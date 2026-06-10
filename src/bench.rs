//! Benchmark harness (quality axis).
//!
//! The token gate proves savings; this proves the savings keep the task **correct**,
//! on real corpora with a real model. Two axes per run:
//!
//! - **tokens saved** — real tokenizer, measured at compress time.
//! - **quality retained** — the A/B delta between the model's answer on the
//!   ORIGINAL request and on the COMPRESSED one. A preset is only honest if
//!   retention stays high at its token saving; the frontier of (saved, retained) is
//!   the benchmark, not the saving alone.
//!
//! Scoring is ground-truth where possible — numeric-exact for GSM8K, pass@1 that runs
//! the unit tests for HumanEval — so there is no judge noise. Token-F1 / span-EM cover
//! extractive QA; an LLM judge is reserved for open-ended shapes only.

use anyhow::{Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Value, json};
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;
use std::time::Duration;
use wait_timeout::ChildExt;

use crate::compress_with_config;
use crate::config::DenseConfig;
use crate::ir::ProviderKind;
use crate::quality::Model;
use crate::tokenizer::TokenCounter;

/// How a case's answer is graded against its gold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scorer {
    /// Final numeric answer must match (GSM8K, MATH). Commas/units/prose ignored.
    NumericExact,
    /// Normalized exact match: the model's answer reduces to the gold span.
    SpanEm,
    /// Token-level F1 over normalized tokens (SQuAD / HotpotQA standard).
    TokenF1,
    /// Gold phrase appears verbatim in the answer (loose containment).
    ContainsMatch,
    /// Run the model's code against provided unit tests (HumanEval/MBPP). Gold is
    /// JSON `{"test":…, "entry_point":…}`; scored in [`exec`] (needs a subprocess).
    PassAtOne,
    /// Compare the emitted tool call (name + args) against gold JSON (agent corpora).
    ToolCallMatch,
    /// A cheap model judges answer-vs-gold equivalence (open-ended shapes only).
    LlmJudge,
}

impl Scorer {
    /// Parse a scorer name (corpus manifests, CLI).
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_lowercase().as_str() {
            "numeric" | "numeric_exact" => Scorer::NumericExact,
            "span" | "span_em" | "em" => Scorer::SpanEm,
            "f1" | "token_f1" => Scorer::TokenF1,
            "contains" | "contains_match" => Scorer::ContainsMatch,
            "pass@1" | "pass_at_one" | "passatone" => Scorer::PassAtOne,
            "tool" | "tool_call" | "tool_call_match" => Scorer::ToolCallMatch,
            "judge" | "llm_judge" => Scorer::LlmJudge,
            _ => return None,
        })
    }

    /// True when scoring needs an external resource (subprocess or judge model),
    /// handled outside [`score_text`].
    pub fn needs_resource(self) -> bool {
        matches!(
            self,
            Scorer::PassAtOne | Scorer::ToolCallMatch | Scorer::LlmJudge
        )
    }
}

/// One benchmark case: a provider-shaped request, the gold answer, and how to grade.
pub struct BenchCase {
    pub name: String,
    pub request: String,
    pub provider: ProviderKind,
    pub gold: String,
    pub scorer: Scorer,
}

/// Score a text answer against gold for the resource-free scorers. Returns `None`
/// for scorers that need a subprocess or judge model (handled elsewhere).
pub fn score_text(scorer: Scorer, answer: &str, gold: &str) -> Option<f64> {
    Some(match scorer {
        Scorer::NumericExact => numeric_exact(answer, gold),
        Scorer::SpanEm => span_em(answer, gold),
        Scorer::TokenF1 => token_f1(answer, gold),
        Scorer::ContainsMatch => contains_match(answer, gold),
        _ => return None,
    })
}

/// SQuAD-style normalization for answer comparison: lowercase, drop punctuation and
/// the articles a/an/the, collapse whitespace.
///
/// This is **eval-time answer scoring**, not prompt compression — the
/// no-stopword-stripping guardrail (LLM-Microscope) governs what we send to the
/// model, not how we grade its reply against a reference. Article/punctuation
/// removal here is the canonical SQuAD metric.
fn normalize(s: &str) -> String {
    let lowered = s.to_lowercase();
    let spaced: String = lowered
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect();
    spaced
        .split_whitespace()
        .filter(|w| !matches!(*w, "a" | "an" | "the"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Token-level F1 between answer and gold over normalized tokens (SQuAD metric):
/// multiset overlap → harmonic mean of precision and recall.
fn token_f1(answer: &str, gold: &str) -> f64 {
    let a = normalize(answer);
    let g = normalize(gold);
    let at: Vec<&str> = a.split_whitespace().collect();
    let gt: Vec<&str> = g.split_whitespace().collect();
    if at.is_empty() || gt.is_empty() {
        // Both empty → trivially equal; one empty → no overlap.
        return if at.is_empty() && gt.is_empty() {
            1.0
        } else {
            0.0
        };
    }
    // Common tokens = sum of min(count_answer, count_gold) per distinct token.
    let mut common = 0usize;
    let mut seen: Vec<&str> = Vec::new();
    for tok in &gt {
        if seen.contains(tok) {
            continue;
        }
        seen.push(tok);
        let in_a = at.iter().filter(|t| *t == tok).count();
        let in_g = gt.iter().filter(|t| *t == tok).count();
        common += in_a.min(in_g);
    }
    if common == 0 {
        return 0.0;
    }
    let precision = common as f64 / at.len() as f64;
    let recall = common as f64 / gt.len() as f64;
    2.0 * precision * recall / (precision + recall)
}

/// Normalized exact match: the answer, reduced, equals the gold — or contains it as
/// a contiguous token run (verbose chat answers wrap the span in prose).
fn span_em(answer: &str, gold: &str) -> f64 {
    let a = normalize(answer);
    let g = normalize(gold);
    if g.is_empty() {
        return if a.is_empty() { 1.0 } else { 0.0 };
    }
    if a == g {
        return 1.0;
    }
    // Contiguous token-subsequence containment (so "the answer is paris" matches gold "paris").
    let at: Vec<&str> = a.split_whitespace().collect();
    let gt: Vec<&str> = g.split_whitespace().collect();
    if gt.len() <= at.len() && at.windows(gt.len()).any(|w| w == gt.as_slice()) {
        1.0
    } else {
        0.0
    }
}

/// Loose containment: the gold phrase appears verbatim in the answer.
fn contains_match(answer: &str, gold: &str) -> f64 {
    if gold.is_empty() || answer.contains(gold) {
        1.0
    } else {
        0.0
    }
}

static NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"-?\d[\d,]*(?:\.\d+)?").unwrap());

/// Parse the **last** number in a string (models state the final answer last),
/// stripping thousands commas. `None` if there is none.
fn last_number(s: &str) -> Option<f64> {
    NUMBER
        .find_iter(s)
        .last()
        .and_then(|m| m.as_str().replace(',', "").parse::<f64>().ok())
}

/// Numeric exactness: the last number in the answer equals the last number in the
/// gold (GSM8K gold is often `#### 72`). Tolerant to commas, units, and prose.
fn numeric_exact(answer: &str, gold: &str) -> f64 {
    match (last_number(answer), last_number(gold)) {
        (Some(a), Some(g)) if (a - g).abs() <= 1e-6 * g.abs().max(1.0) => 1.0,
        _ => 0.0,
    }
}

/// Load a normalized benchmark corpus (one JSON object per line) into cases.
///
/// Two authoring forms per line:
/// - **explicit**: `{"request": "<full provider request JSON>", "gold": …, "scorer": …}`
///   — used by tool/structured shapes that need a hand-shaped request (tools, schema).
/// - **friendly**: `{"system"?, "context"?, "question"|"prompt", "gold", "scorer"}`
///   — the request is assembled (one system + context + question message chain).
///
/// `gold` may be a string or an array (multi-answer QA); the first element is the
/// reference. The A/B retention metric grades original and compressed answers against
/// the same gold, so any first-answer simplification cancels in the delta.
pub fn load_bench_corpus(
    jsonl: &str,
    provider: ProviderKind,
    default_model: &str,
) -> Result<Vec<BenchCase>> {
    let mut cases = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .with_context(|| format!("bench corpus line {} is not valid JSON", i + 1))?;
        let scorer = v
            .get("scorer")
            .and_then(Value::as_str)
            .and_then(Scorer::parse)
            .unwrap_or(Scorer::ContainsMatch);
        let gold = gold_of(&v);
        let request = match v.get("request").and_then(Value::as_str) {
            Some(req) => req.to_string(),
            None => build_request(&v, default_model),
        };
        let name = v
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("case-{}", i + 1));
        cases.push(BenchCase {
            name,
            request,
            provider,
            gold,
            scorer,
        });
    }
    Ok(cases)
}

/// Extract the reference answer: a string field, or the first element of an array
/// (`gold`/`answer`/`expected`). Objects (e.g. pass@1 `{test, entry_point}`) are kept
/// verbatim as JSON so the exec scorer can parse them.
fn gold_of(v: &Value) -> String {
    for key in ["gold", "answer", "expected"] {
        match v.get(key) {
            Some(Value::String(s)) => return s.clone(),
            Some(Value::Array(a)) => {
                // First element as text: a string verbatim, a scalar via to_string()
                // (so a numeric/bool array gold isn't silently dropped).
                return match a.first() {
                    Some(Value::String(s)) => s.clone(),
                    Some(n @ (Value::Number(_) | Value::Bool(_))) => n.to_string(),
                    _ => String::new(),
                };
            }
            // Objects (e.g. pass@1 `{test, entry_point}`) stay verbatim JSON for the exec scorer.
            Some(obj @ Value::Object(_)) => return obj.to_string(),
            // Scalars: a numeric/bool gold (`{"gold": 7}`) is a valid reference — render it
            // so numeric/contains scorers see "7", not "" (which scores 0 on BOTH arms).
            Some(n @ (Value::Number(_) | Value::Bool(_))) => return n.to_string(),
            _ => {}
        }
    }
    String::new()
}

/// Assemble a provider request from friendly fields: optional system, optional
/// context, then the question/prompt as the final user turn.
fn build_request(v: &Value, default_model: &str) -> String {
    let str_of = |keys: &[&str]| -> Option<String> {
        keys.iter()
            .find_map(|k| v.get(*k).and_then(Value::as_str))
            .map(str::to_string)
    };
    let model = str_of(&["model"]).unwrap_or_else(|| default_model.to_string());
    let mut messages = Vec::new();
    if let Some(sys) = str_of(&["system"]) {
        messages.push(json!({"role": "system", "content": sys}));
    }
    if let Some(ctx) = str_of(&["context", "input", "passage", "document"]) {
        messages.push(json!({"role": "user", "content": ctx}));
    }
    if let Some(q) = str_of(&["question", "query", "prompt"]) {
        messages.push(json!({"role": "user", "content": q}));
    }
    json!({"model": model, "messages": messages}).to_string()
}

static EXEC_SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Extract code from a chat answer: the first fenced ```code block``` (dropping an
/// optional language tag), else the whole answer.
fn extract_code(answer: &str) -> String {
    if let Some(start) = answer.find("```") {
        let after = &answer[start + 3..];
        let body_start = after.find('\n').map(|i| i + 1).unwrap_or(0);
        let body = &after[body_start..];
        return match body.find("```") {
            Some(end) => body[..end].to_string(),
            None => body.to_string(),
        };
    }
    answer.to_string()
}

/// pass@1: run the model's function against HumanEval's `test` harness (`gold` is JSON
/// `{"test":…, "entry_point":…}`). Returns 1.0 iff the assembled program exits cleanly.
///
/// SECURITY: this executes **untrusted, model-generated code**. It is sandboxed only by
/// best-effort, defense-in-depth measures — NOT a security boundary. Run the bench on
/// throwaway/CI hosts, never against secrets or production credentials. The hardening:
/// `python3 -I` (isolated: ignore `PYTHON*` env, user site-packages, and `$CWD` on the
/// import path), POSIX `setrlimit` caps (CPU seconds, address space, file size, no new
/// processes) injected as a preamble, and a hard `wait-timeout` wall-clock kill (no
/// dependency on an external `timeout` binary). A small import preamble covers the
/// typing/math names HumanEval prompts assume, so a correct body isn't failed for an
/// import the chat model omitted.
fn pass_at_one(answer: &str, gold: &str, timeout_secs: u64) -> f64 {
    let Ok(g) = serde_json::from_str::<Value>(gold) else {
        return 0.0;
    };
    let (Some(test), Some(entry)) = (
        g.get("test").and_then(Value::as_str),
        g.get("entry_point").and_then(Value::as_str),
    ) else {
        return 0.0;
    };
    let code = extract_code(answer);
    // Best-effort in-process resource caps (POSIX): CPU time ~= the wall budget, 512 MiB
    // address space, no file writes, no forked subprocesses. Wrapped in try/except so a
    // platform without `resource` (e.g. Windows) still runs under the wall-clock bound.
    let limits = format!(
        "import resource as _r\n\
         try:\n\
         \x20   _r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))\n\
         \x20   _r.setrlimit(_r.RLIMIT_AS, (512*1024*1024, 512*1024*1024))\n\
         \x20   _r.setrlimit(_r.RLIMIT_FSIZE, (0, 0))\n\
         \x20   _r.setrlimit(_r.RLIMIT_NPROC, (0, 0))\n\
         except Exception:\n\
         \x20   pass\n",
        cpu = timeout_secs.max(1)
    );
    let preamble = "from typing import List, Dict, Tuple, Optional, Any\nimport math, re, collections, itertools, functools\n";
    let program = format!("{limits}{preamble}\n{code}\n\n{test}\n\ncheck({entry})\n");

    let seq = EXEC_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("dp_passk_{}_{}.py", std::process::id(), seq));
    if std::fs::write(&path, program.as_bytes()).is_err() {
        return 0.0;
    }
    let spawned = std::process::Command::new("python3")
        .arg("-I") // isolated mode: ignore env/user-site/cwd on the import path
        .arg(&path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
    let passed = match spawned {
        Ok(mut child) => match child.wait_timeout(Duration::from_secs(timeout_secs)) {
            Ok(Some(status)) => status.success(),
            Ok(None) => {
                let _ = child.kill(); // overran the budget
                let _ = child.wait();
                false
            }
            Err(_) => false,
        },
        Err(_) => false,
    };
    let _ = std::fs::remove_file(&path);
    passed as u8 as f64
}

/// The function name from a tool call, across the common chat shapes:
/// `{"name":…}`, `{"function":{"name":…}}`, or `{"tool_calls":[{"function":{"name":…}}]}`.
fn tool_call_name(v: &Value) -> Option<&str> {
    v.get("name")
        .and_then(Value::as_str)
        .or_else(|| v.pointer("/function/name").and_then(Value::as_str))
        .or_else(|| {
            v.pointer("/tool_calls/0/function/name")
                .and_then(Value::as_str)
        })
}

/// The argument object keys from a tool call (sorted), for optional arg-key matching.
/// `arguments` may be a nested object or a JSON-encoded string (OpenAI emits the latter).
fn tool_call_arg_keys(v: &Value) -> Option<Vec<String>> {
    let args = v
        .pointer("/arguments")
        .or_else(|| v.pointer("/function/arguments"))
        .or_else(|| v.pointer("/tool_calls/0/function/arguments"))?;
    let obj = match args {
        Value::Object(_) => args.clone(),
        Value::String(s) => serde_json::from_str::<Value>(s).ok()?,
        _ => return None,
    };
    let mut keys: Vec<String> = obj.as_object()?.keys().cloned().collect();
    keys.sort();
    Some(keys)
}

/// Match the model's tool call against the gold call (agent corpora). `gold` is JSON
/// `{"name":…}` (optionally `{"arguments":…}`); the answer is the serialized tool call (or
/// prose). Scores 1.0 iff the expected function name was invoked — the primary tool-selection
/// signal. The name is matched by **equality** on the extracted call (not substring, which
/// would let gold `get_weather` match a different `get_weather_forecast`); when the gold
/// pins argument keys and the answer is a parseable call, those keys must match too.
fn tool_call_match(answer: &str, gold: &str) -> f64 {
    let Some(gold_v) = serde_json::from_str::<Value>(gold).ok() else {
        return 0.0;
    };
    let Some(want_name) = tool_call_name(&gold_v).map(str::to_string) else {
        return 0.0;
    };
    let want_keys = tool_call_arg_keys(&gold_v);

    // Preferred path: the answer is (or contains) a structured call — compare by equality.
    if let Ok(ans_v) = serde_json::from_str::<Value>(answer.trim())
        && let Some(got_name) = tool_call_name(&ans_v)
    {
        if got_name != want_name {
            return 0.0;
        }
        // Arg keys only gate when the gold specifies them and the answer carries args.
        if let (Some(want), Some(got)) = (&want_keys, tool_call_arg_keys(&ans_v))
            && *want != got
        {
            return 0.0;
        }
        return 1.0;
    }

    // Fallback for prose answers ("I'll call generate_password now"): whole-word match on the
    // name (a word boundary, so it won't match a longer different tool name as a substring).
    let bytes = answer.as_bytes();
    let is_word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let matched = answer.match_indices(&want_name).any(|(i, _)| {
        let before_ok = i == 0 || !is_word(bytes[i - 1]);
        let after = i + want_name.len();
        let after_ok = after >= bytes.len() || !is_word(bytes[after]);
        before_ok && after_ok
    });
    if matched { 1.0 } else { 0.0 }
}

/// Ask a cheap judge model whether `answer` is equivalent to the reference `gold`
/// (open-ended shapes only). The reply is constrained to a single 1/0 token;
/// `judge_model` is the id the judge endpoint routes to.
fn judge_score(judge: &dyn Model, judge_model: &str, route: &str, answer: &str, gold: &str) -> f64 {
    // Cap the texts so the judge prompt stays cheap; equivalence needs the gist, not
    // the full long-form answer.
    let clip = |s: &str| s.chars().take(1500).collect::<String>();
    let prompt = format!(
        "Grade whether the ANSWER is correct and equivalent to the REFERENCE. \
         End your reply with a single digit: 1 if correct/equivalent, 0 if not.\n\n\
         REFERENCE:\n{}\n\nANSWER:\n{}\n\nVerdict (1 or 0):",
        clip(gold),
        clip(answer)
    );
    let mut req = json!({
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        // Reasoning models (e.g. gpt-oss) spend tokens thinking before the content — too
        // small a cap leaves nothing for the verdict. Give room + keep reasoning low.
        "max_tokens": 256,
        "temperature": 0,
        "reasoning": {"effort": "low"},
    });
    // Pin the judge to the same upstream as the bench (else it hits a random provider).
    if !route.is_empty()
        && let Some(obj) = req.as_object_mut()
    {
        let (provider, quant) = match route.split_once('/') {
            Some((p, q)) => (p, Some(q)),
            None => (route, None),
        };
        let mut routing = json!({"order": [provider], "allow_fallbacks": false});
        if let Some(q) = quant {
            routing["quantizations"] = json!([q]);
        }
        obj.insert("provider".to_string(), routing);
    }
    match judge.answer(&req.to_string()) {
        Ok(t) => parse_judge_verdict(&t),
        Err(_) => 0.0,
    }
}

/// Interpret a judge reply as 1.0 (correct/equivalent) or 0.0. We constrain the judge to end
/// with a single digit, so parse the **final standalone token**: the last maximal run of digit
/// characters in the reply, read as a whole number. This avoids the old last-0/1-char bug where
/// "score: 10" → '0' → fail and "1.5" → '1'…'5' picks a stray digit. A trailing "1"/"10"/… reads
/// as correct; "0" reads as incorrect; anything else (no digits) is treated as not-correct.
fn parse_judge_verdict(reply: &str) -> f64 {
    // Walk from the end, skipping trailing non-digits (punctuation/whitespace), then take the
    // contiguous digit run as the verdict token.
    let chars: Vec<char> = reply.chars().collect();
    let mut end = chars.len();
    while end > 0 && !chars[end - 1].is_ascii_digit() {
        end -= 1;
    }
    let mut start = end;
    while start > 0 && chars[start - 1].is_ascii_digit() {
        start -= 1;
    }
    if start == end {
        return 0.0; // no digit token at all
    }
    let token: String = chars[start..end].iter().collect();
    // Any non-zero whole number ("1", "10", …) = correct; "0"/"00" = incorrect.
    match token.parse::<u64>() {
        Ok(0) => 0.0,
        Ok(_) => 1.0,
        Err(_) => 0.0,
    }
}

/// The full bench scorer: dispatches each [`Scorer`] to its method — pass@1 runs the
/// code, tool-call matches the function name, the LLM judge calls `judge` (if set), and
/// everything else is resource-free text scoring.
pub struct BenchScorer<'a> {
    pub exec_timeout: u64,
    pub judge: Option<&'a dyn Model>,
    pub judge_model: String,
    /// Upstream route for judge calls (same `provider`/`provider/quant` as the bench),
    /// so the judge hits the same pinned endpoint rather than a random one.
    pub route: String,
}

impl BenchScorer<'_> {
    /// Grade a model answer against `gold` for the case's [`Scorer`] shape: pass@1 runs
    /// the code, tool-call matches the function name, the LLM judge calls `judge` (if
    /// set), and everything else is resource-free text scoring.
    pub fn score(&self, scorer: Scorer, answer: &str, gold: &str) -> f64 {
        match scorer {
            Scorer::PassAtOne => pass_at_one(answer, gold, self.exec_timeout),
            Scorer::ToolCallMatch => tool_call_match(answer, gold),
            Scorer::LlmJudge => self
                .judge
                .map(|j| judge_score(j, &self.judge_model, &self.route, answer, gold))
                .unwrap_or(0.0),
            other => score_text(other, answer, gold).unwrap_or(0.0),
        }
    }
}

/// Provider token pricing in USD per 1K tokens.
#[derive(Debug, Clone, Copy)]
pub struct Pricing {
    pub input_per_1k: f64,
    pub output_per_1k: f64,
    /// Price of a cached-input token (Stage A prompt cache). 0 if unknown.
    pub cache_per_1k: f64,
}

impl Pricing {
    /// Uncached round-trip cost: input + output.
    pub fn cost(&self, tokens_in: usize, tokens_out: usize) -> f64 {
        tokens_in as f64 / 1000.0 * self.input_per_1k
            + tokens_out as f64 / 1000.0 * self.output_per_1k
    }

    /// Round-trip cost with prompt-cache accounting: `cached_in` input tokens are billed
    /// at the cheaper `cache_read` rate, the rest at the full input rate, plus output.
    pub fn cost_cached(&self, tokens_in: usize, cached_in: usize, tokens_out: usize) -> f64 {
        let cached = cached_in.min(tokens_in);
        let uncached = tokens_in - cached;
        uncached as f64 / 1000.0 * self.input_per_1k
            + cached as f64 / 1000.0 * self.cache_per_1k
            + tokens_out as f64 / 1000.0 * self.output_per_1k
    }
}

/// A model→pricing table parsed from the pinned `bench/pricing.json` snapshot.
pub type PriceTable = std::collections::HashMap<String, Pricing>;

/// Parse the pinned models.dev snapshot (`{models: {id: {input,output,cache_read}}}`,
/// USD per 1M) into per-1K [`Pricing`]. Returns an empty table on parse failure, so
/// the caller transparently falls back to [`pricing_for`].
pub fn load_pricing(json: &str) -> PriceTable {
    let mut table = PriceTable::new();
    let Ok(v) = serde_json::from_str::<Value>(json) else {
        return table;
    };
    let Some(models) = v.get("models").and_then(Value::as_object) else {
        return table;
    };
    for (id, m) in models {
        let per_1k = |k: &str| m.get(k).and_then(Value::as_f64).unwrap_or(0.0) / 1000.0;
        let input_per_1k = per_1k("input");
        // Conservative fallback: if cache_read is absent, bill cached tokens at the
        // full input rate so benchmark savings estimates are never overstated.
        let cache_per_1k = m
            .get("cache_read")
            .and_then(Value::as_f64)
            .map(|v| v / 1000.0)
            .unwrap_or(input_per_1k);
        table.insert(
            id.clone(),
            Pricing {
                input_per_1k,
                output_per_1k: per_1k("output"),
                cache_per_1k,
            },
        );
    }
    table
}

/// Resolve pricing for a model: the pinned table first (exact id, then with the
/// `provider/` prefix stripped), else the hardcoded fallback. Lets the live snapshot
/// drive cost while staying correct offline.
pub fn resolve_pricing(table: &PriceTable, model: &str) -> Pricing {
    if let Some(p) = table.get(model) {
        return *p;
    }
    if let Some((_, bare)) = model.split_once('/')
        && let Some(p) = table.get(bare)
    {
        return *p;
    }
    pricing_for(model)
}

/// Hardcoded fallback pricing (USD per 1K) for when the pinned snapshot is absent or
/// lacks a model. Unknown models (e.g. free OpenRouter tiers) price at zero, so cost
/// columns read 0 rather than mislead.
pub fn pricing_for(model: &str) -> Pricing {
    let m = model.to_lowercase();
    let p = |input_per_1k, output_per_1k, cache_per_1k| Pricing {
        input_per_1k,
        output_per_1k,
        cache_per_1k,
    };
    if m.contains("gpt-3.5-turbo") {
        p(0.0005, 0.0015, 0.0)
    } else if m.contains("gpt-4o-mini") {
        p(0.00015, 0.0006, 0.000075)
    } else if m.contains("gpt-4o") {
        p(0.0025, 0.010, 0.00125)
    } else if m.contains("gpt-4.1-mini") {
        p(0.0004, 0.0016, 0.0001)
    } else if m.contains("gpt-4.1") {
        p(0.002, 0.008, 0.0005)
    } else {
        p(0.0, 0.0, 0.0)
    }
}

/// One case's A/B outcome: tokens and quality on the original vs the compressed
/// request, plus the round-trip cost of each.
#[derive(Debug, Clone)]
pub struct CaseOutcome {
    pub name: String,
    /// Tiktoken counts — our deterministic compression measure (drives input/output %).
    pub tokens_in_before: usize,
    pub tokens_in_after: usize,
    pub tokens_out_orig: usize,
    pub tokens_out_comp: usize,
    /// Provider-reported prompt tokens (`usage`) — the real billing/cache denominator;
    /// the model's own tokenizer differs from tiktoken, so cache% must divide by THIS,
    /// not the tiktoken count (else cache can exceed 100%). 0 if the provider is silent.
    pub prompt_orig: usize,
    pub prompt_comp: usize,
    /// Input tokens served from the provider's prompt cache (Stage A), from `usage`.
    pub cached_in_orig: usize,
    pub cached_in_comp: usize,
    pub quality_orig: f64,
    pub quality_comp: f64,
    pub cost_orig: f64,
    pub cost_comp: f64,
}

/// True if an upstream send error is **transient** (worth skipping, not scoring as a
/// regression): rate limits (429) and server errors (5xx). A deterministic 4xx (e.g. a 400
/// from a body the compression broke) is NOT transient — that's the worst regression and must
/// be counted, not silently dropped. Matched on the error message text (the live transport
/// surfaces the status/string; there's no typed status here).
fn is_transient_error(msg: &str) -> bool {
    let m = msg.to_lowercase();
    [
        "429",
        "rate",
        "temporarily",
        "overloaded",
        "timeout",
        "timed out",
    ]
    .iter()
    .any(|p| m.contains(p))
        || ["500", "502", "503", "504"].iter().any(|p| m.contains(p))
}

/// Prepend a unique per-arm nonce as a leading system message, to bust the provider's prefix
/// cache between the back-to-back ORIGINAL and COMPRESSED sends. Without it, the 2nd call reuses
/// the 1st's cached prefix and gets a cache discount even when the preset's `cache` stage is OFF,
/// inflating cost/cache numbers by run ordering. A tiny leading marker shifts the cached prefix
/// so neither arm hits the other's cache. Non-fatal: returns the input unchanged if it can't parse.
fn cache_bust(request_json: &str, nonce: &str) -> String {
    let Ok(mut v) = serde_json::from_str::<Value>(request_json) else {
        return request_json.to_string();
    };
    let Some(obj) = v.as_object_mut() else {
        return request_json.to_string();
    };
    let marker = json!({"role": "system", "content": format!("[bench-nonce {nonce}]")});
    if let Some(Value::Array(msgs)) = obj.get_mut("messages") {
        msgs.insert(0, marker);
    } else if let Some(Value::Array(input)) = obj.get_mut("input") {
        // Responses-style payloads use `input` instead of `messages`.
        input.insert(0, marker);
    }
    v.to_string()
}

/// Result of an A/B run: the scored case outcomes plus what happened to cases that didn't
/// complete cleanly, and whether cache-busting was applied — surfaced so the frontier is honest.
#[derive(Debug, Clone, Default)]
pub struct AbRun {
    pub outcomes: Vec<CaseOutcome>,
    /// Cases where ORIGINAL succeeded but COMPRESSED failed with a deterministic 4xx
    /// (compression broke the body): scored `quality_comp = 0.0` and KEPT in `outcomes`.
    pub failed: usize,
    /// Cases dropped for a transient reason (429/5xx/timeout on either arm), not scored.
    pub skipped: usize,
    /// Whether a per-arm cache-busting nonce was injected (preset's `cache` stage OFF).
    pub cache_busted: bool,
}

/// Run the A/B benchmark: for each case, compress with `config`, ask the model on
/// BOTH the original and the compressed request, score each answer, and price the
/// round-trip. Output tokens are counted on the answer text with the same tokenizer,
/// so input and output savings are on one consistent scale.
///
/// Measurement hygiene:
/// - **Cache fairness (#3):** when the preset's `cache` stage is OFF we inject a per-arm nonce
///   to bust the cross-arm prefix cache AND zero the cached-token discount in costing, so the
///   compressed arm isn't credited a cache hit it only got from running second.
/// - **Failure honesty (#4):** a COMPRESSED 4xx after a successful ORIGINAL is the worst
///   regression — it's scored `quality_comp = 0.0` and counted, not dropped. Only transient
///   429/5xx are skipped.
pub fn run_ab(
    cases: &[BenchCase],
    config: &DenseConfig,
    model: &dyn Model,
    counter: &dyn TokenCounter,
    scorer: &BenchScorer,
    pricing: Pricing,
) -> Result<AbRun> {
    // When the cache stage is NOT under test, neutralize the cross-arm cache-warm artifact.
    let cache_under_test =
        config.cache || (config.auto && cases.iter().any(|c| c.request.contains("cache_control")));
    let bust = !cache_under_test;
    let mut run = AbRun {
        cache_busted: bust,
        ..Default::default()
    };
    for (i, case) in cases.iter().enumerate() {
        let compressed = compress_with_config(&case.request, Some(case.provider), config)?;

        // Per-arm cache-busting nonces (distinct between arms and across cases).
        let (req_orig, req_comp) = if bust {
            (
                cache_bust(&case.request, &format!("o{i}")),
                cache_bust(&compressed.request_json, &format!("c{i}")),
            )
        } else {
            (case.request.clone(), compressed.request_json.clone())
        };

        // ORIGINAL is the baseline: if it fails we have nothing to compare against, so skip
        // the case regardless of cause (can't attribute a regression without the baseline).
        let (answer_orig, usage_orig) = match model.answer_with_usage(&req_orig) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("  skip {} (original send): {e}", case.name);
                run.skipped += 1;
                continue;
            }
        };

        // COMPRESSED: a transient failure → skip; a deterministic 4xx (compression broke the
        // request) → score the case 0 and KEEP it, so the worst regression can't vanish.
        let (answer_comp, usage_comp, comp_failed) = match model.answer_with_usage(&req_comp) {
            Ok((a, u)) => (a, u, false),
            Err(e) => {
                let msg = e.to_string();
                if is_transient_error(&msg) {
                    eprintln!("  skip {} (compressed send, transient): {e}", case.name);
                    run.skipped += 1;
                    continue;
                }
                eprintln!(
                    "  FAIL {} (compressed send broke the request, scoring 0): {e}",
                    case.name
                );
                run.failed += 1;
                (String::new(), crate::quality::Usage::default(), true)
            }
        };

        let tokens_out_orig = counter.count(&answer_orig);
        let tokens_out_comp = counter.count(&answer_comp);
        // Zero the cache discount when busting (cache not under test) so a residual hit can't
        // inflate savings; keep the real cached counts when the cache stage IS being measured.
        let cached_in_orig = if bust {
            0
        } else {
            usage_orig.cached_tokens.unwrap_or(0) as usize
        };
        let cached_in_comp = if bust {
            0
        } else {
            usage_comp.cached_tokens.unwrap_or(0) as usize
        };
        let quality_orig = scorer.score(case.scorer, &answer_orig, &case.gold);
        // A broken compressed request scores 0 outright (no answer to grade).
        let quality_comp = if comp_failed {
            0.0
        } else {
            scorer.score(case.scorer, &answer_comp, &case.gold)
        };

        // Prefer the provider's own token counts for billing/cache; fall back to our
        // tiktoken counts when the provider doesn't report usage (incl. the failed comp arm).
        let prompt_orig = usage_orig.prompt_tokens.map(|x| x as usize);
        let prompt_comp = usage_comp.prompt_tokens.map(|x| x as usize);
        let billed_in_orig = prompt_orig.unwrap_or(compressed.input_tokens_before.0);
        let billed_in_comp = prompt_comp.unwrap_or(compressed.input_tokens_after.0);
        let billed_out_orig = usage_orig
            .completion_tokens
            .map(|x| x as usize)
            .unwrap_or(tokens_out_orig);
        let billed_out_comp = usage_comp
            .completion_tokens
            .map(|x| x as usize)
            .unwrap_or(tokens_out_comp);

        run.outcomes.push(CaseOutcome {
            name: case.name.clone(),
            tokens_in_before: compressed.input_tokens_before.0,
            tokens_in_after: compressed.input_tokens_after.0,
            tokens_out_orig,
            tokens_out_comp,
            prompt_orig: prompt_orig.unwrap_or(0),
            prompt_comp: prompt_comp.unwrap_or(0),
            cached_in_orig,
            cached_in_comp,
            quality_orig,
            quality_comp,
            cost_orig: pricing.cost_cached(billed_in_orig, cached_in_orig, billed_out_orig),
            cost_comp: pricing.cost_cached(billed_in_comp, cached_in_comp, billed_out_comp),
        });
    }
    Ok(run)
}

/// A mean with a 95% confidence half-width (normal approximation of the standard
/// error of the mean). For n < 30 treat the interval as indicative, not exact.
#[derive(Debug, Clone, Copy)]
pub struct Stat {
    pub mean: f64,
    pub ci95: f64,
    pub n: usize,
}

/// Mean ± 95% CI half-width for a sample, via `statrs`: sample mean/std-dev and the
/// two-sided Student-t critical value at dof = n−1 (correct for small n, unlike a flat
/// 1.96 normal approximation). Empty → all zeros; n=1 → CI 0.
pub fn mean_ci(xs: &[f64]) -> Stat {
    let n = xs.len();
    if n == 0 {
        return Stat {
            mean: 0.0,
            ci95: 0.0,
            n: 0,
        };
    }
    let mean = xs.mean();
    if n == 1 {
        return Stat { mean, ci95: 0.0, n };
    }
    let std_err = xs.std_dev() / (n as f64).sqrt();
    let t = StudentsT::new(0.0, 1.0, (n - 1) as f64)
        .map(|d| d.inverse_cdf(0.975))
        .unwrap_or(1.96);
    Stat {
        mean,
        ci95: t * std_err,
        n,
    }
}

/// Aggregate frontier point for one (corpus × preset) run: how much was saved and how
/// much quality survived.
#[derive(Debug, Clone)]
pub struct Frontier {
    pub n: usize,
    pub tokens_in_saved_pct: f64,
    pub tokens_out_saved_pct: f64,
    pub cost_saved_pct: f64,
    pub quality_orig: Stat,
    pub quality_comp: Stat,
    /// (compressed − original) quality, in percentage points. Negative = harm.
    pub retention_pp: f64,
    /// 95% CI half-width (pp) of the **paired** per-case retention delta (comp−orig) — the
    /// correct interval for the retention number, not the CI of the compressed mean.
    pub retention_ci95_pp: f64,
    /// Share of compressed input tokens served from the prompt cache (Stage A), %.
    pub cache_used_pct: f64,
    /// Cases scored 0 because the compressed send failed with a deterministic 4xx (#4).
    pub failed: usize,
    /// Cases dropped for a transient reason (429/5xx), not scored (#4).
    pub skipped: usize,
    /// Whether a per-arm cache-busting nonce neutralized the cross-arm cache artifact (#3).
    pub cache_busted: bool,
}

fn pct_drop(before: f64, after: f64) -> f64 {
    if before <= 0.0 {
        0.0
    } else {
        (before - after) / before * 100.0
    }
}

/// Roll an A/B run into one frontier point, carrying its failure/skip/cache-busting flags.
pub fn summarize(run: &AbRun) -> Frontier {
    let mut f = summarize_outcomes(&run.outcomes);
    f.failed = run.failed;
    f.skipped = run.skipped;
    f.cache_busted = run.cache_busted;
    f
}

/// Roll case outcomes into one frontier point (without the run-level flags).
pub fn summarize_outcomes(outcomes: &[CaseOutcome]) -> Frontier {
    let sum = |f: &dyn Fn(&CaseOutcome) -> f64| outcomes.iter().map(f).sum::<f64>();
    let in_before = sum(&|o| o.tokens_in_before as f64);
    let in_after = sum(&|o| o.tokens_in_after as f64);
    let out_orig = sum(&|o| o.tokens_out_orig as f64);
    let out_comp = sum(&|o| o.tokens_out_comp as f64);
    let cost_orig = sum(&|o| o.cost_orig);
    let cost_comp = sum(&|o| o.cost_comp);
    let cached_comp = sum(&|o| o.cached_in_comp as f64);
    // Cache % denominator is the PROVIDER's prompt-token count (same tokenizer as the
    // cached count), not our tiktoken estimate — else cache_used can exceed 100%.
    let prompt_comp = sum(&|o| o.prompt_comp as f64);
    let q_orig = mean_ci(&outcomes.iter().map(|o| o.quality_orig).collect::<Vec<_>>());
    let q_comp = mean_ci(&outcomes.iter().map(|o| o.quality_comp).collect::<Vec<_>>());
    // Paired retention: per-case (comp − orig). Its mean equals the unpaired difference, but
    // its CI is computed over the per-case deltas (a paired design), which is the honest
    // interval for "did compression change quality?" — far tighter than the compressed mean's
    // CI when orig/comp are correlated (they usually are: same case, same gold).
    let deltas: Vec<f64> = outcomes
        .iter()
        .map(|o| o.quality_comp - o.quality_orig)
        .collect();
    let delta = mean_ci(&deltas);
    Frontier {
        n: outcomes.len(),
        tokens_in_saved_pct: pct_drop(in_before, in_after),
        tokens_out_saved_pct: pct_drop(out_orig, out_comp),
        cost_saved_pct: pct_drop(cost_orig, cost_comp),
        quality_orig: q_orig,
        quality_comp: q_comp,
        retention_pp: delta.mean * 100.0,
        retention_ci95_pp: delta.ci95 * 100.0,
        cache_used_pct: if prompt_comp > 0.0 {
            (cached_comp / prompt_comp * 100.0).min(100.0)
        } else {
            0.0
        },
        failed: 0,
        skipped: 0,
        cache_busted: false,
    }
}

/// Ablation variants of a base config: `"full"`, then one variant per *enabled* stage
/// with that stage turned off. Comparing `full` against `-stage` isolates each stage's
/// contribution (à la arXiv:2606.01326's per-transformation table). Stages already off
/// in the base are skipped (they'd be no-ops).
pub fn ablation_configs(base: &DenseConfig) -> Vec<(String, DenseConfig)> {
    type On = fn(&DenseConfig) -> bool;
    type Off = fn(&mut DenseConfig);
    let stages: &[(&str, On, Off)] = &[
        ("retrieve", |c| c.retrieve, |c| c.retrieve = false),
        ("serialize", |c| c.serialize, |c| c.serialize = false),
        (
            "serialize_flatten",
            |c| c.serialize_flatten,
            |c| c.serialize_flatten = false,
        ),
        (
            "serialize_buckets",
            |c| c.serialize_buckets,
            |c| c.serialize_buckets = false,
        ),
        ("dedup", |c| c.dedup, |c| c.dedup = false),
        ("dedup_near", |c| c.dedup_near, |c| c.dedup_near = false),
        ("hygiene", |c| c.hygiene, |c| c.hygiene = false),
        (
            "strip_base64",
            |c| c.strip_base64,
            |c| c.strip_base64 = false,
        ),
        (
            "normalize_unicode",
            |c| c.normalize_unicode,
            |c| c.normalize_unicode = false,
        ),
        ("json_crush", |c| c.json_crush, |c| c.json_crush = false),
        (
            "output_control",
            |c| c.output_control,
            |c| c.output_control = false,
        ),
        ("skeletonize", |c| c.skeletonize, |c| c.skeletonize = false),
        ("minify_code", |c| c.minify_code, |c| c.minify_code = false),
        ("ngram", |c| c.ngram, |c| c.ngram = false),
        ("tool_select", |c| c.tool_select, |c| c.tool_select = false),
        ("toolout", |c| c.toolout, |c| c.toolout = false),
        ("multimodal", |c| c.multimodal, |c| c.multimodal = false),
        ("cache", |c| c.cache, |c| c.cache = false),
    ];
    let mut out = vec![("full".to_string(), base.clone())];
    for (name, is_on, turn_off) in stages {
        if !is_on(base) {
            continue;
        }
        let mut c = base.clone();
        turn_off(&mut c);
        out.push((format!("-{name}"), c));
    }
    out
}

/// Offline per-config input-token totals (no model calls): compress every case under
/// each config and sum input tokens before/after. Reproducible and free — isolates each
/// stage's INPUT contribution without burning the API.
pub fn run_token_ablation(
    cases: &[BenchCase],
    configs: &[(String, DenseConfig)],
) -> Result<Vec<(String, usize, usize)>> {
    let mut rows = Vec::with_capacity(configs.len());
    for (label, cfg) in configs {
        let (mut before, mut after) = (0usize, 0usize);
        for c in cases {
            let r = compress_with_config(&c.request, Some(c.provider), cfg)?;
            before += r.input_tokens_before.0;
            after += r.input_tokens_after.0;
        }
        rows.push((label.clone(), before, after));
    }
    Ok(rows)
}

/// Render frontier rows (one per corpus×preset) as a Markdown table for the README. The
/// retention column carries the **paired** delta CI; `fail/skip` surfaces broken-compressed
/// (scored 0) and transient-dropped case counts so a clean table can't hide regressions.
pub fn frontier_markdown(rows: &[(String, Frontier)]) -> String {
    let mut s = String::from(
        "| run | n | input saved | output saved | cost saved | cache used | quality (orig→comp) | retention | fail/skip |\n\
         |---|--:|--:|--:|--:|--:|:--:|--:|--:|\n",
    );
    for (label, f) in rows {
        s.push_str(&format!(
            "| {label} | {} | {:.1}% | {:.1}% | {:.1}% | {:.1}% | {:.0}%→{:.0}% | {:+.1}±{:.1}pp | {}/{} |\n",
            f.n,
            f.tokens_in_saved_pct,
            f.tokens_out_saved_pct,
            f.cost_saved_pct,
            f.cache_used_pct,
            f.quality_orig.mean * 100.0,
            f.quality_comp.mean * 100.0,
            f.retention_pp,
            f.retention_ci95_pp,
            f.failed,
            f.skipped,
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_exact_ignores_prose_commas_and_units() {
        assert_eq!(
            numeric_exact("So the total is 1,250 dollars.", "#### 1250"),
            1.0
        );
        assert_eq!(numeric_exact("The answer is 72.", "72"), 1.0);
        assert_eq!(numeric_exact("I get 30.", "#### 70000"), 0.0);
        assert_eq!(numeric_exact("no number here", "5"), 0.0);
    }

    #[test]
    fn numeric_exact_takes_last_number() {
        // Reasoning mentions 3 and 4 along the way; final answer 12.
        assert_eq!(numeric_exact("3 times 4 is 12", "12"), 1.0);
    }

    #[test]
    fn token_f1_rewards_partial_overlap() {
        assert_eq!(token_f1("the quick brown fox", "quick brown fox"), 1.0); // articles dropped
        let partial = token_f1("Paris is the capital of France", "Paris");
        assert!(
            partial > 0.0 && partial < 1.0,
            "partial overlap scored, got {partial}"
        );
        assert_eq!(token_f1("totally unrelated", "Paris France"), 0.0);
    }

    #[test]
    fn token_f1_empty_handling() {
        assert_eq!(token_f1("", ""), 1.0);
        assert_eq!(token_f1("something", ""), 0.0);
    }

    #[test]
    fn span_em_matches_span_inside_prose() {
        assert_eq!(span_em("The answer is Paris.", "Paris"), 1.0);
        assert_eq!(span_em("Paris", "paris"), 1.0); // case/normalize
        assert_eq!(span_em("It is London.", "Paris"), 0.0);
    }

    #[test]
    fn contains_is_loose() {
        assert_eq!(contains_match("result: 42 ok", "42"), 1.0);
        assert_eq!(contains_match("nope", "42"), 0.0);
    }

    #[test]
    fn score_text_returns_none_for_resource_scorers() {
        assert!(score_text(Scorer::PassAtOne, "x", "y").is_none());
        assert!(score_text(Scorer::ToolCallMatch, "x", "y").is_none());
        assert!(score_text(Scorer::LlmJudge, "x", "y").is_none());
        assert!(score_text(Scorer::NumericExact, "5", "5").is_some());
    }

    #[test]
    fn scorer_parse_roundtrips_names() {
        assert_eq!(Scorer::parse("numeric"), Some(Scorer::NumericExact));
        assert_eq!(Scorer::parse("F1"), Some(Scorer::TokenF1));
        assert_eq!(Scorer::parse("pass@1"), Some(Scorer::PassAtOne));
        assert_eq!(Scorer::parse("nonsense"), None);
    }

    #[test]
    fn needs_resource_flags_exec_and_judge() {
        assert!(Scorer::PassAtOne.needs_resource());
        assert!(Scorer::LlmJudge.needs_resource());
        assert!(!Scorer::NumericExact.needs_resource());
        assert!(!Scorer::TokenF1.needs_resource());
    }

    struct StubModel(String);
    impl Model for StubModel {
        fn answer(&self, _req: &str) -> Result<String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn run_ab_computes_outcomes_and_frontier() {
        use crate::tokenizer::counter_for;
        use serde_json::json;
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let cfg = DenseConfig::default();
        let model = StubModel("The answer is 42.".to_string());
        let cases = vec![BenchCase {
            name: "a".into(),
            request:
                json!({"model":"gpt-4o","messages":[{"role":"user","content":"what is 6*7?"}]})
                    .to_string(),
            provider: ProviderKind::OpenAi,
            gold: "42".into(),
            scorer: Scorer::NumericExact,
        }];
        let scorer = BenchScorer {
            exec_timeout: 10,
            judge: None,
            judge_model: String::new(),
            route: String::new(),
        };
        let run = run_ab(
            &cases,
            &cfg,
            &model,
            counter.as_ref(),
            &scorer,
            pricing_for("gpt-4o"),
        )
        .unwrap();
        assert_eq!(run.outcomes.len(), 1);
        assert_eq!(run.failed, 0);
        assert_eq!(run.skipped, 0);
        assert!(
            run.cache_busted,
            "default preset has cache off → busting on"
        );
        let o = &run.outcomes[0];
        assert_eq!(o.quality_orig, 1.0);
        assert_eq!(o.quality_comp, 1.0);
        assert!(o.tokens_out_orig > 0);
        assert!(o.cost_orig > 0.0, "gpt-4o is priced");
        assert_eq!(o.cached_in_comp, 0, "cache discount zeroed when busting");

        let f = summarize(&run);
        assert_eq!(f.n, 1);
        assert_eq!(f.retention_pp, 0.0, "stub answers identically → no harm");
        assert_eq!(f.retention_ci95_pp, 0.0, "n=1 paired delta → CI 0");
        assert_eq!(f.quality_comp.mean, 1.0);
    }

    /// A stub model that fails the COMPRESSED arm (second `answer_with_usage` call) with a
    /// chosen error — to drive the #4 fail-vs-skip classification deterministically.
    struct FailCompModel {
        ok_answer: String,
        comp_error: String,
        calls: std::cell::Cell<u32>,
    }
    impl Model for FailCompModel {
        fn answer(&self, _req: &str) -> Result<String> {
            unreachable!("bench uses answer_with_usage")
        }
        fn answer_with_usage(&self, _req: &str) -> Result<(String, crate::quality::Usage)> {
            let n = self.calls.get();
            self.calls.set(n + 1);
            if n == 0 {
                Ok((self.ok_answer.clone(), crate::quality::Usage::default()))
            } else {
                Err(anyhow::anyhow!("{}", self.comp_error))
            }
        }
    }

    fn one_case() -> Vec<BenchCase> {
        use serde_json::json;
        vec![BenchCase {
            name: "a".into(),
            request: json!({"model":"m","messages":[{"role":"user","content":"q"}]}).to_string(),
            provider: ProviderKind::OpenAi,
            gold: "42".into(),
            scorer: Scorer::NumericExact,
        }]
    }

    #[test]
    fn compressed_4xx_is_scored_zero_not_dropped() {
        use crate::tokenizer::counter_for;
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let model = FailCompModel {
            ok_answer: "The answer is 42.".into(),
            comp_error: "OpenRouter request failed: 400 Bad Request (invalid body)".into(),
            calls: std::cell::Cell::new(0),
        };
        let scorer = BenchScorer {
            exec_timeout: 5,
            judge: None,
            judge_model: String::new(),
            route: String::new(),
        };
        let run = run_ab(
            &one_case(),
            &DenseConfig::default(),
            &model,
            counter.as_ref(),
            &scorer,
            pricing_for("gpt-4o"),
        )
        .unwrap();
        // The worst regression is COUNTED, not silently skipped.
        assert_eq!(run.outcomes.len(), 1, "4xx case kept");
        assert_eq!(run.failed, 1, "counted as failed");
        assert_eq!(run.skipped, 0);
        assert_eq!(
            run.outcomes[0].quality_comp, 0.0,
            "broken compressed scores 0"
        );
        assert_eq!(run.outcomes[0].quality_orig, 1.0, "original still graded");
        let f = summarize(&run);
        assert_eq!(f.failed, 1);
        assert!(
            f.retention_pp < 0.0,
            "regression shows as negative retention"
        );
    }

    #[test]
    fn compressed_transient_error_is_skipped() {
        use crate::tokenizer::counter_for;
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let model = FailCompModel {
            ok_answer: "The answer is 42.".into(),
            comp_error: "OpenRouter request failed: 429 Too Many Requests".into(),
            calls: std::cell::Cell::new(0),
        };
        let scorer = BenchScorer {
            exec_timeout: 5,
            judge: None,
            judge_model: String::new(),
            route: String::new(),
        };
        let run = run_ab(
            &one_case(),
            &DenseConfig::default(),
            &model,
            counter.as_ref(),
            &scorer,
            pricing_for("gpt-4o"),
        )
        .unwrap();
        assert_eq!(run.outcomes.len(), 0, "transient case dropped");
        assert_eq!(run.skipped, 1, "counted as skipped");
        assert_eq!(run.failed, 0);
    }

    #[test]
    fn transient_error_classifier() {
        assert!(is_transient_error("429 Too Many Requests"));
        assert!(is_transient_error("503 Service Unavailable"));
        assert!(is_transient_error("upstream temporarily overloaded"));
        assert!(!is_transient_error("400 Bad Request"));
        assert!(!is_transient_error("422 Unprocessable Entity"));
    }

    #[test]
    fn cache_bust_prepends_unique_nonce() {
        use serde_json::json;
        let req = json!({"model":"m","messages":[{"role":"user","content":"hi"}]}).to_string();
        let a = cache_bust(&req, "o0");
        let b = cache_bust(&req, "c0");
        assert!(a.contains("bench-nonce o0") && b.contains("bench-nonce c0"));
        assert_ne!(a, b, "arms get distinct leading nonces");
        let v: Value = serde_json::from_str(&a).unwrap();
        assert_eq!(
            v["messages"][0]["role"], "system",
            "nonce is the new leading message"
        );
    }

    #[test]
    fn mean_ci_basic() {
        let s = mean_ci(&[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.mean, 1.0);
        assert_eq!(s.ci95, 0.0);
        assert_eq!(s.n, 4);
        let s2 = mean_ci(&[0.0, 1.0]);
        assert_eq!(s2.mean, 0.5);
        assert!(s2.ci95 > 0.0);
        assert_eq!(mean_ci(&[]).n, 0);
    }

    #[test]
    fn pricing_known_and_unknown() {
        assert!(pricing_for("openai/gpt-3.5-turbo").output_per_1k > 0.0);
        let free = pricing_for("meta-llama/llama-3-8b-instruct:free");
        assert_eq!(free.input_per_1k, 0.0);
        assert_eq!(free.output_per_1k, 0.0);
        let pr = Pricing {
            input_per_1k: 1.0,
            output_per_1k: 2.0,
            cache_per_1k: 0.0,
        };
        assert!((pr.cost(1000, 1000) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn pinned_table_drives_pricing_with_fallback() {
        // models.dev snapshot shape: USD per 1M → converted to per-1K on load.
        let snap = r#"{"source":"x","unit":"usd_per_1m","models":{
            "gpt-4o":{"input":2.5,"output":10,"cache_read":1.25},
            "openai/gpt-3.5-turbo":{"input":0.5,"output":1.5,"cache_read":0}}}"#;
        let table = load_pricing(snap);
        let p = resolve_pricing(&table, "gpt-4o");
        assert!((p.input_per_1k - 0.0025).abs() < 1e-9, "2.5/1M = 0.0025/1k");
        assert!((p.output_per_1k - 0.010).abs() < 1e-9);
        assert!((p.cache_per_1k - 0.00125).abs() < 1e-9);
        // prefix strip: send "openai/gpt-3.5-turbo"; also resolves bare key.
        assert!(
            (resolve_pricing(&table, "openai/gpt-3.5-turbo").input_per_1k - 0.0005).abs() < 1e-9
        );
        // missing model → hardcoded fallback (gpt-4o-mini known there).
        assert!(resolve_pricing(&table, "gpt-4o-mini").output_per_1k > 0.0);
        // garbage json → empty table → fallback still works.
        assert!(load_pricing("not json").is_empty());
    }

    #[test]
    fn summarize_reports_savings_and_retention() {
        let outcomes = vec![
            CaseOutcome {
                name: "x".into(),
                tokens_in_before: 100,
                tokens_in_after: 60,
                tokens_out_orig: 50,
                tokens_out_comp: 30,
                prompt_orig: 100,
                prompt_comp: 60,
                cached_in_orig: 0,
                cached_in_comp: 30,
                quality_orig: 1.0,
                quality_comp: 1.0,
                cost_orig: 1.0,
                cost_comp: 0.5,
            },
            CaseOutcome {
                name: "y".into(),
                tokens_in_before: 100,
                tokens_in_after: 60,
                tokens_out_orig: 50,
                tokens_out_comp: 30,
                prompt_orig: 100,
                prompt_comp: 60,
                cached_in_orig: 0,
                cached_in_comp: 30,
                quality_orig: 1.0,
                quality_comp: 0.0,
                cost_orig: 1.0,
                cost_comp: 0.5,
            },
        ];
        let f = summarize_outcomes(&outcomes);
        assert_eq!(f.tokens_in_saved_pct, 40.0);
        assert_eq!(f.tokens_out_saved_pct, 40.0);
        assert_eq!(f.cost_saved_pct, 50.0);
        assert_eq!(f.quality_orig.mean, 1.0);
        assert_eq!(f.quality_comp.mean, 0.5);
        assert_eq!(
            f.retention_pp, -50.0,
            "one case broke → 50pp drop (paired mean)"
        );
        // Paired delta CI over [0, -1]: non-zero (n=2), the honest interval for the delta.
        assert!(f.retention_ci95_pp > 0.0, "paired retention CI is reported");
        assert_eq!(f.cache_used_pct, 50.0, "60 cached / 120 compressed input");
    }

    #[test]
    fn load_corpus_friendly_and_explicit_forms() {
        let jsonl = r#"
{"name":"g1","context":"Ann has 3 apples, Bob has 4.","question":"How many total?","gold":"7","scorer":"numeric"}
{"request":"{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}","gold":["Paris","paris"],"scorer":"f1"}
"#;
        let cases = load_bench_corpus(jsonl, ProviderKind::OpenAi, "gpt-4o").unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].name, "g1");
        assert_eq!(cases[0].scorer, Scorer::NumericExact);
        assert_eq!(cases[0].gold, "7");
        assert!(
            cases[0].request.contains("3 apples") && cases[0].request.contains("How many"),
            "friendly form assembles context + question"
        );
        // explicit request used verbatim; array gold → first element.
        assert!(cases[1].request.contains("\"content\":\"hi\""));
        assert_eq!(cases[1].gold, "Paris");
        assert_eq!(cases[1].scorer, Scorer::TokenF1);
    }

    #[test]
    fn extract_code_strips_fences() {
        assert_eq!(
            extract_code("here:\n```python\ndef f():\n    pass\n```\ndone").trim(),
            "def f():\n    pass"
        );
        assert_eq!(extract_code("def g(): pass"), "def g(): pass");
    }

    fn python_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[test]
    fn pass_at_one_runs_unit_tests() {
        if !python_available() {
            return; // no interpreter on this host → skip (suite stays green)
        }
        use serde_json::json;
        let gold = json!({
            "test": "def check(candidate):\n    assert candidate(2) == 4\n    assert candidate(3) == 9",
            "entry_point": "sq"
        })
        .to_string();
        let good = "```python\ndef sq(x):\n    return x * x\n```";
        let bad = "```python\ndef sq(x):\n    return x + x\n```";
        assert_eq!(pass_at_one(good, &gold, 10), 1.0, "correct solution passes");
        assert_eq!(pass_at_one(bad, &gold, 10), 0.0, "wrong solution fails");
        // malformed gold → 0, never panics.
        assert_eq!(pass_at_one(good, "not json", 10), 0.0);
    }

    #[test]
    fn bench_scorer_delegates_text_scorers() {
        let s = BenchScorer {
            exec_timeout: 10,
            judge: None,
            judge_model: String::new(),
            route: String::new(),
        };
        assert_eq!(s.score(Scorer::NumericExact, "the answer is 42", "42"), 1.0);
        assert_eq!(s.score(Scorer::ContainsMatch, "nope", "42"), 0.0);
    }

    #[test]
    fn tool_call_match_on_function_name() {
        let gold = serde_json::json!({"name":"generate_password"}).to_string();
        assert_eq!(
            tool_call_match(
                "{\"name\":\"generate_password\",\"arguments\":\"x\"}",
                &gold
            ),
            1.0
        );
        assert_eq!(
            tool_call_match("I'll call generate_password now", &gold),
            1.0
        );
        assert_eq!(tool_call_match("{\"name\":\"get_weather\"}", &gold), 0.0);
        assert_eq!(tool_call_match("anything", "not json"), 0.0);
    }

    #[test]
    fn judge_and_bench_scorer_dispatch() {
        let yes = StubModel("1".to_string());
        let no = StubModel("0".to_string());
        assert_eq!(judge_score(&yes, "m", "", "ans", "gold"), 1.0);
        assert_eq!(judge_score(&no, "m", "", "ans", "gold"), 0.0);

        let s = BenchScorer {
            exec_timeout: 5,
            judge: Some(&yes),
            judge_model: "m".into(),
            route: String::new(),
        };
        assert_eq!(s.score(Scorer::LlmJudge, "ans", "gold"), 1.0);
        assert_eq!(s.score(Scorer::TokenF1, "paris", "paris"), 1.0);
        assert_eq!(
            s.score(
                Scorer::ToolCallMatch,
                "{\"name\":\"f\"}",
                "{\"name\":\"f\"}"
            ),
            1.0
        );

        // No judge wired → LlmJudge scores 0 rather than panicking.
        let s2 = BenchScorer {
            exec_timeout: 5,
            judge: None,
            judge_model: "m".into(),
            route: String::new(),
        };
        assert_eq!(s2.score(Scorer::LlmJudge, "ans", "gold"), 0.0);
    }

    #[test]
    fn ablation_skips_already_off_stages() {
        let agg = DenseConfig::preset("aggressive").unwrap();
        let variants = ablation_configs(&agg);
        assert_eq!(variants[0].0, "full");
        let labels: Vec<&str> = variants.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"-retrieve"));
        assert!(labels.contains(&"-output_control"));

        // safe leaves most stages off → fewer ablation rows, and no -retrieve.
        let safe = DenseConfig::preset("safe").unwrap();
        let sv = ablation_configs(&safe);
        assert!(sv.len() < variants.len(), "safe ablates fewer stages");
        assert!(!sv.iter().any(|(l, _)| l == "-retrieve"));
    }

    #[test]
    fn token_ablation_offline_sums_tokens() {
        use serde_json::json;
        let agg = DenseConfig::preset("aggressive").unwrap();
        let configs = ablation_configs(&agg);
        let cases = vec![BenchCase {
            name: "x".into(),
            request: json!({"model":"gpt-4o","messages":[{"role":"user","content":"[{\"a\":1},{\"a\":2},{\"a\":3}]"}]})
                .to_string(),
            provider: ProviderKind::OpenAi,
            gold: String::new(),
            scorer: Scorer::ContainsMatch,
        }];
        let rows = run_token_ablation(&cases, &configs).unwrap();
        assert_eq!(rows.len(), configs.len());
        assert_eq!(rows[0].0, "full");
        assert!(rows[0].1 > 0, "before-tokens counted");
    }

    #[test]
    fn frontier_markdown_renders_table() {
        let f = Frontier {
            n: 10,
            tokens_in_saved_pct: 30.0,
            tokens_out_saved_pct: 50.0,
            cost_saved_pct: 45.0,
            quality_orig: Stat {
                mean: 0.9,
                ci95: 0.05,
                n: 10,
            },
            quality_comp: Stat {
                mean: 0.88,
                ci95: 0.06,
                n: 10,
            },
            retention_pp: -2.0,
            retention_ci95_pp: 1.5,
            cache_used_pct: 25.0,
            failed: 1,
            skipped: 2,
            cache_busted: true,
        };
        let md = frontier_markdown(&[("gsm8k-aggressive".into(), f)]);
        assert!(md.contains("| run |"));
        assert!(md.contains("cache used"));
        assert!(md.contains("fail/skip"));
        assert!(md.contains("gsm8k-aggressive"));
        assert!(md.contains("-2.0±1.5pp"), "paired retention CI rendered");
        assert!(md.contains("1/2"), "fail/skip counts rendered");
    }

    #[test]
    fn gold_of_handles_scalar_and_array_numbers() {
        use serde_json::json;
        // Numeric gold: {"gold": 7} → "7" (not "", which would score 0 on both arms).
        assert_eq!(gold_of(&json!({"gold": 7})), "7");
        assert_eq!(gold_of(&json!({"gold": true})), "true");
        // Numeric array gold → first element stringified.
        assert_eq!(gold_of(&json!({"answer": [7, 8]})), "7");
        // String and object forms still work.
        assert_eq!(gold_of(&json!({"gold": "paris"})), "paris");
        assert!(gold_of(&json!({"gold": {"test": "x"}})).contains("test"));
    }

    #[test]
    fn tool_call_match_requires_name_equality_not_substring() {
        // gold get_weather must NOT match a different, longer tool name as a substring.
        let gold = serde_json::json!({"name": "get_weather"}).to_string();
        assert_eq!(
            tool_call_match("{\"name\":\"get_weather_forecast\"}", &gold),
            0.0,
            "substring of a different tool name is not a match"
        );
        assert_eq!(tool_call_match("{\"name\":\"get_weather\"}", &gold), 1.0);
        // nested function shape
        assert_eq!(
            tool_call_match("{\"function\":{\"name\":\"get_weather\"}}", &gold),
            1.0
        );
        // prose word-boundary match still works; a longer word does not.
        assert_eq!(tool_call_match("I'll call get_weather now", &gold), 1.0);
        assert_eq!(tool_call_match("calling get_weatherly", &gold), 0.0);
    }

    #[test]
    fn tool_call_match_checks_arg_keys_when_gold_pins_them() {
        let gold =
            serde_json::json!({"name": "f", "arguments": {"city": "x", "unit": "c"}}).to_string();
        // Same name + same arg keys → match (values are free).
        assert_eq!(
            tool_call_match(
                "{\"name\":\"f\",\"arguments\":{\"city\":\"paris\",\"unit\":\"k\"}}",
                &gold
            ),
            1.0
        );
        // Same name but missing an arg key → no match.
        assert_eq!(
            tool_call_match("{\"name\":\"f\",\"arguments\":{\"city\":\"paris\"}}", &gold),
            0.0
        );
        // OpenAI emits arguments as a JSON-encoded string — handled.
        assert_eq!(
            tool_call_match(
                "{\"name\":\"f\",\"arguments\":\"{\\\"city\\\":\\\"x\\\",\\\"unit\\\":\\\"c\\\"}\"}",
                &gold
            ),
            1.0
        );
    }

    #[test]
    fn judge_verdict_parses_final_token_robustly() {
        // The old last-0/1-char bug: "score: 10" → must be correct (10), not fail on '0'.
        assert_eq!(parse_judge_verdict("score: 10"), 1.0);
        assert_eq!(parse_judge_verdict("Verdict: 1"), 1.0);
        assert_eq!(parse_judge_verdict("0"), 0.0);
        assert_eq!(parse_judge_verdict("The answer is wrong. 0"), 0.0);
        assert_eq!(parse_judge_verdict("reasoning... finally 1."), 1.0);
        // No digit → not correct (never panics).
        assert_eq!(parse_judge_verdict("undecided"), 0.0);
        assert_eq!(parse_judge_verdict(""), 0.0);
    }

    #[test]
    fn ablation_covers_newer_stages() {
        // A config with the newer stages on should produce ablation rows for each.
        let cfg = DenseConfig {
            toolout: true,
            json_crush: true,
            strip_base64: true,
            normalize_unicode: true,
            dedup_near: true,
            multimodal: true,
            cache: true,
            serialize_flatten: true,
            serialize_buckets: true,
            ..DenseConfig::default()
        };
        let labels: Vec<String> = ablation_configs(&cfg).into_iter().map(|(l, _)| l).collect();
        for stage in [
            "-toolout",
            "-json_crush",
            "-strip_base64",
            "-normalize_unicode",
            "-dedup_near",
            "-multimodal",
            "-cache",
            "-serialize_flatten",
            "-serialize_buckets",
        ] {
            assert!(labels.contains(&stage.to_string()), "ablates {stage}");
        }
    }
}
