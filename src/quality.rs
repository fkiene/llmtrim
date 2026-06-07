//! Quality evaluation harness (spec §6).
//!
//! The token gate proves savings; this proves the savings don't break the task.
//! Per §6, input and output quality are evaluated separately, and a lossy stage may
//! only ship once it passes here.
//!
//! - **Recall** (network-free): for lossy retrieval (Stage B), does the
//!   answer-bearing content survive compression? Sweeping `keep_ratio` and reading
//!   recall tells you the safe drop ratio — the gate for turning Stage B default-on.
//! - **Task success** (needs a model): does the model still answer correctly given
//!   the compressed request? The [`Model`] trait lets a live API (the proxy phase)
//!   or recorded answers plug in; the scaffold runs offline today.

use anyhow::{Context, Result};
use serde_json::{Value, json};

use crate::compress_with_config;
use crate::config::DenseConfig;
use crate::ir::ProviderKind;
use crate::provider::{self, Provider};
use crate::transport::Endpoint;

/// A case asserting that answer-bearing phrases survive compression.
pub struct RecallCase {
    pub name: String,
    pub request: String,
    pub provider: ProviderKind,
    /// Phrases the answer depends on; each must remain in the compressed request.
    pub must_keep: Vec<String>,
}

/// Fraction of `must_keep` phrases present in `haystack` (1.0 = all retained).
pub fn recall(must_keep: &[String], haystack: &str) -> f64 {
    if must_keep.is_empty() {
        return 1.0;
    }
    let hit = must_keep
        .iter()
        .filter(|p| haystack.contains(p.as_str()))
        .count();
    hit as f64 / must_keep.len() as f64
}

/// Recall + token measurement for one case.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub name: String,
    pub recall: f64,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

impl RecallResult {
    pub fn savings_pct(&self) -> f64 {
        if self.tokens_before == 0 {
            0.0
        } else {
            (self.tokens_before as f64 - self.tokens_after as f64) / self.tokens_before as f64
                * 100.0
        }
    }
}

/// Compress each case with `config` and measure recall + token reduction.
pub fn run_recall(cases: &[RecallCase], config: &DenseConfig) -> Result<Vec<RecallResult>> {
    cases
        .iter()
        .map(|c| {
            let r = compress_with_config(&c.request, Some(c.provider), config)?;
            Ok(RecallResult {
                name: c.name.clone(),
                recall: recall(&c.must_keep, &r.request_json),
                tokens_before: r.input_tokens_before.0,
                tokens_after: r.input_tokens_after.0,
            })
        })
        .collect()
}

/// Mean recall across results (1.0 when empty).
pub fn mean_recall(results: &[RecallResult]) -> f64 {
    if results.is_empty() {
        return 1.0;
    }
    results.iter().map(|r| r.recall).sum::<f64>() / results.len() as f64
}

/// Load a held-out corpus from JSONL into recall cases — so existing benchmark
/// datasets (LongBench, ZeroSCROLLS, prompt-compression-benchmarker exports) drive
/// the §6 gate with no bespoke framework. Each line may use `context`|`input`,
/// `question`|`query`, and `answers`|`answer`|`expected`.
pub fn load_corpus(jsonl: &str, provider: ProviderKind) -> Result<Vec<RecallCase>> {
    let mut cases = Vec::new();
    for (i, line) in jsonl.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line)
            .with_context(|| format!("corpus line {} is not valid JSON", i + 1))?;
        let context = field(&v, &["context", "input", "passage", "document"]).unwrap_or_default();
        let question = field(&v, &["question", "query", "prompt"]).unwrap_or_default();

        let mut messages = vec![json!({"role": "user", "content": context})];
        if !question.is_empty() {
            messages.push(json!({"role": "user", "content": question}));
        }
        cases.push(RecallCase {
            name: format!("case-{}", i + 1),
            request: json!({"model": "gpt-4o", "messages": messages}).to_string(),
            provider,
            must_keep: answers(&v),
        });
    }
    Ok(cases)
}

fn field(v: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| v.get(*k).and_then(Value::as_str))
        .map(str::to_string)
}

fn answers(v: &Value) -> Vec<String> {
    if let Some(arr) = v.get("answers").and_then(Value::as_array) {
        return arr
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }
    field(v, &["answer", "expected"]).into_iter().collect()
}

/// Token usage reported by the provider for one response, including prompt-cache hits
/// (Stage A). `None` fields mean the provider didn't report that figure.
#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    /// Input tokens served from the prompt cache (the cheap `cache_read` rate).
    pub cached_tokens: Option<u64>,
}

/// Extract usage from a chat/messages response, tolerant to OpenAI
/// (`prompt_tokens_details.cached_tokens`) and Anthropic (`cache_read_input_tokens`)
/// shapes, both of which OpenRouter may surface.
pub fn parse_usage(response: &Value) -> Usage {
    let u = response.get("usage");
    let num = |k: &str| u.and_then(|u| u.get(k)).and_then(Value::as_u64);
    let cached = u
        .and_then(|u| u.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_u64)
        .or_else(|| num("cache_read_input_tokens"))
        .or_else(|| num("cached_tokens"));
    Usage {
        prompt_tokens: num("prompt_tokens").or_else(|| num("input_tokens")),
        completion_tokens: num("completion_tokens").or_else(|| num("output_tokens")),
        cached_tokens: cached,
    }
}

/// An abstract LLM endpoint, so task-success evals can run against a live API
/// later or recorded answers now.
pub trait Model {
    fn answer(&self, request_json: &str) -> Result<String>;

    /// Also return provider token usage (prompt/completion/cached) alongside the answer.
    /// Default: the answer with empty usage — only live endpoints fill it in.
    fn answer_with_usage(&self, request_json: &str) -> Result<(String, Usage)> {
        Ok((self.answer(request_json)?, Usage::default()))
    }
}

/// A task-success case: the model's answer to the (compressed) request must
/// contain `expected`.
pub struct TaskCase {
    pub name: String,
    pub request: String,
    pub provider: ProviderKind,
    pub expected: String,
}

/// Compress each case, ask the model, and return the fraction whose answer
/// contains the expected phrase. Lights up once a real [`Model`] is wired (proxy
/// phase); runs offline today with a recorded/stub model.
pub fn run_task_success(
    cases: &[TaskCase],
    config: &DenseConfig,
    model: &dyn Model,
) -> Result<f64> {
    if cases.is_empty() {
        return Ok(1.0);
    }
    let mut ok = 0usize;
    for case in cases {
        let r = compress_with_config(&case.request, Some(case.provider), config)?;
        let answer = model.answer(&r.request_json)?;
        if answer.contains(case.expected.as_str()) {
            ok += 1;
        }
    }
    Ok(ok as f64 / cases.len() as f64)
}

/// A live LLM endpoint as a quality [`Model`]: sends the (compressed) request and
/// extracts the answer text. Lets task-success evals run against a real API once
/// keys are available (the proxy phase) — the harness is otherwise model-agnostic.
pub struct HttpModel {
    endpoint: Endpoint,
    provider: Box<dyn Provider>,
}

impl HttpModel {
    pub fn from_env(provider: ProviderKind) -> Result<Self> {
        Ok(Self {
            endpoint: Endpoint::from_env(provider)?,
            provider: provider::for_kind(provider),
        })
    }

    /// Build with explicit credentials (e.g. an OpenRouter base URL + key resolved
    /// from a `.env`), bypassing the fixed `OPENAI_*` env var names.
    pub fn new(provider: ProviderKind, base_url: String, api_key: String) -> Self {
        Self {
            endpoint: Endpoint {
                provider,
                base_url,
                api_key,
            },
            provider: provider::for_kind(provider),
        }
    }
}

impl Model for HttpModel {
    fn answer(&self, request_json: &str) -> Result<String> {
        let response = self.endpoint.send(request_json)?;
        let value: serde_json::Value =
            serde_json::from_str(&response).context("response is not valid JSON")?;
        self.provider
            .answer_text(&value)
            .context("no answer text found in response")
    }
}

/// A live OpenRouter endpoint as a quality [`Model`], via the `async-openai` client's
/// bring-your-own-types path: it POSTs the **exact** (compressed) request body as raw
/// JSON and returns the raw response, so llmtrim's injected fields — provider
/// routing, `cache_control`, reordered messages — reach the API unchanged. A typed SDK
/// request builder would silently drop them and benchmark a different request.
///
/// Wraps a single-thread Tokio runtime so the otherwise-async client satisfies the sync
/// [`Model`] interface; this is the dev/bench transport and stays out of the no-async
/// compressor core.
#[cfg(feature = "live")]
pub struct OpenRouterModel {
    client: async_openai::Client<async_openai::config::OpenAIConfig>,
    runtime: tokio::runtime::Runtime,
    provider: Box<dyn Provider>,
}

#[cfg(feature = "live")]
impl OpenRouterModel {
    /// Build against OpenRouter's OpenAI-compatible endpoint with the given key.
    pub fn new(api_key: String, provider: ProviderKind) -> Result<Self> {
        let config = async_openai::config::OpenAIConfig::new()
            .with_api_base("https://openrouter.ai/api/v1")
            .with_api_key(api_key);
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to start Tokio runtime for the OpenRouter client")?;
        Ok(Self {
            client: async_openai::Client::with_config(config),
            runtime,
            provider: provider::for_kind(provider),
        })
    }
}

#[cfg(feature = "live")]
impl OpenRouterModel {
    /// POST the raw (compressed) body and return the raw response Value, retrying with
    /// exponential backoff on transient upstream errors (HTTP 429 / rate-limit / 5xx) —
    /// Groq free-tier routing rate-limits in bursts, so a short wait usually clears it.
    fn send(&self, request_json: &str) -> Result<Value> {
        let body: Value =
            serde_json::from_str(request_json).context("request is not valid JSON")?;
        let mut last = anyhow::anyhow!("no attempt made");
        const ATTEMPTS: u32 = 8;
        for attempt in 0..ATTEMPTS {
            // Light pre-call pace to stay under strict free-tier rate limits (Groq).
            std::thread::sleep(std::time::Duration::from_millis(350));
            match self
                .runtime
                .block_on(self.client.chat().create_byot::<Value, Value>(body.clone()))
            {
                Ok(v) => return Ok(v),
                Err(e) => {
                    let msg = e.to_string();
                    let transient = ["429", "rate", "temporarily", "502", "503", "overloaded"]
                        .iter()
                        .any(|p| msg.contains(p));
                    last = anyhow::anyhow!("OpenRouter request failed: {msg}");
                    if !transient || attempt == ATTEMPTS - 1 {
                        break;
                    }
                    // Exponential backoff capped at 20s: 0.8,1.6,3.2,6.4,12.8,20,20…
                    let backoff = (800u64 << attempt).min(20_000);
                    std::thread::sleep(std::time::Duration::from_millis(backoff));
                }
            }
        }
        Err(last)
    }
}

#[cfg(feature = "live")]
impl Model for OpenRouterModel {
    fn answer(&self, request_json: &str) -> Result<String> {
        let response = self.send(request_json)?;
        self.provider
            .answer_text(&response)
            .context("no answer text found in response")
    }

    fn answer_with_usage(&self, request_json: &str) -> Result<(String, Usage)> {
        let response = self.send(request_json)?;
        let text = self
            .provider
            .answer_text(&response)
            .context("no answer text found in response")?;
        Ok((text, parse_usage(&response)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_corpus_parses_common_schemas() {
        let jsonl = "{\"context\":\"The vault code is 7741.\",\"question\":\"code?\",\"answers\":[\"7741\"]}\n\
                     {\"input\":\"Paris is the capital.\",\"query\":\"capital?\",\"answer\":\"Paris\"}";
        let cases = load_corpus(jsonl, ProviderKind::OpenAi).unwrap();
        assert_eq!(cases.len(), 2);
        assert_eq!(cases[0].must_keep, vec!["7741"]);
        assert_eq!(cases[1].must_keep, vec!["Paris"]);
        let v: Value = serde_json::from_str(&cases[0].request).unwrap();
        assert_eq!(
            v["messages"].as_array().unwrap().len(),
            2,
            "context + question"
        );
    }

    #[test]
    fn recall_counts_surviving_phrases() {
        let keep = vec!["alpha".to_string(), "beta".to_string()];
        assert_eq!(recall(&keep, "x alpha y"), 0.5);
        assert_eq!(recall(&keep, "alpha beta"), 1.0);
        assert_eq!(recall(&[], "anything"), 1.0);
    }

    /// Stub model that echoes the request — proves the task-success plumbing offline
    /// (a real model plugs in unchanged via the `Model` trait).
    struct EchoModel;
    impl Model for EchoModel {
        fn answer(&self, request_json: &str) -> Result<String> {
            Ok(request_json.to_string())
        }
    }

    #[test]
    fn task_success_harness_runs_offline() {
        let cases = vec![TaskCase {
            name: "secret".to_string(),
            request: r#"{"model":"gpt-4o","messages":[{"role":"user","content":"the secret code is ALPHA7"}]}"#.to_string(),
            provider: ProviderKind::OpenAi,
            expected: "ALPHA7".to_string(),
        }];
        let rate = run_task_success(&cases, &DenseConfig::default(), &EchoModel).unwrap();
        assert_eq!(
            rate, 1.0,
            "echoed compressed request still contains the answer"
        );
    }

    #[test]
    fn provider_answer_text_extraction() {
        use crate::provider::for_kind;
        use serde_json::json;
        let oa = for_kind(ProviderKind::OpenAi);
        assert_eq!(
            oa.answer_text(&json!({"choices":[{"message":{"content":"hi"}}]})),
            Some("hi".to_string())
        );
        assert_eq!(oa.answer_text(&json!({"nope": 1})), None);
        let an = for_kind(ProviderKind::Anthropic);
        assert_eq!(
            an.answer_text(
                &json!({"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]})
            ),
            Some("ab".to_string())
        );
    }
}
