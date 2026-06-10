//! Stage G — tool layer: static tool selection + schema-description trimming. Opt-in.
//!
//! Tool/function schemas are resent every request and are often the largest hidden
//! input cost in agent loops. Static tool selection keeps only the tools whose
//! name/description lexically overlaps the conversation (keyword match, no model),
//! dropping the rest; description trimming caps verbose descriptions. Lossy — a
//! dropped or trimmed tool may be the one the model needed — so off by default and
//! InputTokens-gated. (Tool-output hygiene — collapsing repeated log lines in tool
//! results — is handled by Stage E dedup.)
//!
//! Stopwords (which prevent spurious overlap like "the" matching a SQL tool) come
//! from the `stop-words` crate, for the language `whatlang` detects in the request —
//! not a hardcoded English list.

use std::collections::HashSet;

use anyhow::Result;
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct ToolStage {
    pub select: bool,
    pub trim_desc: bool,
    pub max_desc_chars: usize,
}

impl Transform for ToolStage {
    fn name(&self) -> &str {
        "tools"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn scope(&self) -> crate::gate::Scope {
        crate::gate::Scope::Tools // selects/trims tool schemas; content text untouched
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        if self.select {
            select_tools(req, provider);
        }
        if self.trim_desc {
            provider.truncate_tool_descriptions(req, self.max_desc_chars);
        }
        Ok(())
    }
}

/// The reliably-detected language of `sample` (whatlang), or `None` when detection
/// is unreliable or absent. The single language-detection seam in the crate — Stage B
/// retrieval (BM25 + pruning stopwords) and tool selection all route through it, so
/// "what language is this" is decided in exactly one place.
pub(crate) fn detect_lang(sample: &str) -> Option<whatlang::Lang> {
    whatlang::detect(sample)
        .filter(|info| info.is_reliable())
        .map(|info| info.lang())
}

/// Stopwords for the language detected in `sample` (NLTK/ISO lists via the
/// `stop-words` crate), falling back to English when detection is unreliable or the
/// language isn't in our supported map. The map is enum→enum glue; the word lists
/// themselves come from the crate. Shared with Stage B sentence pruning.
pub(crate) fn stopword_set(sample: &str) -> HashSet<&'static str> {
    use stop_words::LANGUAGE as L;
    use whatlang::Lang;
    // Detect on a leading slice of a large segment (see LANG_DETECT_MAX_BYTES): matches
    // whole-text detection for monolingual inputs while avoiding a multi-KB rescan.
    // Char-boundary-safe.
    let head = if sample.len() > LANG_DETECT_MAX_BYTES {
        let mut end = LANG_DETECT_MAX_BYTES;
        while !sample.is_char_boundary(end) {
            end -= 1;
        }
        &sample[..end]
    } else {
        sample
    };
    let language = match detect_lang(head) {
        Some(Lang::Fra) => L::French,
        Some(Lang::Spa) => L::Spanish,
        Some(Lang::Deu) => L::German,
        Some(Lang::Ita) => L::Italian,
        Some(Lang::Por) => L::Portuguese,
        Some(Lang::Nld) => L::Dutch,
        Some(Lang::Rus) => L::Russian,
        Some(Lang::Jpn) => L::Japanese,
        Some(Lang::Kor) => L::Korean,
        Some(Lang::Cmn) => L::Chinese,
        Some(Lang::Ara) => L::Arabic,
        Some(Lang::Tur) => L::Turkish,
        Some(Lang::Pol) => L::Polish,
        Some(Lang::Swe) => L::Swedish,
        Some(Lang::Dan) => L::Danish,
        Some(Lang::Fin) => L::Finnish,
        Some(Lang::Ell) => L::Greek,
        Some(Lang::Hun) => L::Hungarian,
        Some(Lang::Ron) => L::Romanian,
        Some(Lang::Ces) => L::Czech,
        Some(Lang::Ukr) => L::Ukrainian,
        Some(Lang::Vie) => L::Vietnamese,
        Some(Lang::Ind) => L::Indonesian,
        Some(Lang::Hin) => L::Hindi,
        // Any other (or undetected) language falls back to English (graceful, never panics).
        _ => L::English,
    };
    stop_words::get(language).iter().copied().collect()
}

/// Lowercased lexical tokens via the Unicode word segmenter (UAX#29) — works across
/// scripts (CJK, Cyrillic, …) rather than an ASCII `is_alphanumeric` split, which
/// would collapse a space-less script into one token. The shared tokenizer for Stage
/// B retrieval ranking, Stage E SimHash dedup, and tool selection.
pub(crate) fn lex_words(s: &str) -> Vec<String> {
    s.unicode_words().map(str::to_lowercase).collect()
}

/// Content words of `lower` (already lowercased) as a set of **borrowed** slices —
/// Unicode-segmented (universal), snake_case split (`run_sql` → `run`, `sql`), stopwords +
/// single chars dropped. Borrows from `lower`, so no per-word allocation.
fn content_words<'a>(lower: &'a str, stop: &HashSet<&str>) -> HashSet<&'a str> {
    lower
        .unicode_words()
        .flat_map(|w| w.split('_'))
        .filter(|w| w.len() >= 2 && !stop.contains(w))
        .collect()
}

/// Enough leading content to detect the language (whatlang needs a sample, not the whole
/// prompt) — bounded so we never join tens of KB of context just to pick a stopword list.
const LANG_SAMPLE_BYTES: usize = 2048;

/// Cap for language detection on a large segment. `whatlang` is O(input) and its verdict
/// stabilizes within a few KB, so above this we detect on a leading slice rather than
/// rescanning tens of KB (Stage B sentence pruning on big RAG contexts — ~5ms on a 200KB
/// request). 8 KB is generous and representative, so the detected language — hence the
/// stopword set — matches whole-text detection for any monolingual input.
const LANG_DETECT_MAX_BYTES: usize = 8 * 1024;

/// Cap on the (recent) content scanned to build the tool-selection query word-set. Lowercasing
/// and word-segmenting the whole resent prompt every call dominated this stage (~15ms on a
/// 120K request); tool relevance tracks the current task, so a bounded slice of the newest
/// content suffices (already-invoked tools are protected separately).
const TOOL_QUERY_MAX_BYTES: usize = 16 * 1024;

/// Keep only tools whose name/description shares a content word with the
/// conversation. Safety: if nothing matches, keep all tools (never strip the whole
/// toolset on a weak query).
fn select_tools(req: &mut Request, provider: &dyn Provider) {
    let descriptors = provider.tool_descriptors(req);
    if descriptors.len() < 2 {
        return; // nothing meaningful to prune
    }

    let pointers = provider.content_text_pointers(req);
    // Build the query word-set from the most-recent content only, bounded by
    // `TOOL_QUERY_MAX_BYTES`: scanning newest-first and stopping at the cap keeps this O(cap)
    // instead of O(whole resent prompt), which was the stage's dominant cost. The query word-set
    // borrows slices out of `lower` (no per-word allocation); already-invoked tools are kept
    // regardless (below), so bounding the scan can't dangle a `tool_use`.
    let mut lower = String::new();
    for p in pointers.iter().rev() {
        if let Some(s) = req.get_str(p) {
            lower.push_str(&s.to_lowercase());
            lower.push(' ');
            if lower.len() >= TOOL_QUERY_MAX_BYTES {
                break;
            }
        }
    }
    let sample_end = lower.len().min(LANG_SAMPLE_BYTES);
    let stop = stopword_set(lower.get(..sample_end).unwrap_or(&lower));
    let query = content_words(&lower, &stop);
    if query.is_empty() {
        return;
    }

    // Never drop a tool the agent already invoked earlier in the conversation: its
    // `tool_use` block would dangle (and the agent clearly needs it). Multi-turn safety.
    let used = tools_used_in_history(req);
    let keep: Vec<bool> = descriptors
        .iter()
        .map(|(name, desc)| {
            if used.contains(name) {
                return true;
            }
            let tool_lower = format!("{name} {desc}").to_lowercase();
            !query.is_disjoint(&content_words(&tool_lower, &stop))
        })
        .collect();
    if keep.iter().any(|&k| k) {
        provider.retain_tools(req, &keep);
    }
}

/// Names of tools already invoked in the conversation — OpenAI `tool_calls[].function.name`,
/// Anthropic `{type: tool_use, name}` content blocks, and Google `parts[].functionCall.name`.
fn tools_used_in_history(req: &Request) -> HashSet<String> {
    let mut used = HashSet::new();
    let raw = req.raw();
    // OpenAI/Anthropic use "messages"; Google uses "contents".
    let turns = raw
        .get("messages")
        .or_else(|| raw.get("contents"))
        .and_then(Value::as_array);
    let Some(turns) = turns else {
        return used;
    };
    for m in turns {
        // OpenAI: tool_calls[].function.name
        if let Some(calls) = m.get("tool_calls").and_then(Value::as_array) {
            for c in calls {
                if let Some(n) = c.pointer("/function/name").and_then(Value::as_str) {
                    used.insert(n.to_string());
                }
            }
        }
        // Anthropic: content[].{type:tool_use, name}
        if let Some(blocks) = m.get("content").and_then(Value::as_array) {
            for b in blocks {
                if b.get("type").and_then(Value::as_str) == Some("tool_use")
                    && let Some(n) = b.get("name").and_then(Value::as_str)
                {
                    used.insert(n.to_string());
                }
            }
        }
        // Google: parts[].functionCall.name
        if let Some(parts) = m.get("parts").and_then(Value::as_array) {
            for p in parts {
                if let Some(n) = p.pointer("/functionCall/name").and_then(Value::as_str) {
                    used.insert(n.to_string());
                }
            }
        }
    }
    used
}

/// True if a text segment is **structured / positional data** — JSON, CSV/TSV, a table, a
/// key-value/config block, or symbol-dense code/markup — rather than natural-language prose.
///
/// Lossy prose transforms (n-gram abbreviation, near-duplicate line collapse) are safe on
/// prose but corrupt structured data, where token *position* and *count* are load-bearing:
/// the model aligns columns, counts records, or parses syntax, so even a byte-reversible
/// change makes it misread the data (a record array abbreviated by n-gram miscounts rows —
/// `adult` −100pp in the bench). Such segments are left verbatim.
///
/// Format- and language-universal: structure is detected by *shape*, not keywords, and
/// scripts are classified with Unicode `char` categories (not ASCII), so prose in any
/// script — Latin, CJK, Arabic, Cyrillic, Indic … — reads as prose.
pub(crate) fn is_structured_segment(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }

    // 1. JSON — a whole value, or adjacent objects forming a record array.
    if t.starts_with(['{', '[']) && serde_json::from_str::<Value>(t).is_ok() {
        return true;
    }
    if t.contains("},{") || t.contains("}, {") {
        return true;
    }

    let lines: Vec<&str> = t.lines().map(str::trim).filter(|l| !l.is_empty()).collect();

    // 2. Tabular — one column delimiter recurring with the same count across most lines
    //    (CSV / TSV / Markdown table). Fields are position-indexed, so collapsing or
    //    abbreviating rows corrupts the alignment.
    if lines.len() >= 3 {
        for delim in [',', '\t', '|', ';'] {
            let counts: Vec<usize> = lines.iter().map(|l| l.matches(delim).count()).collect();
            let cols = counts.iter().copied().max().unwrap_or(0);
            if cols >= 1 && counts.iter().filter(|&&c| c == cols).count() * 4 >= lines.len() * 3 {
                return true; // ≥75% of lines share the same ≥2-column shape
            }
        }
    }

    // 3. Key-value / config — most lines are `key: value` / `key = value` (YAML, TOML, ini,
    //    env, headers). Keys are positional; abbreviating them breaks lookups.
    if lines.len() >= 3 && lines.iter().filter(|l| is_kv_line(l)).count() * 4 >= lines.len() * 3 {
        return true;
    }

    // 4. Symbol density — code / markup / dense structure. Prose in *any* script is mostly
    //    letters; punctuation + symbols stay well under ~15%. Above ~22% means structure.
    let mut symbols = 0usize;
    let mut nonspace = 0usize;
    for c in t.chars() {
        if c.is_whitespace() {
            continue;
        }
        nonspace += 1;
        if !c.is_alphanumeric() {
            symbols += 1;
        }
    }
    nonspace >= 40 && symbols * 100 >= nonspace * 22
}

/// A `key: value` / `key = value` line with a short, single-clause key — the shape of a
/// config / header line, not a prose sentence that merely contains a colon.
fn is_kv_line(line: &str) -> bool {
    match line.find([':', '=']) {
        Some(i) if i > 0 && i + 1 < line.len() => {
            let key = line[..i].trim();
            !key.is_empty() && key.chars().count() <= 40 && !key.contains(['.', '!', '?'])
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::{AnthropicProvider, OpenAiProvider};
    use crate::tokenizer::counter_for;
    use serde_json::{Value, json};

    #[test]
    fn structured_detects_json_and_record_arrays() {
        assert!(is_structured_segment("[{\"a\":1},{\"a\":2}]"));
        assert!(is_structured_segment("{\"k\": \"v\"}"));
        // JSON records followed by a question (whole thing doesn't parse) → adjacency signal.
        assert!(is_structured_segment(
            "[{\"occupation\":\"Sales\"},{\"occupation\":\"Tech\"}] then a question"
        ));
    }

    #[test]
    fn structured_detects_csv_tsv_and_markdown_tables() {
        assert!(is_structured_segment(
            "name,age,city\nJohn,30,NYC\nJane,25,LA\nBob,40,SF"
        ));
        assert!(is_structured_segment(
            "| col | val |\n|-----|-----|\n| a | 1 |\n| b | 2 |"
        ));
    }

    #[test]
    fn structured_detects_key_value_config() {
        assert!(is_structured_segment(
            "host: localhost\nport: 8080\ndebug: true\nname: app"
        ));
        assert!(is_structured_segment("KEY=val\nFOO=bar\nBAZ=qux"));
    }

    #[test]
    fn structured_detects_code_by_symbol_density() {
        assert!(is_structured_segment(
            "for (let i = 0; i < n; i++) { out[i] = (a[i] + b[i]) * w - bias / 2; }"
        ));
    }

    #[test]
    fn prose_is_not_structured_in_any_script() {
        assert!(!is_structured_segment(
            "The quick brown fox jumps over the lazy dog. It was a calm, bright morning, \
             and nothing at all seemed out of the ordinary on that particular day."
        ));
        // CJK prose: enough characters to pass the length floor, but few symbols, and the
        // ideographs are alphabetic → must read as prose, not a table.
        assert!(!is_structured_segment(
            "这是一段用于测试的中文散文文本，它包含足够多的汉字以超过长度阈值，\
             但是标点符号很少，因此不应该被误判成结构化数据或者表格。"
        ));
        // A single prose line with a colon is not key-value.
        assert!(!is_structured_segment(
            "Note: this is an ordinary sentence that merely happens to contain a colon."
        ));
    }

    fn openai_tools() -> Value {
        json!([
            {"type":"function","function":{"name":"get_weather","description":"Get the weather forecast for a city","parameters":{}}},
            {"type":"function","function":{"name":"send_email","description":"Send an email to a recipient","parameters":{}}},
            {"type":"function","function":{"name":"run_sql","description":"Execute a SQL query against the database","parameters":{}}}
        ])
    }

    fn select_stage() -> Box<dyn Transform> {
        Box::new(ToolStage {
            select: true,
            trim_desc: false,
            max_desc_chars: 200,
        })
    }

    #[test]
    fn openai_selection_keeps_relevant_tool() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"what is the weather forecast in Paris today?"}],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let out = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert!(
            out.stages[0].applied,
            "dropping irrelevant tools reduces tokens"
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["get_weather"], "only the weather tool is kept");
    }

    #[test]
    fn selection_keeps_tools_invoked_earlier() {
        // `run_sql` was called earlier; the latest turn is about weather. Multi-turn safety:
        // a tool already invoked must survive even when irrelevant to the current turn,
        // else its `tool_use` dangles and the agent loses it.
        let body = json!({
            "model":"gpt-4o",
            "messages":[
                {"role":"assistant","tool_calls":[{"function":{"name":"run_sql"}}]},
                {"role":"user","content":"now what is the weather forecast in Paris?"}
            ],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert!(names.contains(&"get_weather"), "relevant tool kept");
        assert!(
            names.contains(&"run_sql"),
            "tool invoked earlier kept despite being irrelevant now"
        );
        assert!(
            !names.contains(&"send_email"),
            "unused irrelevant tool dropped"
        );
    }

    #[test]
    fn keeps_all_when_nothing_matches() {
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"hello there friend"}],
            "tools": openai_tools()
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        assert_eq!(
            req.raw()
                .get("tools")
                .and_then(Value::as_array)
                .unwrap()
                .len(),
            3,
            "weak query keeps the whole toolset (safety)"
        );
    }

    #[test]
    fn french_query_uses_french_stopwords() {
        // "des", "la", "pour" are French stopwords; without French detection they
        // would survive and create spurious overlap. The relevant tool still wins.
        let body = json!({
            "model":"gpt-4o",
            "messages":[{"role":"user","content":"quelle est la météo pour la ville de Paris aujourd'hui"}],
            "tools":[
                {"type":"function","function":{"name":"meteo","description":"Obtenir les prévisions météo pour une ville","parameters":{}}},
                {"type":"function","function":{"name":"envoyer_email","description":"Envoyer un courriel à un destinataire","parameters":{}}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let _ = pipeline::run(
            &mut req,
            &OpenAiProvider,
            counter.as_ref(),
            &[select_stage()],
        );
        let names: Vec<&str> = req
            .raw()
            .get("tools")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["meteo"], "only the weather tool kept (French)");
    }

    #[test]
    fn anthropic_selection_and_trim() {
        let long_desc = "x".repeat(400);
        let body = json!({
            "max_tokens":100,
            "messages":[{"role":"user","content":"run a sql query on the orders table please"}],
            "tools":[
                {"name":"run_sql","description": long_desc,"input_schema":{}},
                {"name":"get_weather","description":"weather forecast","input_schema":{}}
            ]
        });
        let mut req = Request::from_value(ProviderKind::Anthropic, body);
        let counter = counter_for(ProviderKind::Anthropic, None).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(ToolStage {
            select: true,
            trim_desc: true,
            max_desc_chars: 50,
        })];
        pipeline::run(&mut req, &AnthropicProvider, counter.as_ref(), &stages);
        let tools = req.raw().get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(tools.len(), 1, "only run_sql kept");
        let desc = tools[0].get("description").and_then(Value::as_str).unwrap();
        assert!(
            desc.chars().count() <= 51,
            "description trimmed to max+ellipsis"
        );
    }

    #[test]
    fn google_function_call_not_dropped() {
        // tools_used_in_history must find Google functionCall parts so select_tools
        // never orphans a tool already invoked in a Gemini conversation.
        let body = json!({
            "contents": [
                {"role": "user", "parts": [{"text": "what is the weather?"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp": "15°C"}}}]}
            ],
            "tools": [{"functionDeclarations": [
                {"name": "get_weather", "description": "weather forecast"},
                {"name": "run_sql", "description": "database query"}
            ]}]
        });
        // Ensure get_weather is recognised as already-used; a full pipeline run would
        // keep it even if the keyword selector otherwise wouldn't.
        let req = Request::from_value(ProviderKind::Google, body);
        let used = tools_used_in_history(&req);
        assert!(
            used.contains("get_weather"),
            "Google functionCall must be tracked"
        );
    }
}
