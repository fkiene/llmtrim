//! llmtrim — static, deterministic prompt/payload compression for LLM APIs.
//!
//! This crate is a zero-LLM-call middleware: it ingests a provider-shaped request
//! body, compresses it with deterministic algorithms only (no auxiliary model, no
//! embeddings), and can reverse the lossless transforms on the response. The
//! functions here are the **pure transform core** — no network calls live in this
//! crate. The CLI (`main.rs`), and later a proxy/library surface, wrap them.
//!

use anyhow::{Context, Result};
use serde_json::Value;

pub mod autostart;
pub mod bench;
pub mod cache_zone;
pub mod config;
pub mod daemon;
pub mod doctor;
pub mod gate;
pub mod ir;
pub mod media;
pub mod monitor;
pub mod pipeline;
pub mod provider;
pub mod quality;
pub mod select;
pub mod serve;
pub mod setup;
pub mod stages;
pub mod tokenizer;
pub mod tracking;
pub mod transport;
pub mod ui;
pub mod update;

use gate::{PlanEntry, Transform};
use ir::{ProviderKind, Request};
use pipeline::StageReport;
use tokenizer::Tokens;

/// The outcome of compressing one request: the compressed body, the rehydration
/// plan, and the measured token deltas (for the ledger and reporting).
#[derive(Debug, Clone)]
pub struct CompressResult {
    pub request_json: String,
    pub plan: Vec<PlanEntry>,
    pub provider: ProviderKind,
    pub model: Option<String>,
    pub tokenizer_label: String,
    pub tokenizer_exact: bool,
    pub input_tokens_before: Tokens,
    pub input_tokens_after: Tokens,
    pub stages: Vec<StageReport>,
    /// Whether Stage F (output shaping) ran on this request — i.e. the *effective* config
    /// (after `auto` routing) enabled it. The ledger needs this to project the benchmark
    /// output reduction only onto traffic that actually carried the instruction.
    pub output_shaped: bool,
}

/// The ordered MVP stage list for a provider. Empty until Stage D/F land in later
/// build steps; the gated pipeline already runs over it so wiring stages in is a
/// one-line change here.
fn stages_for(_provider: ProviderKind, config: &config::DenseConfig) -> Vec<Box<dyn Transform>> {
    let mut stages: Vec<Box<dyn Transform>> = Vec::new();
    // Stage T (input, lossy): compress tool outputs (logs/diffs/grep) first, so the
    // structure-aware windowing runs before generic prose pruning sees a giant log.
    if config.toolout {
        stages.push(Box::new(stages::ToolOutputStage {
            max_lines: config.toolout_max_lines,
            min_lines: config.toolout_min_lines,
            template: config.toolout_template,
            mode: stages::toolout::ModeSetting::parse(&config.toolout_mode),
        }));
    }
    // Stage B (input-side, lossy): prune large context to the relevant chunks first.
    if config.retrieve {
        stages.push(Box::new(stages::RetrieveStage {
            keep_ratio: config.retrieve_keep_ratio,
            min_segment_chars: config.retrieve_min_segment_chars,
            reorder: config.retrieve_reorder,
            mmr: config.retrieve_mmr,
            mmr_lambda: config.retrieve_mmr_lambda,
            sentence: config.retrieve_sentence,
        }));
    }
    // Stage C (input, lossy): skeletonize non-focus code in fenced blocks.
    if config.skeletonize {
        stages.push(Box::new(stages::SkeletonStage {
            keep_full_top_k: config.skeleton_keep_full_top_k,
            drop_unmatched: config.skeleton_drop_unmatched,
            drop_min_body_lines: config.skeleton_drop_min_body_lines,
        }));
    }
    // Stage C (input, lossless): minify brace-language code (strip whitespace).
    if config.minify_code {
        stages.push(Box::new(stages::MinifyCodeStage));
    }
    // Stage H (input, lossy): image detail tier + downscale embedded images.
    if config.multimodal {
        stages.push(Box::new(stages::ImageStage {
            detail: config.image_detail.clone(),
        }));
    }
    // Stage D (input-side, lossless): clean, then columnar-encode uniform arrays.
    if config.hygiene {
        stages.push(Box::new(stages::HygieneStage {
            strip_base64: config.strip_base64,
            sig_figs: config.numeric_sig_figs,
            normalize_unicode: config.normalize_unicode,
        }));
    }
    // Lossy sample of huge record arrays (keeps anomalies) FIRST — drops rows while it's
    // still JSON; then the columnar encoder below packs the survivors. `safe` (json_crush
    // off) keeps every row and relies on serialize's lossless union CSV instead.
    if config.json_crush {
        stages.push(Box::new(stages::JsonCrushStage {
            max_rows: config.json_crush_max_rows,
        }));
    }
    // Columnar-encode record arrays (incl. near-uniform → union CSV). Lossless.
    if config.serialize {
        stages.push(Box::new(stages::SerializeStage {
            min_rows: config.serialize_min_rows,
            nested: config.serialize_nested,
            csv: config.serialize_csv,
            flatten: config.serialize_flatten,
            buckets: config.serialize_buckets,
        }));
    }
    // Stage E (input, lossy-ish): collapse duplicate lines.
    if config.dedup {
        stages.push(Box::new(stages::DedupStage {
            near: config.dedup_near,
            near_max_distance: config.dedup_near_max_distance,
        }));
    }
    // Stage E+ (input, lossless): abbreviate repeated multi-word phrases with a legend.
    if config.ngram {
        stages.push(Box::new(stages::NgramStage {
            max_entries: config.ngram_max_entries,
        }));
    }
    // Stage G (input, lossy): trim/select tool schemas (resent every call).
    if config.tool_select || config.tool_trim_desc {
        stages.push(Box::new(stages::ToolStage {
            select: config.tool_select,
            trim_desc: config.tool_trim_desc,
            max_desc_chars: config.tool_max_desc_chars,
        }));
    }
    // Stage F (output-side): request-shaping output controls (terse / Chain-of-Draft / budget).
    if config.output_control || config.output_compact_code {
        stages.push(Box::new(stages::OutputControlStage {
            level: stages::output::OutputLevel::parse(&config.output_level),
            max_tokens: config.output_max_tokens,
            token_budget: config.output_token_budget,
            compact_code: config.output_compact_code,
        }));
    }
    // Stage A (lossless, latent payoff): mark the final prefix for provider caching.
    // Last, so it fingerprints system+tools after the other stages have shaped them.
    if config.cache {
        stages.push(Box::new(stages::CacheStage {
            max_breakpoints: config.cache_max_breakpoints,
        }));
    }
    stages
}

/// Pick the workload preset for a request from its structure alone (no model): tool
/// calls → `agent`; fenced code → `code`; a long context segment alongside a question
/// (≥2 messages) → `rag` (sentence pruning, not blanket-`aggressive`, which misfires on
/// RAG); everything else → `aggressive`. Backs the `auto` default.
pub fn route(req: &Request, provider: &dyn provider::Provider) -> &'static str {
    let raw = req.raw();
    if raw
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return "agent";
    }
    let texts: Vec<String> = provider
        .content_text_pointers(req)
        .iter()
        .filter_map(|p| req.get_str(p).map(str::to_string))
        .collect();
    if texts.iter().any(|t| t.contains("```")) {
        return "code";
    }
    // Turn count across every wire shape (Chat `messages`, Responses `input`, Gemini
    // `contents`) — not just `messages`, else Gemini/Responses RAG misroutes to aggressive.
    let turns = ["messages", "input", "contents"]
        .iter()
        .filter_map(|k| raw.get(*k).and_then(Value::as_array))
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    if turns >= 2 && texts.iter().any(|t| t.chars().count() >= 1200) {
        return "rag";
    }
    "aggressive"
}

/// Compress a provider request body (JSON), loading per-stage config from the
/// environment/default path. `provider` may be `None` to auto-detect from shape.
pub fn compress(input: &str, provider: Option<ProviderKind>) -> Result<CompressResult> {
    let config = config::DenseConfig::load().unwrap_or_else(|e| {
        eprintln!("llmtrim: {e}; using defaults");
        config::DenseConfig::default()
    });
    compress_with_config(input, provider, &config)
}

/// Compress with an explicit [`DenseConfig`] (no environment access — the
/// deterministic core used by tests and embedders).
///
/// The request is parsed into the neutral [`Request`], measured with the real
/// target tokenizer, and run through the gated stage pipeline.
pub fn compress_with_config(
    input: &str,
    provider: Option<ProviderKind>,
    config: &config::DenseConfig,
) -> Result<CompressResult> {
    let value: Value = serde_json::from_str(input).context("request body is not valid JSON")?;
    let kind = match provider {
        Some(k) => k,
        None => provider::detect(&value).context(
            "could not auto-detect provider from request shape; pass --provider openai|anthropic",
        )?,
    };
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);

    let counter = tokenizer::counter_for(kind, model.as_deref())?;
    let adapter = provider::for_kind(kind);
    let mut req = Request::from_value(kind, value);

    // `auto` resolves the preset from the request shape (structural, zero-model).
    let routed;
    let config = if config.auto {
        routed = config::DenseConfig::preset(route(&req, adapter.as_ref())).unwrap_or_default();
        &routed
    } else {
        config
    };

    let stages = stages_for(kind, config);
    let outcome = pipeline::run(&mut req, adapter.as_ref(), counter.as_ref(), &stages);

    Ok(CompressResult {
        request_json: req.to_json_string()?,
        plan: outcome.plan,
        provider: kind,
        model,
        tokenizer_label: counter.label().to_string(),
        tokenizer_exact: counter.is_exact(),
        input_tokens_before: outcome.input_tokens_before,
        input_tokens_after: outcome.input_tokens_after,
        stages: outcome.stages,
        output_shaped: config.output_control || config.output_compact_code,
    })
}

/// Reverse the lossless output transforms recorded in a rehydration plan. Internal: no
/// output-side transform ships today (Stage D is input-only; DSS was removed), so this is an
/// inert passthrough — a JSON response is normalized, plain text returned unchanged. Kept
/// `pub(crate)` as the interceptor's inverse hook; not part of the public API.
// Only the `intercept` transport calls this; allow it to be unused in the tokio-free
// embedder build (`--no-default-features`) without tripping `-D warnings`.
#[cfg_attr(not(feature = "intercept"), allow(dead_code))]
pub(crate) fn rehydrate(response: &str, _plan: &str) -> Result<String> {
    match serde_json::from_str::<Value>(response) {
        Ok(value) => {
            serde_json::to_string(&value).context("failed to serialize rehydrated response")
        }
        Err(_) => Ok(response.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_default_is_behavior_preserving() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let cfg = config::DenseConfig::default();
        let result =
            compress_with_config(input, Some(ProviderKind::OpenAi), &cfg).expect("compress");
        assert!(result.tokenizer_exact);
        let body: Value = serde_json::from_str(&result.request_json).unwrap();
        let msgs = body.get("messages").and_then(Value::as_array).unwrap();
        // Default = lossless input only: content intact, no injected system
        // instruction (the model's output behavior is unchanged).
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].get("content").and_then(Value::as_str), Some("hi"));
        assert!(
            !msgs
                .iter()
                .any(|m| m.get("role").and_then(Value::as_str) == Some("system")),
            "default must not change the model's output behavior"
        );
    }

    #[test]
    fn route_picks_preset_by_shape() {
        use serde_json::json;
        let p = provider::for_kind(ProviderKind::OpenAi);
        let mk = |v: Value| Request::from_value(ProviderKind::OpenAi, v);

        let tools = mk(json!({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f"}}]}));
        assert_eq!(route(&tools, p.as_ref()), "agent");

        let code =
            mk(json!({"messages":[{"role":"user","content":"fix:\n```rust\nfn x(){}\n```"}]}));
        assert_eq!(route(&code, p.as_ref()), "code");

        let long = "the report covers revenue and costs. ".repeat(60); // >1200 chars
        let rag = mk(json!({"messages":[{"role":"user","content":long},
            {"role":"user","content":"what was the revenue?"}]}));
        assert_eq!(route(&rag, p.as_ref()), "rag");

        let plain = mk(json!({"messages":[{"role":"user","content":"write a poem about spring"}]}));
        assert_eq!(route(&plain, p.as_ref()), "aggressive");
    }

    #[test]
    fn auto_routes_at_compress_time() {
        use serde_json::json;
        // auto on a tools request → agent preset → its long description gets trimmed.
        let input = json!({"model":"gpt-4o",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","description":"x".repeat(500)}}]})
        .to_string();
        let r = compress_with_config(
            &input,
            Some(ProviderKind::OpenAi),
            &config::DenseConfig::auto(),
        )
        .expect("compress");
        assert!(
            r.input_tokens_after < r.input_tokens_before,
            "auto routed to agent and trimmed the tool description"
        );
        assert!(
            !config::DenseConfig::default().auto,
            "plain default is not auto"
        );
    }

    #[test]
    fn compress_is_identity_when_all_stages_off() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let cfg = config::DenseConfig {
            hygiene: false,
            serialize: false,
            output_control: false,
            ..config::DenseConfig::default()
        };
        let result =
            compress_with_config(input, Some(ProviderKind::OpenAi), &cfg).expect("compress");
        let a: Value = serde_json::from_str(input).unwrap();
        let b: Value = serde_json::from_str(&result.request_json).unwrap();
        assert_eq!(a, b, "all stages off => identity");
        assert_eq!(result.input_tokens_before, result.input_tokens_after);
        assert!(result.input_tokens_before.0 > 0);
    }

    #[test]
    fn compress_auto_detects_anthropic() {
        let input = r#"{"system":"s","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let result = compress(input, None).expect("auto-detect anthropic");
        assert_eq!(result.provider, ProviderKind::Anthropic);
        assert!(!result.tokenizer_exact, "anthropic counts are approximate");
    }

    #[test]
    fn compress_rejects_invalid_json() {
        assert!(compress("not json", Some(ProviderKind::OpenAi)).is_err());
    }

    #[test]
    fn rehydrate_passes_through_without_transforms() {
        let resp = r#"{"content":"hello"}"#;
        let out = rehydrate(resp, "{}").expect("rehydrate");
        let a: Value = serde_json::from_str(resp).unwrap();
        let b: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn agent_preset_compresses_tool_result_diff() {
        // A 40-file diff returned as a tool_result — over the toolout file cap, so the
        // least-changed files are dropped to positional elision markers.
        let mut diff = String::new();
        for i in 0..40 {
            diff.push_str(&format!(
                "diff --git a/f{i}.rs b/f{i}.rs\n--- a/f{i}.rs\n+++ b/f{i}.rs\n\
                 @@ -1,3 +1,3 @@\n ctx_{i}\n-old line {i}\n+new line {i}\n trailing_{i}\n"
            ));
        }
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": diff}],
            }],
            "max_tokens": 1024,
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let result =
            compress_with_config(&input, Some(ProviderKind::Anthropic), &cfg).expect("compress");

        assert!(
            result.input_tokens_after < result.input_tokens_before,
            "toolout compressed the diff ({} -> {})",
            result.input_tokens_before,
            result.input_tokens_after
        );
        assert!(
            result.request_json.contains("omitted"),
            "dropped files left a positional elision marker in the body"
        );
    }

    #[test]
    fn frozen_prefix_untouched_while_live_zone_compresses() {
        // message 0 is the cached prefix (`cache_control`) holding a big log; message 1 is
        // the live user turn with another big log. The agent preset must compress the live
        // log but leave the cached one byte-identical — else it busts the prompt cache.
        let cached_log = (0..80)
            .map(|i| format!("INFO  step {i} routine nominal pass"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\nERROR failure inside the cached prefix";
        let live_log = (0..80)
            .map(|i| format!("DEBUG worker {i} idle waiting for work"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\nERROR failure inside the live turn";
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": cached_log,
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "b", "content": live_log}
                ]},
            ],
            "max_tokens": 1024,
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let result =
            compress_with_config(&input, Some(ProviderKind::Anthropic), &cfg).expect("compress");
        let body: Value = serde_json::from_str(&result.request_json).unwrap();

        let m0 = body
            .pointer("/messages/0/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(
            m0, cached_log,
            "cached prefix must be byte-identical (cache stays warm)"
        );

        let m1 = body
            .pointer("/messages/1/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert!(
            m1.len() < live_log.len(),
            "live turn was compressed ({} -> {})",
            live_log.len(),
            m1.len()
        );
    }
}
