//! UniFFI surface over [`llmtrim_core`] — one Rust definition, idiomatic bindings for
//! Python, Ruby, Swift and Kotlin.
//!
//! The binding API is deliberately flat (FFI-shaped): the rich internal `CompressResult`
//! (rehydration plan, per-stage reports, newtype token counts) is projected onto a plain
//! [`CompressOutput`] record with scalar fields. Embedders that need the full plan should
//! depend on `llmtrim-core` directly in Rust.

use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;

uniffi::setup_scaffolding!();

/// Target LLM API shape. `None` on the call lets the engine auto-detect from the body.
#[derive(uniffi::Enum, Clone, Copy)]
pub enum Provider {
    OpenAi,
    Anthropic,
    Google,
}

impl From<Provider> for ProviderKind {
    fn from(p: Provider) -> Self {
        match p {
            Provider::OpenAi => ProviderKind::OpenAi,
            Provider::Anthropic => ProviderKind::Anthropic,
            Provider::Google => ProviderKind::Google,
        }
    }
}

/// What one pipeline stage did to the request, projected from
/// [`llmtrim_core::pipeline::StageReport`]. The token counts are over the content the
/// stage saw, so `tokens_before - tokens_after` is that stage's own contribution to the
/// total input reduction (mirrors the per-strategy breakdown other compressors report).
#[derive(uniffi::Record, Debug)]
pub struct StageReport {
    /// Stage identifier, e.g. `toolout`, `retrieve`, `dedup`, `output`.
    pub name: String,
    /// Whether the stage actually changed the request (a stage can run yet be a no-op, or
    /// be reverted by the token gate when it would not have saved tokens).
    pub applied: bool,
    /// Content tokens the stage saw before running.
    pub tokens_before: u64,
    /// Content tokens after the stage ran.
    pub tokens_after: u64,
    /// Optional human-readable note (e.g. why a stage was skipped or gated).
    pub note: Option<String>,
}

/// The result of compressing one request body.
#[derive(uniffi::Record, Debug)]
pub struct CompressOutput {
    /// The compressed, provider-shaped request body, ready to send.
    pub request_json: String,
    /// The provider the engine compressed for (after any auto-detection): `openai`,
    /// `anthropic` or `google`.
    pub provider: String,
    /// The `model` field from the request, if present.
    pub model: Option<String>,
    /// Human-readable tokenizer used for the counts (e.g. `cl100k_base`).
    pub tokenizer_label: String,
    /// Whether the token counts are exact (the provider's real tokenizer) or estimated.
    pub tokenizer_exact: bool,
    /// Input tokens before compression.
    pub input_tokens_before: u64,
    /// Input tokens after compression.
    pub input_tokens_after: u64,
    /// Tokens in the frozen (cache-controlled) prefix the stages skipped.
    pub frozen_input_tokens: u64,
    /// Whether output-shaping (Stage F) ran on this request.
    pub output_shaped: bool,
    /// Per-stage breakdown, in pipeline order. Each entry is one stage's own token
    /// contribution; sum the deltas of applied stages to attribute the total reduction.
    pub stages: Vec<StageReport>,
}

/// Errors surfaced to the bound language.
#[derive(uniffi::Error, Debug)]
pub enum LlmtrimError {
    /// The request body was not valid JSON, the provider could not be detected, or a
    /// stage failed. `detail` carries the full context chain. (Named `detail`, not
    /// `message`, because UniFFI maps this onto a Kotlin exception whose `Throwable`
    /// supertype already defines `message`.)
    Compress { detail: String },
    /// `preset` named a workload that does not exist.
    UnknownPreset { name: String },
}

impl std::fmt::Display for LlmtrimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmtrimError::Compress { detail } => write!(f, "{detail}"),
            LlmtrimError::UnknownPreset { name } => write!(f, "unknown preset: {name}"),
        }
    }
}

impl std::error::Error for LlmtrimError {}

fn project(r: llmtrim_core::CompressResult) -> CompressOutput {
    CompressOutput {
        request_json: r.request_json,
        provider: r.provider.as_str().to_string(),
        model: r.model,
        tokenizer_label: r.tokenizer_label,
        tokenizer_exact: r.tokenizer_exact,
        input_tokens_before: r.input_tokens_before.0 as u64,
        input_tokens_after: r.input_tokens_after.0 as u64,
        frozen_input_tokens: r.frozen_input_tokens.0 as u64,
        output_shaped: r.output_shaped,
        stages: r
            .stages
            .into_iter()
            .map(|s| StageReport {
                name: s.name,
                applied: s.applied,
                tokens_before: s.tokens_before.0 as u64,
                tokens_after: s.tokens_after.0 as u64,
                note: s.note,
            })
            .collect(),
    }
}

/// Compress an LLM API request body (JSON string).
///
/// - `provider`: the target API shape, or `None` to auto-detect from the body.
/// - `preset`: a named workload preset (`aggressive`, `agent`, `code`, `rag`, `safe`, …)
///   to compress with. `None` uses the configuration from the environment / config file
///   (the same defaults the `llmtrim` CLI uses).
#[uniffi::export]
pub fn compress(
    input: String,
    provider: Option<Provider>,
    preset: Option<String>,
) -> Result<CompressOutput, LlmtrimError> {
    let kind = provider.map(ProviderKind::from);
    let result = match preset {
        Some(name) => {
            let config = DenseConfig::preset(&name).ok_or(LlmtrimError::UnknownPreset { name })?;
            llmtrim_core::compress_with_config(&input, kind, &config)
        }
        None => llmtrim_core::compress(&input, kind),
    };
    result.map(project).map_err(|e| LlmtrimError::Compress {
        detail: format!("{e:#}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_roundtrips_a_basic_openai_request() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let out = compress(input.to_string(), Some(Provider::OpenAi), None).expect("compress");
        assert_eq!(out.provider, "openai");
        assert_eq!(out.model.as_deref(), Some("gpt-4o"));
        assert!(out.tokenizer_exact);
        assert!(out.input_tokens_before > 0);
    }

    #[test]
    fn preset_string_dispatch_reaches_core() {
        // FFI-layer concern: a `preset` string selects the right DenseConfig and the
        // Anthropic `provider` enum maps through. The magnitude of compression is covered
        // by llmtrim-core's own eval tests — here we only assert the binding wired it up
        // (a known-compressing input shrinks, and the projection fields are coherent).
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
                "content": "ERROR boom\n".repeat(60)}]}],
            "max_tokens": 1024,
        })
        .to_string();
        let out =
            compress(input, Some(Provider::Anthropic), Some("agent".into())).expect("compress");
        assert_eq!(out.provider, "anthropic");
        assert!(out.input_tokens_after <= out.input_tokens_before);
        assert!(!out.request_json.is_empty());
        // The per-stage breakdown projects through: a tool-output-heavy request under the
        // `agent` preset runs at least one stage, and at least one of them applies and saves.
        assert!(!out.stages.is_empty());
        assert!(
            out.stages
                .iter()
                .any(|s| s.applied && s.tokens_after < s.tokens_before)
        );
    }

    #[test]
    fn unknown_preset_is_an_error() {
        let err = compress("{}".into(), Some(Provider::OpenAi), Some("nope".into())).unwrap_err();
        assert!(matches!(err, LlmtrimError::UnknownPreset { .. }));
    }

    #[test]
    fn invalid_json_is_a_compress_error() {
        let err = compress("not json".into(), Some(Provider::OpenAi), None).unwrap_err();
        assert!(matches!(err, LlmtrimError::Compress { .. }));
    }
}
