//! WebAssembly/JS bindings over [`llmtrim_core`] — the compression engine in a browser,
//! Node, or a Cloudflare Worker.
//!
//! The surface mirrors [`llmtrim-uniffi`](../llmtrim_uniffi): one [`compress`] call that
//! takes a provider-shaped request body and returns a plain JS object with the compressed
//! body, the per-stage report, and the before/after token counts.
//!
//! This build links `llmtrim-core` with `default-features = false`, so the C-linking and
//! large dependencies (tree-sitter, the tiktoken BPE vocab, image decoders) are dropped:
//! token counts come from the estimate tokenizer (approximate; savings percentages are
//! unchanged) and code skeletonization / image downscaling are no-ops.
//!
//! Build with `cargo` + `wasm-bindgen`, supplying the JS-backed getrandom backend via
//! RUSTFLAGS (see the crate README for the full recipe, including the optional `wasm-opt`
//! size pass):
//!
//! ```sh
//! RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
//!   cargo build -p llmtrim-wasm --release --target wasm32-unknown-unknown
//! wasm-bindgen target/wasm32-unknown-unknown/release/llmtrim_wasm.wasm \
//!   --out-dir pkg --target bundler   # or: nodejs | web
//! ```

use std::str::FromStr;

use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;
use serde::Serialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

/// What one pipeline stage did to the request, projected from
/// [`llmtrim_core::pipeline::StageReport`]. `tokens_before - tokens_after` is that stage's
/// own contribution to the input reduction.
#[derive(Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct StageReport {
    pub name: String,
    pub applied: bool,
    pub tokens_before: u64,
    pub tokens_after: u64,
    pub note: Option<String>,
}

/// The result of compressing one request body.
#[derive(Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct CompressOutput {
    pub request_json: String,
    pub provider: String,
    pub model: Option<String>,
    pub tokenizer_label: String,
    pub tokenizer_exact: bool,
    pub input_tokens_before: u64,
    pub input_tokens_after: u64,
    pub frozen_input_tokens: u64,
    pub output_shaped: bool,
    pub stages: Vec<StageReport>,
}

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

/// Compress an LLM API request body (a JSON string).
///
/// - `provider`: `"openai"`, `"anthropic"`, or `"google"`; omit (`null`/`undefined`) to
///   auto-detect from the body shape.
/// - `preset`: a named workload preset (`aggressive`, `agent`, `code`, `rag`, `safe`, …);
///   omit to use the shipped default (`auto` shape-routing), matching the native CLI.
///   Unlike the native crate, this binding never reads the environment or a config file
///   (there is none in a Worker), so the configuration comes only from the preset or the
///   `auto` default. Pass `"safe"` to restore the lossless-only behavior this binding had
///   before `auto` became the no-preset default.
///
/// Cross-binding note: with no preset, this binding applies `auto`, while the UniFFI binding
/// reads the host's environment/config. For output that matches every binding regardless of
/// host, pass an explicit preset.
///
/// Returns the [`CompressOutput`] as a typed JS object (TypeScript types are generated for
/// it), or throws on invalid JSON, an undetectable provider, or an unknown preset/provider
/// name.
#[wasm_bindgen]
pub fn compress(
    input: &str,
    provider: Option<String>,
    preset: Option<String>,
) -> Result<CompressOutput, JsError> {
    let kind = match provider.as_deref() {
        None => None,
        Some(p) => Some(ProviderKind::from_str(p).map_err(|e| JsError::new(&format!("{e:#}")))?),
    };
    let config = match preset.as_deref() {
        Some(name) => DenseConfig::preset(name)
            .ok_or_else(|| JsError::new(&format!("unknown preset: {name}")))?,
        None => DenseConfig::default(),
    };
    let result = llmtrim_core::compress_with_config(input, kind, &config)
        .map_err(|e| JsError::new(&format!("{e:#}")))?;
    Ok(project(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the binding's config/provider wiring without the wasm-bindgen JS boundary
    // (the JsValue projection is a thin serde step covered by serde-wasm-bindgen itself).
    fn run(input: &str, provider: Option<&str>, preset: Option<&str>) -> CompressOutput {
        let kind = provider.map(|p| ProviderKind::from_str(p).unwrap());
        let config = preset
            .map(|n| DenseConfig::preset(n).unwrap())
            .unwrap_or_default();
        project(llmtrim_core::compress_with_config(input, kind, &config).unwrap())
    }

    #[test]
    fn compresses_a_tool_heavy_agent_request() {
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
                "content": "ERROR boom\n".repeat(60)}]}],
            "max_tokens": 1024,
        })
        .to_string();
        let out = run(&input, Some("anthropic"), Some("agent"));
        assert_eq!(out.provider, "anthropic");
        assert!(out.input_tokens_after <= out.input_tokens_before);
        assert!(
            out.stages
                .iter()
                .any(|s| s.applied && s.tokens_after < s.tokens_before)
        );
    }

    #[test]
    fn no_preset_compresses_a_real_body_via_auto_default() {
        // The point of the `auto` no-preset default: a compressible body actually shrinks
        // when called with `None`, not just when an explicit preset is passed.
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
                "content": "ERROR boom\n".repeat(60)}]}],
            "max_tokens": 1024,
        })
        .to_string();
        let out = run(&input, Some("anthropic"), None);
        assert_eq!(out.provider, "anthropic");
        assert!(
            out.input_tokens_after < out.input_tokens_before,
            "auto default must compress a tool-heavy body with no preset"
        );
    }

    #[test]
    fn default_preset_preserves_a_basic_request() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let out = run(input, Some("openai"), None);
        assert_eq!(out.provider, "openai");
        assert_eq!(out.model.as_deref(), Some("gpt-4o"));
        // The `auto` default preserves a trivial message (nothing to shape away).
        assert!(out.request_json.contains("\"hi\""));
        // The real wasm build has no tiktoken, so counts are approximate; but a host
        // workspace build unifies `tiktoken` on from other crates, so the exactness flag
        // is build-variant-dependent and not asserted here. The count is non-zero either way.
        assert!(out.input_tokens_before > 0);
    }
}
