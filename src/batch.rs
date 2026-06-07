//! Stage I — Batch. Compress every request in a batch JSONL so non-interactive
//! jobs stack llmtrim's compression *on top of* the provider's ~50% batch
//! discount.
//!
//! llmtrim rewrites the request bodies; the caller submits the compressed JSONL
//! to the provider's Batch API (the discount + async execution are provider-side,
//! and that upload/poll flow is provider-specific multipart, out of scope here).
//! Handles the OpenAI batch envelope `{custom_id, method, url, body}` and bare
//! request objects.

use anyhow::{Context, Result};
use serde_json::Value;

use crate::config::DenseConfig;
use crate::ir::ProviderKind;

/// One compressed batch line + its token measurement.
pub struct BatchLine {
    pub line: String,
    pub tokens_before: usize,
    pub tokens_after: usize,
}

/// Compress the request in one batch JSONL line, preserving an OpenAI batch
/// envelope (`{custom_id, method, url, body}`) when present.
pub fn compress_line(
    line: &str,
    provider: ProviderKind,
    config: &DenseConfig,
) -> Result<BatchLine> {
    let mut value: Value = serde_json::from_str(line).context("batch line is not valid JSON")?;
    let has_envelope = value.get("body").is_some_and(Value::is_object);

    let body_str = if has_envelope {
        value["body"].to_string()
    } else {
        line.to_string()
    };
    let result = crate::compress_with_config(&body_str, Some(provider), config)?;

    let line = if has_envelope {
        value["body"] = serde_json::from_str(&result.request_json)
            .context("compressed body re-parse failed")?;
        serde_json::to_string(&value).context("failed to serialize batch line")?
    } else {
        result.request_json
    };

    Ok(BatchLine {
        line,
        tokens_before: result.input_tokens_before.0,
        tokens_after: result.input_tokens_after.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn compresses_openai_batch_envelope_preserving_metadata() {
        let pretty =
            serde_json::to_string_pretty(&json!({"a": 1, "b": 2, "c": 3, "d": 4})).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":pretty}]});
        let line = json!({
            "custom_id":"req-1","method":"POST","url":"/v1/chat/completions","body":body
        })
        .to_string();

        let out = compress_line(&line, ProviderKind::OpenAi, &DenseConfig::default()).unwrap();
        let v: Value = serde_json::from_str(&out.line).unwrap();
        // Envelope metadata is preserved...
        assert_eq!(v["custom_id"], "req-1");
        assert_eq!(v["url"], "/v1/chat/completions");
        // ...and the body is still a valid request (the pipeline ran on it).
        assert!(v["body"]["messages"].is_array());
        assert!(out.tokens_before > 0);
    }

    #[test]
    fn compresses_bare_request_line() {
        let body =
            json!({"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}).to_string();
        let out = compress_line(&body, ProviderKind::OpenAi, &DenseConfig::default()).unwrap();
        let v: Value = serde_json::from_str(&out.line).unwrap();
        assert!(v["messages"].is_array(), "bare request stays a request");
    }

    #[test]
    fn invalid_line_is_an_error() {
        assert!(compress_line("{not json", ProviderKind::OpenAi, &DenseConfig::default()).is_err());
    }
}
