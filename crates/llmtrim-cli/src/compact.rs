//! Claude Code compaction request detection and ordered model planning.
//!
//! Claude Code handles `/compact` locally and sends a normal Anthropic Messages request. The
//! internal summarization prompt is the protocol marker; ordinary user text mentioning `/compact`
//! must never trigger model substitution.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::reroute::{SubProvider, resolve_model};

const MARKERS: [&str; 3] = [
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
    "Your task is to create a detailed summary of the conversation so far",
    "an <analysis> block followed by a <summary> block",
];

/// Fallback ids for the `haiku`/`sonnet` aliases used only when the embedded models.dev snapshot
/// has no entry for the family (see [`direct_model`]); the live value is scanned from the snapshot
/// via [`llmtrim_core::latest_model_for_family`] so a new release doesn't need a code bump here.
pub const DEFAULT_HAIKU: &str = "claude-haiku-4-5";
pub const DEFAULT_SONNET: &str = "claude-sonnet-5";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// User-facing configured entry (`haiku`, a concrete id, …), or the original model id.
    pub logical_model: String,
    /// Model sent to the active upstream after subscription tier mapping.
    pub upstream_model: String,
    pub is_original: bool,
}

/// Detect Claude Code's internal compaction summarization request.
///
/// All three independently distinctive prompt markers are required in the final user turn. The
/// supporting request shape is checked to avoid treating a copied prompt fragment as protocol.
pub fn detect(body: &Value) -> Option<String> {
    let original_model = body.get("model")?.as_str()?.to_string();
    if body.get("stream").and_then(Value::as_bool) != Some(true)
        || body.get("max_tokens").and_then(Value::as_u64) != Some(64_000)
        || body
            .pointer("/output_config/effort")
            .and_then(Value::as_str)
            != Some("low")
    {
        return None;
    }
    let messages = body.get("messages")?.as_array()?;
    let last = messages.last()?;
    if last.get("role").and_then(Value::as_str) != Some("user") {
        return None;
    }
    let mut text = String::new();
    collect_text(last.get("content")?, &mut text);
    MARKERS
        .iter()
        .all(|marker| text.contains(marker))
        .then_some(original_model)
}

fn collect_text(value: &Value, out: &mut String) {
    match value {
        Value::String(text) => {
            out.push_str(text);
            out.push('\n');
        }
        Value::Array(items) => items.iter().for_each(|item| collect_text(item, out)),
        Value::Object(object) => {
            if let Some(text) = object.get("text").and_then(Value::as_str) {
                out.push_str(text);
                out.push('\n');
            }
        }
        _ => {}
    }
}

/// Resolve a `haiku`/`sonnet` family alias to the newest concrete id in the embedded models.dev
/// snapshot, falling back to the pinned default only if the family is missing from the snapshot.
/// Any other value is passed through as an explicit model id.
fn direct_model(model: &str) -> String {
    let alias = model.trim().to_ascii_lowercase();
    match alias.as_str() {
        "haiku" => llmtrim_core::latest_model_for_family("haiku")
            .unwrap_or_else(|| DEFAULT_HAIKU.to_string()),
        "sonnet" => llmtrim_core::latest_model_for_family("sonnet")
            .unwrap_or_else(|| DEFAULT_SONNET.to_string()),
        _ => model.trim().to_string(),
    }
}

/// Resolve configured alternatives for the active backend, append the original model implicitly,
/// and deduplicate by effective upstream model while preserving order.
pub fn plan(
    configured: &[String],
    original: &str,
    sub: Option<SubProvider>,
    tiers: &BTreeMap<String, String>,
) -> Vec<Candidate> {
    let mut out = Vec::new();
    for (logical, is_original) in configured
        .iter()
        .map(|model| (model.as_str(), false))
        .chain(std::iter::once((original, true)))
    {
        let logical_model = if is_original {
            original.to_string()
        } else {
            direct_model(logical)
        };
        let upstream_model = match sub {
            Some(provider) => resolve_model(provider, &logical_model, tiers),
            None => logical_model.clone(),
        };
        if out
            .iter()
            .any(|candidate: &Candidate| candidate.upstream_model == upstream_model)
        {
            continue;
        }
        out.push(Candidate {
            logical_model,
            upstream_model,
            is_original,
        });
    }
    out
}

/// Whether a known candidate context can hold both the compressed input and requested output.
pub fn fits(context_window: u64, input_tokens: i64, max_tokens: u64) -> bool {
    u64::try_from(input_tokens)
        .ok()
        .and_then(|input| input.checked_add(max_tokens))
        .is_some_and(|total| total <= context_window)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn compact_body(text: &str) -> Value {
        json!({
            "model": "claude-opus-4-8",
            "stream": true,
            "max_tokens": 64000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}]
        })
    }

    fn signature() -> String {
        format!("{}\n{}\n{}", MARKERS[0], MARKERS[1], MARKERS[2])
    }

    #[test]
    fn detects_verified_signature() {
        assert_eq!(
            detect(&compact_body(&signature())).as_deref(),
            Some("claude-opus-4-8")
        );
    }

    #[test]
    fn fails_closed_when_marker_or_shape_is_missing() {
        let mut body = compact_body(&format!("{}\n{}", MARKERS[0], MARKERS[1]));
        assert!(detect(&body).is_none());
        body = compact_body(&signature());
        body["output_config"]["effort"] = json!("high");
        assert!(detect(&body).is_none());
    }

    #[test]
    fn ordinary_compact_discussion_does_not_match() {
        assert!(detect(&compact_body("How does /compact work?")).is_none());
    }

    #[test]
    fn plan_appends_original_and_deduplicates_after_mapping() {
        let configured = vec!["haiku".into(), "sonnet".into(), "haiku".into()];
        let direct = plan(&configured, "claude-opus-4-8", None, &BTreeMap::new());
        assert_eq!(
            direct
                .iter()
                .map(|c| c.upstream_model.as_str())
                .collect::<Vec<_>>(),
            vec![DEFAULT_HAIKU, DEFAULT_SONNET, "claude-opus-4-8"]
        );
        let kimi = plan(
            &configured,
            "claude-opus-4-8",
            Some(SubProvider::Kimi),
            &BTreeMap::new(),
        );
        assert_eq!(kimi.len(), 1);
    }

    #[test]
    fn capacity_reserves_requested_output() {
        assert!(fits(200_000, 120_000, 64_000));
        assert!(!fits(200_000, 140_000, 64_000));
    }
}
