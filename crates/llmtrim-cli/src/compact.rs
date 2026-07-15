//! Claude Code compaction request detection and ordered model planning.
//!
//! Claude Code handles `/compact` locally and sends a normal Anthropic Messages request. The
//! internal summarization prompt is the protocol marker; ordinary user text mentioning `/compact`
//! must never trigger model substitution.

use std::collections::BTreeMap;

use serde_json::{Value, json};

use crate::reroute::{SubProvider, resolve_model};

pub(crate) const MARKERS: [&str; 3] = [
    "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.",
    "Your task is to create a detailed summary of the conversation so far",
    "an <analysis> block followed by a <summary> block",
];

/// Fallback ids for the `haiku`/`sonnet` aliases used only when the embedded models.dev snapshot
/// has no entry for the family (see `direct_model`); the live value is scanned from the snapshot
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

/// Thinking budget (tokens) substituted for `adaptive` when a compact candidate can't take the
/// adaptive mode. Ample for a summarization turn's reasoning; clamped below the request's
/// `max_tokens` (and above Anthropic's 1024 floor) so it's valid on every model.
const COMPACT_THINKING_BUDGET: u64 = 8_192;

/// Reconcile a compact candidate's `thinking` block with the swapped-in model. Claude Code's
/// `/compact` uses `thinking.type = "adaptive"`, which only some models accept — the Haiku family
/// rejects it ("adaptive thinking is not supported on this model") even though it *is*
/// reasoning-capable. So:
///
/// - a reasoning-capable model keeps thinking, with `adaptive` downgraded to an explicit
///   `enabled` budget (universally accepted);
/// - a non-reasoning model has the block stripped entirely.
///
/// Only this isolated compact request is edited — the client's ordinary turns keep their thinking,
/// so it resumes after compaction. Reasoning capability comes from the embedded models.dev flag
/// ([`llmtrim_core::model_is_reasoning_capable`]), not a hardcoded model list. Returns `true` when
/// the body was modified.
pub fn normalize_compact_thinking(body: &mut Value, model: &str, max_tokens: u64) -> bool {
    if body.pointer("/thinking/type").and_then(Value::as_str) != Some("adaptive") {
        return false;
    }
    if llmtrim_core::model_is_reasoning_capable(model) {
        // Keep thinking, but as an explicit budget every model accepts. Stay under `max_tokens`
        // (Anthropic requires `budget_tokens < max_tokens`) and above the 1024 floor.
        let budget = COMPACT_THINKING_BUDGET
            .min(max_tokens.saturating_sub(1))
            .max(1_024);
        body["thinking"] = json!({ "type": "enabled", "budget_tokens": budget });
    } else if let Some(obj) = body.as_object_mut() {
        obj.remove("thinking");
    }
    true
}

/// Drop assistant `thinking`/`redacted_thinking` blocks from the conversation history. Anthropic
/// binds a thinking block's `signature` to the model (tier) that produced it, so a swapped-in
/// compact candidate can't validate the original model's signatures ("Invalid signature in thinking
/// block") and rejects the whole request. Compaction summarizes the text/tool content, so dropping
/// the history's reasoning is safe; a turn left with no content blocks is removed so the request
/// stays well-formed. Only this isolated compact request is edited. Returns `true` when the body
/// was modified.
pub fn strip_history_thinking(body: &mut Value) -> bool {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return false;
    };
    let mut changed = false;
    for message in messages.iter_mut() {
        if message.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        let before = content.len();
        content.retain(|block| {
            !matches!(
                block.get("type").and_then(Value::as_str),
                Some("thinking") | Some("redacted_thinking")
            )
        });
        changed |= content.len() != before;
    }
    // An assistant turn stripped down to nothing would be an empty-content error: drop it whole.
    if changed && let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        messages.retain(|message| {
            message
                .get("content")
                .and_then(Value::as_array)
                .map(|blocks| blocks.is_empty())
                != Some(true)
        });
    }
    changed
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
    fn adaptive_thinking_downgrades_to_a_budget_when_the_model_can_reason() {
        // A reasoning-capable model keeps thinking, with `adaptive` swapped for an explicit budget
        // (clamped below max_tokens, above the 1024 floor). The decision is the models.dev flag, not
        // the id — this fixture just needs one model the registry marks reasoning-capable.
        assert!(llmtrim_core::model_is_reasoning_capable("claude-haiku-4-5"));
        let mut body = compact_body("x");
        assert!(normalize_compact_thinking(
            &mut body,
            "claude-haiku-4-5",
            64_000
        ));
        assert_eq!(body["thinking"]["type"], "enabled");
        let budget = body["thinking"]["budget_tokens"].as_u64().unwrap();
        assert!((1_024..64_000).contains(&budget), "{budget}");
    }

    #[test]
    fn adaptive_thinking_is_stripped_when_the_model_cannot_reason() {
        // A model the registry marks non-reasoning can't take any thinking block: strip it.
        assert!(!llmtrim_core::model_is_reasoning_capable("gpt-4o-mini"));
        let mut body = compact_body("x");
        assert!(normalize_compact_thinking(&mut body, "gpt-4o-mini", 64_000));
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn history_thinking_blocks_are_stripped_but_other_content_kept() {
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "opus-sig"},
                    {"type": "text", "text": "answer"}
                ]},
                {"role": "user", "content": "go"}
            ]
        });
        assert!(strip_history_thinking(&mut body));
        let assistant = &body["messages"][1]["content"];
        assert_eq!(assistant.as_array().unwrap().len(), 1);
        assert_eq!(assistant[0]["type"], "text");
        // User turns and a string-content turn are untouched.
        assert_eq!(body["messages"][0]["content"], "start");
    }

    #[test]
    fn assistant_turn_reduced_to_nothing_is_dropped() {
        let mut body = json!({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "only reasoning", "signature": "s"}
                ]},
                {"role": "user", "content": "summarize"}
            ]
        });
        assert!(strip_history_thinking(&mut body));
        // The pure-thinking assistant turn is removed so no empty-content block is sent.
        let roles: Vec<&str> = body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["user", "user"]);
    }

    #[test]
    fn strip_history_thinking_is_a_noop_without_thinking() {
        let mut body = json!({
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]
        });
        assert!(!strip_history_thinking(&mut body));
    }

    #[test]
    fn non_adaptive_thinking_is_left_untouched() {
        // Only `adaptive` needs reconciling; an explicit budget is already universal.
        let mut body = json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 64000,
            "thinking": {"type": "enabled", "budget_tokens": 8000},
        });
        assert!(!normalize_compact_thinking(
            &mut body,
            "claude-haiku-4-5",
            64_000
        ));
        assert_eq!(body["thinking"]["budget_tokens"], 8000);
    }

    #[test]
    fn budget_is_clamped_below_a_small_max_tokens() {
        // Defensive: if the request cap were ever tiny, the budget must stay strictly under it.
        let mut body = compact_body("x");
        body["max_tokens"] = json!(1500);
        assert!(normalize_compact_thinking(
            &mut body,
            "claude-haiku-4-5",
            1500
        ));
        assert!(body["thinking"]["budget_tokens"].as_u64().unwrap() < 1500);
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
