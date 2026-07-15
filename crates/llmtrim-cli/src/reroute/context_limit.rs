//! Session-scoped guard for subscription context-window rejections.
//!
//! When a Codex/Kimi backend refuses a turn because the conversation no longer fits its context
//! window, retrying or compressing cannot recover: the *history* is too large. This module:
//!
//! 1. Keeps a last-known-good Anthropic request snapshot per Claude Code session
//!    (`x-claude-code-session-id`).
//! 2. On a pre-output context-limit rejection, marks the session `must_compact` and returns a
//!    local explanation (no retry, no truncate, no llmtrim compression).
//! 3. While blocked, rejects ordinary turns locally and only allows Claude Code's real `/compact`
//!    request — rewritten to use the stored snapshot — or a cleared/new session.
//!
//! State is process-local, TTL-bound, and capped (same spirit as [`super::continuation`]).

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;

/// Client-facing explanation when a subscription backend rejected a turn for context length, or
/// when a blocked session receives another ordinary request.
pub const GUARD_MESSAGE: &str = "\
Subscription context limit reached.\n\
This turn was not accepted by the subscription backend. Run /compact to continue this session, or /clear to start fresh. After compaction, resend your last request.";

const TTL_MS: u64 = 30 * 60 * 1000;
const MAX_STATES: usize = 10_000;
/// Cap a single snapshot so a pathological history cannot pin tens of MB per session.
const MAX_SNAPSHOT_BYTES: u64 = 2_000_000;
const MAX_TOTAL_SNAPSHOT_BYTES: u64 = 20_000_000;

#[derive(Clone)]
struct SessionState {
    must_compact: bool,
    /// Anthropic-shaped body of the last request the backend accepted (messages + system).
    last_good: Option<Value>,
    last_good_bytes: u64,
    updated_at: u64,
}

static STATES: Mutex<Option<HashMap<String, SessionState>>> = Mutex::new(None);
static TOTAL_BYTES: Mutex<u64> = Mutex::new(0);

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Slim the full Anthropic body down to the fields compaction needs to restore history.
fn snapshot_body(body: &Value) -> Value {
    let mut out = serde_json::Map::new();
    if let Some(messages) = body.get("messages") {
        out.insert("messages".to_string(), messages.clone());
    }
    if let Some(system) = body.get("system") {
        out.insert("system".to_string(), system.clone());
    }
    Value::Object(out)
}

fn message_count(body: &Value) -> usize {
    body.get("messages")
        .and_then(Value::as_array)
        .map(|a| a.len())
        .unwrap_or(0)
}

/// True when `text` (HTTP body, SSE error message, or status phrase) describes a context-window
/// rejection from Codex, Kimi, OpenAI-compatible, or Anthropic-shaped error payloads.
pub fn is_context_limit_text(text: &str) -> bool {
    let m = text.to_lowercase();
    if m.is_empty() {
        return false;
    }
    // Explicit codes / types first.
    if m.contains("context_length_exceeded")
        || m.contains("context_length")
        || m.contains("max_tokens_exceeded")
        || m.contains("token_limit")
        || m.contains("prompt_too_long")
        || m.contains("input_too_long")
        || m.contains("string_above_max_length")
    {
        return true;
    }
    // Natural-language forms used by ChatGPT/Codex, Kimi, and Anthropic.
    let contextish = m.contains("context")
        && (m.contains("window")
            || m.contains("length")
            || m.contains("limit")
            || m.contains("overflow")
            || m.contains("exceed"));
    let tokenish =
        (m.contains("too many tokens") || m.contains("too long") || m.contains("maximum"))
            && (m.contains("token")
                || m.contains("prompt")
                || m.contains("input")
                || m.contains("message"));
    let exceeds_model = m.contains("exceeds")
        && (m.contains("context") || m.contains("maximum") || m.contains("model"));
    contextish || tokenish || exceeds_model
}

/// HTTP-path detection: non-success status whose body (or extracted error message) is a
/// context-window rejection. Success statuses are handled via the SSE reducer path instead.
pub fn is_context_limit_http(status: u16, body: &[u8]) -> bool {
    if (200..300).contains(&status) {
        return false;
    }
    // Most providers use 400; some return 413. Be liberal on status when the body is clear.
    let text = String::from_utf8_lossy(body);
    if is_context_limit_text(&text) {
        return true;
    }
    if let Ok(v) = serde_json::from_slice::<Value>(body) {
        let msg = v
            .pointer("/error/message")
            .or_else(|| v.pointer("/error/code"))
            .or_else(|| v.get("detail"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if is_context_limit_text(msg) {
            return true;
        }
        let code = v
            .pointer("/error/type")
            .or_else(|| v.pointer("/error/code"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if is_context_limit_text(code) {
            return true;
        }
    }
    false
}

/// Whether this session is currently blocked pending `/compact` (or a cleared session).
pub fn must_compact(session_id: Option<&str>) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    let now = now_ms();
    let guard = STATES.lock().unwrap();
    guard.as_ref().is_some_and(|m| {
        m.get(session_id)
            .is_some_and(|s| s.must_compact && now.saturating_sub(s.updated_at) <= TTL_MS)
    })
}

/// Mark the session blocked after a pre-output context-limit rejection. Does not replace an
/// existing last-known-good snapshot (the rejected turn is intentionally discarded).
pub fn mark_must_compact(session_id: Option<&str>) {
    let Some(session_id) = session_id else {
        return;
    };
    let now = now_ms();
    let mut guard = STATES.lock().unwrap();
    let map = guard.get_or_insert_with(HashMap::new);
    match map.get_mut(session_id) {
        Some(state) => {
            state.must_compact = true;
            state.updated_at = now;
        }
        None => {
            map.insert(
                session_id.to_string(),
                SessionState {
                    must_compact: true,
                    last_good: None,
                    last_good_bytes: 0,
                    updated_at: now,
                },
            );
        }
    }
    drop(guard);
    evict_oldest();
}

/// Record a request the subscription backend accepted (content committed / successful completion).
/// Clears `must_compact` — a successful compact or ordinary turn unblocks the session.
pub fn record_accepted(session_id: Option<&str>, anthropic_body: &Value) {
    let Some(session_id) = session_id else {
        return;
    };
    let snap = snapshot_body(anthropic_body);
    let bytes = serde_json::to_vec(&snap)
        .map(|b| b.len() as u64)
        .unwrap_or(0);
    if bytes == 0 || bytes > MAX_SNAPSHOT_BYTES {
        // Oversized or empty: keep must_compact clear but drop the snapshot rather than OOM.
        let mut guard = STATES.lock().unwrap();
        if let Some(map) = guard.as_mut()
            && let Some(state) = map.get_mut(session_id)
        {
            if let Some(old) = state.last_good.take() {
                let old_b = state.last_good_bytes;
                state.last_good_bytes = 0;
                drop(old);
                let mut total = TOTAL_BYTES.lock().unwrap();
                *total = total.saturating_sub(old_b);
            }
            state.must_compact = false;
            state.updated_at = now_ms();
        }
        return;
    }

    // Evict prior bytes for this session first.
    clear_snapshot_bytes(session_id);

    let state = SessionState {
        must_compact: false,
        last_good: Some(snap),
        last_good_bytes: bytes,
        updated_at: now_ms(),
    };
    {
        let mut total = TOTAL_BYTES.lock().unwrap();
        *total = total.saturating_add(bytes);
    }
    {
        let mut guard = STATES.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        map.insert(session_id.to_string(), state);
    }
    evict_oldest();
}

fn clear_snapshot_bytes(session_id: &str) {
    let mut guard = STATES.lock().unwrap();
    if let Some(map) = guard.as_mut()
        && let Some(state) = map.get_mut(session_id)
        && let Some(old) = state.last_good.take()
    {
        let old_b = state.last_good_bytes;
        state.last_good_bytes = 0;
        drop(old);
        let mut total = TOTAL_BYTES.lock().unwrap();
        *total = total.saturating_sub(old_b);
    }
}

/// Drop all guard state for a session (session reset / `/clear` with a new id naturally misses).
pub fn clear(session_id: Option<&str>) {
    let Some(session_id) = session_id else {
        return;
    };
    let mut guard = STATES.lock().unwrap();
    if let Some(map) = guard.as_mut()
        && let Some(existing) = map.remove(session_id)
    {
        let mut total = TOTAL_BYTES.lock().unwrap();
        *total = total.saturating_sub(existing.last_good_bytes);
    }
}

/// True when the inbound request looks like a fresh conversation on the same session id
/// (history shorter than the stored snapshot — e.g. after a local clear that reused the id).
pub fn looks_like_session_reset(session_id: Option<&str>, body: &Value) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    let cur = message_count(body);
    if cur == 0 {
        return true;
    }
    let guard = STATES.lock().unwrap();
    let Some(state) = guard.as_ref().and_then(|m| m.get(session_id)) else {
        return false;
    };
    let good = state.last_good.as_ref().map(message_count).unwrap_or(0);
    // A reset is shorter than what we last accepted; ordinary turns only grow (or stay
    // similar when the client re-sends). Compact requests keep the full overflowing history
    // and are handled separately.
    good > 0 && cur < good
}

/// While `must_compact`, rewrite a real Claude Code compact request so its conversation history
/// is the last-known-good snapshot, preserving the final summarization user turn. Returns whether
/// the body was rewritten.
pub fn apply_last_good_to_compact(session_id: Option<&str>, body: &mut Value) -> bool {
    let Some(session_id) = session_id else {
        return false;
    };
    let last_good = {
        let guard = STATES.lock().unwrap();
        guard
            .as_ref()
            .and_then(|m| m.get(session_id))
            .and_then(|s| s.last_good.clone())
    };
    let Some(last_good) = last_good else {
        return false;
    };
    let Some(good_messages) = last_good.get("messages").and_then(Value::as_array).cloned() else {
        return false;
    };
    let compact_prompt = body
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|a| a.last().cloned());
    let mut messages = good_messages;
    if let Some(prompt) = compact_prompt {
        messages.push(prompt);
    }
    if let Some(obj) = body.as_object_mut() {
        obj.insert("messages".to_string(), Value::Array(messages));
        // Prefer the system prompt that last fit, when the snapshot carried one.
        if let Some(system) = last_good.get("system") {
            obj.insert("system".to_string(), system.clone());
        }
        return true;
    }
    false
}

/// Decision for an inbound subscription request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InboundAction {
    /// Forward normally (optionally after compact rewrite).
    Allow { rewrote_compact: bool },
    /// Reject locally with [`GUARD_MESSAGE`].
    Block,
}

/// Gate an inbound Anthropic `/v1/messages` body for a subscription-backed session.
///
/// - Not blocked → allow.
/// - Blocked + real `/compact` → rewrite history from last-known-good and allow.
/// - Blocked + session reset → clear state and allow.
/// - Blocked + ordinary turn → block.
///
/// Compact is checked before the reset heuristic: a compact request often carries fewer
/// messages than the overflowing local transcript (or just a long history that still
/// happens to be shorter than a prior snapshot after client-side trimming), and must not
/// be mistaken for `/clear`.
pub fn gate_inbound(session_id: Option<&str>, body: &mut Value, is_compact: bool) -> InboundAction {
    if !must_compact(session_id) {
        return InboundAction::Allow {
            rewrote_compact: false,
        };
    }
    if is_compact {
        let rewrote = apply_last_good_to_compact(session_id, body);
        return InboundAction::Allow {
            rewrote_compact: rewrote,
        };
    }
    if looks_like_session_reset(session_id, body) {
        clear(session_id);
        return InboundAction::Allow {
            rewrote_compact: false,
        };
    }
    InboundAction::Block
}

fn evict_oldest() {
    let mut guard = STATES.lock().unwrap();
    let map = match guard.as_mut() {
        Some(m) if !m.is_empty() => m,
        _ => return,
    };
    let now = now_ms();
    // Drop expired first.
    let expired: Vec<String> = map
        .iter()
        .filter(|(_, s)| now.saturating_sub(s.updated_at) > TTL_MS)
        .map(|(k, _)| k.clone())
        .collect();
    let mut total = TOTAL_BYTES.lock().unwrap();
    for key in expired {
        if let Some(old) = map.remove(&key) {
            *total = total.saturating_sub(old.last_good_bytes);
        }
    }
    while (map.len() > MAX_STATES || *total > MAX_TOTAL_SNAPSHOT_BYTES) && !map.is_empty() {
        if let Some((oldest_key, _)) = map.iter().min_by_key(|(_, s)| s.updated_at) {
            let key = oldest_key.clone();
            if let Some(old) = map.remove(&key) {
                *total = total.saturating_sub(old.last_good_bytes);
            }
        } else {
            break;
        }
    }
}

// ── test helpers ──────────────────────────────────────────────────────────────

/// Process-global state is shared across tests; serialize them so `clear_all_for_tests`
/// cannot wipe another test mid-flight.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub fn clear_all_for_tests() {
    let mut guard = STATES.lock().unwrap();
    *guard = None;
    let mut b = TOTAL_BYTES.lock().unwrap();
    *b = 0;
}

#[cfg(test)]
pub fn last_good_message_count_for_tests(session_id: &str) -> Option<usize> {
    let guard = STATES.lock().unwrap();
    guard
        .as_ref()
        .and_then(|m| m.get(session_id))
        .and_then(|s| s.last_good.as_ref())
        .map(message_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn body_with_n_messages(n: usize) -> Value {
        let messages: Vec<Value> = (0..n)
            .map(|i| {
                json!({
                    "role": if i % 2 == 0 { "user" } else { "assistant" },
                    "content": format!("m{i}")
                })
            })
            .collect();
        json!({ "model": "claude-opus-4-8", "messages": messages })
    }

    fn compact_body(history_n: usize) -> Value {
        let mut messages: Vec<Value> = (0..history_n)
            .map(|i| {
                json!({
                    "role": if i % 2 == 0 { "user" } else { "assistant" },
                    "content": format!("h{i}")
                })
            })
            .collect();
        messages.push(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": format!(
                    "{}\n{}\n{}",
                    crate::compact::MARKERS[0],
                    crate::compact::MARKERS[1],
                    crate::compact::MARKERS[2]
                )}
            ]
        }));
        json!({
            "model": "claude-opus-4-8",
            "stream": true,
            "max_tokens": 64000,
            "output_config": {"effort": "low"},
            "messages": messages
        })
    }

    #[test]
    fn detects_context_limit_phrases_for_codex_and_kimi() {
        let _lock = test_lock();
        assert!(is_context_limit_text(
            "Your input exceeds the context window of this model."
        ));
        assert!(is_context_limit_text("context_length_exceeded"));
        assert!(is_context_limit_text(
            "This model's maximum context length is 128000 tokens."
        ));
        assert!(is_context_limit_text("prompt is too long: 200000 tokens"));
        assert!(is_context_limit_http(
            400,
            br#"{"error":{"message":"Your input exceeds the context window of this model.","type":"invalid_request_error"}}"#
        ));
        assert!(is_context_limit_http(
            400,
            br#"{"error":{"code":"context_length_exceeded","message":"too big"}}"#
        ));
        assert!(!is_context_limit_text("rate limit reached"));
        assert!(!is_context_limit_http(
            429,
            br#"{"error":{"message":"rate limit"}}"#
        ));
        assert!(!is_context_limit_http(
            200,
            br#"{"error":{"message":"context_length_exceeded"}}"#
        ));
    }

    #[test]
    fn block_compact_resume_flow() {
        let _lock = test_lock();
        clear_all_for_tests();
        let sid = Some("sess-guard-1");

        // Successful history lands in the snapshot.
        let good = body_with_n_messages(4);
        record_accepted(sid, &good);
        assert!(!must_compact(sid));
        assert_eq!(last_good_message_count_for_tests("sess-guard-1"), Some(4));

        // Context limit on the next turn: block, keep snapshot.
        mark_must_compact(sid);
        assert!(must_compact(sid));
        assert_eq!(last_good_message_count_for_tests("sess-guard-1"), Some(4));

        // Ordinary request while blocked.
        let mut ordinary = body_with_n_messages(5);
        assert_eq!(
            gate_inbound(sid, &mut ordinary, false),
            InboundAction::Block
        );

        // Real compact: rewrite history to LKG + compact prompt.
        let mut compact = compact_body(20);
        let action = gate_inbound(sid, &mut compact, true);
        assert_eq!(
            action,
            InboundAction::Allow {
                rewrote_compact: true
            }
        );
        let msgs = compact["messages"].as_array().unwrap();
        // 4 from LKG + 1 compact prompt.
        assert_eq!(msgs.len(), 5);
        assert_eq!(msgs[0]["content"], "m0");
        assert!(
            msgs[4]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains(crate::compact::MARKERS[0])
        );

        // Successful compact clears the block and refreshes the snapshot.
        record_accepted(sid, &compact);
        assert!(!must_compact(sid));
        assert_eq!(last_good_message_count_for_tests("sess-guard-1"), Some(5));

        // Ordinary traffic resumes.
        let mut next = body_with_n_messages(2);
        assert_eq!(
            gate_inbound(sid, &mut next, false),
            InboundAction::Allow {
                rewrote_compact: false
            }
        );
    }

    #[test]
    fn session_reset_clears_block() {
        let _lock = test_lock();
        clear_all_for_tests();
        let sid = Some("sess-reset");
        record_accepted(sid, &body_with_n_messages(6));
        mark_must_compact(sid);
        assert!(must_compact(sid));

        // A short new conversation is treated as /clear on a reused id.
        let mut fresh = body_with_n_messages(1);
        assert_eq!(
            gate_inbound(sid, &mut fresh, false),
            InboundAction::Allow {
                rewrote_compact: false
            }
        );
        assert!(!must_compact(sid));
    }

    #[test]
    fn clear_drops_state() {
        let _lock = test_lock();
        clear_all_for_tests();
        let sid = Some("sess-clear");
        record_accepted(sid, &body_with_n_messages(3));
        mark_must_compact(sid);
        clear(sid);
        assert!(!must_compact(sid));
        assert_eq!(last_good_message_count_for_tests("sess-clear"), None);
    }

    #[test]
    fn rejected_turn_is_not_recorded_as_last_good() {
        let _lock = test_lock();
        clear_all_for_tests();
        let sid = Some("sess-no-replay");
        record_accepted(sid, &body_with_n_messages(2));
        // Overflowing turn is never passed to record_accepted — only mark_must_compact.
        mark_must_compact(sid);
        assert_eq!(last_good_message_count_for_tests("sess-no-replay"), Some(2));
    }

    #[test]
    fn missing_session_never_blocks() {
        let _lock = test_lock();
        clear_all_for_tests();
        mark_must_compact(None);
        assert!(!must_compact(None));
        let mut body = body_with_n_messages(3);
        assert_eq!(
            gate_inbound(None, &mut body, false),
            InboundAction::Allow {
                rewrote_compact: false
            }
        );
    }

    #[test]
    fn guard_message_tells_user_to_compact_or_clear() {
        let _lock = test_lock();
        assert!(GUARD_MESSAGE.contains("Subscription context limit reached."));
        assert!(GUARD_MESSAGE.contains("Run /compact"));
        assert!(GUARD_MESSAGE.contains("/clear"));
        assert!(GUARD_MESSAGE.contains("resend your last request"));
    }
}
