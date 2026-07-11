//! Server-side continuation for Codex (Responses API) reroute.
//!
//! Ports the core idea from claude-code-proxy:
//! - Use `previous_response_id` + a small `input` delta instead of resending the full
//!   conversation history on every turn.
//! - This keeps the backend's internal state (and associated caching/prefix reuse) warm.
//! - Combined with `prompt_cache_key`, this should produce better `cached_tokens` reports
//!   from the backend, which then translate into higher "♻ % cached" in the status line.
//!
//! We work with raw serde_json::Value because the reroute path builds codex bodies as Value.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{Value, json};

const TTL_MS: u64 = 30 * 60 * 1000;
const MAX_STATES: usize = 10_000;
const MAX_SESSION_TRANSCRIPT_BYTES: u64 = 2_000_000;
const MAX_TOTAL_TRANSCRIPT_BYTES: u64 = 20_000_000;

#[derive(Clone)]
struct ContinuationState {
    response_id: String,
    /// Signature of the non-input parts of the request (model, instructions, tools, etc.).
    /// If this changes we cannot safely continue.
    prompt_signature: String,
    /// The logical full `input` array (codex-shaped) that represents the conversation
    /// up to and including the assistant output of that turn.
    transcript: Vec<Value>,
    transcript_bytes: u64,
    updated_at: u64,
}

static STATES: Mutex<Option<HashMap<String, ContinuationState>>> = Mutex::new(None);
static TOTAL_TRANSCRIPT_BYTES: Mutex<u64> = Mutex::new(0);

#[derive(Clone, Default)]
pub struct ContinuationCandidate {
    pub previous_response_id: Option<String>,
    pub input_delta: Option<Vec<Value>>,
    pub input_delta_count: usize,
    pub disabled_reason: Option<String>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Compute a signature over everything except the "input" array.
/// This lets us detect when tools, model, instructions, reasoning, etc. changed.
/// Uses a stable JSON representation (sorted keys, normalized values) for robustness.
fn prompt_signature(body: &Value) -> String {
    let value = serde_json::to_value(body).unwrap_or_default();
    let obj = match value.as_object() {
        Some(o) => o,
        None => return String::new(),
    };
    let mut entries: Vec<(&String, &Value)> = obj.iter().filter(|(k, _)| *k != "input").collect();
    entries.sort_by_key(|(a, _)| *a);
    let mut sig = String::from("{");
    for (i, (key, val)) in entries.iter().enumerate() {
        if i > 0 {
            sig.push(',');
        }
        sig.push_str(&format!("\"{}\":{}", key, stable_json(val)));
    }
    sig.push('}');
    sig
}

fn stable_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => serde_json::to_string(s).unwrap_or_default(),
        Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(stable_json).collect();
            format!("[{}]", items.join(","))
        }
        Value::Object(obj) => {
            let mut entries: Vec<(&String, &Value)> = obj.iter().collect();
            entries.sort_by_key(|(a, _)| *a);
            let items: Vec<String> = entries
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(k).unwrap_or_default(),
                        stable_json(v)
                    )
                })
                .collect();
            format!("{{{}}}", items.join(","))
        }
    }
}

fn input_suffix_after_prefix(input: &[Value], prefix: &[Value]) -> Option<Vec<Value>> {
    if prefix.len() > input.len() {
        return None;
    }
    for (i, p) in prefix.iter().enumerate() {
        let a = &input[i];
        if a != p {
            return None;
        }
    }
    Some(input[prefix.len()..].to_vec())
}

pub fn continuation_candidate(
    session_id: Option<&str>,
    body: &Value,
    enabled: bool,
) -> ContinuationCandidate {
    let now = now_ms();

    if !enabled {
        return ContinuationCandidate {
            previous_response_id: None,
            input_delta: None,
            input_delta_count: body
                .get("input")
                .and_then(|i| i.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
            disabled_reason: Some("disabled".to_string()),
        };
    }

    let session_id = match session_id {
        Some(s) => s,
        None => {
            return ContinuationCandidate {
                previous_response_id: None,
                input_delta: None,
                input_delta_count: body
                    .get("input")
                    .and_then(|i| i.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0),
                disabled_reason: Some("missing_session".to_string()),
            };
        }
    };

    let state = {
        let guard = STATES.lock().unwrap();
        guard.as_ref().and_then(|m| m.get(session_id)).cloned()
    };

    let state = match state {
        Some(s) if now - s.updated_at <= TTL_MS => s,
        Some(_) => {
            clear_continuation(Some(session_id));
            return ContinuationCandidate {
                previous_response_id: None,
                input_delta: None,
                input_delta_count: body
                    .get("input")
                    .and_then(|i| i.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0),
                disabled_reason: Some("missing_state".to_string()),
            };
        }
        None => {
            return ContinuationCandidate {
                previous_response_id: None,
                input_delta: None,
                input_delta_count: body
                    .get("input")
                    .and_then(|i| i.as_array())
                    .map(|a| a.len())
                    .unwrap_or(0),
                disabled_reason: Some("missing_state".to_string()),
            };
        }
    };

    let sig = prompt_signature(body);
    if sig != state.prompt_signature {
        clear_continuation(Some(session_id));
        return ContinuationCandidate {
            previous_response_id: None,
            input_delta: None,
            input_delta_count: body
                .get("input")
                .and_then(|i| i.as_array())
                .map(|a| a.len())
                .unwrap_or(0),
            disabled_reason: Some("prompt_changed".to_string()),
        };
    }

    let current_input = body
        .get("input")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let suffix = match input_suffix_after_prefix(&current_input, &state.transcript) {
        Some(s) => s,
        None => {
            clear_continuation(Some(session_id));
            return ContinuationCandidate {
                previous_response_id: None,
                input_delta: None,
                input_delta_count: current_input.len(),
                disabled_reason: Some("not_append_only".to_string()),
            };
        }
    };

    if suffix.is_empty() {
        return ContinuationCandidate {
            previous_response_id: None,
            input_delta: None,
            input_delta_count: 0,
            disabled_reason: Some("empty_delta".to_string()),
        };
    }

    ContinuationCandidate {
        previous_response_id: Some(state.response_id.clone()),
        input_delta: Some(suffix.clone()),
        input_delta_count: suffix.len(),
        disabled_reason: None,
    }
}

/// Mutate the codex request body Value to use continuation (previous_response_id + delta input)
/// if the candidate provides them.
pub fn apply_codex_continuation(body: &mut Value, candidate: &ContinuationCandidate) {
    if let Some(obj) = body.as_object_mut() {
        if let Some(prev) = &candidate.previous_response_id {
            obj.insert("previous_response_id".to_string(), json!(prev));
        }
        if let Some(delta) = &candidate.input_delta {
            obj.insert("input".to_string(), json!(delta));
        }
    }
}

pub fn record_continuation(
    session_id: Option<&str>,
    logical_body: &Value,
    response_id: Option<&str>,
    output_items: &[Value],
) {
    let session_id = match session_id {
        Some(s) => s,
        None => return,
    };

    let response_id = match response_id {
        Some(id) => id.to_string(),
        None => {
            clear_continuation(Some(session_id));
            return;
        }
    };

    let mut transcript: Vec<Value> = logical_body
        .get("input")
        .and_then(|i| i.as_array())
        .cloned()
        .unwrap_or_default();
    transcript.extend_from_slice(output_items);

    let transcript_json = serde_json::to_string(&transcript).unwrap_or_default();
    let transcript_bytes = transcript_json.len() as u64;

    if transcript_bytes > MAX_SESSION_TRANSCRIPT_BYTES {
        clear_continuation(Some(session_id));
        return;
    }

    // Evict this session's previous bytes first
    clear_continuation(Some(session_id));

    let state = ContinuationState {
        response_id,
        prompt_signature: prompt_signature(logical_body),
        transcript,
        transcript_bytes,
        updated_at: now_ms(),
    };

    {
        let mut guard = TOTAL_TRANSCRIPT_BYTES.lock().unwrap();
        *guard = guard.saturating_add(state.transcript_bytes);
    }
    {
        let mut guard = STATES.lock().unwrap();
        let map = guard.get_or_insert_with(HashMap::new);
        map.insert(session_id.to_string(), state);
    }
    evict_oldest();
}

pub fn clear_continuation(session_id: Option<&str>) {
    let session_id = match session_id {
        Some(s) => s,
        None => return,
    };
    let mut guard = STATES.lock().unwrap();
    if let Some(map) = guard.as_mut()
        && let Some(existing) = map.remove(session_id)
    {
        let mut bytes_guard = TOTAL_TRANSCRIPT_BYTES.lock().unwrap();
        *bytes_guard = bytes_guard.saturating_sub(existing.transcript_bytes);
    }
}

fn evict_oldest() {
    // Evict oldest-first (by updated_at) when either the state count or total bytes
    // exceed the caps. Matches the spirit of the proxy (which also caps count) while
    // preferring LRU-style eviction over arbitrary HashMap order.
    let mut guard = STATES.lock().unwrap();
    let map = match guard.as_mut() {
        Some(m) if !m.is_empty() => m,
        _ => return,
    };

    let mut bytes_guard = TOTAL_TRANSCRIPT_BYTES.lock().unwrap();
    while (map.len() > MAX_STATES || *bytes_guard > MAX_TOTAL_TRANSCRIPT_BYTES) && !map.is_empty() {
        if let Some((&ref oldest_key, _)) = map.iter().min_by_key(|(_, s)| s.updated_at) {
            let key = oldest_key.clone();
            if let Some(old) = map.remove(&key) {
                *bytes_guard = bytes_guard.saturating_sub(old.transcript_bytes);
            }
        } else {
            break;
        }
    }
}

// Test helpers
#[cfg(test)]
pub fn has_continuation_for_tests(session_id: &str) -> bool {
    let guard = STATES.lock().unwrap();
    guard.as_ref().is_some_and(|m| m.contains_key(session_id))
}

#[cfg(test)]
pub fn clear_all_continuations_for_tests() {
    let mut guard = STATES.lock().unwrap();
    *guard = None;
    let mut b = TOTAL_TRANSCRIPT_BYTES.lock().unwrap();
    *b = 0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn first_turn_has_no_continuation() {
        clear_all_continuations_for_tests();
        let body = json!({
            "model": "gpt-5.5",
            "input": [ {"type":"message", "role":"user", "content":[{"type":"input_text","text":"hi"}]} ],
            "instructions": "be nice"
        });
        let c = continuation_candidate(Some("sess-1"), &body, true);
        assert!(c.previous_response_id.is_none());
        assert_eq!(c.disabled_reason, Some("missing_state".to_string()));
    }

    #[test]
    fn records_and_produces_delta_on_append() {
        clear_all_continuations_for_tests();
        let body1 = json!({
            "model": "gpt-5.5",
            "input": [ {"type":"message", "role":"user", "content":[{"type":"input_text","text":"first"}]} ],
        });
        // Simulate a completed turn
        record_continuation(Some("sess-1"), &body1, Some("resp_123"), &[]);
        assert!(has_continuation_for_tests("sess-1"));

        // Next logical input appends a new user message (CC will include prior assistant in real, but here we test basic suffix)
        let body2 = json!({
            "model": "gpt-5.5",
            "input": [
                {"type":"message", "role":"user", "content":[{"type":"input_text","text":"first"}]},
                {"type":"message", "role":"user", "content":[{"type":"input_text","text":"second"}]}
            ],
        });
        let c = continuation_candidate(Some("sess-1"), &body2, true);
        assert_eq!(c.previous_response_id, Some("resp_123".to_string()));
        assert!(c.input_delta.is_some());
        assert_eq!(c.input_delta.as_ref().unwrap().len(), 1);
    }
}
