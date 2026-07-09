//! Sanitize the `Read` tool calls a Codex model emits so Claude Code's Read tool doesn't choke on
//! them. Codex occasionally hallucinates an absurd `offset` (millions of lines) or sends an empty
//! `pages` string; both make the client-side Read fail. We strip those args before the tool call
//! reaches the client, and record the stripped offset keyed by the tool-call id so the next turn's
//! `tool_result` can carry a short note back to the model ("the offset you asked for was ignored").
//!
//! Ported from the reference proxy's read-arg rewrite; the note store is bounded (FIFO-evicted by
//! insertion order) so a long session can't grow it without limit. Keyed by the provider's
//! per-call tool-call id, which is globally unique, so the process-wide store is safe to share
//! across concurrent sessions.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use once_cell::sync::Lazy;
use serde_json::Value;

/// Cap on remembered rewrites (FIFO-evicted by insertion order past this).
const MAX_REWRITE_NOTES: usize = 4_096;
/// An `offset` at or above this is treated as a hallucination and dropped.
const READ_OFFSET_REWRITE_THRESHOLD: i64 = 1_000_000;

/// A dropped `offset` remembered by tool-call id, so the matching `tool_result` can note it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadOffsetRewrite {
    pub offset: i64,
    pub file_path: Option<String>,
}

#[derive(Debug, Default)]
struct RewriteStore {
    order: VecDeque<String>,
    entries: HashMap<String, ReadOffsetRewrite>,
}

static READ_OFFSET_REWRITES: Lazy<Mutex<RewriteStore>> =
    Lazy::new(|| Mutex::new(RewriteStore::default()));

/// Clean a `Read` tool call's argument JSON: drop an empty `pages`, and drop an absurd `offset`
/// (recording it under `call_id` for a later note). Any other tool, empty args, or non-JSON args
/// pass through unchanged.
pub fn sanitize_read_args(name: &str, args: &str, call_id: Option<&str>) -> String {
    if name != "Read" || args.is_empty() {
        return args.to_string();
    }
    let Ok(parsed) = serde_json::from_str::<Value>(args) else {
        return args.to_string();
    };
    let Some(obj) = parsed.as_object() else {
        return args.to_string();
    };

    let mut sanitized = obj.clone();
    let mut changed = false;

    let has_empty_pages = obj
        .get("pages")
        .and_then(Value::as_str)
        .is_some_and(str::is_empty);
    if has_empty_pages {
        sanitized.remove("pages");
        changed = true;
    }

    if let Some(offset) = obj.get("offset").and_then(Value::as_i64)
        && offset >= READ_OFFSET_REWRITE_THRESHOLD
    {
        sanitized.remove("offset");
        changed = true;
        if let Some(call_id) = call_id.filter(|id| !id.is_empty()) {
            record_read_offset_rewrite(
                call_id,
                ReadOffsetRewrite {
                    offset,
                    file_path: obj
                        .get("file_path")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                },
            );
        }
    }

    if changed {
        serde_json::to_string(&sanitized).unwrap_or_else(|_| args.to_string())
    } else {
        args.to_string()
    }
}

/// The rewrite recorded for `call_id`, if the model's Read for that call had its offset stripped.
pub fn read_offset_rewrite(call_id: &str) -> Option<ReadOffsetRewrite> {
    READ_OFFSET_REWRITES
        .lock()
        .ok()
        .and_then(|store| store.entries.get(call_id).cloned())
}

/// A short note appended to a Read `tool_result` telling the model the offset it asked for was
/// dropped, so it doesn't keep re-issuing the same impossible read.
pub fn read_offset_rewrite_note(rewrite: &ReadOffsetRewrite) -> String {
    match &rewrite.file_path {
        Some(path) => format!(
            "[llmtrim: the requested offset {} for {path} was out of range and ignored; the file was read from the start]",
            rewrite.offset
        ),
        None => format!(
            "[llmtrim: the requested offset {} was out of range and ignored; the file was read from the start]",
            rewrite.offset
        ),
    }
}

fn record_read_offset_rewrite(call_id: &str, note: ReadOffsetRewrite) {
    let Ok(mut store) = READ_OFFSET_REWRITES.lock() else {
        return;
    };
    if !store.entries.contains_key(call_id) {
        store.order.push_back(call_id.to_string());
    }
    store.entries.insert(call_id.to_string(), note);
    while store.entries.len() > MAX_REWRITE_NOTES {
        let Some(oldest) = store.order.pop_front() else {
            break;
        };
        store.entries.remove(&oldest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_empty_pages() {
        let out = sanitize_read_args("Read", r#"{"file_path":"/tmp/a","pages":""}"#, None);
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("pages").is_none());
        assert_eq!(v.get("file_path").and_then(Value::as_str), Some("/tmp/a"));
    }

    #[test]
    fn drops_and_records_absurd_offset() {
        let out = sanitize_read_args(
            "Read",
            r#"{"file_path":"/tmp/a","offset":1300000,"limit":20}"#,
            Some("call_rw_test"),
        );
        let v: Value = serde_json::from_str(&out).unwrap();
        assert!(v.get("offset").is_none());
        assert_eq!(v.get("limit").and_then(Value::as_i64), Some(20));
        let note = read_offset_rewrite("call_rw_test").unwrap();
        assert_eq!(note.offset, 1_300_000);
        assert_eq!(note.file_path.as_deref(), Some("/tmp/a"));
        assert!(read_offset_rewrite_note(&note).contains("/tmp/a"));
    }

    #[test]
    fn keeps_normal_offset_and_non_read() {
        let kept = sanitize_read_args(
            "Read",
            r#"{"file_path":"/tmp/a","offset":1300}"#,
            Some("c1"),
        );
        assert_eq!(
            serde_json::from_str::<Value>(&kept).unwrap()["offset"].as_i64(),
            Some(1300)
        );
        assert!(read_offset_rewrite("c1").is_none());
        // A non-Read tool is untouched even with an absurd offset.
        let other = sanitize_read_args("Bash", r#"{"offset":9999999}"#, Some("c2"));
        assert_eq!(other, r#"{"offset":9999999}"#);
        assert!(read_offset_rewrite("c2").is_none());
    }
}
