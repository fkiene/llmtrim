//! Per-result exemptions from tool-output windowing (issue #281).
//!
//! A courier that must deliver stdout verbatim — including a terminal trailer a
//! downstream attester binds to — cannot recover from clipping via `llmtrim recall`
//! (it is sandboxed to the dispatch command). Three signals skip lossy shaping and
//! the ANSI/CR pre-pass for that result only:
//!
//! 1. Config/env command globs ([`crate::config::RuntimeConfig::toolout_passthrough`]).
//!    `*` matches every command. `LLMTRIM_TOOL_OUTPUT=passthrough` is this glob.
//! 2. The producing bash command assigns `LLMTRIM_TOOL_OUTPUT=passthrough`.
//! 3. A line of the result is exactly that assignment (in-band, for wrappers that
//!    cannot change the recorded command).
//!
//! Independently, lines starting with [`KEEP_PREFIX`] are force-kept when windowing
//! still runs.

use std::collections::HashMap;

use serde_json::Value;

/// Documented force-keep prefix. Windowing always retains a line that starts with
/// this (optional leading whitespace allowed).
pub(crate) const KEEP_PREFIX: &str = "LLMTRIM_KEEP:";

/// In-band / command-line passthrough assignment the interceptor honours.
pub(crate) const PASSTHROUGH_SENTINEL: &str = "LLMTRIM_TOOL_OUTPUT=passthrough";

/// True when this tool result must ship byte-identical: no normalize, no window,
/// no recall trailer.
pub(crate) fn should_passthrough(text: &str, command: Option<&str>, patterns: &[String]) -> bool {
    if output_requests_passthrough(text) {
        return true;
    }
    if patterns.iter().any(|p| p == "*") {
        return true;
    }
    let Some(cmd) = command else {
        return false;
    };
    command_requests_passthrough(cmd) || patterns.iter().any(|pat| matches_command(pat, cmd))
}

pub(crate) fn is_keep_line(line: &str) -> bool {
    line.trim_start().starts_with(KEEP_PREFIX)
}

fn output_requests_passthrough(text: &str) -> bool {
    text.lines()
        .any(|line| line.trim().eq_ignore_ascii_case(PASSTHROUGH_SENTINEL))
}

fn command_requests_passthrough(command: &str) -> bool {
    contains_ignore_ascii_case(command, PASSTHROUGH_SENTINEL)
}

fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle.as_bytes()))
}

/// Glob-match `command` against `pat`. `~/` in the pattern expands to `$HOME/` so a
/// config written with a tilde still hits an expanded path on the wire.
fn matches_command(pat: &str, command: &str) -> bool {
    if glob_match(pat, command) {
        return true;
    }
    let expanded_pat = expand_home(pat);
    if expanded_pat != pat && glob_match(&expanded_pat, command) {
        return true;
    }
    let expanded_cmd = expand_home(command);
    if expanded_cmd != command && glob_match(pat, &expanded_cmd) {
        return true;
    }
    if expanded_pat != pat && expanded_cmd != command && glob_match(&expanded_pat, &expanded_cmd) {
        return true;
    }
    false
}

fn expand_home(s: &str) -> String {
    let Ok(home) = std::env::var("HOME") else {
        return s.to_string();
    };
    if home.is_empty() {
        return s.to_string();
    }
    s.replace("~/", &format!("{home}/"))
}

/// `*` = any sequence (including `/` and spaces), `?` = one byte. Consecutive stars
/// collapse. Empty pattern matches only empty text.
pub(crate) fn glob_match(pat: &str, text: &str) -> bool {
    glob_rec(pat.as_bytes(), text.as_bytes())
}

fn glob_rec(pat: &[u8], text: &[u8]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < pat.len() {
        match pat[i] {
            b'*' => {
                while i < pat.len() && pat[i] == b'*' {
                    i += 1;
                }
                if i == pat.len() {
                    return true;
                }
                while j <= text.len() {
                    if glob_rec(&pat[i..], &text[j..]) {
                        return true;
                    }
                    if j == text.len() {
                        break;
                    }
                    j += 1;
                }
                return false;
            }
            b'?' => {
                if j >= text.len() {
                    return false;
                }
                i += 1;
                j += 1;
            }
            c => {
                if j >= text.len() || text[j] != c {
                    return false;
                }
                i += 1;
                j += 1;
            }
        }
    }
    j == text.len()
}

/// tool_use / function_call / tool_calls id → `command` argument.
pub(crate) fn commands_by_id(raw: &Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    walk_commands(raw, &mut out);
    out
}

fn walk_commands(v: &Value, out: &mut HashMap<String, String>) {
    match v {
        Value::Array(items) => {
            for item in items {
                walk_commands(item, out);
            }
        }
        Value::Object(map) => {
            ingest_invocation(map, out);
            for child in map.values() {
                walk_commands(child, out);
            }
        }
        _ => {}
    }
}

fn ingest_invocation(map: &serde_json::Map<String, Value>, out: &mut HashMap<String, String>) {
    let ty = map.get("type").and_then(Value::as_str).unwrap_or("");
    if ty == "tool_use"
        && let Some(id) = map.get("id").and_then(Value::as_str)
        && let Some(cmd) = map
            .get("input")
            .and_then(|i| i.get("command"))
            .and_then(Value::as_str)
    {
        out.insert(id.to_string(), cmd.to_string());
        return;
    }
    if ty == "function_call"
        && let Some(id) = map
            .get("call_id")
            .or_else(|| map.get("id"))
            .and_then(Value::as_str)
        && let Some(cmd) = command_from_arguments(map.get("arguments"))
    {
        out.insert(id.to_string(), cmd);
        return;
    }
    if let Some(id) = map.get("id").and_then(Value::as_str)
        && let Some(cmd) =
            command_from_arguments(map.get("function").and_then(|f| f.get("arguments")))
    {
        out.insert(id.to_string(), cmd);
    }
}

fn command_from_arguments(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|j| j.get("command").and_then(Value::as_str).map(str::to_string)),
        Value::Object(o) => o.get("command").and_then(Value::as_str).map(str::to_string),
        _ => None,
    }
}

/// Id on the tool-result block that `pointer` addresses (`tool_use_id` / `tool_call_id`
/// / `call_id`), walking toward the document root.
pub(crate) fn result_call_id<'a>(raw: &'a Value, pointer: &str) -> Option<&'a str> {
    let mut p = pointer;
    loop {
        if let Some(node) = raw.pointer(p) {
            for key in ["tool_use_id", "tool_call_id", "call_id"] {
                if let Some(id) = node.get(key).and_then(Value::as_str) {
                    return Some(id);
                }
            }
        }
        match p.rsplit_once('/') {
            None | Some(("", _)) => return None,
            Some((parent, _)) => p = parent,
        }
    }
}

pub(crate) fn command_for<'a>(
    commands: &'a HashMap<String, String>,
    raw: &Value,
    pointer: &str,
) -> Option<&'a str> {
    result_call_id(raw, pointer).and_then(|id| commands.get(id).map(String::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn glob_star_and_literal() {
        assert!(glob_match("*", "anything at all"));
        assert!(glob_match(
            "bash ~/.claude/bin/gpt.sh *",
            "bash ~/.claude/bin/gpt.sh --job abc"
        ));
        assert!(glob_match(
            "*gpt.sh*",
            "bash /Users/x/.claude/bin/gpt.sh foo"
        ));
        assert!(!glob_match(
            "bash ~/.claude/bin/gpt.sh *",
            "bash cargo test"
        ));
        assert!(glob_match("a?c", "abc"));
        assert!(!glob_match("a?c", "ac"));
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }

    #[test]
    fn keep_line_allows_indent() {
        assert!(is_keep_line("LLMTRIM_KEEP: LANE_DELIVERY job=1 sha256=ab"));
        assert!(is_keep_line("  LLMTRIM_KEEP: trailer"));
        assert!(!is_keep_line("LANE_DELIVERY job=1 sha256=ab"));
        assert!(!is_keep_line("note LLMTRIM_KEEP: not a prefix"));
    }

    #[test]
    fn sentinel_in_output_or_command_passthroughs() {
        let dump = "INFO a\nLLMTRIM_TOOL_OUTPUT=passthrough\nLANE_DELIVERY x";
        assert!(should_passthrough(dump, None, &[]));
        assert!(should_passthrough(
            "INFO only",
            Some("LLMTRIM_TOOL_OUTPUT=passthrough bash ~/.claude/bin/gpt.sh"),
            &[]
        ));
        assert!(!should_passthrough("INFO only", Some("echo hi"), &[]));
        assert!(should_passthrough(
            "INFO only",
            Some("echo hi"),
            &["*".into()]
        ));
    }

    #[test]
    fn star_pattern_passthroughs_without_a_command() {
        assert!(should_passthrough("INFO only", None, &["*".into()]));
        assert!(!should_passthrough("INFO only", None, &["gpt.sh *".into()]));
    }

    #[test]
    fn command_glob_passthroughs_matching_bash() {
        assert!(should_passthrough(
            "INFO only",
            Some("bash ~/.claude/bin/gpt.sh --job 1"),
            &["bash ~/.claude/bin/gpt.sh *".into()]
        ));
        assert!(!should_passthrough(
            "INFO only",
            Some("cargo test"),
            &["bash ~/.claude/bin/gpt.sh *".into()]
        ));
    }

    #[test]
    fn anthropic_tool_use_command_maps_to_result() {
        let raw = json!({
            "messages": [
                {"role":"assistant","content":[
                    {"type":"tool_use","id":"call-1","name":"Bash",
                     "input":{"command":"bash ~/.claude/bin/gpt.sh --job 1"}}
                ]},
                {"role":"user","content":[
                    {"type":"tool_result","tool_use_id":"call-1","content":"stdout"}
                ]}
            ]
        });
        let cmds = commands_by_id(&raw);
        assert_eq!(
            command_for(&cmds, &raw, "/messages/1/content/0/content"),
            Some("bash ~/.claude/bin/gpt.sh --job 1")
        );
    }

    #[test]
    fn openai_chat_and_responses_command_maps() {
        let chat = json!({
            "messages": [
                {"role":"assistant","tool_calls":[
                    {"id":"c1","type":"function",
                     "function":{"name":"Bash","arguments":"{\"command\":\"ls -l\"}"}}
                ]},
                {"role":"tool","tool_call_id":"c1","content":"total 0"}
            ]
        });
        let cmds = commands_by_id(&chat);
        assert_eq!(
            command_for(&cmds, &chat, "/messages/1/content"),
            Some("ls -l")
        );

        let responses = json!({
            "input": [
                {"type":"function_call","call_id":"r1","name":"bash",
                 "arguments":{"command":"pwd"}},
                {"type":"function_call_output","call_id":"r1","output":"/tmp"}
            ]
        });
        let cmds = commands_by_id(&responses);
        assert_eq!(
            command_for(&cmds, &responses, "/input/1/output"),
            Some("pwd")
        );
    }
}
