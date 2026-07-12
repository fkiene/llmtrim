//! Anthropic extended-thinking invariants across compression (issue #157).
//!
//! Anthropic verifies the `signature` on every `thinking` block it is handed back, and rejects
//! an assistant turn that contains `tool_use` but does not *begin* with its thinking block. A
//! compression stage that rewrote thinking text, dropped a block, or reordered one would break
//! Claude Code's thinking outright — so pin the invariants here rather than rely on the stages
//! happening not to produce a text pointer for those block types.

use serde_json::{Value, json};
use std::collections::BTreeSet;

fn compress(body: &str) -> Value {
    let cfg = llmtrim_core::config::DenseConfig::default();
    let out = llmtrim_core::compress_with_config_model(
        body,
        Some(llmtrim_core::ir::ProviderKind::Anthropic),
        &cfg,
        None,
    )
    .expect("compress");
    serde_json::from_str(&out.request_json).expect("compressed body is JSON")
}

/// A Claude Code agent loop: interleaved thinking + tool_use turns, with tool_results big and
/// repetitive enough that the input-side stages actually fire.
fn cc_thinking_conversation(turns: usize) -> String {
    let mut messages = vec![json!({
        "role": "user",
        "content": [{"type": "text", "text": "Refactor the proxy and fix the dropped connections."}],
    })];
    for i in 0..turns {
        messages.push(json!({"role": "assistant", "content": [
            {"type": "thinking",
             "thinking": format!("Turn {i}: inspect the handler, then reason about the streaming path and the timeout budget. ").repeat(3),
             "signature": format!("SIG{i}/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa==")},
            {"type": "text", "text": format!("Reading module {i}.")},
            {"type": "tool_use", "id": format!("toolu_{i:03}"), "name": "Read",
             "input": {"file_path": format!("src/mod{i}.rs")}},
        ]}));
        messages.push(json!({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": format!("toolu_{i:03}"),
             "content": [{"type": "text", "text":
                format!("pub fn handler_{i}() {{\n    let x = compute();\n    log::info!(\"step\");\n    x\n}}\n").repeat(40)}]},
        ]}));
    }
    messages.push(json!({"role": "user",
        "content": [{"type": "text", "text": "Now summarize the root cause."}]}));

    json!({
        "model": "claude-sonnet-4-5",
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
        "system": [{"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."}],
        "tools": [{"name": "Read", "description": "Read a file.",
                   "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}}}],
        "messages": messages,
        "stream": true,
    })
    .to_string()
}

fn blocks_of<'a>(v: &'a Value, ty: &str) -> Vec<&'a Value> {
    v["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|b| b["type"].as_str() == Some(ty))
        .collect()
}

/// Anthropic verifies the signature over the thinking block it gets back: a single rewritten
/// byte invalidates the turn.
#[test]
fn thinking_blocks_survive_compression_byte_identical() {
    let body = cc_thinking_conversation(12);
    let before: Value = serde_json::from_str(&body).unwrap();
    let after = compress(&body);

    let (b, a) = (
        blocks_of(&before, "thinking"),
        blocks_of(&after, "thinking"),
    );
    assert_eq!(b.len(), 12, "fixture sanity");
    assert_eq!(a.len(), b.len(), "compression dropped a thinking block");
    assert_eq!(
        a, b,
        "compression rewrote a thinking block or its signature"
    );
}

/// With thinking enabled, an assistant turn carrying `tool_use` must *start* with its thinking
/// block — Anthropic rejects the request otherwise.
#[test]
fn tool_use_turns_still_begin_with_their_thinking_block() {
    let after = compress(&cc_thinking_conversation(12));
    for (i, m) in after["messages"].as_array().unwrap().iter().enumerate() {
        if m["role"].as_str() != Some("assistant") {
            continue;
        }
        let types: Vec<&str> = m["content"]
            .as_array()
            .map(|bs| {
                bs.iter()
                    .map(|b| b["type"].as_str().unwrap_or("?"))
                    .collect()
            })
            .unwrap_or_default();
        if types.contains(&"tool_use") {
            assert_eq!(
                types.first(),
                Some(&"thinking"),
                "messages[{i}] leads with {types:?}, not its thinking block"
            );
        }
    }
}

/// Dropping one side of a tool_use/tool_result pair is a hard 400 — and with a `thinking` turn
/// in between it is the easiest way for a retrieval stage to corrupt the transcript.
#[test]
fn tool_use_and_tool_result_ids_still_pair_up() {
    let after = compress(&cc_thinking_conversation(12));
    let ids = |ty: &str, key: &str| -> BTreeSet<String> {
        blocks_of(&after, ty)
            .iter()
            .filter_map(|b| b[key].as_str().map(str::to_string))
            .collect()
    };
    let uses = ids("tool_use", "id");
    let results = ids("tool_result", "tool_use_id");
    assert_eq!(
        uses,
        results,
        "unpaired tool blocks: use-only={:?} result-only={:?}",
        uses.difference(&results).collect::<Vec<_>>(),
        results.difference(&uses).collect::<Vec<_>>()
    );
}

/// `max_tokens` must stay above `thinking.budget_tokens`, and the thinking config itself must
/// reach the provider untouched.
#[test]
fn thinking_config_and_max_tokens_are_left_alone() {
    let after = compress(&cc_thinking_conversation(12));
    assert_eq!(
        after["thinking"],
        json!({"type": "enabled", "budget_tokens": 4096})
    );
    assert_eq!(after["max_tokens"], json!(8192));
    assert!(
        after["max_tokens"].as_i64().unwrap()
            > after["thinking"]["budget_tokens"].as_i64().unwrap(),
        "Anthropic rejects max_tokens <= thinking.budget_tokens"
    );
}
