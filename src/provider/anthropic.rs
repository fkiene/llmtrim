//! Anthropic Messages adapter.

use serde_json::{Value, json};

use super::{Provider, append_stop};
use crate::ir::{ProviderKind, Request};

/// Does the request already carry any `cache_control` block (client-managed caching)?
fn has_cache_control(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.contains_key("cache_control") || m.values().any(has_cache_control),
        Value::Array(a) => a.iter().any(has_cache_control),
        _ => false,
    }
}

pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    fn content_text_pointers(&self, req: &Request) -> Vec<String> {
        let mut out = Vec::new();
        let root = req.raw();

        // Top-level `system`: a string, or an array of (cacheable) text blocks.
        match root.get("system") {
            Some(Value::String(_)) => out.push("/system".to_string()),
            Some(Value::Array(blocks)) => {
                for (j, b) in blocks.iter().enumerate() {
                    if let Some(p) = super::text_block_ptr(b, &format!("/system/{j}")) {
                        out.push(p);
                    }
                }
            }
            _ => {}
        }

        if let Some(messages) = root.get("messages") {
            super::message_text_pointers(messages, &mut out);
        }
        out
    }

    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64) {
        if let Some(obj) = req.raw_mut().as_object_mut() {
            obj.insert("max_tokens".to_string(), json!(max_tokens));
        }
    }

    fn max_tokens(&self, req: &Request) -> Option<u64> {
        req.raw().get("max_tokens").and_then(Value::as_u64)
    }

    fn add_stop_sequence(&self, req: &mut Request, stop: &str) {
        append_stop(req.raw_mut(), "stop_sequences", stop);
    }

    fn add_system_instruction(&self, req: &mut Request, text: &str) {
        // Anthropic carries `system` as a top-level field (string or block array). APPEND our
        // instruction after the existing system content, never prepend: prepending changes the
        // bytes *before* a client `cache_control` breakpoint (Claude Code marks the last system
        // block), busting the prompt cache at 1.25× write cost every turn. Appending leaves the
        // cached prefix byte-identical — our (small, possibly per-request) instruction simply
        // sits after the breakpoint, uncached, which is correct since it can vary.
        let Some(obj) = req.raw_mut().as_object_mut() else {
            return;
        };
        match obj.get("system") {
            Some(Value::String(existing)) => {
                let combined = format!("{existing}\n\n{text}");
                obj.insert("system".to_string(), Value::String(combined));
            }
            Some(Value::Array(_)) => {
                if let Some(Value::Array(arr)) = obj.get_mut("system") {
                    arr.push(json!({"type": "text", "text": text}));
                }
            }
            _ => {
                obj.insert("system".to_string(), Value::String(text.to_string()));
            }
        }
    }

    fn bind_structured_output(&self, req: &mut Request, name: &str, schema: Value) {
        // Anthropic has no `response_format`; forced structured output is achieved
        // by exposing a single tool with the schema and forcing tool_choice to it.
        let Some(obj) = req.raw_mut().as_object_mut() else {
            return;
        };
        let tool = json!({
            "name": name,
            "description": "Return the result via this tool.",
            "input_schema": schema,
        });
        match obj.get_mut("tools") {
            Some(Value::Array(arr)) => arr.push(tool),
            _ => {
                obj.insert("tools".to_string(), json!([tool]));
            }
        }
        obj.insert(
            "tool_choice".to_string(),
            json!({"type": "tool", "name": name}),
        );
    }

    fn set_prompt_cache_key(&self, _req: &mut Request, _key: &str) {
        // Anthropic uses explicit `cache_control` breakpoints, not a prompt cache key.
    }

    fn set_cache_breakpoints(&self, req: &mut Request, max: usize) {
        if max == 0 {
            return;
        }
        // If the client already manages caching (e.g. Claude Code sets its own
        // `cache_control` blocks with specific ttls), leave it alone — adding our ephemeral
        // breakpoints corrupts the ttl ordering and the API rejects the request.
        if has_cache_control(req.raw()) {
            return;
        }
        let Some(obj) = req.raw_mut().as_object_mut() else {
            return;
        };
        let mut used = 0usize;

        // Cache the tool block (resent every call) by marking the last tool.
        if used < max
            && let Some(Value::Array(tools)) = obj.get_mut("tools")
            && let Some(last) = tools.last_mut()
            && let Some(t) = last.as_object_mut()
        {
            t.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            used += 1;
        }

        // Mark the end of the system prefix.
        if used < max {
            match obj.get("system") {
                Some(Value::Array(_)) => {
                    if let Some(Value::Array(blocks)) = obj.get_mut("system")
                        && let Some(last) = blocks.last_mut()
                        && let Some(b) = last.as_object_mut()
                    {
                        b.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
                    }
                }
                Some(Value::String(s)) => {
                    let text = s.clone();
                    obj.insert(
                        "system".to_string(),
                        json!([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]),
                    );
                }
                _ => {}
            }
        }

        // History breakpoint: on a multi-turn conversation (≥2 messages) with breakpoints to
        // spare, cache the conversation prefix by marking the last block of the last message.
        // The growing history is the bulk of a raw-SDK agent's input and is otherwise re-billed
        // at full rate every turn; this turn writes the cache, the next reads it. Skipped on
        // single-shot requests (no next turn to amortize the +25% write).
        if used < max
            && let Some(Value::Array(messages)) = obj.get_mut("messages")
            && messages.len() >= 2
            && let Some(last) = messages.last_mut()
        {
            mark_last_block(last);
        }
    }

    fn tool_descriptors(&self, req: &Request) -> Vec<(String, String)> {
        let Some(tools) = req.raw().get("tools").and_then(Value::as_array) else {
            return Vec::new();
        };
        tools
            .iter()
            .map(|t| {
                let name = t.get("name").and_then(Value::as_str).unwrap_or_default();
                let desc = t
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                (name.to_string(), desc.to_string())
            })
            .collect()
    }

    fn retain_tools(&self, req: &mut Request, keep: &[bool]) {
        super::retain_tools_array(req, keep);
    }

    fn truncate_tool_descriptions(&self, req: &mut Request, max_chars: usize) {
        if let Some(Value::Array(tools)) = req.raw_mut().get_mut("tools") {
            for t in tools.iter_mut() {
                if let Some(Value::String(d)) = t.get_mut("description") {
                    super::truncate_chars(d, max_chars);
                }
            }
        }
    }

    fn answer_text(&self, response: &Value) -> Option<String> {
        let blocks = response.get("content")?.as_array()?;
        let text: String = blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect();
        (!text.is_empty()).then_some(text)
    }

    fn set_image_detail(&self, _req: &mut Request, _tier: &str) {
        // Anthropic has no per-image detail tier.
    }

    fn downscale_images(&self, req: &mut Request) {
        super::for_each_content_block(req, |b| {
            downscale_anthropic_block(b);
            // Computer-use screenshots are image blocks nested inside tool_result content.
            if b.get("type").and_then(Value::as_str) == Some("tool_result")
                && let Some(inner) = b.get_mut("content").and_then(Value::as_array_mut)
            {
                for ib in inner.iter_mut() {
                    downscale_anthropic_block(ib);
                }
            }
        });
    }
}

/// Mark a message's last content block with an ephemeral cache breakpoint, promoting a bare
/// string content to a single text block (a string can't carry `cache_control`). This is the
/// current turn (not a cached prefix), so reshaping it here is safe.
fn mark_last_block(msg: &mut Value) {
    let Some(obj) = msg.as_object_mut() else {
        return;
    };
    match obj.get_mut("content") {
        Some(Value::Array(blocks)) => {
            if let Some(b) = blocks.last_mut().and_then(Value::as_object_mut) {
                b.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
            }
        }
        Some(Value::String(s)) => {
            let text = s.clone();
            obj.insert(
                "content".to_string(),
                json!([{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]),
            );
        }
        _ => {}
    }
}

/// Downscale one Anthropic `image` block (base64 source) in place, to the resolution cap.
fn downscale_anthropic_block(b: &mut Value) {
    if b.get("type").and_then(Value::as_str) == Some("image")
        && b.pointer("/source/type").and_then(Value::as_str) == Some("base64")
        && let Some(Value::String(data)) = b.pointer_mut("/source/data")
        && let Some(new_data) = crate::media::fit_to_cap(data, crate::media::CAP_ANTHROPIC)
    {
        *data = new_data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: &str) -> Request {
        Request::parse(ProviderKind::Anthropic, body).unwrap()
    }

    #[test]
    fn text_pointers_cover_system_and_messages() {
        let r = req(r#"{
                "system":[{"type":"text","text":"sys"}],
                "messages":[{"role":"user","content":"hi"},
                            {"role":"assistant","content":[{"type":"text","text":"yo"}]}]
            }"#);
        let p = AnthropicProvider.content_text_pointers(&r);
        assert_eq!(
            p,
            vec![
                "/system/0/text",
                "/messages/0/content",
                "/messages/1/content/0/text"
            ]
        );
    }

    #[test]
    fn cache_breakpoints_skipped_when_client_already_caches() {
        // Mimics Claude Code: client sets its own cache_control (with ttls). We must not
        // add ours — it corrupts the ttl ordering and the API rejects it (real 400 bug).
        let mut r = req(r#"{
            "system":[{"type":"text","text":"sys","cache_control":{"type":"ephemeral","ttl":"1h"}}],
            "tools":[{"name":"t","description":"d"}],
            "messages":[{"role":"user","content":"hi"}]
        }"#);
        AnthropicProvider.set_cache_breakpoints(&mut r, 4);
        // Exactly the client's one cache_control remains — we added none.
        fn count(v: &Value) -> usize {
            match v {
                Value::Object(m) => {
                    m.values().map(count).sum::<usize>() + m.contains_key("cache_control") as usize
                }
                Value::Array(a) => a.iter().map(count).sum(),
                _ => 0,
            }
        }
        assert_eq!(count(r.raw()), 1);
    }

    #[test]
    fn cache_breakpoints_added_when_client_does_not_cache() {
        let mut r = req(r#"{"system":[{"type":"text","text":"sys"}],"messages":[]}"#);
        AnthropicProvider.set_cache_breakpoints(&mut r, 4);
        assert!(
            has_cache_control(r.raw()),
            "we add a breakpoint when none exists"
        );
    }

    #[test]
    fn tool_result_content_is_covered() {
        // Tool results (string and block-array forms) carry the bulk of agent context —
        // they must be in the text pointers so the content stages can compress them.
        let r = req(r#"{"messages":[
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"a","content":"FILE DATA"}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"b","content":[{"type":"text","text":"BLOCK DATA"}]}]}
        ]}"#);
        let p = AnthropicProvider.content_text_pointers(&r);
        assert!(
            p.contains(&"/messages/0/content/0/content".to_string()),
            "string tool_result content"
        );
        assert!(
            p.contains(&"/messages/1/content/0/content/0/text".to_string()),
            "array tool_result text block"
        );
    }

    #[test]
    fn tool_use_input_and_document_are_covered() {
        // Assistant tool_use echoes the call args (Write/Edit carries whole files in `input`);
        // text `document` blocks carry plain-text data. Both were uncovered before.
        let r = req(r#"{"messages":[
            {"role":"assistant","content":[
                {"type":"tool_use","id":"u1","name":"write","input":{"path":"a.rs","content":"FILE BODY"}}]},
            {"role":"user","content":[
                {"type":"document","source":{"type":"text","media_type":"text/plain","data":"DOC BODY"}}]}
        ]}"#);
        let p = AnthropicProvider.content_text_pointers(&r);
        assert!(
            p.contains(&"/messages/0/content/0/input/content".to_string()),
            "{p:?}"
        );
        assert!(
            p.contains(&"/messages/0/content/0/input/path".to_string()),
            "{p:?}"
        );
        assert!(
            p.contains(&"/messages/1/content/0/source/data".to_string()),
            "{p:?}"
        );
    }

    #[test]
    fn stop_uses_stop_sequences_key() {
        let mut r = req(r#"{"messages":[],"max_tokens":1}"#);
        AnthropicProvider.add_stop_sequence(&mut r, "STOP");
        assert_eq!(r.raw().get("stop_sequences").unwrap(), &json!(["STOP"]));
    }

    #[test]
    fn system_instruction_appends_to_string() {
        // Appended (not prepended) so the cached system prefix stays byte-identical.
        let mut r = req(r#"{"system":"old","messages":[],"max_tokens":1}"#);
        AnthropicProvider.add_system_instruction(&mut r, "new");
        assert_eq!(
            r.raw().get("system").and_then(Value::as_str),
            Some("old\n\nnew")
        );
    }

    #[test]
    fn system_instruction_appends_after_cache_breakpoint() {
        // Client (Claude Code) put cache_control on the last system block. Our instruction must
        // land *after* it, leaving the marked/cached block untouched (cache stays warm).
        let mut r = req(r#"{"system":[
            {"type":"text","text":"stable","cache_control":{"type":"ephemeral"}}
        ],"messages":[],"max_tokens":1}"#);
        AnthropicProvider.add_system_instruction(&mut r, "legend");
        let sys = r.raw().get("system").and_then(Value::as_array).unwrap();
        assert_eq!(sys.len(), 2);
        // The breakpoint block is unchanged; ours is the new trailing, uncached block.
        assert!(sys[0].get("cache_control").is_some());
        assert_eq!(sys[1].get("text").and_then(Value::as_str), Some("legend"));
        assert!(sys[1].get("cache_control").is_none());
    }

    #[test]
    fn structured_output_forces_tool() {
        let mut r = req(r#"{"messages":[],"max_tokens":1}"#);
        AnthropicProvider.bind_structured_output(&mut r, "Out", json!({"type":"object"}));
        assert_eq!(
            r.raw().pointer("/tool_choice/name").and_then(Value::as_str),
            Some("Out")
        );
        assert_eq!(
            r.raw().pointer("/tools/0/name").and_then(Value::as_str),
            Some("Out")
        );
    }
}
