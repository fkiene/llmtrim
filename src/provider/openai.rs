//! OpenAI adapter — Chat Completions *and* the Responses API (`/v1/responses`). Both share
//! one adapter: same tokenizer, pricing, and provider identity. Each method dispatches on the
//! body shape, since Responses replaces `messages` with `input`/`instructions`.

use serde_json::{Value, json};

use super::{Provider, append_stop};
use crate::ir::{ProviderKind, Request};

pub struct OpenAiProvider;

/// True for a Responses API body (`input`, no `messages`). Chat Completions carries `messages`;
/// the two never coexist, so this cleanly distinguishes the shapes.
fn is_responses(req: &Request) -> bool {
    req.raw()
        .as_object()
        .is_some_and(|o| !o.contains_key("messages") && o.contains_key("input"))
}

/// True for a Responses text content block (`input_text`/`output_text`/`text` with a string body).
fn is_responses_text_block(b: &Value) -> bool {
    matches!(
        b.get("type").and_then(Value::as_str),
        Some("input_text" | "output_text" | "text")
    ) && b.get("text").is_some_and(Value::is_string)
}

/// Text segments of a Responses body: the `instructions` system string, plus `input` — a bare
/// string, or an array of items whose `content` is a string or typed text blocks.
fn responses_text_pointers(root: &Value, out: &mut Vec<String>) {
    if root.get("instructions").is_some_and(Value::is_string) {
        out.push("/instructions".to_string());
    }
    match root.get("input") {
        Some(Value::String(_)) => out.push("/input".to_string()),
        Some(Value::Array(items)) => {
            for (i, item) in items.iter().enumerate() {
                match item.get("content") {
                    Some(Value::String(_)) => out.push(format!("/input/{i}/content")),
                    Some(Value::Array(blocks)) => {
                        for (j, b) in blocks.iter().enumerate() {
                            if is_responses_text_block(b) {
                                out.push(format!("/input/{i}/content/{j}/text"));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
}

impl Provider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    fn content_text_pointers(&self, req: &Request) -> Vec<String> {
        let mut out = Vec::new();
        if is_responses(req) {
            responses_text_pointers(req.raw(), &mut out);
        } else if let Some(messages) = req.raw().get("messages") {
            super::message_text_pointers(messages, &mut out);
        }
        out
    }

    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64) {
        let responses = is_responses(req);
        if let Some(obj) = req.raw_mut().as_object_mut() {
            // Responses caps with `max_output_tokens`; Chat Completions prefers whichever cap
            // field is already present, defaulting to the modern one.
            let key = if responses {
                "max_output_tokens"
            } else if obj.contains_key("max_tokens") {
                "max_tokens"
            } else {
                "max_completion_tokens"
            };
            obj.insert(key.to_string(), json!(max_tokens));
        }
    }

    fn max_tokens(&self, req: &Request) -> Option<u64> {
        let obj = req.raw().as_object()?;
        if is_responses(req) {
            return obj.get("max_output_tokens").and_then(Value::as_u64);
        }
        obj.get("max_tokens")
            .or_else(|| obj.get("max_completion_tokens"))
            .and_then(Value::as_u64)
    }

    fn add_stop_sequence(&self, req: &mut Request, stop: &str) {
        // The Responses API has no stop-sequence field — leave it untouched (lossless).
        if is_responses(req) {
            return;
        }
        append_stop(req.raw_mut(), "stop", stop);
    }

    fn add_system_instruction(&self, req: &mut Request, text: &str) {
        // Responses carries the system prompt in the top-level `instructions` string; prepend
        // to any existing one.
        if is_responses(req) {
            if let Some(obj) = req.raw_mut().as_object_mut() {
                let combined = match obj.get("instructions").and_then(Value::as_str) {
                    Some(existing) if !existing.is_empty() => format!("{text}\n{existing}"),
                    _ => text.to_string(),
                };
                obj.insert("instructions".to_string(), Value::String(combined));
            }
            return;
        }
        // Chat Completions carries it as a `role: system` message. Insert at the front so it
        // joins the stable prefix (Stage A ordering, later phase).
        if let Some(obj) = req.raw_mut().as_object_mut()
            && let Some(Value::Array(messages)) = obj.get_mut("messages")
        {
            messages.insert(0, json!({"role": "system", "content": text}));
        }
    }

    fn bind_structured_output(&self, req: &mut Request, name: &str, schema: Value) {
        let responses = is_responses(req);
        if let Some(obj) = req.raw_mut().as_object_mut() {
            if responses {
                // Responses binds the schema under `text.format`.
                obj.insert(
                    "text".to_string(),
                    json!({
                        "format": {
                            "type": "json_schema",
                            "name": name,
                            "schema": schema,
                            "strict": true,
                        }
                    }),
                );
            } else {
                obj.insert(
                    "response_format".to_string(),
                    json!({
                        "type": "json_schema",
                        "json_schema": {"name": name, "schema": schema, "strict": true},
                    }),
                );
            }
        }
    }

    fn set_cache_breakpoints(&self, _req: &mut Request, _max: usize) {
        // OpenAI caches the longest matching prefix automatically; no breakpoint API.
    }

    fn tool_descriptors(&self, req: &Request) -> Vec<(String, String)> {
        let Some(tools) = req.raw().get("tools").and_then(Value::as_array) else {
            return Vec::new();
        };
        tools
            .iter()
            .map(|t| {
                // Responses tools are flat (`{name, description}`); Chat Completions nests them
                // under `function`.
                let scope = t.get("function").unwrap_or(t);
                let name = scope
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let desc = scope
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
                // Description lives under `function` (Chat) or at the top level (Responses).
                let desc = match t.get_mut("function").and_then(Value::as_object_mut) {
                    Some(f) => f.get_mut("description"),
                    None => t.get_mut("description"),
                };
                if let Some(Value::String(d)) = desc {
                    super::truncate_chars(d, max_chars);
                }
            }
        }
    }

    fn answer_text(&self, response: &Value) -> Option<String> {
        // Chat Completions.
        if let Some(content) = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        {
            return Some(content.to_string());
        }
        if let Some(f) = response.pointer("/choices/0/message/tool_calls/0/function") {
            // A tool-call response has null content; the call itself is the answer.
            // Serialize the first call's function ({name, arguments}) for the tool-match scorer.
            return Some(f.to_string());
        }
        // Responses API: concatenate the `output_text` of every message item in `output[]`.
        if let Some(output) = response.get("output").and_then(Value::as_array) {
            let mut text = String::new();
            for item in output {
                if item.get("type").and_then(Value::as_str) == Some("message")
                    && let Some(blocks) = item.get("content").and_then(Value::as_array)
                {
                    for b in blocks {
                        if b.get("type").and_then(Value::as_str) == Some("output_text")
                            && let Some(t) = b.get("text").and_then(Value::as_str)
                        {
                            text.push_str(t);
                        }
                    }
                }
            }
            if !text.is_empty() {
                return Some(text);
            }
            // No text — a function_call item is the answer.
            if let Some(fc) = output
                .iter()
                .find(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
            {
                return Some(fc.to_string());
            }
        }
        None
    }

    fn set_image_detail(&self, req: &mut Request, tier: &str) {
        super::for_each_content_block(req, |b| {
            if b.get("type").and_then(Value::as_str) == Some("image_url")
                && let Some(iu) = b.get_mut("image_url").and_then(Value::as_object_mut)
            {
                iu.insert("detail".to_string(), Value::String(tier.to_string()));
            }
        });
    }

    fn downscale_images(&self, req: &mut Request) {
        super::for_each_content_block(req, |b| {
            if b.get("type").and_then(Value::as_str) == Some("image_url")
                && let Some(Value::String(url)) = b.pointer_mut("/image_url/url")
                && url.starts_with("data:")
                && let Some(new_url) = crate::media::fit_data_uri(url, crate::media::CAP_OPENAI)
            {
                *url = new_url;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(body: &str) -> Request {
        Request::parse(ProviderKind::OpenAi, body).unwrap()
    }

    #[test]
    fn text_pointers_string_and_block_content() {
        let r = req(r#"{"messages":[
                {"role":"system","content":"sys"},
                {"role":"user","content":[{"type":"text","text":"hello"},{"type":"image_url","image_url":{"url":"x"}}]}
            ]}"#);
        let p = OpenAiProvider.content_text_pointers(&r);
        assert_eq!(p, vec!["/messages/0/content", "/messages/1/content/0/text"]);
    }

    #[test]
    fn max_tokens_prefers_existing_field() {
        let mut r = req(r#"{"max_tokens":50,"messages":[]}"#);
        OpenAiProvider.set_max_tokens(&mut r, 10);
        assert_eq!(OpenAiProvider.max_tokens(&r), Some(10));
        assert!(r.raw().get("max_completion_tokens").is_none());

        let mut r2 = req(r#"{"messages":[]}"#);
        OpenAiProvider.set_max_tokens(&mut r2, 20);
        assert_eq!(
            r2.raw()
                .get("max_completion_tokens")
                .and_then(Value::as_u64),
            Some(20)
        );
    }

    #[test]
    fn stop_promotes_string_to_array() {
        let mut r = req(r#"{"stop":"END","messages":[]}"#);
        OpenAiProvider.add_stop_sequence(&mut r, "STOP");
        assert_eq!(r.raw().get("stop").unwrap(), &json!(["END", "STOP"]));
    }

    #[test]
    fn system_instruction_inserts_front_message() {
        let mut r = req(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        OpenAiProvider.add_system_instruction(&mut r, "be terse");
        let first = &r.raw().get("messages").unwrap()[0];
        assert_eq!(first, &json!({"role":"system","content":"be terse"}));
    }

    #[test]
    fn structured_output_sets_response_format() {
        let mut r = req(r#"{"messages":[]}"#);
        OpenAiProvider.bind_structured_output(&mut r, "Out", json!({"type":"object"}));
        assert_eq!(
            r.raw()
                .pointer("/response_format/type")
                .and_then(Value::as_str),
            Some("json_schema"),
        );
    }

    // ── Responses API (/v1/responses) ──────────────────────────────────────────────────

    #[test]
    fn responses_text_pointers_cover_instructions_and_input() {
        // String `input`.
        let r = req(r#"{"instructions":"be terse","input":"hello"}"#);
        assert_eq!(
            OpenAiProvider.content_text_pointers(&r),
            vec!["/instructions", "/input"]
        );
        // Array `input` with a string content and a typed `input_text` block (the latter sits
        // beside a non-text block that must be skipped).
        let r2 = req(r#"{"input":[
                {"role":"user","content":"plain"},
                {"role":"user","content":[{"type":"input_text","text":"typed"},{"type":"input_image","image_url":"x"}]}
            ]}"#);
        assert_eq!(
            OpenAiProvider.content_text_pointers(&r2),
            vec!["/input/0/content", "/input/1/content/0/text"]
        );
    }

    #[test]
    fn responses_uses_max_output_tokens() {
        let mut r = req(r#"{"input":"hi"}"#);
        OpenAiProvider.set_max_tokens(&mut r, 32);
        assert_eq!(OpenAiProvider.max_tokens(&r), Some(32));
        assert_eq!(
            r.raw().get("max_output_tokens").and_then(Value::as_u64),
            Some(32)
        );
        assert!(r.raw().get("max_tokens").is_none());
    }

    #[test]
    fn responses_system_prepends_instructions() {
        let mut r = req(r#"{"instructions":"keep","input":"hi"}"#);
        OpenAiProvider.add_system_instruction(&mut r, "be terse");
        assert_eq!(
            r.raw().get("instructions").and_then(Value::as_str),
            Some("be terse\nkeep")
        );
    }

    #[test]
    fn responses_tools_are_flat() {
        let r = req(
            r#"{"input":"hi","tools":[{"type":"function","name":"grep","description":"search"}]}"#,
        );
        assert_eq!(
            OpenAiProvider.tool_descriptors(&r),
            vec![("grep".to_string(), "search".to_string())]
        );
    }

    #[test]
    fn responses_answer_text_walks_output() {
        let body = json!({
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"}
                ]}
            ]
        });
        assert_eq!(
            OpenAiProvider.answer_text(&body).as_deref(),
            Some("hello world")
        );
    }
}
