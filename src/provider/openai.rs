//! OpenAI Chat Completions adapter.

use serde_json::{Value, json};

use super::{Provider, append_stop};
use crate::ir::{ProviderKind, Request};

pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    fn content_text_pointers(&self, req: &Request) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(messages) = req.raw().get("messages") {
            super::message_text_pointers(messages, &mut out);
        }
        out
    }

    fn set_max_tokens(&self, req: &mut Request, max_tokens: u64) {
        if let Some(obj) = req.raw_mut().as_object_mut() {
            // Prefer whichever cap field is already present; default to the modern one.
            let key = if obj.contains_key("max_tokens") {
                "max_tokens"
            } else {
                "max_completion_tokens"
            };
            obj.insert(key.to_string(), json!(max_tokens));
        }
    }

    fn max_tokens(&self, req: &Request) -> Option<u64> {
        let obj = req.raw().as_object()?;
        obj.get("max_tokens")
            .or_else(|| obj.get("max_completion_tokens"))
            .and_then(Value::as_u64)
    }

    fn add_stop_sequence(&self, req: &mut Request, stop: &str) {
        append_stop(req.raw_mut(), "stop", stop);
    }

    fn add_system_instruction(&self, req: &mut Request, text: &str) {
        // OpenAI carries the system prompt as a `role: system` message. Insert at
        // the front so it joins the stable prefix (Stage A ordering, later phase).
        if let Some(obj) = req.raw_mut().as_object_mut()
            && let Some(Value::Array(messages)) = obj.get_mut("messages")
        {
            messages.insert(0, json!({"role": "system", "content": text}));
        }
    }

    fn bind_structured_output(&self, req: &mut Request, name: &str, schema: Value) {
        if let Some(obj) = req.raw_mut().as_object_mut() {
            obj.insert(
                "response_format".to_string(),
                json!({
                    "type": "json_schema",
                    "json_schema": {"name": name, "schema": schema, "strict": true},
                }),
            );
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
                let f = t.get("function");
                let name = f
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let desc = f
                    .and_then(|f| f.get("description"))
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
                if let Some(f) = t.get_mut("function").and_then(Value::as_object_mut)
                    && let Some(Value::String(d)) = f.get_mut("description")
                {
                    super::truncate_chars(d, max_chars);
                }
            }
        }
    }

    fn answer_text(&self, response: &Value) -> Option<String> {
        if let Some(content) = response
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
        {
            return Some(content.to_string());
        }
        // A tool-call response has null content; the call itself is the answer.
        // Serialize the first call's function ({name, arguments}) so callers and the
        // tool-match scorer can read it.
        response
            .pointer("/choices/0/message/tool_calls/0/function")
            .map(|f| f.to_string())
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
}
