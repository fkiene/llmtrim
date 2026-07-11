//! Kimi (Moonshot) subscription reroute provider.
//!
//! Translates an intercepted Anthropic `/v1/messages` request into the Kimi coding chat-completions
//! wire shape (POST `https://api.kimi.com/coding/v1/chat/completions`) and reduces Kimi's streamed
//! `chat.completion.chunk` SSE back into the shared [`ReduceEvent`] stream the
//! [`crate::reroute::sse::AnthropicSseEncoder`] turns into Anthropic SSE.
//!
//! Kimi exposes a single wire model (`kimi-for-coding`); the resolved `model` argument is ignored
//! for the request body. The endpoint is OpenAI-compatible chat completions with a few Moonshot
//! extensions (`reasoning_content`, `thinking`, `reasoning_effort`).

use anyhow::Result;
use serde_json::{Value, json};

use crate::reroute::sse::{ReduceEvent, SseLineParser, StopReason, Usage};

pub const HOST: &str = "api.kimi.com";
pub const PATH: &str = "/coding/v1/chat/completions";

/// The only model id the Kimi coding endpoint accepts. Every incoming tier collapses to it.
const WIRE_MODEL: &str = "kimi-for-coding";
/// Kimi caps `max_tokens` for the coding endpoint at 32k.
const MAX_TOKENS_CAP: i64 = 32_000;
/// System text blocks that begin with this marker are llmtrim/Claude-Code billing metadata and must
/// not be forwarded upstream.
const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

/// Device id advertised in the `X-Msh-Device-Id` header.
///
/// `LLMTRIM_KIMI_DEVICE_ID` overrides for tests/debugging; otherwise the persistent per-install id
/// generated and stored by [`crate::reroute::auth::kimi_device_id`] (bound into the Kimi JWT at
/// login, so it must stay stable for the life of the install).
pub fn device_id() -> String {
    std::env::var("LLMTRIM_KIMI_DEVICE_ID")
        .unwrap_or_else(|_| crate::reroute::auth::kimi_device_id())
}

fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

/// Headers to SET on the rewritten upstream request. `account_id` is unused by Kimi.
pub fn request_headers(
    access_token: &str,
    _account_id: Option<&str>,
    _session_id: Option<&str>,
) -> Vec<(String, String)> {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "application/json".to_string()),
        (
            "authorization".to_string(),
            format!("Bearer {access_token}"),
        ),
        ("X-Msh-Platform".to_string(), "kimi_cli".to_string()),
        ("X-Msh-Version".to_string(), "1.37.0".to_string()),
        ("X-Msh-Device-Name".to_string(), hostname()),
        ("X-Msh-Device-Model".to_string(), format!("{os} {arch}")),
        ("X-Msh-Os-Version".to_string(), arch.to_string()),
        ("X-Msh-Device-Id".to_string(), device_id()),
        ("user-agent".to_string(), "KimiCLI/1.37.0".to_string()),
    ]
}

/// Anthropic `/v1/messages` body -> Kimi chat-completions request body.
pub fn build_request_body(
    anthropic: &Value,
    _model: &str,
    session_id: Option<&str>,
) -> Result<Value> {
    let thinking_disabled = anthropic
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        == Some("disabled");

    let max_tokens = anthropic
        .get("max_tokens")
        .and_then(Value::as_i64)
        .map(|m| m.min(MAX_TOKENS_CAP))
        .unwrap_or(MAX_TOKENS_CAP);

    let mut messages: Vec<Value> = Vec::new();

    // System -> a leading {role:"system"} message (billing-header blocks dropped, joined "\n\n").
    if let Some(system) = build_system(anthropic.get("system")) {
        messages.push(json!({"role": "system", "content": system}));
    }

    if let Some(arr) = anthropic.get("messages").and_then(Value::as_array) {
        for msg in arr {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
            match role {
                "assistant" => messages.push(build_assistant(msg)),
                _ => build_user(msg, &mut messages),
            }
        }
    }

    let mut body = json!({
        "model": WIRE_MODEL,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": max_tokens,
    });
    let obj = body
        .as_object_mut()
        .expect("json! literal is always an object");

    if let Some(tools) = build_tools(anthropic.get("tools")) {
        obj.insert("tools".to_string(), tools);
    }
    if let Some(choice) = build_tool_choice(anthropic.get("tool_choice")) {
        obj.insert("tool_choice".to_string(), choice);
    }

    if !thinking_disabled {
        obj.insert(
            "reasoning_effort".to_string(),
            Value::String(reasoning_effort(anthropic)),
        );
        obj.insert("thinking".to_string(), json!({"type": "enabled"}));
    }

    if let Some(sid) = session_id {
        obj.insert(
            "prompt_cache_key".to_string(),
            Value::String(sid.to_string()),
        );
    }

    Ok(body)
}

/// `output_config.effort` -> Kimi `reasoning_effort`. `max`/`xhigh` collapse to `high`;
/// `low`/`medium`/`high` pass through; anything else defaults to `medium`.
fn reasoning_effort(anthropic: &Value) -> String {
    let effort = anthropic
        .get("output_config")
        .and_then(|c| c.get("effort"))
        .and_then(Value::as_str)
        .unwrap_or("");
    match effort {
        "max" | "xhigh" | "high" => "high",
        "low" => "low",
        "medium" => "medium",
        _ => "medium",
    }
    .to_string()
}

/// Flatten the Anthropic `system` (string or block array) into a single string, dropping any text
/// block whose text starts with the billing-header marker. Returns `None` when nothing survives.
fn build_system(system: Option<&Value>) -> Option<String> {
    match system {
        Some(Value::String(s)) => {
            if s.starts_with(BILLING_HEADER_PREFIX) || s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        Some(Value::Array(blocks)) => {
            let parts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .filter(|t| !t.starts_with(BILLING_HEADER_PREFIX))
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join("\n\n"))
            }
        }
        _ => None,
    }
}

/// Anthropic `tools` -> Kimi function tools, stripping the hosted `web_search_20250305` tool.
fn build_tools(tools: Option<&Value>) -> Option<Value> {
    let arr = tools?.as_array()?;
    let mapped: Vec<Value> = arr
        .iter()
        .filter(|t| t.get("name").and_then(Value::as_str) != Some("web_search_20250305"))
        .map(|t| {
            let name = t.get("name").and_then(Value::as_str).unwrap_or_default();
            let parameters = t.get("input_schema").cloned().unwrap_or_else(|| json!({}));
            let mut func = json!({"name": name, "parameters": parameters});
            if let Some(desc) = t.get("description").and_then(Value::as_str) {
                func["description"] = Value::String(desc.to_string());
            }
            json!({"type": "function", "function": func})
        })
        .collect();
    if mapped.is_empty() {
        None
    } else {
        Some(Value::Array(mapped))
    }
}

/// Anthropic `tool_choice` -> Kimi `tool_choice`. Emits the STRING form (`"none"`/`"required"`) or a
/// function object; `auto` (and anything targeting the stripped web-search tool) omits the field.
/// Never an untagged serde enum (which would serialize to `null`).
fn build_tool_choice(choice: Option<&Value>) -> Option<Value> {
    let choice = choice?;
    let ty = choice.get("type").and_then(Value::as_str)?;
    match ty {
        "auto" => None,
        "none" => Some(Value::String("none".to_string())),
        "any" | "required" => Some(Value::String("required".to_string())),
        "tool" => {
            let name = choice.get("name").and_then(Value::as_str)?;
            if name == "web_search_20250305" {
                return None;
            }
            Some(json!({"type": "function", "function": {"name": name}}))
        }
        _ => None,
    }
}

/// Map one Anthropic `role:"user"` message, appending one or more Kimi messages. Text/image blocks
/// buffer into a `role:"user"` message; a `tool_result` flushes that buffer then emits a
/// `role:"tool"` message (Kimi requires tool results as their own turn).
fn build_user(msg: &Value, out: &mut Vec<Value>) {
    let content = msg.get("content");
    // Plain-string content is the common short case.
    if let Some(Value::String(s)) = content {
        out.push(json!({"role": "user", "content": s}));
        return;
    }
    let Some(blocks) = content.and_then(Value::as_array) else {
        return;
    };

    let mut parts: Vec<Value> = Vec::new();
    let mut has_image = false;

    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    parts.push(json!({"type": "text", "text": t}));
                }
            }
            Some("image") => {
                if let Some(url) = image_url(block.get("source")) {
                    has_image = true;
                    parts.push(json!({"type": "image_url", "image_url": {"url": url}}));
                }
            }
            Some("tool_result") => {
                flush_user(out, &mut parts, &mut has_image);
                out.push(build_tool_result(block));
            }
            _ => {}
        }
    }
    flush_user(out, &mut parts, &mut has_image);
}

/// Flush the accumulated user parts into a `role:"user"` message: a bare string when there are no
/// images, an OpenAI-style parts array otherwise. Clears the accumulators.
fn flush_user(out: &mut Vec<Value>, parts: &mut Vec<Value>, has_image: &mut bool) {
    if parts.is_empty() {
        *has_image = false;
        return;
    }
    let content = if *has_image {
        Value::Array(std::mem::take(parts))
    } else {
        let text = parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        parts.clear();
        Value::String(text)
    };
    *has_image = false;
    out.push(json!({"role": "user", "content": content}));
}

fn build_tool_result(block: &Value) -> Value {
    let tool_call_id = block
        .get("tool_use_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let is_error = block
        .get("is_error")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut content = tool_result_text(block.get("content"));
    if is_error {
        content = format!("[tool execution error]\n{content}");
    }
    json!({"role": "tool", "tool_call_id": tool_call_id, "content": content})
}

/// A `tool_result` content is a string or an array of blocks; flatten to text.
fn tool_result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Convert an Anthropic image `source` into a URL Kimi accepts (a `data:` URL for base64 sources).
fn image_url(source: Option<&Value>) -> Option<String> {
    let source = source?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png");
            let data = source.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media};base64,{data}"))
        }
        Some("url") => source
            .get("url")
            .and_then(Value::as_str)
            .map(str::to_string),
        _ => None,
    }
}

/// Map one Anthropic `role:"assistant"` message. Text -> `content`, thinking blocks ->
/// `reasoning_content` (joined "\n\n"), `tool_use` blocks -> `tool_calls`.
fn build_assistant(msg: &Value) -> Value {
    let content = msg.get("content");
    if let Some(Value::String(s)) = content {
        return json!({"role": "assistant", "content": s});
    }

    let mut texts: Vec<String> = Vec::new();
    let mut thinking: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(blocks) = content.and_then(Value::as_array) {
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        texts.push(t.to_string());
                    }
                }
                Some("thinking") => {
                    if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                        thinking.push(t.to_string());
                    }
                }
                Some("tool_use") => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or_default();
                    let name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    let arguments =
                        serde_json::to_string(&input).unwrap_or_else(|_| "{}".to_string());
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }));
                }
                _ => {}
            }
        }
    }

    let mut m = json!({"role": "assistant"});
    let obj = m.as_object_mut().expect("json! literal is an object");
    obj.insert(
        "content".to_string(),
        if texts.is_empty() {
            Value::Null
        } else {
            Value::String(texts.join("\n"))
        },
    );
    if !thinking.is_empty() {
        obj.insert(
            "reasoning_content".to_string(),
            Value::String(thinking.join("\n\n")),
        );
    }
    if !tool_calls.is_empty() {
        obj.insert("tool_calls".to_string(), Value::Array(tool_calls));
    }
    m
}

/// Stateful reducer: Kimi `chat.completion.chunk` SSE -> shared [`ReduceEvent`] stream.
pub struct Reducer {
    parser: SseLineParser,
    thinking_open: bool,
    text_open: bool,
    /// The `delta.tool_calls[].index` of the currently open tool block, if any.
    tool_index: Option<i64>,
    saw_tool: bool,
    pending_stop: Option<StopReason>,
    usage: Usage,
    terminal: bool,
}

impl Reducer {
    pub fn new(_model: &str) -> Self {
        Self {
            parser: SseLineParser::new(),
            thinking_open: false,
            text_open: false,
            tool_index: None,
            saw_tool: false,
            pending_stop: None,
            usage: Usage::default(),
            terminal: false,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<ReduceEvent> {
        let mut events = Vec::new();
        for value in self.parser.push(chunk) {
            self.handle(&value, &mut events);
        }
        events
    }

    /// Flush any open block and emit a terminal `Finish` if the stream ended without one.
    pub fn finish(&mut self) -> Vec<ReduceEvent> {
        let mut events = Vec::new();
        if !self.terminal {
            self.close_blocks(&mut events);
            let stop = self.pending_stop.unwrap_or(StopReason::EndTurn);
            events.push(ReduceEvent::Finish {
                stop_reason: stop,
                usage: self.usage,
                response_id: None,
                continuation_eligible: false,
            });
            self.terminal = true;
        }
        events
    }

    fn handle(&mut self, value: &Value, events: &mut Vec<ReduceEvent>) {
        // An upstream error payload aborts translation.
        if let Some(err) = value.get("error") {
            let message = err
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| err.to_string());
            events.push(ReduceEvent::Error { message });
            self.terminal = true;
            return;
        }

        if let Some(choice) = value
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
        {
            if let Some(delta) = choice.get("delta") {
                self.handle_reasoning(delta, events);
                self.handle_content(delta, events);
                self.handle_tool_calls(delta, events);
            }
            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                self.close_blocks(events);
                self.pending_stop = Some(self.map_finish(reason));
            }
        }

        // Usage arrives either on the finish chunk or a trailing usage-only chunk.
        if let Some(usage) = value.get("usage").filter(|u| !u.is_null()) {
            self.usage = map_usage(usage);
        }

        self.maybe_finish(events);
    }

    fn handle_reasoning(&mut self, delta: &Value, events: &mut Vec<ReduceEvent>) {
        let Some(text) = delta.get("reasoning_content").and_then(Value::as_str) else {
            return;
        };
        if text.is_empty() {
            return;
        }
        if !self.thinking_open {
            events.push(ReduceEvent::ThinkingStart);
            self.thinking_open = true;
        }
        events.push(ReduceEvent::ThinkingDelta(text.to_string()));
    }

    fn handle_content(&mut self, delta: &Value, events: &mut Vec<ReduceEvent>) {
        let Some(text) = delta.get("content").and_then(Value::as_str) else {
            return;
        };
        if text.is_empty() {
            return;
        }
        // Text closes any open thinking/tool block first (strict nesting).
        if self.thinking_open {
            events.push(ReduceEvent::ThinkingStop);
            self.thinking_open = false;
        }
        if self.tool_index.is_some() {
            events.push(ReduceEvent::ToolStop);
            self.tool_index = None;
        }
        if !self.text_open {
            events.push(ReduceEvent::TextStart);
            self.text_open = true;
        }
        events.push(ReduceEvent::TextDelta(text.to_string()));
    }

    fn handle_tool_calls(&mut self, delta: &Value, events: &mut Vec<ReduceEvent>) {
        let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) else {
            return;
        };
        for call in calls {
            self.saw_tool = true;
            let index = call.get("index").and_then(Value::as_i64).unwrap_or(0);
            let is_new = self.tool_index != Some(index);
            if is_new {
                // Close whatever block is currently open before opening this tool.
                if self.thinking_open {
                    events.push(ReduceEvent::ThinkingStop);
                    self.thinking_open = false;
                }
                if self.text_open {
                    events.push(ReduceEvent::TextStop);
                    self.text_open = false;
                }
                if self.tool_index.is_some() {
                    events.push(ReduceEvent::ToolStop);
                }
                let id = call.get("id").and_then(Value::as_str).unwrap_or_default();
                let name = call
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                events.push(ReduceEvent::ToolStart {
                    id: id.to_string(),
                    name: name.to_string(),
                });
                self.tool_index = Some(index);
            }
            if let Some(args) = call
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                && !args.is_empty()
            {
                events.push(ReduceEvent::ToolDelta(args.to_string()));
            }
        }
    }

    fn map_finish(&self, reason: &str) -> StopReason {
        match reason {
            "length" => StopReason::MaxTokens,
            "tool_calls" => StopReason::ToolUse,
            _ if self.saw_tool => StopReason::ToolUse,
            _ => StopReason::EndTurn,
        }
    }

    fn close_blocks(&mut self, events: &mut Vec<ReduceEvent>) {
        if self.tool_index.is_some() {
            events.push(ReduceEvent::ToolStop);
            self.tool_index = None;
        }
        if self.text_open {
            events.push(ReduceEvent::TextStop);
            self.text_open = false;
        }
        if self.thinking_open {
            events.push(ReduceEvent::ThinkingStop);
            self.thinking_open = false;
        }
    }

    /// Emit the terminal `Finish` once we have both a stop reason and usage. Usage may lag the
    /// finish chunk (`stream_options.include_usage`), so this is checked after every chunk.
    fn maybe_finish(&mut self, events: &mut Vec<ReduceEvent>) {
        if self.terminal {
            return;
        }
        if let Some(stop) = self.pending_stop
            && self.usage != Usage::default()
        {
            events.push(ReduceEvent::Finish {
                stop_reason: stop,
                usage: self.usage,
                response_id: None,
                continuation_eligible: false,
            });
            self.terminal = true;
        }
    }
}

/// Kimi usage -> Anthropic's four-way split. `input` is fresh (non-cached) prompt tokens.
fn map_usage(usage: &Value) -> Usage {
    let prompt = usage
        .get("prompt_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let cached = usage
        .get("prompt_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_i64)
        .or_else(|| usage.get("cached_tokens").and_then(Value::as_i64))
        .unwrap_or(0);
    Usage {
        input: (prompt - cached).max(0),
        output: completion,
        cache_read: cached,
        cache_write: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body(v: Value) -> Value {
        build_request_body(&v, "ignored-model", None).expect("build_request_body")
    }

    #[test]
    fn model_is_always_wire_model_and_max_tokens_clamped() {
        let out = body(json!({
            "model": "claude-opus-4-8",
            "max_tokens": 128000,
            "messages": [],
        }));
        assert_eq!(out["model"], "kimi-for-coding");
        assert_eq!(out["max_tokens"], 32000);
        assert_eq!(out["stream"], true);
        assert_eq!(out["stream_options"]["include_usage"], true);
    }

    #[test]
    fn max_tokens_defaults_when_absent() {
        let out = body(json!({"messages": []}));
        assert_eq!(out["max_tokens"], 32000);
    }

    #[test]
    fn system_joins_and_drops_billing_header() {
        let out = body(json!({
            "max_tokens": 100,
            "system": [
                {"type": "text", "text": "x-anthropic-billing-header: 123"},
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [],
        }));
        let sys = &out["messages"][0];
        assert_eq!(sys["role"], "system");
        assert_eq!(sys["content"], "You are helpful.\n\nBe concise.");
    }

    #[test]
    fn user_text_and_tool_result_and_assistant_mapping() {
        let out = body(json!({
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think"},
                    {"type": "text", "text": "calling a tool"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "a.rs"}},
                ]},
                {"role": "user", "content": [
                    {"type": "text", "text": "here"},
                    {"type": "tool_result", "tool_use_id": "toolu_1", "is_error": true,
                     "content": "boom"},
                ]},
            ],
        }));
        let msgs = out["messages"].as_array().unwrap();
        // user, assistant, user (flushed text), tool
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "hello");

        let asst = &msgs[1];
        assert_eq!(asst["role"], "assistant");
        assert_eq!(asst["content"], "calling a tool");
        assert_eq!(asst["reasoning_content"], "let me think");
        assert_eq!(asst["tool_calls"][0]["id"], "toolu_1");
        assert_eq!(asst["tool_calls"][0]["type"], "function");
        assert_eq!(asst["tool_calls"][0]["function"]["name"], "Read");
        assert_eq!(
            asst["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a.rs\"}"
        );

        assert_eq!(msgs[2]["role"], "user");
        assert_eq!(msgs[2]["content"], "here");

        let tool = &msgs[3];
        assert_eq!(tool["role"], "tool");
        assert_eq!(tool["tool_call_id"], "toolu_1");
        assert_eq!(tool["content"], "[tool execution error]\nboom");
    }

    #[test]
    fn user_image_produces_parts_array() {
        let out = body(json!({
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                 "data": "AAAA"}},
            ]}],
        }));
        let content = &out["messages"][0]["content"];
        assert!(content.is_array());
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AAAA");
    }

    #[test]
    fn tools_mapping_and_web_search_stripped() {
        let out = body(json!({
            "max_tokens": 100,
            "messages": [],
            "tools": [
                {"name": "Read", "description": "read a file",
                 "input_schema": {"type": "object"}},
                {"name": "web_search_20250305", "input_schema": {"type": "object"}},
            ],
        }));
        let tools = out["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1, "web_search_20250305 must be stripped");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "Read");
        assert_eq!(tools[0]["function"]["description"], "read a file");
        assert_eq!(tools[0]["function"]["parameters"]["type"], "object");
    }

    #[test]
    fn tool_choice_none_serializes_to_string() {
        let out = body(json!({
            "max_tokens": 100,
            "messages": [],
            "tool_choice": {"type": "none"},
        }));
        // Assert on serialized JSON: it must be the bare string "none", never null or an object.
        let serialized = serde_json::to_string(&out["tool_choice"]).unwrap();
        assert_eq!(serialized, "\"none\"");
    }

    #[test]
    fn tool_choice_auto_is_omitted_and_any_becomes_required() {
        let auto = body(json!({
            "max_tokens": 100, "messages": [], "tool_choice": {"type": "auto"},
        }));
        assert!(auto.get("tool_choice").is_none());

        let any = body(json!({
            "max_tokens": 100, "messages": [], "tool_choice": {"type": "any"},
        }));
        assert_eq!(any["tool_choice"], "required");
    }

    #[test]
    fn tool_choice_tool_becomes_function_and_web_search_dropped() {
        let out = body(json!({
            "max_tokens": 100, "messages": [],
            "tool_choice": {"type": "tool", "name": "Read"},
        }));
        assert_eq!(out["tool_choice"]["type"], "function");
        assert_eq!(out["tool_choice"]["function"]["name"], "Read");

        let dropped = body(json!({
            "max_tokens": 100, "messages": [],
            "tool_choice": {"type": "tool", "name": "web_search_20250305"},
        }));
        assert!(dropped.get("tool_choice").is_none());
    }

    #[test]
    fn effort_max_maps_to_high_and_thinking_enabled() {
        let out = body(json!({
            "max_tokens": 100, "messages": [],
            "output_config": {"effort": "max"},
        }));
        assert_eq!(out["reasoning_effort"], "high");
        assert_eq!(out["thinking"]["type"], "enabled");
    }

    #[test]
    fn effort_default_is_medium() {
        let out = body(json!({"max_tokens": 100, "messages": []}));
        assert_eq!(out["reasoning_effort"], "medium");
    }

    #[test]
    fn thinking_disabled_drops_reasoning_and_thinking() {
        let out = body(json!({
            "max_tokens": 100, "messages": [],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "max"},
        }));
        assert!(out.get("reasoning_effort").is_none());
        assert!(out.get("thinking").is_none());
    }

    #[test]
    fn prompt_cache_key_from_session_id() {
        let out = build_request_body(
            &json!({"max_tokens": 100, "messages": []}),
            "m",
            Some("sess-1"),
        )
        .unwrap();
        assert_eq!(out["prompt_cache_key"], "sess-1");
    }

    #[test]
    fn headers_carry_kimi_identity() {
        let headers = request_headers("tok", None, None);
        let get = |k: &str| headers.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("authorization"), Some("Bearer tok".to_string()));
        assert_eq!(get("X-Msh-Platform"), Some("kimi_cli".to_string()));
        assert_eq!(get("X-Msh-Version"), Some("1.37.0".to_string()));
        assert_eq!(get("user-agent"), Some("KimiCLI/1.37.0".to_string()));
        assert!(get("X-Msh-Device-Id").is_some());
    }

    // ---- Reducer ----

    fn frame(v: Value) -> Vec<u8> {
        format!("data: {v}\n\n").into_bytes()
    }

    fn content_chunk(text: &str) -> Value {
        json!({"choices": [{"delta": {"content": text}}]})
    }

    #[test]
    fn multi_frame_content_deltas() {
        let mut r = Reducer::new("m");
        let mut events = Vec::new();
        events.extend(r.push(&frame(content_chunk("Hel"))));
        events.extend(r.push(&frame(content_chunk("lo"))));
        assert_eq!(events[0], ReduceEvent::TextStart);
        assert_eq!(events[1], ReduceEvent::TextDelta("Hel".into()));
        assert_eq!(events[2], ReduceEvent::TextDelta("lo".into()));
    }

    #[test]
    fn reasoning_then_content_closes_thinking_before_text() {
        let mut r = Reducer::new("m");
        let mut events = Vec::new();
        events.extend(r.push(&frame(json!({"choices": [{"delta":
            {"reasoning_content": "hmm"}}]}))));
        events.extend(r.push(&frame(content_chunk("answer"))));
        assert_eq!(events[0], ReduceEvent::ThinkingStart);
        assert_eq!(events[1], ReduceEvent::ThinkingDelta("hmm".into()));
        assert_eq!(events[2], ReduceEvent::ThinkingStop);
        assert_eq!(events[3], ReduceEvent::TextStart);
        assert_eq!(events[4], ReduceEvent::TextDelta("answer".into()));
    }

    #[test]
    fn tool_calls_split_across_frames() {
        let mut r = Reducer::new("m");
        let mut events = Vec::new();
        events.extend(r.push(&frame(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "Read", "arguments": "{\"pa"}}]}}]}))));
        events.extend(r.push(&frame(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": "th\":\"a\"}"}}]}}]}))));
        assert_eq!(
            events[0],
            ReduceEvent::ToolStart {
                id: "call_1".into(),
                name: "Read".into()
            }
        );
        assert_eq!(events[1], ReduceEvent::ToolDelta("{\"pa".into()));
        assert_eq!(events[2], ReduceEvent::ToolDelta("th\":\"a\"}".into()));
    }

    #[test]
    fn finish_tool_calls_emits_finish_tool_use_with_usage() {
        let mut r = Reducer::new("m");
        let mut events = Vec::new();
        events.extend(r.push(&frame(json!({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "type": "function",
             "function": {"name": "Read", "arguments": "{}"}}]}}]}))));
        // finish chunk (no usage yet)
        events.extend(r.push(&frame(json!({"choices": [{"delta": {},
            "finish_reason": "tool_calls"}]}))));
        // trailing usage-only chunk
        events.extend(r.push(&frame(json!({"choices": [],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                      "prompt_tokens_details": {"cached_tokens": 30}}}))));
        // ToolStop happens at finish, Finish at usage arrival.
        assert!(events.contains(&ReduceEvent::ToolStop));
        let finish = events.last().unwrap();
        assert_eq!(
            finish,
            &ReduceEvent::Finish {
                stop_reason: StopReason::ToolUse,
                usage: Usage {
                    input: 70,
                    output: 10,
                    cache_read: 30,
                    cache_write: 0
                },
                response_id: None,
                continuation_eligible: false,
            }
        );
        // No double finish.
        assert!(r.finish().is_empty());
    }

    #[test]
    fn truncated_stream_finish_emits_finish() {
        let mut r = Reducer::new("m");
        let mut events = r.push(&frame(content_chunk("partial")));
        events.extend(r.finish());
        assert!(events.contains(&ReduceEvent::TextStart));
        assert!(events.contains(&ReduceEvent::TextStop));
        assert_eq!(
            events.last().unwrap(),
            &ReduceEvent::Finish {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
                response_id: None,
                continuation_eligible: false,
            }
        );
    }

    #[test]
    fn error_payload_becomes_error_event() {
        let mut r = Reducer::new("m");
        let events = r.push(&frame(json!({"error": {"message": "rate limited"}})));
        assert_eq!(
            events[0],
            ReduceEvent::Error {
                message: "rate limited".into()
            }
        );
    }

    #[test]
    fn empty_reasoning_emits_nothing() {
        let mut r = Reducer::new("m");
        let events = r.push(&frame(json!({"choices": [{"delta":
            {"reasoning_content": ""}}]})));
        assert!(events.is_empty());
    }
}
