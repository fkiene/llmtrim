//! Codex (ChatGPT backend) provider for the subscription reroute.
//!
//! Translates an Anthropic `/v1/messages` request into the OpenAI *Responses* API shape the
//! ChatGPT `codex` backend accepts (`POST https://chatgpt.com/backend-api/codex/responses`), and
//! reduces the Responses SSE stream back into the shared [`ReduceEvent`] stream that
//! [`crate::reroute::sse::AnthropicSseEncoder`] re-encodes as Anthropic SSE.
//!
//! The Anthropic body is read positionally with `serde_json::Value` accessors (`.get()` /
//! `.as_*`) rather than deriving structs, matching how `serve.rs` handles intercepted bodies.

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::reroute::sse::{ReduceEvent, SseLineParser, StopReason, Usage};

pub const HOST: &str = "chatgpt.com";
pub const PATH: &str = "/backend-api/codex/responses";

/// The Claude Code hosted web-search tool. Translated to the Codex `web_search` tool (not a
/// function tool) so the model can actually search; see [`web_search_tool`].
const WEB_SEARCH_TOOL: &str = "web_search_20250305";

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

/// Build the Codex Responses request body from an intercepted Anthropic `/v1/messages` body.
///
/// `model` is already resolved to the upstream id (the caller stripped `[1m]`/`-fast` and mapped
/// tiers). `session_id` is the `x-claude-code-session-id` header value if present; it becomes the
/// Responses `prompt_cache_key`.
pub fn build_request_body(
    anthropic: &Value,
    model: &str,
    session_id: Option<&str>,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));

    // instructions = flattened system text (skip when empty).
    if let Some(instr) = anthropic.get("system").map(flatten_text)
        && !instr.is_empty()
    {
        body.insert("instructions".into(), json!(instr));
    }

    body.insert("input".into(), Value::Array(build_input(anthropic)));

    let tools = build_tools(anthropic);
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }

    if let Some(tc) = build_tool_choice(anthropic.get("tool_choice")) {
        body.insert("tool_choice".into(), tc);
    }

    body.insert("store".into(), json!(false));
    body.insert("stream".into(), json!(true));
    body.insert("parallel_tool_calls".into(), json!(true));

    // Reasoning + encrypted-content include are coupled: both present only when an effort is set.
    if let Some(effort) = reasoning_effort(anthropic) {
        body.insert("include".into(), json!(["reasoning.encrypted_content"]));
        body.insert(
            "reasoning".into(),
            json!({ "effort": effort, "summary": "auto" }),
        );
    }

    if let Some(sid) = session_id {
        body.insert("prompt_cache_key".into(), json!(sid));
    }

    // text = { verbosity, format? }
    let mut text = Map::new();
    text.insert("verbosity".into(), json!("low"));
    if let Some(fmt) = anthropic
        .get("output_config")
        .and_then(|c| c.get("format"))
        .map(build_format)
    {
        text.insert("format".into(), fmt);
    }
    body.insert("text".into(), Value::Object(text));

    Ok(Value::Object(body))
}

/// Flatten an Anthropic `system` (string or array of `{type:"text", text}` blocks) to one string.
fn flatten_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Normalize a message `content` (string or array of blocks) into a slice of block Values.
/// A bare string becomes a single synthetic text block.
fn content_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({ "type": "text", "text": s })],
        Value::Array(arr) => arr.clone(),
        _ => Vec::new(),
    }
}

/// Convert an Anthropic image block into a Responses `image_url` (a data URL for base64 sources).
fn image_url(block: &Value) -> Option<String> {
    let source = block.get("source")?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source.get("media_type").and_then(Value::as_str)?;
            let data = source.get("data").and_then(Value::as_str)?;
            Some(format!("data:{media};base64,{data}"))
        }
        Some("url") => source.get("url").and_then(Value::as_str).map(String::from),
        _ => None,
    }
}

/// Render a `tool_result` block's content into the Responses `function_call_output.output` string.
fn tool_result_output(block: &Value) -> String {
    let body = match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p.get("type").and_then(Value::as_str) {
                Some("image") => "[image omitted]".to_string(),
                _ => p
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    if block.get("is_error").and_then(Value::as_bool) == Some(true) {
        format!("[tool execution error]\n{body}")
    } else {
        body
    }
}

/// Build the Responses `input[]` array from Anthropic `messages`.
fn build_input(anthropic: &Value) -> Vec<Value> {
    let mut input = Vec::new();
    let Some(messages) = anthropic.get("messages").and_then(Value::as_array) else {
        return input;
    };

    for msg in messages {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");
        let content = msg.get("content").cloned().unwrap_or(Value::Null);
        let blocks = content_blocks(&content);

        match role {
            "assistant" => {
                let mut parts: Vec<Value> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(Value::as_str) {
                                parts.push(json!({ "type": "output_text", "text": t }));
                            }
                        }
                        Some("tool_use") => {
                            flush_message(&mut input, "assistant", &mut parts);
                            let args = b.get("input").cloned().unwrap_or(json!({}));
                            let args_str = if args.is_null() {
                                "{}".to_string()
                            } else {
                                serde_json::to_string(&args).unwrap_or_else(|_| "{}".into())
                            };
                            input.push(json!({
                                "type": "function_call",
                                "call_id": b.get("id").and_then(Value::as_str).unwrap_or(""),
                                "name": b.get("name").and_then(Value::as_str).unwrap_or(""),
                                "arguments": args_str,
                            }));
                        }
                        _ => {}
                    }
                }
                flush_message(&mut input, "assistant", &mut parts);
            }
            "system" | "developer" => {
                let text = flatten_text(&content);
                input.push(json!({
                    "type": "message",
                    "role": "developer",
                    "content": [{ "type": "input_text", "text": text }],
                }));
            }
            _ => {
                // Treat any other role as user.
                let mut parts: Vec<Value> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(Value::as_str) {
                                parts.push(json!({ "type": "input_text", "text": t }));
                            }
                        }
                        Some("image") => {
                            if let Some(url) = image_url(b) {
                                parts.push(json!({
                                    "type": "input_image",
                                    "image_url": url,
                                    "detail": Value::Null,
                                }));
                            }
                        }
                        Some("tool_result") => {
                            flush_message(&mut input, "user", &mut parts);
                            let call_id =
                                b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                            let mut output = tool_result_output(b);
                            // If we stripped an absurd offset from this Read call earlier, tell the
                            // model so it stops re-issuing the same impossible read.
                            if let Some(rw) =
                                crate::reroute::read_rewrite::read_offset_rewrite(call_id)
                            {
                                output.push_str("\n\n");
                                output.push_str(
                                    &crate::reroute::read_rewrite::read_offset_rewrite_note(&rw),
                                );
                            }
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output,
                            }));
                        }
                        _ => {}
                    }
                }
                flush_message(&mut input, "user", &mut parts);
            }
        }
    }
    input
}

/// Emit a `message` item from accumulated parts (if any) and clear the buffer.
fn flush_message(input: &mut Vec<Value>, role: &str, parts: &mut Vec<Value>) {
    if parts.is_empty() {
        return;
    }
    input.push(json!({
        "type": "message",
        "role": role,
        "content": Value::Array(std::mem::take(parts)),
    }));
}

/// Map Anthropic `tools[]` to Responses tools. The hosted web-search tool becomes the Codex
/// `web_search` tool (so the model can actually search); every other tool becomes a function tool.
fn build_tools(anthropic: &Value) -> Vec<Value> {
    let Some(tools) = anthropic.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .map(|t| {
            if is_web_search(t) {
                return web_search_tool(t);
            }
            let mut obj = Map::new();
            obj.insert("type".into(), json!("function"));
            obj.insert(
                "name".into(),
                json!(t.get("name").and_then(Value::as_str).unwrap_or("")),
            );
            if let Some(desc) = t.get("description").and_then(Value::as_str) {
                obj.insert("description".into(), json!(desc));
            }
            obj.insert(
                "parameters".into(),
                t.get("input_schema").cloned().unwrap_or(json!({})),
            );
            Value::Object(obj)
        })
        .collect()
}

/// Build the Codex `web_search` tool from Anthropic's hosted web-search tool, carrying over the
/// `allowed_domains`/`blocked_domains` filters when non-empty.
fn web_search_tool(tool: &Value) -> Value {
    let domains = |key: &str| {
        tool.get(key)
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
    };
    let mut filters = Map::new();
    if let Some(allowed) = domains("allowed_domains") {
        filters.insert("allowed_domains".into(), json!(allowed));
    }
    if let Some(blocked) = domains("blocked_domains") {
        filters.insert("blocked_domains".into(), json!(blocked));
    }
    let mut obj = Map::new();
    obj.insert("type".into(), json!("web_search"));
    obj.insert("external_web_access".into(), json!(false));
    obj.insert("search_content_types".into(), json!(["text", "image"]));
    if !filters.is_empty() {
        obj.insert("filters".into(), Value::Object(filters));
    }
    Value::Object(obj)
}

fn is_web_search(tool: &Value) -> bool {
    tool.get("name").and_then(Value::as_str) == Some(WEB_SEARCH_TOOL)
        || tool.get("type").and_then(Value::as_str) == Some(WEB_SEARCH_TOOL)
}

/// Translate Anthropic `tool_choice` into the Responses form.
///
/// Returns a STRING `"none"`/`"required"` or a `{type:"function", name}` object. `auto` (and an
/// absent choice) return `None` so the key is omitted. A choice targeting the hosted web-search
/// tool is dropped.
fn build_tool_choice(tc: Option<&Value>) -> Option<Value> {
    let tc = tc?;
    match tc.get("type").and_then(Value::as_str) {
        Some("auto") | None => None,
        Some("none") => Some(json!("none")),
        Some("any") | Some("required") => Some(json!("required")),
        Some("tool") => {
            let name = tc.get("name").and_then(Value::as_str)?;
            if name == WEB_SEARCH_TOOL {
                return None;
            }
            Some(json!({ "type": "function", "name": name }))
        }
        _ => None,
    }
}

/// Resolve the reasoning effort from `output_config.effort`.
///
/// `max`/`xhigh` -> `"xhigh"`; `low`/`medium`/`high` pass through; anything else (including
/// `none`/absent) -> `None`, which drops both `reasoning` and the encrypted-content `include`.
fn reasoning_effort(anthropic: &Value) -> Option<String> {
    let effort = anthropic
        .get("output_config")
        .and_then(|c| c.get("effort"))
        .and_then(Value::as_str)?;
    match effort {
        "max" | "xhigh" => Some("xhigh".into()),
        "low" | "medium" | "high" => Some(effort.into()),
        _ => None,
    }
}

/// Build the Responses `text.format` from an Anthropic `output_config.format`.
fn build_format(fmt: &Value) -> Value {
    match fmt.get("type").and_then(Value::as_str) {
        Some("json_schema") => json!({
            "type": "json_schema",
            "name": fmt.get("name").and_then(Value::as_str).unwrap_or("response"),
            "schema": fmt.get("schema").cloned().unwrap_or(json!({})),
            "strict": true,
        }),
        Some("json_object") => json!({ "type": "json_object" }),
        _ => json!({ "type": "text" }),
    }
}

// ---------------------------------------------------------------------------------------------
// Headers
// ---------------------------------------------------------------------------------------------

/// Headers to SET on the rewritten upstream request (hyper sets host/content-length).
pub fn request_headers(
    access_token: &str,
    account_id: Option<&str>,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "text/event-stream".to_string()),
        (
            "authorization".to_string(),
            format!("Bearer {access_token}"),
        ),
        ("originator".to_string(), "llmtrim".to_string()),
        (
            "openai-beta".to_string(),
            "responses=experimental".to_string(),
        ),
        (
            "user-agent".to_string(),
            format!("llmtrim/{}", env!("CARGO_PKG_VERSION")),
        ),
    ];
    if let Some(acc) = account_id {
        headers.push(("ChatGPT-Account-Id".to_string(), acc.to_string()));
    }
    if let Some(sid) = session_id {
        headers.push(("session_id".to_string(), sid.to_string()));
        headers.push(("x-client-request-id".to_string(), sid.to_string()));
        headers.push(("x-codex-window-id".to_string(), format!("{sid}:0")));
    }
    headers
}

// ---------------------------------------------------------------------------------------------
// Response reducer
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    None,
    Thinking,
    Text,
    Tool,
}

/// Stateful reducer: Codex Responses SSE -> shared [`ReduceEvent`] stream.
pub struct Reducer {
    parser: SseLineParser,
    open: Open,
    saw_tool_use: bool,
    /// Current tool call's name / id / accumulated argument JSON. Tool args are buffered rather
    /// than streamed so they can be sanitized ([`read_rewrite`]) as a complete JSON value before
    /// the tool call reaches the client (Claude Code only executes a tool after the full block).
    tool_name: String,
    tool_id: String,
    tool_buf: String,
    /// Whether the buffered args have been sanitized + emitted for the open tool.
    tool_flushed: bool,
    /// The Responses `output_index` of the open tool call. Since args are buffered (not streamed),
    /// a different item's `output_item.done` must NOT close/flush a half-buffered tool — so a `done`
    /// is only honored for the item that owns the currently open tool.
    tool_output_index: Option<i64>,
    terminal_seen: bool,
}

impl Reducer {
    pub fn new(_model: &str) -> Self {
        Self {
            parser: SseLineParser::new(),
            open: Open::None,
            saw_tool_use: false,
            tool_name: String::new(),
            tool_id: String::new(),
            tool_buf: String::new(),
            tool_flushed: false,
            tool_output_index: None,
            terminal_seen: false,
        }
    }

    /// Sanitize + emit the buffered tool args once (idempotent per tool call).
    fn flush_tool(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.tool_flushed {
            return;
        }
        self.tool_flushed = true;
        let sanitized = crate::reroute::read_rewrite::sanitize_read_args(
            &self.tool_name,
            &self.tool_buf,
            Some(&self.tool_id),
        );
        if !sanitized.is_empty() {
            out.push(ReduceEvent::ToolDelta(sanitized));
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<ReduceEvent> {
        let mut out = Vec::new();
        for v in self.parser.push(chunk) {
            self.handle(&v, &mut out);
        }
        out
    }

    /// Flush any still-open block at stream end; emit a `Finish EndTurn` if no terminal was seen.
    pub fn finish(&mut self) -> Vec<ReduceEvent> {
        let mut out = Vec::new();
        self.close_open(&mut out);
        if !self.terminal_seen {
            self.terminal_seen = true;
            out.push(ReduceEvent::Finish {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
            });
        }
        out
    }

    /// Close whatever block is open, emitting its `*Stop`.
    fn close_open(&mut self, out: &mut Vec<ReduceEvent>) {
        match self.open {
            Open::Thinking => out.push(ReduceEvent::ThinkingStop),
            Open::Text => out.push(ReduceEvent::TextStop),
            Open::Tool => {
                self.flush_tool(out);
                out.push(ReduceEvent::ToolStop);
            }
            Open::None => {}
        }
        self.open = Open::None;
    }

    fn handle(&mut self, v: &Value, out: &mut Vec<ReduceEvent>) {
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "response.output_item.added" => {
                let item = v.get("item").cloned().unwrap_or(Value::Null);
                match item.get("type").and_then(Value::as_str) {
                    Some("reasoning") => {} // ignore reasoning items
                    Some("message") => {
                        self.close_open(out);
                        out.push(ReduceEvent::TextStart);
                        self.open = Open::Text;
                    }
                    Some("function_call") => {
                        self.close_open(out);
                        self.saw_tool_use = true;
                        self.tool_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        self.tool_name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        self.tool_buf.clear();
                        self.tool_flushed = false;
                        self.tool_output_index = v.get("output_index").and_then(Value::as_i64);
                        out.push(ReduceEvent::ToolStart {
                            id: self.tool_id.clone(),
                            name: self.tool_name.clone(),
                        });
                        self.open = Open::Tool;
                    }
                    _ => {}
                }
            }
            "response.reasoning_summary_part.added" => {
                if self.open == Open::Thinking {
                    out.push(ReduceEvent::ThinkingDelta("\n\n".to_string()));
                }
            }
            "response.reasoning_summary_text.delta" => {
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                // Lazily open a thinking block only on the first non-empty summary delta.
                if delta.is_empty() {
                    return;
                }
                if self.open != Open::Thinking {
                    self.close_open(out);
                    out.push(ReduceEvent::ThinkingStart);
                    self.open = Open::Thinking;
                }
                out.push(ReduceEvent::ThinkingDelta(delta.to_string()));
            }
            "response.output_text.delta" => {
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if self.open != Open::Text {
                    self.close_open(out);
                    out.push(ReduceEvent::TextStart);
                    self.open = Open::Text;
                }
                out.push(ReduceEvent::TextDelta(delta.to_string()));
            }
            "response.function_call_arguments.delta" => {
                // Buffer, don't stream: the full args are needed to sanitize as one JSON value.
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                self.tool_buf.push_str(delta);
            }
            "response.function_call_arguments.done" => {
                // Prefer the terminal `arguments` (authoritative) when the deltas were empty.
                if self.tool_buf.is_empty()
                    && let Some(args) = v.get("arguments").and_then(Value::as_str)
                {
                    self.tool_buf.push_str(args);
                }
                self.flush_tool(out);
            }
            "response.output_item.done" => {
                // With buffered tool args, ignore a `done` that belongs to a different item than the
                // open tool (parallel tool use): closing here would flush a half-buffered tool as if
                // complete. A non-matching `done` is a no-op; the tool closes on its own `done`, the
                // next item's `added`, or stream end.
                if self.open == Open::Tool
                    && let Some(done_idx) = v.get("output_index").and_then(Value::as_i64)
                    && self.tool_output_index.is_some()
                    && Some(done_idx) != self.tool_output_index
                {
                    return;
                }
                self.close_open(out);
            }
            "response.completed" | "response.done" => {
                self.finish_terminal(v, false, out);
            }
            "response.incomplete" => {
                self.finish_terminal(v, true, out);
            }
            "codex.rate_limits" => {
                if rate_limited(v) {
                    self.terminal_seen = true;
                    out.push(ReduceEvent::Error {
                        message: "rate limit reached".to_string(),
                    });
                }
            }
            "response.failed" | "response.error" | "error" => {
                self.terminal_seen = true;
                out.push(ReduceEvent::Error {
                    message: error_message(v),
                });
            }
            _ => {}
        }
    }

    fn finish_terminal(&mut self, v: &Value, incomplete: bool, out: &mut Vec<ReduceEvent>) {
        if self.terminal_seen {
            return;
        }
        self.close_open(out);
        let stop_reason = if incomplete {
            StopReason::MaxTokens
        } else if self.saw_tool_use {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };
        let usage = v
            .get("response")
            .and_then(|r| r.get("usage"))
            .or_else(|| v.get("usage"))
            .map(map_usage)
            .unwrap_or_default();
        self.terminal_seen = true;
        out.push(ReduceEvent::Finish { stop_reason, usage });
    }
}

fn rate_limited(v: &Value) -> bool {
    v.get("limit_reached").and_then(Value::as_bool) == Some(true)
        || v.get("rate_limits")
            .and_then(|r| r.get("limit_reached"))
            .and_then(Value::as_bool)
            == Some(true)
}

fn error_message(v: &Value) -> String {
    v.get("error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .or_else(|| {
            v.get("response")
                .and_then(|r| r.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
        })
        .or_else(|| v.get("message").and_then(Value::as_str))
        .unwrap_or("upstream error")
        .to_string()
}

/// Map a Responses `usage` object onto the shared four-way [`Usage`] split.
fn map_usage(u: &Value) -> Usage {
    let input_tokens = u.get("input_tokens").and_then(Value::as_i64).unwrap_or(0);
    let cached = u
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output = u.get("output_tokens").and_then(Value::as_i64).unwrap_or(0);
    Usage {
        input: (input_tokens - cached).max(0),
        output,
        cache_read: cached,
        cache_write: 0,
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- request ----

    #[test]
    fn system_becomes_instructions() {
        let body = build_request_body(
            &json!({ "system": "Be concise.", "messages": [] }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        assert_eq!(body["instructions"], "Be concise.");
    }

    #[test]
    fn system_array_is_flattened() {
        let body = build_request_body(
            &json!({
                "system": [
                    { "type": "text", "text": "Line 1" },
                    { "type": "text", "text": "Line 2" }
                ],
                "messages": []
            }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        assert_eq!(body["instructions"], "Line 1\nLine 2");
    }

    #[test]
    fn user_text_turn_maps_to_input_text() {
        let body = build_request_body(
            &json!({
                "messages": [
                    { "role": "user", "content": [{ "type": "text", "text": "hi there" }] }
                ]
            }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().expect("input array");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][0]["text"], "hi there");
    }

    #[test]
    fn tool_use_and_result_round_trip() {
        let body = build_request_body(
            &json!({
                "messages": [
                    { "role": "user", "content": "run it" },
                    {
                        "role": "assistant",
                        "content": [
                            { "type": "text", "text": "calling" },
                            { "type": "tool_use", "id": "call_1", "name": "Read", "input": { "path": "x" } }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            { "type": "tool_result", "tool_use_id": "call_1", "content": "file body" }
                        ]
                    }
                ]
            }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().expect("input array");
        // user msg, assistant text msg, function_call, function_call_output
        assert_eq!(input.len(), 4);
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "message");
        assert_eq!(input[1]["role"], "assistant");
        assert_eq!(input[1]["content"][0]["type"], "output_text");
        assert_eq!(input[2]["type"], "function_call");
        assert_eq!(input[2]["call_id"], "call_1");
        assert_eq!(input[2]["name"], "Read");
        assert_eq!(input[2]["arguments"], "{\"path\":\"x\"}");
        assert_eq!(input[3]["type"], "function_call_output");
        assert_eq!(input[3]["call_id"], "call_1");
        assert_eq!(input[3]["output"], "file body");
    }

    #[test]
    fn tool_result_error_is_prefixed() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            { "type": "tool_result", "tool_use_id": "c1", "content": "boom", "is_error": true }
                        ]
                    }
                ]
            }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        assert_eq!(body["input"][0]["output"], "[tool execution error]\nboom");
    }

    #[test]
    fn tools_map_and_web_search_translated() {
        let body = build_request_body(
            &json!({
                "messages": [],
                "tools": [
                    {
                        "name": "Read",
                        "description": "read a file",
                        "input_schema": { "type": "object" }
                    },
                    { "type": "web_search_20250305", "name": "web_search",
                      "allowed_domains": ["docs.rs"] },
                ]
            }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        let tools = body["tools"].as_array().expect("tools");
        assert_eq!(tools.len(), 2, "Read function + translated web_search");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["name"], "Read");
        // web_search is now a real Codex tool, not stripped, with its domain filter carried over.
        assert_eq!(tools[1]["type"], "web_search");
        assert_eq!(tools[1]["external_web_access"], false);
        assert_eq!(tools[1]["search_content_types"][0], "text");
        assert_eq!(tools[1]["filters"]["allowed_domains"][0], "docs.rs");
    }

    #[test]
    fn web_search_without_filters_omits_filters() {
        let body = build_request_body(
            &json!({ "messages": [], "tools": [ { "name": "web_search_20250305" } ] }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        let tools = body["tools"].as_array().expect("tools");
        assert_eq!(tools[0]["type"], "web_search");
        assert!(
            tools[0].get("filters").is_none(),
            "no filters key when empty"
        );
    }

    #[test]
    fn tool_choice_none_serializes_to_string() {
        let body = build_request_body(
            &json!({ "messages": [], "tool_choice": { "type": "none" } }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        let s = serde_json::to_string(&body).expect("serialize");
        assert!(
            s.contains("\"tool_choice\":\"none\""),
            "tool_choice must be the string \"none\", got: {s}"
        );
    }

    #[test]
    fn tool_choice_required_and_specific() {
        assert_eq!(
            build_tool_choice(Some(&json!({ "type": "any" }))),
            Some(json!("required"))
        );
        assert_eq!(
            build_tool_choice(Some(&json!({ "type": "required" }))),
            Some(json!("required"))
        );
        assert_eq!(
            build_tool_choice(Some(&json!({ "type": "tool", "name": "Read" }))),
            Some(json!({ "type": "function", "name": "Read" }))
        );
    }

    #[test]
    fn tool_choice_auto_is_omitted() {
        let body = build_request_body(
            &json!({ "messages": [], "tool_choice": { "type": "auto" } }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        assert!(body.get("tool_choice").is_none(), "auto omits tool_choice");
    }

    #[test]
    fn tool_choice_web_search_is_dropped() {
        assert_eq!(
            build_tool_choice(Some(
                &json!({ "type": "tool", "name": "web_search_20250305" })
            )),
            None
        );
    }

    #[test]
    fn effort_max_maps_to_xhigh_with_include() {
        let body = build_request_body(
            &json!({ "messages": [], "output_config": { "effort": "max" } }),
            "gpt-5.5",
            None,
        )
        .expect("build");
        assert_eq!(body["reasoning"]["effort"], "xhigh");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["include"][0], "reasoning.encrypted_content");
    }

    #[test]
    fn no_effort_omits_reasoning_and_include() {
        let body = build_request_body(&json!({ "messages": [] }), "gpt-5.5", None).expect("build");
        assert!(body.get("reasoning").is_none());
        assert!(body.get("include").is_none());
    }

    #[test]
    fn max_tokens_and_sampling_dropped_static_fields_present() {
        let body = build_request_body(
            &json!({
                "messages": [],
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["x"]
            }),
            "gpt-5.4",
            Some("sess-1"),
        )
        .expect("build");
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("stop_sequences").is_none());
        assert_eq!(body["model"], "gpt-5.4");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], true);
        assert_eq!(body["parallel_tool_calls"], true);
        assert_eq!(body["text"]["verbosity"], "low");
        assert_eq!(body["prompt_cache_key"], "sess-1");
    }

    #[test]
    fn model_passed_through_verbatim() {
        // Caller already stripped [1m]; this fn must not touch the model string.
        let body =
            build_request_body(&json!({ "messages": [] }), "gpt-5.3-codex", None).expect("build");
        assert_eq!(body["model"], "gpt-5.3-codex");
    }

    // ---- headers ----

    #[test]
    fn headers_include_static_and_conditional() {
        let h = request_headers("tok", Some("acc-1"), Some("sess-9"));
        let get = |k: &str| h.iter().find(|(n, _)| n == k).map(|(_, v)| v.clone());
        assert_eq!(get("authorization").as_deref(), Some("Bearer tok"));
        assert_eq!(get("content-type").as_deref(), Some("application/json"));
        assert_eq!(get("accept").as_deref(), Some("text/event-stream"));
        assert_eq!(get("originator").as_deref(), Some("llmtrim"));
        assert_eq!(
            get("openai-beta").as_deref(),
            Some("responses=experimental")
        );
        assert_eq!(get("ChatGPT-Account-Id").as_deref(), Some("acc-1"));
        assert_eq!(get("session_id").as_deref(), Some("sess-9"));
        assert_eq!(get("x-client-request-id").as_deref(), Some("sess-9"));
        assert_eq!(get("x-codex-window-id").as_deref(), Some("sess-9:0"));
        assert!(get("user-agent").unwrap().starts_with("llmtrim/"));
    }

    #[test]
    fn headers_omit_account_and_session_when_absent() {
        let h = request_headers("tok", None, None);
        assert!(h.iter().all(|(n, _)| n != "ChatGPT-Account-Id"));
        assert!(h.iter().all(|(n, _)| n != "session_id"));
        assert!(h.iter().all(|(n, _)| n != "x-codex-window-id"));
    }

    // ---- reducer ----

    /// Feed a whole SSE string through a fresh reducer in one shot.
    fn reduce(sse: &str) -> (Vec<ReduceEvent>, Reducer) {
        let mut r = Reducer::new("gpt-5.5");
        let events = r.push(sse.as_bytes());
        (events, r)
    }

    #[test]
    fn text_stream_produces_wellformed_events() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"Hello\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\" world\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"input_tokens_details\":{\"cached_tokens\":3},\"output_tokens\":5}}}\n\n",
        );
        let (events, _) = reduce(sse);
        assert_eq!(
            events,
            vec![
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("Hello".into()),
                ReduceEvent::TextDelta(" world".into()),
                ReduceEvent::TextStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage {
                        input: 7,
                        output: 5,
                        cache_read: 3,
                        cache_write: 0
                    },
                },
            ]
        );
    }

    #[test]
    fn function_call_args_split_across_frames() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"Read\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"{\\\"path\\\":\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":0,\"delta\":\"\\\"x\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"path\\\":\\\"x\\\"}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":2}}}\n\n",
        );
        let (events, _) = reduce(sse);
        assert_eq!(
            events,
            vec![
                ReduceEvent::ToolStart {
                    id: "call_1".into(),
                    name: "Read".into()
                },
                // Args are buffered across frames and emitted once, complete, on `done` — so they
                // can be sanitized ([`read_rewrite`]) as a whole JSON value before the client runs
                // the tool. A well-formed Read passes through unchanged.
                ReduceEvent::ToolDelta("{\"path\":\"x\"}".into()),
                ReduceEvent::ToolStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::ToolUse,
                    usage: Usage {
                        input: 4,
                        output: 2,
                        cache_read: 0,
                        cache_write: 0
                    },
                },
            ]
        );
    }

    #[test]
    fn foreign_item_done_does_not_flush_a_half_buffered_tool() {
        // A `function_call` on output_index 1 is mid-buffer when a DIFFERENT item (index 0) reports
        // `done`. That must NOT flush the tool with truncated args; the tool completes on its own
        // `done`, emitting the full, well-formed args exactly once.
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_9\",\"name\":\"Read\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"{\\\"file_path\\\":\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"web_search_call\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"output_index\":1,\"delta\":\"\\\"/a\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":1,\"arguments\":\"{\\\"file_path\\\":\\\"/a\\\"}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let (events, _) = reduce(sse);
        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ReduceEvent::ToolDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            deltas,
            vec!["{\"file_path\":\"/a\"}"],
            "one complete delta, not truncated"
        );
    }

    #[test]
    fn read_tool_call_with_absurd_offset_is_sanitized() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_off\",\"name\":\"Read\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{\\\"file_path\\\":\\\"/a\\\",\\\"offset\\\":5000000}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"type\":\"function_call\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n",
        );
        let (events, _) = reduce(sse);
        let emitted: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                ReduceEvent::ToolDelta(s) => Some(s.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(emitted.len(), 1, "one complete, sanitized args delta");
        let args: serde_json::Value = serde_json::from_str(emitted[0]).unwrap();
        assert!(
            args.get("offset").is_none(),
            "absurd offset stripped: {}",
            emitted[0]
        );
        assert_eq!(args["file_path"], "/a");
    }

    #[test]
    fn args_done_fills_when_no_deltas_arrived() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"c\",\"name\":\"Ls\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"output_index\":0,\"arguments\":\"{}\"}\n\n",
        );
        let (events, _) = reduce(sse);
        assert_eq!(
            events,
            vec![
                ReduceEvent::ToolStart {
                    id: "c".into(),
                    name: "Ls".into()
                },
                ReduceEvent::ToolDelta("{}".into()),
            ]
        );
    }

    #[test]
    fn reasoning_then_text_closes_thinking_first() {
        let sse = concat!(
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"output_index\":0,\"delta\":\"pondering\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"output_index\":1,\"item\":{\"type\":\"message\",\"id\":\"m\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":1,\"delta\":\"answer\"}\n\n",
        );
        let (events, _) = reduce(sse);
        assert_eq!(
            events,
            vec![
                ReduceEvent::ThinkingStart,
                ReduceEvent::ThinkingDelta("pondering".into()),
                ReduceEvent::ThinkingStop,
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("answer".into()),
            ]
        );
    }

    #[test]
    fn empty_reasoning_delta_emits_nothing() {
        let sse = "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"\"}\n\n";
        let (events, _) = reduce(sse);
        assert!(events.is_empty());
    }

    #[test]
    fn rate_limit_frame_becomes_error() {
        let sse = "data: {\"type\":\"codex.rate_limits\",\"limit_reached\":true}\n\n";
        let (events, r) = reduce(sse);
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0], ReduceEvent::Error { .. }));
        assert!(r.terminal_seen);
    }

    #[test]
    fn failed_frame_carries_message() {
        let sse = "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"nope\"}}}\n\n";
        let (events, _) = reduce(sse);
        assert_eq!(
            events,
            vec![ReduceEvent::Error {
                message: "nope".into()
            }]
        );
    }

    #[test]
    fn truncated_stream_finish_closes_open_block() {
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"message\",\"id\":\"m\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"delta\":\"partial\"}\n\n",
        );
        let mut r = Reducer::new("gpt-5.5");
        let streamed = r.push(sse.as_bytes());
        assert_eq!(
            streamed,
            vec![
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("partial".into())
            ]
        );
        let flushed = r.finish();
        assert_eq!(
            flushed,
            vec![
                ReduceEvent::TextStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                },
            ]
        );
    }

    #[test]
    fn finish_is_noop_after_terminal() {
        let sse = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}\n\n";
        let (_, mut r) = reduce(sse);
        assert!(r.finish().is_empty());
    }
}
