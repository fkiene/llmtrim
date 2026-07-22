//! Grok (xAI SuperGrok / Grok Build) subscription reroute provider.
//!
//! Translates an Anthropic `/v1/messages` request into the Grok CLI Responses API shape
//! (`POST https://cli-chat-proxy.grok.com/v1/responses`) and reduces the Responses SSE stream
//! back into the shared [`ReduceEvent`] stream that
//! [`crate::reroute::sse::AnthropicSseEncoder`] re-encodes as Anthropic SSE.
//!
//! Wire models: `grok-4.5` (flagship) and `grok-composer-2.5-fast` (cheap/fast). Auth is OAuth
//! against `auth.x.ai` (see [`crate::reroute::auth`]).

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::reroute::sse::{ReduceEvent, SseLineParser, StopReason, Usage};

pub const HOST: &str = "cli-chat-proxy.grok.com";
pub const PATH: &str = "/v1/responses";
const CLIENT_VERSION: &str = "0.2.93";

/// Claude Code hosted web-search tool id (translated to Grok `web_search`).
const WEB_SEARCH_TOOL: &str = "web_search_20250305";

// ---------------------------------------------------------------------------------------------
// Request
// ---------------------------------------------------------------------------------------------

/// Headers for the rewritten upstream request.
///
/// When `session_id` is present it is also sent as `x-grok-conv-id` (Chat Completions cache
/// affinity header from xAI docs). Responses caching itself is driven by `prompt_cache_key` in
/// the body; dual-sending is harmless (live probe: HTTP 200) and covers any proxy path that still
/// keys on the header.
pub fn request_headers(
    access_token: &str,
    _account_id: Option<&str>,
    session_id: Option<&str>,
) -> Vec<(String, String)> {
    let mut headers = vec![
        ("content-type".to_string(), "application/json".to_string()),
        ("accept".to_string(), "text/event-stream".to_string()),
        (
            "authorization".to_string(),
            format!("Bearer {access_token}"),
        ),
        ("x-xai-token-auth".to_string(), "xai-grok-cli".to_string()),
        (
            "x-grok-client-identifier".to_string(),
            "grok-shell".to_string(),
        ),
        (
            "x-grok-client-version".to_string(),
            CLIENT_VERSION.to_string(),
        ),
        (
            "user-agent".to_string(),
            format!("llmtrim/{}", env!("CARGO_PKG_VERSION")),
        ),
    ];
    if let Some(sid) = session_id {
        headers.push(("x-grok-conv-id".to_string(), sid.to_string()));
    }
    headers
}

/// Build the Grok Responses request body from an intercepted Anthropic `/v1/messages` body.
///
/// `model` is already resolved to an upstream id. `session_id` is the
/// `x-claude-code-session-id` header value if present; it becomes the Responses
/// `prompt_cache_key` so cli-chat-proxy can pin automatic prefix caching to the Claude Code
/// session (same field Codex/Kimi already set). Live probe: the field is accepted (HTTP 200)
/// and subsequent turns report `input_tokens_details.cached_tokens`.
pub fn build_request_body(
    anthropic: &Value,
    model: &str,
    session_id: Option<&str>,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));

    let mut instructions = crate::reroute::flatten_system_text(anthropic.get("system"));
    // Only advertise hosted tools when Claude Code offered them. We do not yet reduce hosted
    // search streams back into Anthropic server_tool blocks, so auto-injecting x_search every
    // turn would steer the model into tools Claude Code never sees.
    let tools = build_tools(anthropic);
    if tools
        .iter()
        .any(|t| t.get("type").and_then(Value::as_str) == Some("x_search"))
    {
        append_guidance(
            &mut instructions,
            "For requests to search X or Twitter, use the hosted x_search tool. Do not use Bash, curl, HTTP clients, or general web_search for X searches.",
        );
    }
    if tools
        .iter()
        .any(|t| t.get("type").and_then(Value::as_str) == Some("web_search"))
    {
        append_guidance(
            &mut instructions,
            "For general web searches, use the hosted web_search tool. Do not use shell commands, HTTP clients, or local tools to search the web.",
        );
    }
    if let Some(instr) = instructions {
        body.insert("instructions".into(), json!(instr));
    }

    body.insert("input".into(), Value::Array(build_input(anthropic)));
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
    }
    if let Some(tc) = build_tool_choice(anthropic.get("tool_choice")) {
        body.insert("tool_choice".into(), tc);
    }

    body.insert("store".into(), json!(false));
    body.insert("stream".into(), json!(true));
    // Ask cli-chat-proxy for `encrypted_content` on reasoning items so the next turn can
    // replay it. Live probe: without `include`, items only carry `summary`; with it, done
    // items gain `encrypted_content`. xAI docs list omitted prior reasoning as a top cause
    // of cache misses on reasoning models (grok-4.5 bills large `reasoning_tokens`).
    body.insert("include".into(), json!(["reasoning.encrypted_content"]));

    if let Some(max) = anthropic.get("max_tokens").and_then(Value::as_u64) {
        body.insert("max_output_tokens".into(), json!(max));
    }

    // Pin the automatic prefix cache to the Claude Code session. Without this, Grok's
    // cache affinity is account/content-hash only and multi-session concurrency (or any
    // server-side routing change) shows up as a sudden `cached_tokens` collapse.
    if let Some(sid) = session_id {
        body.insert("prompt_cache_key".into(), json!(sid));
    }

    Ok(Value::Object(body))
}

fn append_guidance(instructions: &mut Option<String>, guidance: &str) {
    *instructions = Some(match instructions.take() {
        Some(existing) if !existing.is_empty() => format!("{existing}\n\n{guidance}"),
        _ => guidance.into(),
    });
}

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

fn content_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({ "type": "text", "text": s })],
        Value::Array(arr) => arr.clone(),
        _ => Vec::new(),
    }
}

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

/// Build Responses `input[]` from Anthropic `messages`.
///
/// Assistant `thinking` / `redacted_thinking` blocks are replayed as Responses `reasoning`
/// items (encrypted when we tunnelled Grok's blob out as the thinking signature; otherwise
/// best-effort `summary` from plaintext thinking). Hosted-search history is still dropped —
/// we never re-emit server tool blocks Claude Code cannot round-trip.
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
                        Some("thinking") | Some("redacted_thinking") => {
                            flush_message(&mut input, "assistant", &mut parts);
                            if let Some(item) = thinking_input_item(b) {
                                input.push(item);
                            }
                        }
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
                        Some("server_tool_use")
                        | Some("web_search_tool_result")
                        | Some("x_search_tool_result") => {}
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
                let mut parts: Vec<Value> = Vec::new();
                for b in &blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(t) = b.get("text").and_then(Value::as_str) {
                                parts.push(json!({ "type": "input_text", "text": t }));
                            }
                        }
                        Some("image") => {
                            // Keep a visible placeholder so multimodal context is not silently lost.
                            parts.push(json!({ "type": "input_text", "text": "[image omitted]" }));
                        }
                        Some("tool_result") => {
                            flush_message(&mut input, "user", &mut parts);
                            let call_id =
                                b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": tool_result_output(b),
                            }));
                        }
                        Some("web_search_tool_result") | Some("x_search_tool_result") => {}
                        _ => {}
                    }
                }
                flush_message(&mut input, "user", &mut parts);
            }
        }
    }
    input
}

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

/// Sentinel prefixing a thinking-block signature that carries llmtrim's own Grok
/// `encrypted_content`. Same role as Codex's marker: Claude Code stores the blob as the thinking
/// `signature`, and the marker lets the next turn tell our blobs apart from foreign Anthropic
/// signatures left in history when `/sub grok` is switched on mid-conversation. Replaying a
/// foreign blob as `encrypted_content` makes cli-chat-proxy 400 ("Could not decrypt…").
const GROK_SIG_MARK: &str = "llmtrim-grok-v1:";

fn mark_grok_signature(encrypted: &str) -> String {
    format!("{GROK_SIG_MARK}{encrypted}")
}

fn unmark_grok_signature(signature: &str) -> Option<&str> {
    signature.strip_prefix(GROK_SIG_MARK)
}

/// Build a Grok Responses `reasoning` input item.
///
/// Prefer `encrypted_content` when present (live probe: replaying it keeps subsequent-turn
/// `reasoning_tokens` lower; foreign blobs 400). Plaintext `summary` alone is accepted but is
/// only a best-effort stand-in for history that never received a Grok-issued signature
/// (e.g. Anthropic thinking carried over when `/sub` is enabled mid-session).
fn reasoning_item(encrypted: Option<&str>, summary_text: &str) -> Option<Value> {
    let summary = if summary_text.is_empty() {
        json!([])
    } else {
        json!([{"type": "summary_text", "text": summary_text}])
    };
    match encrypted {
        Some(enc) if !enc.is_empty() => Some(json!({
            "type": "reasoning",
            "encrypted_content": enc,
            "summary": summary,
        })),
        _ if !summary_text.is_empty() => Some(json!({
            "type": "reasoning",
            "summary": summary,
        })),
        _ => None,
    }
}

/// Map an Anthropic thinking / redacted_thinking block into a Grok `reasoning` input item.
fn thinking_input_item(block: &Value) -> Option<Value> {
    let thinking = block
        .get("thinking")
        .and_then(Value::as_str)
        .or_else(|| block.get("data").and_then(Value::as_str))
        .unwrap_or_default();
    let signature = block
        .get("signature")
        .and_then(Value::as_str)
        .unwrap_or_default();
    // Only treat marked signatures as Grok encrypted_content. An empty/foreign signature falls
    // through to summary-only replay from plaintext thinking when available.
    let encrypted = unmark_grok_signature(signature);
    reasoning_item(encrypted, thinking)
}

fn build_tools(anthropic: &Value) -> Vec<Value> {
    let Some(tools) = anthropic.get("tools").and_then(Value::as_array) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for t in tools {
        let name = t.get("name").and_then(Value::as_str).unwrap_or("");
        let ty = t.get("type").and_then(Value::as_str).unwrap_or("");
        if name == "WebSearch"
            || name == WEB_SEARCH_TOOL
            || ty == WEB_SEARCH_TOOL
            || name.eq_ignore_ascii_case("web_search")
        {
            out.push(json!({ "type": "web_search" }));
            continue;
        }
        if name == "XSearch" || name.eq_ignore_ascii_case("x_search") {
            out.push(json!({ "type": "x_search" }));
            continue;
        }
        // Skip pure hosted-type entries already handled.
        if ty == "web_search" || ty == "x_search" {
            out.push(json!({ "type": ty }));
            continue;
        }
        if name.is_empty() {
            continue;
        }
        let mut obj = Map::new();
        obj.insert("type".into(), json!("function"));
        obj.insert("name".into(), json!(name));
        if let Some(desc) = t.get("description").and_then(Value::as_str) {
            obj.insert("description".into(), json!(desc));
        }
        obj.insert(
            "parameters".into(),
            t.get("input_schema").cloned().unwrap_or(json!({})),
        );
        out.push(Value::Object(obj));
    }
    out
}

fn build_tool_choice(tc: Option<&Value>) -> Option<Value> {
    let tc = tc?;
    match tc.get("type").and_then(Value::as_str) {
        Some("auto") | None => None,
        Some("none") => Some(json!("none")),
        Some("any") | Some("required") => Some(json!("required")),
        Some("tool") => {
            let name = tc.get("name").and_then(Value::as_str)?;
            if name == WEB_SEARCH_TOOL || name == "WebSearch" {
                return None;
            }
            Some(json!({ "type": "function", "name": name }))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Response reducer (Responses SSE → ReduceEvent)
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Open {
    None,
    Thinking,
    Text,
    /// Currently streaming Anthropic tool block for this call_id (blocks do not interleave).
    Tool,
}

struct ToolCall {
    name: String,
    buf: String,
    /// ToolStart already emitted on the Anthropic stream.
    started: bool,
    flushed: bool,
    stopped: bool,
}

/// Stateful reducer: Grok Responses SSE → shared [`ReduceEvent`] stream.
///
/// Tool calls are keyed by `call_id` (with `item_id` → `call_id` fallback) so interleaved
/// argument deltas on the wire buffer correctly. Anthropic SSE still requires non-interleaved
/// blocks, so each tool is emitted as ToolStart/Delta/Stop only when that call completes or
/// when a different block (text/thinking) must open.
///
/// Reasoning items with `encrypted_content` (when requested via `include`) are tunnelled out as
/// Anthropic thinking `signature_delta` so the next turn can rebuild a Grok `reasoning` input
/// item. Summary/text deltas still stream as thinking text.
pub struct Reducer {
    /// Resolved upstream model id (for upstream-usage capture metadata only).
    model: String,
    parser: SseLineParser,
    open: Open,
    /// call_id of the Anthropic tool block currently open (if `open == Tool`).
    open_tool: Option<String>,
    saw_tool_use: bool,
    tools: std::collections::HashMap<String, ToolCall>,
    item_to_call: std::collections::HashMap<String, String>,
    /// Stable emission order for tools registered this turn.
    tool_order: Vec<String>,
    terminal_seen: bool,
    /// Grok `encrypted_content` for the open (or most recent) reasoning item, mapped to Anthropic
    /// `signature_delta` before the thinking block closes.
    thinking_encrypted: Option<String>,
    /// Last encrypted blob emitted as a signature, so the repeated `added`/`done` copies of one
    /// reasoning item dedupe while a later item still gets its own signature.
    last_signature: Option<String>,
}

impl Reducer {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            parser: SseLineParser::new(),
            open: Open::None,
            open_tool: None,
            saw_tool_use: false,
            tools: std::collections::HashMap::new(),
            item_to_call: std::collections::HashMap::new(),
            tool_order: Vec::new(),
            terminal_seen: false,
            thinking_encrypted: None,
            last_signature: None,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<ReduceEvent> {
        let mut out = Vec::new();
        for v in self.parser.push(chunk) {
            self.handle(&v, &mut out);
        }
        out
    }

    pub fn finish(&mut self) -> Vec<ReduceEvent> {
        let mut out = Vec::new();
        self.close_open(&mut out);
        self.emit_remaining_tools(&mut out);
        if !self.terminal_seen {
            self.terminal_seen = true;
            out.push(ReduceEvent::Finish {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
                response_id: None,
                continuation_eligible: false,
            });
        }
        out
    }

    fn close_open(&mut self, out: &mut Vec<ReduceEvent>) {
        match self.open {
            Open::Thinking => {
                self.close_thinking(out);
            }
            Open::Text => {
                out.push(ReduceEvent::TextStop);
                self.open = Open::None;
            }
            Open::Tool => {
                if let Some(id) = self.open_tool.clone() {
                    self.emit_tool_complete(&id, out);
                }
                self.open = Open::None;
                self.open_tool = None;
            }
            Open::None => {}
        }
    }

    /// Emit `ThinkingSignatureDelta` when encrypted reasoning content is known, then close the
    /// thinking block. Signature must precede `content_block_stop` or Claude Code rejects the turn.
    fn close_thinking(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.open != Open::Thinking {
            return;
        }
        if let Some(sig) = self.thinking_encrypted.take()
            && self.last_signature.as_deref() != Some(sig.as_str())
        {
            out.push(ReduceEvent::ThinkingSignatureDelta(mark_grok_signature(
                &sig,
            )));
            self.last_signature = Some(sig);
        }
        out.push(ReduceEvent::ThinkingStop);
        self.open = Open::None;
    }

    /// Record encrypted reasoning content from a Responses `reasoning` item. Opens a thinking
    /// block when the upstream omitted summary text (signature-only shape).
    ///
    /// Grok (like Codex) can repeat the same `encrypted_content` on an item's `added` and `done`
    /// events; a turn with tool calls may emit several distinct reasoning items.
    fn note_reasoning_encrypted(
        &mut self,
        out: &mut Vec<ReduceEvent>,
        encrypted: String,
        done: bool,
    ) {
        if self.last_signature.as_deref() == Some(encrypted.as_str()) {
            return;
        }
        self.thinking_encrypted = Some(encrypted);
        // On `added` the summary deltas are still to come: stash the signature and let the
        // summary open the block.
        if !done {
            return;
        }
        if self.open != Open::Thinking {
            // Don't interrupt an open tool for a late reasoning signature-only item.
            if self.open == Open::Tool {
                return;
            }
            self.close_open(out);
            out.push(ReduceEvent::ThinkingStart);
            self.open = Open::Thinking;
        }
        self.close_thinking(out);
    }

    fn emit_tool_complete(&mut self, call_id: &str, out: &mut Vec<ReduceEvent>) {
        let Some(tool) = self.tools.get_mut(call_id) else {
            return;
        };
        if tool.stopped {
            return;
        }
        if !tool.started {
            out.push(ReduceEvent::ToolStart {
                id: call_id.to_string(),
                name: tool.name.clone(),
            });
            tool.started = true;
        }
        if !tool.flushed {
            let sanitized = crate::reroute::read_rewrite::sanitize_read_args(
                &tool.name,
                &tool.buf,
                Some(call_id),
            );
            if !sanitized.is_empty() {
                out.push(ReduceEvent::ToolDelta(sanitized));
            } else if !tool.buf.is_empty() {
                out.push(ReduceEvent::ToolDelta(tool.buf.clone()));
            }
            tool.flushed = true;
        }
        out.push(ReduceEvent::ToolStop);
        tool.stopped = true;
    }

    /// Emit any tools that received args/registration but were never closed (stream end).
    fn emit_remaining_tools(&mut self, out: &mut Vec<ReduceEvent>) {
        let order = self.tool_order.clone();
        for id in order {
            if self.tools.get(&id).is_some_and(|t| {
                !t.stopped && (t.started || !t.buf.is_empty() || !t.name.is_empty())
            }) {
                // Close thinking/text first so nesting stays valid.
                if matches!(self.open, Open::Thinking | Open::Text) {
                    self.close_open(out);
                }
                if self.open == Open::Tool && self.open_tool.as_deref() != Some(id.as_str()) {
                    self.close_open(out);
                }
                self.open = Open::Tool;
                self.open_tool = Some(id.clone());
                self.emit_tool_complete(&id, out);
                self.open = Open::None;
                self.open_tool = None;
            }
        }
    }

    fn resolve_call_id(&self, v: &Value) -> Option<String> {
        if let Some(id) = v
            .get("call_id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            return Some(id.to_string());
        }
        v.get("item_id")
            .and_then(Value::as_str)
            .and_then(|item| self.item_to_call.get(item).cloned())
    }

    fn ensure_tool(&mut self, call_id: &str, name: &str) {
        if !self.tools.contains_key(call_id) {
            self.tool_order.push(call_id.to_string());
            self.tools.insert(
                call_id.to_string(),
                ToolCall {
                    name: name.to_string(),
                    buf: String::new(),
                    started: false,
                    flushed: false,
                    stopped: false,
                },
            );
        } else if !name.is_empty()
            && let Some(t) = self.tools.get_mut(call_id)
            && t.name.is_empty()
        {
            t.name = name.to_string();
        }
    }

    fn open_thinking(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.open == Open::Thinking {
            return;
        }
        self.close_open(out);
        out.push(ReduceEvent::ThinkingStart);
        self.open = Open::Thinking;
    }

    fn open_text(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.open == Open::Text {
            return;
        }
        self.close_open(out);
        out.push(ReduceEvent::TextStart);
        self.open = Open::Text;
    }

    /// Begin Anthropic streaming for `call_id` if nothing else is open (or after closing it).
    fn open_tool_stream(&mut self, call_id: &str, out: &mut Vec<ReduceEvent>) {
        if self.open == Open::Tool && self.open_tool.as_deref() == Some(call_id) {
            return;
        }
        self.close_open(out);
        let Some(tool) = self.tools.get_mut(call_id) else {
            return;
        };
        if tool.stopped {
            return;
        }
        if !tool.started {
            out.push(ReduceEvent::ToolStart {
                id: call_id.to_string(),
                name: tool.name.clone(),
            });
            tool.started = true;
        }
        self.open = Open::Tool;
        self.open_tool = Some(call_id.to_string());
    }

    fn handle(&mut self, v: &Value, out: &mut Vec<ReduceEvent>) {
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "response.output_item.added" => {
                let item = v.get("item").cloned().unwrap_or(Value::Null);
                match item.get("type").and_then(Value::as_str) {
                    Some("message") => {
                        self.open_text(out);
                    }
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        if call_id.is_empty() {
                            return;
                        }
                        let name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        self.saw_tool_use = true;
                        self.ensure_tool(&call_id, &name);
                        if let Some(item_id) = item.get("id").and_then(Value::as_str) {
                            self.item_to_call
                                .insert(item_id.to_string(), call_id.clone());
                        }
                        // Do not close an already-open tool for a different call_id here —
                        // argument deltas may still arrive for the first call. Start streaming
                        // only when no other tool is open.
                        if self.open != Open::Tool {
                            if matches!(self.open, Open::Thinking | Open::Text) {
                                self.close_open(out);
                            }
                            self.open_tool_stream(&call_id, out);
                        }
                    }
                    Some("reasoning") => {
                        if let Some(enc) = item
                            .get("encrypted_content")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            self.note_reasoning_encrypted(out, enc.to_string(), false);
                        }
                    }
                    // Hosted search / custom tools — no Claude function block (Grok runs them).
                    // Must not close an open client function tool when these complete later.
                    Some("custom_tool_call") | Some("web_search_call") => {}
                    _ => {}
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if delta.is_empty() {
                    return;
                }
                self.open_thinking(out);
                out.push(ReduceEvent::ThinkingDelta(delta.to_string()));
            }
            "response.output_text.delta" => {
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if delta.is_empty() {
                    return;
                }
                self.open_text(out);
                out.push(ReduceEvent::TextDelta(delta.to_string()));
            }
            "response.function_call_arguments.delta" => {
                let Some(call_id) = self.resolve_call_id(v) else {
                    return;
                };
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                self.ensure_tool(&call_id, "");
                if let Some(tool) = self.tools.get_mut(&call_id) {
                    tool.buf.push_str(delta);
                }
                // Stream deltas live only for the currently open Anthropic tool block.
                if self.open == Open::Tool && self.open_tool.as_deref() == Some(call_id.as_str()) {
                    // Keep buffering; flush happens on done so sanitize_read_args sees full JSON.
                } else if self.open != Open::Tool
                    && !matches!(self.open, Open::Thinking | Open::Text)
                {
                    self.open_tool_stream(&call_id, out);
                }
            }
            "response.function_call_arguments.done" => {
                let Some(call_id) = self.resolve_call_id(v) else {
                    return;
                };
                self.ensure_tool(&call_id, "");
                if let Some(tool) = self.tools.get_mut(&call_id)
                    && tool.buf.is_empty()
                    && let Some(args) = v.get("arguments").and_then(Value::as_str)
                {
                    tool.buf.push_str(args);
                }
                // Prefer completing this call if it is the open one; otherwise leave buffered
                // until its output_item.done or stream end.
                if self.open_tool.as_deref() == Some(call_id.as_str()) || self.open != Open::Tool {
                    self.open_tool_stream(&call_id, out);
                    self.emit_tool_complete(&call_id, out);
                    self.open = Open::None;
                    self.open_tool = None;
                }
            }
            "response.output_item.done" => {
                let item = v.get("item").cloned().unwrap_or(Value::Null);
                match item.get("type").and_then(Value::as_str) {
                    Some("function_call") => {
                        let call_id = item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                            .or_else(|| {
                                item.get("id")
                                    .and_then(Value::as_str)
                                    .and_then(|id| self.item_to_call.get(id).cloned())
                            });
                        let Some(call_id) = call_id else {
                            return;
                        };
                        self.ensure_tool(&call_id, "");
                        self.open_tool_stream(&call_id, out);
                        self.emit_tool_complete(&call_id, out);
                        self.open = Open::None;
                        self.open_tool = None;
                    }
                    Some("message") => {
                        if self.open == Open::Text {
                            self.close_open(out);
                        }
                    }
                    Some("reasoning") => {
                        // Prefer closing an open thinking block with the encrypted signature
                        // when available. Do not close an open function tool for hosted/reasoning
                        // completion (same rule as custom_tool_call / web_search_call).
                        if let Some(enc) = item
                            .get("encrypted_content")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                        {
                            self.note_reasoning_encrypted(out, enc.to_string(), true);
                        } else if self.open == Open::Thinking {
                            self.close_thinking(out);
                        }
                    }
                    // Hosted done must not close an open function tool.
                    Some("custom_tool_call") | Some("web_search_call") | None => {}
                    _ => {}
                }
            }
            "response.completed" | "response.done" => {
                self.finish_terminal(v, false, out);
            }
            "response.incomplete" => {
                self.finish_terminal(v, true, out);
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
        self.emit_remaining_tools(out);
        let stop_reason = if incomplete {
            StopReason::MaxTokens
        } else if self.saw_tool_use {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };
        let raw_usage = v
            .get("response")
            .and_then(|r| r.get("usage"))
            .or_else(|| v.get("usage"));
        // Opt-in: when `LLMTRIM_CAPTURE_DIR` is set, keep the *raw* upstream usage object
        // (pre-mapping) so a cache-collapse investigation can compare Grok's
        // `input_tokens_details.cached_tokens` against the ledger without guessing the
        // schema. Best-effort; capture must never break streaming.
        if let Some(raw) = raw_usage {
            capture_upstream_usage(raw, &self.model);
        }
        let usage = raw_usage.map(map_usage).unwrap_or_default();
        self.terminal_seen = true;
        out.push(ReduceEvent::Finish {
            stop_reason,
            usage,
            response_id: v
                .get("response")
                .and_then(|r| r.get("id"))
                .and_then(|i| i.as_str())
                .map(|s| s.to_string())
                .or_else(|| v.get("id").and_then(|i| i.as_str()).map(|s| s.to_string())),
            continuation_eligible: false,
        });
    }
}

/// Write one `*-upstream-usage.json` record under the capture corpus, if configured.
/// Public for tests; production call sites go through the reducer terminal path.
pub fn capture_upstream_usage(raw_usage: &Value, model: &str) {
    let Some(dir) = llmtrim_core::config::RuntimeConfig::get()
        .capture_dir
        .clone()
    else {
        return;
    };
    write_upstream_usage_capture(&dir, raw_usage, model, "grok");
}

/// Env-independent body of [`capture_upstream_usage`] (testable without RuntimeConfig).
fn write_upstream_usage_capture(
    dir: &std::path::Path,
    raw_usage: &Value,
    model: &str,
    provider: &str,
) {
    let mapped = map_usage(raw_usage);
    let record = json!({
        "ts": chrono::Utc::now().to_rfc3339(),
        "kind": "upstream_usage",
        "provider": provider,
        "model": model,
        "usage": raw_usage,
        // Pre-computed mapping so a cold-turn audit can compare without re-running map_usage.
        "mapped": {
            "input": mapped.input,
            "output": mapped.output,
            "cache_read": mapped.cache_read,
            "cache_write": mapped.cache_write,
        },
    });
    let name = format!(
        "{}-{:x}-upstream-usage.json",
        chrono::Utc::now().timestamp_micros(),
        std::process::id()
    );
    let path = dir.join(name);
    if let Err(e) =
        std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&path, record.to_string()))
    {
        eprintln!(
            "llmtrim: upstream usage capture failed ({}): {e}",
            path.display()
        );
    }
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

    #[test]
    fn system_becomes_instructions() {
        let body = build_request_body(
            &json!({ "system": "Be concise.", "messages": [] }),
            "grok-4.5",
            None,
        )
        .expect("build");
        assert!(
            body["instructions"]
                .as_str()
                .unwrap()
                .starts_with("Be concise.")
        );
        assert_eq!(body["model"], "grok-4.5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["store"], false);
        assert_eq!(
            body["include"],
            json!(["reasoning.encrypted_content"]),
            "always request encrypted reasoning for multi-turn replay"
        );
    }

    #[test]
    fn thinking_history_replays_as_reasoning_with_encrypted_signature() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":[
                        {
                            "type":"thinking",
                            "thinking":"consider options",
                            "signature": mark_grok_signature("enc-blob-1")
                        },
                        {"type":"text","text":"hello"}
                    ]},
                    {"role":"user","content":"again"}
                ]
            }),
            "grok-4.5",
            Some("sess-think"),
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        let reasoning = input
            .iter()
            .find(|i| i["type"] == "reasoning")
            .expect("reasoning item from thinking history");
        assert_eq!(reasoning["encrypted_content"], "enc-blob-1");
        assert_eq!(
            reasoning["summary"],
            json!([{"type":"summary_text","text":"consider options"}])
        );
        // Order: user msg, reasoning, assistant text, next user.
        assert_eq!(input[0]["type"], "message");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[1]["type"], "reasoning");
        assert_eq!(input[2]["type"], "message");
        assert_eq!(input[2]["role"], "assistant");
        assert_eq!(input[3]["type"], "message");
        assert_eq!(input[3]["role"], "user");
        assert_eq!(body["prompt_cache_key"], "sess-think");
    }

    #[test]
    fn plaintext_thinking_without_grok_signature_replays_as_summary_only() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"anthropic leftover","signature":""},
                        {"type":"text","text":"ok"}
                    ]}
                ]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        let reasoning = input
            .iter()
            .find(|i| i["type"] == "reasoning")
            .expect("summary-only reasoning");
        assert!(reasoning.get("encrypted_content").is_none());
        assert_eq!(
            reasoning["summary"],
            json!([{"type":"summary_text","text":"anthropic leftover"}])
        );
    }

    #[test]
    fn foreign_thinking_signature_is_not_replayed_as_encrypted() {
        // Unmarked signature would 400 cli-chat-proxy; drop it and fall back to summary.
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":[
                        {
                            "type":"thinking",
                            "thinking":"foreign",
                            "signature":"not-a-grok-blob"
                        },
                        {"type":"text","text":"ok"}
                    ]}
                ]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let reasoning = body["input"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["type"] == "reasoning")
            .expect("summary fallback");
        assert!(reasoning.get("encrypted_content").is_none());
        assert_eq!(
            reasoning["summary"],
            json!([{"type":"summary_text","text":"foreign"}])
        );
    }

    #[test]
    fn empty_thinking_without_signature_is_dropped() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"assistant","content":[
                        {"type":"thinking","thinking":"","signature":""},
                        {"type":"text","text":"ok"}
                    ]}
                ]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        assert!(
            input.iter().all(|i| i["type"] != "reasoning"),
            "empty thinking must not inject an empty reasoning item"
        );
        assert!(
            input
                .iter()
                .any(|i| i["type"] == "message" && i["role"] == "assistant")
        );
    }

    #[test]
    fn non_thinking_assistant_unchanged() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":"plain reply"}
                ]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        assert!(input.iter().all(|i| i["type"] != "reasoning"));
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn tools_map_functions_and_web_search() {
        let body = build_request_body(
            &json!({
                "messages": [],
                "tools": [
                    {
                        "name": "Bash",
                        "description": "run",
                        "input_schema": { "type": "object", "properties": {} }
                    },
                    { "name": "WebSearch", "description": "search", "input_schema": {} }
                ]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let tools = body["tools"].as_array().unwrap();
        assert!(
            tools
                .iter()
                .any(|t| t["type"] == "function" && t["name"] == "Bash")
        );
        assert!(tools.iter().any(|t| t["type"] == "web_search"));
        // x_search is not auto-injected unless Claude Code offered XSearch.
        assert!(!tools.iter().any(|t| t["type"] == "x_search"));
    }

    #[test]
    fn assistant_tool_use_and_user_result_roundtrip() {
        let body = build_request_body(
            &json!({
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":[
                        {"type":"tool_use","id":"call_1","name":"Bash","input":{"command":"ls"}}
                    ]},
                    {"role":"user","content":[
                        {"type":"tool_result","tool_use_id":"call_1","content":"ok"}
                    ]}
                ]
            }),
            "grok-composer-2.5-fast",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        assert!(
            input
                .iter()
                .any(|i| i["type"] == "function_call" && i["call_id"] == "call_1")
        );
        assert!(
            input
                .iter()
                .any(|i| i["type"] == "function_call_output" && i["output"] == "ok")
        );
    }

    #[test]
    fn headers_carry_grok_identity() {
        let h = request_headers("tok", None, None);
        assert!(
            h.iter()
                .any(|(k, v)| k == "authorization" && v == "Bearer tok")
        );
        assert!(
            h.iter()
                .any(|(k, v)| k == "x-xai-token-auth" && v == "xai-grok-cli")
        );
        assert!(
            h.iter()
                .any(|(k, v)| k == "x-grok-client-identifier" && v == "grok-shell")
        );
        assert!(h.iter().all(|(k, _)| k != "x-grok-conv-id"));
    }

    #[test]
    fn headers_set_conv_id_from_session() {
        let h = request_headers("tok", None, Some("sess-42"));
        assert!(
            h.iter()
                .any(|(k, v)| k == "x-grok-conv-id" && v == "sess-42")
        );
    }

    #[test]
    fn prompt_cache_key_from_session_id() {
        let body = build_request_body(
            &json!({ "messages": [{"role": "user", "content": "hi"}] }),
            "grok-4.5",
            Some("sess-1"),
        )
        .expect("build");
        assert_eq!(body["prompt_cache_key"], "sess-1");
    }

    #[test]
    fn prompt_cache_key_omitted_without_session() {
        let body = build_request_body(
            &json!({ "messages": [{"role": "user", "content": "hi"}] }),
            "grok-4.5",
            None,
        )
        .expect("build");
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn map_usage_reads_input_tokens_details_cached_tokens() {
        let u = map_usage(&json!({
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 40},
            "output_tokens": 7,
        }));
        assert_eq!(
            u,
            Usage {
                input: 60,
                output: 7,
                cache_read: 40,
                cache_write: 0,
            }
        );
    }

    #[test]
    fn map_usage_zero_cache_on_miss() {
        let u = map_usage(&json!({
            "input_tokens": 50,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 2},
            "context_details": {"input_tokens": 50, "output_tokens": 3},
        }));
        assert_eq!(u.cache_read, 0);
        assert_eq!(u.input, 50);
        assert_eq!(u.output, 3);
    }

    #[test]
    fn write_upstream_usage_capture_records_raw_and_mapped() {
        let dir = std::env::temp_dir().join(format!(
            "llmtrim-grok-usage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("mkdir");
        let raw = json!({
            "input_tokens": 218,
            "input_tokens_details": {"cached_tokens": 128},
            "output_tokens": 41,
            "output_tokens_details": {"reasoning_tokens": 40},
        });
        write_upstream_usage_capture(&dir, &raw, "grok-4.5", "grok");
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("-upstream-usage.json"))
            })
            .collect();
        assert_eq!(files.len(), 1, "one capture file");
        let rec: Value =
            serde_json::from_str(&std::fs::read_to_string(&files[0]).unwrap()).unwrap();
        assert_eq!(rec["kind"], "upstream_usage");
        assert_eq!(rec["provider"], "grok");
        assert_eq!(rec["model"], "grok-4.5");
        assert_eq!(rec["usage"]["input_tokens_details"]["cached_tokens"], 128);
        assert_eq!(rec["mapped"]["cache_read"], 128);
        assert_eq!(rec["mapped"]["input"], 90); // 218 - 128
        assert_eq!(rec["mapped"]["output"], 41);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reducer_finish_maps_cached_tokens_into_usage() {
        let mut r = Reducer::new("grok-4.5");
        let chunk = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":100,\"input_tokens_details\":{\"cached_tokens\":40},\"output_tokens\":5}}}\n\n",
        );
        let events: Vec<_> = r
            .push(chunk.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        let finish = events.iter().find_map(|e| match e {
            ReduceEvent::Finish { usage, .. } => Some(*usage),
            _ => None,
        });
        assert_eq!(
            finish,
            Some(Usage {
                input: 60,
                output: 5,
                cache_read: 40,
                cache_write: 0,
            })
        );
    }

    #[test]
    fn reducer_streams_text_and_tools() {
        let mut r = Reducer::new("grok-4.5");
        let chunk = concat!(
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"Bash\"},\"output_index\":1}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"c1\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"c1\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"output_index\":1,\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":3}}}\n\n",
        );
        let events = r.push(chunk.as_bytes());
        let finish = r.finish();
        let all: Vec<_> = events.into_iter().chain(finish).collect();
        assert!(all.iter().any(|e| matches!(e, ReduceEvent::TextStart)));
        assert!(
            all.iter()
                .any(|e| matches!(e, ReduceEvent::TextDelta(s) if s == "hi"))
        );
        assert!(all.iter().any(
            |e| matches!(e, ReduceEvent::ToolStart { id, name } if id == "c1" && name == "Bash")
        ));
        assert!(
            all.iter()
                .any(|e| matches!(e, ReduceEvent::Finish { stop_reason, .. } if *stop_reason == StopReason::ToolUse))
        );
    }

    #[test]
    fn reducer_maps_reasoning_to_thinking() {
        let mut r = Reducer::new("grok-4.5");
        let chunk = concat!(
            "data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"hmm\"}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        );
        let events = r.push(chunk.as_bytes());
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::ThinkingDelta(s) if s == "hmm"))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::TextDelta(s) if s == "ok"))
        );
    }

    #[test]
    fn reducer_tunnels_encrypted_reasoning_as_thinking_signature() {
        let mut r = Reducer::new("grok-4.5");
        let chunk = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[],\"status\":\"in_progress\"}}\n\n",
            "data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"ponder\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"id\":\"rs_1\",\"status\":\"completed\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"ponder\"}],\"encrypted_content\":\"encXYZ\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"answer\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        );
        let events: Vec<_> = r
            .push(chunk.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::ThinkingStart))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::ThinkingDelta(s) if s == "ponder"))
        );
        let sig = events.iter().find_map(|e| match e {
            ReduceEvent::ThinkingSignatureDelta(s) => Some(s.as_str()),
            _ => None,
        });
        assert_eq!(sig, Some(mark_grok_signature("encXYZ").as_str()));
        // signature before stop
        let sig_pos = events
            .iter()
            .position(|e| matches!(e, ReduceEvent::ThinkingSignatureDelta(_)))
            .unwrap();
        let stop_pos = events
            .iter()
            .position(|e| matches!(e, ReduceEvent::ThinkingStop))
            .unwrap();
        assert!(sig_pos < stop_pos);
    }

    #[test]
    fn reducer_buffers_interleaved_tool_args_by_call_id() {
        let mut r = Reducer::new("grok-4.5");
        // Two function calls; argument deltas arrive interleaved without output_index.
        let chunk = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\",\"name\":\"Bash\",\"id\":\"item_1\"}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c2\",\"name\":\"Read\",\"id\":\"item_2\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"c2\",\"delta\":\"{\\\"file\\\"\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"c1\",\"delta\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.delta\",\"call_id\":\"c2\",\"delta\":\":\\\"a.rs\\\"}\"}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"c1\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"}\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c1\"}}\n\n",
            "data: {\"type\":\"response.function_call_arguments.done\",\"call_id\":\"c2\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"c2\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"web_search_call\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":2}}}\n\n",
        );
        let events: Vec<_> = r
            .push(chunk.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        let starts: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                ReduceEvent::ToolStart { id, name } => Some((id.as_str(), name.as_str())),
                _ => None,
            })
            .collect();
        assert!(starts.contains(&("c1", "Bash")), "starts={starts:?}");
        assert!(starts.contains(&("c2", "Read")), "starts={starts:?}");
        assert!(
            events.iter().any(
                |e| matches!(e, ReduceEvent::ToolDelta(s) if s.contains("command") && s.contains("ls"))
            ),
            "c1 args present: {events:?}"
        );
        assert!(
            events.iter().any(
                |e| matches!(e, ReduceEvent::ToolDelta(s) if s.contains("file") && s.contains("a.rs"))
            ),
            "c2 args present: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::Finish { stop_reason, .. } if *stop_reason == StopReason::ToolUse))
        );
    }

    #[test]
    fn user_image_becomes_placeholder() {
        let body = build_request_body(
            &json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "see"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xx"}}
                    ]
                }]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        let input = body["input"].as_array().unwrap();
        let text = serde_json::to_string(input).unwrap();
        assert!(text.contains("[image omitted]"), "{text}");
    }
}
