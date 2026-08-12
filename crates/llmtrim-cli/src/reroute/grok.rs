//! Grok (xAI SuperGrok / Grok Build) subscription reroute provider.
//!
//! Translates an Anthropic `/v1/messages` request into the Grok CLI Responses API shape
//! (`POST https://cli-chat-proxy.grok.com/v1/responses`) and reduces the Responses SSE stream
//! back into the shared [`ReduceEvent`] stream that
//! [`crate::reroute::sse::AnthropicSseEncoder`] re-encodes as Anthropic SSE.
//!
//! Wire models: `grok-4.6` (Fable), `grok-4.5` (Opus/Sonnet flagship), and
//! `grok-composer-2.5-fast` (cheap/fast). Auth is OAuth against `auth.x.ai` (see
//! [`crate::reroute::auth`]).

use anyhow::Result;
use serde_json::{Map, Value, json};

use crate::reroute::sse::{
    ReduceEvent, SERVER_TOOL_ID_PREFIX, SseLineParser, StopReason, Usage, WEB_SEARCH_RESULT_TOOL,
    X_SEARCH_RESULT_TOOL,
};

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
    // Only advertise hosted tools when Claude Code offered them. Hosted search streams are
    // reduced back into Anthropic server_tool blocks, but x_search is still opt-in (only when
    // Claude offered XSearch) so we do not invent X tools the client never registered.
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

/// Tool-result images: base64 becomes a data URL; remote URLs stay textual placeholders.
fn tool_result_image_url(block: &Value) -> Result<String, String> {
    let Some(source) = block.get("source") else {
        return Err("[image omitted]".into());
    };
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media = source.get("media_type").and_then(Value::as_str);
            let data = source.get("data").and_then(Value::as_str);
            match (media, data) {
                (Some(media), Some(data)) if !data.is_empty() => {
                    Ok(format!("data:{media};base64,{data}"))
                }
                _ => Err("[image omitted]".into()),
            }
        }
        Some("url") if source.get("url").and_then(Value::as_str).is_some() => {
            Err("[image omitted: url]".into())
        }
        _ => Err("[image omitted]".into()),
    }
}

/// Render a `tool_result` into Responses `function_call_output.output`.
/// Pure text stays a string; base64 images become a content-parts array.
fn tool_result_output(block: &Value) -> Value {
    let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);

    match block.get("content") {
        Some(Value::String(s)) => {
            let body = if is_error {
                format!("[tool execution error]\n{s}")
            } else {
                s.clone()
            };
            Value::String(body)
        }
        Some(Value::Array(parts)) => {
            let mut out: Vec<Value> = Vec::new();
            let mut has_image = false;
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("image") => match tool_result_image_url(p) {
                        Ok(url) => {
                            has_image = true;
                            out.push(json!({
                                "type": "input_image",
                                "image_url": url,
                            }));
                        }
                        Err(placeholder) => {
                            out.push(json!({
                                "type": "input_text",
                                "text": placeholder,
                            }));
                        }
                    },
                    _ => {
                        let text = p.get("text").and_then(Value::as_str).unwrap_or_default();
                        out.push(json!({
                            "type": "input_text",
                            "text": text,
                        }));
                    }
                }
            }

            if !has_image {
                let body = out
                    .iter()
                    .filter_map(|p| p.get("text").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join("\n");
                let body = if is_error {
                    format!("[tool execution error]\n{body}")
                } else {
                    body
                };
                return Value::String(body);
            }

            if is_error {
                out.insert(
                    0,
                    json!({ "type": "input_text", "text": "[tool execution error]" }),
                );
            }
            Value::Array(out)
        }
        _ => {
            if is_error {
                Value::String("[tool execution error]\n".into())
            } else {
                Value::String(String::new())
            }
        }
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

/// Hosted search recorded from a Grok `web_search_call` or `custom_tool_call` (`x_search`),
/// emitted as Anthropic `server_tool_use` + `*_tool_result` once citations/text are known.
struct PendingHostedSearch {
    id: String,
    /// Anthropic server tool name: `web_search` or `x_search`.
    name: String,
    query: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HostedSearchResultItem {
    title: String,
    url: String,
}

/// Stateful reducer: Grok Responses SSE → shared [`ReduceEvent`] stream.
///
/// Tool calls are keyed by `call_id` (with `item_id` → `call_id` fallback) so interleaved
/// argument deltas on the wire buffer correctly. Anthropic SSE still requires non-interleaved
/// blocks, so each tool is emitted as ToolStart/Delta/Stop only when that call completes or
/// when a different block (text/thinking) must open.
///
/// Hosted `web_search` / `x_search` complete server-side on Grok; they are buffered and reduced
/// into Anthropic `server_tool_use` + result blocks (same encoding path as Codex) before answer
/// text, so Claude Code sees the search turn.
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
    /// Hosted searches not yet emitted as Anthropic server-tool blocks. Answer text after a
    /// search is deferred until these flush so the result block can include citations scraped
    /// from the answer (and any `url_citation` annotations).
    pending_hosted_searches: Vec<PendingHostedSearch>,
    /// In-flight Grok `custom_tool_call` items (id → (name, input buffer)). Used for x_search.
    hosted_custom_calls: std::collections::HashMap<String, (String, String)>,
    /// Citations from `response.output_text.annotation.added` (`url_citation`).
    search_citations: Vec<HostedSearchResultItem>,
    /// Answer text held while a hosted search is pending (for citation scrape + order).
    deferred_text_deltas: Vec<String>,
    /// Accumulated assistant text this turn (for scraping URLs into search results).
    current_assistant_text: String,
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
            pending_hosted_searches: Vec::new(),
            hosted_custom_calls: std::collections::HashMap::new(),
            search_citations: Vec::new(),
            deferred_text_deltas: Vec::new(),
            current_assistant_text: String::new(),
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
        self.flush_hosted_searches(&mut out);
        self.flush_deferred_text(&mut out);
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

    fn has_unflushed_hosted_search(&self) -> bool {
        !self.pending_hosted_searches.is_empty()
    }

    /// Emit buffered hosted searches as Anthropic server-tool blocks, using collected citations
    /// and any markdown/bare URLs from the (possibly deferred) answer text.
    fn flush_hosted_searches(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.pending_hosted_searches.is_empty() {
            return;
        }
        // Close an open client function / text / thinking block before server-tool blocks.
        if matches!(self.open, Open::Tool | Open::Thinking | Open::Text) {
            self.close_open(out);
        }
        let mut text = self.current_assistant_text.clone();
        for delta in &self.deferred_text_deltas {
            text.push_str(delta);
        }
        let mut results = self.search_citations.clone();
        for scraped in scrape_search_results_from_text(&text) {
            if results.iter().any(|r| r.url == scraped.url) {
                continue;
            }
            results.push(scraped);
        }
        let searches = std::mem::take(&mut self.pending_hosted_searches);
        let result_content: Vec<Value> = results
            .iter()
            .map(|r| {
                json!({
                    "type": "web_search_result",
                    "title": r.title,
                    "url": r.url,
                })
            })
            .collect();
        let result_json =
            serde_json::to_string(&result_content).unwrap_or_else(|_| "[]".to_string());
        for search in searches {
            let input = json!({ "query": search.query });
            let partial = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
            let result_tool = if search.name == "x_search" {
                X_SEARCH_RESULT_TOOL
            } else {
                WEB_SEARCH_RESULT_TOOL
            };
            // Encoder special-cases srvtoolu_* + name web_search/x_search → server_tool_use.
            out.push(ReduceEvent::ToolStart {
                id: search.id.clone(),
                name: search.name,
            });
            out.push(ReduceEvent::ToolDelta(partial));
            out.push(ReduceEvent::ToolStop);
            out.push(ReduceEvent::ToolStart {
                id: search.id,
                name: result_tool.into(),
            });
            out.push(ReduceEvent::ToolDelta(result_json.clone()));
            out.push(ReduceEvent::ToolStop);
        }
    }

    /// Replay text that arrived while hosted search was still pending.
    fn flush_deferred_text(&mut self, out: &mut Vec<ReduceEvent>) {
        if self.deferred_text_deltas.is_empty() {
            return;
        }
        self.flush_hosted_searches(out);
        if self.open != Open::Text {
            out.push(ReduceEvent::TextStart);
            self.open = Open::Text;
        }
        for delta in std::mem::take(&mut self.deferred_text_deltas) {
            self.current_assistant_text.push_str(&delta);
            out.push(ReduceEvent::TextDelta(delta));
        }
    }

    fn note_hosted_search(&mut self, raw_id: &str, name: &str, query: String) {
        if raw_id.is_empty() {
            return;
        }
        let id = server_tool_use_id_from_grok_id(raw_id);
        if let Some(existing) = self.pending_hosted_searches.iter_mut().find(|s| s.id == id) {
            if existing.query.is_empty() && !query.is_empty() {
                existing.query = query;
            }
            if existing.name.is_empty() && !name.is_empty() {
                existing.name = name.to_string();
            }
            return;
        }
        self.pending_hosted_searches.push(PendingHostedSearch {
            id,
            name: name.to_string(),
            query,
        });
    }

    fn note_url_citation(&mut self, v: &Value) {
        let Some(annotation) = v.get("annotation") else {
            return;
        };
        if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
            return;
        }
        let Some(url) = annotation.get("url").and_then(Value::as_str) else {
            return;
        };
        if url.is_empty() || self.search_citations.iter().any(|r| r.url == url) {
            return;
        }
        let title = annotation
            .get("title")
            .and_then(Value::as_str)
            .filter(|t| !t.is_empty())
            .unwrap_or(url)
            .to_string();
        self.search_citations.push(HostedSearchResultItem {
            title,
            url: url.to_string(),
        });
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
        // Hosted search blocks must precede the answer that cites them.
        self.flush_hosted_searches(out);
        self.close_open(out);
        out.push(ReduceEvent::TextStart);
        self.open = Open::Text;
    }

    /// Begin Anthropic streaming for `call_id` if nothing else is open (or after closing it).
    fn open_tool_stream(&mut self, call_id: &str, out: &mut Vec<ReduceEvent>) {
        if self.open == Open::Tool && self.open_tool.as_deref() == Some(call_id) {
            return;
        }
        self.flush_hosted_searches(out);
        self.flush_deferred_text(out);
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
                        if self.has_unflushed_hosted_search() {
                            // Answer text is held until hosted search blocks flush.
                        } else {
                            self.open_text(out);
                        }
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
                    // Hosted search / custom tools — Grok runs them; reduce to server_tool blocks
                    // on `done`. Must not close an open client function tool when these appear.
                    Some("web_search_call") => {
                        if let Some(query) = web_search_query(&item) {
                            self.note_hosted_search(
                                item.get("id").and_then(Value::as_str).unwrap_or(""),
                                "web_search",
                                query,
                            );
                        }
                    }
                    Some("custom_tool_call") => {
                        let id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .unwrap_or("");
                        if id.is_empty() {
                            return;
                        }
                        let raw_name = item
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .unwrap_or("x_search");
                        // Grok may emit x_keyword_search / x_semantic_search; normalize to x_search.
                        let name = if raw_name.starts_with("x_") {
                            "x_search"
                        } else {
                            raw_name
                        };
                        self.hosted_custom_calls
                            .insert(id.to_string(), (name.to_string(), String::new()));
                    }
                    _ => {}
                }
            }
            "response.custom_tool_call_input.delta" => {
                let Some(id) = v.get("item_id").and_then(Value::as_str) else {
                    return;
                };
                let delta = v.get("delta").and_then(Value::as_str).unwrap_or("");
                if let Some((_, input)) = self.hosted_custom_calls.get_mut(id) {
                    input.push_str(delta);
                }
            }
            "response.custom_tool_call_input.done"
            | "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed" => {}
            "response.output_text.annotation.added" => {
                self.note_url_citation(v);
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
                // Hold answer text until hosted searches flush so result blocks can include
                // citations scraped from the answer (and any url_citation annotations).
                if self.has_unflushed_hosted_search() {
                    self.deferred_text_deltas.push(delta.to_string());
                    return;
                }
                self.open_text(out);
                self.current_assistant_text.push_str(delta);
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
                    Some("web_search_call") => {
                        // Hosted done must not close an open function tool — only record the
                        // search; flush happens before text / finish / client tools.
                        let id = item.get("id").and_then(Value::as_str).unwrap_or("");
                        let query = web_search_query(&item).unwrap_or_default();
                        self.note_hosted_search(id, "web_search", query);
                    }
                    Some("custom_tool_call") => {
                        let id = item
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .unwrap_or("");
                        if id.is_empty() {
                            return;
                        }
                        let (name, input) =
                            self.hosted_custom_calls.remove(id).unwrap_or_else(|| {
                                let raw = item
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or("x_search");
                                let name = if raw.starts_with("x_") {
                                    "x_search".to_string()
                                } else {
                                    raw.to_string()
                                };
                                (name, String::new())
                            });
                        // Only x_search is reduced to Anthropic server tools (CCP parity).
                        if name != "x_search" {
                            return;
                        }
                        let query = serde_json::from_str::<Value>(&input)
                            .ok()
                            .and_then(|v| {
                                v.get("query").and_then(Value::as_str).map(str::to_string)
                            })
                            .unwrap_or_default();
                        self.note_hosted_search(id, "x_search", query);
                    }
                    None => {}
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
        // Emit hosted search blocks before any deferred answer text so Claude Code sees the
        // server_tool_use / *_tool_result pair ahead of the prose.
        self.flush_hosted_searches(out);
        self.flush_deferred_text(out);
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

/// Anthropic server-tool ids must look like `srvtoolu_*`.
fn server_tool_use_id_from_grok_id(id: &str) -> String {
    let suffix: String = id
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{SERVER_TOOL_ID_PREFIX}{suffix}")
}

fn web_search_query(item: &Value) -> Option<String> {
    let action = item.get("action")?;
    if let Some(q) = action.get("query").and_then(Value::as_str) {
        return Some(q.to_string());
    }
    if let Some(queries) = action.get("queries").and_then(Value::as_array) {
        for q in queries {
            if let Some(s) = q.as_str() {
                return Some(s.to_string());
            }
        }
    }
    None
}

/// Pull markdown links and bare URLs out of the answer text as a best-effort result list when
/// Grok does not stream structured search hits.
fn scrape_search_results_from_text(text: &str) -> Vec<HostedSearchResultItem> {
    let mut results = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // [title](https://...)
    let mut rest = text;
    while let Some(open) = rest.find('[') {
        let after_open = &rest[open + 1..];
        let Some(close) = after_open.find(']') else {
            break;
        };
        let title = after_open[..close].trim();
        let after_close = &after_open[close + 1..];
        if !after_close.starts_with('(') {
            rest = &after_open[close + 1..];
            continue;
        }
        let after_paren = &after_close[1..];
        let Some(end) = after_paren.find(')') else {
            break;
        };
        let mut url = after_paren[..end].trim().to_string();
        trim_trailing_url_punct(&mut url);
        if (url.starts_with("http://") || url.starts_with("https://")) && seen.insert(url.clone()) {
            let display = if title.is_empty() {
                fallback_title_from_url(&url)
            } else {
                title.to_string()
            };
            results.push(HostedSearchResultItem {
                title: display,
                url,
            });
        }
        rest = &after_paren[end + 1..];
    }

    // Bare URLs not already captured via markdown.
    let mut rest = text;
    while let Some(start) = {
        let http = rest.find("http://");
        let https = rest.find("https://");
        match (http, https) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    } {
        let candidate = &rest[start..];
        let end = candidate
            .find(|c: char| c.is_whitespace() || matches!(c, '<' | '>' | ')' | '|' | '"' | '\''))
            .unwrap_or(candidate.len());
        let mut url = candidate[..end].to_string();
        trim_trailing_url_punct(&mut url);
        if !url.is_empty() && seen.insert(url.clone()) {
            results.push(HostedSearchResultItem {
                title: fallback_title_from_url(&url),
                url,
            });
        }
        rest = &candidate[end.max(1)..];
    }

    results
}

fn trim_trailing_url_punct(url: &mut String) {
    while url
        .chars()
        .last()
        .is_some_and(|c| matches!(c, '.' | ',' | ';' | ':' | '!' | '?' | ')'))
    {
        url.pop();
    }
}

fn fallback_title_from_url(url: &str) -> String {
    url.trim_start_matches("https://")
        .trim_start_matches("http://")
        .split('/')
        .next()
        .unwrap_or(url)
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
        // Hosted web_search_call mid-stream must not invent a client tool start.
        assert!(
            !events.iter().any(|e| matches!(
                e,
                ReduceEvent::ToolStart { name, .. } if name == "web_search"
            )),
            "empty web_search_call without id/query should not emit: {events:?}"
        );
    }

    #[test]
    fn web_search_call_emits_server_tool_blocks_before_text() {
        let mut r = Reducer::new("grok-4.5");
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\"}}\n\n",
            "data: {\"type\":\"response.web_search_call.in_progress\",\"item_id\":\"ws_1\"}\n\n",
            "data: {\"type\":\"response.web_search_call.searching\",\"item_id\":\"ws_1\"}\n\n",
            "data: {\"type\":\"response.web_search_call.completed\",\"item_id\":\"ws_1\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"action\":{\"query\":\"rust news\"}}}\n\n",
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"message\",\"id\":\"msg_1\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"See [Example](https://example.com).\"}\n\n",
            "data: {\"type\":\"response.output_text.annotation.added\",\"annotation\":{\"type\":\"url_citation\",\"url\":\"https://docs.rs/tokio\",\"title\":\"Tokio\"}}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}}\n\n",
        );
        let events: Vec<_> = r
            .push(sse.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        let result_json = json!([
            {
                "type": "web_search_result",
                "title": "Tokio",
                "url": "https://docs.rs/tokio"
            },
            {
                "type": "web_search_result",
                "title": "Example",
                "url": "https://example.com"
            }
        ])
        .to_string();
        assert_eq!(
            events,
            vec![
                ReduceEvent::ToolStart {
                    id: "srvtoolu_ws_1".into(),
                    name: "web_search".into(),
                },
                ReduceEvent::ToolDelta(r#"{"query":"rust news"}"#.into()),
                ReduceEvent::ToolStop,
                ReduceEvent::ToolStart {
                    id: "srvtoolu_ws_1".into(),
                    name: WEB_SEARCH_RESULT_TOOL.into(),
                },
                ReduceEvent::ToolDelta(result_json),
                ReduceEvent::ToolStop,
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("See [Example](https://example.com).".into()),
                ReduceEvent::TextStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage {
                        input: 3,
                        output: 2,
                        cache_read: 0,
                        cache_write: 0,
                    },
                    response_id: None,
                    continuation_eligible: false,
                },
            ]
        );
    }

    #[test]
    fn x_search_custom_tool_call_emits_server_tool_blocks() {
        let mut r = Reducer::new("grok-4.5");
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"custom_tool_call\",\"name\":\"x_search\",\"id\":\"xs_1\"}}\n\n",
            "data: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"xs_1\",\"delta\":\"{\\\"query\\\":\\\"claude-code-proxy\\\"}\"}\n\n",
            "data: {\"type\":\"response.custom_tool_call_input.done\",\"item_id\":\"xs_1\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"custom_tool_call\",\"name\":\"x_search\",\"id\":\"xs_1\"}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Recent post https://x.com/example/status/1\"}\n\n",
            "data: {\"type\":\"response.output_text.annotation.added\",\"annotation\":{\"type\":\"url_citation\",\"url\":\"https://x.com/example/status/1\",\"title\":\"Example post\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":3}}}\n\n",
        );
        let events: Vec<_> = r
            .push(sse.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        assert!(
            events.iter().any(|e| matches!(
                e,
                ReduceEvent::ToolStart { id, name }
                    if id == "srvtoolu_xs_1" && name == "x_search"
            )),
            "x_search server tool: {events:?}"
        );
        assert!(
            events.iter().any(|e| matches!(
                e,
                ReduceEvent::ToolStart { name, .. } if name == X_SEARCH_RESULT_TOOL
            )),
            "x_search result block: {events:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ReduceEvent::ToolDelta(s) if s.contains("claude-code-proxy"))),
            "query in input: {events:?}"
        );
        // Search blocks before answer text.
        let search_pos = events
            .iter()
            .position(|e| matches!(e, ReduceEvent::ToolStart { name, .. } if name == "x_search"))
            .expect("x_search start");
        let text_pos = events
            .iter()
            .position(|e| matches!(e, ReduceEvent::TextStart))
            .expect("text");
        assert!(search_pos < text_pos, "search before text: {events:?}");
    }

    #[test]
    fn x_keyword_search_normalizes_to_x_search() {
        let mut r = Reducer::new("grok-4.5");
        let sse = concat!(
            "data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"custom_tool_call\",\"name\":\"x_keyword_search\",\"id\":\"search_1\"}}\n\n",
            "data: {\"type\":\"response.custom_tool_call_input.delta\",\"item_id\":\"search_1\",\"delta\":\"{\\\"query\\\":\\\"test\\\"}\"}\n\n",
            "data: {\"type\":\"response.custom_tool_call_input.done\",\"item_id\":\"search_1\"}\n\n",
            "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"custom_tool_call\",\"name\":\"x_keyword_search\",\"id\":\"search_1\"}}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{}}}\n\n",
        );
        let events: Vec<_> = r
            .push(sse.as_bytes())
            .into_iter()
            .chain(r.finish())
            .collect();
        assert!(
            events.iter().any(|e| matches!(
                e,
                ReduceEvent::ToolStart { name, .. } if name == "x_search"
            )),
            "x_keyword_search → x_search: {events:?}"
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

    #[test]
    fn tool_result_base64_image_becomes_content_parts() {
        let body = build_request_body(
            &json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_img",
                        "content": [
                            {"type": "text", "text": "shot"},
                            {"type": "image", "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "xx"
                            }}
                        ]
                    }]
                }]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        assert_eq!(
            body["input"][0]["output"],
            json!([
                {"type": "input_text", "text": "shot"},
                {"type": "input_image", "image_url": "data:image/png;base64,xx"}
            ])
        );
    }

    #[test]
    fn tool_result_text_only_stays_string() {
        let body = build_request_body(
            &json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_t",
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": "b"}
                        ]
                    }]
                }]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        assert_eq!(body["input"][0]["output"], "a\nb");
    }

    #[test]
    fn tool_result_error_prefix_with_image() {
        let body = build_request_body(
            &json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "call_e",
                        "is_error": true,
                        "content": [{
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "yy"
                            }
                        }]
                    }]
                }]
            }),
            "grok-4.5",
            None,
        )
        .expect("build");
        assert_eq!(
            body["input"][0]["output"],
            json!([
                {"type": "input_text", "text": "[tool execution error]"},
                {"type": "input_image", "image_url": "data:image/jpeg;base64,yy"}
            ])
        );
    }
}
