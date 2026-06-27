//! Per-source attribution of a request body.
//!
//! Splits an outgoing provider request into the individual content blocks the model
//! reads — system prompt, tool schemas, conversation text, thinking, tool calls and
//! tool results — and tags each with a `(group, label)` source category plus, for MCP
//! tools, the originating server. This is the data behind the cost-breakdown TUI's
//! "where did the tokens / dollars go" view.
//!
//! Attribution is descriptive only: it never mutates the request and is best-effort —
//! a body shape it doesn't recognize yields fewer blocks, never an error.

use serde_json::Value;

use crate::ir::ProviderKind;
use crate::tokenizer::TokenCounter;

/// Which side of the exchange a block belongs to. Only `Input` blocks occupy the
/// context window; `Output` is reserved for response attribution (cost view).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Zone {
    Input,
    Output,
}

impl Zone {
    pub fn as_str(self) -> &'static str {
        match self {
            Zone::Input => "input",
            Zone::Output => "output",
        }
    }
}

/// Whether a block is part of the cacheable static prefix (system prompt + tool
/// schemas, resent verbatim every turn) or the growing message history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Static,
    Messages,
}

impl Section {
    pub fn as_str(self) -> &'static str {
        match self {
            Section::Static => "static",
            Section::Messages => "messages",
        }
    }
}

/// The wire kind of a content block, normalized across providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bucket {
    System,
    Schema,
    Text,
    Thinking,
    ToolUse,
    ToolResult,
}

impl Bucket {
    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::System => "system",
            Bucket::Schema => "schema",
            Bucket::Text => "text",
            Bucket::Thinking => "thinking",
            Bucket::ToolUse => "tool_use",
            Bucket::ToolResult => "tool_result",
        }
    }
}

/// Message author for text blocks (drives the "user text" vs "assistant text" split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }

    fn parse(s: &str) -> Option<Role> {
        match s {
            "user" => Some(Role::User),
            "assistant" | "model" => Some(Role::Assistant),
            _ => None,
        }
    }
}

/// One attributed content block of a request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockAttribution {
    pub zone: Zone,
    pub section: Section,
    pub bucket: Bucket,
    /// MCP server name parsed from a `mcp__<server>__<tool>` tool name, else `None`.
    pub mcp_server: Option<String>,
    /// Full tool name for schema / tool_use / tool_result blocks.
    pub tool_name: Option<String>,
    pub role: Option<Role>,
    /// Ordinal of the owning message in the request's `messages`/`input`/`contents`
    /// array (`None` for static blocks).
    pub msg_index: Option<usize>,
    /// Token count of this block under the request's tokenizer.
    pub tokens: usize,
}

impl BlockAttribution {
    /// The `(group, label)` source category for grouping rows in the breakdown view.
    /// `group` is one of `Static` / `Messages` / `Output`; `label` is the human source
    /// name the TUI groups rows under.
    pub fn category(&self) -> (&'static str, &'static str) {
        match self.zone {
            Zone::Output => {
                let label = match self.bucket {
                    Bucket::ToolUse | Bucket::ToolResult => {
                        if self.mcp_server.is_some() {
                            "MCP tool use+output"
                        } else {
                            "system tool use+output"
                        }
                    }
                    _ => self.bucket.as_str(),
                };
                ("Output", label)
            }
            Zone::Input => match self.section {
                Section::Static => match self.bucket {
                    Bucket::System => ("Static", "System prompt"),
                    Bucket::Schema => {
                        if self.mcp_server.is_some() {
                            ("Static", "MCP tools")
                        } else {
                            ("Static", "System tools")
                        }
                    }
                    _ => ("Static", self.bucket.as_str()),
                },
                Section::Messages => match self.bucket {
                    Bucket::Text => match self.role {
                        Some(Role::User) => ("Messages", "user text"),
                        _ => ("Messages", "assistant text"),
                    },
                    Bucket::Thinking => ("Messages", "thinking"),
                    Bucket::ToolUse | Bucket::ToolResult => {
                        if self.mcp_server.is_some() {
                            ("Messages", "MCP tool use+output")
                        } else {
                            ("Messages", "system tool use+output")
                        }
                    }
                    _ => ("Messages", self.bucket.as_str()),
                },
            },
        }
    }
}

/// Extract the MCP server segment from a `mcp__<server>__<tool>` tool name.
/// Returns `None` for built-in tools (names without the `mcp__` prefix).
pub fn mcp_server_of(tool: &str) -> Option<&str> {
    let rest = tool.strip_prefix("mcp__")?;
    // `mcp__<server>__<tool>` → server is the segment up to the next `__`; a bare
    // `mcp__server` (no tool) still yields the server. A degenerate `mcp__` (empty
    // segment) is not a real server, so reject it rather than emit a blank label.
    let server = rest.split("__").next().unwrap_or(rest);
    (!server.is_empty()).then_some(server)
}

/// Walk a request body into its attributed content blocks (input side only).
///
/// Provider-aware: Anthropic (`system`/`tools`/`messages`), OpenAI chat
/// (`messages`/`tools`) and the Responses API (`instructions`/`input`/`tools`), and
/// Google (`systemInstruction`/`tools`/`contents`). Unknown shapes contribute nothing.
pub fn attribute(
    body: &Value,
    kind: ProviderKind,
    counter: &dyn TokenCounter,
) -> Vec<BlockAttribution> {
    let mut out = Vec::new();
    match kind {
        ProviderKind::Anthropic => attribute_anthropic(body, counter, &mut out),
        ProviderKind::OpenAi => attribute_openai(body, counter, &mut out),
        ProviderKind::Google => attribute_google(body, counter, &mut out),
    }
    // Drop empty blocks (e.g. a `tool_result` with no content) — they bill nothing and would
    // only add zero-token clutter to the breakdown.
    out.retain(|b| b.tokens > 0);
    out
}

/// Count tokens of an arbitrary JSON value: a string is counted as-is, anything else by
/// its compact serialization (matches how the pipeline counts tool schemas and tool-call
/// arguments, which the model reads as structured JSON).
fn count_value(v: &Value, counter: &dyn TokenCounter) -> usize {
    match v {
        Value::String(s) => counter.count(s),
        Value::Null => 0,
        other => counter.count(&other.to_string()),
    }
}

/// Count tokens of a *text-bearing* value — a plain string, an array of content blocks, or
/// an object carrying a `text` field. Unlike [`count_value`] this extracts the natural
/// language and skips JSON punctuation, so a `tool_result`/message whose `content` is the
/// documented `[{"type":"text","text":"…"}]` block array is counted as the model bills it,
/// not inflated by the array's braces and keys.
fn count_text(v: &Value, counter: &dyn TokenCounter) -> usize {
    match v {
        Value::String(s) => counter.count(s),
        _ => counter.count(&flatten_text(v)),
    }
}

/// Tool name from a schema entry across provider shapes: Anthropic/Google `name`,
/// OpenAI chat `function.name`, OpenAI Responses flattened `name`.
fn schema_tool_name(tool: &Value) -> Option<&str> {
    tool.get("name").and_then(Value::as_str).or_else(|| {
        tool.get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
    })
}

/// Emit one `schema` block per tool in a `tools` array.
fn attribute_tools(tools: &Value, counter: &dyn TokenCounter, out: &mut Vec<BlockAttribution>) {
    let Some(arr) = tools.as_array() else { return };
    for tool in arr {
        let name = schema_tool_name(tool);
        out.push(BlockAttribution {
            zone: Zone::Input,
            section: Section::Static,
            bucket: Bucket::Schema,
            mcp_server: name.and_then(mcp_server_of).map(str::to_string),
            tool_name: name.map(str::to_string),
            role: None,
            msg_index: None,
            tokens: count_value(tool, counter),
        });
    }
}

fn attribute_anthropic(body: &Value, counter: &dyn TokenCounter, out: &mut Vec<BlockAttribution>) {
    // `system`: a string or an array of text blocks.
    if let Some(system) = body.get("system") {
        let tokens = count_value(system, counter);
        if tokens > 0 {
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Static,
                bucket: Bucket::System,
                mcp_server: None,
                tool_name: None,
                role: None,
                msg_index: None,
                tokens,
            });
        }
    }
    if let Some(tools) = body.get("tools") {
        attribute_tools(tools, counter, out);
    }
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return;
    };
    for (i, msg) in messages.iter().enumerate() {
        let role = msg
            .get("role")
            .and_then(Value::as_str)
            .and_then(Role::parse);
        match msg.get("content") {
            Some(Value::String(s)) => push_text(out, Section::Messages, role, i, counter.count(s)),
            Some(Value::Array(blocks)) => {
                for b in blocks {
                    attribute_anthropic_block(b, role, i, counter, out);
                }
            }
            _ => {}
        }
    }
}

/// Classify one Anthropic content block (`text`/`thinking`/`tool_use`/`tool_result`).
fn attribute_anthropic_block(
    b: &Value,
    role: Option<Role>,
    msg_index: usize,
    counter: &dyn TokenCounter,
    out: &mut Vec<BlockAttribution>,
) {
    let ty = b.get("type").and_then(Value::as_str).unwrap_or("");
    let (bucket, tokens) = match ty {
        "text" => (
            Bucket::Text,
            count_text(b.get("text").unwrap_or(b), counter),
        ),
        "thinking" | "redacted_thinking" => {
            let t = b.get("thinking").or_else(|| b.get("data")).unwrap_or(b);
            (Bucket::Thinking, count_text(t, counter))
        }
        "tool_use" | "server_tool_use" => (Bucket::ToolUse, count_value(b, counter)),
        "tool_result" | "web_search_tool_result" => {
            // `content` may be a string or the documented array of `{type,text}` blocks —
            // count the text it carries, not the JSON envelope.
            let c = b.get("content").unwrap_or(b);
            (Bucket::ToolResult, count_text(c, counter))
        }
        _ => (Bucket::Text, count_value(b, counter)),
    };
    // tool_use names the tool directly; tool_result references it indirectly (no name on
    // the wire), so MCP attribution only applies to tool_use here.
    let tool_name = b.get("name").and_then(Value::as_str).map(str::to_string);
    out.push(BlockAttribution {
        zone: Zone::Input,
        section: Section::Messages,
        bucket,
        mcp_server: tool_name
            .as_deref()
            .and_then(mcp_server_of)
            .map(str::to_string),
        tool_name,
        role,
        msg_index: Some(msg_index),
        tokens,
    });
}

fn attribute_openai(body: &Value, counter: &dyn TokenCounter, out: &mut Vec<BlockAttribution>) {
    // Responses API: `instructions` (system) + `input` (array). Chat API: `messages`.
    if let Some(instr) = body.get("instructions") {
        let tokens = count_value(instr, counter);
        if tokens > 0 {
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Static,
                bucket: Bucket::System,
                mcp_server: None,
                tool_name: None,
                role: None,
                msg_index: None,
                tokens,
            });
        }
    }
    if let Some(tools) = body.get("tools") {
        attribute_tools(tools, counter, out);
    }
    let items = body
        .get("messages")
        .or_else(|| body.get("input"))
        .and_then(Value::as_array);
    let Some(items) = items else { return };
    for (i, msg) in items.iter().enumerate() {
        attribute_openai_item(msg, i, counter, out);
    }
}

/// Classify one OpenAI chat message or Responses input item.
fn attribute_openai_item(
    msg: &Value,
    i: usize,
    counter: &dyn TokenCounter,
    out: &mut Vec<BlockAttribution>,
) {
    let role_str = msg.get("role").and_then(Value::as_str).unwrap_or("");
    let item_type = msg.get("type").and_then(Value::as_str).unwrap_or("");

    // Responses-API function call / output items (no `role`).
    match item_type {
        "function_call" | "custom_tool_call" => {
            let name = msg.get("name").and_then(Value::as_str).map(str::to_string);
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Messages,
                bucket: Bucket::ToolUse,
                mcp_server: name.as_deref().and_then(mcp_server_of).map(str::to_string),
                tool_name: name,
                role: Some(Role::Assistant),
                msg_index: Some(i),
                tokens: count_value(msg, counter),
            });
            return;
        }
        "function_call_output" | "custom_tool_call_output" => {
            let c = msg.get("output").unwrap_or(msg);
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Messages,
                bucket: Bucket::ToolResult,
                mcp_server: None,
                tool_name: None,
                role: Some(Role::User),
                msg_index: Some(i),
                tokens: count_text(c, counter),
            });
            return;
        }
        _ => {}
    }

    // System / developer role → static system prompt. `msg_index` stays `None` to match the
    // static-block contract (and the Anthropic/Google system blocks).
    if matches!(role_str, "system" | "developer") {
        out.push(BlockAttribution {
            zone: Zone::Input,
            section: Section::Static,
            bucket: Bucket::System,
            mcp_server: None,
            tool_name: None,
            role: None,
            msg_index: None,
            tokens: count_text(msg.get("content").unwrap_or(msg), counter),
        });
        return;
    }

    // Chat `tool` role = a tool result.
    if role_str == "tool" {
        out.push(BlockAttribution {
            zone: Zone::Input,
            section: Section::Messages,
            bucket: Bucket::ToolResult,
            mcp_server: None,
            tool_name: None,
            role: Some(Role::User),
            msg_index: Some(i),
            tokens: count_text(msg.get("content").unwrap_or(msg), counter),
        });
        return;
    }

    let role = Role::parse(role_str);
    // Assistant `tool_calls` (chat API) are tool_use blocks.
    if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
        for call in calls {
            let name = call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .map(str::to_string);
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Messages,
                bucket: Bucket::ToolUse,
                mcp_server: name.as_deref().and_then(mcp_server_of).map(str::to_string),
                tool_name: name,
                role: Some(Role::Assistant),
                msg_index: Some(i),
                tokens: count_value(call, counter),
            });
        }
    }
    if let Some(content) = msg.get("content") {
        let tokens = count_text(content, counter);
        if tokens > 0 {
            push_text(out, Section::Messages, role, i, tokens);
        }
    }
}

fn attribute_google(body: &Value, counter: &dyn TokenCounter, out: &mut Vec<BlockAttribution>) {
    if let Some(si) = body
        .get("systemInstruction")
        .or_else(|| body.get("system_instruction"))
    {
        let tokens = count_value(si, counter);
        if tokens > 0 {
            out.push(BlockAttribution {
                zone: Zone::Input,
                section: Section::Static,
                bucket: Bucket::System,
                mcp_server: None,
                tool_name: None,
                role: None,
                msg_index: None,
                tokens,
            });
        }
    }
    // Google nests function declarations under tools[].functionDeclarations[].
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for tool in tools {
            if let Some(decls) = tool.get("functionDeclarations").and_then(Value::as_array) {
                attribute_tools(&Value::Array(decls.clone()), counter, out);
            }
        }
    }
    let Some(contents) = body.get("contents").and_then(Value::as_array) else {
        return;
    };
    for (i, msg) in contents.iter().enumerate() {
        let role = msg
            .get("role")
            .and_then(Value::as_str)
            .and_then(Role::parse);
        let Some(parts) = msg.get("parts").and_then(Value::as_array) else {
            continue;
        };
        for part in parts {
            if let Some(fc) = part.get("functionCall") {
                let name = fc.get("name").and_then(Value::as_str).map(str::to_string);
                out.push(BlockAttribution {
                    zone: Zone::Input,
                    section: Section::Messages,
                    bucket: Bucket::ToolUse,
                    mcp_server: name.as_deref().and_then(mcp_server_of).map(str::to_string),
                    tool_name: name,
                    role: Some(Role::Assistant),
                    msg_index: Some(i),
                    tokens: count_value(fc, counter),
                });
            } else if let Some(fr) = part.get("functionResponse") {
                out.push(BlockAttribution {
                    zone: Zone::Input,
                    section: Section::Messages,
                    bucket: Bucket::ToolResult,
                    mcp_server: None,
                    tool_name: None,
                    role: Some(Role::User),
                    msg_index: Some(i),
                    tokens: count_value(fr, counter),
                });
            } else if let Some(t) = part.get("text") {
                let tokens = counter.count(t.as_str().unwrap_or(""));
                if tokens > 0 {
                    push_text(out, Section::Messages, role, i, tokens);
                }
            }
        }
    }
}

fn push_text(
    out: &mut Vec<BlockAttribution>,
    section: Section,
    role: Option<Role>,
    msg_index: usize,
    tokens: usize,
) {
    if tokens == 0 {
        return;
    }
    out.push(BlockAttribution {
        zone: Zone::Input,
        section,
        bucket: Bucket::Text,
        mcp_server: None,
        tool_name: None,
        role,
        msg_index: Some(msg_index),
        tokens,
    });
}

/// Best-effort session/agent/project identity inferred from a request body.
///
/// This is the fallback path: the proxy first tries to cross-reference the local agent's
/// own session logs; when no log matches, these heuristics keep traffic grouped. All
/// fields are best-effort and may be `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RequestIdentity {
    /// Stable id for the conversation: a hash of the request's static prefix (system
    /// prompt), which is identical across the turns of one session.
    pub session_id: Option<String>,
    /// Coarse agent label fingerprinted from the system prompt ("claude-code", "codex").
    pub agent: Option<String>,
    /// Working directory the agent embedded in its context, when present.
    pub project: Option<String>,
}

/// Infer identity from a parsed request body (the fallback when no agent log matches).
pub fn extract_identity(body: &Value, kind: ProviderKind) -> RequestIdentity {
    let system = system_text(body, kind);
    RequestIdentity {
        session_id: system.as_deref().map(stable_hash),
        agent: system
            .as_deref()
            .and_then(fingerprint_agent)
            .map(str::to_string),
        project: system.as_deref().and_then(extract_cwd),
    }
}

/// The system-prompt text across provider shapes, concatenated.
fn system_text(body: &Value, kind: ProviderKind) -> Option<String> {
    let field = match kind {
        ProviderKind::Anthropic => body.get("system"),
        ProviderKind::OpenAi => body.get("instructions"),
        ProviderKind::Google => body
            .get("systemInstruction")
            .or_else(|| body.get("system_instruction")),
    };
    if let Some(v) = field {
        return Some(flatten_text(v));
    }
    // OpenAI chat / Responses: first system/developer message.
    let items = body
        .get("messages")
        .or_else(|| body.get("input"))
        .and_then(Value::as_array)?;
    items
        .iter()
        .find(|m| {
            matches!(
                m.get("role").and_then(Value::as_str),
                Some("system") | Some("developer")
            )
        })
        .map(|m| flatten_text(m.get("content").unwrap_or(m)))
}

/// Collect every string in a value (string, array of blocks, or object with `text`).
fn flatten_text(v: &Value) -> String {
    fn walk(v: &Value, acc: &mut String) {
        match v {
            Value::String(s) => {
                acc.push_str(s);
                acc.push('\n');
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, acc)),
            Value::Object(o) => {
                if let Some(t) = o.get("text").and_then(Value::as_str) {
                    acc.push_str(t);
                    acc.push('\n');
                }
            }
            _ => {}
        }
    }
    let mut acc = String::new();
    walk(v, &mut acc);
    acc
}

/// A short stable hex hash of a string (FNV-1a, 64-bit). Used to key a session by its
/// system-prompt prefix without pulling in a crypto dependency.
fn stable_hash(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

/// A known coding agent and the verbatim product strings it embeds in its system prompt.
///
/// Identification is by system-prompt body, never by `User-Agent` or other headers: some
/// agents deliberately spoof another's headers (Oh My Pi sends the Gemini CLI / Antigravity
/// `User-Agent`; Forge sends Anthropic's `anthropic-beta: claude-code-20250219`), so a header
/// match would mislabel them. The `markers` are registered ASCII brand strings that appear
/// regardless of the user's locale, so the match is brand-name based, not localized.
struct AgentFingerprint {
    label: &'static str,
    markers: &'static [&'static str],
}

/// The Tier A roster: agents that route through a host the live proxy intercepts and carry a
/// verified system-prompt marker. Order matters only where a generic fallback marker could
/// collide: Qwen Code is a Gemini CLI fork and can carry Gemini's legacy generic phrase, so
/// `qwen` is listed before `gemini` and the brand-specific marker wins.
static AGENTS: &[AgentFingerprint] = &[
    AgentFingerprint {
        label: "claude-code",
        markers: &["Claude Code"],
    },
    AgentFingerprint {
        label: "codex",
        markers: &["running as a coding agent in the Codex CLI", "Codex CLI"],
    },
    AgentFingerprint {
        label: "cursor",
        markers: &["You operate exclusively in Cursor"],
    },
    AgentFingerprint {
        label: "cline",
        markers: &["You are Cline,"],
    },
    AgentFingerprint {
        label: "roo",
        markers: &["You are Roo,"],
    },
    AgentFingerprint {
        label: "kilo",
        markers: &["You are Kilo Code,"],
    },
    AgentFingerprint {
        label: "goose",
        markers: &["agent called goose"],
    },
    AgentFingerprint {
        label: "opencode",
        markers: &["You are OpenCode,", "You are opencode,"],
    },
    AgentFingerprint {
        label: "crush",
        markers: &["You are Crush,"],
    },
    AgentFingerprint {
        label: "qwen",
        markers: &["You are Qwen Code,"],
    },
    AgentFingerprint {
        label: "grok",
        markers: &["You are Grok CLI"],
    },
    AgentFingerprint {
        label: "kimi",
        markers: &["You are Kimi Code CLI,"],
    },
    AgentFingerprint {
        label: "mistral-vibe",
        markers: &["operating as and within Mistral Vibe"],
    },
    AgentFingerprint {
        label: "mux",
        markers: &["agent called Mux"],
    },
    AgentFingerprint {
        label: "pi",
        markers: &["Oh My Pi"],
    },
    AgentFingerprint {
        label: "forge",
        markers: &["You are Forge,"],
    },
    AgentFingerprint {
        label: "openclaw",
        markers: &["OpenClaw"],
    },
    // Listed after qwen: its primary marker is brand-specific, but the legacy fallback phrase
    // is generic and also appears in Gemini CLI forks like Qwen Code.
    AgentFingerprint {
        label: "gemini",
        markers: &[
            "You are Gemini CLI,",
            "interactive CLI agent specializing in software engineering tasks",
        ],
    },
];

/// Identify the agent from well-known system-prompt markers. Returns the first registered
/// agent whose marker appears verbatim in the system prompt; an unrecognized agent maps to
/// `None` and is grouped under "unknown".
fn fingerprint_agent(system: &str) -> Option<&'static str> {
    AGENTS
        .iter()
        .find(|a| a.markers.iter().any(|m| system.contains(m)))
        .map(|a| a.label)
}

/// Pull a working-directory hint out of the system prompt. Claude Code and Codex embed a
/// line like `Current working directory: /path` (and a `cwd:` variant) in English markers,
/// which is what we match. This is deliberately English-only: it's a best-effort label used
/// only to group sessions by project, and the supported agents emit the English header
/// regardless of the user's locale. A prompt without a recognized marker yields `None`
/// (the session is still grouped by id and agent), so no functionality is lost — only the
/// project label is absent. It is not a universal natural-language parser.
fn extract_cwd(system: &str) -> Option<String> {
    for line in system.lines() {
        let line = line.trim();
        for marker in ["Current working directory:", "cwd:", "Working directory:"] {
            if let Some(rest) = line.strip_prefix(marker) {
                let p = rest.trim().trim_matches(|c| c == '`' || c == '"');
                if !p.is_empty() {
                    return Some(p.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Whitespace token counter (deterministic, tokenizer-independent) for tests.
    struct WordCounter;
    impl TokenCounter for WordCounter {
        fn count(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
        fn is_exact(&self) -> bool {
            false
        }
        fn label(&self) -> &str {
            "words"
        }
    }

    fn cats(blocks: &[BlockAttribution]) -> Vec<(&'static str, &'static str)> {
        blocks.iter().map(BlockAttribution::category).collect()
    }

    #[test]
    fn mcp_server_extraction() {
        assert_eq!(mcp_server_of("mcp__github__create_issue"), Some("github"));
        assert_eq!(mcp_server_of("mcp__filesystem__read"), Some("filesystem"));
        assert_eq!(mcp_server_of("Read"), None);
        assert_eq!(mcp_server_of("mcp__solo"), Some("solo"));
        // Degenerate `mcp__` (empty server segment) is not a real server.
        assert_eq!(mcp_server_of("mcp__"), None);
    }

    #[test]
    fn tool_result_content_array_counts_inner_text_not_json() {
        // content as the documented block array: only the text is billed, not the braces.
        let body = json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "content": [{"type": "text", "text": "one two three"}]}
                ]}
            ]
        });
        let blocks = attribute(&body, ProviderKind::Anthropic, &WordCounter);
        let tr = blocks
            .iter()
            .find(|b| b.bucket == Bucket::ToolResult)
            .unwrap();
        // WordCounter counts whitespace tokens: "one two three" = 3, not inflated by JSON.
        assert_eq!(tr.tokens, 3);
    }

    #[test]
    fn openai_content_array_counts_text_only() {
        let body = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "alpha beta"}]}
            ]
        });
        let blocks = attribute(&body, ProviderKind::OpenAi, &WordCounter);
        let txt = blocks.iter().find(|b| b.bucket == Bucket::Text).unwrap();
        assert_eq!(txt.tokens, 2);
    }

    #[test]
    fn anthropic_full_request() {
        let body = json!({
            "model": "claude-sonnet-4",
            "system": "You are Claude Code. Current working directory: /home/me/proj",
            "tools": [
                {"name": "Read", "description": "read a file"},
                {"name": "mcp__github__create_issue", "description": "open an issue"}
            ],
            "messages": [
                {"role": "user", "content": "hello there friend"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "let me think about this"},
                    {"type": "tool_use", "name": "mcp__github__create_issue", "input": {"title": "bug"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "content": "issue created ok"}
                ]}
            ]
        });
        let blocks = attribute(&body, ProviderKind::Anthropic, &WordCounter);
        assert_eq!(
            cats(&blocks),
            vec![
                ("Static", "System prompt"),
                ("Static", "System tools"),
                ("Static", "MCP tools"),
                ("Messages", "user text"),
                ("Messages", "thinking"),
                ("Messages", "MCP tool use+output"),
                ("Messages", "system tool use+output"),
            ]
        );
        // MCP server attribution survives onto the tool schema and the tool_use block.
        let mcp_tool = blocks
            .iter()
            .find(|b| b.bucket == Bucket::Schema && b.mcp_server.is_some());
        assert_eq!(mcp_tool.unwrap().mcp_server.as_deref(), Some("github"));
    }

    #[test]
    fn openai_chat_request() {
        let body = json!({
            "messages": [
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "do a thing"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"function": {"name": "mcp__db__query"}}
                ]},
                {"role": "tool", "content": "42 rows"}
            ],
            "tools": [{"function": {"name": "mcp__db__query"}}]
        });
        let blocks = attribute(&body, ProviderKind::OpenAi, &WordCounter);
        assert_eq!(
            cats(&blocks),
            vec![
                ("Static", "MCP tools"),
                ("Static", "System prompt"),
                ("Messages", "user text"),
                ("Messages", "MCP tool use+output"),
                ("Messages", "system tool use+output"),
            ]
        );
    }

    #[test]
    fn empty_and_malformed_bodies_yield_no_panic() {
        assert!(attribute(&json!({}), ProviderKind::Anthropic, &WordCounter).is_empty());
        assert!(
            attribute(
                &json!({"messages": "not an array"}),
                ProviderKind::OpenAi,
                &WordCounter
            )
            .is_empty()
        );
        assert!(attribute(&json!(7), ProviderKind::Google, &WordCounter).is_empty());
    }

    #[test]
    fn identity_from_claude_code_system() {
        let body = json!({
            "system": "You are Claude Code, Anthropic's CLI.\nCurrent working directory: /home/me/app",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let id = extract_identity(&body, ProviderKind::Anthropic);
        assert_eq!(id.agent.as_deref(), Some("claude-code"));
        assert_eq!(id.project.as_deref(), Some("/home/me/app"));
        assert!(id.session_id.is_some());
        // Session id is stable across turns (same system prefix).
        let body2 = json!({
            "system": "You are Claude Code, Anthropic's CLI.\nCurrent working directory: /home/me/app",
            "messages": [{"role": "user", "content": "second turn"}]
        });
        assert_eq!(
            id.session_id,
            extract_identity(&body2, ProviderKind::Anthropic).session_id
        );
    }

    #[test]
    fn every_registered_agent_fingerprints() {
        // Drive the test from the registry itself, not a parallel table: every agent's first
        // marker, fed through the real extraction path (which also guards `system_text`), must
        // resolve back to that agent. A new `AGENTS` entry is covered the day it is added.
        for agent in AGENTS {
            let marker = agent.markers[0];
            let body = json!({ "system": marker, "messages": [] });
            let id = extract_identity(&body, ProviderKind::Anthropic);
            assert_eq!(
                id.agent.as_deref(),
                Some(agent.label),
                "marker {marker:?} did not fingerprint as {}",
                agent.label
            );
        }
    }

    #[test]
    fn qwen_legacy_phrase_does_not_steal_from_gemini_fork() {
        // Qwen Code is a Gemini CLI fork: its prompt can carry both its own brand marker and
        // Gemini's generic legacy phrase. The brand-specific qwen marker must win.
        let body = json!({
            "system": "You are Qwen Code, an interactive CLI agent specializing in software engineering tasks.",
            "messages": []
        });
        assert_eq!(
            extract_identity(&body, ProviderKind::Anthropic)
                .agent
                .as_deref(),
            Some("qwen")
        );
    }

    #[test]
    fn unknown_agent_is_none() {
        let body = json!({ "system": "You are a helpful assistant.", "messages": [] });
        assert_eq!(extract_identity(&body, ProviderKind::Anthropic).agent, None);
    }

    #[test]
    fn google_request() {
        let body = json!({
            "systemInstruction": {"parts": [{"text": "be concise"}]},
            "tools": [{"functionDeclarations": [{"name": "mcp__x__y"}]}],
            "contents": [
                {"role": "user", "parts": [{"text": "question here"}]},
                {"role": "model", "parts": [{"functionCall": {"name": "mcp__x__y"}}]}
            ]
        });
        let blocks = attribute(&body, ProviderKind::Google, &WordCounter);
        assert_eq!(
            cats(&blocks),
            vec![
                ("Static", "System prompt"),
                ("Static", "MCP tools"),
                ("Messages", "user text"),
                ("Messages", "MCP tool use+output"),
            ]
        );
    }
}
