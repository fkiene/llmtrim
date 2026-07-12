//! Shared streaming model for the subscription reroute.
//!
//! Both provider reducers (Codex, Kimi) translate their own wire SSE into a common
//! [`ReduceEvent`] stream; a single [`AnthropicSseEncoder`] turns that stream into the Anthropic
//! `/v1/messages` SSE the client (Claude Code) expects. Centralising the Anthropic side here keeps
//! content-block index bookkeeping in one place — the reducers never compute Anthropic indices,
//! they just say "a text block started / a delta arrived / it stopped" and the encoder assigns and
//! tracks the index. This is deliberately the *only* place that emits `message_start`,
//! `content_block_*`, `message_delta`, and `message_stop`, so the wire contract can't drift between
//! providers.

use serde_json::{Value, json};

/// Why the model stopped, mapped onto Anthropic `stop_reason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

impl StopReason {
    pub fn as_str(self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::ToolUse => "tool_use",
            StopReason::MaxTokens => "max_tokens",
        }
    }
}

/// Token accounting harvested from the upstream response, already mapped to Anthropic's four-way
/// split. `input` is *fresh* (non-cached) input, matching Anthropic's `input_tokens`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    pub input: i64,
    pub output: i64,
    pub cache_read: i64,
    pub cache_write: i64,
}

impl Usage {
    /// The Anthropic `usage` object for a `message_delta` / `message_start`.
    fn to_json(self) -> Value {
        json!({
            "input_tokens": self.input,
            "output_tokens": self.output,
            "cache_creation_input_tokens": self.cache_write,
            "cache_read_input_tokens": self.cache_read,
        })
    }
}

/// A single normalized event from a provider reducer, in emission order. The encoder relies on
/// well-formed nesting: every `*Start` is eventually followed by its `*Stop`, and blocks do not
/// interleave (thinking closes before text opens, etc.). Reducers are responsible for closing an
/// open block before opening another — the encoder only assigns indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceEvent {
    ThinkingStart,
    ThinkingDelta(String),
    ThinkingStop,
    TextStart,
    TextDelta(String),
    TextStop,
    /// A tool call opened: Anthropic needs the id + name up front on the `content_block_start`.
    ToolStart {
        id: String,
        name: String,
    },
    /// A fragment of the tool-call arguments JSON (streamed as `input_json_delta.partial_json`).
    ToolDelta(String),
    ToolStop,
    /// Terminal event: emit `message_delta` (stop_reason + usage) then `message_stop`.
    Finish {
        stop_reason: StopReason,
        usage: Usage,
        /// Codex (and similar) response id, used for server-side continuation via
        /// previous_response_id. Present when the upstream provided one.
        response_id: Option<String>,
        /// Whether this terminal is eligible for continuation recording.
        /// Mirrors the proxy: true only for completed non-incomplete turns.
        continuation_eligible: bool,
    },
    /// A mid-stream upstream failure. Once `message_start` has been sent we cannot flip the HTTP
    /// status, so this is surfaced as an Anthropic SSE `error` event (which Claude Code renders).
    Error {
        message: String,
    },
}

/// Turns an ordered [`ReduceEvent`] stream into Anthropic `/v1/messages` SSE bytes. Stateful:
/// tracks whether `message_start` was emitted and the running content-block index.
pub struct AnthropicSseEncoder {
    message_id: String,
    model: String,
    started: bool,
    next_index: i64,
    /// The index of the currently open block (set on any `*Start`), used for its deltas + stop.
    current_index: i64,
    stopped: bool,
}

impl AnthropicSseEncoder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            message_id: format!("msg_{}", uuid::Uuid::new_v4().simple()),
            model: model.into(),
            started: false,
            next_index: 0,
            current_index: 0,
            stopped: false,
        }
    }

    /// Encode one event into zero or more SSE frames appended to `out`.
    pub fn encode(&mut self, event: &ReduceEvent, out: &mut String) {
        // Any content event implies the message has begun; emit the opener exactly once.
        if !self.started && !matches!(event, ReduceEvent::Error { .. }) {
            self.emit_message_start(out);
        }
        match event {
            ReduceEvent::ThinkingStart => {
                let idx = self.open_block(
                    out,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                );
                self.current_index = idx;
            }
            ReduceEvent::ThinkingDelta(text) => {
                self.block_delta(out, json!({"type": "thinking_delta", "thinking": text}))
            }
            ReduceEvent::ThinkingStop => self.close_block(out),
            ReduceEvent::TextStart => {
                let idx = self.open_block(out, json!({"type": "text", "text": ""}));
                self.current_index = idx;
            }
            ReduceEvent::TextDelta(text) => {
                self.block_delta(out, json!({"type": "text_delta", "text": text}))
            }
            ReduceEvent::TextStop => self.close_block(out),
            ReduceEvent::ToolStart { id, name } => {
                let idx = self.open_block(
                    out,
                    json!({"type": "tool_use", "id": id, "name": name, "input": {}}),
                );
                self.current_index = idx;
            }
            ReduceEvent::ToolDelta(partial) => self.block_delta(
                out,
                json!({"type": "input_json_delta", "partial_json": partial}),
            ),
            ReduceEvent::ToolStop => self.close_block(out),
            ReduceEvent::Finish {
                stop_reason, usage, ..
            } => self.emit_finish(out, *stop_reason, *usage),
            ReduceEvent::Error { message } => self.emit_error(out, message),
        }
    }

    /// Emit the terminal frames if the reducer never produced a `Finish` (e.g. the upstream stream
    /// was truncated). Idempotent — a no-op once a `Finish`/`Error` already closed the message.
    pub fn finish_if_open(&mut self, out: &mut String) {
        if self.started && !self.stopped {
            self.emit_finish(out, StopReason::EndTurn, Usage::default());
        }
    }

    fn emit_message_start(&mut self, out: &mut String) {
        self.started = true;
        let data = json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        });
        frame(out, "message_start", &data);
    }

    fn open_block(&mut self, out: &mut String, content_block: Value) -> i64 {
        let index = self.next_index;
        self.next_index += 1;
        let data = json!({
            "type": "content_block_start",
            "index": index,
            "content_block": content_block,
        });
        frame(out, "content_block_start", &data);
        index
    }

    fn block_delta(&self, out: &mut String, delta: Value) {
        let data = json!({
            "type": "content_block_delta",
            "index": self.current_index,
            "delta": delta,
        });
        frame(out, "content_block_delta", &data);
    }

    fn close_block(&self, out: &mut String) {
        let data = json!({"type": "content_block_stop", "index": self.current_index});
        frame(out, "content_block_stop", &data);
    }

    fn emit_finish(&mut self, out: &mut String, stop_reason: StopReason, usage: Usage) {
        if self.stopped {
            return;
        }
        self.stopped = true;
        let delta = json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason.as_str(), "stop_sequence": Value::Null},
            "usage": usage.to_json(),
        });
        frame(out, "message_delta", &delta);
        frame(out, "message_stop", &json!({"type": "message_stop"}));
    }

    fn emit_error(&mut self, out: &mut String, message: &str) {
        // Anthropic's SSE error frame. Safe to send before or after `message_start`.
        self.stopped = true;
        let data = json!({
            "type": "error",
            "error": {"type": "api_error", "message": message},
        });
        frame(out, "error", &data);
    }
}

/// Encode one Anthropic SSE frame (`event:` + `data:` line + blank line) onto `out`. Anthropic
/// sends compact single-line JSON in `data:`; match that.
fn frame(out: &mut String, event: &str, data: &Value) {
    out.push_str("event: ");
    out.push_str(event);
    out.push('\n');
    out.push_str("data: ");
    out.push_str(&data.to_string());
    out.push_str("\n\n");
}

/// A stateful incremental parser for `text/event-stream` upstream bodies: feed it raw byte chunks
/// as they arrive off the wire and it yields the JSON payload of each complete `data:` line. Buffers
/// partial lines across chunk boundaries (SSE events routinely split across TCP reads). Lines whose
/// data is empty or `[DONE]` are skipped. Both provider reducers use this to avoid re-implementing
/// SSE framing.
#[derive(Default)]
pub struct SseLineParser {
    buf: String,
}

impl SseLineParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a raw chunk; returns the parsed JSON of every `data:` line completed by this chunk.
    /// Non-JSON data lines are skipped (returned as `None` filtered out) so a stray keepalive can't
    /// abort translation.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<Value> {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        let mut out = Vec::new();
        // Process every complete line; keep the trailing partial in the buffer.
        while let Some(nl) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=nl).collect();
            let line = line.trim_end_matches(['\r', '\n']);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                out.push(v);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_all(model: &str, events: &[ReduceEvent]) -> String {
        let mut enc = AnthropicSseEncoder::new(model);
        let mut out = String::new();
        for e in events {
            enc.encode(e, &mut out);
        }
        enc.finish_if_open(&mut out);
        out
    }

    #[test]
    fn synthesized_sse_preserves_client_model_id() {
        // Sub reroute maps claude-sonnet-5 → gpt-5.6-luna upstream; the encoder must still
        // advertise the Claude Code selection in message_start or the client rejects the turn.
        let out = encode_all(
            "claude-sonnet-5",
            &[
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("ok".into()),
                ReduceEvent::TextStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage::default(),
                },
            ],
        );
        assert!(out.contains("\"model\":\"claude-sonnet-5\""));
        assert!(!out.contains("gpt-5.6-luna"));
    }

    #[test]
    fn text_turn_emits_well_formed_anthropic_sse() {
        let out = encode_all(
            "gpt-5.5",
            &[
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("Hello".into()),
                ReduceEvent::TextStop,
                // Note: include continuation fields (response_id + continuation_eligible)
                // so that MSRV/coverage/test jobs under all feature sets see complete literals.
                ReduceEvent::Finish {
                    stop_reason: StopReason::EndTurn,
                    usage: Usage {
                        input: 10,
                        output: 2,
                        cache_read: 3,
                        cache_write: 0,
                    },
                    response_id: None,
                    continuation_eligible: true,
                },
            ],
        );
        assert!(out.contains("event: message_start"));
        assert!(out.contains("\"model\":\"gpt-5.5\""));
        assert!(out.contains("event: content_block_start"));
        assert!(out.contains("\"text_delta\""));
        assert!(out.contains("event: content_block_stop"));
        assert!(out.contains("\"stop_reason\":\"end_turn\""));
        assert!(out.contains("\"cache_read_input_tokens\":3"));
        assert!(out.contains("event: message_stop"));
    }

    #[test]
    fn block_indices_increment_per_block() {
        let out = encode_all(
            "m",
            &[
                ReduceEvent::ThinkingStart,
                ReduceEvent::ThinkingDelta("t".into()),
                ReduceEvent::ThinkingStop,
                ReduceEvent::TextStart,
                ReduceEvent::TextDelta("x".into()),
                ReduceEvent::TextStop,
                ReduceEvent::ToolStart {
                    id: "toolu_1".into(),
                    name: "Read".into(),
                },
                ReduceEvent::ToolDelta("{\"path\":".into()),
                ReduceEvent::ToolStop,
                ReduceEvent::Finish {
                    stop_reason: StopReason::ToolUse,
                    usage: Usage::default(),
                    response_id: None,
                    continuation_eligible: false,
                },
            ],
        );
        assert!(out.contains("\"index\":0"));
        assert!(out.contains("\"index\":1"));
        assert!(out.contains("\"index\":2"));
        assert!(out.contains("\"type\":\"tool_use\""));
        assert!(out.contains("\"stop_reason\":\"tool_use\""));
    }

    #[test]
    fn finish_if_open_is_idempotent() {
        let mut enc = AnthropicSseEncoder::new("m");
        let mut out = String::new();
        enc.encode(&ReduceEvent::TextStart, &mut out);
        enc.encode(&ReduceEvent::TextStop, &mut out);
        enc.encode(
            &ReduceEvent::Finish {
                stop_reason: StopReason::EndTurn,
                usage: Usage::default(),
                response_id: None,
                continuation_eligible: false,
            },
            &mut out,
        );
        let len_after_finish = out.len();
        enc.finish_if_open(&mut out);
        assert_eq!(
            out.len(),
            len_after_finish,
            "finish_if_open must not double-emit"
        );
    }

    #[test]
    fn sse_line_parser_reassembles_split_frames() {
        let mut p = SseLineParser::new();
        assert!(p.push(b"data: {\"a\":").is_empty(), "partial line buffered");
        let got = p.push(b"1}\n\ndata: [DONE]\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["a"], 1);
    }

    #[test]
    fn sse_line_parser_skips_non_json_and_keepalive() {
        let mut p = SseLineParser::new();
        let got = p.push(b": keepalive\ndata: \ndata: not json\ndata: {\"ok\":true}\n");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["ok"], true);
    }
}
