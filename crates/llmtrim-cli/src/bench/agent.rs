//! Agent-loop benchmark (issue #14): per-iteration token economics for tool-calling loops.
//!
//! The single-shot harness in the parent module measures one request. Agent loops are
//! different: the model calls tools over many turns, and cost is dominated by how the
//! provider's prompt cache reuses the prefix across those turns (plus how many turns the
//! model takes). This module drives a real (or stubbed) tool loop and records, per turn,
//! the input / cached / output tokens and the tool-call count — the data contract issue #14
//! asks for — so a preset's agent-loop value can be measured on a golden set instead of one
//! noisy live session.
//!
//! ## The contract
//!
//! [`AgentRunResult`] (one per task × condition × repeat) carries the per-iteration breakdown
//! ([`IterUsage`]) plus totals and cost. The loop driver [`run_agent_loop`] is
//! provider-agnostic and transport-agnostic: a [`AgentProvider`] handles the wire shape, and a
//! `send` closure performs the round-trip (live HTTP, or a stub for tests / `--dry-run`). One
//! reference provider ships ([`OpenAiAgent`], Chat Completions); others implement the same trait.
//!
//! ## Conditions
//!
//! A [`Condition`] is how each outgoing request is transformed before sending: `Baseline`
//! (unchanged) or `Preset(name)` (the proxy's transform — `compress_with_config` plus the
//! turn-stability memo, so message-content prefixes stay byte-stable across turns exactly as
//! the live proxy does). Comparing conditions on the same task isolates llmtrim's effect.

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::Pricing;
use crate::quality::{Usage, parse_usage};
use llmtrim_core::compress_with_config;
use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;
use llmtrim_core::memo::{self, Memo};

// ── Data contract ───────────────────────────────────────────────────────────────────────

/// Token usage for one model round-trip (one loop iteration).
#[derive(Debug, Clone, Default, Serialize)]
pub struct IterUsage {
    pub index: usize,
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    /// Tool calls the model requested this turn (0 on the final, answer-only turn).
    pub tool_calls: usize,
}

/// The result of one agent loop: the per-iteration breakdown plus totals and cost.
#[derive(Debug, Clone, Default, Serialize)]
pub struct AgentRunResult {
    pub task_id: String,
    pub provider: String,
    pub model: String,
    pub condition: String,
    pub iterations: usize,
    /// Whether the loop reached a final (non-tool) answer within `max_iterations`.
    pub completed: bool,
    pub per_iter: Vec<IterUsage>,
    pub input_tokens: usize,
    pub cached_tokens: usize,
    pub output_tokens: usize,
    /// Billed cost with cache accounting (cached input at the cache-read rate).
    pub cost_usd: f64,
}

// ── Golden task ─────────────────────────────────────────────────────────────────────────

/// A golden agent task: the opening request plus deterministic tool stubs so the loop is
/// bounded and comparable across conditions. The model still chooses how many rounds to take,
/// which is how the benchmark captures iteration drift.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub model: String,
    pub system: String,
    pub user: String,
    #[serde(default)]
    pub tools: Value,
    /// Canned output per tool name; unlisted tools fall back to `default_stub`.
    #[serde(default)]
    pub tool_stubs: HashMap<String, String>,
    #[serde(default = "default_stub")]
    pub default_stub: String,
    #[serde(default = "default_max_iter")]
    pub max_iterations: usize,
    /// Per-turn output-token cap. Too low truncates the final answer turn, so the loop never
    /// sees a clean final and keeps calling tools (false iteration drift); 512 leaves room.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u64,
}

fn default_stub() -> String {
    "ok".to_string()
}
fn default_max_iter() -> usize {
    12
}
fn default_max_tokens() -> u64 {
    512
}

impl AgentTask {
    pub fn from_json(s: &str) -> Result<Self> {
        serde_json::from_str(s).context("invalid agent task JSON")
    }

    fn stub_for(&self, name: &str) -> String {
        self.tool_stubs
            .get(name)
            .cloned()
            .unwrap_or_else(|| self.default_stub.clone())
    }
}

// ── Provider seam ───────────────────────────────────────────────────────────────────────

pub struct ToolCall {
    pub id: String,
    pub name: String,
}

pub enum TurnAction {
    Final(String),
    Calls(Vec<ToolCall>),
}

/// One parsed model turn: its usage, what it did (tool calls vs final answer), and the raw
/// assistant message to replay back into the conversation.
pub struct Turn {
    pub usage: Usage,
    pub action: TurnAction,
    pub assistant_msg: Value,
}

/// Wire-shape adapter. Implement per provider; the loop driver stays generic.
pub trait AgentProvider {
    fn name(&self) -> &str;
    /// The provider kind for compression/tokenization.
    fn kind(&self) -> ProviderKind;
    /// Build the opening request body (system + tools + first user turn).
    fn initial_request(&self, task: &AgentTask) -> Value;
    /// Parse a raw response into usage + action + the assistant message.
    fn parse_turn(&self, response: &Value) -> Result<Turn>;
    /// Append the assistant turn and the tool results to the conversation for the next round.
    fn append_results(&self, body: &mut Value, assistant_msg: &Value, results: &[(String, String)]);
}

/// OpenAI Chat Completions reference provider.
pub struct OpenAiAgent;

impl AgentProvider for OpenAiAgent {
    fn name(&self) -> &str {
        "openai"
    }
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    fn initial_request(&self, task: &AgentTask) -> Value {
        json!({
            "model": task.model,
            "messages": [
                {"role": "system", "content": task.system},
                {"role": "user", "content": task.user},
            ],
            "tools": task.tools,
            "max_tokens": task.max_tokens,
            "temperature": 0,
        })
    }

    fn parse_turn(&self, response: &Value) -> Result<Turn> {
        let usage = parse_usage(response);
        let msg = response
            .pointer("/choices/0/message")
            .cloned()
            .unwrap_or(Value::Null);
        let calls = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|a| !a.is_empty());
        let action = match calls {
            Some(calls) => TurnAction::Calls(
                calls
                    .iter()
                    .map(|c| ToolCall {
                        id: c
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        name: c
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    })
                    .collect(),
            ),
            None => TurnAction::Final(
                msg.get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ),
        };
        Ok(Turn {
            usage,
            action,
            assistant_msg: msg,
        })
    }

    fn append_results(
        &self,
        body: &mut Value,
        assistant_msg: &Value,
        results: &[(String, String)],
    ) {
        if let Some(msgs) = body.get_mut("messages").and_then(Value::as_array_mut) {
            msgs.push(assistant_msg.clone());
            for (id, out) in results {
                msgs.push(json!({"role": "tool", "tool_call_id": id, "content": out}));
            }
        }
    }
}

// ── Conditions ──────────────────────────────────────────────────────────────────────────

/// How each outgoing request is transformed before it is sent.
pub enum Condition {
    /// Send the request unchanged (the unproxied baseline).
    Baseline,
    /// Apply the named preset's transform, mirroring the live proxy (compress + turn memo).
    Preset(String),
}

impl Condition {
    pub fn label(&self) -> String {
        match self {
            Condition::Baseline => "baseline".to_string(),
            Condition::Preset(p) => p.clone(),
        }
    }

    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("baseline") {
            Condition::Baseline
        } else {
            Condition::Preset(s.to_string())
        }
    }
}

// ── Loop driver ─────────────────────────────────────────────────────────────────────────

/// Drive one agent loop under one condition, recording the per-iteration token contract.
///
/// `send` performs the round-trip (live HTTP, or a stub). It is given the exact bytes that
/// would go on the wire — already compressed + memo-frozen for a `Preset` condition — so what
/// the benchmark measures is what the proxy would send.
pub fn run_agent_loop(
    provider: &dyn AgentProvider,
    task: &AgentTask,
    condition: &Condition,
    price: &Pricing,
    send: &mut dyn FnMut(&str) -> Result<Value>,
) -> Result<AgentRunResult> {
    let kind = provider.kind();
    // Per-conversation turn-stability memo, exactly as the serve proxy uses, so a preset's
    // earlier-turn message content stays byte-identical across turns (warm provider cache).
    let memo = Memo::with_capacity(memo::DEFAULT_CAPACITY);
    let config = match condition {
        Condition::Baseline => None,
        Condition::Preset(p) => {
            Some(DenseConfig::preset(p).with_context(|| format!("unknown preset '{p}'"))?)
        }
    };
    // Salt scopes the memo to this (provider, config) — the same key the serve layer derives.
    let salt = config.as_ref().map(|c| {
        format!(
            "{:?}|{}",
            kind,
            serde_json::to_string(c).unwrap_or_default()
        )
    });

    let mut body = provider.initial_request(task);
    let mut result = AgentRunResult {
        task_id: task.id.clone(),
        provider: provider.name().to_string(),
        model: task.model.clone(),
        condition: condition.label(),
        ..Default::default()
    };

    for i in 0..task.max_iterations {
        let to_send = match (&config, &salt) {
            (Some(cfg), Some(salt)) => {
                let original = body.clone();
                let compressed_json =
                    compress_with_config(&body.to_string(), Some(kind), cfg)?.request_json;
                let mut compressed: Value = serde_json::from_str(&compressed_json)
                    .context("compressed body is not valid JSON")?;
                memo::apply(&memo, salt.as_bytes(), &original, &mut compressed);
                compressed
            }
            _ => body.clone(),
        };

        let response = send(&to_send.to_string())?;
        let turn = provider.parse_turn(&response)?;
        let u = turn.usage;
        let tool_calls = match &turn.action {
            TurnAction::Calls(c) => c.len(),
            TurnAction::Final(_) => 0,
        };
        let iter = IterUsage {
            index: i,
            input_tokens: u.prompt_tokens.unwrap_or(0) as usize,
            cached_tokens: u.cached_tokens.unwrap_or(0) as usize,
            output_tokens: u.completion_tokens.unwrap_or(0) as usize,
            tool_calls,
        };
        result.input_tokens += iter.input_tokens;
        result.cached_tokens += iter.cached_tokens;
        result.output_tokens += iter.output_tokens;
        result.per_iter.push(iter);

        match turn.action {
            TurnAction::Final(_) => {
                result.completed = true;
                break;
            }
            TurnAction::Calls(calls) => {
                let results: Vec<(String, String)> = calls
                    .iter()
                    .map(|c| (c.id.clone(), task.stub_for(&c.name)))
                    .collect();
                provider.append_results(&mut body, &turn.assistant_msg, &results);
            }
        }
    }

    result.iterations = result.per_iter.len();
    result.cost_usd = price.cost_cached(
        result.input_tokens,
        result.cached_tokens,
        result.output_tokens,
    );
    Ok(result)
}

// ── Zero-cost dry-run transport ─────────────────────────────────────────────────────────

/// A deterministic, no-network transport for `--dry-run` and tests. It estimates usage from
/// the request itself: `prompt_tokens` ≈ chars/4, `cached_tokens` ≈ the byte-prefix shared
/// with the previous turn's sent body (a stand-in for provider longest-prefix caching), and a
/// fixed small completion. It calls the first tool for `tool_turns` rounds, then answers — so
/// the loop runs end to end and a stable prefix (preset) shows more cached tokens than a
/// churning one, all without spending a cent.
pub fn dry_run_transport(tool_turns: usize) -> impl FnMut(&str) -> Result<Value> {
    let mut prev: Option<String> = None;
    let mut turn = 0usize;
    move |request_json: &str| {
        let prompt = request_json.chars().count() / 4;
        let cached = prev
            .as_deref()
            .map(|p| common_prefix_chars(p, request_json) / 4)
            .unwrap_or(0)
            .min(prompt);
        prev = Some(request_json.to_string());

        let body: Value = serde_json::from_str(request_json).unwrap_or(Value::Null);
        let first_tool = body
            .pointer("/tools/0/function/name")
            .and_then(Value::as_str)
            .unwrap_or("noop")
            .to_string();

        let message = if turn < tool_turns {
            json!({
                "role": "assistant", "content": Value::Null,
                "tool_calls": [{
                    "id": format!("call_{turn}"), "type": "function",
                    "function": {"name": first_tool, "arguments": "{}"}
                }]
            })
        } else {
            json!({"role": "assistant", "content": "done"})
        };
        turn += 1;

        Ok(json!({
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": cached}
            }
        }))
    }
}

/// Length (in chars) of the shared leading prefix of two strings.
fn common_prefix_chars(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task() -> AgentTask {
        AgentTask::from_json(
            r#"{
                "id": "demo", "model": "gpt-4o-mini",
                "system": "You are a coding agent.",
                "user": "read the file then summarize it",
                "tools": [
                    {"type":"function","function":{"name":"read_file","description":"read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}},
                    {"type":"function","function":{"name":"grep","description":"search","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}
                ],
                "tool_stubs": {"read_file": "fn main() {}"},
                "max_iterations": 6
            }"#,
        )
        .unwrap()
    }

    fn price() -> Pricing {
        Pricing {
            input_per_1k: 0.15 / 1000.0 * 1000.0,
            output_per_1k: 0.6,
            cache_per_1k: 0.0375,
        }
    }

    #[test]
    fn loop_runs_to_completion_and_records_per_iteration() {
        let t = task();
        let mut send = dry_run_transport(2); // two tool turns, then a final answer
        let r =
            run_agent_loop(&OpenAiAgent, &t, &Condition::Baseline, &price(), &mut send).unwrap();
        assert_eq!(r.iterations, 3, "two tool rounds + one final answer");
        assert!(r.completed, "reached a final answer");
        assert_eq!(r.per_iter.len(), 3);
        assert_eq!(r.per_iter[0].tool_calls, 1);
        assert_eq!(r.per_iter[2].tool_calls, 0, "final turn has no tool call");
        assert!(r.input_tokens > 0 && r.output_tokens > 0);
        assert_eq!(r.condition, "baseline");
    }

    #[test]
    fn caps_at_max_iterations_when_never_final() {
        let t = task();
        let mut send = dry_run_transport(usize::MAX); // always calls a tool, never answers
        let r =
            run_agent_loop(&OpenAiAgent, &t, &Condition::Baseline, &price(), &mut send).unwrap();
        assert_eq!(r.iterations, t.max_iterations, "stops at the cap");
        assert!(!r.completed, "never reached a final answer");
    }

    #[test]
    fn preset_condition_compresses_and_stays_valid() {
        // The agent preset must run end to end and keep the request valid each turn.
        let t = task();
        let mut send = dry_run_transport(2);
        let r = run_agent_loop(
            &OpenAiAgent,
            &t,
            &Condition::Preset("agent".to_string()),
            &price(),
            &mut send,
        )
        .unwrap();
        assert!(r.completed && r.iterations == 3);
        assert_eq!(r.condition, "agent");
    }

    #[test]
    fn unknown_preset_is_an_error() {
        let t = task();
        let mut send = dry_run_transport(1);
        assert!(
            run_agent_loop(
                &OpenAiAgent,
                &t,
                &Condition::Preset("nope".to_string()),
                &price(),
                &mut send
            )
            .is_err()
        );
    }
}
