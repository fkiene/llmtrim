//! Stage A — cache discipline (provider prefix caching). Lossless / opt-in.
//!
//! The #2 lever: mark the invariant prefix (system prompt + tool schemas)
//! with provider cache breakpoints so the prefix is billed once and reused across
//! calls. On Anthropic this places `cache_control: {ephemeral}` (≤4 breakpoints); on
//! OpenAI it's a no-op (the longest matching prefix is cached automatically).
//!
//! Lossless — adds caching hints, never changes content — so it uses the
//! `Structural` gate (always applied; the discount is latent, realized on a later
//! call, not in per-call input tokens). Off by default: Anthropic cache *writes*
//! cost ~25% more, so it only pays off when the prefix is read again (multi-turn or
//! templated/structural reuse). Runs last so it fingerprints the final prefix.

use anyhow::Result;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::stages::tools::fnv1a;

pub struct CacheStage {
    /// Maximum cache breakpoints to place (Anthropic allows up to 4).
    pub max_breakpoints: usize,
}

impl Transform for CacheStage {
    fn name(&self) -> &str {
        "cache"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::Structural
    }

    fn scope(&self) -> crate::gate::Scope {
        // Adds `cache_control` metadata (to tool/system blocks); content TEXT is unchanged.
        crate::gate::Scope::Tools
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        // Stabilize the prefix so it's byte-identical across SDK restarts (raises the
        // provider's cache-hit rate). Skip when the client manages its own caching —
        // reordering tools or injecting a key would bust the cache it set up.
        if !crate::cache_zone::has_cache_control(req.raw()) {
            sort_tools(req);
            let key = format!("{:016x}", cache_prefix_hash(req));
            provider.set_prompt_cache_key(req, &key);
        }
        provider.set_cache_breakpoints(req, self.max_breakpoints);
        Ok(())
    }
}

/// Canonicalize `tools[]`: recursively sort every JSON-object key (schemas included), then
/// sort the tools by name. Object key order and tool order are semantically irrelevant, so
/// this is lossless — but it makes the prefix deterministic across SDKs that emit tools in
/// hash-randomized order, which otherwise bust the provider cache on every restart.
fn sort_tools(req: &mut Request) {
    let Some(Value::Array(tools)) = req.raw_mut().get_mut("tools") else {
        return;
    };
    for tool in tools.iter_mut() {
        sort_keys(tool);
    }
    tools.sort_by(|a, b| tool_name(a).cmp(tool_name(b)));
}

/// Tool name across wire shapes: Anthropic top-level `name`, OpenAI `function.name`.
fn tool_name(tool: &Value) -> &str {
    tool.get("name")
        .or_else(|| tool.get("function").and_then(|f| f.get("name")))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// Recursively sort object keys in place (relies on serde_json's `preserve_order`).
fn sort_keys(v: &mut Value) {
    match v {
        Value::Object(map) => {
            for child in map.values_mut() {
                sort_keys(child);
            }
            let mut entries: Vec<(String, Value)> = std::mem::take(map).into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            for (k, val) in entries {
                map.insert(k, val);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(sort_keys),
        _ => {}
    }
}

/// Fingerprint of the cacheable prefix (system + tool schemas). Two requests with a
/// byte-identical prefix share this hash → eligible for the provider prefix cache,
/// including across independent single-turn calls (structural reuse, *UniCache*).
/// Non-cryptographic (used only for equality of the prefix).
pub fn cache_prefix_hash(req: &Request) -> u64 {
    let raw = req.raw();
    let mut buf = String::new();
    let mut hashed_anything = false;

    // Anthropic-style: top-level `system` + `tools`.
    if let Some(sys) = raw.get("system") {
        buf.push_str(&sys.to_string());
        buf.push('\u{1f}'); // unit separator keeps adjacent fields distinct
        hashed_anything = true;
    }
    if let Some(tools) = raw.get("tools") {
        buf.push_str(&tools.to_string());
        buf.push('\u{1f}');
        hashed_anything = true;
    }

    // OpenAI-style: the leading run of system-role messages.
    if !hashed_anything && let Some(msgs) = raw.get("messages").and_then(Value::as_array) {
        for m in msgs {
            if m.get("role").and_then(Value::as_str) == Some("system") {
                buf.push_str(&m.to_string());
                buf.push('\u{1f}');
            } else {
                break;
            }
        }
    }
    fnv1a(buf.bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::provider::{AnthropicProvider, OpenAiProvider};
    use serde_json::json;

    fn anthropic(body: Value) -> Request {
        Request::from_value(ProviderKind::Anthropic, body)
    }

    #[test]
    fn anthropic_caches_system_string_as_block() {
        let mut req = anthropic(json!({"system":"you are helpful","max_tokens":1,"messages":[]}));
        AnthropicProvider.set_cache_breakpoints(&mut req, 4);
        let sys = req.raw().get("system").unwrap();
        assert_eq!(
            sys.pointer("/0/cache_control/type").and_then(Value::as_str),
            Some("ephemeral"),
            "string system becomes a cached text block"
        );
        assert_eq!(
            sys.pointer("/0/text").and_then(Value::as_str),
            Some("you are helpful")
        );
    }

    #[test]
    fn anthropic_caches_last_tool() {
        let mut req = anthropic(json!({
            "max_tokens":1, "messages":[],
            "tools":[{"name":"a","input_schema":{}},{"name":"b","input_schema":{}}]
        }));
        AnthropicProvider.set_cache_breakpoints(&mut req, 4);
        assert_eq!(
            req.raw()
                .pointer("/tools/1/cache_control/type")
                .and_then(Value::as_str),
            Some("ephemeral")
        );
        assert!(req.raw().pointer("/tools/0/cache_control").is_none());
    }

    #[test]
    fn respects_max_breakpoints() {
        let mut req = anthropic(json!({
            "system":"sys","max_tokens":1,"messages":[],
            "tools":[{"name":"a","input_schema":{}}]
        }));
        AnthropicProvider.set_cache_breakpoints(&mut req, 1);
        // Only one breakpoint: the tool is marked, the system is left untouched.
        assert!(req.raw().pointer("/tools/0/cache_control").is_some());
        assert!(
            req.raw().get("system").unwrap().is_string(),
            "system not converted (budget spent)"
        );
    }

    #[test]
    fn openai_is_noop() {
        let body =
            json!({"messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body.clone());
        OpenAiProvider.set_cache_breakpoints(&mut req, 4);
        assert_eq!(
            req.raw(),
            &body,
            "OpenAI request is unchanged (automatic caching)"
        );
    }

    fn run_cache_stage(req: &mut Request, provider: &dyn Provider) {
        let mut plan: Vec<PlanEntry> = Vec::new();
        CacheStage { max_breakpoints: 4 }
            .apply(req, provider, &mut plan)
            .unwrap();
    }

    #[test]
    fn stabilize_sorts_tools_and_schema_keys() {
        let mut req = Request::from_value(
            ProviderKind::OpenAi,
            json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {"type": "function", "function": {"name": "zebra", "parameters": {"b": 1, "a": 2}}},
                    {"type": "function", "function": {"name": "apple", "parameters": {}}},
                ]
            }),
        );
        run_cache_stage(&mut req, &OpenAiProvider);
        let tools = req.raw().get("tools").and_then(Value::as_array).unwrap();
        assert_eq!(
            tools[0].pointer("/function/name").unwrap(),
            "apple",
            "tools sorted by name"
        );
        assert_eq!(tools[1].pointer("/function/name").unwrap(), "zebra");
        let keys: Vec<&str> = tools[1]
            .pointer("/function/parameters")
            .and_then(Value::as_object)
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["a", "b"], "schema keys canonicalized");
    }

    #[test]
    fn openai_gets_a_stable_prompt_cache_key() {
        let mut req = Request::from_value(
            ProviderKind::OpenAi,
            json!({"model": "gpt-4o", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]}),
        );
        run_cache_stage(&mut req, &OpenAiProvider);
        assert!(
            req.raw()
                .get("prompt_cache_key")
                .and_then(Value::as_str)
                .is_some(),
            "prompt_cache_key injected for OpenAI"
        );
    }

    #[test]
    fn stabilize_defers_to_client_managed_caching() {
        // A client `cache_control` marker means it manages its own cache → we must not
        // reorder tools (that would bust it).
        let mut req = anthropic(json!({
            "max_tokens": 1, "messages": [],
            "tools": [
                {"name": "zebra", "input_schema": {}, "cache_control": {"type": "ephemeral"}},
                {"name": "apple", "input_schema": {}},
            ]
        }));
        run_cache_stage(&mut req, &AnthropicProvider);
        assert_eq!(
            req.raw().pointer("/tools/0/name").unwrap(),
            "zebra",
            "tool order preserved when the client manages caching"
        );
    }

    #[test]
    fn prefix_hash_is_stable_and_distinct() {
        let a = anthropic(json!({"system":"SAME","messages":[{"role":"user","content":"q1"}]}));
        let b = anthropic(
            json!({"system":"SAME","messages":[{"role":"user","content":"q2 different"}]}),
        );
        let c = anthropic(json!({"system":"OTHER","messages":[{"role":"user","content":"q1"}]}));
        // Same prefix (system) → same hash even with different turns (structural reuse).
        assert_eq!(cache_prefix_hash(&a), cache_prefix_hash(&b));
        // Different prefix → different hash.
        assert_ne!(cache_prefix_hash(&a), cache_prefix_hash(&c));
    }
}
