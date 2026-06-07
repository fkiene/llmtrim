//! Stage A — cache discipline (provider prefix caching). Lossless / opt-in.
//!
//! The #2 lever (spec §2): mark the invariant prefix (system prompt + tool schemas)
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
        provider.set_cache_breakpoints(req, self.max_breakpoints);
        Ok(())
    }
}

/// 64-bit FNV-1a. Unlike `DefaultHasher`, its output is **stable across Rust versions
/// and platforms**, so a prefix fingerprint stays comparable over time. Non-crypto —
/// used only to test equality of the cacheable prefix.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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
    fnv1a(buf.as_bytes())
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
