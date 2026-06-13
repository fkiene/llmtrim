//! Cache-zone discipline — never recompress the provider's frozen (cached) prefix.
//!
//! When a request carries `cache_control` markers (Claude Code sets these on the stable
//! prefix), the provider caches everything up to the last marker and bills it at ~0.1×.
//! Rewriting that content — even to save tokens — changes the cached bytes and busts the
//! cache, which usually costs *more* than the tokens saved (the "input compression is a
//! false economy" trap). So the content-mutating stages compress only the **live zone**:
//! the segments after the last `cache_control` marker. Each new tool result is therefore
//! compressed exactly once — when it first arrives in the live zone — then frozen.
//!
//! No markers ⇒ no known cache ⇒ everything is compressible (behavior unchanged):
//! determinism keeps an identical prefix cache-stable across calls, and Stage A's OpenAI
//! `prompt_cache_key` pins auto-cached prefixes.

use std::collections::HashSet;

use serde_json::Value;

use crate::ir::Request;
use crate::provider::Provider;

/// Content-text pointers safe to compress: every content pointer minus those inside the
/// frozen (cached) prefix. The stages iterate this instead of
/// [`Provider::content_text_pointers`]; the token gate still counts *all* content.
pub fn compressible_pointers(req: &Request, provider: &dyn Provider) -> Vec<String> {
    let frozen = frozen_pointers(req, provider);
    provider
        .content_text_pointers(req)
        .into_iter()
        .filter(|p| !frozen.contains(p))
        .collect()
}

/// Content-text pointers inside the frozen prefix — everything up to and including the
/// last `cache_control`-marked message, plus a cache-controlled `system`. Empty when the
/// request carries no `cache_control` markers (nothing known-cached to protect).
pub fn frozen_pointers(req: &Request, provider: &dyn Provider) -> HashSet<String> {
    let raw = req.raw();
    let system_frozen = raw.get("system").is_some_and(has_cache_control);
    let frozen_until = raw
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|msgs| {
            msgs.iter()
                .enumerate()
                .filter(|(_, m)| has_cache_control(m))
                .map(|(i, _)| i)
                .max()
        });

    if frozen_until.is_none() && !system_frozen {
        return HashSet::new();
    }
    provider
        .content_text_pointers(req)
        .into_iter()
        .filter(|p| is_frozen(p, frozen_until, system_frozen))
        .collect()
}

/// `cache_control` present anywhere within `v` (a block, a message, or nested content).
pub(crate) fn has_cache_control(v: &Value) -> bool {
    match v {
        Value::Object(m) => m.contains_key("cache_control") || m.values().any(has_cache_control),
        Value::Array(a) => a.iter().any(has_cache_control),
        _ => false,
    }
}

/// A pointer is frozen if it addresses the cache-controlled `system`, or a message at or
/// before `frozen_until`.
fn is_frozen(ptr: &str, frozen_until: Option<usize>, system_frozen: bool) -> bool {
    if let Some(rest) = ptr.strip_prefix("/system") {
        return system_frozen && (rest.is_empty() || rest.starts_with('/'));
    }
    if let Some(rest) = ptr.strip_prefix("/messages/") {
        let idx = rest
            .split('/')
            .next()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(usize::MAX);
        return frozen_until.is_some_and(|until| idx <= until);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::provider::for_kind;
    use serde_json::json;

    fn req(v: Value) -> Request {
        Request::from_value(ProviderKind::Anthropic, v)
    }

    #[test]
    fn no_markers_means_everything_compressible() {
        let r = req(json!({
            "messages": [
                {"role": "user", "content": "first turn"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second turn"},
            ]
        }));
        let p = for_kind(ProviderKind::Anthropic);
        assert!(frozen_pointers(&r, p.as_ref()).is_empty());
        assert_eq!(
            compressible_pointers(&r, p.as_ref()).len(),
            p.content_text_pointers(&r).len(),
            "no cache_control → all content compressible"
        );
    }

    #[test]
    fn cache_control_freezes_the_prefix_through_the_last_marker() {
        // Marker on message 1 → messages 0 and 1 frozen, message 2 (the live turn) free.
        let r = req(json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "cached A"}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "cached B", "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [{"type": "text", "text": "live turn"}]},
            ]
        }));
        let p = for_kind(ProviderKind::Anthropic);
        let comp = compressible_pointers(&r, p.as_ref());
        assert!(
            comp.iter().all(|x| x.starts_with("/messages/2")),
            "only the live turn: {comp:?}"
        );
        let frozen = frozen_pointers(&r, p.as_ref());
        assert!(frozen.contains("/messages/0/content/0/text"));
        assert!(frozen.contains("/messages/1/content/0/text"));
    }

    #[test]
    fn cache_controlled_system_is_frozen() {
        let r = req(json!({
            "system": [{"type": "text", "text": "stable instructions", "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": "ask"}],
        }));
        let p = for_kind(ProviderKind::Anthropic);
        let frozen = frozen_pointers(&r, p.as_ref());
        assert!(
            frozen.contains("/system/0/text"),
            "marked system is frozen: {frozen:?}"
        );
        // The (unmarked) user turn stays compressible.
        assert!(
            compressible_pointers(&r, p.as_ref())
                .iter()
                .any(|x| x.starts_with("/messages/0"))
        );
    }
}
