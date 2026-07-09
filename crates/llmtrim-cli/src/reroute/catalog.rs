//! Model catalog for the reroute mapping-editor TUI: the upstream models a `sub` provider can be
//! mapped to, each enriched with per-1M-token pricing when the models.dev snapshot knows it.
//!
//! Pure and offline: pricing comes from the `pricing.json` snapshot embedded at compile time
//! (the same file the bench uses). A missing price is never an error — it just shows blank in the
//! editor.

use once_cell::sync::Lazy;
use serde_json::Value;

use crate::reroute::{CODEX_MODELS, KIMI_MODEL, SubProvider};

/// The models.dev pricing snapshot, embedded at build time. Ids are provider-prefixed, e.g.
/// `"openai/gpt-5.5"` or `"moonshotai/kimi-k2.7-code"`.
static PRICING: Lazy<Value> = Lazy::new(|| {
    serde_json::from_str(include_str!("../../bench/pricing.json")).unwrap_or(Value::Null)
});

/// One selectable upstream model, with pricing when known.
pub struct CatalogEntry {
    pub id: String, // upstream model id to map TO (e.g. "gpt-5.5", "kimi-for-coding")
    pub input: Option<f64>, // $ per 1M input tokens, if known
    pub output: Option<f64>, // $ per 1M output tokens
    pub cache_read: Option<f64>,
}

/// The models a `sub` provider can be mapped to, best-first-ish, de-duplicated.
/// Codex: the curated [`CODEX_MODELS`] set (the ids the ChatGPT backend accepts), each enriched
///   with pricing looked up from `pricing.json`. Kimi: a single entry [`KIMI_MODEL`] enriched from
///   pricing if present.
pub fn models_for(provider: SubProvider) -> Vec<CatalogEntry> {
    match provider {
        SubProvider::Codex => CODEX_MODELS
            .iter()
            .map(|id| entry_for(id, &["openai/"]))
            .collect(),
        SubProvider::Kimi => {
            vec![entry_for(KIMI_MODEL, &["moonshotai/", "moonshot/"])]
        }
    }
}

/// Build a [`CatalogEntry`] for `id`, looking up pricing by trying, in order: each
/// `prefix + id`, the bare `id`, then a case-insensitive suffix match on the pricing map keys
/// (a key equal to `id` or ending in `/id`). All-`None` when nothing matches.
fn entry_for(id: &str, prefixes: &[&str]) -> CatalogEntry {
    let models = PRICING.get("models");
    let price = lookup(models, id, prefixes);
    CatalogEntry {
        id: id.to_string(),
        input: price.and_then(|p| field(p, "input")),
        output: price.and_then(|p| field(p, "output")),
        cache_read: price.and_then(|p| field(p, "cache_read")),
    }
}

/// Find the pricing object for `id` in the `models` map, or `None`.
fn lookup<'a>(models: Option<&'a Value>, id: &str, prefixes: &[&str]) -> Option<&'a Value> {
    let map = models?.as_object()?;
    // 1. prefix + id (e.g. "openai/gpt-5.5")
    for p in prefixes {
        if let Some(v) = map.get(&format!("{p}{id}")) {
            return Some(v);
        }
    }
    // 2. bare id
    if let Some(v) = map.get(id) {
        return Some(v);
    }
    // 3. case-insensitive suffix match: key == id or key ends with "/id"
    let want = id.to_ascii_lowercase();
    let want_slash = format!("/{want}");
    map.iter()
        .find(|(k, _)| {
            let k = k.to_ascii_lowercase();
            k == want || k.ends_with(&want_slash)
        })
        .map(|(_, v)| v)
}

/// Read a numeric field from a pricing object as `f64`, if present and numeric.
fn field(price: &Value, key: &str) -> Option<f64> {
    price.get(key).and_then(Value::as_f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_pricing_is_a_models_object() {
        let models = PRICING
            .get("models")
            .expect("pricing.json must have a `models` key");
        assert!(models.is_object(), "`models` must be an object");
        assert!(
            !models.as_object().unwrap().is_empty(),
            "`models` map must not be empty"
        );
    }

    #[test]
    fn codex_returns_exactly_curated_ids() {
        let ids: Vec<String> = models_for(SubProvider::Codex)
            .into_iter()
            .map(|e| e.id)
            .collect();
        assert_eq!(ids, CODEX_MODELS.to_vec());
    }

    #[test]
    fn kimi_returns_single_entry() {
        let entries = models_for(SubProvider::Kimi);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, KIMI_MODEL);
    }

    #[test]
    fn known_id_has_pricing_matching_snapshot() {
        // gpt-5.5 is keyed "openai/gpt-5.5" in the snapshot.
        let gpt55 = models_for(SubProvider::Codex)
            .into_iter()
            .find(|e| e.id == "gpt-5.5")
            .expect("gpt-5.5 in the Codex catalog");
        let expected = PRICING["models"]["openai/gpt-5.5"]["input"]
            .as_f64()
            .expect("gpt-5.5 input price in snapshot");
        assert_eq!(gpt55.input, Some(expected));
        assert!(gpt55.output.is_some());
        assert!(gpt55.cache_read.is_some());
    }

    #[test]
    fn bogus_id_yields_all_none() {
        let e = entry_for("totally-not-a-real-model-xyz", &["openai/"]);
        assert_eq!(e.input, None);
        assert_eq!(e.output, None);
        assert_eq!(e.cache_read, None);
    }
}
