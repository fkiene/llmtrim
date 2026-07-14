//! Subscription reroute: send intercepted Anthropic `/v1/messages` traffic to a *different*
//! subscription's backend (ChatGPT/Codex or Kimi) instead of Anthropic, translating the request
//! and streamed response between wire shapes.
//!
//! This is opt-in (`sub = "codex"|"kimi"` in the config, off by default) and rides the existing
//! MITM path: [`crate::serve`] rewrites the intercepted request's URI authority to the provider
//! host and swaps in the translated body + provider auth, so hudsucker forwards it over the same
//! client and `handle_response` streams the translated reply back. Nothing here opens its own
//! socket except the one-time OAuth flows in [`auth`].
//!
//! **Terms of service:** driving a ChatGPT/Kimi *subscription* through a non-official client is a
//! gray area and can get that account restricted. Reroute is off by default and the `auth login`
//! commands print this warning. Use at your own risk.

pub mod auth;
pub mod catalog;
pub mod codex;
pub mod continuation;
pub mod kimi;
pub mod read_rewrite;
pub mod sse;
#[cfg(feature = "breakdown")]
pub mod tui;

use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::{Value, json};

/// The rewritten upstream request: how [`crate::serve`] should retarget the intercepted Anthropic
/// request at the provider. The caller sets the request URI authority to `host`, path to `path`,
/// replaces the body with `body`, and applies `headers` (after stripping the client's Anthropic
/// auth). `model` is the resolved upstream model (for the ledger + the response reducer).
pub struct UpstreamRewrite {
    pub host: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub model: String,
    pub provider: SubProvider,
}

/// Translate an intercepted Anthropic `/v1/messages` body into a provider-targeted request, using
/// the resolved subscription token. Pure except that model resolution reads `overrides`.
pub fn build_upstream(
    provider: SubProvider,
    anthropic_body: &Value,
    overrides: &BTreeMap<String, String>,
    token: &auth::TokenSet,
    session_id: Option<&str>,
) -> Result<UpstreamRewrite> {
    build_upstream_for_model(provider, anthropic_body, None, overrides, token, session_id)
}

/// Internal variant that resolves a compact candidate without changing the client-visible model in
/// the Anthropic body. The public wrapper above retains its stable five-argument API.
pub(crate) fn build_upstream_for_model(
    provider: SubProvider,
    anthropic_body: &Value,
    logical_model: Option<&str>,
    overrides: &BTreeMap<String, String>,
    token: &auth::TokenSet,
    session_id: Option<&str>,
) -> Result<UpstreamRewrite> {
    let incoming = logical_model.unwrap_or_else(|| {
        anthropic_body
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
    });
    let model = resolve_model(provider, incoming, overrides);
    let (host, path, body, headers) = match provider {
        SubProvider::Codex => {
            let b = codex::build_request_body(anthropic_body, &model, session_id)?;
            let h = codex::request_headers_with_mode(
                &token.access,
                token.account_id.as_deref(),
                session_id,
                codex::uses_official_codex_client(&model),
            );
            (codex::HOST, codex::PATH, serde_json::to_vec(&b)?, h)
        }
        SubProvider::Kimi => {
            let b = kimi::build_request_body(anthropic_body, &model, session_id)?;
            let h = kimi::request_headers(&token.access, token.account_id.as_deref(), session_id);
            (kimi::HOST, kimi::PATH, serde_json::to_vec(&b)?, h)
        }
    };
    Ok(UpstreamRewrite {
        host: host.to_string(),
        path: path.to_string(),
        headers,
        body,
        model,
        provider,
    })
}

/// Provider-dispatching wrapper over the two [`sse`] reducers, so [`crate::serve`] can hold one
/// reducer regardless of provider and stream the translated Anthropic SSE incrementally.
pub enum StreamReducer {
    Codex(codex::Reducer),
    Kimi(kimi::Reducer),
}

impl StreamReducer {
    pub fn new(provider: SubProvider, model: &str) -> Self {
        match provider {
            SubProvider::Codex => StreamReducer::Codex(codex::Reducer::new(model)),
            SubProvider::Kimi => StreamReducer::Kimi(kimi::Reducer::new(model)),
        }
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<sse::ReduceEvent> {
        match self {
            StreamReducer::Codex(r) => r.push(chunk),
            StreamReducer::Kimi(r) => r.push(chunk),
        }
    }

    pub fn finish(&mut self) -> Vec<sse::ReduceEvent> {
        match self {
            StreamReducer::Codex(r) => r.finish(),
            StreamReducer::Kimi(r) => r.finish(),
        }
    }

    /// For Codex continuation: take the assistant output items (text + function calls)
    /// accumulated during reduction for this turn, to append to the transcript for next-turn
    /// delta detection. Safe no-op for Kimi.
    pub fn take_codex_output_items(&mut self) -> Vec<Value> {
        match self {
            StreamReducer::Codex(r) => r.take_output_items(),
            StreamReducer::Kimi(_) => vec![],
        }
    }
}

/// Answer Claude Code's `/v1/messages/count_tokens` locally (the request is being billed against
/// the sub provider, not Anthropic, so we can't proxy it). Returns the Anthropic-shaped
/// `{"input_tokens": N}` JSON. Deliberately biased to *over*-estimate: an undercount makes Claude
/// Code compact late and overflow the real upstream context window. We count the concatenated text
/// with the Anthropic tokenizer and add a small per-message + safety margin.
pub fn count_tokens_json(anthropic_body: &Value) -> Value {
    let mut text = String::new();
    if let Some(system) = anthropic_body.get("system") {
        collect_text(system, &mut text);
    }
    let mut messages = 0usize;
    if let Some(arr) = anthropic_body.get("messages").and_then(|v| v.as_array()) {
        messages = arr.len();
        for m in arr {
            if let Some(c) = m.get("content") {
                collect_text(c, &mut text);
            }
        }
    }
    if let Some(tools) = anthropic_body.get("tools") {
        collect_text(tools, &mut text);
    }
    let model = anthropic_body.get("model").and_then(|v| v.as_str());
    let base = count_text_tokens(&text, model);
    // +4 tokens/message envelope, then a 10% safety margin (never undercount).
    let est = (((base + messages as i64 * 4) as f64) * 1.1).ceil() as i64;
    json!({ "input_tokens": est })
}

/// Recursively harvest every string leaf's text from an Anthropic content value into `out`.
fn collect_text(v: &Value, out: &mut String) {
    match v {
        Value::String(s) => {
            out.push_str(s);
            out.push(' ');
        }
        Value::Array(a) => a.iter().for_each(|e| collect_text(e, out)),
        Value::Object(o) => o.values().for_each(|e| collect_text(e, out)),
        _ => {}
    }
}

/// Token count of `text` via llmtrim's Anthropic tokenizer; falls back to a chars/3.5 estimate
/// (which over-counts typical English/code) if a counter can't be built.
fn count_text_tokens(text: &str, model: Option<&str>) -> i64 {
    use llmtrim_core::ir::ProviderKind;
    match llmtrim_core::tokenizer::counter_for(ProviderKind::Anthropic, model) {
        Ok(counter) => counter.count(text) as i64,
        Err(_) => (text.chars().count() as f64 / 3.5).ceil() as i64,
    }
}

/// Which subscription backend intercepted Anthropic traffic is rerouted to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubProvider {
    Codex,
    Kimi,
}

impl SubProvider {
    /// Parse the `sub` config value. `None` for unset/`off`/unknown (the serve layer logs an
    /// unknown value once; unknown never silently reroutes).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "codex" | "chatgpt" | "openai" => Some(SubProvider::Codex),
            "kimi" | "moonshot" => Some(SubProvider::Kimi),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SubProvider::Codex => "codex",
            SubProvider::Kimi => "kimi",
        }
    }
}

/// The four Claude model tiers Claude Code selects between (plus the background small/fast tier,
/// `haiku`). Every incoming model id classifies into one of these; the tier maps to a concrete
/// provider model via the preset + user overrides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Opus,
    Sonnet,
    Haiku,
    Fable,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Opus => "opus",
            Tier::Sonnet => "sonnet",
            Tier::Haiku => "haiku",
            Tier::Fable => "fable",
        }
    }

    pub const ALL: [Tier; 4] = [Tier::Opus, Tier::Sonnet, Tier::Haiku, Tier::Fable];
}

/// Classify an incoming Anthropic model id into a [`Tier`] by substring, matching both the family
/// aliases (`opus`) and the dated ids (`claude-opus-4-8`). `None` when the id names no known tier
/// (e.g. it is already a concrete provider id like `gpt-5.5`, or a future/unknown Claude id).
pub fn classify_tier(model: &str) -> Option<Tier> {
    let m = model.to_ascii_lowercase();
    // Order matters only in that each keyword is unique across tiers.
    if m.contains("opus") {
        Some(Tier::Opus)
    } else if m.contains("sonnet") {
        Some(Tier::Sonnet)
    } else if m.contains("haiku") {
        Some(Tier::Haiku)
    } else if m.contains("fable") {
        Some(Tier::Fable)
    } else {
        None
    }
}

/// The built-in default preset for Codex (the single `balanced` preset). Opus maps to the deep
/// reasoning flagship (`gpt-5.6-terra`), Sonnet to the balanced flagship (`gpt-5.6-luna`), Fable
/// to the fast flagship (`gpt-5.6-sol`), and Haiku to the small/fast model (Claude Code's
/// background title/token calls land on Haiku, so it should be cheap). Kimi has one model and
/// ignores tiers.
pub fn default_codex_tier_model(tier: Tier) -> &'static str {
    match tier {
        Tier::Opus => "gpt-5.6-terra",
        Tier::Sonnet => "gpt-5.6-luna",
        Tier::Haiku => "gpt-5.4-mini",
        Tier::Fable => "gpt-5.6-sol",
    }
}

/// Kimi exposes a single wire model; every tier and alias collapses to it.
pub const KIMI_MODEL: &str = "kimi-for-coding";

/// Resolve the incoming Anthropic model id to the upstream provider model for `provider`, applying
/// (in order): an exact-id override, then the tier's override, then the preset default. `overrides`
/// is [`llmtrim_core::config::RuntimeConfig::sub_tiers`] (keys lowercased: tier names or exact ids).
///
/// - A model already in concrete provider form (no tier keyword) with no exact override passes
///   through unchanged after `[1m]`/`-fast` normalization by the caller — here it falls back to the
///   `sonnet` tier only if it also isn't a provider id. See [`normalize_incoming`].
pub fn resolve_model(
    provider: SubProvider,
    incoming: &str,
    overrides: &std::collections::BTreeMap<String, String>,
) -> String {
    let (base, _fast) = normalize_incoming(incoming);
    if provider == SubProvider::Kimi {
        return KIMI_MODEL.to_string();
    }
    let key = base.to_ascii_lowercase();
    // 1. Exact-id override.
    if let Some(m) = overrides.get(&key) {
        return m.clone();
    }
    // 2. Tier override, then preset default.
    if let Some(tier) = classify_tier(&base) {
        if let Some(m) = overrides.get(tier.as_str()) {
            return m.clone();
        }
        return default_codex_tier_model(tier).to_string();
    }
    // 3. Not a Claude tier: if it already looks like a Codex model, pass it through; otherwise fall
    //    back to the sonnet tier (conservative, always-works default).
    if is_codex_model(&base) {
        return base;
    }
    overrides
        .get(Tier::Sonnet.as_str())
        .cloned()
        .unwrap_or_else(|| default_codex_tier_model(Tier::Sonnet).to_string())
}

/// System text blocks beginning with this marker are Claude Code billing metadata smuggled as a
/// system block, not prompt content. Its `cch=` field changes every turn, so forwarding it poisons
/// the head of the provider's cached prefix and every turn pays a cold prompt cache.
pub const BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

/// Flatten an Anthropic `system` (string or block array) into one string for providers that take a
/// single system/instructions field, dropping [`BILLING_HEADER_PREFIX`] blocks. `None` when nothing
/// survives.
pub fn flatten_system_text(system: Option<&Value>) -> Option<String> {
    let texts: Vec<&str> = match system? {
        Value::String(s) => vec![s.as_str()],
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect(),
        _ => return None,
    };
    let kept: Vec<&str> = texts
        .into_iter()
        .filter(|t| !t.starts_with(BILLING_HEADER_PREFIX) && !t.is_empty())
        .collect();
    if kept.is_empty() {
        None
    } else {
        Some(kept.join("\n\n"))
    }
}

/// The Codex models the ChatGPT backend actually accepts. A model outside this set 400s upstream;
/// the caller can still send it (forward-compat) but the mapping editor picks from this list.
pub const CODEX_MODELS: [&str; 9] = [
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.5",
];

fn is_codex_model(m: &str) -> bool {
    let m = m.to_ascii_lowercase();
    CODEX_MODELS.contains(&m.as_str()) || m.starts_with("gpt-")
}

/// Strip Claude Code's local `[1m]` context-window hint and a trailing `-fast` service-tier
/// suffix, returning `(base_model, fast_requested)`. The `[1m]` suffix is a client-only compaction
/// hint and must never reach upstream; `-fast` maps to a priority service tier, not a model id.
pub fn normalize_incoming(model: &str) -> (String, bool) {
    let mut m = model.trim();
    // `[1m]` is case-insensitive per Claude Code.
    if m.len() >= 4 && m[m.len() - 4..].eq_ignore_ascii_case("[1m]") {
        m = m[..m.len() - 4].trim();
    }
    if let Some(base) = m.strip_suffix("-fast") {
        (base.to_string(), true)
    } else {
        (m.to_string(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn classify_handles_aliases_and_dated_ids() {
        assert_eq!(classify_tier("opus"), Some(Tier::Opus));
        assert_eq!(classify_tier("claude-opus-4-8"), Some(Tier::Opus));
        assert_eq!(classify_tier("claude-sonnet-5"), Some(Tier::Sonnet));
        assert_eq!(
            classify_tier("claude-haiku-4-5-20251001"),
            Some(Tier::Haiku)
        );
        assert_eq!(classify_tier("claude-fable-5"), Some(Tier::Fable));
        assert_eq!(classify_tier("gpt-5.5"), None);
    }

    #[test]
    fn normalize_strips_1m_and_fast() {
        assert_eq!(normalize_incoming("gpt-5.5[1m]"), ("gpt-5.5".into(), false));
        assert_eq!(normalize_incoming("gpt-5.5[1M]"), ("gpt-5.5".into(), false));
        assert_eq!(
            normalize_incoming("gpt-5.4-mini-fast"),
            ("gpt-5.4-mini".into(), true)
        );
        assert_eq!(
            normalize_incoming("claude-opus-4-8[1m]"),
            ("claude-opus-4-8".into(), false)
        );
    }

    #[test]
    fn default_preset_maps_fable_to_flagship() {
        let ov = BTreeMap::new();
        assert_eq!(
            resolve_model(SubProvider::Codex, "claude-opus-4-8", &ov),
            "gpt-5.6-terra"
        );
        assert_eq!(
            resolve_model(SubProvider::Codex, "sonnet", &ov),
            "gpt-5.6-luna"
        );
        assert_eq!(
            resolve_model(SubProvider::Codex, "claude-sonnet-5", &ov),
            "gpt-5.6-luna"
        );
        assert_eq!(
            resolve_model(SubProvider::Codex, "haiku", &ov),
            "gpt-5.4-mini"
        );
        assert_eq!(
            resolve_model(SubProvider::Codex, "claude-fable-5", &ov),
            "gpt-5.6-sol"
        );
    }

    #[test]
    fn overrides_win_over_preset() {
        let mut ov = BTreeMap::new();
        ov.insert("sonnet".to_string(), "gpt-5.3-codex".to_string());
        assert_eq!(
            resolve_model(SubProvider::Codex, "claude-sonnet-5", &ov),
            "gpt-5.3-codex"
        );
        // exact-id override beats the tier
        ov.insert("claude-sonnet-5".to_string(), "gpt-5.5".to_string());
        assert_eq!(
            resolve_model(SubProvider::Codex, "claude-sonnet-5", &ov),
            "gpt-5.5"
        );
    }

    #[test]
    fn kimi_collapses_every_tier() {
        let ov = BTreeMap::new();
        assert_eq!(
            resolve_model(SubProvider::Kimi, "claude-opus-4-8", &ov),
            KIMI_MODEL
        );
        assert_eq!(resolve_model(SubProvider::Kimi, "gpt-5.5", &ov), KIMI_MODEL);
    }

    #[test]
    fn concrete_codex_model_passes_through() {
        let ov = BTreeMap::new();
        assert_eq!(
            resolve_model(SubProvider::Codex, "gpt-5.3-codex", &ov),
            "gpt-5.3-codex"
        );
        assert_eq!(
            resolve_model(SubProvider::Codex, "gpt-5.5[1m]", &ov),
            "gpt-5.5"
        );
    }

    #[test]
    fn unknown_non_codex_falls_back_to_sonnet() {
        let ov = BTreeMap::new();
        assert_eq!(
            resolve_model(SubProvider::Codex, "mystery-model", &ov),
            "gpt-5.6-luna"
        );
    }
}
