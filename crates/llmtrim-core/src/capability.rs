//! Model-capability gates for the output-steering directives.
//!
//! Two independent signals, both data-driven snapshots (not hand-maintained model tables):
//! [`model_honors_steering`] gates the agent-loop frugality directive on an LMArena Elo bar, and
//! [`model_is_reasoning_capable`] gates the anti-overthink directive on the models.dev `reasoning`
//! flag. The frugality half is documented below; the reasoning half is on its own function.
//!
//! The frugality directive (see [`crate::stages::output`]) steers a tool-using agent toward the
//! fewest tool-use turns. Benches show only *capable* harnesses act on that system-level steer:
//! flagship models (Claude Opus/Sonnet 5, GPT-5-high, Gemini 3) batch and cap their exploration,
//! while cheap models (gpt-4o-mini, gpt-oss-20b) ignore it and just eat the directive's input
//! cost. So the directive is gated on a capability signal — inject only for models that clear a
//! bar, skip the rest.
//!
//! The signal is a static LMArena text-leaderboard snapshot (Elo), embedded at compile time. This
//! is a *data-driven snapshot* refreshed from an external source (`tools/refresh_lmarena.py`), not
//! a hand-maintained model-family table — the pattern the rest of the crate avoids. Refresh it on
//! release the same way `bench/pricing.json` is refreshed.
//!
//! Semantics are **opt-out**: an unknown model id injects. A brand-new flagship absent from the
//! snapshot still gets the steer (correct — it is capable); the only cost is a not-yet-listed weak
//! model eating the directive until the next snapshot. Matching a MISS to "inject" keeps the gate
//! from silently disabling the feature on the newest models, which is the case that matters most.

use std::collections::HashMap;

use once_cell::sync::Lazy;
use serde_json::Value;

/// LMArena text leaderboard (overall), 2026-07-02 snapshot. See module docs for provenance.
const LMARENA_SNAPSHOT: &str = include_str!("../data/lmarena_text.json");

/// Per-model `reasoning` flag from models.dev, embedded at compile time. Backs the
/// anti-overthink gate: refreshed by `tools/refresh_reasoning.py` on release.
const REASONING_SNAPSHOT: &str = include_str!("../data/model_reasoning.json");

/// `model_id` (lowercased) -> whether the model has a reasoning mode, from [`REASONING_SNAPSHOT`].
static REASONING: Lazy<HashMap<String, bool>> = Lazy::new(|| {
    let parsed: Value = serde_json::from_str(REASONING_SNAPSHOT)
        .expect("embedded reasoning snapshot is valid json");
    parsed["models"]
        .as_object()
        .expect("snapshot has a `models` object")
        .iter()
        .filter_map(|(name, flag)| Some((name.to_ascii_lowercase(), flag.as_bool()?)))
        .collect()
});

/// Per-model context window (tokens) from models.dev, embedded at compile time. Backs the breakdown
/// occupancy view: refreshed by `tools/refresh_context.py` on release.
const CONTEXT_SNAPSHOT: &str = include_str!("../data/model_context.json");

/// `model_id` (lowercased) -> context window in tokens, from [`CONTEXT_SNAPSHOT`].
static CONTEXT: Lazy<HashMap<String, u32>> = Lazy::new(|| {
    let parsed: Value =
        serde_json::from_str(CONTEXT_SNAPSHOT).expect("embedded context snapshot is valid json");
    parsed["models"]
        .as_object()
        .expect("snapshot has a `models` object")
        .iter()
        .filter_map(|(name, window)| {
            Some((
                name.to_ascii_lowercase(),
                u32::try_from(window.as_u64()?).ok()?,
            ))
        })
        .collect()
});

/// Elo bar: models rated strictly above obey the trajectory steer in our benches; the models
/// proven to ignore it sit well below (gpt-4o-mini 1287, gpt-oss-20b 1288, claude-haiku-4-5 1393),
/// so the bar has margin on the weak side while still admitting the GPT-5-high class (1405).
const CAPABILITY_THRESHOLD: u64 = 1400;

/// `model_name` (lowercased) -> Elo rating, parsed once from the embedded snapshot.
static RATINGS: Lazy<HashMap<String, u64>> = Lazy::new(|| {
    let parsed: Value =
        serde_json::from_str(LMARENA_SNAPSHOT).expect("embedded lmarena snapshot is valid json");
    parsed["models"]
        .as_object()
        .expect("snapshot has a `models` object")
        .iter()
        .filter_map(|(name, rating)| Some((name.to_ascii_lowercase(), rating.as_u64()?)))
        .collect()
});

/// Look up a wire model id's Elo, tolerating the id shapes providers actually send.
///
/// Normalize to lowercase and drop any `provider/` prefix (`openai/gpt-oss-20b` -> `gpt-oss-20b`),
/// then:
/// 1. exact match, else
/// 2. a *date-suffix* match — the leaderboard pins dated ids (`gpt-4o-mini-2024-07-18`) while the
///    wire often sends the bare id (`gpt-4o-mini`). Accept a key `{id}-{digit…}` so a bare id
///    resolves to its dated entry. The digit guard is deliberate: it matches a date/version suffix
///    but NOT a sibling model (`gpt-5` must not borrow `gpt-5.5-high`'s rating), so it can't
///    inflate a weak model into the capable band. Ties pick the highest-rated dated variant.
///
/// Returns `None` on a genuine miss (caller treats that as "inject", per the opt-out contract).
fn rating_for(model_id: &str) -> Option<u64> {
    let id = model_id.to_ascii_lowercase();
    let id = id.split_once('/').map_or(id.as_str(), |(_, rest)| rest);

    if let Some(&r) = RATINGS.get(id) {
        return Some(r);
    }
    let prefix = format!("{id}-");
    RATINGS
        .iter()
        .filter(|(k, _)| {
            k.strip_prefix(&prefix)
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        })
        .map(|(_, &r)| r)
        .max()
}

/// True when the model is capable enough to act on the frugality directive — i.e. its leaderboard
/// Elo clears [`CAPABILITY_THRESHOLD`], or it is unknown (opt-out: unknown models inject). An empty
/// / missing id is treated as unknown.
pub(crate) fn model_honors_steering(model_id: &str) -> bool {
    rating_for(model_id).is_none_or(|r| r > CAPABILITY_THRESHOLD)
}

/// True when the wire model id names a model with a reasoning mode, per the embedded models.dev
/// `reasoning` flag ([`REASONING_SNAPSHOT`]). This is the authoritative registry signal — a bare
/// id the harness sends (Claude Code sends `claude-sonnet-5`) resolves directly, no `-thinking`
/// guessing. Normalizes like [`rating_for`]: lowercase, drop a `provider/` prefix, then a
/// date-suffix fallback for a dated registry entry behind a bare wire id.
///
/// Semantics are **opt-in** (the opposite of [`model_honors_steering`]): a genuine miss returns
/// `false`. The anti-overthink directive should ride only models known to reason — injecting it on
/// a non-reasoning model is the wrong-behavior case, so an unlisted model gets no directive
/// ("no help", never "misapplied").
pub(crate) fn model_is_reasoning_capable(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    let id = id.split_once('/').map_or(id.as_str(), |(_, rest)| rest);

    if let Some(&flag) = REASONING.get(id) {
        return flag;
    }
    let prefix = format!("{id}-");
    REASONING
        .iter()
        .filter(|(k, _)| {
            k.strip_prefix(&prefix)
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()))
        })
        .any(|(_, &flag)| flag)
}

/// Context window (tokens) for a wire model id, from the embedded models.dev snapshot
/// ([`CONTEXT_SNAPSHOT`]). Normalizes the id (lowercase, drop a `provider/` prefix) and looks it up
/// directly — models.dev keys the registry by the bare ids the wire sends (`gpt-5.6-terra`,
/// `claude-opus-4-8`), so no date-suffix fallback is needed here (unlike the dated LMArena board).
/// `None` on a genuine miss, so the caller keeps its own default.
pub(crate) fn context_window_for(model_id: &str) -> Option<u32> {
    let id = model_id.to_ascii_lowercase();
    let id = id.split_once('/').map_or(id.as_str(), |(_, rest)| rest);
    CONTEXT.get(id).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weak_models_are_below_the_bar() {
        // The two models our live agent benches proved ignore the steer, plus other cheap tiers.
        for id in [
            "openai/gpt-oss-20b",
            "gpt-oss-20b",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o",
            "claude-3-5-haiku",
            "claude-3-5-sonnet-20241022",
        ] {
            assert!(!model_honors_steering(id), "{id} should be gated out");
        }
    }

    #[test]
    fn capable_models_clear_the_bar() {
        for id in [
            "claude-opus-4-8",
            "claude-opus-4-8-thinking",
            "claude-sonnet-4-6",
            "gpt-5.5",
            "gpt-5.5-high",
            "gemini-3-pro",
            "anthropic/claude-opus-4-6",
        ] {
            assert!(model_honors_steering(id), "{id} should be injected");
        }
    }

    #[test]
    fn reasoning_capability_read_from_models_dev_snapshot() {
        // Bare ids the harness sends (Claude Code sends `claude-sonnet-5`) resolve directly against
        // the models.dev `reasoning` flag — no `-thinking` guessing.
        for id in [
            "claude-sonnet-5",
            "anthropic/claude-sonnet-5", // provider prefix stripped
            "claude-opus-4-8",
            "gpt-5",
            "o3",
        ] {
            assert!(
                model_is_reasoning_capable(id),
                "{id} should read as reasoning-capable"
            );
        }
        // Registry says reasoning=false, or a genuine miss (opt-in) -> false.
        for id in [
            "gpt-4o-mini",
            "gpt-4o",
            "claude-3-5-haiku",
            "",
            "totally-unknown-model",
        ] {
            assert!(
                !model_is_reasoning_capable(id),
                "{id} must not read as reasoning-capable"
            );
        }
    }

    #[test]
    fn context_window_read_from_models_dev_snapshot() {
        // Real per-model windows, resolved from the id shapes the wire actually sends.
        for (id, window) in [
            ("gpt-5", 400_000),
            ("gpt-5-codex", 400_000),
            ("gpt-5.6-terra", 1_050_000),
            ("openai/gpt-5", 400_000), // provider prefix stripped
            ("gpt-4o", 128_000),
        ] {
            assert_eq!(context_window_for(id), Some(window), "{id}");
        }
        // A genuine miss returns None so the caller keeps its own default.
        for id in ["totally-unknown-model", ""] {
            assert_eq!(context_window_for(id), None, "{id}");
        }
    }

    #[test]
    fn unknown_and_empty_ids_inject() {
        // Opt-out: a brand-new flagship absent from the snapshot must still get the steer, and a
        // missing model field must not silently disable the feature.
        assert!(model_honors_steering("claude-sonnet-5")); // only `-thinking` is on the board
        assert!(model_honors_steering("some-model-shipped-tomorrow"));
        assert!(model_honors_steering(""));
    }

    #[test]
    fn date_suffix_match_resolves_a_bare_id_without_borrowing_a_sibling() {
        // The board has no bare `gpt-4o-mini` key, only the dated `gpt-4o-mini-2024-07-18` (1287).
        // The bare wire id must resolve to that dated entry via the date-suffix path (so it gates
        // out), and must NOT borrow a stronger sibling. This exercises the fallback the exact-match
        // branch would otherwise short-circuit.
        assert_eq!(RATINGS.get("gpt-4o-mini"), None); // no exact key -> forces the fallback
        assert_eq!(rating_for("gpt-4o-mini"), Some(1287));
        assert!(!model_honors_steering("gpt-4o-mini"));
    }

    #[test]
    fn snapshot_parses_and_is_populated() {
        assert!(RATINGS.len() > 100, "snapshot should carry the full board");
        // The anchor model must stay above the bar; a refresh dropping it below would silently
        // disable the gate's intent, so assert the threshold relationship, not a pinned number.
        assert!(RATINGS["claude-sonnet-5-thinking"] > CAPABILITY_THRESHOLD);
    }
}
