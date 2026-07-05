//! Model-capability gate for the agent-loop frugality directive.
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
