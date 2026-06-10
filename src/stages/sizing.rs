//! Adaptive keep-count (K) sizing — *how many* items to retain when pruning.
//!
//! Replaces a blind keep-ratio with an information-saturation estimate: keep items
//! until they stop adding new content, not a fixed fraction. Two deterministic
//! signals, both dependency-light (no model, no embeddings):
//!
//! 1. **Unique-bigram saturation (Kneedle).** Walk the items in order, tracking the
//!    cumulative set of distinct word-bigrams. Coverage rises fast then flattens once
//!    later items repeat earlier ones; the "knee" of that curve is the point past
//!    which extra items add little (Satopää et al. 2011). A curve with no knee
//!    (every item adds new bigrams) means the content is all-diverse → keep up to the
//!    budget; a sharp early knee → keep few.
//! 2. **SimHash dedup ceiling.** Near-duplicate items (gaoya 64-bit SimHash, the same
//!    fingerprint Stage E uses) collapse to one cluster; K is never set *above* the
//!    number of distinct clusters, so near-duplicate spam can't inflate the keep set.
//!
//! K is clamped to `[min_k, max_k]` and never exceeds the item count. This is a pure
//! function: the caller decides *which* items to keep (first/last/by-score); this
//! answers only *how many*.

use std::collections::HashSet;

use gaoya::simhash::SimHashBits;

use crate::stages::dedup::make_simhasher;

use crate::stages::tools::lex_words;

/// Minimum normalized deviation above the chord for a point to count as a knee.
/// Below this the curve is ~linear (no saturation) and we keep up to the budget.
const KNEE_MIN_DEV: f64 = 0.05;

/// SimHash Hamming distance under which two items are the "same" cluster. Matches the
/// conservative default Stage E uses for exact-ish near-duplicate lines.
const SIMHASH_NEAR_DIST: u32 = 3;

/// Decide how many of `items` to keep, between `min_k` and `max_k` (both clamped to
/// the item count). Picks the unique-bigram saturation knee, then caps that by the
/// number of distinct (non-near-duplicate) clusters so redundant items never enlarge
/// the keep set.
pub fn optimal_keep(items: &[&str], min_k: usize, max_k: usize) -> usize {
    let n = items.len();
    let lo = min_k.min(n);
    let hi = max_k.min(n);
    if n <= lo {
        return n;
    }
    if hi <= lo {
        return lo;
    }

    let curve = unique_bigram_curve(items);
    let knee = match knee_index(&curve) {
        // Keep through the knee item (1-based count).
        Some(i) => i + 1,
        // No clear knee: either no lexical signal (keep the floor) or an all-diverse
        // curve where every item earns its place (keep up to the budget).
        None if curve.last().copied().unwrap_or(0) == 0 => lo,
        None => hi,
    };

    // Distinct clusters is an upper bound on useful items — beyond it, items are
    // near-duplicates of ones already kept. Floor at `lo` so the minimum still holds
    // when everything is similar.
    let clusters = distinct_clusters(items).max(lo);

    knee.clamp(lo, hi).min(clusters)
}

/// Cumulative count of distinct word-bigrams after including each item, in order.
/// `curve[i]` is the size of the running bigram set through item `i`. Single-word
/// items contribute a unigram sentinel so they still register as content.
fn unique_bigram_curve(items: &[&str]) -> Vec<usize> {
    let mut seen: HashSet<u64> = HashSet::new();
    let mut curve = Vec::with_capacity(items.len());
    for item in items {
        let words = lex_words(item);
        match words.as_slice() {
            [] => {}
            [single] => {
                seen.insert(token_hash(single, ""));
            }
            _ => {
                for pair in words.windows(2) {
                    seen.insert(token_hash(&pair[0], &pair[1]));
                }
            }
        }
        curve.push(seen.len());
    }
    curve
}

/// Index of the saturation knee: the point of maximum normalized distance above the
/// chord from `(0,0)` to `(1,1)` on the cumulative-coverage curve (concave-increasing,
/// so the knee is the largest `y - x`). `None` when the curve is too short, has no
/// content, or never bows more than [`KNEE_MIN_DEV`] above the chord (≈ linear).
fn knee_index(curve: &[usize]) -> Option<usize> {
    let n = curve.len();
    if n < 3 {
        return None;
    }
    let total = *curve.last().unwrap() as f64;
    if total <= 0.0 {
        return None;
    }
    let x_span = (n - 1) as f64;
    let mut best = (0usize, 0.0f64);
    for (i, &c) in curve.iter().enumerate() {
        let x = i as f64 / x_span;
        let y = c as f64 / total;
        let deviation = y - x;
        if deviation > best.1 {
            best = (i, deviation);
        }
    }
    (best.1 > KNEE_MIN_DEV).then_some(best.0)
}

/// Number of distinct items by 64-bit SimHash, greedily clustering any item within
/// [`SIMHASH_NEAR_DIST`] Hamming bits onto an existing representative. Mirrors Stage
/// E's near-duplicate detection so the two stages agree on what "the same" means.
fn distinct_clusters(items: &[&str]) -> usize {
    let hasher = make_simhasher();
    let mut reps: Vec<u64> = Vec::new();
    for item in items {
        let words = lex_words(item);
        let sig = if words.is_empty() {
            0
        } else {
            hasher.create_signature(words.iter())
        };
        if reps
            .iter()
            .any(|&r| r.hamming_distance(&sig) <= SIMHASH_NEAR_DIST as usize)
        {
            continue;
        }
        reps.push(sig);
    }
    reps.len()
}

/// Non-cryptographic FNV-1a hash of an ordered token pair (unit-separated), so the
/// bigram set can be a `HashSet<u64>` without per-pair allocation. Matches the FNV-1a
/// used for the cache prefix fingerprint.
fn token_hash(a: &str, b: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in a.bytes().chain(std::iter::once(0x1f)).chain(b.bytes()) {
        h ^= byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` lexically distinct lines with *no word shared between lines* (every token
    /// carries the line index), so SimHash sees `n` separate clusters and every item
    /// adds fresh bigrams — a ~linear coverage curve with no knee.
    fn diverse(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                (0..6)
                    .map(|j| format!("w{i}x{j}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect()
    }

    fn as_refs(v: &[String]) -> Vec<&str> {
        v.iter().map(String::as_str).collect()
    }

    #[test]
    fn returns_all_when_under_floor() {
        let items = ["a", "b"];
        assert_eq!(optimal_keep(&items, 5, 10), 2, "n below min_k keeps all");
        assert_eq!(optimal_keep(&[], 1, 10), 0, "empty keeps none");
    }

    #[test]
    fn diverse_content_keeps_up_to_budget() {
        let v = diverse(20);
        let k = optimal_keep(&as_refs(&v), 3, 12);
        assert_eq!(k, 12, "all-diverse curve has no knee → keep the max budget");
    }

    #[test]
    fn near_duplicate_spam_is_capped_by_clusters() {
        // One distinct line repeated 20×: a single SimHash cluster caps K low,
        // regardless of a generous max budget.
        let v: Vec<String> =
            std::iter::repeat_n("WARN cache miss for user session".to_string(), 20).collect();
        let k = optimal_keep(&as_refs(&v), 2, 15);
        assert!(
            k <= 3,
            "20 near-identical lines collapse to ~1 cluster, got {k}"
        );
    }

    #[test]
    fn early_saturation_keeps_fewer_than_diverse() {
        // Rich prefix, then redundant tail: knee lands early, so K is well under the
        // budget and under the all-diverse case.
        let mut v = diverse(4);
        v.extend(std::iter::repeat_n(
            "retry pending retry pending retry".to_string(),
            16,
        ));
        let saturating = optimal_keep(&as_refs(&v), 2, 18);
        let all_diverse = optimal_keep(&as_refs(&diverse(20)), 2, 18);
        assert!(
            saturating < all_diverse,
            "saturating set ({saturating}) keeps fewer than diverse ({all_diverse})"
        );
    }

    #[test]
    fn never_exceeds_item_count() {
        let v = diverse(6);
        assert!(optimal_keep(&as_refs(&v), 2, 100) <= 6);
    }

    #[test]
    fn knee_on_concave_curve_is_near_the_elbow() {
        // Fast rise (0,5,9,12) then flat (13,13,13,13): the elbow sits at the
        // transition, not at either end.
        let curve = [0usize, 5, 9, 12, 13, 13, 13, 13];
        let knee = knee_index(&curve).expect("a concave curve has a knee");
        assert!(
            (2..=4).contains(&knee),
            "knee at the elbow region, got {knee}"
        );
    }

    #[test]
    fn linear_curve_has_no_knee() {
        let curve = [0usize, 2, 4, 6, 8, 10];
        assert_eq!(knee_index(&curve), None, "a straight line has no knee");
    }
}
