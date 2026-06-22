//! Token-budgeted submodular selection — pick a high-value, low-redundancy subset of
//! items that fits a token budget. The shared engine behind Stage B retrieval chunk
//! selection and Stage's JSON-row diverse sampling.
//!
//! ## Objective (monotone submodular)
//!
//! For a selected set `S`, with a tradeoff `λ ∈ [0,1]`:
//!
//! ```text
//! F(S) = λ · Σ_{i∈S} rel(i)
//!      + (1−λ) · Σ_b min( cov_S(b),  α · totalW(b) )
//! ```
//!
//! where `b` ranges over the corpus's word-**bigrams**, `cov_S(b)` is the weight of `b`
//! accumulated across the items in `S`, `totalW(b)` is `b`'s weight summed over *all*
//! items, and `α ∈ (0,1]` caps how much any one feature can pay off. The first term is
//! the **modular relevance** (linear in `S`); the second is the **saturating coverage**
//! of Lin & Bilmes ("A Class of Submodular Functions for Document Summarization", ACL
//! 2011) — once a bigram is covered to its ceiling, re-covering it (a near-duplicate
//! item) earns nothing, so diversity falls out of the objective rather than a separate
//! penalty. `F` is monotone submodular, so the cost-scaled greedy below has a constant-
//! factor guarantee.
//!
//! ## Algorithm — lazy (CELF) cost-ratio greedy under a knapsack
//!
//! The token budget is a **knapsack** constraint. We greedily take the item maximizing
//! marginal-gain **per token** — `Δ(i | S) / cost(i)` — the cost-ratio rule of
//! "Revisiting Modified Greedy Algorithm for Monotone Submodular Maximization with a
//! Knapsack Constraint" (arXiv:2008.05391), which carries a 0.405 guarantee.
//!
//! Recomputing every item's marginal each round is `O(rounds · n · features)`.
//! Submodularity makes marginals **non-increasing** as `S` grows, so a stale gain is a
//! valid *upper bound*: we keep a max-heap of `(stale_ratio, idx)`, pop the top, and if
//! its gain was computed against the current `S` it is the true best (CELF; the lazy-
//! greedy reuse Chen et al., "Fast Greedy MAP Inference for DPP", NeurIPS 2018, also rely
//! on). Otherwise we refresh just that one item and push it back. Ties (equal ratio) break
//! by **original index**, so selection is fully deterministic.
//!
//! ## Coverage representation
//!
//! Coverage is tracked as a `bigram → covered-weight` map (`O(n · avg_bigrams)` memory),
//! never an `n×n` similarity matrix — the marginal of one item touches only *its own*
//! bigrams. Tokenization is the crate's shared Unicode word splitter (`lex_words`,
//! UAX#29), so it is script-universal, not English-only.

use std::collections::HashMap;

use crate::stages::tools::{fnv1a, lex_words};

/// One candidate for selection: its token cost, modular relevance, and the lexical
/// features (word-bigrams) used for saturating coverage / redundancy.
pub struct Item {
    /// Token cost charged against the budget when this item is selected.
    pub cost: usize,
    /// Modular relevance score (e.g. a normalized BM25 rank or query overlap). Higher is
    /// better; need not be normalized, but mixing scales across items distorts the λ blend.
    pub relevance: f64,
    /// FNV-1a hashes of the item's word-bigrams (its coverage features). Built by
    /// [`Item::from_text`]; a caller may supply its own feature hashes instead.
    pub features: Vec<u64>,
}

impl Item {
    /// Build an item from raw text: cost is caller-supplied (the real tokenizer count),
    /// relevance is caller-supplied, features are the text's distinct word-bigrams
    /// (Unicode-segmented). A single-word item contributes one unigram sentinel so it
    /// still has a feature; empty text has none (it can only ever add relevance).
    pub fn from_text(text: &str, cost: usize, relevance: f64) -> Self {
        Item {
            cost,
            relevance,
            features: bigram_hashes(text),
        }
    }
}

/// Tuning weights for [`select`]. `lambda` blends modular relevance against saturating
/// coverage (`1.0` = pure relevance/top-k-like, `0.0` = pure coverage/diversity);
/// `saturation` is the Lin-Bilmes `α` — the fraction of a feature's total corpus weight
/// at which it stops paying off (lower = stronger diversity pressure).
#[derive(Clone, Copy)]
pub struct Weights {
    pub lambda: f64,
    pub saturation: f64,
}

impl Default for Weights {
    /// Balanced relevance/coverage with a half-saturation ceiling — a feature seen in
    /// half the items it occurs in is "covered enough".
    fn default() -> Self {
        Weights {
            lambda: 0.5,
            saturation: 0.5,
        }
    }
}

/// Distinct word-bigram hashes of `text` (FNV-1a over each ordered, unit-separated word
/// pair), deduplicated and in first-seen order. A single word yields one unigram
/// sentinel; no words yields none. Mirrors `sizing::unique_bigram_curve` so the two
/// modules agree on what a "feature" is.
fn bigram_hashes(text: &str) -> Vec<u64> {
    let words = lex_words(text);
    let mut seen = Vec::new();
    let mut set = std::collections::HashSet::new();
    let mut push = |h: u64, seen: &mut Vec<u64>| {
        if set.insert(h) {
            seen.push(h);
        }
    };
    match words.as_slice() {
        [] => {}
        [single] => push(token_hash(single, ""), &mut seen),
        _ => {
            for pair in words.windows(2) {
                push(token_hash(&pair[0], &pair[1]), &mut seen);
            }
        }
    }
    seen
}

/// FNV-1a of an ordered token pair (unit-separated). Identical to `sizing`'s hashing so
/// the bigram space is shared and allocation-free.
fn token_hash(a: &str, b: &str) -> u64 {
    fnv1a(a.bytes().chain(std::iter::once(0x1f)).chain(b.bytes()))
}

/// Select a budget-fitting subset of `items` maximizing the Lin-Bilmes objective via
/// lazy (CELF) cost-ratio greedy. Returns the chosen indices in **ascending original
/// order** (the set, not the selection order — callers emit in document order).
///
/// - `budget`: total token cost the selection may not exceed.
/// - `weights`: λ / α tuning (see [`Weights`]).
///
/// Items costlier than the whole budget are skipped. An item with zero marginal value
/// (no relevance, all features already saturated) is never added — selection stops once
/// nothing affordable improves `F`, so it won't pad the set with filler just because
/// budget remains. Deterministic: equal cost-ratios break by original index.
pub fn select(items: &[Item], budget: usize, weights: &Weights) -> Vec<usize> {
    let n = items.len();
    if n == 0 || budget == 0 {
        return Vec::new();
    }

    // Per-feature saturation ceiling = α · (total weight of the feature across all items).
    // Weight is presence (1 per item that has the feature), matching the unweighted bigram
    // sets above; `cap[b]` is therefore α · (number of items containing b).
    let mut total: HashMap<u64, f64> = HashMap::new();
    for it in items {
        for &f in &it.features {
            *total.entry(f).or_insert(0.0) += 1.0;
        }
    }
    let cap: HashMap<u64, f64> = total
        .into_iter()
        .map(|(f, t)| (f, weights.saturation * t))
        .collect();

    // Covered weight so far, per feature (the `cov_S` accumulator).
    let mut covered: HashMap<u64, f64> = HashMap::new();
    let mut chosen: Vec<usize> = Vec::new();
    let mut picked = vec![false; n];
    let mut spent = 0usize;

    // CELF heap of stale upper bounds: (cost-ratio, valid_at, idx). `valid_at` is the
    // selection count the ratio was computed against; an entry is fresh iff it equals the
    // current count. `HeapEntry`'s ordering gives a deterministic total order (ratio desc,
    // then idx asc), so `BinaryHeap` pops the true best-first.
    let mut heap: std::collections::BinaryHeap<HeapEntry> = items
        .iter()
        .enumerate()
        .filter(|(_, it)| it.cost > 0 && it.cost <= budget)
        .map(|(i, it)| {
            let gain = marginal(it, &covered, &cap, weights);
            HeapEntry::new(gain / it.cost as f64, 0, i)
        })
        .collect();

    let mut selections = 0u64;
    while let Some(top) = heap.pop() {
        let i = top.idx;
        if picked[i] {
            continue;
        }
        let it = &items[i];
        // Doesn't fit the remaining budget any more → drop it permanently.
        if spent + it.cost > budget {
            continue;
        }
        if top.valid_at == selections {
            // Fresh marginal ⇒ this is the true best affordable item. A non-positive gain
            // means nothing left improves the objective: stop (don't pad with filler).
            if top.ratio <= 0.0 {
                break;
            }
            for &f in &it.features {
                let c = covered.entry(f).or_insert(0.0);
                *c = (*c + 1.0).min(cap.get(&f).copied().unwrap_or(0.0));
            }
            picked[i] = true;
            chosen.push(i);
            spent += it.cost;
            selections += 1;
        } else {
            // Stale ⇒ refresh against the current S and re-insert (CELF lazy re-eval).
            let gain = marginal(it, &covered, &cap, weights);
            heap.push(HeapEntry::new(gain / it.cost as f64, selections, i));
        }
    }

    chosen.sort_unstable();
    chosen
}

/// Marginal gain of adding `it` to the current selection: the modular relevance term plus
/// the increase in saturating coverage from its features. `O(it.features)`.
fn marginal(
    it: &Item,
    covered: &HashMap<u64, f64>,
    cap: &HashMap<u64, f64>,
    weights: &Weights,
) -> f64 {
    let cov_gain: f64 = it
        .features
        .iter()
        .map(|f| {
            let ceiling = cap.get(f).copied().unwrap_or(0.0);
            let now = covered.get(f).copied().unwrap_or(0.0);
            // min(now+1, ceiling) − min(now, ceiling): the unsaturated headroom this item
            // fills (0 once the feature is already at its ceiling).
            (now + 1.0).min(ceiling) - now.min(ceiling)
        })
        .sum();
    weights.lambda * it.relevance + (1.0 - weights.lambda) * cov_gain
}

/// A CELF heap entry. Ordered by cost-ratio descending, ties broken by **ascending**
/// original index, so `BinaryHeap` (a max-heap) yields a deterministic best-first order.
struct HeapEntry {
    ratio: f64,
    valid_at: u64,
    idx: usize,
}

impl HeapEntry {
    fn new(ratio: f64, valid_at: u64, idx: usize) -> Self {
        HeapEntry {
            ratio,
            valid_at,
            idx,
        }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher ratio first. NaN can't arise (finite costs > 0, finite gains), but treat
        // any incomparable pair as equal rather than panicking. Tie-break: SMALLER index
        // is "greater" so it pops first from the max-heap (stable, original-order ties).
        self.ratio
            .partial_cmp(&other.ratio)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(other.idx.cmp(&self.idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Token cost proxy used in tests (whitespace words), matching the crate's
    /// `count_tokens` test convention. Production passes the real tokenizer count.
    fn cost(text: &str) -> usize {
        text.split_whitespace().count().max(1)
    }

    fn item(text: &str, relevance: f64) -> Item {
        Item::from_text(text, cost(text), relevance)
    }

    #[test]
    fn respects_the_token_budget() {
        // Five 4-token items, budget 10 ⇒ at most two fit.
        let items: Vec<Item> = (0..5)
            .map(|i| item(&format!("alpha{i} beta{i} gamma{i} delta{i}"), 1.0))
            .collect();
        let picked = select(&items, 10, &Weights::default());
        let spent: usize = picked.iter().map(|&i| items[i].cost).sum();
        assert!(spent <= 10, "selection cost {spent} exceeds budget 10");
        assert!(!picked.is_empty(), "something fits a budget of 10");
    }

    #[test]
    fn empty_and_degenerate_inputs() {
        assert!(select(&[], 100, &Weights::default()).is_empty(), "no items");
        let items = vec![item("hello world", 1.0)];
        assert!(
            select(&items, 0, &Weights::default()).is_empty(),
            "zero budget selects nothing"
        );
        // An item bigger than the whole budget is skipped, not force-fit.
        let big = vec![Item::from_text("a b c d e", 5, 1.0)];
        assert!(
            select(&big, 3, &Weights::default()).is_empty(),
            "an over-budget item can't be selected"
        );
    }

    #[test]
    fn returns_ascending_original_order() {
        let items: Vec<Item> = (0..6)
            .map(|i| item(&format!("token{i}a token{i}b token{i}c"), 1.0))
            .collect();
        let picked = select(&items, 100, &Weights::default());
        let mut sorted = picked.clone();
        sorted.sort_unstable();
        assert_eq!(picked, sorted, "indices come back in ascending order");
    }

    #[test]
    fn saturating_coverage_prefers_diverse_over_redundant() {
        // Three items at EQUAL relevance: 0 and 1 are near-duplicates (same bigrams), 2 is
        // novel. With a tight budget that fits exactly two, pure top-k would be ambiguous;
        // the coverage term must break it toward {0, 2} (or {1,2}) — never {0,1}.
        let items = vec![
            item(
                "quarterly revenue for logistics was four million dollars",
                1.0,
            ),
            item(
                "quarterly revenue for logistics was four million dollars",
                1.0,
            ),
            item(
                "parking available north visitor lot near south elevators",
                1.0,
            ),
        ];
        // All three are 8 words; a budget of 16 fits exactly two of them.
        let picked = select(
            &items,
            16,
            &Weights {
                lambda: 0.3,
                saturation: 0.5,
            },
        );
        assert!(picked.contains(&2), "the novel item must be selected");
        assert!(
            !(picked.contains(&0) && picked.contains(&1)),
            "the two near-duplicates must not both be chosen: {picked:?}"
        );
    }

    #[test]
    fn pure_relevance_lambda_one_is_cost_ratio_top_k() {
        // λ=1 ⇒ coverage ignored; selection is by relevance-per-token. Item 2 has the
        // highest relevance and should be picked first / included under a tight budget.
        let items = vec![
            item("aaa bbb ccc", 1.0),
            item("ddd eee fff", 2.0),
            item("ggg hhh iii", 9.0),
        ];
        let picked = select(
            &items,
            3,
            &Weights {
                lambda: 1.0,
                saturation: 0.5,
            },
        );
        assert_eq!(
            picked,
            vec![2],
            "highest relevance-per-token wins under λ=1"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        // Same input, repeated selection ⇒ byte-identical result (heap tie-break is stable).
        let items: Vec<Item> = (0..20)
            .map(|i| {
                let rel = ((i * 7) % 5) as f64; // a few ties
                item(&format!("word{}a shared common token{}b", i % 3, i), rel)
            })
            .collect();
        let first = select(&items, 25, &Weights::default());
        for _ in 0..5 {
            assert_eq!(
                select(&items, 25, &Weights::default()),
                first,
                "selection must be deterministic"
            );
        }
    }

    #[test]
    fn stops_instead_of_padding_with_zero_value_filler() {
        // Two identical items, λ=0 (pure coverage). The first covers all bigrams; the
        // second adds zero marginal coverage and zero relevance ⇒ must NOT be selected
        // even though the budget easily fits it.
        let items = vec![
            item("alpha beta gamma delta epsilon", 0.0),
            item("alpha beta gamma delta epsilon", 0.0),
        ];
        let picked = select(
            &items,
            100,
            &Weights {
                lambda: 0.0,
                saturation: 0.5,
            },
        );
        assert_eq!(
            picked.len(),
            1,
            "no zero-value duplicate padding: {picked:?}"
        );
    }

    #[test]
    fn lazy_greedy_matches_naive_greedy() {
        // CELF must select exactly what an eager recompute-everything greedy would. Cross-
        // check on a mixed instance with varied costs, relevances, and overlaps.
        let texts = [
            ("red green blue", 3.0),
            ("red green blue extra", 1.0),
            ("yellow orange purple", 2.0),
            ("yellow orange", 0.5),
            ("cyan magenta", 4.0),
            ("cyan magenta black white", 2.5),
        ];
        let items: Vec<Item> = texts.iter().map(|(t, r)| item(t, *r)).collect();
        let w = Weights {
            lambda: 0.4,
            saturation: 0.6,
        };
        let lazy = select(&items, 9, &w);
        let eager = naive_greedy(&items, 9, &w);
        assert_eq!(lazy, eager, "CELF output diverged from naive greedy");
    }

    /// Reference cost-ratio greedy: recompute every affordable item's marginal each round,
    /// take the best (ties → lowest index), stop on non-positive gain. Same objective as
    /// [`select`], used only to validate the lazy version.
    fn naive_greedy(items: &[Item], budget: usize, weights: &Weights) -> Vec<usize> {
        let mut total: HashMap<u64, f64> = HashMap::new();
        for it in items {
            for &f in &it.features {
                *total.entry(f).or_insert(0.0) += 1.0;
            }
        }
        let cap: HashMap<u64, f64> = total
            .into_iter()
            .map(|(f, t)| (f, weights.saturation * t))
            .collect();
        let mut covered: HashMap<u64, f64> = HashMap::new();
        let mut chosen = Vec::new();
        let mut picked = vec![false; items.len()];
        let mut spent = 0usize;
        loop {
            let mut best: Option<(f64, usize)> = None;
            for (i, it) in items.iter().enumerate() {
                if picked[i] || it.cost == 0 || spent + it.cost > budget {
                    continue;
                }
                let ratio = marginal(it, &covered, &cap, weights) / it.cost as f64;
                if best.is_none_or(|(br, _)| ratio > br) {
                    best = Some((ratio, i));
                }
            }
            match best {
                Some((ratio, i)) if ratio > 0.0 => {
                    for &f in &items[i].features {
                        let c = covered.entry(f).or_insert(0.0);
                        *c = (*c + 1.0).min(cap.get(&f).copied().unwrap_or(0.0));
                    }
                    picked[i] = true;
                    chosen.push(i);
                    spent += items[i].cost;
                }
                _ => break,
            }
        }
        chosen.sort_unstable();
        chosen
    }
}
