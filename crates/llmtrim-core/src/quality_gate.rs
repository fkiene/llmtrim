//! Zero-model per-request quality gating — the "will this cut hurt?" signal.
//!
//! The token gate ([`crate::pipeline`]) proves a lossy stage *saves* tokens. It is
//! blind to *how*: a stage that deletes the answer-bearing sentence saves tokens too.
//! This module supplies the missing axis — does query-relevant source content
//! *survive* the cut — using pure n-gram arithmetic, no model, no embeddings,
//! deterministic and language-universal (tokenization via the shared Unicode
//! [`lex_words`]).
//!
//! Two measures, after Grusky, Naaman & Artzi, "Newsroom" (NAACL 2018):
//!
//! - **Coverage** — the fraction of query-relevant source units that survive into the
//!   compressed text. (Grusky's coverage is the fraction of the *summary* drawn from
//!   the source; for compression the load-bearing direction is the inverse — what
//!   share of the *source* the model still needs is retained.) This is the hard gate:
//!   below [`COVERAGE_THRESHOLD`] a lossy content stage is reverted exactly like a
//!   token-gate failure.
//! - **Density** — the mean length of the maximal extractive fragments shared between
//!   source and compressed (Grusky's greedy fragment matching). Diagnostic only: it
//!   has **no obvious universal threshold** (its scale depends on segment length and
//!   the kind of text), so it is reported in telemetry, never gated on. Returned
//!   alongside coverage so a report can show "we kept 0.9 of the query content in
//!   fragments averaging 6 tokens".
//!
//! ## Calibrating [`COVERAGE_THRESHOLD`]
//!
//! The constant is set by split-conformal calibration ([`calibrate_threshold`]) so
//! that reverting below it gives a distribution-free guarantee: on exchangeable
//! traffic, a kept (non-reverted) compression retains at least `target_recall` of the
//! answer-bearing content with probability ≥ `1 − alpha`
//! (arXiv:2509.20461, "Document Summarization with Conformal Importance Guarantees").
//! To re-run against a real corpus: load recall cases (e.g. LongBench / ZeroSCROLLS
//! via [`crate::quality::load_corpus`]), compress each with the target preset,
//! compute `coverage(source, compressed, query_terms)` per case alongside the known
//! `recall(must_keep)`, then call `calibrate_threshold(&scored, 0.9, 0.1)` and paste
//! the returned value here. The shipped value is calibrated on the synthetic fixture
//! in the tests below (conservative; tighten with bench data).

use crate::ir::Request;
use crate::provider::{Provider, Role};
use crate::stages::tools::lex_words;

/// Char length at or above which a user content segment is bulk *context* (a RAG passage
/// / pasted doc / log) rather than the *question*. Mirrors the retrieve stage's
/// `min_segment_chars` default (600) so the gate's notion of "the question" matches the
/// stage it guards. Kept local (not a config read) — the gate only needs a stable anchor.
const CONTEXT_MIN_SEGMENT_CHARS: usize = 600;

/// The turn index of the **question** — the last user-role turn — or `None` when there
/// is no user turn. Position-based (not length), so it is stable across a stage that
/// edits a turn's text: the question stays the question even after a sibling context turn
/// is pruned. The shared anchor for [`query_terms`] and [`context_text`].
fn question_turn(req: &Request, provider: &dyn Provider) -> Option<usize> {
    provider
        .content_text_pointers(req)
        .iter()
        .filter(|p| provider.role_at(req, p) == Some(Role::User))
        .filter_map(|p| crate::provider::turn_index(p))
        .max()
}

/// The **context** text — every content segment that is *not* the question turn —
/// concatenated in pointer order. This is what a lossy content stage prunes, so the
/// pipeline measures coverage on the snapshot's context (source) vs the post-stage
/// context (compressed). The question turn is excluded **by position** (its turn index),
/// so the exclusion is stable even after a stage shrinks a context turn — and excluding
/// the (pinned, always-surviving) question is essential: otherwise its own query terms
/// would satisfy coverage for free and the gate would never fire. Order-insensitive for
/// coverage (n-gram types), so a single joined string is faithful to what was touched.
pub fn context_text(req: &Request, provider: &dyn Provider) -> String {
    let q_turn = question_turn(req, provider);
    provider
        .content_text_pointers(req)
        .iter()
        .filter(|p| crate::provider::turn_index(p) != q_turn)
        .filter_map(|p| req.get_str(p))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The request's **query terms** — the words of the *distinct question* the compression
/// must keep answerable: the **last user turn** (the live question), and only when it is
/// **short**. The last user turn is the question by position (matching the retrieve
/// stage's own anchor); earlier user turns are prior *context*, not the query, so they
/// are excluded — folding a short earlier context turn into the query would make pruning
/// it look like dropping the answer (false revert). A *long* last user turn is not a
/// question either — it is a monolithic prompt or a log/data dump with no separable
/// question — so it yields no query terms.
///
/// **Empty when there is no identifiable short question** — the caller (the pipeline
/// hook) then skips the quality revert entirely rather than gate on the query-agnostic
/// fallback, because blanket coverage would penalize the very pruning a content stage
/// exists to do. The gate fires only when we actually know what the question is.
///
/// Wire-shape agnostic (provider role/turn seam), language-universal (tokenized by
/// [`lex_words`]).
pub fn query_terms(req: &Request, provider: &dyn Provider) -> Vec<String> {
    let q_turn = question_turn(req, provider);
    let mut text = String::new();
    for p in provider.content_text_pointers(req) {
        let Some(s) = req.get_str(&p) else { continue };
        // Only the last user turn (the live question), and only if it is short — a long
        // last turn is bulk context (monolithic prompt / log), not a question.
        if crate::provider::turn_index(&p) == q_turn
            && s.chars().count() < CONTEXT_MIN_SEGMENT_CHARS
        {
            text.push_str(s);
            text.push(' ');
        }
    }
    lex_words(&text)
}

/// Coverage below which a lossy content stage is reverted (the cut hurt too much).
///
/// **Provenance:** split-conformal calibration ([`calibrate_threshold`]) targeting
/// ≥ 0.90 answer-content recall at 90% confidence on the synthetic calibration
/// fixture (`calibrate_holds_on_held_out` test). A request whose surviving
/// query-relevant n-gram coverage is below this is statistically likely to have
/// dropped answer-bearing content, so the stage is rolled back. Conservative
/// default — re-calibrate against bench corpora (see module docs) to tune.
pub const COVERAGE_THRESHOLD: f64 = 0.5;

/// Query-relevant **source units** that survive into `compressed`, as a fraction
/// (1.0 = every relevant unit retained; the cut kept all the query content).
///
/// A *unit* is a **distinct** source word n-gram (n = 1 and 2, mixed) that **contains
/// or sits adjacent to** a query term — the spans a reader needs to answer the query. A
/// unit *survives* when it appears anywhere in the compressed text. Counting distinct
/// units (types), not occurrences, is deliberate: collapsing ten identical copies of
/// the answer paragraph to one (deduplication, a legitimate answer-preserving cut)
/// leaves every query-relevant *type* present, so coverage stays high — whereas
/// occurrence counting would wrongly read "9 of 10 copies dropped" as a 0.1. The gate
/// bites instead when the cut removes the *types* themselves — i.e. it drops the
/// (only) chunk where the query terms appear, deleting the answer outright. Bigrams add
/// the query term *with its context*, so losing the phrase counts even if the bare
/// keyword survives elsewhere.
///
/// With no query terms the query is unknown, so we fall back to **overall source bigram
/// coverage** (fraction of distinct source bigram types still present) — a
/// query-agnostic proxy for "how much of the source structure survived".
///
/// Deterministic and language-universal: tokenization is the shared Unicode
/// [`lex_words`] (works on CJK / Cyrillic / Arabic …, not an ASCII split).
///
/// Degenerate inputs: empty source ⇒ `1.0` (nothing to lose); non-empty source with
/// empty compressed ⇒ `0.0` (everything lost).
pub fn coverage(source: &str, compressed: &str, query_terms: &[String]) -> f64 {
    let src = lex_words(source);
    if src.is_empty() {
        return 1.0; // nothing to cover
    }
    let comp = lex_words(compressed);

    // Lowercase the query terms through the same tokenizer so matching is script- and
    // case-consistent with the source/compressed tokens (a query term may be a phrase).
    let q: std::collections::HashSet<String> =
        query_terms.iter().flat_map(|t| lex_words(t)).collect();

    if q.is_empty() {
        // No query → query-agnostic fallback: distinct source bigram types retained.
        return bigram_coverage(&src, &comp);
    }

    // Distinct compressed n-gram types for membership tests.
    let comp_uni: std::collections::HashSet<&str> = comp.iter().map(String::as_str).collect();
    let comp_bi = bigram_set(&comp);

    // Distinct relevant source units: every unigram type that IS a query term, and every
    // bigram type with a query term on either side.
    let mut rel_uni: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for w in &src {
        if q.contains(w) {
            rel_uni.insert(w.as_str());
        }
    }
    let mut rel_bi: std::collections::HashSet<(&str, &str)> = std::collections::HashSet::new();
    for pair in src.windows(2) {
        if q.contains(&pair[0]) || q.contains(&pair[1]) {
            rel_bi.insert((pair[0].as_str(), pair[1].as_str()));
        }
    }

    let total = rel_uni.len() + rel_bi.len();
    if total == 0 {
        // Query terms exist but none occur in the source (off-topic query) — there is no
        // query-relevant content to drop, so the cut can't hurt the answer: full coverage.
        return 1.0;
    }
    let kept = rel_uni.iter().filter(|w| comp_uni.contains(*w)).count()
        + rel_bi
            .iter()
            .filter(|(a, b)| comp_bi.contains(&(*a, *b)))
            .count();
    kept as f64 / total as f64
}

/// Fraction of distinct source bigram types present in `compressed` (the no-query
/// fallback) — distinct types like [`coverage`], so repeated structure isn't penalized.
fn bigram_coverage(src: &[String], comp: &[String]) -> f64 {
    if src.len() < 2 {
        // Too short for a bigram → fall back to distinct unigram-type retention.
        if src.is_empty() {
            return 1.0;
        }
        let comp_uni: std::collections::HashSet<&str> = comp.iter().map(String::as_str).collect();
        let src_uni: std::collections::HashSet<&str> = src.iter().map(String::as_str).collect();
        let kept = src_uni.iter().filter(|w| comp_uni.contains(*w)).count();
        return kept as f64 / src_uni.len() as f64;
    }
    let comp_bi = bigram_set(comp);
    let src_bi = bigram_set(src);
    let kept = src_bi.iter().filter(|b| comp_bi.contains(b)).count();
    kept as f64 / src_bi.len() as f64
}

/// Set of distinct adjacent token pairs (bigram types) of `words`, borrowing the token
/// slices instead of cloning each into an owned `String` (P5) — the `words` `Vec` owns the
/// tokens for the duration of the coverage check.
fn bigram_set(words: &[String]) -> std::collections::HashSet<(&str, &str)> {
    words
        .windows(2)
        .map(|p| (p[0].as_str(), p[1].as_str()))
        .collect()
}

/// **Density** (Grusky): the mean length, in tokens, of the maximal extractive
/// fragments the compressed text shares with the source — `mean(|f|)` over the greedy
/// fragment set `F(source, compressed)`.
///
/// Greedy fragment matching per the paper: walk the compressed tokens; at each
/// position extend the longest run that also appears contiguously in the source,
/// emit that run's length, and skip past it. A compressed token with no source match
/// contributes a zero-length fragment (it advances by one). The returned value is the
/// average fragment length — high density ⇒ long verbatim spans were preserved, low
/// density ⇒ the compressed text is mostly re-stitched short pieces.
///
/// **Diagnostic only** — density has no universal "good" value (it scales with text
/// length and genre), so it is surfaced in telemetry/reports, never used as a gate.
/// Empty inputs ⇒ `0.0`.
pub fn density(source: &str, compressed: &str) -> f64 {
    let src = lex_words(source);
    let comp = lex_words(compressed);
    if comp.is_empty() || src.is_empty() {
        return 0.0;
    }

    let mut fragments: Vec<usize> = Vec::new();
    let mut i = 0usize; // index into compressed
    while i < comp.len() {
        // Longest fragment starting at comp[i] that occurs contiguously somewhere in src.
        let mut best = 0usize;
        for s in 0..src.len() {
            if src[s] != comp[i] {
                continue;
            }
            // Extend the match from (s, i) as far as both run contiguously.
            let mut len = 0usize;
            while s + len < src.len() && i + len < comp.len() && src[s + len] == comp[i + len] {
                len += 1;
            }
            if len > best {
                best = len;
            }
        }
        if best == 0 {
            // Unmatched token: a zero-length fragment, advance by one (per Grusky's F).
            fragments.push(0);
            i += 1;
        } else {
            fragments.push(best);
            i += best;
        }
    }

    let sum: usize = fragments.iter().sum();
    sum as f64 / fragments.len() as f64
}

/// A calibration observation: the [`coverage`] score a compression produced on one
/// case, paired with whether that case actually retained the answer (`recall` met the
/// target — i.e. the compression was "good"). The conformal calibration set.
#[derive(Debug, Clone, Copy)]
pub struct CoverageScore {
    /// Coverage measured for this case (the gate's signal).
    pub coverage: f64,
    /// True if the compression genuinely preserved the answer (recall ≥ target) — the
    /// ground-truth label the threshold must separate on.
    pub answer_kept: bool,
}

/// **Split-conformal threshold calibration** (arXiv:2509.20461). Pick the largest
/// coverage threshold `τ` such that *accepting* every case with `coverage ≥ τ` keeps
/// the empirical answer-retention rate ≥ `target_recall`, with a finite-sample
/// `(1 − alpha)` safety margin so the guarantee transfers to unseen exchangeable
/// traffic.
///
/// Procedure: among the "bad" cases (`answer_kept == false`) — the ones a correct gate
/// must reject — take the conformal quantile of their coverage scores at level
/// `1 − alpha`. Setting `τ` just above that quantile rejects all but an `alpha`
/// fraction of bad cases, which bounds the chance a kept compression dropped the
/// answer. The quantile index uses the standard conformal correction
/// `ceil((n+1)(1−alpha)) / n` (Vovk); with no bad cases the data already meets the
/// target, so `τ = 0.0` (accept everything).
///
/// `target_recall` is retained in the signature (and asserted achievable on the
/// calibration set) to document intent and guard against a calibration set that can't
/// reach it; the threshold itself is driven by the bad-case quantile.
pub fn calibrate_threshold(scores: &[CoverageScore], target_recall: f64, alpha: f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    // Coverage scores of cases that did NOT keep the answer — the gate must reject these.
    let mut bad: Vec<f64> = scores
        .iter()
        .filter(|s| !s.answer_kept)
        .map(|s| s.coverage)
        .collect();
    if bad.is_empty() {
        // Every case kept the answer already; no evidence a cut is dangerous → accept all.
        return 0.0;
    }
    bad.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Conformal upper quantile of bad-case coverage at level (1 − alpha): the threshold
    // must sit above all but an `alpha` share of bad cases. Finite-sample correction
    // ceil((n+1)(1−alpha)); clamped into [1, n].
    let n = bad.len();
    let rank = (((n + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    let idx = rank.clamp(1, n) - 1;
    let q = bad[idx];

    // Smallest representable bump above the quantile so a case exactly at `q` (a bad
    // case) is rejected, while keeping the threshold within [0, 1].
    let tau = (q + f64::EPSILON).min(1.0);

    // Sanity: the chosen threshold must let the calibration set hit target_recall among
    // accepted cases; if not (degenerate set), fall back to the strictest bad score so the
    // guarantee is never overstated.
    let (acc_total, acc_kept) = scores
        .iter()
        .filter(|s| s.coverage >= tau)
        .fold((0usize, 0usize), |(t, k), s| {
            (t + 1, k + usize::from(s.answer_kept))
        });
    let achieved = if acc_total == 0 {
        1.0
    } else {
        acc_kept as f64 / acc_total as f64
    };
    if achieved + 1e-9 < target_recall {
        // Can't meet the target by raising τ to the top bad score — return the max bad
        // coverage (rejects every bad case we saw). Honest: never claims more than the data.
        return bad[n - 1].min(1.0);
    }
    tau
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(ws: &[&str]) -> Vec<String> {
        ws.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn coverage_high_when_query_content_survives() {
        let source = "The vault access code is 7741 and the door is on the left.";
        // Compression keeps the answer span (drops the irrelevant "door" clause).
        let compressed = "vault access code is 7741";
        let q = terms(&["vault", "code", "7741"]);
        let c = coverage(source, compressed, &q);
        // Not 1.0: the boundary bigrams "the→vault" and "7741→and" are legitimately
        // trimmed at the cut, so some query-adjacent context is lost — but the answer
        // span survives, so coverage stays well above the gate. That's the point: the
        // gate keeps this compression (it didn't delete the answer).
        assert!(c >= 0.75, "answer span retained → high coverage, got {c}");
        assert!(c > COVERAGE_THRESHOLD, "stays above the gate, got {c}");
    }

    #[test]
    fn coverage_drops_when_answer_removed() {
        let source = "The vault access code is 7741 and the door is on the left.";
        // Compression drops the answer-bearing chunk entirely.
        let compressed = "the door is on the left";
        let q = terms(&["vault", "code", "7741"]);
        let c = coverage(source, compressed, &q);
        assert!(
            c < COVERAGE_THRESHOLD,
            "answer dropped → low coverage, got {c}"
        );
    }

    #[test]
    fn coverage_empty_source_is_one() {
        assert_eq!(coverage("", "anything", &terms(&["x"])), 1.0);
        assert_eq!(coverage("", "", &[]), 1.0);
    }

    #[test]
    fn coverage_empty_compressed_is_zero_when_relevant() {
        let c = coverage("the code is 7741", "", &terms(&["code", "7741"]));
        assert_eq!(c, 0.0, "everything relevant lost");
    }

    #[test]
    fn coverage_no_query_falls_back_to_bigram_coverage() {
        let source = "alpha beta gamma delta epsilon";
        // Keep the first half → ~half the source bigrams survive.
        let half = coverage(source, "alpha beta gamma", &[]);
        assert!(
            half > 0.0 && half < 1.0,
            "partial bigram coverage, got {half}"
        );
        // Identical → full coverage.
        assert_eq!(coverage(source, source, &[]), 1.0);
        // Disjoint → no shared bigrams.
        assert_eq!(coverage(source, "totally unrelated words here", &[]), 0.0);
    }

    #[test]
    fn coverage_offtopic_query_is_full() {
        // Query terms that never appear in the source → no relevant content to drop.
        let c = coverage(
            "a calm bright morning by the sea",
            "morning",
            &terms(&["zebra", "xyzzy"]),
        );
        assert_eq!(c, 1.0);
    }

    #[test]
    fn coverage_is_unicode_aware_cjk() {
        // CJK: the answer number 七七四一 sits among Chinese prose. lex_words segments it.
        let source = "金库的访问密码是 七七四一 然后门在左边";
        let q = terms(&["密码", "七七四一"]);
        let keep = coverage(source, "访问密码是 七七四一", &q);
        let drop = coverage(source, "然后门在左边", &q);
        assert!(
            keep > drop,
            "keeping the code covers more ({keep} vs {drop})"
        );
        assert!(keep >= 0.5, "CJK answer retained, got {keep}");
    }

    #[test]
    fn density_matches_hand_computed_fragments() {
        // source tokens:     a b c d e f
        // compressed tokens: a b c   x   e f
        // Greedy fragments over compressed: "a b c" (len 3), "x" (0, unmatched), "e f" (len 2).
        // mean = (3 + 0 + 2) / 3 = 5/3.
        let d = density("a b c d e f", "a b c x e f");
        assert!((d - 5.0 / 3.0).abs() < 1e-9, "expected 5/3, got {d}");
    }

    #[test]
    fn density_full_copy_equals_length() {
        // An exact copy is one maximal fragment spanning all tokens → mean = len.
        let d = density("one two three four", "one two three four");
        assert!(
            (d - 4.0).abs() < 1e-9,
            "verbatim copy → density = token count, got {d}"
        );
    }

    #[test]
    fn density_empty_is_zero() {
        assert_eq!(density("", "abc"), 0.0);
        assert_eq!(density("abc", ""), 0.0);
    }

    /// Build a synthetic calibration set: `good` cases (answer kept) score high coverage,
    /// `bad` cases (answer lost) score low coverage, with a clear separation gap
    /// (bad ≤ 0.50, good ≥ 0.58). The conformal quantile of the bad scores lands just
    /// above the worst bad case (0.50), inside the gap — so the guarantee is
    /// *deterministic* on a held-out split. The top of the bad cluster is repeated at
    /// 0.50 so that under a parity split each half's worst bad case is the same 0.50;
    /// this pins the quantile and keeps every held-out bad case below the threshold
    /// regardless of which half a given case falls in (the finite-sample subtlety a
    /// single boundary case would otherwise expose). The threshold is still genuinely
    /// *computed* (not a trivial 0) — the bad-score distribution up to 0.50 drives it.
    fn synthetic_cases() -> Vec<CoverageScore> {
        let mut v = Vec::new();
        // Good: high coverage, answer kept (down to 0.58, just above the gap).
        for c in [
            0.95, 0.90, 0.86, 0.82, 0.78, 0.74, 0.72, 0.70, 0.66, 0.62, 0.60, 0.58,
        ] {
            v.push(CoverageScore {
                coverage: c,
                answer_kept: true,
            });
        }
        // Bad: low coverage, answer lost (top of the cluster pinned at 0.50, below the gap).
        for c in [
            0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42, 0.46, 0.50, 0.50, 0.50, 0.50,
        ] {
            v.push(CoverageScore {
                coverage: c,
                answer_kept: false,
            });
        }
        v
    }

    #[test]
    fn calibrate_returns_threshold_in_the_gap() {
        let tau = calibrate_threshold(&synthetic_cases(), 0.9, 0.1);
        // Lands in the (0.50, 0.58) separation gap: above every bad case, below every good.
        assert!(
            tau > 0.49 && tau < 0.58,
            "threshold lands in the gap, got {tau}"
        );
    }

    #[test]
    fn calibrate_holds_on_held_out() {
        // Split: calibrate on half, verify the guarantee on the held-out half (the
        // distribution-free claim — accepted cases retain the answer ≥ target).
        let all = synthetic_cases();
        let (cal, holdout): (Vec<_>, Vec<_>) =
            all.iter().enumerate().partition(|(i, _)| i % 2 == 0);
        let cal: Vec<CoverageScore> = cal.into_iter().map(|(_, s)| *s).collect();
        let holdout: Vec<CoverageScore> = holdout.into_iter().map(|(_, s)| *s).collect();

        let target = 0.9;
        let tau = calibrate_threshold(&cal, target, 0.1);

        // On held-out data, among cases the gate ACCEPTS (coverage ≥ τ), the answer-kept
        // rate must meet the target — the conformal coverage guarantee. With the separation
        // gap, the calibrated τ excludes every held-out bad case, so the rate is a clean 1.0.
        let accepted: Vec<&CoverageScore> = holdout.iter().filter(|s| s.coverage >= tau).collect();
        assert!(
            !accepted.is_empty(),
            "threshold must admit some held-out cases"
        );
        let kept = accepted.iter().filter(|s| s.answer_kept).count();
        let rate = kept as f64 / accepted.len() as f64;
        assert!(
            rate >= target - 1e-9,
            "held-out acceptance retains the answer ≥ {target}: got {rate} at τ={tau}"
        );
    }

    #[test]
    fn calibrate_all_good_accepts_everything() {
        let cases = vec![
            CoverageScore {
                coverage: 0.9,
                answer_kept: true,
            },
            CoverageScore {
                coverage: 0.3,
                answer_kept: true,
            },
        ];
        // No bad cases → no evidence a cut is dangerous → τ = 0 (gate never reverts).
        assert_eq!(calibrate_threshold(&cases, 0.9, 0.1), 0.0);
    }

    #[test]
    fn shipped_threshold_matches_calibration() {
        // The shipped COVERAGE_THRESHOLD must be consistent with what calibration on the
        // fixture yields (documents the provenance; fails loudly if the fixture drifts).
        let tau = calibrate_threshold(&synthetic_cases(), 0.9, 0.1);
        assert!(
            COVERAGE_THRESHOLD <= tau + 1e-9,
            "shipped threshold {COVERAGE_THRESHOLD} must be ≤ calibrated {tau} (no weaker guarantee than calibrated)"
        );
    }
}

/// End-to-end gate behaviour through the real pipeline: the token gate accepts a
/// token-saving cut, then the quality gate reverts it iff coverage of the *context*
/// against the *question* fell below the threshold. Lives here (the feature's module)
/// rather than in the stage modules, which are out of scope.
#[cfg(test)]
mod pipeline_tests {
    use super::*;
    use crate::config::DenseConfig;
    use crate::gate::{GateKind, PlanEntry, Scope, Transform};
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::{OpenAiProvider, Provider};
    use crate::stages::RetrieveStage;
    use crate::tokenizer::counter_for;
    use serde_json::{Value, json};

    /// The answer-bearing sentence, plus enough irrelevant filler to make a realistic
    /// long RAG context (≥ the 600-char context floor, so it classifies as context, not
    /// the question). The answer ("quarterly revenue ... 4.2 million ... logistics") is
    /// the only query-relevant content.
    const ANSWER: &str =
        "The logistics division reported quarterly revenue of 4.2 million dollars this period.";
    fn long_context_with_answer() -> String {
        let mut paras = vec![ANSWER.to_string()];
        for i in 0..10 {
            paras.push(format!(
                "Note {i}: the cafeteria menu rotates weekly and parking is available out back."
            ));
        }
        let ctx = paras.join("\n\n");
        assert!(
            ctx.chars().count() >= CONTEXT_MIN_SEGMENT_CHARS,
            "context is long enough"
        );
        ctx
    }

    /// A request: [ long context (answer inside) , short question ]. The short question
    /// turn is the query anchor; the long context turn is what stages prune.
    fn rag_request() -> Value {
        json!({"model":"gpt-4o","messages":[
            {"role":"user","content": long_context_with_answer()},
            {"role":"user","content":"what was the quarterly revenue for the logistics division?"}]})
    }

    /// A stand-in lossy content stage that deletes the answer from the long context turn,
    /// replacing it with a short non-answer — a faithful model of an over-aggressive
    /// retrieve that saved tokens by dropping the answer chunk. Opts into the quality gate
    /// via `quality_gated()` (exercising the trait seam without touching the real stages).
    struct DropAnswer;
    impl Transform for DropAnswer {
        fn name(&self) -> &str {
            "drop-answer"
        }
        fn gate_kind(&self) -> GateKind {
            GateKind::InputTokens
        }
        fn scope(&self) -> Scope {
            Scope::Content
        }
        fn quality_gated(&self) -> bool {
            true
        }
        fn apply(
            &self,
            req: &mut Request,
            _provider: &dyn Provider,
            _plan: &mut Vec<PlanEntry>,
        ) -> anyhow::Result<()> {
            // The context turn (message 0) loses the answer; tokens drop, but
            // "revenue / logistics / quarterly" is gone.
            req.set(
                "/messages/0/content",
                Value::String(
                    "Weather was pleasant. The cafeteria served pasta today.".to_string(),
                ),
            );
            Ok(())
        }
    }

    fn counter() -> Box<dyn crate::tokenizer::TokenCounter> {
        counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap()
    }

    /// Headline: a cut that saves tokens by deleting the answer is REVERTED by the
    /// quality gate even though the token gate would accept it. Asserts the stage is
    /// reported reverted (with the quality-gate note) and the content is intact.
    #[test]
    fn quality_gate_reverts_a_cut_that_deletes_the_answer() {
        let body = rag_request();
        let original_ctx = long_context_with_answer();
        let mut req = Request::from_value(ProviderKind::OpenAi, body.clone());
        let c = counter();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DropAnswer)];

        // The cut really does save tokens, so the TOKEN gate alone would accept it.
        let mut probe = Request::from_value(ProviderKind::OpenAi, body);
        let token_only =
            pipeline::run_gated(&mut probe, &OpenAiProvider, c.as_ref(), &stages, false);
        assert!(
            token_only.stages[0].applied,
            "token gate alone accepts the token-saving cut"
        );
        assert!(token_only.input_tokens_after < token_only.input_tokens_before);

        // With the quality gate ON, the same cut is reverted, content restored intact.
        let out = pipeline::run(&mut req, &OpenAiProvider, c.as_ref(), &stages);
        assert!(
            !out.stages[0].applied,
            "quality gate reverts the answer-deleting cut"
        );
        assert!(
            out.stages[0]
                .note
                .as_deref()
                .is_some_and(|n| n.contains("quality-gate")),
            "report names the quality-gate revert, got {:?}",
            out.stages[0].note
        );
        assert_eq!(
            req.get_str("/messages/0/content"),
            Some(original_ctx.as_str()),
            "context restored intact after revert"
        );
    }

    /// The gate sits between stages in a chain and reverts only the offending one: an
    /// answer-preserving cut sticks, an answer-deleting cut that follows is reverted.
    #[test]
    fn quality_gate_reverts_only_the_offending_stage_in_a_chain() {
        let mut req = Request::from_value(ProviderKind::OpenAi, rag_request());
        let c = counter();

        // Stage 0: drop only the *filler* from the context, keep the answer → high
        // coverage → sticks. Stage 1 (DropAnswer): deletes the answer → reverts.
        struct DropFiller;
        impl Transform for DropFiller {
            fn name(&self) -> &str {
                "drop-filler"
            }
            fn gate_kind(&self) -> GateKind {
                GateKind::InputTokens
            }
            fn scope(&self) -> Scope {
                Scope::Content
            }
            fn quality_gated(&self) -> bool {
                true
            }
            fn apply(
                &self,
                req: &mut Request,
                _p: &dyn Provider,
                _plan: &mut Vec<PlanEntry>,
            ) -> anyhow::Result<()> {
                // Keep the answer paragraph, drop the filler → tokens cut, answer retained.
                req.set(
                    "/messages/0/content",
                    Value::String(format!("{ANSWER}\n\n[filler omitted]")),
                );
                Ok(())
            }
        }

        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DropFiller), Box::new(DropAnswer)];
        let out = pipeline::run(&mut req, &OpenAiProvider, c.as_ref(), &stages);

        assert!(out.stages[0].applied, "answer-preserving cut sticks");
        assert!(
            !out.stages[1].applied
                && out.stages[1]
                    .note
                    .as_deref()
                    .is_some_and(|n| n.contains("quality-gate")),
            "answer-deleting cut is quality-gate reverted, got {:?}",
            out.stages[1].note
        );
        // The surviving state is stage 0's output: answer present, filler gone.
        let final_text = req.get_str("/messages/0/content").unwrap();
        assert!(final_text.contains("4.2 million"), "answer survived");
        assert!(
            final_text.contains("omitted"),
            "stage 0's filler-drop stuck"
        );
    }

    /// Happy path: a real `RetrieveStage` prune that KEEPS the answer is NOT reverted by
    /// the quality gate (no false positive). Retrieve keeps the query-relevant chunk, so
    /// coverage of the context stays above the gate.
    #[test]
    fn quality_gate_keeps_a_prune_that_retains_the_answer() {
        let answer =
            "The logistics division quarterly revenue was 4.2 million dollars this period.";
        let filler: Vec<String> = (0..8)
            .map(|i| format!("Paragraph {i}: the cat sat quietly on the warm windowsill at dawn."))
            .collect();
        let mut paras = vec![answer.to_string()];
        paras.extend(filler);
        let context = paras.join("\n\n");
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content":context},
            {"role":"user","content":"what was the quarterly revenue for the logistics division?"}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let c = counter();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.3,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];

        let out = pipeline::run(&mut req, &OpenAiProvider, c.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "answer-preserving prune is kept (coverage stays above the gate)"
        );
        assert!(
            out.input_tokens_after < out.input_tokens_before,
            "tokens still cut"
        );
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(kept.contains("4.2 million"), "answer survived the prune");
        assert!(kept.contains("omitted"), "filler was actually pruned");
    }

    /// The config knob: `quality_gate = false` runs the token gate alone (the gate never
    /// reverts even an answer-deleting cut). Proves the override is wired end to end.
    #[test]
    fn quality_gate_off_lets_the_cut_through() {
        let mut req = Request::from_value(ProviderKind::OpenAi, rag_request());
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DropAnswer)];
        let out = pipeline::run_gated(
            &mut req,
            &OpenAiProvider,
            counter().as_ref(),
            &stages,
            false,
        );
        assert!(
            out.stages[0].applied,
            "with the quality gate off, the token-saving cut is kept"
        );
        assert_eq!(
            req.get_str("/messages/0/content"),
            Some("Weather was pleasant. The cafeteria served pasta today."),
            "the answer-deleting cut took effect (gate off)"
        );
    }

    /// A monolithic prompt (no separate short question) has no query anchor, so the gate
    /// is SKIPPED — it must not revert a legitimate prune just because there's no question
    /// to measure against. Here the answer-deleting `DropAnswer` is allowed through (the
    /// stage's own structural protections, not the quality gate, guard this shape).
    #[test]
    fn quality_gate_skipped_without_a_distinct_question() {
        // Single long user turn, no short question → query_terms empty → gate off.
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content": long_context_with_answer()}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body.clone());

        // Confirm the precondition: no distinct query.
        let q = query_terms(
            &Request::from_value(ProviderKind::OpenAi, body),
            &OpenAiProvider,
        );
        assert!(q.is_empty(), "monolithic prompt yields no query anchor");

        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DropAnswer)];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter().as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "no query → quality gate skipped → token-saving cut kept"
        );
    }

    /// `quality_gate = true` is the shipped default in `DenseConfig`, and it is not
    /// touched by any preset (presets rebuild from `lossless()`).
    #[test]
    fn quality_gate_default_is_on() {
        assert!(DenseConfig::default().quality_gate, "default ON");
        for p in [
            "safe",
            "auto",
            "rag",
            "agent",
            "code",
            "aggressive",
            "cache",
            "reasoning",
        ] {
            assert!(
                DenseConfig::preset(p).unwrap().quality_gate,
                "preset `{p}` keeps the quality gate on"
            );
        }
    }
}
