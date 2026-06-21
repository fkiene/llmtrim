//! Stage B — lexical retrieval (BM25 + TextRank). LOSSY / opt-in.
//!
//! The #1 savings lever: instead of stuffing a whole large context
//! segment, keep only the chunks relevant to the query — BM25 against the short
//! conversation text — or, when there is no query, the most central chunks
//! (TextRank over a lexical-similarity graph). Pure lexical: no model, no
//! embeddings (spec scope rule 1).
//!
//! Lossy: dropped chunks are gone, replaced by a positional elision marker
//! (referenced by position, never a hash). Off by default; quality is
//! checked offline via recall@k (see the unit tests) until the live quality gate
//! lands. The token gate reverts the stage if it doesn't reduce tokens.

use std::collections::HashSet;

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::select::{self, Item, Weights};
use crate::stages::sizing::optimal_keep;
use crate::stages::tools::{detect_lang, lex_words, stopword_set};

pub struct RetrieveStage {
    /// Fraction of chunks to keep (0.0–1.0).
    pub keep_ratio: f64,
    /// Only segments at least this many chars are eligible for pruning; shorter
    /// segments are treated as the query.
    pub min_segment_chars: usize,
    /// Reorder kept chunks into a head+tail U-shape by relevance (lost-in-the-middle).
    pub reorder: bool,
    /// Use MMR diversity-aware selection instead of plain top-k.
    pub mmr: bool,
    /// MMR tradeoff (1.0 = pure relevance, 0.0 = pure diversity).
    pub mmr_lambda: f64,
    /// Chunk at sentence granularity instead of paragraph/line (DSLR, arXiv:2407.03627)
    /// — finer pruning; pairs with `reorder=false` for original-source-order reassembly.
    pub sentence: bool,
}

impl Transform for RetrieveStage {
    fn name(&self) -> &str {
        "retrieve"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        // Compress only the live zone — never re-prune content in the provider's frozen
        // (cache_control) prefix, which would bust the prompt cache.
        let pointers = crate::cache_zone::compressible_pointers(req, provider);
        let min = self.min_segment_chars;

        // Role-aware classification: prune only bulk *context*, never the
        // instruction (system) or the live question (final user turn). The token
        // gate can't see that dropping the instruction/question breaks the task, so
        // the protection has to be structural. Roles come from the provider seam
        // (`turn_index` + `role_at`), so this works on every wire shape — `/messages`,
        // Google `/contents`, OpenAI Responses `/input` — not just `/messages`.
        // Decisions computed up front (one immutable pass) before any mutation.
        use crate::provider::Role;
        let segs: Vec<Seg> = pointers
            .iter()
            .map(|p| Seg {
                idx: crate::provider::turn_index(p),
                role: provider.role_at(req, p),
                len: req.get_str(p).map(|s| s.chars().count()).unwrap_or(0),
                ptr: p.clone(),
            })
            .collect();
        // Top-level text (`role_at` → None, e.g. Anthropic `system`) or an explicit
        // System role is instruction.
        let is_system = |s: &Seg| s.role.is_none() || s.role == Some(Role::System);
        let last_user = segs
            .iter()
            .filter(|s| s.role == Some(Role::User))
            .filter_map(|s| s.idx)
            .max();
        let is_last_user = |s: &Seg| s.idx.is_some() && s.idx == last_user;
        // Only pin the final user turn when there is *other* long context to prune
        // instead — otherwise a single monolithic prompt would never compress (the
        // within-segment boundary pin in `select` still protects its edges).
        let other_long_context = segs
            .iter()
            .any(|s| s.len >= min && !is_system(s) && !is_last_user(s));
        let pinned: Vec<bool> = segs
            .iter()
            .map(|s| is_system(s) || (is_last_user(s) && other_long_context))
            .collect();

        // Query anchor = the final user turn + any genuinely short segments. Large pinned
        // system text (a multi-KB instruction / CLAUDE.md) is deliberately excluded: folding
        // it into the query makes nearly every context sentence "overlap", so pruning
        // underperforms. The question carries the actual information need.
        let query_text: String = segs
            .iter()
            .filter(|s| is_last_user(s) || s.len < min)
            .filter_map(|s| req.get_str(&s.ptr))
            .collect::<Vec<_>>()
            .join(" ");
        let query = lex_words(&query_text);

        for (k, s) in segs.iter().enumerate() {
            if pinned[k] || s.len < min {
                continue; // instruction/question, or too short to be context
            }
            let Some(text) = req.get_str(&s.ptr).map(str::to_string) else {
                continue;
            };
            // Prose pruning would corrupt a JSON array/object — leave those to serialize /
            // json_crush, which encode them by shape.
            if serde_json::from_str::<Value>(text.trim())
                .is_ok_and(|v| v.is_array() || v.is_object())
            {
                continue;
            }
            // Prune the segment, but never touch directive regions embedded in it
            // (e.g. `<system-reminder>` blocks carry CLAUDE.md + harness instructions
            // that must survive regardless of query relevance). `None` = unchanged.
            if let Some(rebuilt) = prune_protecting_directives(&text, &query, self) {
                req.set(&s.ptr, Value::String(rebuilt));
            }
        }
        Ok(())
    }
}

/// Harness-injected directive blocks whose contents must never be pruned — they carry
/// always-on instructions (CLAUDE.md, system reminders) that aren't relevant to the
/// immediate query but still govern behavior. Matched structurally (the tag), not by
/// wording, so it stays language-agnostic.
static DIRECTIVE_BLOCK: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?is)<(?:system-reminder|system|instructions|important[_-]?instructions|directive)\b[^>]*>.*?</(?:system-reminder|system|instructions|important[_-]?instructions|directive)>",
    )
    .unwrap()
});

/// Byte ranges of directive blocks in `text` (in order, non-overlapping).
fn directive_spans(text: &str) -> Vec<(usize, usize)> {
    DIRECTIVE_BLOCK
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect()
}

/// Prune one segment with `prune_one`, but keep every directive block verbatim: prune only
/// the gaps between them. `None` when nothing changed.
fn prune_protecting_directives(
    text: &str,
    query: &[String],
    cfg: &RetrieveStage,
) -> Option<String> {
    let spans = directive_spans(text);
    if spans.is_empty() {
        return prune_one(text, query, cfg);
    }
    let mut out = String::new();
    let mut pos = 0usize;
    let mut changed = false;
    for (start, end) in spans {
        changed |= prune_gap(&text[pos..start], query, cfg, &mut out);
        out.push_str(&text[start..end]); // directive block: verbatim
        pos = end;
    }
    changed |= prune_gap(&text[pos..], query, cfg, &mut out);
    changed.then_some(out)
}

/// Prune a non-directive gap into `out` (verbatim if too short / nothing dropped).
/// Returns whether it pruned anything.
fn prune_gap(gap: &str, query: &[String], cfg: &RetrieveStage, out: &mut String) -> bool {
    if gap.len() >= cfg.min_segment_chars
        && let Some(pruned) = prune_one(gap, query, cfg)
    {
        out.push_str(&pruned);
        true
    } else {
        out.push_str(gap);
        false
    }
}

/// Prune one (already directive-free) span, sentence- or chunk-grained per config.
fn prune_one(text: &str, query: &[String], cfg: &RetrieveStage) -> Option<String> {
    if cfg.sentence {
        rebuild_sentence(text, query, cfg.keep_ratio)
    } else {
        rebuild_chunked(text, query, cfg)
    }
}

/// Sentence-grained pruning (DSLR, arXiv:2407.03627) for one context segment:
/// recall-oriented — keep every sentence sharing a query term, its neighbours, and
/// the boundaries; drop only zero-relevance filler — reassembled in ORIGINAL order
/// (coherence matters at this grain). `None` when the segment can't/shouldn't prune:
/// no query (blind pruning drops answer-bearing sentences), too few sentences, or
/// nothing dropped.
fn rebuild_sentence(text: &str, query: &[String], keep_ratio: f64) -> Option<String> {
    if query.is_empty() {
        return None;
    }
    let chunks = sentence_chunks(text);
    if chunks.len() < 3 {
        return None;
    }
    // Stopwords for the context's own language (whatlang-detected).
    let stops = stopword_set(text);
    let kept = prune_sentences(&chunks, query, stops, keep_ratio);
    if kept.len() >= chunks.len() {
        return None; // nothing dropped
    }
    Some(rebuild(&chunks, &kept, PARA_SEP))
}

/// Chunk-grained retrieval for one context segment: split into chunks, rank (BM25 with
/// a query, else TextRank centrality), then select — plain top-k, MMR diversity, and/or
/// a U-shape reorder — always keeping the boundary chunks. `None` when the segment is
/// too small to prune.
fn rebuild_chunked(text: &str, query: &[String], cfg: &RetrieveStage) -> Option<String> {
    let (chunks, sep) = chunk_with_sep(text);
    if chunks.len() < 2 {
        return None;
    }
    let keep = ((chunks.len() as f64) * cfg.keep_ratio).ceil().max(1.0) as usize;
    if keep >= chunks.len() {
        return None; // nothing to drop
    }
    // Failure-signal chunks survive regardless of query relevance: a test/build failure
    // ("not ok 19 - …", "ERROR …") rarely shares words with the user's ask ("run the
    // tests"), so pure relevance ranking elides exactly the lines the agent needs.
    // Forced into every selection path below, over budget if need be — correctness over
    // budget; the token gate still arbitrates the net result.
    let protected = failure_protected(&chunks);
    let with_protected = |mut idx: Vec<usize>| -> Vec<usize> {
        idx.extend_from_slice(&protected);
        idx.sort_unstable();
        idx.dedup();
        idx
    };
    // Query-less ranking uses TextRank, whose dense n×n similarity matrix is O(n²) memory —
    // tens of thousands of line-chunks (a large log) would allocate gigabytes and abort the
    // process. Above the cap, skip centrality entirely and fall back to a head+tail keep
    // (boundary-safe, O(n)). Never block the user.
    let ranked = if query.is_empty() {
        if chunks.len() > TEXTRANK_MAX_CHUNKS {
            return Some(rebuild(
                &chunks,
                &with_protected(head_tail_keep(keep, chunks.len())),
                sep,
            ));
        }
        textrank_rank(&chunks)
    } else {
        bm25_rank(&chunks, query)
    };
    if !cfg.reorder && !cfg.mmr {
        return Some(rebuild(
            &chunks,
            &with_protected(budgeted_select(&chunks, &ranked, keep, cfg)),
            sep,
        ));
    }
    let mut chosen = if cfg.mmr {
        mmr_order(&ranked, &chunks, keep, cfg.mmr_lambda)
    } else {
        ranked.iter().copied().take(keep).collect::<Vec<usize>>()
    };
    chosen.extend_from_slice(&protected);
    pin_boundaries(&mut chosen, chunks.len());
    if cfg.reorder {
        Some(rebuild_ordered(&chunks, &u_shape(&chosen), sep))
    } else {
        chosen.sort_unstable();
        chosen.dedup();
        Some(rebuild(&chunks, &chosen, sep))
    }
}

/// Indices of chunks that carry a failure-level signal (the toolout stage's STRONG
/// machine-token set: `error`, `panic`, TAP's `not ok`, …) plus the continuation
/// chunks that follow one — the indented traceback frames, or for a TAP failure the
/// whole YAML diagnostic up to the next test point. These must survive pruning no
/// matter how little they overlap the query.
fn failure_protected(chunks: &[String]) -> Vec<usize> {
    use crate::stages::toolout::signals::STRONG;
    /// A failing TAP test point (`not ok 19 - …`) — opens a YAML diagnostic block.
    static TAP_FAIL: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)^not ok\b").unwrap());
    /// Any new TAP record (test point, plan, comment) — closes the diagnostic block.
    /// Matched on trimmed text: the sentence-grained path strips indentation, so the
    /// block can't be delimited by leading whitespace there.
    static TAP_RECORD: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)^(not ok\b|ok\s+\d|\d+\.\.\d+|#)").unwrap());
    let mut out = Vec::new();
    let mut indent_block = false; // after a STRONG line: indented continuations
    let mut tap_block = false; // after `not ok`: everything until the next TAP record
    for (i, c) in chunks.iter().enumerate() {
        let t = c.trim_start();
        if tap_block && !TAP_RECORD.is_match(t) {
            out.push(i); // inside the failure's YAML diagnostic
            continue;
        }
        tap_block = false;
        if STRONG.is_match(c) {
            out.push(i);
            indent_block = true;
            tap_block = TAP_FAIL.is_match(t);
        } else if indent_block && (c.starts_with(' ') || c.starts_with('\t')) {
            out.push(i); // indented continuation (traceback frame) of the failure
        } else {
            indent_block = false;
        }
    }
    out
}

/// Maximum chunk count for which TextRank's dense O(n²) centrality matrix is built. Above
/// this, the query-less path falls back to a head+tail keep so a huge query-less input can
/// never allocate unboundedly. ~2000² f64 ≈ 32 MB, a safe ceiling.
const TEXTRANK_MAX_CHUNKS: usize = 2000;

/// Boundary-safe head+tail selection (no ranking): keep the first ~⅔ and last ~⅓ of the
/// budget. Used as the O(n) fallback when there are too many chunks for TextRank. Always
/// includes the boundary chunks (instruction/question live at the edges).
fn head_tail_keep(keep: usize, n: usize) -> Vec<usize> {
    let keep = keep.min(n);
    let head = (keep * 2 / 3).max(1);
    let tail = keep.saturating_sub(head);
    let mut idx: Vec<usize> = (0..head.min(n)).collect();
    idx.extend((n.saturating_sub(tail))..n);
    pin_boundaries(&mut idx, n);
    idx.sort_unstable();
    idx.dedup();
    idx
}

/// Ensure the first and last chunk are always kept — a prompt's instruction/question
/// lives at its edges, and the token gate can't see that dropping them breaks the task
/// (it still cuts tokens). Shared by [`budgeted_select`] and the MMR/reorder path.
fn pin_boundaries(chosen: &mut Vec<usize>, n: usize) {
    for b in [0, n - 1] {
        if !chosen.contains(&b) {
            chosen.push(b);
        }
    }
}

/// A classified text segment: its JSON pointer, owning conversational turn index
/// (wire-shape agnostic, `None` for top-level system text), normalized role, and char
/// length.
struct Seg {
    ptr: String,
    idx: Option<usize>,
    role: Option<crate::provider::Role>,
    len: usize,
}

/// Split text into sentence chunks (terminal punctuation followed by space/newline),
/// falling back to paragraph/line chunking when no sentence boundaries are found.
fn sentence_chunks(text: &str) -> Vec<String> {
    let sentences: Vec<String> = text
        .split_sentence_bounds()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.len() > 1 {
        sentences
    } else {
        chunk(text)
    }
}

/// Training-free DSLR sentence selection (arXiv:2407.03627). Build a recall base —
/// every sentence sharing a query CONTENT word, its neighbours, and the boundaries
/// (drop only zero-relevance filler) — then **cap** the kept count at `keep_ratio`,
/// dropping the lowest-relevance kept sentences first while always protecting the
/// boundaries and the single highest-scoring (answer) sentence.
///
/// A low `keep_ratio` therefore prunes *harder than chunk-level* yet stays answer-safe:
/// the cut is per-sentence, so the answer sentence survives even inside a mostly-
/// irrelevant paragraph (which chunk-level would drop whole). Kept indices in ORIGINAL
/// order — sentence coherence depends on it.
fn prune_sentences(
    chunks: &[String],
    query: &[String],
    stops: &HashSet<&str>,
    keep_ratio: f64,
) -> Vec<usize> {
    let n = chunks.len();
    // Content words only: a query term shared with every sentence (a stopword) carries
    // no signal and would keep the whole context. `stops` is the detected language's
    // list, so this works beyond English.
    let qset: HashSet<&str> = query
        .iter()
        .map(String::as_str)
        .filter(|w| w.len() >= 2 && !stops.contains(w))
        .collect();
    if qset.is_empty() {
        return (0..n).collect();
    }
    // relevance = number of distinct query content words present in the sentence.
    // Tokenize exactly as `lex_words` (UAX#29 words, then `char::to_lowercase` — what
    // `str::to_lowercase` does) but reuse one lowercase buffer and a small per-sentence
    // hit set, instead of allocating a `Vec<String>` + `HashSet<String>` of every word.
    // Identical scores, ~no per-word allocation — this loop runs over the whole context.
    let mut buf = String::new();
    let mut hits: HashSet<&str> = HashSet::new();
    let score: Vec<usize> = chunks
        .iter()
        .map(|c| {
            hits.clear();
            for w in c.unicode_words() {
                buf.clear();
                buf.extend(w.chars().flat_map(char::to_lowercase));
                if let Some(&q) = qset.get(buf.as_str()) {
                    hits.insert(q);
                }
            }
            hits.len()
        })
        .collect();
    // No sentence matches any content word → no signal; don't prune blindly.
    if score.iter().all(|&s| s == 0) {
        return (0..n).collect();
    }
    let relevant: Vec<bool> = score.iter().map(|&s| s > 0).collect();
    // Failure-signal sentences are kept unconditionally (see [`failure_protected`]) —
    // a test failure rarely shares content words with the question that ran the tests.
    let mut protected = vec![false; n];
    for i in failure_protected(chunks) {
        protected[i] = true;
    }
    let mut keep = vec![false; n];
    for (i, k) in keep.iter_mut().enumerate() {
        let neighbour = (i > 0 && relevant[i - 1]) || (i + 1 < n && relevant[i + 1]);
        if relevant[i] || neighbour || protected[i] || i == 0 || i == n - 1 {
            *k = true;
        }
    }
    // Cap: trim the base set to `keep_ratio` of the sentences, dropping the lowest-
    // relevance kept ones first — but never the boundaries or the single best (answer)
    // sentence. Low ratio = aggressive yet answer-safe.
    let target = ((n as f64) * keep_ratio).ceil().max(1.0) as usize;
    let kept_count = keep.iter().filter(|&&k| k).count();
    if kept_count > target {
        let best = (0..n).max_by_key(|&i| score[i]).unwrap_or(0);
        let mut droppable: Vec<usize> = (0..n)
            .filter(|&i| keep[i] && !protected[i] && i != 0 && i != n - 1 && i != best)
            .collect();
        droppable.sort_by_key(|&i| (score[i], i)); // lowest relevance first
        let mut excess = kept_count - target;
        for i in droppable {
            if excess == 0 {
                break;
            }
            keep[i] = false;
            excess -= 1;
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// The separator used to rejoin kept chunks — the grain the text was split on, so a
/// line-chunked log isn't reassembled with paragraph gaps (which would *add* tokens and
/// often flip the gate to revert).
const PARA_SEP: &str = "\n\n";
const LINE_SEP: &str = "\n";
/// TextTiling tiles are contiguous prose cut at sentence gaps, so kept tiles rejoin with
/// a plain space — no added structure for the gate to pay for.
const TILE_SEP: &str = " ";

/// Split text into chunks: by blank-line paragraphs; for unstructured *prose* (no blank-line
/// structure) try TextTiling lexical-cohesion boundaries (see [`texttile`]); else fall back
/// to lines / a single blob. Returns the chunks and the matching join separator (so
/// [`rebuild`] rejoins at the same grain).
fn chunk_with_sep(text: &str) -> (Vec<String>, &'static str) {
    let paras: Vec<String> = text
        .split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if paras.len() > 1 {
        // Author-supplied paragraph/structure boundaries already exist (markdown, lists,
        // code) — respect them verbatim; TextTiling only earns its keep where structure is
        // absent (Hearst's continuous-prose setting). Leaves the structured path untouched.
        return (paras, PARA_SEP);
    }
    // Prose: no blank-line structure. Place boundaries at lexical-cohesion valleys instead of
    // arbitrary line breaks; `None` ⇒ too short / no clear valleys ⇒ fall through.
    if let Some(tiles) = texttile(text) {
        return (tiles, TILE_SEP);
    }
    let lines: Vec<String> = text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    if lines.len() > 1 {
        (lines, LINE_SEP)
    } else {
        (vec![text.trim().to_string()], PARA_SEP)
    }
}

/// Chunks only, when the grain doesn't matter (ranking/centrality/recall tests).
fn chunk(text: &str) -> Vec<String> {
    chunk_with_sep(text).0
}

// --- TextTiling lexical-cohesion chunk boundaries (Hearst, "TextTiling: Segmenting Text into
// Multi-paragraph Subtopic Passages", Computational Linguistics 1997; depth scoring after
// Eisenstein & Barzilay, "Bayesian Unsupervised Topic Segmentation", EMNLP 2008) --------------
//
// Continuous prose (a transcript, an article body, a long answer) has no blank-line structure,
// so fixed line/position chunking cuts mid-topic. TextTiling instead cuts where the vocabulary
// *changes*: it slides a window over the sentence sequence, measures the lexical similarity of
// the block before each inter-sentence gap against the block after it, and places a boundary at
// each *valley* (cohesion minimum) whose **depth** — how far it drops below the peaks on either
// side — clears a corpus-relative cutoff. Pure token statistics: no model, no embeddings,
// deterministic. Falls back to the caller's line/blob chunking when the text is too short or has
// no clear valley (so a single-topic passage stays one chunk instead of being split arbitrarily).

/// Sliding block size (sentences per side) for the cohesion comparison. Hearst compares
/// fixed-size token blocks across each gap; at sentence granularity a 2-sentence window each
/// side smooths single-sentence noise while still localizing the boundary.
const TILE_BLOCK_SENTENCES: usize = 2;
/// Minimum sentences before TextTiling is attempted — below this there aren't enough interior
/// gaps to estimate a similarity curve (need a window each side plus interior gaps to compare);
/// the caller falls back to line/blob chunking.
const TILE_MIN_SENTENCES: usize = 6;
/// Minimum sentences per emitted tile: a boundary is suppressed if it would leave either side
/// shorter than this, so cohesion noise can't shave off one-sentence slivers (Hearst's
/// minimum-segment guard).
const TILE_MIN_TILE_SENTENCES: usize = 2;
/// Maximum characters per tile. A long stretch with no cohesion valley would otherwise stay one
/// giant chunk (defeating retrieval granularity); past this we accept the deepest interior gap
/// even if its depth is below the cutoff, so tiles stay prunable. Mirrors the segment scale the
/// rest of the stage already prunes at (`min_segment_chars` defaults to 120–200).
const TILE_MAX_TILE_CHARS: usize = 1200;
/// How far above the mean depth a valley must score to become a boundary, in standard deviations
/// of the (positive) depth scores. This is Hearst's liberal cutoff `mean + σ·k` expressed for the
/// depth convention here (deeper valley = higher score). Higher ⇒ fewer boundaries; 0.5 keeps only
/// valleys clearly deeper than typical, so a single-topic passage (shallow, uniform curve) yields
/// no boundary and the caller falls back to its line/blob chunking.
const TILE_DEPTH_CUTOFF_SIGMA: f64 = 0.5;
/// Absolute-drop guard: a boundary gap's cohesion must fall to at most this fraction of the mean
/// gap cohesion. The depth cutoff alone is *relative*, so a single coherent topic — whose cohesion
/// only drifts gently — still throws up shallow "valleys" that clear it; requiring the valley to
/// also be a real dip (cohesion well below the document's typical) is what distinguishes a genuine
/// subtopic shift (cohesion collapses toward zero) from in-topic noise. 0.6 ⇒ the valley must be
/// ≥40% below average cohesion.
const TILE_VALLEY_SIM_RATIO: f64 = 0.6;

/// TextTiling boundaries for one prose segment. Returns the topical tiles (≥2) on success, or
/// `None` when the text is too short or shows no clear lexical-cohesion valley — the caller then
/// falls back to its existing line/blob chunking, so a single-topic passage is never split
/// arbitrarily.
fn texttile(text: &str) -> Option<Vec<String>> {
    let sentences: Vec<&str> = text
        .split_sentence_bounds()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if sentences.len() < TILE_MIN_SENTENCES {
        return None;
    }
    // Content-word multiset per sentence, in the segment's own language (stopwords dropped so
    // function words don't wash out the topic signal). `lex_words` is Unicode-segmented (UAX#29),
    // so this works across scripts, not just space-delimited English.
    let stops = stopword_set(text);
    let bags: Vec<Vec<String>> = sentences
        .iter()
        .map(|s| {
            lex_words(s)
                .into_iter()
                .filter(|w| w.chars().count() >= 2 && !stops.contains(w.as_str()))
                .collect()
        })
        .collect();

    // Gap g sits between sentence g and g+1 (g = 0..n-2). Score = lexical similarity of the
    // block of up to TILE_BLOCK_SENTENCES sentences ending at g against the block starting at
    // g+1 — high = cohesive (same topic), low = a vocabulary shift (candidate boundary).
    let n = sentences.len();
    let sims: Vec<f64> = (0..n - 1)
        .map(|g| {
            let left = block_counts(&bags, g + 1 - TILE_BLOCK_SENTENCES.min(g + 1), g + 1);
            let right = block_counts(&bags, g + 1, (g + 1 + TILE_BLOCK_SENTENCES).min(n));
            cosine(&left, &right)
        })
        .collect();

    // Depth of each gap: how far its similarity dips below the nearest peak on each side
    // (climb left + climb right). Interior valleys (lower than both neighbours) get the full
    // two-sided depth; a monotone slope gets little. This is the Hearst/Eisenstein depth score.
    let depths = gap_depths(&sims);
    // Per-sentence char lengths drive the max-tile guard in `pick_boundaries`.
    let lens: Vec<usize> = sentences.iter().map(|s| s.chars().count()).collect();
    let boundaries = pick_boundaries(&depths, &sims, &lens);
    if boundaries.is_empty() {
        return None; // no clear valley → single topic → let the caller fall back
    }

    // Cut the sentence stream at the chosen gaps (gap g ends a tile after sentence g).
    let mut tiles: Vec<String> = Vec::new();
    let mut start = 0usize;
    for &g in &boundaries {
        tiles.push(sentences[start..=g].join(" "));
        start = g + 1;
    }
    tiles.push(sentences[start..].join(" "));
    Some(tiles)
}

/// Token→count bag over sentences `[lo, hi)` (a comparison block), merging the per-sentence
/// content-word lists. Empty when the range is empty.
fn block_counts(
    bags: &[Vec<String>],
    lo: usize,
    hi: usize,
) -> std::collections::HashMap<&str, f64> {
    let mut counts: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for bag in &bags[lo..hi] {
        for tok in bag {
            *counts.entry(tok.as_str()).or_insert(0.0) += 1.0;
        }
    }
    counts
}

/// Cosine similarity of two token-count blocks (Hearst's block-comparison score). 0 when either
/// block is empty (so a contentless gap reads as maximally dissimilar — a natural boundary).
fn cosine(
    a: &std::collections::HashMap<&str, f64>,
    b: &std::collections::HashMap<&str, f64>,
) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let dot: f64 = a
        .iter()
        .map(|(t, &av)| av * b.get(t).copied().unwrap_or(0.0))
        .sum();
    let na: f64 = a.values().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.values().map(|v| v * v).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Depth score for each gap from its similarity curve (Hearst/Eisenstein): for a valley, sum how
/// far it sits below the nearest higher point on the left and on the right. A gap that isn't a
/// local minimum (its neighbours aren't both higher up the local slope) scores ~0. Endpoints
/// have only one side. Higher depth = a sharper topic shift = a stronger boundary candidate.
fn gap_depths(sims: &[f64]) -> Vec<f64> {
    let n = sims.len();
    (0..n)
        .map(|i| {
            // Climb left: highest similarity until it stops rising as we walk away from i.
            let mut left_peak = sims[i];
            let mut j = i;
            while j > 0 && sims[j - 1] >= sims[j] {
                j -= 1;
                left_peak = left_peak.max(sims[j]);
            }
            let mut right_peak = sims[i];
            let mut k = i;
            while k + 1 < n && sims[k + 1] >= sims[k] {
                k += 1;
                right_peak = right_peak.max(sims[k]);
            }
            (left_peak - sims[i]) + (right_peak - sims[i])
        })
        .collect()
}

/// Pick boundary gaps from the depth curve. `depths[g]` and `sims[g]` describe inter-sentence gap
/// g (so `sims.len() == depths.len() == lens.len() - 1`). A gap qualifies when (a) it is a local
/// depth maximum (deeper than its immediate neighbours — the valley apex, not its slope), (b) its
/// depth clears Hearst's liberal relative cutoff `mean + σ·k` over the positive depths, AND (c)
/// its cohesion is a *real* dip — at most [`TILE_VALLEY_SIM_RATIO`] of the mean gap cohesion —
/// which separates a genuine subtopic shift from in-topic drift the relative cutoff alone would
/// over-segment. Boundaries that would leave either side shorter than [`TILE_MIN_TILE_SENTENCES`]
/// are skipped. A final pass forces an extra split inside any tile that still exceeds
/// [`TILE_MAX_TILE_CHARS`], so no single topic stays one un-prunable blob. Returned ascending; ties
/// resolve to the earliest gap (deterministic).
fn pick_boundaries(depths: &[f64], sims: &[f64], lens: &[usize]) -> Vec<usize> {
    let positive: Vec<f64> = depths.iter().copied().filter(|&d| d > 0.0).collect();
    if positive.is_empty() {
        return Vec::new();
    }
    let mean_depth = positive.iter().sum::<f64>() / positive.len() as f64;
    let var = positive
        .iter()
        .map(|d| (d - mean_depth).powi(2))
        .sum::<f64>()
        / positive.len() as f64;
    let cutoff = mean_depth + TILE_DEPTH_CUTOFF_SIGMA * var.sqrt();
    let mean_sim = sims.iter().sum::<f64>() / sims.len() as f64;
    let valley_ceiling = TILE_VALLEY_SIM_RATIO * mean_sim;

    let n = depths.len();
    let mut chosen: Vec<usize> = Vec::new();
    let mut last_cut: isize = -1; // gap index where the previous tile ended (−1 = start)
    for g in 0..n {
        // Local-maximum test: a valley shows up as a depth peak; only its apex is a boundary,
        // not the shoulders, so adjacent gaps don't both fire.
        let is_local_max = (g == 0 || depths[g] >= depths[g - 1])
            && (g + 1 == n || depths[g] >= depths[g + 1])
            && depths[g] > 0.0;
        if is_local_max
            && depths[g] >= cutoff
            && sims[g] <= valley_ceiling
            && fits_min_tile(g, last_cut, lens.len())
        {
            chosen.push(g);
            last_cut = g as isize;
        }
    }
    split_oversized_tiles(&mut chosen, lens);
    chosen
}

/// True when cutting at gap `g` leaves both the tile since `last_cut` and the remaining tail at
/// least [`TILE_MIN_TILE_SENTENCES`] long — the minimum-segment guard.
fn fits_min_tile(g: usize, last_cut: isize, n_sentences: usize) -> bool {
    (g as isize - last_cut) >= TILE_MIN_TILE_SENTENCES as isize
        && (n_sentences - 1 - g) >= TILE_MIN_TILE_SENTENCES
}

/// Granularity guard: once at least one real cohesion boundary exists, subdivide any resulting
/// tile whose char span still exceeds [`TILE_MAX_TILE_CHARS`] so no over-long subtopic survives as
/// one un-prunable chunk. Each oversized tile is split at the interior gap nearest its char
/// midpoint that respects [`TILE_MIN_TILE_SENTENCES`] on both sides (the midpoint keeps the two
/// halves balanced even when the cohesion curve is flat — e.g. a uniformly cohesive long topic);
/// the pass repeats until every tile fits or no admissible gap remains. No-op when `chosen` is
/// empty: a passage with *no* cohesion valley must fall back to the caller's chunking (spec: "fall
/// back when no valleys"), not be cut at an arbitrary gap.
fn split_oversized_tiles(chosen: &mut Vec<usize>, lens: &[usize]) {
    if chosen.is_empty() {
        return; // no real boundary → honor the no-valley fallback, don't force an arbitrary cut
    }
    let total: usize = lens.iter().sum();
    if total <= TILE_MAX_TILE_CHARS {
        return; // whole segment already fits — nothing can be oversized
    }
    // Prefix char sums so a tile span's length is O(1): chars(a..=b) = prefix[b+1] - prefix[a].
    let mut prefix = vec![0usize; lens.len() + 1];
    for (i, &l) in lens.iter().enumerate() {
        prefix[i + 1] = prefix[i] + l;
    }
    let span_chars = |start: usize, end: usize| prefix[end + 1] - prefix[start];

    loop {
        // Current tile sentence spans [start, end] (gap g ends a tile at sentence g).
        let starts: Vec<usize> = std::iter::once(0)
            .chain(chosen.iter().map(|&g| g + 1))
            .collect();
        let ends: Vec<usize> = chosen
            .iter()
            .copied()
            .chain(std::iter::once(lens.len() - 1))
            .collect();
        let mut added = false;
        for (&start, &end) in starts.iter().zip(&ends) {
            if span_chars(start, end) <= TILE_MAX_TILE_CHARS {
                continue;
            }
            // Gap closest to the tile's char midpoint, min-size-respecting on both sides.
            let mid = prefix[start] + span_chars(start, end) / 2;
            let best = (start..end)
                .filter(|&c| {
                    (c - start + 1) >= TILE_MIN_TILE_SENTENCES
                        && (end - c) >= TILE_MIN_TILE_SENTENCES
                })
                .min_by_key(|&c| (prefix[c + 1].abs_diff(mid), c));
            if let Some(c) = best {
                chosen.push(c);
                added = true;
            }
        }
        if !added {
            break; // every oversized tile is now atomic (can't be split without a sliver)
        }
        chosen.sort_unstable();
        chosen.dedup();
    }
}

/// Map the corpus's detected language to the BM25 tokenizer's language (Snowball
/// stemmer + stopword set), so ranking is correct beyond English. Unknown or
/// undetected → English (graceful), via the shared [`detect_lang`] seam.
fn bm25_language(sample: &str) -> bm25::Language {
    use bm25::Language as B;
    use whatlang::Lang;
    match detect_lang(sample) {
        Some(Lang::Ara) => B::Arabic,
        Some(Lang::Dan) => B::Danish,
        Some(Lang::Nld) => B::Dutch,
        Some(Lang::Fra) => B::French,
        Some(Lang::Deu) => B::German,
        Some(Lang::Ell) => B::Greek,
        Some(Lang::Hun) => B::Hungarian,
        Some(Lang::Ita) => B::Italian,
        Some(Lang::Nob) => B::Norwegian,
        Some(Lang::Por) => B::Portuguese,
        Some(Lang::Ron) => B::Romanian,
        Some(Lang::Rus) => B::Russian,
        Some(Lang::Spa) => B::Spanish,
        Some(Lang::Swe) => B::Swedish,
        Some(Lang::Tam) => B::Tamil,
        Some(Lang::Tur) => B::Turkish,
        _ => B::English,
    }
}

/// BM25+ lower-bound constant δ (Lv & Zhai, "Lower-Bounding Term Frequency
/// Normalization", CIKM 2011). Plain BM25's length normalization lets a long document's
/// per-term TF weight shrink toward zero, so a long chunk that *does* contain a query
/// term can score below a short chunk that *doesn't* — the over-penalization the paper
/// diagnoses. BM25+ adds δ to the normalized TF of every matched term, flooring its
/// contribution at `idf·δ` independent of document length: any occurrence now beats
/// absence. δ=1.0 is the value recommended in the paper.
const BM25_PLUS_DELTA: f64 = 1.0;

/// A tokenizer + per-chunk BM25 scorer over `chunks`, language-aware and Unicode-safe.
/// Bundles the pieces the BM25+/RM3 path reuses: the `bm25` crate's `DefaultTokenizer`
/// (so query/chunk/expansion terms all live in the same stemmed token space) and a fitted
/// `Scorer` that yields the crate's exact BM25 scores.
struct Bm25Index {
    tokenizer: bm25::DefaultTokenizer,
    embedder: bm25::Embedder<u32, bm25::DefaultTokenizer>,
    scorer: bm25::Scorer<usize>,
    /// Stemmed token strings per chunk (index-aligned to `chunks`) — used for the BM25+ δ
    /// bonus and RM3 term statistics without re-tokenizing.
    chunk_tokens: Vec<Vec<String>>,
    n: usize,
}

impl Bm25Index {
    /// Fit a fresh index to `chunks`.
    fn build(chunks: &[String]) -> Bm25Index {
        use bm25::{DefaultTokenizer, EmbedderBuilder, Scorer, Tokenizer};
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        // Detect the corpus language for stemming + stopwords, but DISABLE bm25's unicode
        // normalization: its default transliterates non-Latin scripts to ASCII (CJK →
        // romaji, Cyrillic/Greek → Latin), which mangles CJK *and* breaks bm25's own
        // non-Latin stemmers (Russian/Greek/Arabic/Tamil would be stemmed on
        // transliterated text). With it off, splitting stays Unicode-aware (UAX#29, the
        // same `unicode_words` as `lex_words`) and stemming runs on the real script —
        // universal across languages.
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(bm25_language(&refs.join(" ")))
            .normalization(false)
            .build();
        let embedder =
            EmbedderBuilder::<u32>::with_tokenizer_and_fit_to_corpus(tokenizer, &refs).build();
        // Re-build the tokenizer for membership/expansion queries (the embedder consumed
        // the first one); same settings ⇒ identical token strings.
        let tokenizer = DefaultTokenizer::builder()
            .language_mode(bm25_language(&refs.join(" ")))
            .normalization(false)
            .build();
        let chunk_tokens: Vec<Vec<String>> = refs.iter().map(|c| tokenizer.tokenize(c)).collect();
        let mut scorer = Scorer::<usize>::new();
        for (i, c) in refs.iter().enumerate() {
            scorer.upsert(&i, embedder.embed(c));
        }
        Bm25Index {
            tokenizer,
            embedder,
            scorer,
            chunk_tokens,
            n: chunks.len(),
        }
    }

    /// Stemmed query tokens, deduplicated (the term *set* the model scores against).
    fn query_terms(&self, query: &[String]) -> Vec<String> {
        use bm25::Tokenizer;
        let mut terms = self.tokenizer.tokenize(&query.join(" "));
        terms.sort_unstable();
        terms.dedup();
        terms
    }

    /// IDF of a stemmed term, using the crate's exact robust formula
    /// `ln(1 + (N - df + 0.5)/(df + 0.5))` so the δ=0 baseline matches the `Scorer`.
    fn idf(&self, term: &str) -> f64 {
        let df = self
            .chunk_tokens
            .iter()
            .filter(|toks| toks.iter().any(|t| t == term))
            .count();
        let n = self.n as f64;
        (1.0 + (n - df as f64 + 0.5) / (df as f64 + 0.5)).ln()
    }

    /// BM25+ score of every chunk against `terms` (index-aligned to the chunks). Takes the
    /// crate's exact BM25 from the `Scorer`, then adds the BM25+ floor `idf(t)·δ` for each
    /// query term *present* in the chunk — a term absent from a chunk gets no bonus, so the
    /// floor lifts only real occurrences (exactly the Lv & Zhai correction). With δ=0 this
    /// equals the crate's BM25 chunk-for-chunk.
    fn scores(&self, terms: &[String], delta: f64) -> Vec<f64> {
        let q = self.embedder.embed(&terms.join(" "));
        let mut score = vec![0.0f64; self.n];
        for m in self.scorer.matches(&q) {
            if m.id < self.n {
                score[m.id] = m.score as f64;
            }
        }
        if delta != 0.0 {
            let idf: Vec<f64> = terms.iter().map(|t| self.idf(t)).collect();
            for (i, toks) in self.chunk_tokens.iter().enumerate() {
                let present: HashSet<&str> = toks.iter().map(String::as_str).collect();
                for (t, &w) in terms.iter().zip(&idf) {
                    if present.contains(t.as_str()) {
                        score[i] += w * delta;
                    }
                }
            }
        }
        score
    }

    /// Per-chunk score against a *weighted* term set: `Σ_w weight(w)·bm25+(w)`, where each
    /// single-term BM25+ score reuses [`Bm25Index::scores`]. RM3 re-scoring (see
    /// [`rm3_rescore`]) drives this with the interpolated original+expansion weights.
    /// Linear in the weighted terms, deterministic.
    fn weighted_scores(&self, weighted: &[(String, f64)], delta: f64) -> Vec<f64> {
        let mut score = vec![0.0f64; self.n];
        let one = [String::new()];
        for (term, weight) in weighted {
            if *weight == 0.0 {
                continue;
            }
            let mut slot = one.clone();
            slot[0] = term.clone();
            for (i, s) in self.scores(&slot, delta).into_iter().enumerate() {
                score[i] += weight * s;
            }
        }
        score
    }
}

/// Per-chunk BM25+ scores against `query` (index-aligned to `chunks`). Thin wrapper over
/// [`Bm25Index`] used by the BM25+ tests to assert the δ lower bound directly (production
/// goes through [`bm25_rank`], which also runs RM3). The `delta` knob lets a test compare
/// the δ=0 baseline against δ>0.
#[cfg(test)]
fn bm25_scores(chunks: &[String], query: &[String], delta: f64) -> Vec<f64> {
    let index = Bm25Index::build(chunks);
    let terms = index.query_terms(query);
    index.scores(&terms, delta)
}

/// Rank chunk indices by BM25+ relevance to `query` (best first). Scores via the δ
/// lower-bounded model ([`Bm25Index`]), runs one RM3 pseudo-relevance feedback round when
/// the query is too sparse to discriminate (see [`rm3_rescore`]), then argsorts. Ties
/// break by original order; zero-overlap chunks score 0 and sort last, so the result is
/// always a full, deterministic permutation.
fn bm25_rank(chunks: &[String], query: &[String]) -> Vec<usize> {
    let index = Bm25Index::build(chunks);
    let terms = index.query_terms(query);
    let mut scores = index.scores(&terms, BM25_PLUS_DELTA);
    rm3_rescore(&index, &terms, &mut scores);
    argsort_desc(&scores)
}

// --- RM3 pseudo-relevance feedback (Lavrenko & Croft, "Relevance-Based Language
// Models", SIGIR 2001) -------------------------------------------------------------------
// Score round 1 → take the top-m chunks as a pseudo-relevant set → extract the top-t terms
// weighted by P(w|chunk)·score(chunk) → interpolate with the original query (weight λ) →
// re-score once. Pure corpus statistics: no model, no external resource, deterministic.
//
// We only fire it when the original query is too *weak* to discriminate — a short/sparse
// query, or a ranking so flat that round 1 barely separates the chunks. A rich, decisive
// query is left untouched (expansion would only add noise and risk drift). One round only.

/// Fire RM3 when the query has at most this many distinct (stemmed) terms — a sparse query
/// under-specifies the need, exactly where feedback helps most (≈ the short-query regime in
/// the literature).
const RM3_SPARSE_QUERY_TERMS: usize = 4;
/// Number of top-ranked chunks taken as the pseudo-relevant set (Lavrenko & Croft use a
/// small handful; m=4 is mid-range).
const RM3_FEEDBACK_CHUNKS: usize = 4;
/// Number of expansion terms drawn from the feedback set (t≈10 is the classic default).
const RM3_EXPANSION_TERMS: usize = 10;
/// Interpolation weight on the original query vs. the expansion model (λ=0.5 — the standard
/// even mix; the original query keeps half the mass so expansion can't hijack the ranking).
const RM3_LAMBDA: f64 = 0.5;
/// "Flat ranking" trigger: if the best score is no more than this multiple of the m-th
/// score, round 1 barely separated the chunks, so feedback is worth a try even for a
/// longer query. 1.15 ⇒ the top is within 15% of the feedback frontier.
const RM3_FLATNESS_RATIO: f64 = 1.15;

/// One RM3 feedback round, mutating `scores` in place. No-op (scores untouched) unless the
/// trigger fires and at least one usable expansion term is found — so a rich, decisive
/// query keeps its round-1 BM25+ ranking exactly.
fn rm3_rescore(index: &Bm25Index, query_terms: &[String], scores: &mut [f64]) {
    let n = index.n;
    if n < 3 || query_terms.is_empty() {
        return; // too few chunks to form a feedback set, or nothing to expand from
    }
    let order = argsort_desc(scores);
    let m = RM3_FEEDBACK_CHUNKS.min(n);
    if !rm3_should_fire(query_terms.len(), scores, &order, m) {
        return;
    }
    let feedback = &order[..m];
    let expansion = rm3_expansion_terms(index, query_terms, feedback, scores);
    if expansion.is_empty() {
        return; // nothing new to add → leave the round-1 ranking untouched
    }
    // Interpolated query model: original terms carry weight λ each, expansion terms share
    // (1−λ) in proportion to their relevance-model weight. Re-score once.
    let mut weighted: Vec<(String, f64)> = query_terms
        .iter()
        .map(|t| (t.clone(), RM3_LAMBDA))
        .collect();
    let total: f64 = expansion.iter().map(|(_, w)| *w).sum();
    if total > 0.0 {
        for (term, w) in &expansion {
            weighted.push((term.clone(), (1.0 - RM3_LAMBDA) * (w / total)));
        }
    }
    let rescored = index.weighted_scores(&weighted, BM25_PLUS_DELTA);
    scores.copy_from_slice(&rescored);
}

/// Whether the RM3 trigger fires: a sparse query (few distinct terms), or a flat round-1
/// ranking (the top score barely exceeds the feedback frontier, so BM25+ alone isn't
/// separating the chunks). Either way the original query is too weak to trust as-is.
fn rm3_should_fire(n_query_terms: usize, scores: &[f64], order: &[usize], m: usize) -> bool {
    if n_query_terms <= RM3_SPARSE_QUERY_TERMS {
        return true;
    }
    let top = scores[order[0]];
    let frontier = scores[order[m - 1]];
    // All-zero or non-positive ranking carries no signal — don't expand blindly.
    top > 0.0 && frontier > 0.0 && top <= frontier * RM3_FLATNESS_RATIO
}

/// The top-t RM3 expansion terms with their relevance-model weights. For each candidate
/// term w appearing in the feedback chunks, weight = `Σ_{c∈feedback} P(w|c)·score(c)` with
/// `P(w|c)=tf(w,c)/|c|` (Lavrenko & Croft). Stopwords (language-aware) and terms already in
/// the query are excluded. Ties break lexicographically for determinism.
fn rm3_expansion_terms(
    index: &Bm25Index,
    query_terms: &[String],
    feedback: &[usize],
    scores: &[f64],
) -> Vec<(String, f64)> {
    // Stopwords for the feedback set's own language (so expansion is universal, not
    // English-locked). `stopword_set` lowercases internally via detection; the bm25
    // tokenizer already lowercases tokens, so membership lines up.
    let sample: String = feedback
        .iter()
        .filter_map(|&i| index.chunk_tokens.get(i))
        .flat_map(|t| t.iter())
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(" ");
    let stops = stopword_set(&sample);
    let query_set: HashSet<&str> = query_terms.iter().map(String::as_str).collect();
    let mut weight: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
    for &c in feedback {
        let Some(toks) = index.chunk_tokens.get(c) else {
            continue;
        };
        let len = toks.len() as f64;
        if len == 0.0 {
            continue;
        }
        let sc = scores.get(c).copied().unwrap_or(0.0);
        if sc <= 0.0 {
            continue; // a non-relevant feedback chunk contributes nothing
        }
        // tf(w,c) via a per-chunk count, then P(w|c)·score accumulated across the set.
        let mut tf: std::collections::HashMap<&str, f64> = std::collections::HashMap::new();
        for t in toks {
            *tf.entry(t.as_str()).or_insert(0.0) += 1.0;
        }
        for (term, count) in tf {
            if term.chars().count() < 2 || stops.contains(term) || query_set.contains(term) {
                continue;
            }
            *weight.entry(term).or_insert(0.0) += (count / len) * sc;
        }
    }
    let mut ranked: Vec<(String, f64)> = weight
        .into_iter()
        .map(|(t, w)| (t.to_string(), w))
        .collect();
    // Highest weight first, ties broken lexicographically (deterministic, locale-free).
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(RM3_EXPANSION_TERMS);
    ranked
}

/// Rank chunk indices by TextRank centrality (query-free salience).
fn textrank_rank(chunks: &[String]) -> Vec<usize> {
    let n = chunks.len();
    let toks: Vec<HashSet<String>> = chunks
        .iter()
        .map(|c| lex_words(c).into_iter().collect())
        .collect();

    // Lexical-similarity weights (overlap normalized by log lengths, LexRank-style).
    // The weight is symmetric — `inter(i,j)` and the log-length denominator are both
    // order-independent — so compute each unordered pair once and mirror it. Halves the
    // set intersections (the O(n²) cost) for a bit-identical matrix; eases the latent
    // blow-up on query-less long docs. `ln_len` hoists the per-row log out of the pair loop.
    let ln_len: Vec<f64> = toks.iter().map(|t| ((t.len() + 1) as f64).ln()).collect();
    let mut w = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let inter = toks[i].intersection(&toks[j]).count() as f64;
            let denom = ln_len[i] + ln_len[j];
            let weight = if denom > 0.0 { inter / denom } else { 0.0 };
            w[i][j] = weight;
            w[j][i] = weight;
        }
    }

    let damping = 0.85;
    let mut score = vec![1.0 / n as f64; n];
    let out_sum: Vec<f64> = w.iter().map(|row| row.iter().sum()).collect();
    for _ in 0..30 {
        let mut next = vec![(1.0 - damping) / n as f64; n];
        for i in 0..n {
            for j in 0..n {
                if out_sum[j] > 0.0 {
                    next[i] += damping * (w[j][i] / out_sum[j]) * score[j];
                }
            }
        }
        score = next;
    }
    argsort_desc(&score)
}

/// Indices sorted by score descending, ties broken by original order (deterministic).
/// Not `toolout::fill_by_score`: this returns a full ranking (selection + boundary
/// pinning happen later in `select`), not a budget-filled keep mask.
fn argsort_desc(scores: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx
}

/// The top `keep` ranked indices, returned in original (ascending) order.
/// Test-only helper for recall@k assertions; production uses [`budgeted_select`].
#[cfg(test)]
fn top_k(ranked: &[usize], keep: usize) -> Vec<usize> {
    let mut kept: Vec<usize> = ranked.iter().copied().take(keep).collect();
    kept.sort_unstable();
    kept
}

/// Token cost of a chunk, as a word count (the cheap, tokenizer-free proxy shared with
/// [`optimal_keep`]'s bigram estimate). The gate re-measures real tokens after the stage,
/// so an exact tokenizer here would only duplicate that work; word count is monotone with
/// it and keeps selection deterministic and fast.
fn chunk_cost(chunk: &str) -> usize {
    lex_words(chunk).len().max(1)
}

/// Selection weights for chunk retrieval. A **low saturation** (`α = 0.3`) makes a
/// word-bigram stop paying off after roughly one covering chunk, so repeated facts
/// (near-duplicate paragraphs) earn ~zero marginal coverage past the first copy and are
/// dropped — the prose-dedup behaviour MMR provided, now inside the budgeted objective.
/// `λ = 0.5` keeps relevance and coverage balanced.
const RETRIEVE_WEIGHTS: Weights = Weights {
    lambda: 0.5,
    saturation: 0.3,
};

/// Budget-constrained submodular chunk selection (replaces a plain top-k keep).
///
/// Instead of "keep the K best-ranked chunks", choose the subset that maximizes
/// relevance + diverse bigram coverage under a **token budget** (Lin-Bilmes via the
/// CELF greedy in [`crate::select`]). Two near-duplicate chunks that top-k would both
/// keep now compete for the same coverage, so the redundant one is dropped in favour of
/// a novel chunk — a strictly better use of the budget.
///
/// Budget derivation (preserving the existing knobs, and repurposing sizing as a budget
/// source rather than deleting it):
/// - When `keep_ratio` is a real fraction (`0 < ratio < 1`, the configured case) it is the
///   budget: `ratio · Σ chunk tokens`. Coverage in [`crate::select`] then drops the chunks
///   the budget can't justify — the redundant near-duplicates first.
/// - When no usable ratio is given (`ratio ≥ 1` ⇒ "keep everything") the bigram-**saturation**
///   estimate [`optimal_keep`] supplies the default budget instead: the token cost of the
///   `k_sat` best-ranked chunks, where `k_sat` is the diversity-justified keep count. So a
///   redundant segment still shrinks even without an explicit ratio.
///
/// Boundary chunks (first + last) are always retained — the same instruction/question
/// safety guard top-k had: the gate can't see that dropping an edge breaks the task.
/// Returned indices are sorted ascending; emission stays in document order.
fn budgeted_select(
    chunks: &[String],
    ranked: &[usize],
    keep: usize,
    cfg: &RetrieveStage,
) -> Vec<usize> {
    let n = chunks.len();
    let costs: Vec<usize> = chunks.iter().map(|c| chunk_cost(c)).collect();
    let total_tokens: usize = costs.iter().sum();
    let min_cost = *costs.iter().min().unwrap_or(&1);
    let budget = if cfg.keep_ratio > 0.0 && cfg.keep_ratio < 1.0 {
        // Explicit ratio: a fraction of the segment's tokens (existing semantics).
        ((total_tokens as f64) * cfg.keep_ratio).ceil() as usize
    } else {
        // No usable ratio → fall back to sizing's saturation estimate as the budget: the
        // token cost of the `k_sat` best-ranked chunks (near-duplicate spam can't inflate it).
        let k_sat = optimal_keep(
            &chunks.iter().map(String::as_str).collect::<Vec<_>>(),
            1,
            keep,
        );
        ranked.iter().take(k_sat).map(|&i| costs[i]).sum()
    }
    .max(min_cost); // always enough for at least one chunk

    // rank position → relevance in (0, 1]: best-ranked chunk scores ~1, worst ~1/n. Mirrors
    // the MMR path's `rel()`. `ranked` is a full permutation, so every chunk gets a score.
    let mut rel = vec![0.0f64; n];
    for (pos, &i) in ranked.iter().enumerate() {
        if i < n {
            rel[i] = (n - pos) as f64 / n as f64;
        }
    }
    let items: Vec<Item> = (0..n)
        .map(|i| Item::from_text(&chunks[i], costs[i], rel[i]))
        .collect();

    let mut chosen = select::select(&items, budget, &RETRIEVE_WEIGHTS);

    // Safety: a prompt's instruction/question lives at its edges; pin them so the gate
    // (blind to task-breakage) can't elide a unique trailing question.
    pin_boundaries(&mut chosen, n);
    chosen.sort_unstable();
    chosen.dedup();
    chosen
}

/// Greedy MMR selection: balance relevance (rank position) against redundancy
/// (Jaccard token overlap with already-picked chunks). Returns `keep` indices in
/// selection order (most-relevant-and-novel first).
fn mmr_order(ranked: &[usize], chunks: &[String], keep: usize, lambda: f64) -> Vec<usize> {
    let toks: Vec<HashSet<String>> = chunks
        .iter()
        .map(|c| lex_words(c).into_iter().collect())
        .collect();
    let total = ranked.len();
    let denom = total.max(1) as f64;
    // Precompute each chunk's rank position once: the MMR loop is k·n and called
    // `rel()` via a linear `position()` scan, making selection O(k·n²). `rank[i]` is
    // chunk i's place in `ranked` (or `total` if absent), so `rel` is now O(1).
    let mut rank = vec![total; chunks.len()];
    for (pos, &i) in ranked.iter().enumerate() {
        if i < rank.len() {
            rank[i] = pos;
        }
    }
    let rel = |idx: usize| -> f64 {
        let pos = rank.get(idx).copied().unwrap_or(total);
        (total - pos) as f64 / denom
    };
    let sim = |a: usize, b: usize| -> f64 {
        let (ta, tb) = (&toks[a], &toks[b]);
        if ta.is_empty() || tb.is_empty() {
            return 0.0;
        }
        let inter = ta.intersection(tb).count() as f64;
        let uni = ta.union(tb).count() as f64;
        if uni > 0.0 { inter / uni } else { 0.0 }
    };
    let mut selected: Vec<usize> = Vec::new();
    let mut pool: Vec<usize> = ranked.to_vec();
    while selected.len() < keep && !pool.is_empty() {
        let best = pool
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let score = |i: usize| {
                    let red = selected.iter().map(|&s| sim(i, s)).fold(0.0_f64, f64::max);
                    lambda * rel(i) - (1.0 - lambda) * red
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        selected.push(best);
        pool.retain(|&x| x != best);
    }
    selected
}

/// Arrange relevance-ranked indices into a head+tail U-shape — most relevant at the
/// edges, least in the middle (lost-in-the-middle, Liu 2307.03172). Deduplicates.
fn u_shape(ranked: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let (mut front, mut back) = (Vec::new(), Vec::new());
    for (i, &x) in ranked.iter().filter(|&&x| seen.insert(x)).enumerate() {
        if i % 2 == 0 {
            front.push(x)
        } else {
            back.push(x)
        }
    }
    back.reverse();
    front.extend(back);
    front
}

/// Reassemble chunks in an explicit (reordered) order, with one summary note for
/// dropped chunks. Used when `reorder` scrambles positions, so per-position elision
/// markers no longer apply. `sep` is the original chunk grain.
fn rebuild_ordered(chunks: &[String], order: &[usize], sep: &str) -> String {
    let dropped = chunks.len() - order.len();
    let mut parts: Vec<String> = order.iter().map(|&i| chunks[i].clone()).collect();
    if dropped > 0 {
        parts.push(format!(
            "[… {dropped} chunk(s) omitted, kept chunks reordered by relevance …]"
        ));
    }
    parts.join(sep)
}

/// Reassemble kept chunks in order, collapsing each dropped run into one marker. `sep` is
/// the original chunk grain ("\n\n" for paragraphs, "\n" for lines) so a line-chunked log
/// isn't re-expanded with paragraph gaps.
fn rebuild(chunks: &[String], keep_idx: &[usize], sep: &str) -> String {
    let keep_set: HashSet<usize> = keep_idx.iter().copied().collect();
    let mut parts: Vec<String> = Vec::new();
    let mut dropped = 0usize;
    for (i, c) in chunks.iter().enumerate() {
        if keep_set.contains(&i) {
            if dropped > 0 {
                parts.push(format!("[… {dropped} chunk(s) omitted …]"));
                dropped = 0;
            }
            parts.push(c.clone());
        } else {
            dropped += 1;
        }
    }
    if dropped > 0 {
        parts.push(format!("[… {dropped} chunk(s) omitted …]"));
    }
    parts.join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    /// A context where exactly one paragraph answers the query.
    fn doc() -> String {
        [
            "The cafeteria serves lunch from noon until two in the afternoon.",
            "Parking is available in the north lot for all visitors and staff.",
            "The quarterly revenue figure for the logistics division was 4.2 million.",
            "Recycling bins are located on every floor near the elevators.",
            "Office hours run from nine to five on weekdays only.",
        ]
        .join("\n\n")
    }

    #[test]
    fn bm25_recall_at_k_finds_the_answer_chunk() {
        let chunks = chunk(&doc());
        let query = lex_words("what was the quarterly revenue for logistics");
        let ranked = bm25_rank(&chunks, &query);
        // The revenue paragraph (index 2) must rank first.
        assert_eq!(ranked[0], 2, "BM25 should rank the answer chunk top");
        // recall@2: answer retained when keeping the top 2.
        assert!(top_k(&ranked, 2).contains(&2));
    }

    #[test]
    fn bm25_ranks_answer_in_non_latin_script() {
        // Chinese (CJK), no inter-word spaces. With bm25's unicode normalization OFF the
        // text is NOT transliterated to ASCII; UAX#29 splits it into character tokens and
        // the answer chunk (sharing the distinctive characters 物流/收入/季度) still ranks
        // top. Proves retrieval is universal, not English-locked.
        let chunks = vec![
            "食堂在中午到下午两点之间供应午餐".to_string(),
            "北方停车场为所有访客和员工提供停车位".to_string(),
            "物流部门本季度的收入为四百二十万".to_string(),
            "回收箱位于每层楼电梯附近".to_string(),
            "工作时间为工作日上午九点到下午五点".to_string(),
        ];
        let query = lex_words("物流部门本季度的收入是多少");
        let ranked = bm25_rank(&chunks, &query);
        assert_eq!(ranked[0], 2, "BM25 ranks the revenue chunk top in Chinese");
    }

    #[test]
    fn textrank_ranks_all_chunks() {
        let chunks = chunk(&doc());
        let ranked = textrank_rank(&chunks);
        assert_eq!(ranked.len(), chunks.len());
        let unique: HashSet<usize> = ranked.iter().copied().collect();
        assert_eq!(unique.len(), chunks.len(), "ranking is a permutation");
    }

    #[test]
    fn rebuild_collapses_dropped_runs() {
        let chunks: Vec<String> = (0..5).map(|i| format!("chunk{i}")).collect();
        let out = rebuild(&chunks, &[0, 3], PARA_SEP);
        assert_eq!(
            out,
            "chunk0\n\n[… 2 chunk(s) omitted …]\n\nchunk3\n\n[… 1 chunk(s) omitted …]"
        );
    }

    #[test]
    fn rebuild_rejoins_line_chunks_at_line_grain() {
        // A log split by LINES must rejoin with "\n", not "\n\n" — doubling newlines
        // would add tokens and often flip the gate to revert.
        let log = (0..6)
            .map(|i| format!("line{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let (chunks, sep) = chunk_with_sep(&log);
        assert_eq!(sep, "\n", "line-chunked text remembers the line grain");
        let out = rebuild(&chunks, &[0, 1, 5], sep);
        assert!(
            !out.contains("\n\n"),
            "no paragraph gaps in a line-chunked rejoin: {out:?}"
        );
        assert!(out.contains("line0\nline1"), "kept lines stay line-joined");
    }

    #[test]
    fn head_tail_keep_is_boundary_safe_and_bounded() {
        // The TextRank OOM fallback: keeps a head+tail slice, always including both edges,
        // and never more than the budget (plus the two pinned boundaries).
        let n = 10_000;
        let keep = head_tail_keep(30, n);
        assert!(keep.contains(&0) && keep.contains(&(n - 1)), "edges pinned");
        assert!(
            keep.len() <= 32,
            "bounded to the budget, got {}",
            keep.len()
        );
        assert!(keep.windows(2).all(|w| w[0] < w[1]), "sorted, deduped");
    }

    #[test]
    fn sentence_chunks_split_on_terminal_punctuation() {
        let c = sentence_chunks("First fact here. Second fact there! Third one? Done.");
        assert_eq!(c.len(), 4, "split into four sentences");
        assert!(c[0].contains("First fact"));
    }

    #[test]
    fn prune_sentences_keeps_relevant_and_neighbours_drops_filler() {
        let chunks: Vec<String> = [
            "The vault access code is 7741.",       // 0 relevant (vault, code)
            "Lunch is served at noon each day.",    // 1 filler (neighbour of 0)
            "Parking validation is at the desk.",   // 2 pure filler (middle)
            "Weather today is sunny and warm.",     // 3 filler (neighbour of 4)
            "Use the vault code to open the safe.", // 4 relevant (vault, code)
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let query = vec!["vault".to_string(), "code".to_string()];
        let stops = stopword_set("the vault is here and the code is there for a test");
        // Loose cap: keep relevant + neighbours, drop only the zero-overlap filler (#2).
        let kept = prune_sentences(&chunks, &query, stops, 0.9);
        assert!(
            kept.contains(&0) && kept.contains(&4),
            "relevant sentences kept"
        );
        assert!(!kept.contains(&2), "zero-overlap middle filler dropped");
        let mut sorted = kept.clone();
        sorted.sort_unstable();
        assert_eq!(kept, sorted, "kept in original order");

        // Aggressive cap: drop the neighbours too, but never the answer (#0/#4).
        let tight = prune_sentences(&chunks, &query, stops, 0.4);
        assert!(
            tight.contains(&0) && tight.contains(&4),
            "answer + boundaries protected under aggressive cap"
        );
        assert!(
            tight.len() < kept.len(),
            "aggressive cap keeps strictly fewer"
        );
    }

    #[test]
    fn u_shape_puts_relevant_at_edges() {
        // ranked best->worst [0,1,2,3,4] => most relevant at both ends, least middle.
        assert_eq!(u_shape(&[0, 1, 2, 3, 4]), vec![0, 2, 4, 3, 1]);
    }

    #[test]
    fn mmr_drops_redundant_in_favor_of_novel() {
        let chunks = vec![
            "the quarterly revenue for logistics was four million".to_string(),
            "the quarterly revenue for logistics was four million dollars".to_string(),
            "parking is available in the north lot for visitors".to_string(),
        ];
        let picked = mmr_order(&[0, 1, 2], &chunks, 2, 0.5);
        assert!(picked.contains(&0));
        assert!(
            picked.contains(&2),
            "MMR picks the novel chunk over the near-duplicate"
        );
        assert!(!picked.contains(&1), "redundant near-duplicate dropped");
    }

    #[test]
    fn reorder_and_mmr_stage_keeps_answer_and_notes_reorder() {
        let big = std::iter::repeat_n(doc(), 3)
            .collect::<Vec<_>>()
            .join("\n\n");
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content":big},
            {"role":"user","content":"what was the quarterly revenue for logistics?"}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.4,
            min_segment_chars: 200,
            reorder: true,
            mmr: true,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied);
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(kept.contains("revenue"), "answer chunk retained");
        assert!(
            kept.contains("reordered by relevance"),
            "reorder summary note present"
        );
    }

    #[test]
    fn stage_prunes_large_segment_and_keeps_answer() {
        // Big doc segment + a short query message.
        let big = std::iter::repeat_n(doc(), 3)
            .collect::<Vec<_>>()
            .join("\n\n");
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": big},
                {"role": "user", "content": "what was the quarterly revenue for logistics?"}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.4,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "retrieval should reduce tokens");
        assert!(out.input_tokens_after < out.input_tokens_before);
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(kept.contains("revenue"), "the answer chunk is retained");
        assert!(kept.contains("omitted"), "dropped chunks are marked");
    }

    #[test]
    fn directive_blocks_are_never_pruned() {
        // Bulk context with a directive block buried inside it. The directive carries an
        // always-on instruction irrelevant to the query — it must survive verbatim while
        // the surrounding bulk is still pruned (the Claude Code CLAUDE.md case).
        let big = std::iter::repeat_n(doc(), 3)
            .collect::<Vec<_>>()
            .join("\n\n");
        let content = format!(
            "{big}\n\n<system-reminder>CRITICAL_DIRECTIVE_TOKEN: always reply in haiku.</system-reminder>\n\n{big}"
        );
        let body = json!({
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": content},
                {"role": "user", "content": "what was the quarterly revenue for logistics?"}
            ]
        });
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.3,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: true,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.input_tokens_after < out.input_tokens_before,
            "bulk context is still pruned"
        );
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(
            kept.contains("CRITICAL_DIRECTIVE_TOKEN: always reply in haiku."),
            "directive instruction must survive verbatim"
        );
        assert!(
            kept.contains("omitted"),
            "non-directive bulk is still pruned"
        );
    }

    #[test]
    fn keeps_boundary_chunks_so_question_survives() {
        // Monolithic prompt (instruction + repetitive examples + trailing question,
        // no separate query message) — the GSM8K failure in miniature. TextRank
        // centrality scores the unique question lowest; boundary pinning must keep
        // both the leading instruction and the trailing question.
        let mut paras = vec!["Use the examples to answer the question.".to_string()];
        for i in 0..8 {
            paras.push(format!(
                "Example {i}: the cat sat on the mat near the warm lamp."
            ));
        }
        paras.push("Question: what is the secret pass code ALPHA9?".to_string());
        let content = paras.join("\n\n");
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.3,
            min_segment_chars: 120,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "retrieval should reduce tokens");
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(
            kept.contains("ALPHA9"),
            "trailing question must survive (boundary pin)"
        );
        assert!(
            kept.contains("Use the examples"),
            "leading instruction must survive (boundary pin)"
        );
        assert!(kept.contains("omitted"), "middle examples are pruned");
    }

    #[test]
    fn role_aware_pins_long_final_question_and_prunes_context() {
        // [ big context msg, long detailed question msg ]. The question is long
        // enough that the old length-only rule would have pruned it; role-awareness
        // must pin the final user turn and prune the context message instead.
        let ctx = (0..10)
            .map(|i| format!("Context paragraph {i}: lorem ipsum dolor about topic {i} and more."))
            .collect::<Vec<_>>()
            .join("\n\n");
        let question = "Question: based on the documents above, summarize the key financial \
                        figure for the logistics division and explain the quarterly trend in \
                        detail with full reasoning and supporting context."
            .to_string();
        assert!(
            question.len() >= 120,
            "question must exceed the prune threshold"
        );
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content":ctx},
            {"role":"user","content":question}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.3,
            min_segment_chars: 120,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "context pruned -> tokens cut");
        assert_eq!(
            req.get_str("/messages/1/content").unwrap(),
            question,
            "final user turn pinned verbatim"
        );
        assert!(
            req.get_str("/messages/0/content")
                .unwrap()
                .contains("omitted"),
            "bulk context is pruned"
        );
    }

    #[test]
    fn role_aware_works_on_google_contents_shape() {
        // Gemini wire shape (`/contents/...`, not `/messages/...`). Before the seam fix the
        // hardcoded `/messages/` parse made every segment look like system → all pinned →
        // the stage silently no-opped. With `turn_index` + `role_at` it must prune the bulk
        // context turn while pinning the final short user question.
        use crate::provider::GoogleProvider;
        let big = std::iter::repeat_n(doc(), 3)
            .collect::<Vec<_>>()
            .join("\n\n");
        let body = json!({"contents":[
            {"role":"user","parts":[{"text":big}]},
            {"role":"user","parts":[{"text":"what was the quarterly revenue for logistics?"}]}],
            "generationConfig":{"maxOutputTokens":64}});
        let mut req = Request::from_value(ProviderKind::Google, body);
        let counter = counter_for(ProviderKind::Google, Some("gemini-1.5-pro")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.4,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &GoogleProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "retrieval must run on the Google shape, not no-op"
        );
        let kept = req.get_str("/contents/0/parts/0/text").unwrap();
        assert!(kept.contains("revenue"), "answer chunk retained");
        assert!(kept.contains("omitted"), "bulk context pruned");
        assert_eq!(
            req.get_str("/contents/1/parts/0/text").unwrap(),
            "what was the quarterly revenue for logistics?",
            "final user question pinned verbatim"
        );
    }

    #[test]
    fn short_segments_are_left_alone() {
        let body =
            json!({"model":"gpt-4o","messages":[{"role":"user","content":"short question?"}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(RetrieveStage {
            keep_ratio: 0.4,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            !out.stages[0].applied,
            "short content is the query, not pruned"
        );
    }

    fn retrieve_cfg(keep_ratio: f64) -> RetrieveStage {
        RetrieveStage {
            keep_ratio,
            min_segment_chars: 200,
            reorder: false,
            mmr: false,
            mmr_lambda: 0.5,
            sentence: false,
        }
    }

    /// A TAP log (node --test / prove): one buried failure with its YAML diagnostic
    /// block, drowned in passing tests. The failure marker is `not ok` — no
    /// error/fail token — and shares no content word with the query.
    fn tap_log() -> String {
        let mut lines = vec!["TAP version 13".to_string()];
        for i in 1..=52 {
            if i == 19 {
                lines.extend(
                    [
                        "not ok 19 - normalize: Windows backslash path matches POSIX entry",
                        "  ---",
                        "  failureType: 'testCodeFailure'",
                        "  error: |-",
                        "    Expected values to be strictly deep-equal:",
                        "    + Symbol(FAIL_OPEN)",
                        "    - true",
                        "  ...",
                    ]
                    .map(String::from),
                );
            } else {
                lines.push(format!(
                    "ok {i} - isInList: case {i} returns expected result"
                ));
                lines.push(format!("  duration_ms: {}.123456", i * 3));
            }
        }
        lines.push("1..52".to_string());
        lines.push("# pass 51".to_string());
        lines.push("# fail 1".to_string());
        lines.join("\n")
    }

    #[test]
    fn tap_failure_block_survives_chunk_pruning() {
        // The Reddit-reported bug: query-relevance ranking elided the only failing
        // test from a TAP log. Failure-signal chunks must survive any keep_ratio.
        let query = lex_words("run the tests");
        let out =
            rebuild_chunked(&tap_log(), &query, &retrieve_cfg(0.1)).expect("large log prunes");
        assert!(out.contains("not ok 19"), "failure line survives: {out}");
        assert!(
            out.contains("testCodeFailure") && out.contains("Symbol(FAIL_OPEN)"),
            "indented diagnostic block survives with its failure: {out}"
        );
        assert!(out.contains("# fail 1"), "summary fail count survives");
        assert!(out.contains("omitted"), "passing noise still pruned");
    }

    #[test]
    fn tap_diagnostic_survives_without_indentation() {
        // The sentence-grained path trims chunks, so the diagnostic block can't be
        // recognized by leading whitespace — only by TAP record structure.
        let chunks: Vec<String> = tap_log().lines().map(|l| l.trim().to_string()).collect();
        let kept = failure_protected(&chunks);
        let joined: String = kept.iter().map(|&i| chunks[i].as_str()).collect();
        assert!(joined.contains("not ok 19"));
        assert!(
            joined.contains("Symbol(FAIL_OPEN)"),
            "trimmed diagnostic protected"
        );
        assert!(
            !joined.contains("ok 20"),
            "block closes at the next test point"
        );
    }

    #[test]
    fn tap_failure_block_survives_sentence_pruning() {
        let chunks: Vec<String> = tap_log().lines().map(String::from).collect();
        let query = lex_words("run the tests");
        let stops = stopword_set(&tap_log());
        let kept = prune_sentences(&chunks, &query, stops, 0.1);
        let joined: String = kept.iter().map(|&i| chunks[i].as_str()).collect();
        assert!(
            joined.contains("not ok 19"),
            "failure line survives the cap"
        );
        assert!(
            joined.contains("Symbol(FAIL_OPEN)"),
            "diagnostic survives the cap"
        );
    }

    #[test]
    fn budgeted_select_prunes_redundant_near_duplicates_topk_kept() {
        // Seven chunks. 0 and 6 are distinct boundaries (always pinned). 1,2,3 are
        // near-identical revenue lines (same bigrams) that rank highest. 4 and 5 are
        // distinct novel paragraphs ranked below the duplicates. A plain top-k keeps the
        // highest-RANKED chunks — the three redundant copies — wasting the budget on
        // duplicates. The budgeted submodular selection saturates the shared revenue
        // bigrams after the FIRST copy, so the remaining duplicates earn only their
        // (relevance-only) marginal and lose the budget to the novel chunks 4/5.
        let chunks: Vec<String> = vec![
            "alpha preamble about the office building main entrance and lobby desk".to_string(),
            "the quarterly revenue for the logistics division was four point two million"
                .to_string(),
            "the quarterly revenue for the logistics division was four point two million"
                .to_string(),
            "the quarterly revenue for the logistics division was four point two million"
                .to_string(),
            "recycling bins sit on every floor right beside the north stairwell doors".to_string(),
            "visitor parking validation happens at the front security reception window".to_string(),
            "omega closing remarks about the parking garage exit gate and ramp barrier".to_string(),
        ];
        // Ranking: the three revenue copies first, then the two novel chunks, then edges.
        let ranked = vec![1, 2, 3, 4, 5, 0, 6];
        let keep = ((chunks.len() as f64) * 0.5).ceil() as usize;

        // Old top-k keep would retain all three redundant copies.
        let old = top_k(&ranked, keep + 2); // generous K, as old code pinned edges on top
        assert!(
            old.contains(&1) && old.contains(&2) && old.contains(&3),
            "baseline top-k kept all three redundant copies: {old:?}"
        );

        // New budgeted selection: at most one revenue copy survives, and a novel chunk is
        // preferred over a redundant one.
        let chosen = budgeted_select(&chunks, &ranked, keep, &retrieve_cfg(0.5));
        let dup_kept = [1usize, 2, 3]
            .iter()
            .filter(|&&i| chosen.contains(&i))
            .count();
        assert!(
            dup_kept <= 1,
            "at most one revenue copy is kept (redundancy pruned): {chosen:?}"
        );
        assert!(
            chosen.contains(&4) || chosen.contains(&5),
            "a novel chunk is preferred over a redundant duplicate: {chosen:?}"
        );
        assert!(
            chosen.contains(&0) && chosen.contains(&6),
            "distinct boundary chunks retained: {chosen:?}"
        );
    }

    // --- BM25+ lower-bounded TF (Lv & Zhai, CIKM 2011) ---------------------------------------

    #[test]
    fn bm25_plus_long_match_beats_absence() {
        // A long chunk that *contains* the query term vs a short chunk that does NOT. The
        // δ floor guarantees the occurrence clears absence regardless of the long chunk's
        // length penalty — and lifts it strictly above the δ=0 (plain BM25) score.
        let filler = "and the report also covered many other routine operational matters \
                      across the wider organisation in considerable repetitive detail";
        let long_with_term = format!("the logistics division summary {filler} {filler} {filler}");
        let short_without = "weather today is mild".to_string();
        let chunks = vec![long_with_term, short_without];
        let query = lex_words("logistics");

        let plain = bm25_scores(&chunks, &query, 0.0);
        let plus = bm25_scores(&chunks, &query, BM25_PLUS_DELTA);

        assert_eq!(plain[1], 0.0, "absence scores zero under plain BM25");
        assert_eq!(plus[1], 0.0, "δ adds nothing to a chunk lacking the term");
        assert!(plus[0] > 0.0, "the matching chunk scores positive");
        assert!(
            plus[0] > plain[0],
            "δ raises the matched long chunk's floor"
        );
        assert!(plus[0] > plus[1], "occurrence beats absence under BM25+");
    }

    #[test]
    fn bm25_plus_keeps_equal_density_ranking() {
        // Two chunks that both contain the query term at the same per-unit density (one
        // "logistics" per identical unit; the long chunk just repeats the unit). Because both
        // contain the term, the BM25+ floor adds the *same* idf·δ to each, so it must shift both
        // equally and never flip their relative order vs the δ=0 baseline (the Lv & Zhai floor
        // corrects absence-vs-presence, not presence-vs-presence).
        let unit = "the logistics division handled freight";
        let short = unit.to_string();
        let long = format!("{unit} {unit} {unit}");
        let chunks = vec![short, long];
        let query = lex_words("logistics");

        let plain = bm25_scores(&chunks, &query, 0.0);
        let plus = bm25_scores(&chunks, &query, BM25_PLUS_DELTA);

        assert_eq!(
            argsort_desc(&plain),
            argsort_desc(&plus),
            "δ floor preserves the equal-density ranking"
        );
        // The δ bonus is identical for both (same term, same idf, present in both), so the score
        // gap is preserved — δ shifts the pair, it doesn't reorder it.
        let bonus_short = plus[0] - plain[0];
        let bonus_long = plus[1] - plain[1];
        assert!(
            (bonus_short - bonus_long).abs() < 1e-9,
            "equal idf·δ bonus applied to both present chunks"
        );
    }

    // --- RM3 pseudo-relevance feedback (Lavrenko & Croft, SIGIR 2001) ------------------------

    /// Corpus where the true answer chunk shares vocabulary with the chunk the sparse query hits
    /// directly, but contains *no* query term itself — the case RM3 is designed to rescue.
    fn rm3_corpus() -> Vec<String> {
        vec![
            // 0: directly hit by the query "telescope"; rich in astronomy vocabulary.
            "the telescope observed the distant galaxy nebula and faint orbiting comet".to_string(),
            // 1: filler, unrelated vocabulary.
            "lunch in the cafeteria is served from noon until two each weekday".to_string(),
            // 2: the ANSWER — astronomy vocabulary (galaxy, nebula, comet) but NOT the word
            //    "telescope", so a literal query misses it; RM3 expansion should lift it.
            "the galaxy and the nebula drift past a comet far beyond the orbiting moon".to_string(),
            // 3: filler.
            "parking validation stamps are available at the front reception desk".to_string(),
            // 4: filler.
            "recycling bins sit on every floor right next to the elevator bank".to_string(),
        ]
    }

    #[test]
    fn rm3_lifts_answer_sharing_vocabulary_with_top_chunk() {
        let chunks = rm3_corpus();
        let query = lex_words("telescope"); // sparse (1 term) → RM3 fires

        // δ-only ranking (no feedback): the answer chunk #2 has no query term → scores 0.
        let baseline = argsort_desc(&bm25_scores(&chunks, &query, BM25_PLUS_DELTA));
        let answer_baseline = baseline.iter().position(|&i| i == 2).unwrap();

        // Full path runs one RM3 round: expansion terms (galaxy/nebula/comet) drawn from the
        // top chunk #0 now reward chunk #2, pulling it up the ranking.
        let with_rm3 = bm25_rank(&chunks, &query);
        let answer_rm3 = with_rm3.iter().position(|&i| i == 2).unwrap();

        assert!(
            answer_rm3 < answer_baseline,
            "RM3 lifts the vocabulary-sharing answer chunk (was #{answer_baseline}, now #{answer_rm3})"
        );
    }

    #[test]
    fn rm3_is_a_noop_for_a_rich_decisive_query() {
        // A rich query (>RM3_SPARSE_QUERY_TERMS distinct terms) that decisively hits one chunk:
        // the trigger must not fire, so the BM25+ ranking is returned unchanged.
        let chunks = vec![
            "the quarterly revenue figure for the logistics division reached four million"
                .to_string(),
            "lunch in the cafeteria is served from noon until two each weekday".to_string(),
            "parking validation stamps are available at the front reception desk".to_string(),
            "recycling bins sit on every floor right next to the elevator bank".to_string(),
            "office working hours run from nine until five on weekdays only".to_string(),
        ];
        let query = lex_words("quarterly revenue logistics division four million figure");
        assert!(
            // sanity: the query really is in the non-sparse regime
            {
                let idx = Bm25Index::build(&chunks);
                idx.query_terms(&query).len() > RM3_SPARSE_QUERY_TERMS
            },
            "fixture must exercise the rich-query branch"
        );

        let baseline = argsort_desc(&bm25_scores(&chunks, &query, BM25_PLUS_DELTA));
        let with_rm3 = bm25_rank(&chunks, &query);
        assert_eq!(
            baseline, with_rm3,
            "rich decisive query: RM3 leaves the ranking untouched"
        );
        assert_eq!(with_rm3[0], 0, "the revenue chunk still ranks first");
    }

    // --- TextTiling chunk boundaries (Hearst CL 1997; Eisenstein & Barzilay EMNLP 2008) ------

    /// Two clearly distinct topics back to back (a garden topic, then a telescope topic), each
    /// several sentences. Each topic *recurs* its own content words (garden/plants/soil …;
    /// telescope/stars/sky …) so within-topic cohesion is high, while the two share no vocabulary
    /// — giving the block-comparison curve one sharp valley exactly at the seam (the recurrence is
    /// what TextTiling keys on; real prose reuses its topic terms the same way).
    fn two_topic_prose() -> String {
        "The garden soil was rich so the garden plants grew fast. \
         Garden plants need water and the soil keeps the plants healthy. \
         A healthy garden has deep soil and the plants love the garden. \
         The garden plants and garden soil made the garden thrive. \
         The telescope showed the stars and the night sky was clear. \
         Stars in the sky shone bright through the telescope each night. \
         The telescope tracked the stars while the sky stayed dark. \
         A clear sky let the telescope find faint stars in the night."
            .to_string()
    }

    #[test]
    fn texttiling_splits_between_two_topics_not_mid_topic() {
        let tiles = texttile(&two_topic_prose()).expect("two-topic prose should tile");
        assert!(tiles.len() >= 2, "at least one boundary placed");
        // The seam must fall between the topics: the first tile is all garden (no telescope
        // vocabulary), and some later tile carries the telescope vocabulary.
        assert!(
            tiles[0].contains("garden"),
            "first tile holds the garden topic"
        );
        assert!(
            !tiles[0].contains("telescope") && !tiles[0].contains("stars"),
            "the boundary is at the topic shift, not mid-astronomy"
        );
        assert!(
            tiles
                .iter()
                .any(|t| t.contains("telescope") || t.contains("stars")),
            "a later tile holds the telescope topic"
        );
    }

    #[test]
    fn budgeted_stage_drops_duplicate_paragraphs_and_cuts_tokens() {
        fn count_tokens(text: &str) -> usize {
            text.split_whitespace().count()
        }
        // A large context: one answer paragraph repeated many times (near-duplicate spam)
        // plus a few distinct paragraphs and a query. The stage must cut tokens and collapse
        // the duplicate block to a single representative + an elision marker.
        let mut paras: Vec<String> = Vec::new();
        for _ in 0..10 {
            paras.push(
                "The quarterly revenue figure for the logistics division was 4.2 million."
                    .to_string(),
            );
        }
        paras.push("Parking is available in the north lot for all visitors.".to_string());
        paras.push("Recycling bins are located on every floor near the elevators.".to_string());
        paras.push("Office hours run from nine to five on weekdays only.".to_string());
        let big = paras.join("\n\n");
        let body = json!({"model":"gpt-4o","messages":[
            {"role":"user","content":big},
            {"role":"user","content":"what was the quarterly revenue for logistics?"}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(retrieve_cfg(0.5))];
        let before = req
            .get_str("/messages/0/content")
            .map(count_tokens)
            .unwrap();
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "duplicate-heavy context is pruned");
        let kept = req.get_str("/messages/0/content").unwrap();
        assert!(kept.contains("revenue"), "the answer survives");
        assert!(kept.contains("omitted"), "redundant copies are elided");
        assert!(
            count_tokens(kept) < before,
            "tokens reduced ({} -> {})",
            before,
            count_tokens(kept)
        );
    }

    #[test]
    fn texttiling_falls_back_on_single_topic() {
        // One coherent topic, vocabulary recurring throughout: no cohesion valley → None, so the
        // caller keeps its existing line/blob chunking instead of an arbitrary cut.
        let single = "The logistics division shipped freight to every regional warehouse. \
                      Freight volumes rose as the division added new regional routes. \
                      Each warehouse tracked its freight against the division forecast. \
                      The division reviewed warehouse freight costs every quarter. \
                      Regional freight delays pushed the division to add more warehouses. \
                      Freight forecasts guided how the division staffed each warehouse."
            .to_string();
        assert_eq!(texttile(&single), None, "single-topic prose falls back");
    }

    #[test]
    fn texttiling_is_deterministic() {
        let text = two_topic_prose();
        assert_eq!(texttile(&text), texttile(&text), "tiling is deterministic");
    }

    #[test]
    fn texttiling_subdivides_oversized_tile_after_a_real_boundary() {
        // A short garden topic, then ONE long coherent telescope topic whose tile exceeds
        // TILE_MAX_TILE_CHARS. The garden|telescope cohesion valley is the real boundary; the
        // oversized telescope tile must then be subdivided at its deepest interior gap, so no
        // tile stays larger than the cap. Exercises the `split_oversized_tiles` post-pass.
        let mut s = String::from(
            "The garden soil was rich so the garden plants grew fast. \
             Garden plants need water and the garden soil keeps the plants healthy. \
             A healthy garden has deep soil and the plants love the garden. ",
        );
        // ~20 telescope sentences (one coherent topic) — comfortably over 1200 chars, so the
        // single forced subdivision leaves both halves under the cap.
        for _ in 0..20 {
            s.push_str(
                "The telescope tracked the bright stars across the clear night sky tonight. ",
            );
        }
        let tiles = texttile(&s).expect("a real boundary plus an oversized tile should tile");
        assert!(tiles.len() >= 3, "oversized telescope tile is subdivided");
        assert!(
            tiles[0].contains("garden") && !tiles[0].contains("telescope"),
            "the first (real) boundary still separates the topics"
        );
        assert!(
            tiles
                .iter()
                .all(|t| t.chars().count() <= TILE_MAX_TILE_CHARS),
            "no tile exceeds the max-char cap after subdivision"
        );
    }
}
