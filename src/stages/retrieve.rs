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
    let kept = prune_sentences(&chunks, query, &stops, keep_ratio);
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
    // Query-less ranking uses TextRank, whose dense n×n similarity matrix is O(n²) memory —
    // tens of thousands of line-chunks (a large log) would allocate gigabytes and abort the
    // process. Above the cap, skip centrality entirely and fall back to a head+tail keep
    // (boundary-safe, O(n)). Never block the user.
    let ranked = if query.is_empty() {
        if chunks.len() > TEXTRANK_MAX_CHUNKS {
            return Some(rebuild(&chunks, &head_tail_keep(keep, chunks.len()), sep));
        }
        textrank_rank(&chunks)
    } else {
        bm25_rank(&chunks, query)
    };
    if !cfg.reorder && !cfg.mmr {
        return Some(rebuild(
            &chunks,
            &budgeted_select(&chunks, &ranked, keep, cfg),
            sep,
        ));
    }
    let mut chosen = if cfg.mmr {
        mmr_order(&ranked, &chunks, keep, cfg.mmr_lambda)
    } else {
        ranked.iter().copied().take(keep).collect::<Vec<usize>>()
    };
    pin_boundaries(&mut chosen, chunks.len());
    if cfg.reorder {
        Some(rebuild_ordered(&chunks, &u_shape(&chosen), sep))
    } else {
        chosen.sort_unstable();
        chosen.dedup();
        Some(rebuild(&chunks, &chosen, sep))
    }
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
    let mut keep = vec![false; n];
    for (i, k) in keep.iter_mut().enumerate() {
        let neighbour = (i > 0 && relevant[i - 1]) || (i + 1 < n && relevant[i + 1]);
        if relevant[i] || neighbour || i == 0 || i == n - 1 {
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
            .filter(|&i| keep[i] && i != 0 && i != n - 1 && i != best)
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

/// Split text into chunks: by blank-line paragraphs, falling back to lines. Returns the
/// chunks and the matching join separator (so [`rebuild`] rejoins at the same grain).
fn chunk_with_sep(text: &str) -> (Vec<String>, &'static str) {
    let paras: Vec<String> = text
        .split("\n\n")
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect();
    if paras.len() > 1 {
        return (paras, PARA_SEP);
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

/// Rank chunk indices by BM25 relevance to `query` (best first) using the `bm25`
/// crate's in-memory scorer. Chunks the scorer omits (zero query overlap) are
/// appended in original order, so the result is always a full permutation.
fn bm25_rank(chunks: &[String], query: &[String]) -> Vec<usize> {
    use bm25::{DefaultTokenizer, EmbedderBuilder, Scorer};
    let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    // Detect the corpus language for stemming + stopwords, but DISABLE bm25's unicode
    // normalization: its default transliterates non-Latin scripts to ASCII (CJK → romaji,
    // Cyrillic/Greek → Latin), which mangles CJK *and* breaks bm25's own non-Latin
    // stemmers (Russian/Greek/Arabic/Tamil would be stemmed on transliterated text). With
    // it off, splitting stays Unicode-aware (UAX#29, the same `unicode_words` as
    // `lex_words`) and stemming runs on the real script — universal across languages.
    let tokenizer = DefaultTokenizer::builder()
        .language_mode(bm25_language(&refs.join(" ")))
        .normalization(false)
        .build();
    let embedder =
        EmbedderBuilder::<u32>::with_tokenizer_and_fit_to_corpus(tokenizer, &refs).build();
    let mut scorer = Scorer::<usize>::new();
    for (i, c) in refs.iter().enumerate() {
        scorer.upsert(&i, embedder.embed(c));
    }
    let q = embedder.embed(&query.join(" "));
    let mut order: Vec<usize> = scorer.matches(&q).into_iter().map(|m| m.id).collect();
    // Append chunks the scorer dropped (score 0) so callers get a full permutation. Track
    // seen indices in a bitset so completion is O(n), not O(n²) via `Vec::contains`.
    let mut seen = vec![false; chunks.len()];
    for &i in &order {
        if i < seen.len() {
            seen[i] = true;
        }
    }
    for (i, &was_seen) in seen.iter().enumerate() {
        if !was_seen {
            order.push(i);
        }
    }
    order
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
        let kept = prune_sentences(&chunks, &query, &stops, 0.9);
        assert!(
            kept.contains(&0) && kept.contains(&4),
            "relevant sentences kept"
        );
        assert!(!kept.contains(&2), "zero-overlap middle filler dropped");
        let mut sorted = kept.clone();
        sorted.sort_unstable();
        assert_eq!(kept, sorted, "kept in original order");

        // Aggressive cap: drop the neighbours too, but never the answer (#0/#4).
        let tight = prune_sentences(&chunks, &query, &stops, 0.4);
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
}
