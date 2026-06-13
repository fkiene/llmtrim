//! Suffix-array maximal-repeat mining for the Stage E+ n-gram dictionary.
//!
//! Replaces the old fixed-window phrase miner (every word n-gram for n in a small
//! fixed range, counted in a hash map) with exact enumeration of *all* maximal
//! repeated word sequences, then a gain-driven greedy selection priced in real
//! target tokens.
//!
//! Method:
//! - Words are interned to `u32` ids using the stage's existing whitespace
//!   segmentation; each content segment is terminated by a unique sentinel id so a
//!   repeat can never span a segment boundary (or wrap into the legend).
//! - A suffix array over the id sequence is built by prefix doubling — O(n log² n),
//!   no external dependency — and the LCP array by Kasai's algorithm. Together they
//!   enumerate every maximal repeat in O(n log n) ("Efficient Repeat Finding via
//!   Suffix Arrays", arXiv:1304.0528).
//! - Candidate phrases are the repeats with length in `2..=MAX_PHRASE_WORDS` and
//!   count ≥ 2. Each is scored by the real token gain of substituting it for a short
//!   placeholder plus a one-line legend entry — the Re-Pair idea of recursively
//!   substituting the most *profitable* repeat (Larsson & Moffat; arXiv:1611.01479),
//!   adapted from pair frequency to true token cost.
//! - Selection is greedy by gain with overlap accounting: the chosen phrase claims
//!   its word spans, occurrences of other candidates overlapping a claimed span stop
//!   counting, and affected candidates are re-priced (n is prompt-sized, so a simple
//!   re-evaluation is fine). Ties break deterministically: longer phrase first, then
//!   the leftmost first occurrence.
//!
//! Prompt-sized input: the id sequence is at most the prompt's word count, so the
//! O(n log² n) construction and the greedy re-pricing are negligible next to the BPE
//! tokenization the gate already runs.

use std::collections::HashMap;

use crate::tokenizer::TokenCounter;

/// Longest phrase (in words) the miner will consider. A glossary entry past this
/// length almost never recurs verbatim, and it bounds the per-candidate work.
const MAX_PHRASE_WORDS: usize = 32;

/// A maximal-repeat candidate: a contiguous run of interned words that occurs more
/// than once. `starts` are the word-sequence offsets of its (non-overlapping within
/// this candidate) occurrences.
struct Candidate {
    /// Interned word ids making up the phrase (its length in words is `ids.len()`).
    ids: Vec<u32>,
    /// First-occurrence offset in the interned sequence (for the leftmost tie-break).
    first: usize,
    /// Occurrence start offsets in the interned sequence.
    starts: Vec<usize>,
}

/// Interns words to `u32` ids per `split_whitespace`, separating segments with unique
/// sentinels. Returns `(ids, vocab)` where `vocab[id]` is the original word string and
/// the sentinels are ids `>= vocab.len()` (never reused, so they can't form a repeat).
fn intern(segments: &[&str]) -> (Vec<u32>, Vec<String>) {
    let mut vocab: Vec<String> = Vec::new();
    let mut lookup: HashMap<&str, u32> = HashMap::new();
    let mut ids: Vec<u32> = Vec::new();
    // Sentinel positions are recorded as we go and patched once the real vocab size is
    // known (sentinels take ids strictly above every word, so they can't form a repeat).
    let mut sentinel_slots: Vec<usize> = Vec::new();
    for seg in segments {
        for w in seg.split_whitespace() {
            let id = *lookup.entry(w).or_insert_with(|| {
                let id = vocab.len() as u32;
                vocab.push(w.to_string());
                id
            });
            ids.push(id);
        }
        // One sentinel after every segment (including the last): guarantees the sequence
        // ends on a unique symbol (the doubling sort relies on it) and walls off
        // cross-segment repeats. Each is patched to a distinct id below.
        sentinel_slots.push(ids.len());
        ids.push(0); // patched below
    }
    let base = vocab.len() as u32;
    for (k, &pos) in sentinel_slots.iter().enumerate() {
        ids[pos] = base + k as u32;
    }
    (ids, vocab)
}

/// Suffix array of `s` by prefix doubling — O(n log² n). `s` must end in a value that
/// is unique (a sentinel), so suffixes are totally ordered without special-casing.
fn suffix_array(s: &[u32]) -> Vec<u32> {
    let n = s.len();
    if n == 0 {
        return Vec::new();
    }
    let mut sa: Vec<u32> = (0..n as u32).collect();
    // `rank[i]` is the order key of the suffix at i for the current prefix length.
    let mut rank: Vec<i64> = s.iter().map(|&c| c as i64).collect();
    let mut tmp: Vec<i64> = vec![0; n];
    let mut k = 1usize;
    while k < n {
        // Sort by (rank[i], rank[i+k]); -1 marks "past the end" (sorts first).
        let key = |i: usize| -> (i64, i64) {
            let second = if i + k < n { rank[i + k] } else { -1 };
            (rank[i], second)
        };
        sa.sort_by_key(|&a| key(a as usize));
        // Recompute ranks from the new order; equal adjacent keys share a rank.
        tmp[sa[0] as usize] = 0;
        for i in 1..n {
            let prev = sa[i - 1] as usize;
            let cur = sa[i] as usize;
            tmp[cur] = tmp[prev] + i64::from(key(prev) != key(cur));
        }
        std::mem::swap(&mut rank, &mut tmp);
        if rank[sa[n - 1] as usize] == (n as i64 - 1) {
            break; // all ranks distinct → fully sorted
        }
        k <<= 1;
    }
    sa
}

/// LCP array (Kasai): `lcp[i]` is the length of the longest common prefix of the
/// suffixes at `sa[i-1]` and `sa[i]`; `lcp[0] = 0`.
fn kasai_lcp(s: &[u32], sa: &[u32]) -> Vec<usize> {
    let n = s.len();
    let mut rank = vec![0usize; n];
    for (i, &p) in sa.iter().enumerate() {
        rank[p as usize] = i;
    }
    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        if rank[i] > 0 {
            let j = sa[rank[i] - 1] as usize;
            while i + h < n && j + h < n && s[i + h] == s[j + h] {
                h += 1;
            }
            lcp[rank[i]] = h;
            h = h.saturating_sub(1);
        } else {
            h = 0;
        }
    }
    lcp
}

/// Enumerate repeated word sequences (count ≥ 2, length `2..=MAX_PHRASE_WORDS`) from
/// the suffix/LCP arrays, capped at the [`MAX_PHRASE_WORDS`]-word prefix of each LCP
/// interval. Returns one [`Candidate`] per distinct phrase, occurrences de-overlapped
/// within the phrase (left-to-right), so `starts.len()` is a sound substitution count.
fn enumerate_repeats(s: &[u32], sa: &[u32], lcp: &[usize], sentinel_floor: u32) -> Vec<Candidate> {
    let n = s.len();
    // For a phrase length L, the set of suffixes that share an L-length prefix is a
    // maximal run of SA positions whose pairwise LCP ≥ L. We collect distinct phrases
    // by scanning each adjacent pair: the prefixes of length 2..=min(lcp, cap) that
    // start at sa[i] are repeated (they also occur at sa[i-1]). Deduplicate by the id
    // slice, union the occurrence offsets, then de-overlap per phrase.
    let mut seen: HashMap<&[u32], usize> = HashMap::new();
    let mut cands: Vec<Candidate> = Vec::new();
    for i in 1..n {
        let common = lcp[i].min(MAX_PHRASE_WORDS);
        if common < 2 {
            continue;
        }
        let a = sa[i - 1] as usize;
        let b = sa[i] as usize;
        for len in 2..=common {
            // A phrase containing a sentinel is not real text — skip it (sentinels sit
            // above every word id, so a quick scan of the prefix suffices).
            let start = b;
            let slice = &s[start..start + len];
            if slice.iter().any(|&id| id >= sentinel_floor) {
                break; // sentinel inside the window → no longer phrase is valid either
            }
            match seen.get(slice).copied() {
                Some(idx) => {
                    let c: &mut Candidate = &mut cands[idx];
                    c.starts.push(a);
                    c.starts.push(b);
                }
                None => {
                    let idx = cands.len();
                    // Key borrows from `s` (stable for the function's lifetime).
                    seen.insert(slice, idx);
                    cands.push(Candidate {
                        ids: slice.to_vec(),
                        first: a.min(b),
                        starts: vec![a, b],
                    });
                }
            }
        }
    }
    // De-overlap each phrase's own occurrences and tidy bookkeeping.
    for c in cands.iter_mut() {
        c.starts.sort_unstable();
        c.starts.dedup();
        let len = c.ids.len();
        let mut kept: Vec<usize> = Vec::with_capacity(c.starts.len());
        let mut last_end = 0usize;
        let mut first_set = false;
        for &st in &c.starts {
            if !first_set || st >= last_end {
                if !first_set {
                    c.first = st;
                    first_set = true;
                }
                kept.push(st);
                last_end = st + len;
            }
        }
        c.starts = kept;
    }
    cands.retain(|c| c.starts.len() >= 2);
    cands
}

/// One selected phrase: the original word strings joined by a single space (byte- and
/// segmentation-compatible with the stage's whole-word replace), in commit order.
pub(crate) struct Selection {
    pub phrase: String,
}

/// Mine `segments` for the highest-token-gain phrase dictionary.
///
/// Returns up to `max_entries` phrases, in the order they should be assigned
/// placeholders `§1, §2, …`, such that substituting each (whole-word) and adding a
/// legend entry `§k=<phrase>` strictly reduces tokens under `counter`.
///
/// `placeholder_for` maps the 1-based placeholder index to its placeholder string
/// (e.g. `|k| format!("§{k}")`) so the gain is priced against the real marker the
/// stage will emit, not an assumed one.
pub(crate) fn mine(
    segments: &[&str],
    max_entries: usize,
    counter: &dyn TokenCounter,
    placeholder_for: impl Fn(usize) -> String,
) -> Vec<Selection> {
    if max_entries == 0 {
        return Vec::new();
    }
    let (ids, vocab) = intern(segments);
    if ids.len() < 3 {
        return Vec::new();
    }
    let sentinel_floor = vocab.len() as u32;
    let sa = suffix_array(&ids);
    let lcp = kasai_lcp(&ids, &sa);
    let cands = enumerate_repeats(&ids, &sa, &lcp, sentinel_floor);
    if cands.is_empty() {
        return Vec::new();
    }

    // Render each candidate's phrase text once (joined originals), and pre-count its
    // per-occurrence phrase tokens — invariant across the greedy loop.
    let phrase_text: Vec<String> = cands
        .iter()
        .map(|c| {
            c.ids
                .iter()
                .map(|&id| vocab[id as usize].as_str())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    let phrase_tokens: Vec<usize> = phrase_text.iter().map(|p| counter.count(p)).collect();

    // `claimed[pos]` is true once some selected phrase owns the word at sequence
    // offset `pos`; overlapping occurrences of other candidates then stop counting.
    let mut claimed = vec![false; ids.len()];
    let mut chosen: Vec<Selection> = Vec::new();
    let mut used: Vec<bool> = vec![false; cands.len()];

    // Gain of phrase `i` at the next placeholder index, counting only occurrences whose
    // span is entirely unclaimed:
    //   gain = live_occ * (phrase_tokens - placeholder_tokens) - legend_entry_tokens
    // The legend entry is `§k=<phrase>` plus the `; ` separator the stage joins with;
    // pricing the separator keeps the estimate aligned with the real injected legend.
    let gain_of = |i: usize, next_index: usize, claimed: &[bool]| -> (i64, usize) {
        let len = cands[i].ids.len();
        let live: usize = cands[i]
            .starts
            .iter()
            .filter(|&&st| !claimed[st..st + len].iter().any(|&c| c))
            .count();
        if live < 2 {
            return (i64::MIN, live);
        }
        let ph = placeholder_for(next_index);
        let ph_tokens = counter.count(&ph);
        let legend_entry = format!("{ph}={}; ", phrase_text[i]);
        let legend_tokens = counter.count(&legend_entry);
        let per = phrase_tokens[i] as i64 - ph_tokens as i64;
        let gain = (live as i64) * per - legend_tokens as i64;
        (gain, live)
    };

    while chosen.len() < max_entries {
        let next_index = chosen.len() + 1;
        // Pick the max-gain candidate; tie-break longer phrase, then leftmost first
        // occurrence, then lexicographically smaller phrase text (full determinism).
        let mut best: Option<(usize, i64)> = None;
        for i in 0..cands.len() {
            if used[i] {
                continue;
            }
            let (g, _live) = gain_of(i, next_index, &claimed);
            if g <= 0 {
                continue;
            }
            let take = match best {
                None => true,
                Some((bi, bg)) => {
                    g > bg
                        || (g == bg
                            && better_tiebreak(
                                cands[i].ids.len(),
                                cands[i].first,
                                &phrase_text[i],
                                cands[bi].ids.len(),
                                cands[bi].first,
                                &phrase_text[bi],
                            ))
                }
            };
            if take {
                best = Some((i, g));
            }
        }
        let Some((bi, _)) = best else {
            break; // no positive-gain candidate remains
        };
        // Commit: claim every word span of its live occurrences.
        used[bi] = true;
        let len = cands[bi].ids.len();
        for &st in &cands[bi].starts {
            if !claimed[st..st + len].iter().any(|&c| c) {
                for c in &mut claimed[st..st + len] {
                    *c = true;
                }
            }
        }
        chosen.push(Selection {
            phrase: phrase_text[bi].clone(),
        });
    }
    chosen
}

/// Deterministic tie-break when two candidates have equal gain: prefer the longer
/// phrase, then the earlier first occurrence, then the lexicographically smaller text.
#[allow(clippy::too_many_arguments)]
fn better_tiebreak(
    a_len: usize,
    a_first: usize,
    a_text: &str,
    b_len: usize,
    b_first: usize,
    b_text: &str,
) -> bool {
    (a_len, std::cmp::Reverse(a_first), std::cmp::Reverse(b_text))
        > (b_len, std::cmp::Reverse(b_first), std::cmp::Reverse(a_text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_array_matches_naive() {
        // Compare doubling SA against a brute-force suffix sort on a small sequence.
        let s: Vec<u32> = vec![3, 1, 2, 1, 2, 1, 0];
        let sa = suffix_array(&s);
        let mut naive: Vec<usize> = (0..s.len()).collect();
        naive.sort_by(|&a, &b| s[a..].cmp(&s[b..]));
        let sa_us: Vec<usize> = sa.iter().map(|&x| x as usize).collect();
        assert_eq!(sa_us, naive, "doubling SA must equal the naive suffix sort");
    }

    #[test]
    fn lcp_is_consistent_with_sa() {
        let s: Vec<u32> = vec![2, 1, 2, 1, 2, 1, 0];
        let sa = suffix_array(&s);
        let lcp = kasai_lcp(&s, &sa);
        // Recompute each LCP directly and compare.
        for i in 1..s.len() {
            let a = sa[i - 1] as usize;
            let b = sa[i] as usize;
            let mut h = 0;
            while a + h < s.len() && b + h < s.len() && s[a + h] == s[b + h] {
                h += 1;
            }
            assert_eq!(lcp[i], h, "Kasai LCP must match the direct prefix length");
        }
    }

    #[test]
    fn enumerate_finds_the_long_repeat() {
        // "a b c d e" repeated; interner gives ids, miner should surface the full run.
        let seg = "a b c d e x a b c d e y a b c d e";
        let (ids, vocab) = intern(&[seg]);
        let floor = vocab.len() as u32;
        let sa = suffix_array(&ids);
        let lcp = kasai_lcp(&ids, &sa);
        let cands = enumerate_repeats(&ids, &sa, &lcp, floor);
        // The maximal repeat "a b c d e" (5 words) must be present with 3 occurrences.
        let abcde: Vec<u32> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|w| ids[seg.split_whitespace().position(|x| x == *w).unwrap()])
            .collect();
        let hit = cands
            .iter()
            .find(|c| c.ids == abcde)
            .expect("full repeat found");
        assert_eq!(hit.starts.len(), 3, "all three non-overlapping occurrences");
    }

    #[test]
    fn no_repeat_spans_a_segment_boundary() {
        // "z y" ends seg 0 and "z y" would only "repeat" across the boundary — the
        // sentinel must prevent that from being mined as a single phrase.
        let segs = ["foo bar z y", "z y baz qux"];
        let (ids, vocab) = intern(&segs);
        let floor = vocab.len() as u32;
        let sa = suffix_array(&ids);
        let lcp = kasai_lcp(&ids, &sa);
        let cands = enumerate_repeats(&ids, &sa, &lcp, floor);
        // "z y" occurs once per segment → it IS a legit cross-segment repeat (count 2),
        // but the *sentinel* between them must keep them as two separate occurrences,
        // never a 4-word "z y z y" phrase. Assert no candidate contains a sentinel id.
        for c in &cands {
            assert!(
                c.ids.iter().all(|&id| id < floor),
                "no phrase may contain a sentinel"
            );
        }
    }
}
