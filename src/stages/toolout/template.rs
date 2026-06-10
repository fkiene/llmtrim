//! Drain-style log-template collapse — information-preserving (feature #5).
//!
//! Logs are full of lines that share a fixed template and differ only in variable
//! tokens (timestamps, ids, counts): exact line-dedup (Stage E) can't fold them
//! because no two lines are byte-identical. This collapses a *consecutive* run of N
//! lines with the same template into one representative carrying every original's
//! variable values inline:
//!
//! ```text
//! Connection to {} timed out after {}ms [×3: (db-01,30) (db-02,12) (db-07,5)]
//! ```
//!
//! The tuples are positional (they map back onto the `{}` slots in order), so the run is
//! reconstructible — information-preserving (every value survives; runs of whitespace are
//! normalized to a single space, so the model reads the data, not the column padding) —
//! yet far shorter when the static part dominates. Normalizing whitespace is what lets
//! *aligned* command output (`ls -l`, `ps aux`, `df` — fixed-width columns whose padding
//! varies per row) collapse to one template instead of fragmenting. The collapse is
//! applied only when it actually shrinks the run (char count), so it never inflates; the
//! model reads the `[×N: …]` notation directly (self-descriptive, like Stage E's `[×N]`).
//!
//! Variable tokens are locale-independent (numbers, hex/UUID, ISO-8601 timestamps,
//! IPv4, quoted strings), so masking is language-agnostic.
//!
//! # Global (non-adjacent) pass — [`collapse_global`]
//!
//! The consecutive pass above misses *interleaved* repeats: parallel build output
//! (`cargo` across crates, `pytest -n`, npm workspaces) emits the same template from
//! several workers at once, so identical-template lines alternate rather than run.
//! [`collapse_global`] catches those with a deterministic bucket → MinHash-LSH → voting
//! pipeline, folding each non-adjacent group into the *same* `[×N: …]` representation
//! (information-preserving, identical downstream contract):
//!
//!  1. **Bucket** (cheap, [`bucket_key`]): key = (token count, the chars at ~0/25/50/75/100 %
//!     relative positions). Char-boundary safe (Unicode scalar values, never byte offsets).
//!     LogLSHD-style coarse grouping (arXiv:2504.02172, 2025), with a *deterministic* key —
//!     no random position sampling. It only has to be a cheap first cut: when a variable
//!     token lands on a sampled position it shatters a template into one-member buckets, and
//!     Stage 2 re-unites them.
//!  2. **Merge** ([`gaoya`] MinHash-LSH): each bucket representative is reduced to its
//!     *value-masked* token shingles ([`masked_tokens`] — numbers/hex/UUID/timestamps/IPs →
//!     `{}`) and MinHashed; buckets whose signatures estimate Jaccard ≥ [`JACCARD_THRESHOLD`]
//!     union. Masking *only the merge key* (we still vote on raw tokens) is what lets the
//!     shattered singletons regroup and length-varied siblings (`crate_0` / `crate_10`) join,
//!     while distinct templates stay apart.
//!  3. **Template** (Brain-style positional voting — Yu et al., "Brain: Log Parsing with
//!     Bidirectional Parallel Tree", IEEE TSC 2023): across the first [`VOTE_SAMPLE`]
//!     members (deterministic — first-N, never random), a token position is a *constant*
//!     when one token uniquely dominates it (≥ [`VOTE_SHARE`] and strictly the most), else a
//!     *variable* slot; adjacent variable slots merge into one `{}`. Training-free; constants
//!     emerge from per-position agreement, not a regex, so error codes (`E0308`) that vary
//!     across members survive as captured values and constant ones stay in the template —
//!     never silently dropped (LogLSHD's alphabetic-only filter would have discarded them).
//!
//! A fold only happens when the voted template has a value-shaped variable slot
//! ([`has_value_shaped_slot`]) and it actually shrinks the group (char count). The
//! value-shaped guard keeps the existing log/prose boundary: a sentence frame that merely
//! shares fixed words ("The {} review of {} examined {} …") votes to a template too, but its
//! slots hold plain words, not values — so it's declined and left to the retrieve stage.
//!
//! Two deviations from LogLSHD are deliberate (both noted above): first-N instead of random
//! sampling (determinism is a hard constraint), and alphanumeric — not alphabetic — tokens
//! (build-log error codes carry meaning). The pass runs *after* the consecutive collapse
//! (cheap first), folds only groups of ≥ [`MIN_GLOBAL_REPEAT`] members, and caps the merge at
//! [`MAX_BUCKETS`] buckets, so the fast path stays fast.

use std::collections::HashMap;

use gaoya::minhash::{MinHasher, MinHasher32};
use once_cell::sync::Lazy;
use regex::Regex;

/// Minimum consecutive same-template lines before a run is collapsed.
const MIN_RUN: usize = 3;

/// Minimum members a merged group must have before the global pass will fold it — fewer
/// than this and the `[×N: …]` notation costs more than the fold saves.
const MIN_GLOBAL_REPEAT: usize = 3;

/// Number of buckets we'll MinHash/merge — a hard ceiling so a pathological input (tens of
/// thousands of distinct templates) can't turn the O(b²) merge into a blow-up. Buckets past
/// this (in first-seen order) keep their lines unfolded.
const MAX_BUCKETS: usize = 512;

/// MinHash signature width. 64 hashes keeps the Jaccard estimate within a few percent at
/// the 0.9 threshold while staying cheap (one pass over each representative's tokens).
const MINHASH_NUM_HASHES: usize = 64;

/// Fixed seed for the MinHasher's permutation coefficients — output must be identical run
/// to run (determinism is a hard constraint), so we never lean on a library RNG default; we
/// pin it.
const MINHASH_SEED: u64 = 1;

/// Two buckets merge into one template group when their masked-token MinHash signatures
/// estimate Jaccard ≥ this. 0.9 = "the same template modulo a token or two" (LogLSHD's
/// recommended merge threshold).
const JACCARD_THRESHOLD: f64 = 0.9;

/// How many members of a group vote on the template (first-N, deterministic). 10 matches
/// LogLSHD's sample size; more rarely changes the constant/variable verdict.
const VOTE_SAMPLE: usize = 10;

/// A token position is a *constant* (part of the template) when **one** token uniquely
/// dominates it — held by at least this share of the voters *and* strictly more of them than
/// any other token. Otherwise it's a variable `{}` slot. The strict-uniqueness rule makes
/// the verdict deterministic (no tie-break on equal-frequency tokens) and stops a 2-member
/// group from reading every position as a "constant".
const VOTE_SHARE: f64 = 0.5;

/// Matches one variable token. Ordered most-specific-first (quoted string, timestamp,
/// UUID, hex, IPv4) so those win over the trailing bare-number alternative.
static VARIABLE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r#""[^"]*""#, // quoted string
        r#"|\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?"#, // ISO-8601
        r"|\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", // UUID
        r"|\b0x[0-9a-fA-F]+\b", // hex literal
        r"|\b[0-9a-fA-F]{12,}\b", // long hex (hashes)
        r"|\b\d{1,3}(?:\.\d{1,3}){3}\b", // IPv4
        r"|\d+(?:\.\d+)?", // unsigned integer / decimal (a leading `-` stays in the
                      // template — it's a separator in `db-01` as often as a sign)
    ))
    .unwrap()
});

/// Collapse consecutive same-template runs in `text`, losslessly. Returns the rebuilt
/// text and whether any run was actually folded. The boolean — not a string compare with
/// the input — is the authoritative "did anything fold?" signal: rebuilding via
/// `join("\n")` strips a trailing newline, so a raw `collapsed != text` reads as "changed"
/// for any input ending in `\n` even when nothing folded, which would defeat the callers'
/// prose-decline / no-op gates.
pub fn collapse(text: &str) -> (String, bool) {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < MIN_RUN {
        return (text.to_string(), false);
    }
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut folded = false;
    let mut i = 0;
    while i < lines.len() {
        let tpl = template_of(lines[i]).0;
        let mut j = i + 1;
        while j < lines.len() && template_of(lines[j]).0 == tpl {
            j += 1;
        }
        let run = &lines[i..j];
        // Only collapse a real run whose template actually has variable slots — runs
        // of identical (slot-free) lines are Stage E's job, not ours.
        if run.len() >= MIN_RUN && tpl.contains("{}") {
            let collapsed = render_run(&tpl, run);
            let original_len: usize = run.iter().map(|l| l.len() + 1).sum();
            if collapsed.len() < original_len {
                out.push(collapsed);
                folded = true;
                i = j;
                continue;
            }
        }
        out.extend(run.iter().map(|l| (*l).to_string()));
        i = j;
    }
    (out.join("\n"), folded)
}

/// `(template, variables)` for one line: each variable token replaced by `{}`, the
/// matched values collected left-to-right.
fn template_of(line: &str) -> (String, Vec<String>) {
    let mut tpl = String::with_capacity(line.len());
    let mut vars = Vec::new();
    let mut last = 0;
    for m in VARIABLE.find_iter(line) {
        push_collapsed(&mut tpl, &line[last..m.start()]);
        tpl.push_str("{}");
        vars.push(m.as_str().to_string());
        last = m.end();
    }
    push_collapsed(&mut tpl, &line[last..]);
    (tpl, vars)
}

/// Append `s` to the template with every run of whitespace collapsed to a single space
/// (and no double space across a `{}` boundary). This makes the template insensitive to
/// column-alignment padding, so rows of aligned output share one template.
fn push_collapsed(out: &mut String, s: &str) {
    let mut prev_ws = out.ends_with(' ');
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_ws {
                out.push(' ');
                prev_ws = true;
            }
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
}

/// Render `<template> [×N: (tuple0) (tuple1) …]`, one comma-joined tuple of variable
/// values per original line, positional against the `{}` slots.
fn render_run(tpl: &str, run: &[&str]) -> String {
    let tuples: Vec<String> = run
        .iter()
        .map(|l| format!("({})", template_of(l).1.join(",")))
        .collect();
    format!("{tpl} [×{}: {}]", run.len(), tuples.join(" "))
}

// ── Global (non-adjacent) collapse ──────────────────────────────────────────────────

/// Run the cheap consecutive [`collapse`] first, then fold *non-adjacent* same-template
/// groups (interleaved parallel-build output) the consecutive pass can't reach. Returns
/// the rebuilt text and whether **either** pass folded anything. Information-preserving:
/// every member's values survive in the `[×N: …]` tuples, in first-seen order, exactly as
/// the consecutive pass already does — so downstream scoring / windowing is unchanged.
///
/// This is the entry point the log and plaintext compressors call; the consecutive-only
/// [`collapse`] stays public for its own focused tests.
pub fn collapse_global(text: &str) -> (String, bool) {
    let (consecutive, folded_consec) = collapse(text);
    let lines: Vec<&str> = consecutive.lines().collect();
    if lines.len() < MIN_GLOBAL_REPEAT {
        return (consecutive, folded_consec);
    }

    // Stage 1 — bucket. A line already folded by the consecutive pass carries the `[×N:`
    // marker; leave it be (re-bucketing a representative would double-count). Empty/blank
    // lines are structure, never folded. The cheap key over-fragments on value length (a
    // 1- vs 2-digit id shifts the relative-position anchors) — Stage 2 re-merges those.
    let mut buckets: HashMap<BucketKey, Vec<usize>> = HashMap::new();
    let mut order: Vec<BucketKey> = Vec::new(); // first-seen bucket order → determinism
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() || line.contains("[×") {
            continue;
        }
        let key = bucket_key(line);
        let entry = buckets.entry(key.clone()).or_insert_with(|| {
            order.push(key.clone());
            Vec::new()
        });
        entry.push(i);
    }

    // Keep buckets in first-seen order, capped (so the O(b²) merge can't blow up). We do
    // *not* drop singletons: the anchor key shatters a recurring template into one-member
    // buckets whenever a variable token lands on a sampled position (`alpha worker 0 done`,
    // `…1…`, `…2…` each key differently). Stage 2 is what regroups them.
    let work: Vec<BucketKey> = order.into_iter().take(MAX_BUCKETS).collect();

    // Stage 2 — MinHash-LSH merge. Each bucket's representative is reduced to its
    // *value-masked* token shingles ([`masked_tokens`] — numbers/hex/UUID/timestamps/IPs →
    // `{}` via the same [`VARIABLE`] regex the consecutive pass uses), then MinHashed. Masking
    // *only for the merge signature* (we still vote on raw tokens below) is what lets two
    // singleton buckets of the same template — `alpha worker 0 done` and `alpha worker 1
    // done` → both `alpha worker {} done` — estimate Jaccard ~1 and union, recovering the
    // grouping the anchor key shattered. It also folds in length-varied siblings (`crate_0`
    // / `crate_10`). Distinct templates stay apart (`alpha …` vs `beta …` share only 3 of 5
    // shingles → below threshold).
    let hasher = MinHasher32::new_with_hasher_and_seed(
        MINHASH_NUM_HASHES,
        gaoya::minhash::SipHasher24BuildHasher::default(),
        MINHASH_SEED,
    );
    let sigs: Vec<Vec<u32>> = work
        .iter()
        .map(|k| {
            let rep = lines[buckets[k][0]];
            hasher.create_signature(masked_tokens(rep).into_iter())
        })
        .collect();
    let groups = merge_buckets(&work, &sigs);

    // Stage 3 — per merged group: re-vote the template over *all* members (so it reflects the
    // full group), then fold them into one representative carrying every member's values.
    let mut fold_at: HashMap<usize, String> = HashMap::new(); // first-member idx → rendered
    let mut drop_line = vec![false; lines.len()];
    let mut folded_global = false;
    for group in &groups {
        // All member line indices across the merged buckets, in original (encounter) order.
        let mut members: Vec<usize> = group
            .iter()
            .flat_map(|k| buckets[k].iter().copied())
            .collect();
        members.sort_unstable();
        if members.len() < MIN_GLOBAL_REPEAT {
            continue;
        }
        let member_lines: Vec<&str> = members.iter().map(|&i| lines[i]).collect();
        let tpl = vote_template(&member_lines);
        // A template with no variable slot means these lines are byte-identical modulo
        // whitespace — that's exact/near dedup's job (Stage E), not template collapse.
        if !tpl.contains("{}") {
            continue;
        }
        // Prose guard: only fold *machine* templates, where the variables are values
        // (numbers, ids, codes, versions, timestamps). Sentences that happen to share a
        // fixed frame — "The {} review of {} examined {} …" — would also vote to a template,
        // but their slots hold plain words; folding those is the retrieve stage's call, not
        // ours. Require at least one slot whose values are predominantly value-shaped
        // ([`value_shaped`], digit- or structure-bearing — language-universal, no word list).
        if !has_value_shaped_slot(&tpl, &member_lines) {
            continue;
        }
        let rendered = render_voted(&tpl, &member_lines);
        // Only fold when it shrinks: the representative vs. the bytes of every member it
        // replaces (newlines included, matching the consecutive pass's accounting).
        let original_len: usize = member_lines.iter().map(|l| l.len() + 1).sum();
        if rendered.len() >= original_len {
            continue;
        }
        let first = members[0];
        for &i in &members[1..] {
            drop_line[i] = true;
        }
        fold_at.insert(first, rendered);
        folded_global = true;
    }

    if !folded_global {
        return (consecutive, folded_consec);
    }

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        if let Some(rep) = fold_at.remove(&i) {
            out.push(rep);
        } else if !drop_line[i] {
            out.push((*line).to_string());
        }
    }
    (out.join("\n"), true)
}

/// A line's coarse bucket key: token count plus the chars at five fixed relative positions
/// (0/25/50/75/100 %). Char-boundary safe — indexes Unicode scalar values, never byte
/// offsets — so it never splits a codepoint and works in any script. Deliberately *coarse*:
/// when a value's length changes the middle anchors shift and same-template lines split into
/// separate buckets; the Stage-2 MinHash merge re-unites them, so this only needs to be a
/// cheap candidate pre-grouping, not the final word.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct BucketKey {
    token_count: usize,
    anchors: [Option<char>; 5],
}

fn bucket_key(line: &str) -> BucketKey {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let at = |num: usize, den: usize| -> Option<char> {
        if n == 0 {
            return None;
        }
        // floor(num/den * (n-1)) — last index is n-1, so 100 % lands on the final char.
        let idx = num * (n - 1) / den;
        chars.get(idx).copied()
    };
    BucketKey {
        token_count: tokens(line).len(),
        anchors: [at(0, 4), at(1, 4), at(2, 4), at(3, 4), at(4, 4)],
    }
}

/// Whitespace-split positional tokens (Unicode whitespace via `split_whitespace`). Unlike
/// [`template_of`]'s regex masking, this keeps every token verbatim — including alphanumeric
/// error codes like `E0308` (LogLSHD drops these with an alphabetic-only filter; build logs
/// need them). These are the positions the template vote runs over, so nothing is dropped.
fn tokens(line: &str) -> Vec<&str> {
    line.split_whitespace().collect()
}

/// Value-masked token shingles for the *merge* signature only — numbers, hex, UUIDs,
/// timestamps, IPs collapse to `{}` via the same [`VARIABLE`] regex the consecutive pass
/// uses, then split on whitespace. Two lines of one template that differ only in such values
/// shingle identically, so MinHash unions their (anchor-shattered) buckets. Voting still runs
/// on the *raw* [`tokens`], so this masking never costs information — a code the regex happens
/// to mask here still survives verbatim as a captured value downstream.
fn masked_tokens(line: &str) -> Vec<String> {
    template_of(line)
        .0
        .split_whitespace()
        .map(str::to_string)
        .collect()
}

/// Union-find merge of `work` buckets: two buckets join when their masked-token MinHash
/// signatures estimate Jaccard ≥ [`JACCARD_THRESHOLD`]. Returns groups of bucket keys, each
/// in first-seen (ascending) order — deterministic. O(b²) in the bucket count, bounded by
/// [`MAX_BUCKETS`].
fn merge_buckets(work: &[BucketKey], sigs: &[Vec<u32>]) -> Vec<Vec<BucketKey>> {
    let n = work.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]]; // path halving
            x = parent[x];
        }
        x
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if gaoya::minhash::compute_minhash_similarity(&sigs[i], &sigs[j]) >= JACCARD_THRESHOLD {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[ri.max(rj)] = ri.min(rj); // lower root wins → stable order
                }
            }
        }
    }
    // Collect members per root, preserving first-seen (ascending index) order.
    let mut by_root: HashMap<usize, Vec<BucketKey>> = HashMap::new();
    let mut root_order: Vec<usize> = Vec::new();
    for (i, key) in work.iter().enumerate() {
        let r = find(&mut parent, i);
        by_root
            .entry(r)
            .or_insert_with(|| {
                root_order.push(r);
                Vec::new()
            })
            .push(key.clone());
    }
    // Drain in root order; every root in `root_order` has an entry, so `unwrap_or_default`
    // is purely defensive (it never fires) and keeps us off `unwrap` in production.
    root_order
        .into_iter()
        .map(|r| by_root.remove(&r).unwrap_or_default())
        .collect()
}

/// Brain-style positional voting over the first [`VOTE_SAMPLE`] members (deterministic).
/// For each token position, the token is a template constant when it *uniquely dominates* —
/// held by ≥ [`VOTE_SHARE`] of the voters and by strictly more of them than any other token —
/// else a variable `{}` slot; adjacent variable slots merge into one `{}`. Members shorter
/// than the widest contribute nothing to the missing positions (counts as disagreement →
/// variable), so a ragged member can't forge a false constant. Returns the template with
/// whitespace-joined constants and `{}` slots.
fn vote_template(members: &[&str]) -> String {
    let toks: Vec<Vec<&str>> = members
        .iter()
        .take(VOTE_SAMPLE)
        .map(|l| tokens(l))
        .collect();
    let width = toks.iter().map(Vec::len).max().unwrap_or(0);
    let voters = toks.len();
    let need = ((voters as f64) * VOTE_SHARE).ceil().max(1.0) as usize;

    let mut out = String::new();
    let mut prev_var = false;
    for pos in 0..width {
        // Tally the token at this position across voters that have it.
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for row in &toks {
            if let Some(t) = row.get(pos) {
                *counts.entry(t).or_insert(0) += 1;
            }
        }
        let constant = dominant_token(&counts, need);
        match constant {
            Some(t) => {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(t);
                prev_var = false;
            }
            None => {
                // Variable slot — merge with an adjacent one rather than emitting `{} {}`.
                if !prev_var {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str("{}");
                    prev_var = true;
                }
            }
        }
    }
    out
}

/// The single token that *uniquely dominates* a position, or `None` (→ variable slot). The
/// winner must reach `need` votes and be held by strictly more voters than any other token,
/// so an exact tie at the top yields `None` (variable). With a unique maximum the result is
/// independent of map-iteration order — deterministic.
fn dominant_token<'a>(counts: &HashMap<&'a str, usize>, need: usize) -> Option<&'a str> {
    let max = counts.values().copied().max().unwrap_or(0);
    if max < need {
        return None;
    }
    let mut at_max = counts.iter().filter(|&(_, &c)| c == max);
    let (&winner, _) = at_max.next()?;
    // Unique only if no second token also sits at the maximum.
    if at_max.next().is_some() {
        return None;
    }
    Some(winner)
}

/// Render a voted group as `<template> [×N: (vals) …]`. Each member contributes one tuple of
/// the tokens that fell in its variable slots — positional against the template's `{}`, and
/// in original order — so the group is fully reconstructible (information-preserving), the
/// same contract as [`render_run`]. Adjacent variable tokens (merged into one `{}` by
/// [`vote_template`]) are space-joined within their slot.
fn render_voted(tpl: &str, members: &[&str]) -> String {
    let tuples: Vec<String> = members
        .iter()
        .map(|l| format!("({})", variable_values(tpl, l).join(",")))
        .collect();
    format!("{tpl} [×{}: {}]", members.len(), tuples.join(" "))
}

/// Extract one line's values for each `{}` slot in `tpl`, walking template and line tokens
/// in lockstep. A run of line tokens facing a single `{}` (slots were merged) joins with a
/// space into that slot's value. Constants are skipped. Trailing line tokens with no slot
/// left are dropped into the last slot if it's variable, else ignored — defensive only;
/// the voter set built `tpl` from these very lines, so the shapes align.
fn variable_values(tpl: &str, line: &str) -> Vec<String> {
    let tpl_toks: Vec<&str> = tpl.split(' ').filter(|t| !t.is_empty()).collect();
    let line_toks: Vec<&str> = tokens(line);
    let mut vals: Vec<String> = Vec::new();
    let mut li = 0;
    for (ti, &tt) in tpl_toks.iter().enumerate() {
        if tt == "{}" {
            // This variable slot absorbs line tokens up to the next constant template token.
            let next_const = tpl_toks[ti + 1..].iter().find(|&&t| t != "{}").copied();
            let mut slot: Vec<&str> = Vec::new();
            while li < line_toks.len() && Some(line_toks[li]) != next_const {
                slot.push(line_toks[li]);
                li += 1;
            }
            vals.push(slot.join(" "));
        } else {
            // Constant: advance past the matching line token if present.
            if li < line_toks.len() && line_toks[li] == tt {
                li += 1;
            }
        }
    }
    vals
}

/// True if `tpl` has at least one variable slot whose values across `members` are
/// *predominantly* value-shaped — the signal that separates a machine template (variables =
/// numbers/ids/codes) from prose that merely shares a sentence frame (variables = words).
/// Without this, "The {} review of {} examined {} …" would fold; with it, only the digit- or
/// structure-bearing slots qualify, so word-only frames are left to the retrieve stage.
fn has_value_shaped_slot(tpl: &str, members: &[&str]) -> bool {
    let per_member: Vec<Vec<String>> = members.iter().map(|l| variable_values(tpl, l)).collect();
    let slots = per_member.iter().map(Vec::len).max().unwrap_or(0);
    (0..slots).any(|s| {
        let (mut shaped, mut total) = (0usize, 0usize);
        for vals in &per_member {
            if let Some(v) = vals.get(s) {
                total += 1;
                if value_shaped(v) {
                    shaped += 1;
                }
            }
        }
        total > 0 && shaped * 2 >= total // ≥ half the slot's values are value-shaped
    })
}

/// Whether a captured value looks like a *value* rather than a plain word — language-
/// universal: it bears an ASCII digit (covers numbers, versions, ids `crate_0`, error codes
/// `E0308`, sizes), or the [`VARIABLE`] regex recognizes it (hex, UUID, timestamp, IPv4,
/// quoted). Pure alphabetic words in any script (`bravo`, `café`, `日本`) are *not*
/// value-shaped, so prose slots don't qualify. Empty values (a slot the member lacked) don't.
fn value_shaped(v: &str) -> bool {
    if v.is_empty() {
        return false;
    }
    v.chars().any(|c| c.is_ascii_digit()) || VARIABLE.is_match(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_mask_variables() {
        let (tpl, vars) = template_of("Connection to db-01 timed out after 30ms");
        assert_eq!(tpl, "Connection to db-{} timed out after {}ms");
        assert_eq!(vars, vec!["01", "30"]);
    }

    #[test]
    fn timestamp_and_uuid_are_single_tokens() {
        let (tpl, vars) =
            template_of("2023-01-02T10:00:00Z req 1f2e3d4c-5b6a-7980-1234-567890abcdef done");
        assert_eq!(tpl, "{} req {} done");
        assert_eq!(vars.len(), 2, "timestamp and UUID each mask as one slot");
    }

    #[test]
    fn collapses_a_run_losslessly_and_shorter() {
        let text = "Connection to db-01 timed out after 30ms\n\
                    Connection to db-02 timed out after 12ms\n\
                    Connection to db-07 timed out after 5ms";
        let (out, folded) = collapse(text);
        assert!(folded, "a real run folded");
        assert_eq!(
            out.lines().count(),
            1,
            "three same-template lines fold to one"
        );
        assert!(out.starts_with("Connection to db-{} timed out after {}ms [×3:"));
        // every original's values survive in the tuples (lossless)
        for tuple in ["(01,30)", "(02,12)", "(07,5)"] {
            assert!(out.contains(tuple), "missing {tuple} in {out}");
        }
        assert!(out.len() < text.len(), "collapse must shrink the run");
    }

    #[test]
    fn leaves_short_runs_untouched() {
        let text = "host a failed 1 time\nhost b failed 2 times";
        let (out, folded) = collapse(text);
        assert_eq!(out, text, "a 2-line run is below MIN_RUN");
        assert!(!folded, "nothing folded");
    }

    #[test]
    fn aligned_columns_fold_despite_padding() {
        // `ls -l`-style: the size column's padding differs per row (right-aligned). Before
        // whitespace-normalization this fragmented into many runs; now all rows share one
        // template and fold together.
        let text = "drwxr-xr-x 2 u g        0 Apr 01 file_0.log\n\
                    drwxr-xr-x 2 u g     1024 Apr 02 file_1.log\n\
                    drwxr-xr-x 2 u g   524288 Apr 03 file_2.log";
        let (out, _) = collapse(text);
        assert_eq!(
            out.lines().count(),
            1,
            "aligned rows fold to one despite padding: {out}"
        );
        assert!(out.contains("[×3:"), "one run of 3");
        assert!(
            out.contains("524288") && out.contains("1024"),
            "every value preserved"
        );
    }

    #[test]
    fn does_not_collapse_slot_free_identical_lines() {
        // No variable tokens → that's exact-dedup territory, not template collapse.
        let text = "starting up\nstarting up\nstarting up";
        let (out, folded) = collapse(text);
        assert_eq!(out, text);
        assert!(!folded, "slot-free identical lines do not fold");
    }

    #[test]
    fn only_consecutive_runs_collapse() {
        // Same template at lines 0 and 2, broken by a different line at 1: neither run
        // reaches MIN_RUN, so nothing collapses (order is preserved).
        let text = "value is 1\ndifferent entirely here\nvalue is 2";
        let (out, folded) = collapse(text);
        assert_eq!(out, text);
        assert!(!folded, "non-consecutive runs below MIN_RUN do not fold");
    }

    #[test]
    fn trailing_newline_with_no_fold_reports_not_folded() {
        // Distinct lines, input ends in '\n'. `join("\n")` strips that newline, so the
        // rebuilt string differs from the input even though nothing folded — the boolean
        // (not a string compare) must still report `false` so callers don't misread the
        // dropped newline as a real change.
        let text = "alpha beta gamma\ndelta epsilon zeta\neta theta iota\n";
        let (out, folded) = collapse(text);
        assert!(
            !folded,
            "nothing folded → false despite the stripped trailing newline"
        );
        assert_ne!(
            out, text,
            "join() drops the trailing newline (would fool a string compare)"
        );
    }

    // ── Global (non-adjacent) collapse ──────────────────────────────────────────────

    /// Whitespace-token count — the same proxy the project's savings tests use.
    fn count_tokens(s: &str) -> usize {
        s.split_whitespace().count()
    }

    /// Simulated parallel build (`cargo`/`pytest -n`): two *distinct* templated lines emitted
    /// by interleaved workers, so identical-template lines alternate and never form a
    /// consecutive run. Each line is mostly a static template with one small numeric value —
    /// the shape where lossless template collapse genuinely wins.
    fn interleaved_build(pairs: usize) -> String {
        let mut lines = Vec::with_capacity(pairs * 2);
        for i in 0..pairs {
            lines.push(format!(
                "[build] compiled object file number {i} successfully, no warnings"
            ));
            lines.push(format!(
                "[test] executed checks for shard {i} of pool, all green ok"
            ));
        }
        lines.join("\n")
    }

    #[test]
    fn global_folds_interleaved_where_consecutive_cannot() {
        let text = interleaved_build(12); // 24 lines, two alternating templates

        // The consecutive pass is helpless: no two same-template lines are adjacent.
        let (consec, folded_consec) = collapse(&text);
        assert!(
            !folded_consec,
            "consecutive pass can't fold interleaved lines"
        );
        assert_eq!(
            consec.lines().count(),
            24,
            "consecutive leaves all 24 lines"
        );

        // The global pass folds each interleaved template into one representative.
        let (out, folded) = collapse_global(&text);
        assert!(folded, "global pass folds the interleaved templates");
        assert_eq!(
            out.lines().count(),
            2,
            "two distinct templates fold to two representatives: {out}"
        );
        assert_eq!(
            out.matches("[×12:").count(),
            2,
            "each of the two templates folds 12 occurrences: {out}"
        );

        // Real token savings (the whole point) — well clear of the 60% gate on this shape.
        let saved = 100.0 - (count_tokens(&out) as f64 / count_tokens(&text) as f64 * 100.0);
        assert!(
            saved >= 60.0,
            "expected ≥60% token savings, got {saved:.1}%"
        );
    }

    #[test]
    fn global_fold_is_information_preserving() {
        // Every per-occurrence value must survive in the tuples, in first-seen order. Here
        // the numeric id 0..3 is the only variable, captured once per template.
        let text = interleaved_build(4);
        let (out, _) = collapse_global(&text);
        for id in ["0", "1", "2", "3"] {
            // Each id appears once in each of the two folded templates' tuples.
            let hits = out.matches(id).count();
            assert!(
                hits >= 2,
                "value {id} preserved for both templates (got {hits}): {out}"
            );
        }
        // The static template words survive verbatim in the representative.
        assert!(
            out.contains("compiled object file"),
            "template 1 kept: {out}"
        );
        assert!(
            out.contains("executed checks for shard"),
            "template 2 kept: {out}"
        );
    }

    #[test]
    fn global_collapse_is_deterministic() {
        // Hard constraint: identical input ⇒ byte-identical output, every run. Bigger,
        // three-template interleave to exercise bucketing/merge ordering.
        let mut lines = Vec::new();
        for i in 0..15 {
            lines.push(format!("worker A processed job {i} in {i}ms"));
            lines.push(format!("worker B fetched record {i} size {i}kb"));
            lines.push(format!("worker C indexed shard {i} ok"));
        }
        let text = lines.join("\n");
        let (a, fa) = collapse_global(&text);
        let (b, fb) = collapse_global(&text);
        assert_eq!(a, b, "same input twice ⇒ identical output");
        assert_eq!(fa, fb, "fold flag stable too");
        assert!(fa, "and it actually folded");
    }

    #[test]
    fn error_codes_survive_global_collapse() {
        // E0308-style codes must never be silently dropped (LogLSHD's alphabetic-only
        // filter would discard them). Here the code is the *variable* part across members:
        // it must show up as a captured value.
        let mut lines = Vec::new();
        for (i, code) in ["E0308", "E0277", "E0425", "E0599"].iter().enumerate() {
            lines.push(format!("error[{code}]: mismatch in module_{i}"));
            lines.push(format!("note: build step {i} continued"));
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded, "the interleaved error/note templates fold");
        for code in ["E0308", "E0277", "E0425", "E0599"] {
            assert!(
                out.contains(code),
                "{code} must survive (not dropped): {out}"
            );
        }
    }

    #[test]
    fn error_code_constant_stays_in_template() {
        // When the *same* code recurs (constant across members), it belongs in the template,
        // not the variable slots — and still must not vanish.
        let mut lines = Vec::new();
        for i in 0..6 {
            lines.push(format!("error[E0308]: mismatched types at site {i}"));
            lines.push(format!("info: checked candidate {i}"));
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded);
        assert!(
            out.contains("E0308"),
            "constant error code kept in template: {out}"
        );
    }

    #[test]
    fn global_collapse_handles_cjk_lines() {
        // Unicode-safe bucketing: CJK lines (no ASCII, multibyte) that share a template must
        // bucket together (anchors index char boundaries, never bytes) and fold.
        let mut lines = Vec::new();
        for i in 0..10 {
            lines.push(format!("处理 任务 {i} 完成 用时 {i} 毫秒"));
            lines.push(format!("读取 文件 {i} 大小 {i} 字节"));
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded, "CJK templates fold: {out}");
        assert!(
            out.contains('处'),
            "CJK content preserved in the template: {out}"
        );
        // No panic, fewer lines.
        assert!(out.lines().count() < text.lines().count());
    }

    #[test]
    fn global_collapse_handles_very_long_lines() {
        // Long lines must not break bucketing (anchors are relative positions) and must fold
        // when they share a template.
        let blob = "x".repeat(4000);
        let mut lines = Vec::new();
        for i in 0..5 {
            lines.push(format!("record {i} payload {blob} tail {i}"));
            lines.push(format!("checksum {i} verified ok ({i} bytes)"));
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded, "long interleaved lines fold without panicking");
        assert!(out.lines().count() < text.lines().count());
    }

    #[test]
    fn global_collapse_leaves_single_line_segment() {
        // One line: nothing to group; return it unchanged, not folded.
        let (out, folded) = collapse_global("just one line here 42");
        assert_eq!(out, "just one line here 42");
        assert!(!folded);
    }

    #[test]
    fn global_declines_diverse_prose() {
        // Distinct sentences: different token counts / anchors ⇒ no shared bucket ⇒ nothing
        // folds. Prose stays the retrieve stage's job, not a false template fold.
        let prose = "The committee reviewed the annual budget on Tuesday morning.\n\
                     A sudden storm delayed the harvest across three northern counties.\n\
                     Engineers traced the outage to a misconfigured load balancer.\n\
                     The museum unveiled a restored fresco after two years of work.";
        let (out, folded) = collapse_global(prose);
        assert!(!folded, "diverse prose must not fold: {out}");
    }

    #[test]
    fn global_declines_word_only_sentence_frame() {
        // The hard case: many sentences share a FIXED frame and differ only in content
        // *words*. Positional voting would vote a template — but the slots hold plain words,
        // not values, so the prose guard declines (this is the retrieve stage's job).
        const W: &[&str] = &[
            "alpha", "bravo", "cobalt", "dune", "ember", "flint", "granite", "harbor",
        ];
        let prose: String = (0..8)
            .map(|i| {
                format!(
                    "The {} review of {} examined {} thoroughly today.",
                    W[i % W.len()],
                    W[(i + 3) % W.len()],
                    W[(i + 5) % W.len()]
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let (out, folded) = collapse_global(&prose);
        assert!(!folded, "word-only sentence frame must not fold: {out}");
    }

    #[test]
    fn global_folds_value_bearing_frame() {
        // The mirror image: the same fixed frame but with *value* slots (a numeric id) DOES
        // fold — the guard passes because the slot is digit-bearing.
        let logs: String = (0..8)
            .map(|i| format!("The job for shard {i} completed in {i} seconds flat."))
            .collect::<Vec<_>>()
            .join("\n");
        let (out, folded) = collapse_global(&logs);
        assert!(folded, "a value-bearing frame folds: {out}");
        assert!(out.contains("[×8:"), "all eight occurrences fold: {out}");
    }

    #[test]
    fn global_preserves_consecutive_fold_then_adds_global() {
        // A consecutive run AND a separate interleaved set in one input: the consecutive
        // pass folds the run, the global pass folds the interleave — both reported.
        let mut lines = Vec::new();
        // consecutive run of 3
        for i in 0..3 {
            lines.push(format!("loading shard shard-{i} ready"));
        }
        // then an interleaved pair set
        for i in 0..6 {
            lines.push(format!("alpha worker {i} done"));
            lines.push(format!("beta worker {i} done"));
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded);
        assert!(out.contains("[×3:"), "consecutive run folded: {out}");
        assert!(out.contains("[×6:"), "interleaved set folded: {out}");
        // The consecutive representative isn't re-bucketed/double-folded.
        assert_eq!(out.matches("[×3:").count(), 1, "no double-fold: {out}");
    }

    #[test]
    fn vote_template_marks_disagreeing_positions_variable() {
        // Brain-style voting: a whitespace token that disagrees across members becomes a
        // `{}` slot (the whole path token here, not its `/`-segments). Two members → a
        // position is variable unless the SAME token holds it in both (strict dominance).
        let members = ["GET /api/v1/users 200 fast", "GET /api/v1/posts 200 fast"];
        assert_eq!(vote_template(&members), "GET {} 200 fast");
        // Two adjacent disagreeing positions collapse to ONE slot.
        let members2 = ["job alpha one done", "job beta two done"];
        assert_eq!(vote_template(&members2), "job {} done");
        // A position where one token uniquely dominates (≥ share, strictly most) stays
        // constant; a true tie at the top stays variable.
        let members3 = ["x A z", "x A z", "x B z"]; // pos1: A×2 vs B×1 → A dominates
        assert_eq!(vote_template(&members3), "x A z");
        let members4 = ["x A z", "x B z"]; // pos1: A×1, B×1 → tie → variable
        assert_eq!(vote_template(&members4), "x {} z");
    }

    #[test]
    fn bucket_key_is_char_boundary_safe() {
        // Anchors must index char boundaries for multibyte input — never panic, never split
        // a codepoint — and an identical line must yield an identical key (determinism).
        let line = "café münster 北京 处理 42";
        let k1 = bucket_key(line);
        let k2 = bucket_key(line);
        assert_eq!(k1, k2, "identical multibyte line ⇒ identical key");
        assert_eq!(k1.token_count, 5, "token count counts whitespace tokens");
        // Empty and single-char lines are handled without panicking.
        let _ = bucket_key("");
        let _ = bucket_key("北");
    }
}

#[cfg(test)]
mod global_grep_records {
    use super::*;

    #[test]
    fn interleaved_grep_records_fold_globally() {
        // 70 same-template `path:line:` records interleaved with `--` separators
        // (rg -C style): runs of 2 sit below MIN_RUN, so only the global pass can fold
        // them. Regression coverage for the enumeration-query shape ("where is X
        // called?") where every record is the answer.
        let mut lines = Vec::new();
        for i in 0..70 {
            lines.push(format!(
                "src/parser/config.rs:{}:    let v = parse_config(input, {});",
                i + 10,
                i
            ));
            if i % 2 == 1 {
                lines.push("--".to_string());
            }
        }
        let text = lines.join("\n");
        let (out, folded) = collapse_global(&text);
        assert!(folded, "global pass must fold the 70-member group");
        assert!(
            out.lines().count() < 40,
            "70 records + seps collapse far below input size"
        );
        // lossless: every line number survives
        for i in 0..70 {
            assert!(
                out.contains(&format!("{}", i + 10)),
                "line number {} lost",
                i + 10
            );
        }
    }
}
