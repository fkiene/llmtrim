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
//! # Range-folded parameter columns
//!
//! When a captured parameter *column* is a regular sequence, listing every tuple is pure
//! waste, so a post-pass on the fold ([`render_tuple_block`]) may swap the row-wise tuples
//! for one column-wise group — `[×N: (col0; col1; …)]`, columns `; `-separated, each one of:
//!
//! - a single value — the column is constant across all N rows;
//! - `start..end` (integers, step 1) or `start..end step k` — an arithmetic sequence,
//!   inclusive of `end`, N values exactly;
//! - `start..end step Ns` — ISO-8601-like timestamps with a constant step of N seconds;
//! - an explicit `,`-joined value list — anything irregular (per-column fallback).
//!
//! Example: `[×30: (2026-06-13T10:02:00Z..2026-06-13T10:02:29Z step 1s; 0..29)]`.
//!
//! Columns are independent: one may range-fold while another stays explicit. The fold is
//! LOSSLESS by construction — a candidate range is emitted only after every original value
//! round-trips byte-identically through the notation (canonical decimal rendering for
//! integers; epoch-seconds re-render for timestamps), so mixed widths, leading zeros,
//! irregular steps, fractional seconds or odd calendar shapes all fall back to the explicit
//! list. And it never inflates: the column form is used only when strictly shorter than the
//! row-wise tuples.
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
/// values per original line, positional against the `{}` slots — or the column-wise
/// range form when that's strictly shorter (see [`render_tuple_block`]).
fn render_run(tpl: &str, run: &[&str]) -> String {
    let rows: Vec<Vec<String>> = run.iter().map(|l| template_of(l).1).collect();
    format!("{tpl} [×{}: {}]", run.len(), render_tuple_block(&rows))
}

/// The inner `[×N: …]` payload for a folded group: row-wise tuples by default, swapped for
/// the column-wise range form (module docs, "Range-folded parameter columns") only when at
/// least one column folds **and** the result is strictly shorter — never inflates, and the
/// irregular case is byte-identical to today's output.
fn render_tuple_block(rows: &[Vec<String>]) -> String {
    let row_wise = rows
        .iter()
        .map(|r| format!("({})", r.join(",")))
        .collect::<Vec<_>>()
        .join(" ");
    match render_columns(rows) {
        Some(col_wise) if col_wise.len() < row_wise.len() => col_wise,
        _ => row_wise,
    }
}

/// Column-wise `(col0; col1; …)` rendering, or `None` when it doesn't apply: ragged rows
/// (defensive), no column folds, or any value contains the notation's own separators
/// (`,`, `;`, `..` — quoted-string captures can), which would make reconstruction
/// ambiguous. Lossless: a fold is per-column and only via [`fold_column`]'s round-trip
/// checks; irregular columns stay an explicit comma list. A datetime-ish column that
/// falls back is reported to the missed-fold telemetry ([`record_missed_fold`]) so real
/// traffic can rank which shapes the future registry should support.
fn render_columns(rows: &[Vec<String>]) -> Option<String> {
    let width = rows.first()?.len();
    if width == 0 || rows.iter().any(|r| r.len() != width) {
        return None;
    }
    if rows
        .iter()
        .flatten()
        .any(|v| v.contains(',') || v.contains(';') || v.contains(".."))
    {
        return None;
    }
    let mut any_folded = false;
    let cols: Vec<String> = (0..width)
        .map(|c| {
            let vals: Vec<&str> = rows.iter().map(|r| r[c].as_str()).collect();
            match fold_column(&vals) {
                Some(folded) => {
                    any_folded = true;
                    folded
                }
                None => {
                    record_missed_fold(&vals);
                    vals.join(",")
                }
            }
        })
        .collect();
    if !any_folded {
        return None;
    }
    Some(format!("({})", cols.join("; ")))
}

/// Fold one parameter column into range notation, or `None` (→ explicit list). Constant
/// first (the single value), then arithmetic integers, then constant-step timestamps.
fn fold_column(vals: &[&str]) -> Option<String> {
    let first = vals.first()?;
    if vals.iter().all(|v| v == first) {
        return Some((*first).to_string());
    }
    fold_int_column(vals).or_else(|| fold_timestamp_column(vals))
}

/// `start..end` (`step k` when k ≠ 1) for an arithmetic integer sequence. Lossless gate:
/// every value must be its own canonical decimal rendering (parse → to_string round-trip),
/// so leading zeros / mixed widths / signs fall back to the explicit list.
fn fold_int_column(vals: &[&str]) -> Option<String> {
    let nums: Vec<i128> = vals
        .iter()
        .map(|v| v.parse::<i128>().ok().filter(|n| n.to_string() == **v))
        .collect::<Option<_>>()?;
    let step = nums.get(1)? - nums[0];
    if step == 0 || nums.windows(2).any(|w| w[1] - w[0] != step) {
        return None;
    }
    let (start, end) = (vals[0], vals[vals.len() - 1]);
    Some(if step == 1 {
        format!("{start}..{end}")
    } else {
        format!("{start}..{end} step {step}")
    })
}

/// `start..end step Ns` for ISO-8601-like timestamps (the shape [`VARIABLE`] already
/// captures, sans fractional seconds) with a constant whole-second step. All values must
/// share the date/time separator and zone suffix, and every one must re-render
/// byte-identically from its epoch seconds — leap-second or invalid-calendar shapes fail
/// that round-trip and fall back to the explicit list.
fn fold_timestamp_column(vals: &[&str]) -> Option<String> {
    let ts: Vec<Ts> = vals.iter().map(|v| parse_ts(v)).collect::<Option<_>>()?;
    let head = &ts[0];
    if ts
        .iter()
        .any(|t| t.sep != head.sep || t.suffix != head.suffix)
    {
        return None;
    }
    let step = ts.get(1)?.epoch - head.epoch;
    if step == 0 || ts.windows(2).any(|w| w[1].epoch - w[0].epoch != step) {
        return None;
    }
    // Round-trip every original: byte equality ⇒ start-epoch + i·step reconstructs exactly.
    let (sep, suffix) = (head.sep, head.suffix.as_str());
    if ts
        .iter()
        .zip(vals)
        .any(|(t, v)| render_ts(t.epoch, sep, suffix) != **v)
    {
        return None;
    }
    Some(format!(
        "{}..{} step {step}s",
        vals[0],
        vals[vals.len() - 1]
    ))
}

/// One parsed timestamp: seconds since epoch in its own (suffix-opaque) timeline, plus the
/// surface details needed to re-render it byte-identically.
struct Ts {
    epoch: i64,
    sep: char,
    suffix: String,
}

/// Strict ISO-8601-like shape — a *subset* of [`VARIABLE`]'s timestamp alternative
/// (fractional seconds deliberately excluded: a range over them is not reconstructible
/// from whole-second steps).
static TS: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(\d{4})-(\d{2})-(\d{2})([T ])(\d{2}):(\d{2}):(\d{2})(Z|[+-]\d{2}:?\d{2})?$")
        .unwrap()
});

fn parse_ts(v: &str) -> Option<Ts> {
    let c = TS.captures(v)?;
    let f = |i: usize| -> Option<i64> { c.get(i)?.as_str().parse().ok() };
    let (y, mo, d) = (f(1)?, f(2)?, f(3)?);
    let (h, mi, s) = (f(5)?, f(6)?, f(7)?);
    let sep = c.get(4)?.as_str().chars().next()?;
    let suffix = c.get(8).map(|m| m.as_str()).unwrap_or("").to_string();
    let epoch = days_from_civil(y, mo, d) * 86_400 + h * 3_600 + mi * 60 + s;
    Some(Ts { epoch, sep, suffix })
}

fn render_ts(epoch: i64, sep: char, suffix: &str) -> String {
    let (y, m, d) = civil_from_days(epoch.div_euclid(86_400));
    let secs = epoch.rem_euclid(86_400);
    format!(
        "{y:04}-{m:02}-{d:02}{sep}{:02}:{:02}:{:02}{suffix}",
        secs / 3_600,
        (secs % 3_600) / 60,
        secs % 60
    )
}

// ── Missed-fold telemetry (shape-registry stage 1) ──────────────────────────────────

/// Capture directory for QA telemetry — the same opt-in capture dir the proxy's before/after
/// capture uses (`LLMTRIM_CAPTURE_DIR` env or `capture_dir` in the config file). Read once;
/// `None` (the default) costs nothing per fold.
static CAPTURE_DIR: Lazy<Option<String>> = Lazy::new(|| {
    crate::config::RuntimeConfig::get()
        .capture_dir
        .as_ref()
        .map(|p| p.to_string_lossy().into_owned())
});

/// Loosely datetime-shaped: a `H:M`-style time, an ISO `YYYY-MM` date prefix, or a
/// slash date. Deliberately broad — this only gates *telemetry*, never output.
static DATETIME_ISH: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\d:\d|\d{4}-\d{2}|\b\d{1,2}/\d{1,2}\b").unwrap());

/// Report a parameter column that looked datetime-ish but fell back to the explicit list.
/// Appends one JSONL record to `<capture dir>/missed_folds.jsonl` so the capture QA loop
/// can rank which timestamp shapes a future shape registry should support (and whether
/// jiff earns its place). Off unless `LLMTRIM_CAPTURE_DIR` is set; write failures are
/// logged and swallowed — telemetry must never break a fold.
fn record_missed_fold(vals: &[&str]) {
    if let Some(dir) = CAPTURE_DIR.as_deref() {
        write_missed_fold(dir, vals);
    }
}

/// Env-independent body of [`record_missed_fold`] (testable without `set_var`).
fn write_missed_fold(dir: &str, vals: &[&str]) {
    if vals.is_empty() || !vals.iter().all(|v| DATETIME_ISH.is_match(v)) {
        return;
    }
    // ISO shapes that parsed but had no constant step vs shapes we can't parse at all —
    // the second bucket is the registry's shopping list.
    let reason = if vals.iter().all(|v| TS.is_match(v)) {
        "irregular_step"
    } else {
        "unsupported_shape"
    };
    let record = serde_json::json!({
        "kind": "missed_fold",
        "reason": reason,
        "count": vals.len(),
        "sample": vals.iter().take(5).collect::<Vec<_>>(),
    });
    use std::io::Write;
    let path = std::path::Path::new(dir).join("missed_folds.jsonl");
    let written = std::fs::create_dir_all(dir).and_then(|_| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| writeln!(f, "{record}"))
    });
    if let Err(e) = written {
        eprintln!(
            "llmtrim: missed-fold telemetry failed ({}): {e}",
            path.display()
        );
    }
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Civil date from days since 1970-01-01 (Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
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
    let rows: Vec<Vec<String>> = members.iter().map(|l| variable_values(tpl, l)).collect();
    format!("{tpl} [×{}: {}]", members.len(), render_tuple_block(&rows))
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

/// Volatile-value-masked fingerprint of a whole output, for the repeat → passthrough
/// rail: every line is reduced to its [`template_of`] template (numbers, hex, UUIDs,
/// timestamps, IPs → `{}`), so two runs of one tool that differ only in such values —
/// TAP's `duration_ms`, log timestamps, ports, PIDs — fingerprint identically, while
/// any change in the *constant* text (a test flipping ok ↔ not ok, a new error line)
/// changes it. A masked false match only ships an output uncompressed — safe.
pub(crate) fn fingerprint(text: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for line in text.lines() {
        template_of(line).0.hash(&mut h);
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_ignores_volatile_values_but_not_results() {
        let run = |d1: &str, d2: &str, t19: &str| {
            format!(
                "TAP version 13\nok 1 - parse\n  duration_ms: {d1}\n{t19} 19 - normalize\n  duration_ms: {d2}\n1..2"
            )
        };
        assert_eq!(
            fingerprint(&run("124.481897", "3.412345", "not ok")),
            fingerprint(&run("64.649202", "9.000001", "not ok")),
            "re-run differing only in timings fingerprints identically"
        );
        assert_ne!(
            fingerprint(&run("124.481897", "3.412345", "not ok")),
            fingerprint(&run("124.481897", "3.412345", "ok")),
            "a test flipping not ok → ok changes the fingerprint"
        );
    }

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
        // The id column 0..3 is a regular sequence, so it range-folds — every value is
        // still exactly reconstructible from the notation, once per folded template.
        assert_eq!(
            out.matches("0..3").count(),
            2,
            "id column range-folded for both templates: {out}"
        );
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

    // ── Range-folded parameter columns ──────────────────────────────────────────────

    fn rows(cols: &[&[&str]]) -> Vec<Vec<String>> {
        let n = cols.first().map_or(0, |c| c.len());
        (0..n)
            .map(|r| cols.iter().map(|c| c[r].to_string()).collect())
            .collect()
    }

    #[test]
    fn integer_column_folds_to_range() {
        let vals: Vec<String> = (0..30).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        let out = render_tuple_block(&rows(&[&refs]));
        assert_eq!(out, "(0..29)");
    }

    #[test]
    fn stepped_integer_column_folds_with_step() {
        let vals: Vec<String> = (0..10).map(|i| (5 + i * 3).to_string()).collect();
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        let out = render_tuple_block(&rows(&[&refs]));
        assert_eq!(out, "(5..32 step 3)");
    }

    #[test]
    fn constant_column_emits_single_value() {
        let out = render_tuple_block(&rows(&[
            &["info", "info", "info", "info"],
            &["1", "2", "3", "4"],
        ]));
        assert_eq!(out, "(info; 1..4)");
    }

    #[test]
    fn timestamp_column_folds_with_seconds_step() {
        let vals: Vec<String> = (0..30)
            .map(|i| format!("2026-06-13T10:02:{i:02}Z"))
            .collect();
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        let out = render_tuple_block(&rows(&[&refs]));
        assert_eq!(out, "(2026-06-13T10:02:00Z..2026-06-13T10:02:29Z step 1s)");
    }

    #[test]
    fn timestamp_range_crosses_minute_and_day_boundaries() {
        // 30s step from 23:59:00 walks across midnight — Hinnant date math, round-tripped.
        let vals: Vec<String> = (0..6)
            .map(|i| {
                render_ts(
                    days_from_civil(2026, 6, 13) * 86_400 + 86_340 + i * 30,
                    'T',
                    "Z",
                )
            })
            .collect();
        assert_eq!(vals[0], "2026-06-13T23:59:00Z");
        assert_eq!(vals[5], "2026-06-14T00:01:30Z");
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        let out = render_tuple_block(&rows(&[&refs]));
        assert_eq!(out, "(2026-06-13T23:59:00Z..2026-06-14T00:01:30Z step 30s)");
    }

    #[test]
    fn irregular_column_keeps_explicit_list() {
        // Irregular step → row-wise tuples exactly as today.
        let out = render_tuple_block(&rows(&[&["1", "2", "4", "9"]]));
        assert_eq!(out, "(1) (2) (4) (9)");
        // Mixed widths (leading zero) break canonical round-trip → explicit.
        let out = render_tuple_block(&rows(&[&["01", "02", "03", "04"]]));
        assert_eq!(out, "(01) (02) (03) (04)");
        // Leap-second-shaped (":60") timestamps never parse → explicit.
        let out = render_tuple_block(&rows(&[&[
            "2026-06-30T23:59:59Z",
            "2026-06-30T23:59:60Z",
            "2026-07-01T00:00:01Z",
        ]]));
        assert!(out.starts_with("(2026-06-30T23:59:59Z) ("), "{out}");
    }

    #[test]
    fn mixed_columns_fold_independently() {
        // Column 0 is a range, column 1 stays an explicit comma list.
        let out = render_tuple_block(&rows(&[
            &["0", "1", "2", "3"],
            &["db-01", "db-07", "db-02", "db-09"],
        ]));
        assert_eq!(out, "(0..3; db-01,db-07,db-02,db-09)");
    }

    #[test]
    fn range_never_inflates_short_lists() {
        // Two timestamps: "(a..b step 1s)" is LONGER than "(a) (b)" → keep row-wise.
        let out = render_tuple_block(&rows(&[&["2026-06-13T10:02:00Z", "2026-06-13T10:02:01Z"]]));
        assert_eq!(out, "(2026-06-13T10:02:00Z) (2026-06-13T10:02:01Z)");
    }

    #[test]
    fn ambiguous_separator_values_decline_column_form() {
        // A captured value containing the notation's own separators would make the column
        // form unreconstructible → row-wise, even though column 0 is a perfect range.
        let out = render_tuple_block(&rows(&[&["1", "2", "3"], &["\"a,b\"", "\"c\"", "\"d\""]]));
        assert_eq!(out, "(1,\"a,b\") (2,\"c\") (3,\"d\")");
    }

    // ── Missed-fold telemetry ───────────────────────────────────────────────────────

    /// Unique scratch dir per test (nextest runs each test in its own process, but keep
    /// names disjoint anyway so the tests never share a file).
    fn scratch_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("llmtrim-missed-fold-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn read_jsonl(dir: &std::path::Path) -> String {
        std::fs::read_to_string(dir.join("missed_folds.jsonl")).unwrap_or_default()
    }

    #[test]
    fn telemetry_records_irregular_iso_timestamps() {
        let dir = scratch_dir("irregular");
        write_missed_fold(
            &dir.to_string_lossy(),
            &[
                "2026-06-13T10:02:32Z",
                "2026-06-13T10:02:51Z",
                "2026-06-13T10:02:33Z",
            ],
        );
        let log = read_jsonl(&dir);
        assert!(log.contains("\"reason\":\"irregular_step\""), "{log}");
        assert!(log.contains("\"count\":3"), "{log}");
        assert!(log.contains("2026-06-13T10:02:32Z"), "sample kept: {log}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn telemetry_records_unsupported_datetime_shape() {
        // Slash dates / bare times don't parse as our ISO shape — the registry's
        // shopping list.
        let dir = scratch_dir("unsupported");
        write_missed_fold(
            &dir.to_string_lossy(),
            &[
                "13/06/2026 10:02:00",
                "13/06/2026 10:02:30",
                "13/06/2026 10:03:00",
            ],
        );
        let log = read_jsonl(&dir);
        assert!(log.contains("\"reason\":\"unsupported_shape\""), "{log}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn telemetry_ignores_non_datetime_columns() {
        // Plain-word / id columns falling back is normal, not a missed shape — no record.
        let dir = scratch_dir("words");
        write_missed_fold(&dir.to_string_lossy(), &["db-01", "db-07", "db-02"]);
        assert!(
            !dir.join("missed_folds.jsonl").exists(),
            "no file for non-datetime columns"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Evidence test for the Drain3 "regex pre-masking" proposal. The existing pipeline
    /// already masks volatile tokens (timestamps, hex addresses, hashes, UUIDs, IPv4,
    /// numbers) via [`VARIABLE`] before grouping. This measures what that pre-masking buys:
    /// distinct templates and token savings WITH masking vs. a no-mask baseline (where each
    /// line is its own group because the volatile fields differ).
    #[test]
    fn premasking_drives_grouping_and_savings() {
        // Realistic repetitive tool output: a stack trace repeated by many worker threads,
        // each frame differing only in volatile fields — a wall-clock timestamp, a hex
        // return address, a thread id, and a line number. This is exactly the shape the
        // Drain3 proposal targets.
        let mut lines = Vec::new();
        for i in 0..40 {
            let addr = 0x7f00_0000u64 + (i as u64) * 0x40;
            lines.push(format!(
                "2026-06-13T10:{:02}:{:02}Z thread-{i} at 0x{addr:x} (worker.rs:{}) panicked: send failed",
                i / 60,
                i % 60,
                120 + i
            ));
        }
        let text = lines.join("\n");

        // Baseline: distinct templates WITHOUT pre-masking = number of byte-distinct lines.
        // Every line differs (timestamp/addr/id/line), so each is its own group → 40.
        let distinct_unmasked = {
            let mut set = std::collections::HashSet::new();
            for l in text.lines() {
                set.insert(l);
            }
            set.len()
        };
        assert_eq!(
            distinct_unmasked, 40,
            "no-mask baseline: every line is unique"
        );

        // With pre-masking: collapse the volatile fields, then count distinct templates.
        let distinct_masked = {
            let mut set = std::collections::HashSet::new();
            for l in text.lines() {
                set.insert(template_of(l).0);
            }
            set.len()
        };
        assert_eq!(
            distinct_masked, 1,
            "pre-masking folds all 40 lines into ONE template"
        );

        // The whole point: real token savings on this shape.
        let (out, folded) = collapse(&text);
        assert!(folded, "the masked run folds");
        let saved = 100.0 - (count_tokens(&out) as f64 / count_tokens(&text) as f64 * 100.0);
        assert!(
            saved >= 60.0,
            "expected >=60% token savings, got {saved:.1}%"
        );

        // Lossless: the line-number column is a regular sequence, so it range-folds to
        // `120..159` (every value reconstructible from the notation) rather than listing all
        // 40. Both endpoints survive verbatim, and the hex-address column keeps its explicit
        // list — no value is dropped.
        assert!(
            out.contains("120..159"),
            "line-number range preserved: {out}"
        );
        assert!(out.contains("0x7f000000"), "first address preserved: {out}");
        assert!(out.contains("0x7f0009c0"), "last address preserved: {out}");
    }

    #[test]
    fn end_to_end_collapse_range_folds_regular_run() {
        let text: String = (0..30)
            .map(|i| format!("[2026-06-13T10:02:{i:02}Z] INFO compiling task_{i} ok"))
            .collect::<Vec<_>>()
            .join("\n");
        let (out, folded) = collapse(&text);
        assert!(folded);
        assert_eq!(
            out,
            "[{}] INFO compiling task_{} ok [×30: \
             (2026-06-13T10:02:00Z..2026-06-13T10:02:29Z step 1s; 0..29)]"
        );
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
