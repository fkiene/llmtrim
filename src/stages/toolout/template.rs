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
//! IPv4, quoted strings), so masking is language-agnostic per spec §5.

use once_cell::sync::Lazy;
use regex::Regex;

/// Minimum consecutive same-template lines before a run is collapsed.
const MIN_RUN: usize = 3;

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
}
