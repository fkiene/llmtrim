//! Terminal-noise normalization pre-pass (feature #6) — lossless for the model.
//!
//! Colored build/test output (cargo, pytest, jest) carries ANSI escape sequences through
//! every stage: a single `error` is wrapped as `\x1b[31m\x1b[1merror\x1b[0m`. Those bytes
//! cost tokens *and* break this stage's own detection — a level token buried inside an
//! escape no longer matches the failure/level regexes, so a colored log reads as prose.
//! Progress output is the dual problem: a `wget`/`pip`/`npm` progress bar is one logical
//! line on screen but thousands of carriage-return-separated frames on the wire.
//!
//! This pass, applied to a tool-output segment *before* detection/windowing, makes both
//! deterministic and model-equivalent:
//!
//! - **strip ANSI** CSI/SGR escapes (`\x1b[…m`, cursor moves) and the lone `\x1b` they
//!   start with — what the terminal would consume to render color, never content;
//! - **collapse carriage returns**: within a line, keep only the segment after the last
//!   `\r` — the final frame the terminal would show, dropping the overwritten ones.
//!
//! Unicode-safe (it only ever removes ASCII control bytes / `\x1b[…]` runs and slices on
//! `\r`, never mid-codepoint) and idempotent. Returns whether it changed anything so the
//! caller can skip re-encoding untouched, already-clean segments.

use once_cell::sync::Lazy;
use regex::Regex;

/// An ANSI escape sequence: ESC `[` … final byte (CSI/SGR: colors, cursor moves), or ESC
/// `]` … terminated by BEL/ST (OSC: title sets), or a two-byte ESC + single char (e.g.
/// `ESC c` reset). Matches what a terminal consumes without printing.
static ANSI: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r"\x1b\[[0-9;?]*[ -/]*[@-~]",          // CSI: ESC [ params interm final
        r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", // OSC: ESC ] … BEL or ST
        r"|\x1b[@-Z\\-_]",                     // two-byte ESC sequences (ESC c, ESC =, …)
    ))
    .unwrap()
});

/// Strip ANSI escapes and collapse carriage-return progress within each line. Returns the
/// cleaned text and whether anything was removed (so an already-clean segment is a no-op
/// the caller can detect and skip).
pub fn normalize(text: &str) -> (String, bool) {
    // Cheap reject: no ESC and no CR means nothing to do — the common case for clean logs.
    if !text.contains('\x1b') && !text.contains('\r') {
        return (text.to_string(), false);
    }

    let stripped = ANSI.replace_all(text, "");

    // Fold `\r\n` (a normal line terminator) down to `\n` first, so the `\r` of a Windows
    // line ending is not treated as a mid-line overwrite. Any `\r` that remains is a bare
    // progress carriage-return: within its line, only the segment after the *last* `\r`
    // is what the terminal would show, so the overwritten frames are dropped.
    let unified = stripped.replace("\r\n", "\n");
    let mut out = String::with_capacity(unified.len());
    for (i, line) in unified.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
        }
        match line.rsplit_once('\r') {
            Some((_, last)) => out.push_str(last),
            None => out.push_str(line),
        }
    }

    let changed = out != text;
    (out, changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_tokens(s: &str) -> usize {
        s.split_whitespace().count()
    }

    #[test]
    fn strips_ansi_color_codes() {
        let colored = "\x1b[32mok\x1b[0m \x1b[1mbold\x1b[0m plain";
        let (out, changed) = normalize(colored);
        assert!(changed, "escapes were removed");
        assert_eq!(out, "ok bold plain", "only the escapes go, content stays");
        assert!(!out.contains('\x1b'), "no escape bytes remain");
    }

    #[test]
    fn colored_error_line_scores_as_failure_after_stripping() {
        // The exact failure the pass exists for: a level token wrapped in SGR escapes does
        // not match the failure regex until the escapes are gone, so a colored log reads
        // as plain prose to detection/scoring. After stripping, the ERROR line scores at
        // the force-keep priority.
        use crate::stages::toolout::{FORCE_PRIORITY, priority};
        let raw = "\x1b[31m\x1b[1merror\x1b[0m: cannot find value x in this scope";
        assert!(
            priority(raw) < FORCE_PRIORITY,
            "wrapped token is NOT a failure raw (the bug)"
        );
        let (out, _) = normalize(raw);
        assert!(
            priority(&out) >= FORCE_PRIORITY,
            "after stripping, the ERROR line is force-kept: {out:?}"
        );
    }

    #[test]
    fn collapses_carriage_return_progress_to_final_frame() {
        // A download progress bar: many CR-overwritten frames, one logical line. Only the
        // last frame (what the terminal shows) survives.
        let progress = "  0% [          ]\r 50% [=====     ]\r100% [==========] done";
        let (out, changed) = normalize(progress);
        assert!(changed, "carriage returns collapsed");
        assert_eq!(
            out, "100% [==========] done",
            "only the final frame remains"
        );
        assert!(
            count_tokens(&out) < count_tokens(progress),
            "fewer tokens after collapse: {} < {}",
            count_tokens(&out),
            count_tokens(progress)
        );
    }

    #[test]
    fn crlf_line_endings_are_preserved_as_lines() {
        // `\r\n` is a normal Windows line ending, not progress: each line's content must
        // survive (the empty post-`\r` segment of a CRLF is dropped, not the line).
        let crlf = "line one\r\nline two\r\nline three";
        let (out, _) = normalize(crlf);
        assert_eq!(
            out, "line one\nline two\nline three",
            "CRLF lines kept, not erased"
        );
    }

    #[test]
    fn clean_text_is_a_no_op() {
        let clean = "plain log line\nanother line\nno control bytes here";
        let (out, changed) = normalize(clean);
        assert!(!changed, "nothing to strip");
        assert_eq!(out, clean);
    }

    #[test]
    fn unicode_content_is_preserved() {
        // Escapes around multi-byte content must not slice a codepoint.
        let colored = "\x1b[33m日本語\x1b[0m café — naïve";
        let (out, _) = normalize(colored);
        assert_eq!(out, "日本語 café — naïve");
    }
}
