//! Tool-output kind detection — shape-based, cheap, zero-model.
//!
//! Each candidate segment is classified by structural shape only (no keywords from the
//! user's language). Diff wins first (unambiguous `@@`/`--- ` markers), then grep
//! (`path:line:` records), then log (a meaningful share of lines carrying a level or
//! failure signal). Anything else returns `None` and is left for the prose stages.

use once_cell::sync::Lazy;
use regex::Regex;

/// The tool-output shapes this stage compresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutKind {
    Log,
    Diff,
    Grep,
}

/// A grep / ripgrep record: `path:line:` or `path:line:col:`. The path field must hold
/// a path-ish character (letter, `.`, `/`, `\`) so a bare `12:34:56` clock — purely
/// numeric before the colon — is not mistaken for a match. An optional leading drive
/// letter (`C:`) is allowed for Windows paths.
static GREP_LINE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(?:[A-Za-z]:)?[^:\n]*[A-Za-z./\\][^:\n]*:\d+:").unwrap());

/// Minimum non-empty lines for the line-oriented kinds (grep, log).
const MIN_LINES: usize = 3;
/// Minimum non-empty lines before a segment is considered for log windowing.
const MIN_LOG_LINES: usize = 8;

/// Classify a tool-output segment, or `None` if it is not a shape this stage handles.
pub fn detect(text: &str) -> Option<OutKind> {
    if is_diff(text.trim_start()) {
        return Some(OutKind::Diff);
    }
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.len() < MIN_LINES {
        return None;
    }
    if is_grep(&lines) {
        return Some(OutKind::Grep);
    }
    if is_log(&lines) {
        return Some(OutKind::Log);
    }
    None
}

/// A unified diff: an explicit `diff --git` header, or a `--- `/`+++ ` file header
/// paired with at least one `@@` hunk header.
fn is_diff(t: &str) -> bool {
    if t.starts_with("diff --git ") {
        return true;
    }
    let has_hunk = t.starts_with("@@ ") || t.contains("\n@@ ");
    let has_file = t.starts_with("--- ") || t.contains("\n--- ") || t.contains("\n+++ ");
    has_hunk && has_file
}

/// At least three records and ≥75% of non-empty lines are `path:line:` matches.
fn is_grep(lines: &[&str]) -> bool {
    let matches = lines.iter().filter(|l| GREP_LINE.is_match(l)).count();
    matches >= MIN_LINES && matches * 4 >= lines.len() * 3
}

/// Log-shaped: enough lines, and either two outright failure lines or ≥30% of lines
/// carrying any level token. Conservative, to avoid windowing ordinary prose.
fn is_log(lines: &[&str]) -> bool {
    if lines.len() < MIN_LOG_LINES {
        return false;
    }
    let strong = lines
        .iter()
        .filter(|l| super::signals::STRONG.is_match(l))
        .count();
    let level = lines
        .iter()
        .filter(|l| super::signals::LEVEL.is_match(l))
        .count();
    strong >= 2 || level * 100 >= lines.len() * 30
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_git_diff() {
        let d = "diff --git a/x.rs b/x.rs\n--- a/x.rs\n+++ b/x.rs\n@@ -1 +1 @@\n-old\n+new";
        assert_eq!(detect(d), Some(OutKind::Diff));
    }

    #[test]
    fn detects_plain_unified_diff() {
        let d = "--- a/x\n+++ b/x\n@@ -1,2 +1,2 @@\n line\n-old\n+new";
        assert_eq!(detect(d), Some(OutKind::Diff));
    }

    #[test]
    fn detects_grep_output() {
        let g = "src/main.rs:10:    let x = 1;\n\
                 src/main.rs:42:    foo(x);\n\
                 src/lib.rs:7:pub fn foo() {}";
        assert_eq!(detect(g), Some(OutKind::Grep));
    }

    #[test]
    fn clock_times_are_not_grep() {
        // Numeric-only field before the colon must not read as a path:line record.
        let log = "12:00:01 service started ok\n\
                   12:00:02 handling request fine\n\
                   12:00:03 all good here now";
        assert_ne!(detect(log), Some(OutKind::Grep));
    }

    #[test]
    fn detects_log_with_failures() {
        let log = "INFO  build started\n\
                   INFO  compiling module a\n\
                   INFO  compiling module b\n\
                   ERROR failed to resolve symbol foo\n\
                   INFO  compiling module c\n\
                   ERROR type mismatch in bar\n\
                   INFO  compiling module d\n\
                   INFO  done with warnings";
        assert_eq!(detect(log), Some(OutKind::Log));
    }

    #[test]
    fn plain_prose_is_not_tool_output() {
        let prose = "The quarterly report covers revenue and costs.\n\
                     Margins improved across every region this year.\n\
                     The board approved the new budget unanimously.";
        assert_eq!(detect(prose), None);
    }
}
