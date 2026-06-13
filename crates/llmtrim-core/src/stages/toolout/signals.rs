//! Shared level/failure signal regexes for the tool-output stage.
//!
//! Both the kind detector ([`super::detect`]) and the line-priority scorer
//! ([`super::priority`]) test lines for failure / log-level tokens. Defining the patterns
//! here once keeps the two in lockstep — they classified the same tokens before, in two
//! verbatim copies that would silently drift apart on the next edit.
//!
//! These are tokens *machine-emitted* by runtimes and build tools (`ERROR`, `FATAL`,
//! `Traceback`, `panicked`), not human prose (see the module note in `mod.rs`), so a fixed
//! English set is appropriate; locale-specific terms from the user's request ride the
//! query-overlap bonus, which is Unicode-segmented.

use once_cell::sync::Lazy;
use regex::Regex;

/// A failure-level signal anywhere in a line (the strongest severity).
pub(crate) static STRONG: Lazy<Regex> = Lazy::new(|| {
    // `not ok` is TAP's failure marker (node --test, prove) — it carries none of the
    // usual tokens. Bare `failure` (no left word boundary) catches camelCase diagnostics
    // like TAP's `failureType: 'testCodeFailure'`, which `\bfailure\b` misses.
    Regex::new(r"(?i)\b(error|fatal|fail(?:ed|ure)?|panic(?:ked)?|exception|traceback|segfault|assert(?:ion)?|not ok)\b|(?i)failure")
        .unwrap()
});

/// A warning-level signal anywhere in a line.
pub(crate) static WARN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(warn(?:ing)?|deprecat)").unwrap());

/// Any log-level token (the strong ones plus informational levels) — used to decide
/// whether a segment is log-shaped at all.
pub(crate) static LEVEL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)\b(error|warn|info|debug|trace|fatal|fail|panic|exception)\b").unwrap()
});

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strong_matches_tap_failure_markers() {
        assert!(STRONG.is_match("not ok 19 - normalize: backslash path"));
        assert!(STRONG.is_match("  failureType: 'testCodeFailure'"));
        assert!(STRONG.is_match("[10:02:31Z] ERROR src/worker/pool.rs:214"));
        assert!(!STRONG.is_match("ok 19 - normalize: backslash path"));
        assert!(!STRONG.is_match("ok 30 - isInList: FAIL_OPEN path returns true"));
    }
}
