//! Shared level/failure signal regexes for the tool-output stage.
//!
//! [`STRONG`] / [`WARN`] score *lines already classified as a log* (keep errors).
//! Kind detection uses [`LINE_LEVEL`] / [`LINE_STRONG`] so identifier hits in
//! source (`throws Exception`, `io::Error`, `console.error`) are not log-shaped.
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

/// Optional leading timestamp / bracketed tag before a line-start level token.
const LINE_PREFIX: &str = r"^[\t ]*(?:\[[^\]]{0,80}\]\s*)?(?:(?:\d{4}-\d{2}-\d{2}[T ]\S+)\s+)?";

/// Log-level token at the start of a line (after optional timestamp/tag). Used to
/// decide whether a segment is log-shaped. Does not include `exception`/`panic`
/// as identifiers — those fire inside source.
pub(crate) static LINE_LEVEL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(&format!(
        r"(?i){LINE_PREFIX}(?:error|warn(?:ing)?|info|debug|trace|fatal|fail)(?:\b|\[)"
    ))
    .unwrap()
});

/// Failure marker at the start of a line (after optional timestamp/tag).
pub(crate) static LINE_STRONG: Lazy<Regex> = Lazy::new(|| {
    Regex::new(&format!(
        r"(?i){LINE_PREFIX}(?:not ok\b|error\b|fatal\b|fail(?:ed|ure)?\b|panic(?:ked)?\b|traceback\b|segfault\b|assert(?:ion)?(?:error)?\b|exception in thread\b)"
    ))
    .unwrap()
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

    #[test]
    fn line_level_is_prefix_not_identifier() {
        assert!(LINE_LEVEL.is_match("INFO  compiling module 3"));
        assert!(LINE_LEVEL.is_match("[10:02:31Z] ERROR src/worker/pool.rs:214"));
        assert!(LINE_LEVEL.is_match("error[E0308]: mismatched types"));
        assert!(LINE_STRONG.is_match("not ok 19 - normalize: backslash path"));
        assert!(
            LINE_STRONG.is_match("Exception in thread \"main\" java.lang.NullPointerException")
        );
        assert!(!LINE_LEVEL.is_match("    throw new Exception(msg);"));
        assert!(!LINE_LEVEL.is_match("    logger.info(\"step\");"));
        assert!(!LINE_LEVEL.is_match("    console.error(err);"));
        assert!(!LINE_LEVEL.is_match("fn boom() -> io::Error {"));
        assert!(!LINE_STRONG.is_match("    throw new Exception(msg);"));
    }
}
