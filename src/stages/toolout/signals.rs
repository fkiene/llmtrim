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
    Regex::new(r"(?i)\b(error|fatal|fail(?:ed|ure)?|panic(?:ked)?|exception|traceback|segfault|assert(?:ion)?)\b")
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
