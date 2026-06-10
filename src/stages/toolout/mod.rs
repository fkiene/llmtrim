//! Stage T — tool-output compression (feature #1).
//!
//! Tool results (logs, diffs, grep output) are the bulk of an agent's context and the
//! noisiest: 10k-line build logs around a handful of errors, 200-line diffs changing
//! three lines, grep dumps with one relevant hit per file. The prose-oriented stages
//! (retrieve, ngram) don't fit this shape. This stage detects the *kind* of each tool
//! output and routes it to a purpose-built lexical compressor:
//!
//! - **log**  → level/stack-aware line selection (+ Drain template collapse, [`template`])
//! - **diff** → file/hunk capping + context trimming
//! - **grep** → per-file match selection
//!
//! All three keep the structurally important lines (errors, changes, query-relevant
//! matches), window the rest under an adaptive budget ([`crate::stages::sizing`]), and
//! replace dropped runs with a positional elision marker (`[… N lines omitted …]`, like
//! the `retrieve` stage). If the agent needs the dropped detail it re-runs the tool —
//! fresher than a stored copy, and what a tool-using agent does naturally. Lossy,
//! `InputTokens`-gated (reverts if it doesn't cut tokens), `Content`-scoped. Zero model
//! calls (spec §5).
//!
//! Note on universality (spec §5): the level/failure keywords scored below are tokens
//! *machine-emitted* by runtimes and build tools (`ERROR`, `FATAL`, `Traceback`,
//! `panicked`), not human prose, so a fixed set is appropriate. Locale-specific terms
//! from the user's request are handled by the query-overlap bonus, which is
//! Unicode-segmented via [`lex_words`].

mod detect;
mod diff;
mod generated;
mod grep;
mod log;
mod normalize;
mod plaintext;
mod signals;
mod template;

use std::collections::HashSet;

use anyhow::Result;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Scope, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::stages::tools::lex_words;
use detect::OutKind;

/// First lines always kept for leading context.
const HEAD: usize = 2;
/// Last lines always kept for trailing context (final status / summary).
const TAIL: usize = 2;
/// Priority at or above which a line is force-kept regardless of budget (an error).
const FORCE_PRIORITY: f64 = 1.0;
/// Floor on the adaptive keep budget — never window below this many lines.
const MIN_KEEP: usize = 3;

/// In `Auto`, a segment goes aggressive only when it has at least this many units …
const AUTO_MIN_LINES: usize = 60;
/// … and its signal (errors / changed lines / distinct files) is at most this percent
/// of them. Big + signal-sparse ⇒ most of it is droppable noise, so signal-only saves
/// hugely with the answer preserved; otherwise stay adaptive (keep more, safer).
const AUTO_SIGNAL_PCT: usize = 25;

/// The compression aggressiveness a run asks for (from config / preset). Public because
/// it is the type of [`ToolOutputStage::mode`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModeSetting {
    /// Always window to the adaptive budget (keep more).
    Adaptive,
    /// Always keep signal-only (errors / changes / one-per-file) + a summary.
    Aggressive,
    /// Decide per segment by noise density (the tuned default).
    Auto,
}

impl ModeSetting {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "adaptive" => Self::Adaptive,
            "aggressive" => Self::Aggressive,
            _ => Self::Auto,
        }
    }
}

/// The resolved mode for one segment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Mode {
    Adaptive,
    Aggressive,
}

/// Resolve the split for one segment. `signal` = must-keep units (errors / changed
/// lines / distinct files); `total` = the unit count the compressor windows over.
pub(crate) fn pick_mode(setting: ModeSetting, total: usize, signal: usize) -> Mode {
    match setting {
        ModeSetting::Adaptive => Mode::Adaptive,
        ModeSetting::Aggressive => Mode::Aggressive,
        ModeSetting::Auto => {
            if total >= AUTO_MIN_LINES && signal * 100 <= total * AUTO_SIGNAL_PCT {
                Mode::Aggressive
            } else {
                Mode::Adaptive
            }
        }
    }
}

pub struct ToolOutputStage {
    /// Upper bound on lines kept per tool-output segment (the adaptive budget ceiling).
    pub max_lines: usize,
    /// Skip segments shorter than this many lines — below it the markers cost more than
    /// the drop saves.
    pub min_lines: usize,
    /// Run the lossless Drain template collapse on logs before windowing.
    pub template: bool,
    /// Adaptive/aggressive split: `Adaptive` (always window), `Aggressive` (always
    /// signal-only), or `Auto` (decide per segment by noise density).
    pub mode: ModeSetting,
}

/// Per-segment knobs handed to each kind's compressor.
pub(crate) struct Ctx {
    pub max_lines: usize,
    pub template: bool,
    /// Adaptive/aggressive split for this run (resolved per-segment when `Auto`).
    pub mode: ModeSetting,
}

impl Transform for ToolOutputStage {
    fn name(&self) -> &str {
        "toolout"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn scope(&self) -> Scope {
        Scope::Content
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        let pointers = crate::cache_zone::compressible_pointers(req, provider);

        // Pre-pass (feature #6): strip ANSI escapes and collapse carriage-return progress
        // on each candidate *before* detection. This is lossless for the model and lets
        // colored cargo/pytest output be detected/scored at all (level tokens are no longer
        // hidden inside escapes). `changed` flags segments the pre-pass altered so a
        // normalized-but-not-windowed segment still ships its (smaller) cleaned form.
        let texts: Vec<Option<(String, bool)>> = pointers
            .iter()
            .map(|p| req.get_str(p).map(normalize::normalize))
            .collect();
        let kinds: Vec<Option<OutKind>> = texts
            .iter()
            .map(|t| t.as_ref().and_then(|(t, _)| detect::detect(t)))
            .collect();

        // The "ask" = the short segments (instruction / question). Tool-output lines
        // overlapping it are biased to survive. Built from length, not kind, so it stays
        // the question even when the long segments are unrecognized (PlainText) output.
        let query: HashSet<String> = texts
            .iter()
            .filter_map(|t| t.as_ref().map(|(t, _)| t.as_str()))
            .filter(|t| t.lines().count() < self.min_lines)
            .flat_map(lex_words)
            .collect();

        let ctx = Ctx {
            max_lines: self.max_lines,
            template: self.template,
            mode: self.mode,
        };

        for ((ptr, text), kind) in pointers.iter().zip(&texts).zip(&kinds) {
            let Some((text, normalized)) = text else {
                continue;
            };
            if text.lines().count() < self.min_lines {
                // Too small to window, but if the pre-pass cleaned it, ship the cleaner
                // form (the gate reverts if it didn't actually save tokens).
                if *normalized {
                    req.set(ptr, Value::String(text.clone()));
                }
                continue;
            }
            let compressed = match kind {
                Some(OutKind::Log) => log::compress(text, &ctx, &query),
                Some(OutKind::Grep) => grep::compress(text, &ctx, &query),
                Some(OutKind::Diff) => diff::compress(text, &ctx, &query),
                // Unrecognized shape. First try the generated/lockfile near-total elision
                // (feature #7, high-confidence machine-noise shapes only); otherwise the
                // redundancy-gated generic fallback ("any tool").
                None => {
                    generated::compress(text).or_else(|| plaintext::compress(text, &ctx, &query))
                }
            };
            if let Some(compressed) = compressed {
                req.set(ptr, Value::String(compressed));
            } else if *normalized {
                // No per-kind compression, but the pre-pass cleaned it → ship the cleaned
                // form so the ANSI/CR savings aren't lost.
                req.set(ptr, Value::String(text.clone()));
            }
        }
        Ok(())
    }
}

/// Priority assigned to a warning line (also the floor for "this is a warning" in the
/// summary count).
pub(crate) const WARN_PRIORITY: f64 = 0.6;

/// Intrinsic importance of a log/output line in `[0,1]`: failures dominate, warnings
/// next, indented stack-frame continuations kept above plain noise.
pub(crate) fn priority(line: &str) -> f64 {
    if signals::STRONG.is_match(line) {
        FORCE_PRIORITY
    } else if signals::WARN.is_match(line) {
        WARN_PRIORITY
    } else if line.starts_with(' ') || line.starts_with('\t') {
        0.5 // stack-trace / continuation line — keep with its error
    } else {
        0.3
    }
}

/// Relevance bonus for a line overlapping the request's words (capped, additive on top
/// of [`priority`]). Unicode-segmented, so it works in any language.
pub(crate) fn query_bonus(line: &str, query: &HashSet<String>) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    let hits = lex_words(line)
        .into_iter()
        .filter(|w| query.contains(w))
        .count();
    (hits as f64 * 0.1).min(0.3)
}

/// Choose which of `scores.len()` lines to keep: always the first [`HEAD`] and last
/// [`TAIL`], always any at or above `force`, then fill by descending score (ties by
/// original order) until `k` are kept. Forced lines may exceed `k` — errors are never
/// dropped to hit a budget.
pub(crate) fn select_keep(scores: &[f64], k: usize, force: f64) -> Vec<bool> {
    let n = scores.len();
    let mut keep = vec![false; n];
    for slot in keep.iter_mut().take(HEAD.min(n)) {
        *slot = true;
    }
    for slot in keep.iter_mut().skip(n.saturating_sub(TAIL)) {
        *slot = true;
    }
    for (i, &s) in scores.iter().enumerate() {
        if s >= force {
            keep[i] = true;
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut count = keep.iter().filter(|&&x| x).count();
    for &i in &order {
        if count >= k {
            break;
        }
        if !keep[i] {
            keep[i] = true;
            count += 1;
        }
    }
    keep
}

/// Replace a dropped run of lines with a positional elision marker — dropped content is
/// referenced by position, never stored (matches the `retrieve` stage's convention). If
/// the agent needs it back, it re-runs the tool, which is fresher than a stored copy.
pub(crate) fn elide(dropped: &[&str]) -> String {
    format!("[… {} lines omitted …]", dropped.len())
}

/// Reassemble `lines` in original order, keeping those flagged in `keep` and collapsing
/// each maximal dropped run into one [`elide`] marker. Shared by the log and grep
/// compressors (both select at line granularity).
pub(crate) fn rebuild(lines: &[&str], keep: &[bool]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        if keep[i] {
            out.push(lines[i].to_string());
            i += 1;
        } else {
            let start = i;
            while i < lines.len() && !keep[i] {
                i += 1;
            }
            out.push(elide(&lines[start..i]));
        }
    }
    out.join("\n")
}

#[cfg(test)]
pub(crate) fn test_ctx() -> Ctx {
    Ctx {
        max_lines: 30,
        template: true,
        mode: ModeSetting::Adaptive,
    }
}

#[cfg(test)]
mod tests {
    use super::{Mode, ModeSetting, pick_mode};

    #[test]
    fn auto_goes_aggressive_only_when_big_and_sparse() {
        assert_eq!(
            pick_mode(ModeSetting::Auto, 100, 5),
            Mode::Aggressive,
            "big + sparse"
        );
        assert_eq!(
            pick_mode(ModeSetting::Auto, 100, 40),
            Mode::Adaptive,
            "big but dense"
        );
        assert_eq!(
            pick_mode(ModeSetting::Auto, 30, 1),
            Mode::Adaptive,
            "too small"
        );
    }

    #[test]
    fn forced_modes_ignore_the_signal() {
        assert_eq!(pick_mode(ModeSetting::Aggressive, 10, 9), Mode::Aggressive);
        assert_eq!(pick_mode(ModeSetting::Adaptive, 1000, 1), Mode::Adaptive);
    }

    #[test]
    fn mode_setting_parses_with_auto_fallback() {
        assert_eq!(ModeSetting::parse("aggressive"), ModeSetting::Aggressive);
        assert_eq!(ModeSetting::parse("ADAPTIVE"), ModeSetting::Adaptive);
        assert_eq!(ModeSetting::parse("auto"), ModeSetting::Auto);
        assert_eq!(ModeSetting::parse("nonsense"), ModeSetting::Auto);
    }
}
