//! Stage E — deduplication (exact + SimHash near-duplicate). Opt-in.
//!
//! Within a content segment, collapse repeated lines: an exact-duplicate line is
//! kept once with a `[×N]` count — semantically lossless, the repetition count is
//! preserved. With `near`, lines within a small SimHash
//! Hamming distance also collapse onto a representative (near-duplicate boilerplate
//! / log spam). Static, no embeddings. The content is kept once
//! and presented — never replaced by a `[REF:hash]` pointer.
//!
//! Off by default; InputTokens-gated, so it reverts if it doesn't reduce tokens.

use std::collections::HashMap;

use anyhow::Result;
use gaoya::simhash::{SimHash, SimHashBits, SimSipHasher64};
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct DedupStage {
    /// Also collapse near-duplicate lines (SimHash within `near_max_distance`).
    pub near: bool,
    /// Max SimHash Hamming distance treated as a near-duplicate.
    pub near_max_distance: u32,
}

impl Transform for DedupStage {
    fn name(&self) -> &str {
        "dedup"
    }

    fn gate_kind(&self) -> GateKind {
        GateKind::InputTokens
    }

    fn apply(
        &self,
        req: &mut Request,
        provider: &dyn Provider,
        _plan: &mut Vec<PlanEntry>,
    ) -> Result<()> {
        for ptr in crate::cache_zone::compressible_pointers(req, provider) {
            let Some(s) = req.get_str(&ptr).map(str::to_string) else {
                continue;
            };
            // On structured/positional data (CSV, tables, record arrays, event logs)
            // both near-dup merging and global first-occurrence exact merging are lossy:
            // they reorder records and hide row counts. Restrict those segments to
            // *adjacent-run* collapse — order-preserving, and contiguous log spam (the
            // big win) still folds. Prose gets the full global exact + optional near.
            let deduped = if crate::stages::tools::is_structured_segment(&s) {
                dedup_adjacent_lines(&s)
            } else {
                dedup_lines(&s, self.near, self.near_max_distance)
            };
            if deduped != s {
                req.set(&ptr, Value::String(deduped));
            }
        }
        Ok(())
    }
}

/// Tag name + close flag when the trimmed line is *solely* an XML/HTML open or
/// close tag — `<name …>` or `</name>` and nothing else (same structural signal
/// as the provider trim's tag-block detection). Identifier chars: alphanumeric,
/// `-`, `_`, `.`, `:`. Comments/PIs (`<!`, `<?`) and self-closing tags (`<x/>`,
/// a complete element, not a block delimiter) return `None`.
fn solo_tag_line(line: &str) -> Option<(&str, bool)> {
    let t = line.trim();
    let body = t.strip_prefix('<')?.strip_suffix('>')?;
    let (body, is_close) = match body.strip_prefix('/') {
        Some(rest) => (rest, true),
        None => (body, false),
    };
    let is_ident = |c: char| c.is_alphanumeric() || c == '-' || c == '_' || c == '.' || c == ':';
    let name_end = body.find(|c| !is_ident(c)).unwrap_or(body.len());
    let (name, rest) = body.split_at(name_end);
    if name.is_empty() || !name.starts_with(|c: char| c.is_alphabetic() || c == '_') {
        return None;
    }
    if is_close {
        // Close tag: only whitespace may follow the name.
        rest.chars()
            .all(char::is_whitespace)
            .then_some((name, true))
    } else {
        // Open tag: optional attributes after whitespace; no nested `<`/`>`
        // (that would mean more content on the line than the tag itself),
        // and not self-closing.
        ((rest.is_empty() || rest.starts_with(char::is_whitespace))
            && !rest.contains(['<', '>'])
            && !rest.trim_end().ends_with('/'))
        .then_some((name, false))
    }
}

/// Per-line fold-boundary signatures. `None` = ordinary line (folds freely).
/// `Some(sig)` = the line is solely an XML open/close tag: it delimits a block
/// and may only fold with a delimiter carrying an *equal* signature — the full
/// text of the delimited block for matched pairs, or a unique sentinel for
/// unmatched delimiters (which therefore never fold). Two delimiters thus merge
/// only when their entire blocks are byte-identical, so the wrapper of every
/// structurally distinct block survives a collapse.
fn block_signatures(lines: &[&str]) -> Vec<Option<String>> {
    let mut sigs: Vec<Option<String>> = vec![None; lines.len()];
    let mut stack: Vec<(&str, usize)> = Vec::new(); // (tag name, open-line index)
    for (i, line) in lines.iter().enumerate() {
        match solo_tag_line(line) {
            None => {}
            Some((name, false)) => {
                // Unique sentinel until matched (line index makes it unmergeable).
                sigs[i] = Some(format!("\u{0}unmatched:{i}"));
                stack.push((name, i));
            }
            Some((name, true)) => {
                sigs[i] = Some(format!("\u{0}unmatched:{i}"));
                if let Some(&(open_name, open_idx)) = stack.last()
                    && open_name == name
                {
                    stack.pop();
                    let block = lines[open_idx..=i].join("\n");
                    sigs[open_idx] = Some(block.clone());
                    sigs[i] = Some(block);
                }
            }
        }
    }
    sigs
}

/// Collapse only *adjacent* runs of identical lines (`line [×N]`), keeping every
/// distinct line in its original position. The safe variant for structured segments,
/// where [`dedup_lines`]'s global first-occurrence merge would reorder records or
/// hide row counts. Blank-line runs pass through (they are structure, not content).
fn dedup_adjacent_lines(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return text.to_string();
    }
    let sigs = block_signatures(&lines);
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut collapsed = false;
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let mut n = 1;
        if !line.trim().is_empty() {
            while i + n < lines.len() && lines[i + n] == line && sigs[i + n] == sigs[i] {
                n += 1;
            }
        }
        if n > 1 {
            collapsed = true;
            out.push(format!("{line} [×{n}]"));
        } else {
            out.push(line.to_string());
        }
        i += n;
    }
    // Nothing collapsed → input verbatim (same trailing-newline rationale as dedup_lines).
    if !collapsed {
        return text.to_string();
    }
    out.join("\n")
}

/// Collapse repeated lines, keeping each group once with a `[×N]` count. Blank
/// lines pass through untouched. Exact duplicates always group; with `near`, lines
/// within `max_dist` SimHash bits group onto the first (representative) line.
fn dedup_lines(text: &str, near: bool, max_dist: u32) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return text.to_string();
    }

    let sigs = block_signatures(&lines);
    let mut group_of: Vec<Option<usize>> = vec![None; lines.len()];
    let mut reps: Vec<u64> = Vec::new(); // representative SimHash per group
    let mut counts: Vec<usize> = Vec::new();
    // Keyed by (line, block signature): delimiter lines only merge when their
    // whole blocks are byte-identical; ordinary lines carry a `None` signature.
    let mut exact: HashMap<(&str, Option<&str>), usize> = HashMap::new();
    let hasher = make_simhasher();

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue; // blanks are structure, not content
        }
        let sig = sigs[i].as_deref();
        if let Some(&g) = exact.get(&(*line, sig)) {
            group_of[i] = Some(g);
            counts[g] += 1;
            continue;
        }
        let sh = line_simhash(&hasher, line);
        if near
            && sig.is_none()
            && let Some(g) = reps
                .iter()
                .position(|&rh| rh.hamming_distance(&sh) <= max_dist as usize)
        {
            group_of[i] = Some(g);
            counts[g] += 1;
            exact.insert((line, sig), g);
            continue;
        }
        let g = reps.len();
        reps.push(sh);
        counts.push(1);
        exact.insert((line, sig), g);
        group_of[i] = Some(g);
    }

    // Nothing collapsed (every group is a singleton) → return the input verbatim. Joining
    // `lines()` would otherwise drop a trailing newline / collapse blank runs, making the
    // caller see a spurious change and pay a needless rewrite + re-tokenization.
    if counts.iter().all(|&n| n <= 1) {
        return text.to_string();
    }

    let mut emitted = vec![false; reps.len()];
    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        match group_of[i] {
            None => out.push((*line).to_string()),
            Some(g) => {
                if !emitted[g] {
                    emitted[g] = true;
                    let n = counts[g];
                    out.push(if n > 1 {
                        format!("{line} [×{n}]")
                    } else {
                        (*line).to_string()
                    });
                }
            }
        }
    }
    out.join("\n")
}

/// Construct the shared 64-bit SimHash instance (SipHash seed 1/2, matching Stage E
/// and the sizing knee). Centralised here so both dedup and sizing use the same hasher
/// configuration without repeating the generic turbofish.
pub(crate) fn make_simhasher() -> SimHash<SimSipHasher64, u64, 64> {
    SimHash::<SimSipHasher64, u64, 64>::new(SimSipHasher64::new(1, 2))
}

/// 64-bit SimHash of a line's lexical word tokens, via gaoya (Charikar). Tokens come
/// from the shared Unicode word segmenter, so near-dup detection works across scripts.
/// An empty line hashes to 0.
fn line_simhash(hasher: &SimHash<SimSipHasher64, u64, 64>, s: &str) -> u64 {
    let words = crate::stages::tools::lex_words(s);
    if words.is_empty() {
        return 0;
    }
    hasher.create_signature(words.iter())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    #[test]
    fn exact_dedup_counts_repeats() {
        let out = dedup_lines("a\nb\na\na", false, 0);
        assert_eq!(out, "a [×3]\nb");
    }

    #[test]
    fn blank_lines_are_preserved() {
        let out = dedup_lines("x\n\nx\n\ny", false, 0);
        assert_eq!(out, "x [×2]\n\n\ny");
    }

    #[test]
    fn no_change_when_all_unique() {
        let out = dedup_lines("alpha\nbeta\ngamma", false, 0);
        assert_eq!(out, "alpha\nbeta\ngamma");
    }

    #[test]
    fn unique_input_with_trailing_newline_is_returned_verbatim() {
        // No collapse: must return the input byte-for-byte (incl. the trailing newline),
        // so the stage doesn't pay a spurious rewrite + re-tokenization.
        let input = "alpha\nbeta\ngamma\n";
        let out = dedup_lines(input, false, 0);
        assert_eq!(
            out, input,
            "trailing newline preserved when nothing collapses"
        );
        // And the stage-level guard: `deduped != s` must be false here.
        let input2 = "one\ntwo\n\nthree\n";
        assert_eq!(
            dedup_lines(input2, false, 0),
            input2,
            "blank runs preserved too"
        );
    }

    #[test]
    fn simhash_distance_small_for_similar_large_for_different() {
        let hasher = make_simhasher();
        let a = line_simhash(&hasher, "the quick brown fox jumps over the lazy dog");
        let b = line_simhash(&hasher, "the quick brown fox jumps over the lazy dogs");
        let c = line_simhash(
            &hasher,
            "completely unrelated content about finance reports",
        );
        assert_eq!(a.hamming_distance(&a), 0);
        assert!(
            a.hamming_distance(&b) < a.hamming_distance(&c),
            "near text is closer than unrelated"
        );
    }

    #[test]
    fn near_dedup_collapses_similar_lines() {
        let text = "Connection retry attempt number one failed\n\
                    Connection retry attempt number two failed\n\
                    Connection retry attempt number three failed";
        let exact = dedup_lines(text, false, 3);
        assert_eq!(exact.lines().count(), 3, "exact mode keeps distinct lines");
        let near = dedup_lines(text, true, 12);
        assert!(
            near.lines().count() < 3,
            "near mode collapses the similar retry lines"
        );
        assert!(near.contains("[×"), "collapsed group carries a count");
    }

    #[test]
    fn near_dedup_skips_structured_segments() {
        // A CSV: near-dup collapse would merge distinct rows. The structured-guard keeps
        // `near` off for this segment, so every record survives (only exact dups would
        // collapse, and there are none here).
        let csv = "id,name,role\n1,Ann,Sales\n2,Bob,Sales\n3,Cy,Sales\n4,Di,Tech";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":csv}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DedupStage {
            near: true,
            near_max_distance: 12,
        })];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(
            now.contains("Bob") && now.contains("Cy") && now.contains("Di"),
            "distinct CSV rows survive (near-dup disabled on structured data)"
        );
        assert!(!now.contains("[×"), "no near-dup collapse on a CSV");
    }

    #[test]
    fn structured_segment_keeps_non_adjacent_duplicates() {
        // A CSV with an identical row at positions 2 and 4: global merge would pull the
        // second occurrence onto the first, reordering records and hiding the row count.
        // On structured segments only adjacent runs may collapse — so this is untouched.
        let csv = "id,name,role\n1,Ann,Sales\n2,Bob,Ops\n1,Ann,Sales\n3,Cy,Ops";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":csv}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DedupStage {
            near: false,
            near_max_distance: 0,
        })];
        let _ = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let now = req.get_str("/messages/0/content").unwrap();
        assert_eq!(
            now, csv,
            "non-adjacent duplicate rows in a CSV must not merge"
        );
    }

    #[test]
    fn adjacent_collapse_preserves_order() {
        // Adjacent runs fold in place; distinct lines keep their relative order.
        let log = "a,1\nb,2\nb,2\nb,2\nc,3";
        let out = dedup_adjacent_lines(log);
        assert_eq!(out, "a,1\nb,2 [×3]\nc,3");
        // Non-adjacent repeats stay separate.
        let interleaved = "x\ny\nx\ny";
        assert_eq!(dedup_adjacent_lines(interleaved), interleaved);
    }

    #[test]
    fn differing_tag_blocks_keep_their_delimiters() {
        // Regression (capture 1781260115569010-3cd0e6): two same-tag blocks
        // sharing header lines but differing tails. The fold must not absorb
        // the second block's open/close delimiter lines — its unique fields
        // must stay inside its own wrappers.
        let text = "<task-notification>\n\
                    <task-id>br0iqqk7p</task-id>\n\
                    <summary>Monitor event</summary>\n\
                    </task-notification>\n\
                    \n\
                    <task-notification>\n\
                    <task-id>br0iqqk7p</task-id>\n\
                    <status>completed</status>\n\
                    <summary>stream ended</summary>\n\
                    </task-notification>";
        let out = dedup_lines(text, false, 0);
        let opens = out.matches("<task-notification>").count();
        let closes = out.matches("</task-notification>").count();
        assert_eq!(opens, 2, "both open tags survive:\n{out}");
        assert_eq!(closes, 2, "both close tags survive:\n{out}");
        // Unique fields of block 2 sit between its own open and close tags.
        let second_open = out
            .match_indices("<task-notification>")
            .nth(1)
            .expect("second open")
            .0;
        let second_close = out.rfind("</task-notification>").expect("second close");
        let status = out.find("<status>completed</status>").expect("status kept");
        assert!(
            second_open < status && status < second_close,
            "unique field stays inside second block's wrappers:\n{out}"
        );
        // The shared header line may still fold with a count.
        assert!(
            out.contains("<task-id>br0iqqk7p</task-id> [×2]"),
            "shared inner lines still fold:\n{out}"
        );
        // Same protection on the adjacent (structured-segment) variant.
        let adj = dedup_adjacent_lines("<t>\n<t>\nx\n</t>\n</t>");
        assert_eq!(
            adj, "<t>\n<t>\nx\n</t>\n</t>",
            "nested delimiters untouched"
        );
    }

    #[test]
    fn byte_identical_tag_blocks_still_fold() {
        // Truly identical duplicate blocks (genuine repeated notifications)
        // may fold whole — existing savings preserved.
        let block = "<note>\nsame body\n</note>";
        let text = format!("{block}\n{block}");
        let out = dedup_lines(&text, false, 0);
        assert_eq!(out, "<note> [×2]\nsame body [×2]\n</note> [×2]");
    }

    #[test]
    fn solo_tag_line_detection() {
        assert_eq!(solo_tag_line("<name>"), Some(("name", false)));
        assert_eq!(
            solo_tag_line("  <a-b_c.d:e attr=\"1\">  "),
            Some(("a-b_c.d:e", false))
        );
        assert_eq!(solo_tag_line("</name>"), Some(("name", true)));
        assert_eq!(solo_tag_line("<x>y</x>"), None, "content on the line");
        assert_eq!(
            solo_tag_line("<x/>"),
            None,
            "self-closing is a whole element"
        );
        assert_eq!(solo_tag_line("<!-- c -->"), None);
        assert_eq!(solo_tag_line("<?xml version=\"1.0\"?>"), None);
        assert_eq!(solo_tag_line("plain text"), None);
        assert_eq!(solo_tag_line("< name>"), None, "no space before name");
    }

    #[test]
    fn stage_reduces_tokens_on_repetitive_content() {
        let spam = std::iter::repeat_n("WARN cache miss for key user:session", 40)
            .collect::<Vec<_>>()
            .join("\n");
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":spam}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(DedupStage {
            near: false,
            near_max_distance: 3,
        })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied);
        assert!(out.input_tokens_after < out.input_tokens_before);
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(now.contains("[×40]"), "40 identical lines collapse to one");
    }
}
