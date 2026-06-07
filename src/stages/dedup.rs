//! Stage E — deduplication (exact + SimHash near-duplicate). Opt-in.
//!
//! Within a content segment, collapse repeated lines: an exact-duplicate line is
//! kept once with a `[×N]` count — semantically lossless, the repetition count is
//! preserved (mirrors RTK's log dedup). With `near`, lines within a small SimHash
//! Hamming distance also collapse onto a representative (near-duplicate boilerplate
//! / log spam). Static, no embeddings (spec §5). Per §5 the content is kept once
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
        for ptr in provider.content_text_pointers(req) {
            let Some(s) = req.get_str(&ptr).map(str::to_string) else {
                continue;
            };
            // Near-dup collapse is lossy: on structured/positional data (CSV, tables,
            // record arrays) it merges distinct rows. Restrict those segments to exact
            // dedup (lossless `[×N]`); prose still gets near-dup.
            let near = self.near && !crate::stages::tools::is_structured_segment(&s);
            let deduped = dedup_lines(&s, near, self.near_max_distance);
            if deduped != s {
                req.set(&ptr, Value::String(deduped));
            }
        }
        Ok(())
    }
}

/// Collapse repeated lines, keeping each group once with a `[×N]` count. Blank
/// lines pass through untouched. Exact duplicates always group; with `near`, lines
/// within `max_dist` SimHash bits group onto the first (representative) line.
fn dedup_lines(text: &str, near: bool, max_dist: u32) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < 2 {
        return text.to_string();
    }

    let mut group_of: Vec<Option<usize>> = vec![None; lines.len()];
    let mut reps: Vec<u64> = Vec::new(); // representative SimHash per group
    let mut counts: Vec<usize> = Vec::new();
    let mut exact: HashMap<&str, usize> = HashMap::new();
    let hasher = SimHash::<SimSipHasher64, u64, 64>::new(SimSipHasher64::new(1, 2));

    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue; // blanks are structure, not content
        }
        if let Some(&g) = exact.get(*line) {
            group_of[i] = Some(g);
            counts[g] += 1;
            continue;
        }
        let sh = line_simhash(&hasher, line);
        if near
            && let Some(g) = reps
                .iter()
                .position(|&rh| rh.hamming_distance(&sh) <= max_dist as usize)
        {
            group_of[i] = Some(g);
            counts[g] += 1;
            exact.insert(line, g);
            continue;
        }
        let g = reps.len();
        reps.push(sh);
        counts.push(1);
        exact.insert(line, g);
        group_of[i] = Some(g);
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
    fn simhash_distance_small_for_similar_large_for_different() {
        let hasher = SimHash::<SimSipHasher64, u64, 64>::new(SimSipHasher64::new(1, 2));
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
