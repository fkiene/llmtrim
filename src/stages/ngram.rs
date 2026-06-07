//! Stage E+ — reversible n-gram abbreviation dictionary (lossless input). Opt-in.
//!
//! Finds the most-repeated multi-word phrases across the request's content, replaces
//! each with a short placeholder (`§1`, `§2`, …), and injects a one-line legend
//! defining them. The model reads the legend to recover meaning (like the TOON
//! legend), so information is preserved while repeated boilerplate — recurring API
//! names, file paths, legal/spec phrases — collapses. This is redundancy that
//! Stage E's line/SimHash dedup misses, because it spans *within* and *across* lines.
//! CompactPrompt n-gram component (arXiv:2510.18043).
//!
//! InputTokens-gated: reverts unless the legend pays for itself. Aborts losslessly
//! if the placeholder marker already occurs in the content.
//!
//! Universality: candidates are word n-grams over whitespace-delimited tokens, so this
//! covers any space-separated script (Latin, Cyrillic, Greek, Arabic, …) and gracefully
//! no-ops on scripts without inter-word spaces (CJK, Thai) — a word-level glossary
//! doesn't apply there, so that content is left verbatim rather than mis-abbreviated.

use std::collections::HashMap;

use anyhow::Result;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct NgramStage {
    /// Maximum dictionary entries (placeholders) to introduce.
    pub max_entries: usize,
}

impl Transform for NgramStage {
    fn name(&self) -> &str {
        "ngram"
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
        // Keep (pointer, text) pairs so write-back stays aligned even if a pointer
        // yields no string.
        let segs: Vec<(String, String)> = provider
            .content_text_pointers(req)
            .into_iter()
            .filter_map(|p| req.get_str(&p).map(|s| (p, s.to_string())))
            .collect();
        if segs.is_empty() || segs.iter().any(|(_, t)| t.contains('§')) {
            return Ok(()); // collision with the placeholder marker → stay lossless
        }

        // Abbreviate PROSE only. Inside structured data (JSON, tables, config, code), every
        // token is load-bearing — glossary-abbreviating it makes the model misread the
        // data (e.g. miscount records: adult −100pp in the bench). Skip those segments.
        let prose: Vec<usize> = (0..segs.len())
            .filter(|&i| !crate::stages::tools::is_structured_segment(&segs[i].1))
            .collect();
        if prose.is_empty() {
            return Ok(());
        }

        let mut working: Vec<String> = prose.iter().map(|&i| segs[i].1.clone()).collect();
        let mut committed: Vec<(String, String)> = Vec::new();
        // Longest phrases first: more savings per hit, and replacing them first
        // consumes their sub-phrases so those drop below the frequency threshold.
        for phrase in candidate_phrases(&working) {
            if committed.len() >= self.max_entries {
                break;
            }
            let occ: usize = working
                .iter()
                .map(|t| t.matches(phrase.as_str()).count())
                .sum();
            if occ < 2 {
                continue; // no longer repeats after prior replacements → wouldn't pay
            }
            let ph = format!("§{}", committed.len() + 1);
            for t in working.iter_mut() {
                *t = t.replace(phrase.as_str(), &ph);
            }
            committed.push((ph, phrase));
        }
        if committed.is_empty() {
            return Ok(());
        }

        for (wi, &i) in prose.iter().enumerate() {
            req.set(&segs[i].0, Value::String(working[wi].clone()));
        }
        let legend = committed
            .iter()
            .map(|(ph, phrase)| format!("{ph}={phrase}"))
            .collect::<Vec<_>>()
            .join("; ");
        const GLOSSARY_TMPL: &str = include_str!("../../prompts/ngram_glossary.txt");
        provider.add_system_instruction(req, &GLOSSARY_TMPL.replace("{terms}", &legend));
        Ok(())
    }
}

/// Repeated multi-word phrases (n = 2..=6 words, frequency ≥ 2), sorted longest
/// first then most frequent — the greedy commit order that maximizes savings and
/// lets long phrases subsume their sub-grams.
fn candidate_phrases(texts: &[String]) -> Vec<String> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for t in texts {
        let words: Vec<&str> = t.split_whitespace().collect();
        for n in 2..=6 {
            if words.len() < n {
                break;
            }
            for w in words.windows(n) {
                *counts.entry(w.join(" ")).or_insert(0) += 1;
            }
        }
    }
    let mut cands: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(p, c)| *c >= 2 && p.len() >= 8)
        .collect();
    cands.sort_by(|a, b| {
        let words = |s: &str| s.split_whitespace().count();
        words(&b.0)
            .cmp(&words(&a.0))
            .then(b.1.cmp(&a.1))
            .then(a.0.cmp(&b.0))
    });
    cands.into_iter().map(|(p, _)| p).collect()
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
    fn candidates_include_the_repeated_phrase() {
        let phrase = "the quarterly financial report";
        let text = format!("{phrase} grew. later {phrase} fell. again {phrase} held.");
        let cands = candidate_phrases(&[text]);
        assert!(
            cands.iter().any(|p| p == phrase),
            "frequent phrase is a candidate"
        );
    }

    #[test]
    fn stage_abbreviates_repeated_boilerplate_with_legend() {
        let p = "the internal configuration service endpoint";
        let content = format!(
            "{p} failed. retry {p}. then {p} again. {p} more. {p} keeps. {p} still. {p} yet. finally {p} ok."
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(NgramStage { max_entries: 32 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "repeated-phrase abbreviation cuts tokens"
        );
        let sys = req
            .raw()
            .pointer("/messages/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert!(
            sys.contains("Glossary") && sys.contains(p),
            "legend defines phrase"
        );
        let user = req
            .raw()
            .pointer("/messages/1/content")
            .and_then(Value::as_str)
            .unwrap();
        assert!(user.contains('§'), "content uses the placeholder");
        // No unused glossary entries: exactly one placeholder defined for this input.
        assert_eq!(
            sys.matches('§').count(),
            1,
            "only the phrase that pays off is committed"
        );
    }

    #[test]
    fn skips_json_record_arrays() {
        // adult-style: repeated "Sales" inside a record array + a counting question.
        // Abbreviating "Sales" would make the model miscount → must be left verbatim.
        let content = "[{\"occupation\":\"Sales\"},{\"occupation\":\"Sales\"},{\"occupation\":\"Sales\"},{\"occupation\":\"Tech\"}]\n\nHow many records have occupation Sales? Answer with the number.";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(NgramStage { max_entries: 32 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            !out.stages[0].applied,
            "structured records are not abbreviated"
        );
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(
            !now.contains('§'),
            "no placeholder injected into record data"
        );
        assert_eq!(now, content, "record segment left exactly verbatim");
    }
}
