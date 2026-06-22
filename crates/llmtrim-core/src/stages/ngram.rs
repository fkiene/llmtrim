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
//! Universality: candidates are word sequences over whitespace-delimited tokens, so this
//! covers any space-separated script (Latin, Cyrillic, Greek, Arabic, …) and gracefully
//! no-ops on scripts without inter-word spaces (CJK, Thai) — a word-level glossary
//! doesn't apply there, so that content is left verbatim rather than mis-abbreviated.
//!
//! Mining (see `crate::stages::ngram_sa`): the repeated phrases are the *maximal
//! repeats* of the word sequence, enumerated from a suffix array + LCP array in
//! O(n log n) ("Efficient Repeat Finding via Suffix Arrays", arXiv:1304.0528), then
//! chosen greedily by their real token gain with overlap accounting — the Re-Pair idea
//! of substituting the most profitable repeat (Larsson & Moffat; arXiv:1611.01479),
//! priced in target tokens rather than raw frequency. This captures whole repeated
//! spans a fixed n-gram window would fragment or miss.

use anyhow::Result;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;
use crate::tokenizer::counter_for;

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
        let segs: Vec<(String, String)> = crate::cache_zone::compressible_pointers(req, provider)
            .into_iter()
            .filter_map(|p| req.get_str(&p).map(|s| (p, s.to_string())))
            .collect();
        if segs.is_empty() {
            return Ok(());
        }
        // Pick a placeholder marker not already present in the content (so the legend
        // stays lossless). German / legal text routinely contains `§`, so don't hard-bail
        // on it — fall through to the next candidate. Only abort if all collide.
        let Some(marker) = pick_marker(&segs) else {
            return Ok(()); // every candidate marker occurs in the text → stay lossless
        };

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

        // Price the dictionary in REAL target tokens: build the same counter the gate
        // uses (provider + model), so the stage's gain estimate and the pipeline's
        // accept/revert decision agree. Unknown/missing model falls back inside
        // `counter_for`, so this never errors out the stage.
        let model = req.raw().get("model").and_then(Value::as_str);
        let counter = counter_for(req.kind(), model)?;

        // Mine maximal repeats and select the positive-token-gain phrases (suffix array
        // + LCP, greedy by gain with overlap accounting). Phrases come back in commit
        // order — longest/most-profitable first.
        let work_refs: Vec<&str> = working.iter().map(String::as_str).collect();
        let selections =
            crate::stages::ngram_sa::mine(&work_refs, self.max_entries, counter.as_ref(), |k| {
                format!("{marker}{k}")
            });
        if selections.is_empty() {
            return Ok(());
        }

        let mut committed: Vec<(String, String)> = Vec::new();
        for sel in selections {
            let ph = format!("{marker}{}", committed.len() + 1);
            // Whole-word replace keeps the legend lossless ("the report" must not match
            // inside "the reporter"); the mined phrases are word-aligned by construction.
            for t in working.iter_mut() {
                *t = replace_word_bounded(t, &sel.phrase, &ph);
            }
            committed.push((ph, sel.phrase));
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

/// The candidate placeholder markers, in preference order. A marker is chosen only if it
/// is absent from every segment, so the legend can losslessly recover the text. `§` is
/// first (compact, one BPE token) but common in German/legal prose, hence the fallbacks.
const MARKERS: &[&str] = &["§", "⟦", "@@", "‡"];

/// The first [`MARKERS`] entry that occurs in none of the segments (so it can't collide
/// with real content), or `None` when they all appear somewhere.
fn pick_marker(segs: &[(String, String)]) -> Option<&'static str> {
    MARKERS
        .iter()
        .copied()
        .find(|&m| !segs.iter().any(|(_, t)| t.contains(m)))
}

/// True when the byte offset `at` in `t` is a word boundary edge: the adjacent char on the
/// given `side` is absent (string edge) or non-alphanumeric (Unicode-aware). Prevents a
/// phrase like "the report" from matching inside "the reporter".
fn boundary_ok(t: &str, at: usize, before: bool) -> bool {
    let adj = if before {
        t[..at].chars().next_back()
    } else {
        t[at..].chars().next()
    };
    adj.is_none_or(|c| !c.is_alphanumeric())
}

/// Count whole-word occurrences of `phrase` in `t`: substring matches whose surrounding
/// chars are both word boundaries. Non-overlapping, scanning left to right. Production
/// counting moved into the suffix-array miner; kept as the word-boundary test oracle.
#[cfg(test)]
fn count_word_bounded(t: &str, phrase: &str) -> usize {
    let mut n = 0;
    let mut start = 0;
    while let Some(rel) = t[start..].find(phrase) {
        let s = start + rel;
        let e = s + phrase.len();
        if boundary_ok(t, s, true) && boundary_ok(t, e, false) {
            n += 1;
        }
        start = e; // non-overlapping; phrase is never empty (≥ 8 chars)
    }
    n
}

/// Replace whole-word occurrences of `phrase` (both edges word boundaries) in `t` with
/// `ph`, leaving partial-word hits ("the reporter") untouched. Lossless w.r.t. the legend.
fn replace_word_bounded(t: &str, phrase: &str, ph: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let mut pos = 0;
    while let Some(rel) = t[pos..].find(phrase) {
        let s = pos + rel;
        let e = s + phrase.len();
        if boundary_ok(t, s, true) && boundary_ok(t, e, false) {
            out.push_str(&t[pos..s]);
            out.push_str(ph);
            pos = e;
        } else {
            // Not a whole-word hit: keep up to and including this char, resume after it.
            let skip = t[s..].chars().next().map_or(e, |c| s + c.len_utf8());
            out.push_str(&t[pos..skip]);
            pos = skip;
        }
    }
    out.push_str(&t[pos..]);
    out
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
        // The miner is gain-gated (positive real-token gain only), so the phrase has to
        // recur enough to outweigh its one legend entry — five hits clears the bar.
        let phrase = "the quarterly financial report";
        let text = format!(
            "{phrase} grew. later {phrase} fell. again {phrase} held. then {phrase} rose. finally {phrase} dipped."
        );
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let sel =
            crate::stages::ngram_sa::mine(&[&text], 32, counter.as_ref(), |k| format!("§{k}"));
        assert!(
            sel.iter().any(|s| s.phrase == phrase),
            "frequent phrase is mined and selected"
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
    fn replace_is_word_bounded_not_substring() {
        // "the report" repeats as a whole phrase, but also appears inside "the reporter".
        // Word-bounded replace must abbreviate the whole-phrase hits and leave "reporter"
        // intact (the old substring replace produced "§1er").
        let t = "the report says X. the report says Y. but the reporter disagreed.";
        let occ = count_word_bounded(t, "the report");
        assert_eq!(
            occ, 2,
            "only the two whole-word hits count, not 'the reporter'"
        );
        let out = replace_word_bounded(t, "the report", "§1");
        assert!(out.contains("the reporter"), "partial-word hit untouched");
        assert!(!out.contains("§1er"), "no corrupted partial replacement");
        assert_eq!(out.matches("§1").count(), 2, "both whole phrases replaced");
    }

    #[test]
    fn picks_fallback_marker_when_section_sign_present() {
        // German/legal text already contains `§` — the stage must use the next free
        // marker instead of bailing, and still abbreviate the repeated phrase.
        let p = "die zuständige aufsichtsbehörde des landes";
        let content = format!(
            "Nach §1 gilt: {p} prüft. Ferner {p} entscheidet. Schließlich {p} bestätigt. Zudem {p} meldet."
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(NgramStage { max_entries: 32 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(
            out.stages[0].applied,
            "abbreviates despite an existing § in the text"
        );
        let user = req.get_str("/messages/1/content").unwrap();
        assert!(user.contains('⟦'), "fallback marker used, not §");
        assert!(
            user.contains("§1"),
            "the original §1 reference is preserved verbatim"
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

    /// Real target-token count of one string (the measure the gate uses).
    fn toks(counter: &dyn crate::tokenizer::TokenCounter, s: &str) -> usize {
        counter.count(s)
    }

    /// The OLD fixed-window miner (word n-grams, n = 2..=6, frequency ≥ 2, ≥ 8 chars,
    /// longest-first greedy) — reproduced here only to prove the suffix-array miner beats
    /// it on a phrase longer than the fixed window.
    fn old_fixed_window_pick(text: &str) -> Option<String> {
        use std::collections::HashMap;
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        for n in 2..=6 {
            if words.len() < n {
                break;
            }
            for w in words.windows(n) {
                *counts.entry(w.join(" ")).or_insert(0) += 1;
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
        cands.into_iter().map(|(p, _)| p).next()
    }

    #[test]
    fn captures_phrase_longer_than_the_fixed_window() {
        // A 9-word boilerplate clause, repeated. The old miner caps n at 6, so it can
        // only ever pick a 6-word fragment; the suffix-array miner takes the whole 9-word
        // span — fewer placeholders, one legend entry, strictly more tokens saved.
        let clause = "the parties hereby agree to indemnify and hold harmless";
        assert_eq!(clause.split_whitespace().count(), 9, "fixture is 9 words");
        // Vary the word before AND after each occurrence so the maximal repeat is exactly
        // the 9-word clause (no shared neighbor silently extends it to 10).
        let text = format!(
            "firstly {clause} promptly. moreover {clause} fully. \
             additionally {clause} jointly. lastly {clause} severally."
        );
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();

        // New miner: the full 9-word clause is the top selection.
        let sel =
            crate::stages::ngram_sa::mine(&[&text], 32, counter.as_ref(), |k| format!("§{k}"));
        let top = sel.first().expect("a phrase is selected");
        assert_eq!(
            top.phrase, clause,
            "suffix-array miner captures the whole 9-word repeat, not a fragment"
        );

        // Old miner can only reach a ≤6-word fragment of the clause.
        let old = old_fixed_window_pick(&text).expect("fixed-window picks something");
        assert!(
            old.split_whitespace().count() <= 6,
            "fixed window is capped at 6 words (got {})",
            old.split_whitespace().count()
        );
        assert!(
            top.phrase.split_whitespace().count() > old.split_whitespace().count(),
            "new phrase is longer than the fixed-window best"
        );

        // Savings comparison: apply each pick once and compare residual tokens. The new
        // longer phrase replaces the same 4 occurrences with one placeholder + one legend
        // entry, beating the fragment which leaves the rest of the clause uncompressed.
        let occ = 4;
        let new_after = occ * toks(counter.as_ref(), "§1")
            + toks(counter.as_ref(), &format!("§1={}; ", top.phrase));
        let new_before = occ * toks(counter.as_ref(), top.phrase.as_str());
        let old_after =
            occ * toks(counter.as_ref(), "§1") + toks(counter.as_ref(), &format!("§1={old}; "));
        let old_before = occ * toks(counter.as_ref(), &old);
        let new_saved = new_before as i64 - new_after as i64;
        let old_saved = old_before as i64 - old_after as i64;
        assert!(
            new_saved > old_saved,
            "suffix-array savings ({new_saved}) exceed fixed-window savings ({old_saved})"
        );
    }

    #[test]
    fn overlap_no_double_claimed_spans_and_reconstructs() {
        // Genuinely overlapping candidates: the 4-word "alpha bravo charlie delta" recurs,
        // and (separately) the 4-word "charlie delta echo foxtrot" recurs, sharing the
        // "charlie delta" sub-span — but the 6-word concatenation only co-occurs sometimes,
        // so neither phrase subsumes the other. Overlap accounting must ensure a word
        // claimed by the first pick isn't recounted for the second, and the legend must
        // reverse to the exact original text.
        let left = "alpha bravo charlie delta";
        let right = "charlie delta echo foxtrot";
        let text = format!(
            "{left} one. {left} two. {left} three. {left} four. \
             {right} five. {right} six. {right} seven. {right} eight."
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":text}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(NgramStage { max_entries: 32 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "overlapping repeats still compress");

        let user = req.get_str("/messages/1/content").unwrap().to_string();
        let sys = req.get_str("/messages/0/content").unwrap().to_string();

        // Parse the legend back out: `§k=phrase` pairs joined by `; `.
        let legend = sys
            .split_once(':')
            .map(|(_, rest)| rest.trim())
            .unwrap_or(&sys);
        let mut pairs: Vec<(String, String)> = legend
            .split("; ")
            .filter_map(|e| {
                e.split_once('=')
                    .map(|(k, v)| (k.trim().to_string(), v.to_string()))
            })
            .collect();
        // Reverse-substitute longest placeholder first so §10 isn't shadowed by §1.
        pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(b.0.cmp(&a.0)));
        let mut restored = user.clone();
        for (ph, phrase) in &pairs {
            restored = restored.replace(ph.as_str(), phrase);
        }
        assert_eq!(
            restored, text,
            "legend reverses placeholders to the exact original"
        );

        // Every placeholder defined in the legend is actually used in the body (no orphan
        // entry from a span that was claimed away), and vice versa.
        for (ph, _) in &pairs {
            assert!(
                user.contains(ph.as_str()),
                "every legend entry is referenced"
            );
        }
        // No placeholder index appears in the body without a matching legend definition.
        for tok in user.split(|c: char| !c.is_alphanumeric() && c != '§') {
            if let Some(rest) = tok.strip_prefix('§')
                && rest.chars().all(|c| c.is_ascii_digit())
                && !rest.is_empty()
            {
                assert!(
                    pairs.iter().any(|(ph, _)| ph == tok),
                    "placeholder {tok} in body must be defined in the legend"
                );
            }
        }
    }

    #[test]
    fn selection_is_deterministic_across_runs() {
        let p = "the internal configuration service endpoint";
        let content =
            format!("{p} failed. retry {p}. then {p} again. {p} more. {p} keeps. {p} still.");
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let run_once = || {
            crate::stages::ngram_sa::mine(&[&content], 32, counter.as_ref(), |k| format!("§{k}"))
                .into_iter()
                .map(|s| s.phrase)
                .collect::<Vec<_>>()
        };
        let a = run_once();
        let b = run_once();
        let c = run_once();
        assert!(!a.is_empty(), "something is selected");
        assert_eq!(a, b, "mining is deterministic");
        assert_eq!(b, c, "mining is deterministic");
    }

    #[test]
    fn unicode_cjk_and_accented_repeats() {
        // CJK is space-separated here (word-segmented upstream); the accented Latin phrase
        // carries combining marks. Both must be mined whole and round-trip losslessly.
        let cjk = "请 立即 提交 季度 财务 报告"; // "submit the quarterly financial report now"
        let accented = "déclaration trimestrielle of résultats financiers";
        let text = format!(
            "{cjk} 一. {cjk} 二. {cjk} 三. {cjk} 四. \
             {accented} A. {accented} B. {accented} C. {accented} D."
        );
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":text}]});
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(NgramStage { max_entries: 32 })];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        assert!(out.stages[0].applied, "unicode repeats compress");

        let sys = req.get_str("/messages/0/content").unwrap();
        assert!(
            sys.contains(cjk) || sys.contains(accented),
            "a unicode phrase is defined in the legend verbatim"
        );

        // Round-trip: reversing the legend rebuilds the exact original (no mangled bytes
        // mid-grapheme — the word-bounded replace only cuts on non-alphanumeric edges).
        let user = req.get_str("/messages/1/content").unwrap().to_string();
        let legend = sys.split_once(':').map(|(_, r)| r.trim()).unwrap_or(sys);
        let mut pairs: Vec<(String, String)> = legend
            .split("; ")
            .filter_map(|e| {
                e.split_once('=')
                    .map(|(k, v)| (k.trim().to_string(), v.to_string()))
            })
            .collect();
        pairs.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(b.0.cmp(&a.0)));
        let mut restored = user.clone();
        for (ph, phrase) in &pairs {
            restored = restored.replace(ph.as_str(), phrase);
        }
        assert_eq!(
            restored, text,
            "unicode legend reverses to the exact original"
        );
    }
}
