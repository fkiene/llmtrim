//! Token measurement — the ground-truth counter.
//!
//! Ground rule: every transform is measured with the *real target tokenizer*, never
//! by byte/char length ("a shorter string can tokenize to more tokens"). All stage
//! code counts tokens only through [`TokenCounter`], so the char≠token rule is
//! enforced structurally.
//!
//! - OpenAI → exact `tiktoken` BPE chosen by model (`o200k_base` for gpt-4o /
//!   o-series / gpt-5, `cl100k_base` for gpt-4 / 3.5), default `o200k_base`.
//! - Anthropic → no public exact tokenizer, so we use `o200k_base` as a BPE *proxy*
//!   and flag the counts as **approximate** (surfaced in `gain`; see plan risk #1).

use anyhow::Result;
use tiktoken_rs::CoreBPE;

use crate::ir::ProviderKind;

/// A token count.
///
/// A newtype over `usize` so a token count can't be silently confused with the many
/// other `usize` quantities the pipeline carries (char caps, row minimums, Hamming
/// distances, indices). Counts are produced by [`TokenCounter`] and stored on the
/// result types ([`crate::CompressResult`], [`crate::pipeline::PipelineOutcome`],
/// [`crate::pipeline::StageReport`]); transient local arithmetic stays plain `usize`,
/// and `.0` drops back to `usize`/`i64` at the SQLite ledger boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Tokens(pub usize);

impl std::fmt::Display for Tokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Counts tokens in text for a target model.
pub trait TokenCounter: Send + Sync {
    /// Number of tokens the target tokenizer produces for `text`.
    fn count(&self, text: &str) -> usize;

    /// `true` if counts are exact for the target model; `false` for an approximation.
    fn is_exact(&self) -> bool;

    /// Short label for diagnostics (e.g. `tiktoken`, `o200k-approx(anthropic)`).
    fn label(&self) -> &str;
}

/// Token counter backed by tiktoken (OpenAI BPE families; also the Anthropic proxy).
/// Holds a cached `&'static` singleton — vocabs load once, lazily.
pub struct TiktokenCounter {
    bpe: &'static CoreBPE,
    label: &'static str,
    exact: bool,
}

impl TokenCounter for TiktokenCounter {
    fn count(&self, text: &str) -> usize {
        // `encode_with_special_tokens` never errors and treats the whole string as
        // input; token count (not the ids) is all we need.
        self.bpe.encode_with_special_tokens(text).len()
    }

    fn is_exact(&self) -> bool {
        self.exact
    }

    fn label(&self) -> &str {
        self.label
    }
}

/// A cheap, BPE-*shaped* token estimate for providers with no public tokenizer — one linear
/// pass, no merge table (~100× faster than tiktoken). It captures the first-order behavior that
/// drives the savings %: alphanumeric "words" are token-dense (~1 token / 4 chars), while
/// punctuation and symbols are each roughly their own token. That's why it tracks
/// structure-stripping stages (JSON→TOON, minify) — which a flat byte/char ratio under-counts
/// ~2×, because it can't see that the punctuation a stage removes was token-dense. The *first*
/// whitespace char in a run is free (BPE folds a leading space into the next word), but a *run*
/// of whitespace — code indentation, blank lines — is token-dense and priced ~1 token / 4 extra
/// chars. Pricing runs (not zeroing all whitespace) is what lets a whitespace-only lossless
/// stage (`minify-code`) register a token win here instead of measuring `after == before` and
/// reverting on every Anthropic/Google request. Unicode letters join word runs (CJK is
/// over-merged, but these counts are flagged approximate and the savings % is what we report).
fn estimate_tokens(text: &str) -> usize {
    const CHARS_PER_WORD_TOKEN: usize = 4;
    // Calibrate the raw word+punct count to o200k's scale: the raw estimate runs ~1.39× o200k
    // on code and ~1× on prose, so 0.72 lands code (the dominant content for coding agents) on
    // o200k. The savings % and the per-stage gate are calibration-invariant (it's a flat factor
    // on both before and after) — this only fixes the absolute counts, so the priced $ stays
    // honest instead of ~40% inflated, and the dashboard numbers don't jump.
    const CALIB: f64 = 0.72;
    let mut raw = 0usize;
    let mut run = 0usize; // length of the current alphanumeric run
    let mut ws = 0usize; // length of the current whitespace run
    // A whitespace run's first char folds into the adjacent word (free); the rest is priced.
    let price_ws = |ws: usize| ws.saturating_sub(1).div_ceil(CHARS_PER_WORD_TOKEN);
    for c in text.chars() {
        if c.is_alphanumeric() {
            raw += price_ws(ws);
            ws = 0;
            run += 1;
        } else {
            if run > 0 {
                raw += run.div_ceil(CHARS_PER_WORD_TOKEN);
                run = 0;
            }
            if c.is_whitespace() {
                ws += 1;
            } else {
                raw += price_ws(ws);
                ws = 0;
                raw += 1; // punctuation / symbol ≈ its own token
            }
        }
    }
    raw += run.div_ceil(CHARS_PER_WORD_TOKEN);
    raw += price_ws(ws);
    (raw as f64 * CALIB).round() as usize
}

/// Token counter for providers without a public tokenizer (Anthropic, Google): the cheap
/// [`estimate_tokens`] heuristic, flagged approximate.
pub struct ApproxCounter {
    label: &'static str,
}

impl TokenCounter for ApproxCounter {
    fn count(&self, text: &str) -> usize {
        estimate_tokens(text)
    }

    fn is_exact(&self) -> bool {
        false
    }

    fn label(&self) -> &str {
        self.label
    }
}

/// Build the token counter for a provider and optional model name.
///
/// OpenAI uses tiktoken's own model→encoding registry (`bpe_for_model`) — so we
/// never hand-maintain a model list; unknown/newer models fall back to o200k_base.
/// Anthropic has no public tokenizer, so o200k_base is used as a *flagged* proxy.
/// Vocabs are cached `&'static` singletons (loaded once, lazily).
pub fn counter_for(provider: ProviderKind, model: Option<&str>) -> Result<Box<dyn TokenCounter>> {
    match provider {
        // OpenAI ships the real tokenizer — use it exactly (its own model→encoding registry;
        // unknown/newer models fall back to o200k_base).
        ProviderKind::OpenAi => {
            // tiktoken is exact only when it actually knows the model. OpenAI-*shaped* hosts
            // (groq/llama, deepseek, qwen, mistral…) miss the registry → o200k_base is then a
            // proxy, not ground truth, so flag it approximate instead of mislabeling the ledger.
            let (bpe, label, exact): (_, &'static str, bool) =
                match model.and_then(|m| tiktoken_rs::bpe_for_model(m).ok()) {
                    Some(bpe) => (bpe, "tiktoken", true),
                    None => (tiktoken_rs::o200k_base_singleton(), "o200k-approx", false),
                };
            Ok(Box::new(TiktokenCounter { bpe, label, exact }))
        }
        // No public tokenizer → the cheap BPE-shaped estimate (flagged approximate). Skips the
        // BPE pass that dominated compress latency while still tracking structural savings.
        ProviderKind::Anthropic => Ok(Box::new(ApproxCounter {
            label: "approx(anthropic)",
        })),
        ProviderKind::Google => Ok(Box::new(ApproxCounter {
            label: "approx(google)",
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_counter_is_exact_and_counts() {
        let c = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        assert!(c.is_exact());
        assert_eq!(c.count(""), 0);
        assert!(c.count("hello world") >= 2);
        // More text => at least as many tokens (monotonic on append).
        assert!(c.count("hello world, this is a longer sentence") > c.count("hello world"));
    }

    #[test]
    fn anthropic_counter_is_flagged_approximate() {
        let c = counter_for(ProviderKind::Anthropic, None).unwrap();
        assert!(!c.is_exact());
        assert!(c.label().contains("approx"));
        assert!(c.count("some tokens here") > 0);
    }

    #[test]
    fn unknown_openai_model_falls_back() {
        // An unrecognized model name must not error — it falls back to o200k_base.
        let c = counter_for(ProviderKind::OpenAi, Some("gpt-99-superfuture")).unwrap();
        assert!(c.count("x") >= 1);
    }
}
