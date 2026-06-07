//! Token measurement — the ground-truth counter.
//!
//! Spec §1/§6: every transform is measured with the *real target tokenizer*, never
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

/// Build the token counter for a provider and optional model name.
///
/// OpenAI uses tiktoken's own model→encoding registry (`bpe_for_model`) — so we
/// never hand-maintain a model list; unknown/newer models fall back to o200k_base.
/// Anthropic has no public tokenizer, so o200k_base is used as a *flagged* proxy.
/// Vocabs are cached `&'static` singletons (loaded once, lazily).
pub fn counter_for(provider: ProviderKind, model: Option<&str>) -> Result<Box<dyn TokenCounter>> {
    let (bpe, label, exact): (&'static CoreBPE, &'static str, bool) = match provider {
        ProviderKind::OpenAi => {
            let bpe = model
                .and_then(|m| tiktoken_rs::bpe_for_model(m).ok())
                .unwrap_or_else(tiktoken_rs::o200k_base_singleton);
            (bpe, "tiktoken", true)
        }
        ProviderKind::Anthropic => (
            tiktoken_rs::o200k_base_singleton(),
            "o200k-approx(anthropic)",
            false,
        ),
        // Gemini's tokenizer (SentencePiece) has no local Rust port; use o200k as a BPE
        // proxy and flag the counts approximate, same as Anthropic.
        ProviderKind::Google => (
            tiktoken_rs::o200k_base_singleton(),
            "o200k-approx(google)",
            false,
        ),
    };
    Ok(Box::new(TiktokenCounter { bpe, label, exact }))
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
