//! Pipeline configuration + embedded assets.
//!
//! Config resolves from `LLMTRIM_PRESET` / a `preset = "<name>"` key (a named profile) or the
//! per-stage flags in a TOML file (`LLMTRIM_CONFIG` or the platform config dir); with neither,
//! the default is `auto` (shape-routing). See [`DenseConfig::load`].

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// The Stage D format legend, embedded at build time and injected into the prompt
/// so the model can read the columnar encoding (spec §4: always include a legend).
/// Validated non-empty by `build.rs`.
pub const FORMAT_LEGEND: &str = include_str!("../prompts/toon_legend.txt");

/// Per-stage enable flags and knobs. `DenseConfig::default()` (= the `safe` preset) is the
/// **lossless** baseline: only quality-neutral lossless input compression runs (`hygiene`,
/// `serialize`, exact-duplicate `dedup`). The **shipped default is `auto`** (shape-routing),
/// which also turns on the lossy stages the eval shows quality-safe — output control on every
/// shape, image downscale (quality-neutral by construction), and retrieve / skeleton / dedup /
/// tools per shape. The bar is cost-at-no-measured-quality-loss, **not** losslessness; a stage
/// is gated to opt-in only when it is *unmeasured* or *shown to regress*, not for being lossy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct DenseConfig {
    /// Stage D — lossless data hygiene (minify, numeric trim, base64/data-URI strip).
    pub hygiene: bool,
    /// Stage D — columnar (TOON) serialization of uniform record arrays.
    pub serialize: bool,
    /// Stage F — request-shaping output controls (terse instruction, max_tokens).
    /// Opt-in: it changes the model's output behavior (visible to the caller), and
    /// its output-side savings aren't measured live by the offline gate.
    pub output_control: bool,
    /// Minimum rows a uniform array must have before columnar encoding is attempted.
    pub serialize_min_rows: usize,
    /// Stage F — optional hard output-token cap, imposed only when the request has none.
    pub output_max_tokens: Option<u64>,
    /// Stage F — output-control tier: `"terse"` (clean) or `"draft"` (Chain-of-Draft
    /// reasoning).
    pub output_level: String,
    /// Stage F — soft output-token budget injected into the prompt ("answer within
    /// N tokens"); complements the hard `output_max_tokens` cap.
    pub output_token_budget: Option<u64>,
    /// Stage F — instruct the model to emit minified code (arXiv:2508.13666; model-gated).
    pub output_compact_code: bool,
    /// Stage D — also encode uniform arrays nested inside content JSON, not only
    /// when the whole content is an array.
    pub serialize_nested: bool,
    /// Stage D — encode a top-level uniform flat array as CSV instead of TOON (opt-in).
    pub serialize_csv: bool,
    /// Stage D — flatten nested-uniform records to dotted columns (`meta.region`) before
    /// columnar encoding. Information-preserving, structurally reshaped; opt-in.
    pub serialize_flatten: bool,
    /// Stage D — partition a heterogeneous record array into uniform groups by shape,
    /// each emitted as its own TOON table. Regroups rows; opt-in.
    pub serialize_buckets: bool,
    /// Stage — lossy down-sampling of record arrays longer than `json_crush_max_rows`:
    /// keep first/last + outliers (errors / rare values) + a query-biased sample.
    pub json_crush: bool,
    /// Row cap a record array is sampled down to when `json_crush` is on.
    pub json_crush_max_rows: usize,
    /// Stage D — strip embedded base64 blobs / `data:` URIs (≥200-char runs → a
    /// `[base64 elided: N]` marker). Lossy, but measured quality-neutral (+0.0pp on
    /// `bench/data/base64.jsonl`), so on in the `auto` presets; `safe` keeps blobs.
    pub strip_base64: bool,
    /// Stage D — opt-in lossy: round float numbers to this many significant figures.
    pub numeric_sig_figs: Option<u32>,
    /// Stage D — tokenizer-aware Unicode normalization: drop invisible/format waste,
    /// fold no-break spaces, NFKC-canonicalize. Meaning-preserving but not byte-
    /// reversible, so opt-in. Universal (biggest wins on non-ASCII / pasted text).
    pub normalize_unicode: bool,
    /// Stage B — lexical retrieval (BM25/TextRank top-k chunk selection). Lossy;
    /// off by default until the live quality gate exists.
    pub retrieve: bool,
    /// Stage B — fraction of chunks to keep when retrieving (0.0–1.0).
    pub retrieve_keep_ratio: f64,
    /// Stage B — only content segments at least this many chars are eligible for
    /// pruning; shorter segments are treated as the query.
    pub retrieve_min_segment_chars: usize,
    /// Stage B — reorder kept chunks into a head+tail U-shape by relevance, to
    /// counter the lost-in-the-middle effect (Liu et al. 2307.03172). Lossless
    /// (reorders, drops nothing extra); replaces positional elision markers with a
    /// single summary note.
    pub retrieve_reorder: bool,
    /// Stage B — MMR diversity-aware selection: when picking the top-k chunks,
    /// penalize ones redundant with already-kept chunks (Carbonell & Goldstein 1998).
    pub retrieve_mmr: bool,
    /// Stage B — MMR tradeoff: 1.0 = pure relevance, 0.0 = pure diversity.
    pub retrieve_mmr_lambda: f64,
    /// Stage B — chunk at sentence granularity (DSLR, arXiv:2407.03627) for finer pruning.
    pub retrieve_sentence: bool,
    /// Stage A — provider prefix caching (cache_control breakpoints). Lossless; off
    /// by default (cache writes cost more, so it only pays off on repeated prefixes).
    pub cache: bool,
    /// Stage A — maximum cache breakpoints to place (Anthropic allows up to 4).
    pub cache_max_breakpoints: usize,
    /// Stage E — collapse exact-duplicate lines in content (with `[×N]` counts).
    pub dedup: bool,
    /// Stage E — also collapse near-duplicate lines (SimHash).
    pub dedup_near: bool,
    /// Stage E — max SimHash Hamming distance treated as a near-duplicate.
    pub dedup_near_max_distance: u32,
    /// Stage E+ — reversible n-gram abbreviation dictionary (lossless input).
    pub ngram: bool,
    /// Stage E+ — maximum abbreviation-dictionary entries to introduce.
    pub ngram_max_entries: usize,
    /// Stage G — static tool selection: keep only tools relevant to the request.
    pub tool_select: bool,
    /// Stage G — truncate verbose tool descriptions.
    pub tool_trim_desc: bool,
    /// Stage G — max characters for a tool description when trimming.
    pub tool_max_desc_chars: usize,
    /// Stage T — tool-output compression: window logs / diffs / grep output coming back
    /// from tools (the agent read path). Lossy; off by default.
    pub toolout: bool,
    /// Stage T — upper bound on lines kept per tool-output segment (adaptive-budget cap).
    pub toolout_max_lines: usize,
    /// Stage T — skip tool-output segments shorter than this many lines.
    pub toolout_min_lines: usize,
    /// Stage T — fold parametric log-line runs with a lossless Drain template pass first.
    pub toolout_template: bool,
    /// Stage T — adaptive/aggressive split: `"adaptive"` (always window to the budget),
    /// `"aggressive"` (always signal-only: errors / changed lines / one match per file +
    /// a summary), or `"auto"` (decide per segment by noise density — the tuned default).
    /// Dropped lines are elided by position (`[… N lines omitted …]`); the agent re-runs
    /// the tool if it needs them.
    pub toolout_mode: String,
    /// Stage C — skeletonize fenced code blocks (drop function bodies to stubs).
    /// Lossy; off by default.
    pub skeletonize: bool,
    /// Stage C — minify fenced brace-language code: strip indentation + blank lines,
    /// protecting string literals (arXiv:2508.13666). Semantically lossless; opt-in.
    pub minify_code: bool,
    /// Stage H — multimodal: lower image detail tier + downscale embedded images.
    /// Lossy; off by default.
    pub multimodal: bool,
    /// Stage H — optionally force the OpenAI image detail tier (e.g. `"low"`).
    /// `None` leaves the caller's choice; downscaling to the provider cap still runs.
    pub image_detail: Option<String>,
    /// Meta: when true, ignore the flags above and route to the shape-matched preset
    /// per request (`route`). Set by `auto()` / `preset("auto")`; the runtime default
    /// when no config file is present. `false` keeps the explicit flags (incl. defaults).
    pub auto: bool,
}

impl Default for DenseConfig {
    fn default() -> Self {
        Self {
            hygiene: true,
            serialize: true,
            output_control: false,
            serialize_min_rows: 2,
            output_max_tokens: None,
            output_level: "terse".to_string(),
            output_token_budget: None,
            output_compact_code: false,
            serialize_nested: true,
            serialize_csv: false,
            serialize_flatten: false,
            serialize_buckets: false,
            json_crush: false,
            json_crush_max_rows: 50,
            strip_base64: false,
            numeric_sig_figs: None,
            normalize_unicode: false,
            retrieve: false,
            retrieve_keep_ratio: 0.5,
            retrieve_min_segment_chars: 600,
            retrieve_reorder: false,
            retrieve_mmr: false,
            retrieve_mmr_lambda: 0.5,
            retrieve_sentence: false,
            cache: false,
            cache_max_breakpoints: 4,
            dedup: true,
            dedup_near: false,
            dedup_near_max_distance: 3,
            ngram: false,
            ngram_max_entries: 32,
            tool_select: false,
            tool_trim_desc: false,
            tool_max_desc_chars: 300,
            toolout: false,
            toolout_max_lines: 40,
            toolout_min_lines: 20,
            toolout_template: true,
            toolout_mode: "auto".to_string(),
            skeletonize: false,
            minify_code: false,
            multimodal: false,
            image_detail: None,
            auto: false,
        }
    }
}

impl DenseConfig {
    /// Load config. Resolution order:
    /// 1. `LLMTRIM_PRESET=<name>` env → that named profile.
    /// 2. A config file (`LLMTRIM_CONFIG` or the platform config dir) with a `preset = "<name>"`
    ///    key → that profile; or otherwise → the explicit per-stage flags in the file.
    /// 3. No env, no file → `auto` (shape-routing), the recommended default.
    ///
    /// Preset names: `auto` · `safe` · `rag` · `agent` · `code` · `aggressive` · `cache` ·
    /// `reasoning`. A `preset` key and raw flags are alternatives — `preset` wins (one knob
    /// instead of ~30); drop the `preset` key to hand-tune flags.
    pub fn load() -> Result<Self> {
        if let Some(name) = std::env::var("LLMTRIM_PRESET")
            .ok()
            .filter(|s| !s.is_empty())
        {
            return Self::preset(&name).with_context(|| {
                format!("unknown LLMTRIM_PRESET '{name}' (auto|safe|rag|agent|code|aggressive|cache|reasoning)")
            });
        }
        let Some(path) = config_path().filter(|p| p.exists()) else {
            return Ok(Self::auto());
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let value: toml::Value =
            toml::from_str(&text).with_context(|| format!("failed to parse {}", path.display()))?;
        Self::from_toml_value(value).with_context(|| format!("invalid config {}", path.display()))
    }

    /// Resolve a parsed config: a `preset = "<name>"` key selects a named profile; otherwise
    /// the value is the explicit per-stage flags. Factored out so it is unit-testable.
    fn from_toml_value(value: toml::Value) -> Result<Self> {
        if let Some(name) = value.get("preset").and_then(toml::Value::as_str) {
            return Self::preset(name).with_context(|| format!("unknown preset '{name}'"));
        }
        // A file with no *compression* keys (empty, or only orthogonal keys like
        // `retention_days`) keeps the shipped `auto` shape-routing default. Otherwise adding
        // e.g. `retention_days = 30` would silently fall through to the bare flag set
        // (`auto = false`, everything-but-lossless off) and disable shape-routing — a
        // surprising downgrade for a key that has nothing to do with compression.
        if let Some(table) = value.as_table()
            && !table.keys().any(|k| k != "retention_days" && k != "preset")
        {
            return Ok(Self::auto());
        }
        value.try_into().context("config does not match the schema")
    }

    /// Config for the live interceptor: the same resolution as [`load`](Self::load), with no
    /// env/file falling back to `auto` (shape-routing). Safe for real clients — the breakers
    /// are in place: the `cache` stage skips client-managed `cache_control`, `retrieve`
    /// protects directive blocks, and `tool_select` never drops an already-invoked tool. A
    /// broken config is surfaced (not silently ignored) before falling back.
    pub fn load_for_interceptor() -> Self {
        Self::load().unwrap_or_else(|e| {
            eprintln!("llmtrim: {e}; using shape-routing defaults");
            Self::auto()
        })
    }

    /// The shape-routing config: at compress time, `route(request)` picks the preset
    /// (tools → agent, code → code, long-context+question → rag, else → aggressive).
    /// The recommended default — captures the per-shape wins without misfiring (RAG
    /// goes to `rag`, not blanket-aggressive). Still zero-model (structural detection).
    pub fn auto() -> Self {
        Self {
            auto: true,
            ..Self::default()
        }
    }

    /// A named bundle of stage flags layered over the defaults, so callers opt into
    /// a workload profile without setting ~20 flags. `None` for an unknown name.
    pub fn preset(name: &str) -> Option<Self> {
        let mut c = Self::default();
        match name.to_ascii_lowercase().as_str() {
            // Defaults already = lossless input only (hygiene + serialize + exact dedup).
            "safe" | "lossless" => {}
            // Shape-routing meta-preset (resolved per request at compress time).
            "auto" => c.auto = true,
            // RAG: training-free DSLR sentence pruning with a tight cap (0.35).
            // Bench-confirmed (hotpotqa n=20) to BEAT chunk-level on BOTH axes — cuts
            // more input (50% vs 43%) at less quality loss (−2.0pp vs −7.6pp) — because
            // it keeps the answer sentence inside an otherwise-irrelevant paragraph,
            // which chunk-level drops whole. `retrieve_sentence` reassembles in original
            // order, so no reorder.
            "rag" => {
                c.retrieve = true;
                c.retrieve_sentence = true;
                c.retrieve_keep_ratio = 0.35;
                // Long context can embed logs/diffs/grep dumps or huge JSON tables —
                // compress those too (shape-gated; prose is left to retrieve above).
                c.toolout = true;
                c.json_crush = true;
                // Output control on by default: terse output holds quality (often improves
                // it) and output tokens cost 3–5× input — the metric is cost-at-no-quality-
                // loss, not losslessness. `safe` is the lossless-only preset.
                c.output_control = true;
                // Image downscale to the provider's resolution cap — quality-neutral by
                // construction (the provider resizes to the same cap regardless), so the
                // model sees identical pixels for fewer upload bytes + image tokens.
                c.multimodal = true;
                // Elide base64 / data-URI blobs (≥200-char runs → `[base64 elided: N]`
                // marker) — measured quality-neutral (+0.0pp on bench/data/base64.jsonl):
                // such blobs are noise the model can't use. Lossy, so `safe` keeps them.
                c.strip_base64 = true;
            }
            "agent" => {
                c.tool_select = true;
                c.tool_trim_desc = true;
                c.cache = true;
                c.toolout = true; // window log/diff/grep tool results (the agent read path)
                c.serialize_flatten = true; // dot-flatten nested tool-result JSON
                c.serialize_buckets = true; // bucket heterogeneous record arrays
                c.json_crush = true; // sample huge record arrays to representatives
                // No terse output here: on tool-calling traffic it gave ~no cost benefit
                // (glaive cost 7%) and a quality dip (100→92 at n=12) — see bench/README.
                c.multimodal = true; // downscale images to the provider cap (see `rag` note)
                c.strip_base64 = true; // elide base64 blobs (measured +0.0pp, see `rag` note)
                // ngram dropped: ~10–106 tok on agent traffic (bench) for an injected
                // glossary that mutates the prompt — not worth it. Opt in explicitly.
            }
            "code" => {
                c.skeletonize = true;
                c.minify_code = true;
                // A coding turn often pastes a build log / diff / grep dump or a big JSON
                // config; these no-op on actual code (shape-gated), fire only when present.
                c.toolout = true;
                c.json_crush = true;
                c.output_control = true;
                c.multimodal = true; // downscale images to the provider cap (see `rag` note)
                c.strip_base64 = true; // elide base64 blobs (measured +0.0pp, see `rag` note)
                // `output_compact_code` (minified-output instruction) is NOT bundled:
                // the bench confirmed it costs pass@1 (humaneval −21.6pp, CI ±14.5 at
                // n=37). The −36% lever (arXiv:2508.13666) holds only via fine-tuning,
                // not a raw instruction to a small model. Opt in explicitly if wanted.
            }
            "aggressive" => {
                c.retrieve = true;
                // DSLR sentence pruning with a TIGHT cap (0.35): prunes harder than
                // chunk-level but protects the answer sentence + boundaries, so it cuts
                // more input at less quality cost than dropping whole paragraphs.
                c.retrieve_sentence = true;
                c.retrieve_keep_ratio = 0.35;
                c.skeletonize = true;
                c.minify_code = true;
                c.dedup_near = true;
                c.ngram = true;
                c.normalize_unicode = true;
                c.tool_select = true;
                c.tool_trim_desc = true;
                c.cache = true;
                c.toolout = true; // compress log/diff/grep tool results (auto split)
                c.serialize_flatten = true;
                c.serialize_buckets = true;
                c.json_crush = true;
                c.output_control = true;
                c.multimodal = true; // downscale images to the provider cap (see `rag` note)
                c.strip_base64 = true; // elide base64 blobs (measured +0.0pp, see `rag` note)
            }
            // Cache-first: lossless input only (no retrieve/reorder that *varies* the
            // prefix per request) + Stage A cache discipline, so a repeated long prefix
            // (agent / RAG-over-fixed-context) is served from the prompt cache. The bench
            // `cache` corpus shows ~92% input served from cache, the biggest cost lever
            // for fixed-context workloads — bigger than squeezing tokens.
            "cache" => {
                c.cache = true;
            }
            // Reasoning: Chain-of-Draft output (terse ≤5-word steps). Focuses the model
            // and cuts output tokens; measured +17pp accuracy on GSM8K vs verbose CoT
            // (compression *helping* quality, not just preserving it).
            "reasoning" => {
                c.output_control = true;
                c.output_level = "draft".to_string();
            }
            _ => return None,
        }
        Some(c)
    }
}

fn config_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LLMTRIM_CONFIG") {
        return Some(PathBuf::from(p));
    }
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .ok()
        .or_else(|| {
            std::env::var("HOME")
                .or_else(|_| std::env::var("USERPROFILE"))
                .ok()
                .map(|h| PathBuf::from(h).join(".config"))
        })?;
    Some(base.join("llmtrim").join("config.toml"))
}

/// Ledger age-retention in days, resolved **independently** of [`DenseConfig`] so selecting a
/// preset never resets it (presets rebuild `DenseConfig` from `default()`). Resolution:
/// `LLMTRIM_RETENTION_DAYS` (env) wins over a top-level `retention_days` key in the config
/// file. `None` when unset or ≤ 0 — age retention off, leaving only the row cap to bound the
/// ledger. The key is ignored by the compression config (no `deny_unknown_fields`).
pub fn retention_days() -> Option<i64> {
    if let Ok(v) = std::env::var("LLMTRIM_RETENTION_DAYS")
        && let Ok(days) = v.trim().parse::<i64>()
    {
        return (days > 0).then_some(days);
    }
    let path = config_path().filter(|p| p.exists())?;
    let text = std::fs::read_to_string(&path).ok()?;
    let value: toml::Value = toml::from_str(&text).ok()?;
    retention_days_from_toml(&value)
}

/// The `retention_days` key from a parsed config (positive only). Factored out so the
/// file-parsing is unit-testable without touching env or the real config path.
fn retention_days_from_toml(value: &toml::Value) -> Option<i64> {
    value
        .get("retention_days")
        .and_then(toml::Value::as_integer)
        .filter(|d| *d > 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_is_embedded_and_nonempty() {
        assert!(!FORMAT_LEGEND.trim().is_empty());
        assert!(FORMAT_LEGEND.contains("TOON"));
    }

    #[test]
    fn defaults_enable_mvp_stages() {
        let c = DenseConfig::default();
        assert!(
            c.hygiene && c.serialize,
            "lossless input compression on by default"
        );
        assert!(
            c.dedup && !c.dedup_near,
            "exact dedup on by default (lossless); near-dedup opt-in (lossy)"
        );
        assert!(
            !c.output_control,
            "default()/`safe` is the lossless baseline — output shaping is on in the shipped `auto` default (via presets), not in this bare base"
        );
        assert!(
            !c.retrieve,
            "retrieval is opt-in (workload-dependent per eval)"
        );
        assert!(c.serialize_nested, "nested array encoding on by default");
        assert!(!c.strip_base64, "base64 strip is opt-in (lossy)");
        assert_eq!(c.serialize_min_rows, 2);
    }

    #[test]
    fn auto_routed_presets_enable_output_control() {
        // The metric is cost-at-no-quality-loss, not losslessness: every shape `auto`
        // routes to (agent / code / rag / aggressive) enables output control by default.
        // Lossless-only lives in `safe`, the opt-in mode.
        for p in ["code", "rag", "aggressive", "reasoning"] {
            assert!(
                DenseConfig::preset(p).unwrap().output_control,
                "preset `{p}` enables output control by default"
            );
        }
        // `agent` (tool-calling): terse gives ~no cost benefit + a noisy quality dip, so off.
        assert!(
            !DenseConfig::preset("agent").unwrap().output_control,
            "agent leaves output unshaped — terse doesn't help tool-calls"
        );
        assert!(
            !DenseConfig::preset("safe").unwrap().output_control,
            "`safe` is the lossless mode — no output shaping"
        );
    }

    #[test]
    fn auto_routed_presets_downscale_images() {
        // Image downscale-to-cap is quality-neutral by construction (the provider resizes to
        // the same cap anyway) → on by default. `safe` leaves images byte-faithful, and the
        // genuinely-lossy `image_detail = "low"` stays opt-in.
        for p in ["agent", "code", "rag", "aggressive"] {
            assert!(
                DenseConfig::preset(p).unwrap().multimodal,
                "preset `{p}` downscales oversized images by default"
            );
        }
        assert!(!DenseConfig::preset("safe").unwrap().multimodal);
        assert!(
            DenseConfig::preset("aggressive")
                .unwrap()
                .image_detail
                .is_none()
        );
    }

    #[test]
    fn auto_routed_presets_strip_base64() {
        // Measured quality-neutral (+0.0pp on bench/data/base64.jsonl): blobs are noise the
        // model can't use, and elision leaves a marker. On by default; `safe` keeps blobs.
        for p in ["agent", "code", "rag", "aggressive"] {
            assert!(
                DenseConfig::preset(p).unwrap().strip_base64,
                "preset `{p}` elides base64 blobs by default"
            );
        }
        assert!(!DenseConfig::preset("safe").unwrap().strip_base64);
    }

    #[test]
    fn config_selects_preset_by_name_else_flags() {
        // `preset = "name"` (env or file) selects a named profile — one knob, not ~30 flags.
        let agg = DenseConfig::from_toml_value(toml::from_str("preset = \"aggressive\"").unwrap())
            .unwrap();
        assert!(
            agg.output_control && agg.retrieve,
            "preset key selects the profile"
        );
        // Unknown preset is a surfaced error, not a silent default.
        assert!(
            DenseConfig::from_toml_value(toml::from_str("preset = \"nope\"").unwrap()).is_err()
        );
        // No preset key → explicit per-stage flags are parsed.
        let flags =
            DenseConfig::from_toml_value(toml::from_str("hygiene = false").unwrap()).unwrap();
        assert!(
            !flags.hygiene,
            "explicit flags parse when no preset is named"
        );
    }

    #[test]
    fn presets_layer_over_defaults() {
        assert!(DenseConfig::preset("nope").is_none());
        let rag = DenseConfig::preset("rag").unwrap();
        assert!(rag.retrieve && rag.retrieve_sentence && rag.hygiene && rag.dedup);
        assert!(
            (rag.retrieve_keep_ratio - 0.35).abs() < 1e-9,
            "tight sentence cap"
        );
        let code = DenseConfig::preset("code").unwrap();
        assert!(code.minify_code && code.skeletonize && code.output_control);
        assert!(
            !code.output_compact_code,
            "compact-code output dropped — bench-confirmed pass@1 harm"
        );
        let agg = DenseConfig::preset("aggressive").unwrap();
        assert!(agg.retrieve && agg.skeletonize && agg.ngram && agg.minify_code);

        // Unknown preset name → None (no silent fallback).
        assert!(DenseConfig::preset("ultra").is_none());

        // Cache-first: cache on, but the prefix-varying retrieve/reorder stay OFF.
        let cache = DenseConfig::preset("cache").unwrap();
        assert!(cache.cache && !cache.retrieve && !cache.retrieve_reorder);
        assert!(cache.hygiene && cache.serialize, "still lossless input");

        // Reasoning: Chain-of-Draft output.
        let reasoning = DenseConfig::preset("reasoning").unwrap();
        assert!(reasoning.output_control && reasoning.output_level == "draft");
    }

    #[test]
    fn partial_toml_fills_remaining_from_default() {
        let c: DenseConfig = toml::from_str("serialize = false\n").unwrap();
        assert!(!c.serialize);
        assert!(c.hygiene, "unset fields take the default");
        assert_eq!(c.serialize_min_rows, 2);
    }

    #[test]
    fn retention_days_parses_positive_only() {
        let p: toml::Value = toml::from_str("retention_days = 30").unwrap();
        assert_eq!(retention_days_from_toml(&p), Some(30));
        let zero: toml::Value = toml::from_str("retention_days = 0").unwrap();
        assert_eq!(
            retention_days_from_toml(&zero),
            None,
            "0 disables age retention"
        );
        let neg: toml::Value = toml::from_str("retention_days = -5").unwrap();
        assert_eq!(retention_days_from_toml(&neg), None);
        let absent: toml::Value = toml::from_str("hygiene = true").unwrap();
        assert_eq!(retention_days_from_toml(&absent), None);
    }

    #[test]
    fn retention_key_does_not_disturb_compression_config() {
        // A config that only sets retention parses as the default compression config — the
        // key is orthogonal and must not be rejected or alter stage flags.
        let c =
            DenseConfig::from_toml_value(toml::from_str("retention_days = 30").unwrap()).unwrap();
        assert!(
            c.hygiene && c.serialize,
            "retention_days is ignored by DenseConfig"
        );
    }
}
