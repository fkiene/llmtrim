//! llmtrim — static, deterministic prompt/payload compression for LLM APIs.
//!
//! This crate is a zero-LLM-call middleware: it ingests a provider-shaped request
//! body, compresses it with deterministic algorithms only (no auxiliary model, no
//! embeddings), and can reverse the lossless transforms on the response. The
//! functions here are the **pure transform core** — no network calls live in this
//! crate. The `llmtrim` CLI/proxy crate wraps them.
//!

use anyhow::{Context, Result};
use serde_json::Value;

pub mod cache_zone;
pub mod config;
pub mod gate;
pub mod ir;
pub mod media;
pub mod memo;
pub mod pipeline;
pub mod provider;
pub mod quality_gate;
pub mod select;
pub mod stages;
pub mod tokenizer;

use gate::{PlanEntry, Transform};
use ir::{ProviderKind, Request};
use pipeline::StageReport;
use tokenizer::Tokens;

/// The outcome of compressing one request: the compressed body, the rehydration
/// plan, and the measured token deltas (for the ledger and reporting).
#[derive(Debug, Clone)]
pub struct CompressResult {
    pub request_json: String,
    pub plan: Vec<PlanEntry>,
    pub provider: ProviderKind,
    pub model: Option<String>,
    pub tokenizer_label: String,
    pub tokenizer_exact: bool,
    pub input_tokens_before: Tokens,
    pub input_tokens_after: Tokens,
    /// Tokens in the frozen (cache-controlled) prefix the stages skipped — see
    /// [`pipeline::PipelineOutcome::frozen_input_tokens`].
    pub frozen_input_tokens: Tokens,
    pub stages: Vec<StageReport>,
    /// Whether Stage F (output shaping) ran on this request — i.e. the *effective* config
    /// (after `auto` routing) enabled it. The ledger needs this to project the benchmark
    /// output reduction only onto traffic that actually carried the instruction.
    pub output_shaped: bool,
}

/// The ordered MVP stage list for a provider. Empty until Stage D/F land in later
/// build steps; the gated pipeline already runs over it so wiring stages in is a
/// one-line change here.
fn stages_for(_provider: ProviderKind, config: &config::DenseConfig) -> Vec<Box<dyn Transform>> {
    let mut stages: Vec<Box<dyn Transform>> = Vec::new();
    // Stage T (input, lossy): compress tool outputs (logs/diffs/grep) first, so the
    // structure-aware windowing runs before generic prose pruning sees a giant log.
    if config.toolout {
        stages.push(Box::new(stages::ToolOutputStage {
            max_lines: config.toolout_max_lines,
            min_lines: config.toolout_min_lines,
            template: config.toolout_template,
            mode: stages::toolout::ModeSetting::parse(&config.toolout_mode),
        }));
    }
    // Stage B (input-side, lossy): prune large context to the relevant chunks first.
    if config.retrieve {
        stages.push(Box::new(stages::RetrieveStage {
            keep_ratio: config.retrieve_keep_ratio,
            min_segment_chars: config.retrieve_min_segment_chars,
            reorder: config.retrieve_reorder,
            mmr: config.retrieve_mmr,
            mmr_lambda: config.retrieve_mmr_lambda,
            sentence: config.retrieve_sentence,
        }));
    }
    // Stage C (input, lossy): skeletonize non-focus code in fenced blocks.
    if config.skeletonize {
        stages.push(Box::new(stages::SkeletonStage {
            keep_full_top_k: config.skeleton_keep_full_top_k,
            drop_unmatched: config.skeleton_drop_unmatched,
            drop_min_body_lines: config.skeleton_drop_min_body_lines,
        }));
    }
    // Stage C (input, lossless): minify brace-language code (strip whitespace).
    if config.minify_code {
        stages.push(Box::new(stages::MinifyCodeStage));
    }
    // Stage H (input, lossy): image detail tier + downscale embedded images.
    if config.multimodal {
        stages.push(Box::new(stages::ImageStage {
            detail: config.image_detail.clone(),
        }));
    }
    // Stage D (input-side, lossless): clean, then columnar-encode uniform arrays.
    if config.hygiene {
        stages.push(Box::new(stages::HygieneStage {
            strip_base64: config.strip_base64,
            sig_figs: config.numeric_sig_figs,
            normalize_unicode: config.normalize_unicode,
        }));
    }
    // Lossy sample of huge record arrays (keeps anomalies) FIRST — drops rows while it's
    // still JSON; then the columnar encoder below packs the survivors. `safe` (json_crush
    // off) keeps every row and relies on serialize's lossless union CSV instead.
    if config.json_crush {
        stages.push(Box::new(stages::JsonCrushStage {
            max_rows: config.json_crush_max_rows,
        }));
    }
    // Columnar-encode record arrays (incl. near-uniform → union CSV). Lossless.
    if config.serialize {
        stages.push(Box::new(stages::SerializeStage {
            min_rows: config.serialize_min_rows,
            nested: config.serialize_nested,
            csv: config.serialize_csv,
            flatten: config.serialize_flatten,
            buckets: config.serialize_buckets,
        }));
    }
    // Stage E (input, lossy-ish): collapse duplicate lines.
    if config.dedup {
        stages.push(Box::new(stages::DedupStage {
            near: config.dedup_near,
            near_max_distance: config.dedup_near_max_distance,
        }));
    }
    // Stage E+ (input, lossless): abbreviate repeated multi-word phrases with a legend.
    if config.ngram {
        stages.push(Box::new(stages::NgramStage {
            max_entries: config.ngram_max_entries,
        }));
    }
    // Stage G (input, lossy): trim/select tool schemas + API-safe schema minification (resent
    // every call).
    if config.tool_select || config.tool_trim_desc || config.tool_minify_schema {
        stages.push(Box::new(stages::ToolStage {
            select: config.tool_select,
            trim_desc: config.tool_trim_desc,
            minify_schema: config.tool_minify_schema,
            max_desc_chars: config.tool_max_desc_chars,
        }));
    }
    // Stage F (output-side): request-shaping output controls (terse / Chain-of-Draft / budget).
    if config.output_control || config.output_compact_code {
        stages.push(Box::new(stages::OutputControlStage {
            level: stages::output::OutputLevel::parse(&config.output_level),
            max_tokens: config.output_max_tokens,
            token_budget: config.output_token_budget,
            compact_code: config.output_compact_code,
        }));
    }
    // Stage A (lossless, latent payoff): mark the final prefix for provider caching.
    // Last, so it fingerprints system+tools after the other stages have shaped them.
    if config.cache {
        stages.push(Box::new(stages::CacheStage {
            max_breakpoints: config.cache_max_breakpoints,
        }));
    }
    stages
}

/// Pick the workload preset for a request from its structure alone (no model): tool
/// calls → `agent`; fenced code → `code`; a long context segment alongside a question
/// (≥2 messages) → `rag` (sentence pruning, not blanket-`aggressive`, which misfires on
/// RAG); everything else → `aggressive`. Backs the `auto` default.
pub fn route(req: &Request, provider: &dyn provider::Provider) -> &'static str {
    let raw = req.raw();
    if raw
        .get("tools")
        .and_then(Value::as_array)
        .is_some_and(|a| !a.is_empty())
    {
        return "agent";
    }
    let texts: Vec<String> = provider
        .content_text_pointers(req)
        .iter()
        .filter_map(|p| req.get_str(p).map(str::to_string))
        .collect();
    if texts.iter().any(|t| t.contains("```")) {
        return "code";
    }
    // Turn count across every wire shape (Chat `messages`, Responses `input`, Gemini
    // `contents`) — not just `messages`, else Gemini/Responses RAG misroutes to aggressive.
    let turns = ["messages", "input", "contents"]
        .iter()
        .filter_map(|k| raw.get(*k).and_then(Value::as_array))
        .map(Vec::len)
        .max()
        .unwrap_or(0);
    if turns >= 2 && texts.iter().any(|t| t.chars().count() >= 1200) {
        return "rag";
    }
    "aggressive"
}

/// Compress a provider request body (JSON), loading per-stage config from the
/// environment/default path. `provider` may be `None` to auto-detect from shape.
pub fn compress(input: &str, provider: Option<ProviderKind>) -> Result<CompressResult> {
    let config = config::DenseConfig::load().unwrap_or_else(|e| {
        eprintln!("llmtrim: {e}; using defaults");
        config::DenseConfig::default()
    });
    compress_with_config(input, provider, &config)
}

/// Compress with an explicit [`DenseConfig`] (no environment access — the
/// deterministic core used by tests and embedders).
///
/// The request is parsed into the neutral [`Request`], measured with the real
/// target tokenizer, and run through the gated stage pipeline.
pub fn compress_with_config(
    input: &str,
    provider: Option<ProviderKind>,
    config: &config::DenseConfig,
) -> Result<CompressResult> {
    let value: Value = serde_json::from_str(input).context("request body is not valid JSON")?;
    let kind = match provider {
        Some(k) => k,
        None => provider::detect(&value).context(
            "could not auto-detect provider from request shape; pass --provider openai|anthropic",
        )?,
    };
    let model = value
        .get("model")
        .and_then(Value::as_str)
        .map(str::to_string);

    let counter = tokenizer::counter_for(kind, model.as_deref())?;
    let adapter = provider::for_kind(kind);
    let mut req = Request::from_value(kind, value);

    // `auto` resolves the preset from the request shape (structural, zero-model).
    let routed;
    let config = if config.auto {
        routed = config::DenseConfig::preset(route(&req, adapter.as_ref())).unwrap_or_default();
        &routed
    } else {
        config
    };

    let stages = stages_for(kind, config);
    let outcome = pipeline::run_gated(
        &mut req,
        adapter.as_ref(),
        counter.as_ref(),
        &stages,
        config.quality_gate,
    );

    Ok(CompressResult {
        request_json: req.to_json_string()?,
        plan: outcome.plan,
        provider: kind,
        model,
        tokenizer_label: counter.label().to_string(),
        tokenizer_exact: counter.is_exact(),
        input_tokens_before: outcome.input_tokens_before,
        input_tokens_after: outcome.input_tokens_after,
        frozen_input_tokens: outcome.frozen_input_tokens,
        stages: outcome.stages,
        output_shaped: config.output_control || config.output_compact_code,
    })
}

// ── Standalone text-blob compression (used by MCP tools and file reading) ──────────────

/// Options for [`compress_file`].
#[derive(Debug, Clone, Default)]
pub struct CompressFileOptions {
    /// Optional inclusive 1-based start line.
    pub start_line: Option<usize>,
    /// Optional inclusive 1-based end line.
    pub end_line: Option<usize>,
    /// Optional maximum bytes to read from the file (before compression).
    pub max_bytes: Option<usize>,
}

/// Result of compressing a standalone text blob (e.g., a file or raw text).
#[derive(Debug, Clone)]
pub struct CompressTextBlobResult {
    /// The compressed text.
    pub text: String,
    /// Tokens before compression.
    pub input_tokens_before: usize,
    /// Tokens after compression.
    pub input_tokens_after: usize,
    /// Human-readable tokenizer name used for the counts.
    pub tokenizer_label: String,
    /// Whether the tokenizer was exact (tiktoken) or approximate.
    pub tokenizer_exact: bool,
}

/// The model `compress_text_blob` wraps a blob under. Arbitrary (the request is
/// synthetic and never sent); it only selects the tokenizer for the reported counts.
const TEXT_WRAP_MODEL: &str = "gpt-4o";

// ── Folder compression ───────────────────────────────────────────────────────────────

/// Options for [`compress_folder`].
#[derive(Debug, Clone, Default)]
pub struct CompressFolderOptions {
    /// File extensions to include (without the leading dot), e.g. `["rs", "ts"]`.
    pub extensions: Vec<String>,
    /// Maximum number of files to include.
    pub max_files: usize,
    /// Maximum total input tokens (before compression) across all included files.
    /// This is a safety/performance limit on how much raw text llmtrim may read.
    pub max_total_input_tokens: usize,
    /// Maximum total output tokens (after compression) that may be returned to the model.
    /// This is the real context budget.
    pub max_total_output_tokens: usize,
    /// Patterns that exclude a file or directory. Supports `*` wildcards.
    pub exclude_patterns: Vec<String>,
    /// Optional glob patterns that further restrict which files are included.
    pub include_globs: Vec<String>,
    /// Whether to walk subdirectories recursively.
    pub recursive: bool,
}

/// Per-file result inside a folder compression.
#[derive(Debug, Clone)]
pub struct FileCompressionResult {
    /// The file path (relative or absolute, as given).
    pub path: String,
    /// The compressed text.
    pub text: String,
    /// Tokens before compression.
    pub input_tokens_before: usize,
    /// Tokens after compression.
    pub input_tokens_after: usize,
}

/// A file that was skipped during folder compression, with a human-readable reason.
#[derive(Debug, Clone)]
pub struct SkippedFile {
    pub path: String,
    pub reason: String,
}

/// Result of compressing a folder.
#[derive(Debug, Clone)]
pub struct CompressFolderResult {
    /// The folder path that was compressed.
    pub folder_path: String,
    /// Files that were successfully compressed.
    pub files: Vec<FileCompressionResult>,
    /// Files that were skipped (excluded, over limit, secret, binary, etc.).
    pub skipped: Vec<SkippedFile>,
    /// Sum of `input_tokens_before` across all included files.
    pub total_input_tokens_before: usize,
    /// Sum of `input_tokens_after` across all included files.
    pub total_input_tokens_after: usize,
    /// Total tokens saved (before - after, signed).
    pub total_tokens_saved: i64,
    /// The `max_total_input_tokens` budget that was applied.
    pub max_total_input_tokens: usize,
    /// The `max_total_output_tokens` budget that was applied.
    pub max_total_output_tokens: usize,
    /// Human-readable tokenizer name used for the counts.
    pub tokenizer_label: String,
    /// Whether the tokenizer was exact (tiktoken) or approximate.
    pub tokenizer_exact: bool,
}

/// Simple glob match: supports `*` at the start, end, or both.
fn matches_glob(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return name.ends_with(suffix);
    }
    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        let mut pos = 0usize;
        for (i, part) in parts.iter().enumerate() {
            if part.is_empty() {
                continue;
            }
            if i == 0 {
                if !name.starts_with(part) {
                    return false;
                }
                pos = part.len();
            } else {
                match name[pos..].find(part) {
                    Some(p) => pos += p + part.len(),
                    None => return false,
                }
            }
        }
        true
    } else {
        name == pattern
    }
}

/// Check whether a path is excluded by any of the exclude patterns.
/// Patterns are matched against both the leaf file/directory name and every parent component.
fn is_excluded(path: &std::path::Path, patterns: &[String]) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    for pattern in patterns {
        if matches_glob(name, pattern) {
            return true;
        }
        for component in path.components() {
            if let Some(s) = component.as_os_str().to_str()
                && matches_glob(s, pattern)
            {
                return true;
            }
        }
    }
    false
}

/// Check whether a file's extension is in the allowed list.
fn extension_matches(path: &std::path::Path, extensions: &[String]) -> bool {
    if extensions.is_empty() {
        return true;
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    extensions.iter().any(|e| e.to_ascii_lowercase() == ext)
}

/// Check whether a file path matches any of the include globs.
fn matches_include_glob(path: &std::path::Path, globs: &[String]) -> bool {
    if globs.is_empty() {
        return true;
    }
    let path_str = path.to_string_lossy();
    for glob in globs {
        if matches_glob(&path_str, glob) {
            return true;
        }
    }
    false
}

/// Assign an importance score to a file path so that package entry files, routes,
/// contexts, hooks, services, components, and utilities are processed in a sensible
/// order. Higher scores are more important.
fn importance_score(path: &std::path::Path) -> i32 {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let path_str = path.to_string_lossy().to_ascii_lowercase();

    // Package / app entry files (highest priority)
    if name == "package.json"
        || name == "cargo.toml"
        || name == "main.rs"
        || name == "lib.rs"
        || name == "mod.rs"
        || name == "app.ts"
        || name == "app.tsx"
        || name == "app.js"
        || name == "app.jsx"
        || name == "index.ts"
        || name == "index.tsx"
        || name == "index.js"
        || name == "index.jsx"
        || name == "main.ts"
        || name == "main.tsx"
        || name == "main.js"
        || name == "main.jsx"
        || name == "lib.ts"
        || name == "lib.js"
        || name.ends_with(".config.ts")
        || name.ends_with(".config.js")
        || name == "tsconfig.json"
    {
        return 100;
    }

    // Routes / pages / app files
    if path_str.contains("/route")
        || path_str.contains("/page")
        || path_str.contains("/app/")
        || path_str.contains("\\route")
        || path_str.contains("\\page")
        || path_str.contains("\\app\\")
        || name.contains("route")
        || name.contains("page")
    {
        return 90;
    }

    // Contexts / providers
    if path_str.contains("/context")
        || path_str.contains("/provider")
        || path_str.contains("\\context")
        || path_str.contains("\\provider")
        || name.contains("context")
        || name.contains("provider")
    {
        return 80;
    }

    // Hooks
    if path_str.contains("/hook") || path_str.contains("\\hook") || name.starts_with("use") {
        return 70;
    }

    // Services / api / db / auth / sync
    if path_str.contains("/service")
        || path_str.contains("/api")
        || path_str.contains("/db")
        || path_str.contains("/auth")
        || path_str.contains("/sync")
        || path_str.contains("\\service")
        || path_str.contains("\\api")
        || path_str.contains("\\db")
        || path_str.contains("\\auth")
        || path_str.contains("\\sync")
        || name.contains("service")
        || name.contains("api")
        || name.contains("auth")
        || name.contains("sync")
    {
        return 60;
    }

    // Components
    if path_str.contains("/component")
        || path_str.contains("/ui/")
        || path_str.contains("\\component")
        || path_str.contains("\\ui\\")
        || name.contains("component")
    {
        return 50;
    }

    // Utilities / types / helpers / common / shared (lowest priority)
    if path_str.contains("/util")
        || path_str.contains("/type")
        || path_str.contains("/helper")
        || path_str.contains("/common")
        || path_str.contains("/shared")
        || path_str.contains("\\util")
        || path_str.contains("\\type")
        || path_str.contains("\\helper")
        || path_str.contains("\\common")
        || path_str.contains("\\shared")
        || name.contains("util")
        || name.contains("type")
        || name.contains("helper")
        || name.contains("common")
        || name.contains("shared")
        || name.starts_with("_")
    {
        return 10;
    }

    // Default: medium-low
    40
}

/// Walk a directory and collect candidate file paths, applying exclude, extension,
/// and include-glob filters. Also returns files/directories skipped due to exclusion
/// patterns so they can be reported to the caller.
fn collect_files(
    dir: &std::path::Path,
    options: &CompressFolderOptions,
) -> Result<(Vec<std::path::PathBuf>, Vec<SkippedFile>)> {
    let mut files = Vec::new();
    let mut skipped = Vec::new();
    let mut dirs_to_visit = vec![dir.to_path_buf()];

    while let Some(current_dir) = dirs_to_visit.pop() {
        let entries = match std::fs::read_dir(&current_dir) {
            Ok(e) => e,
            Err(_) => continue, // skip unreadable directories
        };

        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };

            if metadata.is_dir() {
                if options.recursive {
                    if is_excluded(&path, &options.exclude_patterns) {
                        skipped.push(SkippedFile {
                            path: path.to_string_lossy().to_string(),
                            reason: "excluded".to_string(),
                        });
                    } else {
                        dirs_to_visit.push(path);
                    }
                }
                continue;
            }

            if !metadata.is_file() {
                continue;
            }

            if is_excluded(&path, &options.exclude_patterns) {
                skipped.push(SkippedFile {
                    path: path.to_string_lossy().to_string(),
                    reason: "excluded".to_string(),
                });
                continue;
            }

            if is_secret_file(&path) {
                skipped.push(SkippedFile {
                    path: path.to_string_lossy().to_string(),
                    reason: "secret".to_string(),
                });
                continue;
            }

            if !extension_matches(&path, &options.extensions) {
                continue;
            }

            if !matches_include_glob(&path, &options.include_globs) {
                continue;
            }

            files.push(path);
        }
    }

    Ok((files, skipped))
}

/// Read and compress multiple source files inside a folder.
///
/// Files are sorted by [`importance_score`], then processed in order until
/// `max_files` or `max_total_input_tokens` is reached. Each file is compressed via
/// [`compress_file`] so code-aware skeletonization is applied to recognised source
/// extensions.
pub fn compress_folder(
    path: impl AsRef<std::path::Path>,
    options: &CompressFolderOptions,
) -> Result<CompressFolderResult> {
    let path = path.as_ref();

    let metadata = std::fs::metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    if !metadata.is_dir() {
        anyhow::bail!("path is not a directory: {}", path.display());
    }

    let (mut candidates, mut skipped) = collect_files(path, options)?;

    // Sort by importance (highest first)
    candidates.sort_by(|a, b| {
        let score_a = importance_score(a);
        let score_b = importance_score(b);
        score_b.cmp(&score_a)
    });

    let mut results = Vec::new();
    let mut total_before = 0usize;
    let mut total_after = 0usize;
    let mut files_processed = 0usize;
    let mut tokenizer_label = String::new();
    let mut tokenizer_exact = false;

    for file_path in candidates {
        if files_processed >= options.max_files {
            skipped.push(SkippedFile {
                path: file_path.to_string_lossy().to_string(),
                reason: "max_files".to_string(),
            });
            continue;
        }

        let file_options = CompressFileOptions {
            start_line: None,
            end_line: None,
            max_bytes: None,
        };

        match compress_file(&file_path, &file_options) {
            Ok(result) => {
                if total_before + result.input_tokens_before > options.max_total_input_tokens {
                    skipped.push(SkippedFile {
                        path: file_path.to_string_lossy().to_string(),
                        reason: "input_token_limit".to_string(),
                    });
                    continue;
                }

                if total_after + result.input_tokens_after > options.max_total_output_tokens {
                    skipped.push(SkippedFile {
                        path: file_path.to_string_lossy().to_string(),
                        reason: "output_token_limit".to_string(),
                    });
                    continue;
                }

                total_before += result.input_tokens_before;
                total_after += result.input_tokens_after;
                tokenizer_label = result.tokenizer_label.clone();
                tokenizer_exact = result.tokenizer_exact;

                results.push(FileCompressionResult {
                    path: file_path.to_string_lossy().to_string(),
                    text: result.text,
                    input_tokens_before: result.input_tokens_before,
                    input_tokens_after: result.input_tokens_after,
                });
                files_processed += 1;
            }
            Err(e) => {
                let msg = e.to_string();
                let reason = if msg.contains("secret") || msg.contains("environment") {
                    "secret".to_string()
                } else if msg.contains("binary") {
                    "binary".to_string()
                } else {
                    msg
                };
                skipped.push(SkippedFile {
                    path: file_path.to_string_lossy().to_string(),
                    reason,
                });
            }
        }
    }

    Ok(CompressFolderResult {
        folder_path: path.to_string_lossy().to_string(),
        files: results,
        skipped,
        total_input_tokens_before: total_before,
        total_input_tokens_after: total_after,
        total_tokens_saved: total_before as i64 - total_after as i64,
        max_total_input_tokens: options.max_total_input_tokens,
        max_total_output_tokens: options.max_total_output_tokens,
        tokenizer_label,
        tokenizer_exact,
    })
}

/// Shrink a single text blob. The blob is wrapped in a one-message OpenAI request only so
/// the engine has something to operate on; we then run a **content-only** config (the
/// lossless `safe` preset) so the request-envelope stages never fire.
pub fn compress_text_blob(text: &str) -> Result<CompressTextBlobResult> {
    let body = serde_json::json!({
        "model": TEXT_WRAP_MODEL,
        "messages": [{ "role": "user", "content": text }],
    })
    .to_string();
    let config = config::DenseConfig::preset("safe").expect("built-in preset");
    let result = compress_with_config(&body, Some(ProviderKind::OpenAi), &config)?;
    let out = user_content(&result.request_json);

    let counter = tokenizer::counter_for(ProviderKind::OpenAi, Some(TEXT_WRAP_MODEL))?;
    let before = counter.count(text);
    let after = counter.count(&out);

    Ok(CompressTextBlobResult {
        text: out,
        input_tokens_before: before,
        input_tokens_after: after,
        tokenizer_label: counter.label().to_string(),
        tokenizer_exact: counter.is_exact(),
    })
}

/// Reject list for secret/environment files by exact filename (case-insensitive).
const SECRET_FILES: &[&str] = &[
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    ".envrc",
    ".env.sample",
    ".env.example",
    ".env.template",
    ".gitconfig",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "credentials",
    "credentials.json",
    "service_account.json",
    "token",
    "token.json",
    "tokens.json",
    "secrets.json",
    "secret.json",
];

/// Reject list for secret/environment file extensions (case-insensitive).
const SECRET_EXTENSIONS: &[&str] = &[
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".der",
    ".pub",
    ".gpg",
    ".asc",
    ".keystore",
    ".jks",
];

fn is_secret_file(path: &std::path::Path) -> bool {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if SECRET_FILES.iter().any(|s| name == *s) {
        return true;
    }
    if SECRET_EXTENSIONS.iter().any(|ext| name.ends_with(ext)) {
        return true;
    }
    false
}

fn is_likely_binary(data: &[u8]) -> bool {
    if data.contains(&0) {
        return true;
    }
    let control_count = data
        .iter()
        .filter(|&&b| b < 0x20 && b != b'\n' && b != b'\r' && b != b'\t')
        .count();
    if control_count > data.len() / 20 {
        return true;
    }
    false
}

/// Map a file extension to a fenced-code-block language tag that the
/// skeletonization and minify-code stages understand.
fn detect_language(path: &std::path::Path) -> Option<&'static str> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some("rust"),
        "js" | "mjs" | "cjs" => Some("js"),
        "jsx" => Some("jsx"),
        "ts" | "mts" | "cts" => Some("ts"),
        "tsx" => Some("tsx"),
        "py" => Some("py"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some("cpp"),
        "cs" => Some("csharp"),
        "kt" | "kts" => Some("kotlin"),
        "swift" => Some("swift"),
        "zig" => Some("zig"),
        "rb" => Some("ruby"),
        "php" => Some("php"),
        "ps1" => Some("powershell"), // passes through (unsupported by lang_for)
        _ => None,
    }
}

/// Strip the markdown fence wrapper (```lang\n...\n```) from compressed code.
fn strip_fence(text: &str) -> String {
    if let Some(nl) = text.find('\n') {
        let inner = &text[nl + 1..];
        if let Some(end) = inner.rfind("```") {
            return inner[..end].to_string();
        }
    }
    text.to_string()
}

/// Compress a text blob wrapped in a fenced code block so the code-aware stages
/// (skeletonization, minify-code) can fire. The fence is stripped from the result.
fn compress_fenced_code(text: &str, lang: &str) -> Result<CompressTextBlobResult> {
    let fenced = format!("```{lang}\n{text}\n```");
    let body = serde_json::json!({
        "model": TEXT_WRAP_MODEL,
        "messages": [{ "role": "user", "content": fenced }],
    })
    .to_string();

    // Start from the lossless `safe` preset and enable the two code-only stages.
    let mut config = config::DenseConfig::preset("safe").expect("built-in preset");
    config.skeletonize = true;
    config.minify_code = true;

    let result = compress_with_config(&body, Some(ProviderKind::OpenAi), &config)?;
    let out = user_content(&result.request_json);
    let out = strip_fence(&out);

    let counter = tokenizer::counter_for(ProviderKind::OpenAi, Some(TEXT_WRAP_MODEL))?;
    let before = counter.count(text);
    let after = counter.count(&out);

    Ok(CompressTextBlobResult {
        text: out,
        input_tokens_before: before,
        input_tokens_after: after,
        tokenizer_label: counter.label().to_string(),
        tokenizer_exact: counter.is_exact(),
    })
}

/// Read a local text file, validate it is not binary or a secret/env file, optionally
/// slice a line range, then compress its contents and report token statistics.
///
/// The raw file content is read inside this function and never returned; only the
/// compressed text and token counts are exposed.
///
/// For recognised source-code extensions (`.rs`, `.ts`, `.js`, `.py`, etc.) the file
/// is wrapped in a fenced code block and run through the code-aware stages
/// (skeletonization + minify-code) so function bodies are collapsed to stubs and
/// irrelevant whitespace is stripped. Non-code files use the lossless `safe` preset.
pub fn compress_file(
    path: impl AsRef<std::path::Path>,
    options: &CompressFileOptions,
) -> Result<CompressTextBlobResult> {
    use std::fs;

    let path = path.as_ref();

    if is_secret_file(path) {
        anyhow::bail!(
            "refusing to read secret or environment file: {}",
            path.display()
        );
    }

    let metadata = fs::metadata(path)
        .with_context(|| format!("failed to read metadata for {}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("path is not a file: {}", path.display());
    }

    let data = if let Some(max) = options.max_bytes {
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("failed to open {}", path.display()))?;
        let mut buf = vec![0u8; max];
        let n = std::io::Read::read(&mut file, &mut buf)
            .with_context(|| format!("failed to read {}", path.display()))?;
        buf.truncate(n);
        buf
    } else {
        fs::read(path).with_context(|| format!("failed to read {}", path.display()))?
    };

    if is_likely_binary(&data) {
        anyhow::bail!("file appears to be binary: {}", path.display());
    }

    let text = String::from_utf8(data)
        .with_context(|| format!("file is not valid UTF-8 text: {}", path.display()))?;

    let text = if options.start_line.is_some() || options.end_line.is_some() {
        let lines: Vec<&str> = text.lines().collect();
        let start = options.start_line.unwrap_or(1).saturating_sub(1);
        let end = options
            .end_line
            .unwrap_or(lines.len())
            .saturating_sub(1)
            .min(lines.len().saturating_sub(1));
        if start > end {
            anyhow::bail!("start_line ({}) is after end_line ({})", start + 1, end + 1);
        }
        lines[start..=end].join("\n")
    } else {
        text
    };

    if let Some(lang) = detect_language(path) {
        compress_fenced_code(&text, lang)
    } else {
        compress_text_blob(&text)
    }
}

/// Pull the first user message's text back out of a compressed request, for
/// `compress_text_blob`. Content may be a plain string or an array of typed blocks
/// (any provider, any language); concatenate the text parts. Falls back to the whole
/// compressed JSON if the shape is unexpected.
fn user_content(request_json: &str) -> String {
    let parsed: Value = match serde_json::from_str(request_json) {
        Ok(v) => v,
        Err(_) => return request_json.to_string(),
    };
    let Some(msg) = parsed
        .get("messages")
        .and_then(Value::as_array)
        .and_then(|m| {
            m.iter()
                .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        })
    else {
        return request_json.to_string();
    };
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let text: Vec<&str> = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .collect();
            if text.is_empty() {
                request_json.to_string()
            } else {
                text.join("")
            }
        }
        _ => request_json.to_string(),
    }
}

/// Reverse the lossless output transforms recorded in a rehydration plan. Internal: no
/// output-side transform ships today (Stage D is input-only; DSS was removed), so this is an
/// inert passthrough — a JSON response is normalized, plain text returned unchanged. Kept
/// `pub` so the `llmtrim` CLI's interceptor (a separate crate) can call it as its inverse
/// hook; `#[doc(hidden)]` keeps it off the embedding API — it is an inert passthrough today
/// and embedders should not depend on it.
#[doc(hidden)]
pub fn rehydrate(response: &str, _plan: &str) -> Result<String> {
    match serde_json::from_str::<Value>(response) {
        Ok(value) => {
            serde_json::to_string(&value).context("failed to serialize rehydrated response")
        }
        Err(_) => Ok(response.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_default_is_behavior_preserving() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let cfg = config::DenseConfig::default();
        let result =
            compress_with_config(input, Some(ProviderKind::OpenAi), &cfg).expect("compress");
        assert!(result.tokenizer_exact);
        let body: Value = serde_json::from_str(&result.request_json).unwrap();
        let msgs = body.get("messages").and_then(Value::as_array).unwrap();
        // Default = lossless input only: content intact, no injected system
        // instruction (the model's output behavior is unchanged).
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].get("content").and_then(Value::as_str), Some("hi"));
        assert!(
            !msgs
                .iter()
                .any(|m| m.get("role").and_then(Value::as_str) == Some("system")),
            "default must not change the model's output behavior"
        );
    }

    #[test]
    fn route_picks_preset_by_shape() {
        use serde_json::json;
        let p = provider::for_kind(ProviderKind::OpenAi);
        let mk = |v: Value| Request::from_value(ProviderKind::OpenAi, v);

        let tools = mk(json!({"messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f"}}]}));
        assert_eq!(route(&tools, p.as_ref()), "agent");

        let code =
            mk(json!({"messages":[{"role":"user","content":"fix:\n```rust\nfn x(){}\n```"}]}));
        assert_eq!(route(&code, p.as_ref()), "code");

        let long = "the report covers revenue and costs. ".repeat(60); // >1200 chars
        let rag = mk(json!({"messages":[{"role":"user","content":long},
            {"role":"user","content":"what was the revenue?"}]}));
        assert_eq!(route(&rag, p.as_ref()), "rag");

        let plain = mk(json!({"messages":[{"role":"user","content":"write a poem about spring"}]}));
        assert_eq!(route(&plain, p.as_ref()), "aggressive");
    }

    #[test]
    fn auto_routes_at_compress_time() {
        use serde_json::json;
        // auto on a tools request → agent preset → its long description gets trimmed.
        let input = json!({"model":"gpt-4o",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[{"type":"function","function":{"name":"f","description":"x".repeat(500)}}]})
        .to_string();
        let r = compress_with_config(
            &input,
            Some(ProviderKind::OpenAi),
            &config::DenseConfig::auto(),
        )
        .expect("compress");
        assert!(
            r.input_tokens_after < r.input_tokens_before,
            "auto routed to agent and trimmed the tool description"
        );
        assert!(
            !config::DenseConfig::default().auto,
            "plain default is not auto"
        );
    }

    #[test]
    fn compress_is_identity_when_all_stages_off() {
        let input =
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let cfg = config::DenseConfig {
            hygiene: false,
            serialize: false,
            output_control: false,
            ..config::DenseConfig::default()
        };
        let result =
            compress_with_config(input, Some(ProviderKind::OpenAi), &cfg).expect("compress");
        let a: Value = serde_json::from_str(input).unwrap();
        let b: Value = serde_json::from_str(&result.request_json).unwrap();
        assert_eq!(a, b, "all stages off => identity");
        assert_eq!(result.input_tokens_before, result.input_tokens_after);
        assert!(result.input_tokens_before.0 > 0);
    }

    #[test]
    fn compress_auto_detects_anthropic() {
        let input = r#"{"system":"s","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
        let result = compress(input, None).expect("auto-detect anthropic");
        assert_eq!(result.provider, ProviderKind::Anthropic);
        assert!(!result.tokenizer_exact, "anthropic counts are approximate");
    }

    #[test]
    fn compress_rejects_invalid_json() {
        assert!(compress("not json", Some(ProviderKind::OpenAi)).is_err());
    }

    #[test]
    fn rehydrate_passes_through_without_transforms() {
        let resp = r#"{"content":"hello"}"#;
        let out = rehydrate(resp, "{}").expect("rehydrate");
        let a: Value = serde_json::from_str(resp).unwrap();
        let b: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn agent_preset_compresses_tool_result_diff() {
        // A 40-file diff returned as a tool_result — over the toolout file cap, so the
        // least-changed files are dropped to positional elision markers.
        let mut diff = String::new();
        for i in 0..40 {
            diff.push_str(&format!(
                "diff --git a/f{i}.rs b/f{i}.rs\n--- a/f{i}.rs\n+++ b/f{i}.rs\n\
                 @@ -1,3 +1,3 @@\n ctx_{i}\n-old line {i}\n+new line {i}\n trailing_{i}\n"
            ));
        }
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t1", "content": diff}],
            }],
            "max_tokens": 1024,
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let result =
            compress_with_config(&input, Some(ProviderKind::Anthropic), &cfg).expect("compress");

        assert!(
            result.input_tokens_after < result.input_tokens_before,
            "toolout compressed the diff ({} -> {})",
            result.input_tokens_before,
            result.input_tokens_after
        );
        assert!(
            result.request_json.contains("omitted"),
            "dropped files left a positional elision marker in the body"
        );
    }

    #[test]
    fn frozen_prefix_untouched_while_live_zone_compresses() {
        // message 0 is the cached prefix (`cache_control`) holding a big log; message 1 is
        // the live user turn with another big log. The agent preset must compress the live
        // log but leave the cached one byte-identical — else it busts the prompt cache.
        let cached_log = (0..80)
            .map(|i| format!("INFO  step {i} routine nominal pass"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\nERROR failure inside the cached prefix";
        let live_log = (0..80)
            .map(|i| format!("DEBUG worker {i} idle waiting for work"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\nERROR failure inside the live turn";
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "a", "content": cached_log,
                     "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "b", "content": live_log}
                ]},
            ],
            "max_tokens": 1024,
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let result =
            compress_with_config(&input, Some(ProviderKind::Anthropic), &cfg).expect("compress");
        let body: Value = serde_json::from_str(&result.request_json).unwrap();

        let m0 = body
            .pointer("/messages/0/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(
            m0, cached_log,
            "cached prefix must be byte-identical (cache stays warm)"
        );

        let m1 = body
            .pointer("/messages/1/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert!(
            m1.len() < live_log.len(),
            "live turn was compressed ({} -> {})",
            live_log.len(),
            m1.len()
        );
    }

    #[test]
    fn agent_tool_block_is_byte_stable_across_turns() {
        // Issue #9: on an agent loop the `tools[]` block is part of the cached prompt prefix, so
        // it must be byte-identical turn-to-turn or the provider prompt cache is busted. Two
        // consecutive mid-loop turns (a tool was already invoked, so selection is skipped) must
        // compress to the exact same tools block.
        let tools = serde_json::json!([
            {"type":"function","function":{"name":"read_file","description":"Read a file from disk by path.","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}},
            {"type":"function","function":{"name":"grep_search","description":"Search files with a regex.","parameters":{"type":"object","properties":{"pattern":{"type":"string"}}}}},
            {"type":"function","function":{"name":"run_bash","description":"Run a shell command.","parameters":{"type":"object","properties":{"command":{"type":"string"}}}}},
            {"type":"function","function":{"name":"web_search","description":"Search the web.","parameters":{"type":"object","properties":{"query":{"type":"string"}}}}}
        ]);
        let turn_a = serde_json::json!({
            "model": "gpt-4o-mini", "tools": tools,
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "read main.rs"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"main.rs\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "fn main() {}"},
                {"role": "user", "content": "now grep for the word transform"}
            ]
        })
        .to_string();
        // Turn B = turn A + the grep call/result + a new ask (the agent-loop shape).
        let turn_b = serde_json::json!({
            "model": "gpt-4o-mini", "tools": tools,
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "read main.rs"},
                {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"main.rs\"}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "fn main() {}"},
                {"role": "user", "content": "now grep for the word transform"},
                {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "grep_search", "arguments": "{\"pattern\":\"transform\"}"}}]},
                {"role": "tool", "tool_call_id": "c2", "content": "main.rs:1: transform()"},
                {"role": "user", "content": "now run the tests with bash"}
            ]
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let ra = compress_with_config(&turn_a, Some(ProviderKind::OpenAi), &cfg).unwrap();
        let rb = compress_with_config(&turn_b, Some(ProviderKind::OpenAi), &cfg).unwrap();
        let tools_of =
            |json: &str| -> Value { serde_json::from_str::<Value>(json).unwrap()["tools"].clone() };
        assert_eq!(
            tools_of(&ra.request_json),
            tools_of(&rb.request_json),
            "the agent preset must emit a byte-identical tools[] block across turns (cache prefix stays warm)"
        );
    }

    #[test]
    fn repeated_tool_invocation_ships_full_output() {
        // Rail: repeat → passthrough. The agent re-ran a tool because its first result
        // was compressed — the newest occurrence must ship byte-identical (this is the
        // recovery the elision header promises), while the first still compresses.
        let dump = (0..80)
            .map(|i| format!("src/a.rs:{}:    let v = step({i});", i + 1))
            .collect::<Vec<_>>()
            .join("\n");
        let input = serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": dump}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t2", "content": dump}
                ]},
            ],
            "max_tokens": 1024,
        })
        .to_string();

        let cfg = config::DenseConfig::preset("agent").expect("agent preset");
        let result =
            compress_with_config(&input, Some(ProviderKind::Anthropic), &cfg).expect("compress");
        let body: Value = serde_json::from_str(&result.request_json).unwrap();

        let first = body
            .pointer("/messages/0/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        let second = body
            .pointer("/messages/1/content/0/content")
            .and_then(Value::as_str)
            .unwrap();
        assert_eq!(second, dump, "the repeat ships in full");
        assert_ne!(first, dump, "the first occurrence still compresses");
        assert!(
            first.len() < dump.len(),
            "first occurrence got smaller ({} -> {})",
            dump.len(),
            first.len()
        );
    }

    // ── compress_text_blob ─────────────────────────────────────────────────────────────

    #[test]
    fn compress_text_blob_compresses_text() {
        let text = "repeat me\nrepeat me\ntail words here";
        let result = compress_text_blob(text).expect("compress_text_blob");
        assert!(
            result.input_tokens_after <= result.input_tokens_before,
            "safe preset must not grow tokens"
        );
        assert!(result.tokenizer_exact, "gpt-4o uses tiktoken (exact)");
        assert_eq!(
            result.input_tokens_before as i64 - result.input_tokens_after as i64,
            result.input_tokens_before as i64 - result.input_tokens_after as i64
        );
        assert!(!result.text.is_empty());
    }

    // ── compress_file ──────────────────────────────────────────────────────────────────

    #[test]
    fn compress_file_reads_and_compresses() {
        let dir = std::env::temp_dir().join(format!("llmtrim-file-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.txt");
        std::fs::write(&path, "Hello    world\nwith   extra   spaces.\n").unwrap();

        let result = compress_file(&path, &CompressFileOptions::default()).expect("compress_file");
        assert!(!result.text.is_empty());
        assert!(result.input_tokens_before > 0);
        assert!(result.input_tokens_after > 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_rejects_secret() {
        let dir = std::env::temp_dir().join(format!("llmtrim-secret-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(".env");
        std::fs::write(&path, "SECRET=123").unwrap();

        assert!(
            compress_file(&path, &CompressFileOptions::default()).is_err(),
            ".env must be rejected"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_rejects_binary() {
        let dir = std::env::temp_dir().join(format!("llmtrim-binary-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("data.bin");
        std::fs::write(&path, b"\x00\x01\x02\x03").unwrap();

        assert!(
            compress_file(&path, &CompressFileOptions::default()).is_err(),
            "null bytes must be rejected as binary"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_rejects_directory() {
        let dir = std::env::temp_dir().join(format!("llmtrim-dir-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        assert!(
            compress_file(&dir, &CompressFileOptions::default()).is_err(),
            "directory must be rejected"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_line_range() {
        let dir = std::env::temp_dir().join(format!("llmtrim-range-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lines.txt");
        std::fs::write(&path, "line one\nline two\nline three\nline four\n").unwrap();

        let result = compress_file(
            &path,
            &CompressFileOptions {
                start_line: Some(2),
                end_line: Some(3),
                ..Default::default()
            },
        )
        .expect("compress_file range");
        assert!(
            result.text.contains("line two"),
            "range must include line two"
        );
        assert!(
            result.text.contains("line three"),
            "range must include line three"
        );
        assert!(
            !result.text.contains("line one"),
            "range must exclude line one"
        );
        assert!(
            !result.text.contains("line four"),
            "range must exclude line four"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_skeletonizes_rust_code() {
        let dir = std::env::temp_dir().join(format!("llmtrim-rust-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.rs");
        std::fs::write(
            &path,
            "fn main() {\n    println!(\"hello\");\n}\n\nfn helper() {\n    let x = 42;\n}\n",
        )
        .unwrap();

        let result = compress_file(&path, &CompressFileOptions::default()).expect("compress_file");
        assert!(
            result.text.contains("{ /* … */ }"),
            "rust function bodies should be skeletonized: {}",
            result.text
        );
        assert!(
            !result.text.contains("println!(\"hello\")"),
            "body should be stripped: {}",
            result.text
        );
        assert!(result.input_tokens_after < result.input_tokens_before);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_skeletonizes_typescript_code() {
        let dir = std::env::temp_dir().join(format!("llmtrim-ts-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.ts");
        std::fs::write(
            &path,
            "function greet(name: string) {\n    return `Hello ${name}`;\n}\n",
        )
        .unwrap();

        let result = compress_file(&path, &CompressFileOptions::default()).expect("compress_file");
        assert!(
            result.text.contains("{ /* … */ }"),
            "ts function bodies should be skeletonized: {}",
            result.text
        );
        assert!(
            !result.text.contains("Hello ${name}"),
            "body should be stripped: {}",
            result.text
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_file_plain_text_not_skeletonized() {
        let dir = std::env::temp_dir().join(format!("llmtrim-txt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("sample.txt");
        std::fs::write(&path, "function foo() {\n    return 42;\n}\n").unwrap();

        let result = compress_file(&path, &CompressFileOptions::default()).expect("compress_file");
        assert!(
            !result.text.contains("{ /* … */ }"),
            "plain text must not be skeletonized: {}",
            result.text
        );
        assert!(
            result.text.contains("return 42;"),
            "plain text body should survive: {}",
            result.text
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── compress_folder ────────────────────────────────────────────────────────────────

    #[test]
    fn compress_folder_reads_and_compresses_source_files() {
        let dir = std::env::temp_dir().join(format!("llmtrim-folder-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        std::fs::write(dir.join("helper.rs"), "fn helper() {\n    let x = 42;\n}\n").unwrap();
        std::fs::write(dir.join("readme.md"), "# Hello\n\nWorld.\n").unwrap();

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string(), "md".to_string()],
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(result.folder_path, dir.to_string_lossy().to_string());
        assert_eq!(result.files.len(), 3, "should include .rs and .md files");
        // main.rs should be first (importance score 100 vs 40 for helper.rs)
        assert!(result.files[0].path.contains("main.rs"));
        assert!(
            result.files[0].text.contains("{ /* … */ }"),
            "rust should be skeletonized"
        );
        assert!(
            result.files[2].path.contains("helper.rs")
                || result.files[1].path.contains("helper.rs")
        );
        assert!(result.total_input_tokens_before > 0);
        assert!(result.total_tokens_saved >= 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_respects_exclude_patterns() {
        let dir = std::env::temp_dir().join(format!("llmtrim-exclude-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("good.rs"), "fn good() {}\n").unwrap();
        let bad_dir = dir.join("node_modules");
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::write(bad_dir.join("bad.rs"), "fn bad() {}\n").unwrap();

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string()],
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec!["node_modules".to_string()],
            include_globs: vec![],
            recursive: true,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(result.files.len(), 1);
        assert!(result.files[0].path.contains("good.rs"));
        assert!(
            !result.files.iter().any(|f| f.path.contains("node_modules")),
            "node_modules should be excluded"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_respects_max_files() {
        let dir =
            std::env::temp_dir().join(format!("llmtrim-maxfiles-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..5 {
            std::fs::write(
                dir.join(format!("file{}.rs", i)),
                format!("fn f{}() {{}}\n", i),
            )
            .unwrap();
        }

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string()],
            max_files: 2,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(result.files.len(), 2);
        assert!(result.skipped.iter().any(|s| s.reason == "max_files"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_respects_max_tokens() {
        let dir = std::env::temp_dir().join(format!("llmtrim-maxtok-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..3 {
            std::fs::write(
                dir.join(format!("file{}.rs", i)),
                format!("fn f{}() {{}}\n", i),
            )
            .unwrap();
        }

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string()],
            max_files: 10,
            max_total_input_tokens: 1, // very low limit
            max_total_output_tokens: 100000,
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(
            result.files.len(),
            0,
            "no file should fit under 1 token limit"
        );
        assert!(
            result
                .skipped
                .iter()
                .any(|s| s.reason == "input_token_limit"),
            "files should be skipped due to input token limit"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_skips_secret_files() {
        let dir =
            std::env::temp_dir().join(format!("llmtrim-folder-secret-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("normal.rs"), "fn normal() {}\n").unwrap();
        std::fs::write(dir.join(".env"), "SECRET=123\n").unwrap();

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string(), "".to_string()], // allow no-extension for .env
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert!(result.files.iter().any(|f| f.path.contains("normal.rs")));
        assert!(
            result
                .skipped
                .iter()
                .any(|s| s.path.contains(".env") && s.reason == "secret"),
            "secret file should be skipped with reason 'secret'"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    #[allow(clippy::needless_borrows_for_generic_args)]
    fn compress_folder_non_directory_fails() {
        let file = std::env::temp_dir().join(format!("llmtrim-notdir-{}", std::process::id()));
        std::fs::write(&file, "not a dir\n").unwrap();

        let options = CompressFolderOptions::default();
        assert!(
            compress_folder(&file, &options).is_err(),
            "file path must be rejected as not a directory"
        );

        let _ = std::fs::remove_file(&file);
    }

    #[test]
    fn compress_folder_respects_output_token_limit() {
        let dir = std::env::temp_dir().join(format!("llmtrim-outtok-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..3 {
            std::fs::write(
                dir.join(format!("file{}.rs", i)),
                format!("fn f{}() {{}}\n", i),
            )
            .unwrap();
        }

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string()],
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 1, // very low output limit
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(
            result.files.len(),
            0,
            "no file should fit under 1 output token limit"
        );
        assert!(
            result
                .skipped
                .iter()
                .any(|s| s.reason == "output_token_limit"),
            "files should be skipped due to output token limit"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_tracks_excluded_in_skipped() {
        let dir = std::env::temp_dir().join(format!(
            "llmtrim-excluded-skipped-test-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("good.rs"), "fn good() {}\n").unwrap();
        let excluded_dir = dir.join("node_modules");
        std::fs::create_dir_all(&excluded_dir).unwrap();
        std::fs::write(excluded_dir.join("bad.rs"), "fn bad() {}\n").unwrap();

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string()],
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec!["node_modules".to_string()],
            include_globs: vec![],
            recursive: true,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(result.files.len(), 1);
        assert!(
            result
                .skipped
                .iter()
                .any(|s| s.path.contains("node_modules") && s.reason == "excluded"),
            "excluded directory should appear in skipped with reason 'excluded'"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn compress_folder_importance_sorts_correctly() {
        let dir = std::env::temp_dir().join(format!("llmtrim-sort-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("util.rs"), "fn util() {}\n").unwrap();
        std::fs::write(dir.join("main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\n").unwrap();

        let options = CompressFolderOptions {
            extensions: vec!["rs".to_string(), "toml".to_string()],
            max_files: 10,
            max_total_input_tokens: 100000,
            max_total_output_tokens: 100000,
            exclude_patterns: vec![],
            include_globs: vec![],
            recursive: false,
        };
        let result = compress_folder(&dir, &options).expect("compress_folder");

        assert_eq!(result.files.len(), 3);
        // Cargo.toml (100) > main.rs (100) > util.rs (10)
        assert!(result.files[0].path.contains("Cargo.toml"));
        assert!(result.files[1].path.contains("main.rs"));
        assert!(result.files[2].path.contains("util.rs"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
