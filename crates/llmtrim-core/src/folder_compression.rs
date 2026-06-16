use anyhow::{Context, Result};

use crate::file_compression::{CompressFileOptions, compress_file, is_secret_file};

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

#[cfg(test)]
mod tests {
    use super::*;

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
