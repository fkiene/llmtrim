use anyhow::{Context, Result};
use serde_json::Value;

use crate::{compress_with_config, config, ir::ProviderKind, tokenizer};

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

pub(crate) fn is_secret_file(path: &std::path::Path) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_text_blob_compresses_text() {
        let text = "repeat me\nrepeat me\ntail words here";
        let result = compress_text_blob(text).expect("compress_text_blob");
        assert!(
            result.input_tokens_after <= result.input_tokens_before,
            "safe preset must not grow tokens"
        );
        assert!(result.tokenizer_exact, "gpt-4o uses tiktoken (exact)");
        assert!(!result.text.is_empty());
    }

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
}
