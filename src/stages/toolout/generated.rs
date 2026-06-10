//! Generated / lockfile near-total elision (feature #7).
//!
//! Some tool output is machine-generated noise the model cannot meaningfully reason over:
//! a `Cargo.lock` / `package-lock.json` / `yarn.lock` / `pnpm-lock.yaml` body, a minified
//! `*.min.js` / `*.min.css` bundle (one enormous line), or a long base64 / source-map blob.
//! Echoing thousands of lines of integrity hashes or a 200 KB single-line bundle into the
//! context costs heavily and buys nothing. This collapses such a body to a one-line marker
//! `[generated file elided: N lines, ~M chars]`, the same positional-reference convention
//! the rest of the stage uses — the agent re-reads the file if it genuinely needs it.
//!
//! Detection is by **content shape**, never filename (this stage sees pasted content, not
//! paths), and deliberately conservative: it fires only on high-confidence machine shapes
//! (a recognizable lockfile header, a dense run of `resolved`/`integrity` records, an
//! over-long minified single line, or a long base64-ish run) and declines on anything that
//! looks like ordinary source, JSON, or prose. The pipeline's token gate is a second
//! backstop, but the shape gates are what keep this from ever touching real code.
//!
//! Signals are structural punctuation / hash shapes, not words, so this is language-neutral.

use once_cell::sync::Lazy;
use regex::Regex;

/// Below this many characters a body is too small to be worth a generated-file marker
/// (and too small to be a real lockfile / bundle) — decline.
const MIN_CHARS: usize = 800;

/// A single line longer than this is "not human-authored source": minified bundles put a
/// whole module on one line. Real source wraps; even very long string literals rarely run
/// this far without a newline.
const MINIFIED_LINE_CHARS: usize = 2000;

/// A run of base64/source-map data at least this long (chars) is a generated blob.
const BASE64_RUN_CHARS: usize = 1500;

/// Lockfile-defining header lines (exact machine-emitted preambles).
static LOCK_HEADER: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r#"(?m)^# This file is automatically @?generated"#, // Cargo.lock / many tools
        r#"|^# yarn lockfile v1"#,                          // yarn.lock
        r#"|^"lockfileVersion":"#,                          // package-lock.json (pretty)
        r#"|^lockfileVersion:"#,                            // pnpm-lock.yaml
    ))
    .unwrap()
});

/// A per-dependency record line shared by the lockfile formats: a `resolved`/`integrity`
/// URL-or-hash field, or a Cargo.lock `checksum = "…"`. High density of these is a lockfile
/// even without a recognizable header (a body excerpt).
static LOCK_RECORD: Lazy<Regex> = Lazy::new(|| {
    Regex::new(concat!(
        r#"(?m)^\s*"?(resolved|integrity)"?\s*[:=]"#, // npm/yarn resolved/integrity
        r#"|^\s*checksum = ""#,                       // Cargo.lock
        r#"|^\s*"resolved":"#,
    ))
    .unwrap()
});

/// A `[[package]]` table — the repeating Cargo.lock block.
static CARGO_PACKAGE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?m)^\[\[package\]\]").unwrap());

/// A long contiguous base64-ish token (source maps, embedded fonts/images, inline data
/// URIs). 40+ chars of base64 alphabet with no break.
static BASE64_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9+/=]{40,}").unwrap());

/// If `text` is a high-confidence generated/lockfile/minified body, return the one-line
/// elision marker; otherwise `None` (let the normal detection arm handle it).
pub fn compress(text: &str) -> Option<String> {
    let chars = text.chars().count();
    if chars < MIN_CHARS {
        return None;
    }
    if is_lockfile(text) || is_minified(text) || is_base64_blob(text) {
        let lines = text.lines().count();
        return Some(format!(
            "[generated file elided: {lines} lines, ~{chars} chars]"
        ));
    }
    None
}

/// A lockfile: a defining header line, or — for a header-less excerpt — a high density of
/// `resolved`/`integrity`/`checksum` records, or several `[[package]]` blocks.
fn is_lockfile(text: &str) -> bool {
    if LOCK_HEADER.is_match(text) {
        return true;
    }
    let lines = text.lines().count();
    if lines < 12 {
        return false; // too short to judge by density alone
    }
    let records = LOCK_RECORD.find_iter(text).count();
    // Either a strong record density (npm/yarn lockfiles are ~25% resolved/integrity
    // lines) or several Cargo `[[package]]` blocks each with a checksum.
    records * 100 >= lines * 15 || (CARGO_PACKAGE.find_iter(text).count() >= 5 && records >= 5)
}

/// A minified bundle: at least one line far over [`MINIFIED_LINE_CHARS`] that is dense
/// punctuation and sparse whitespace (the `*.min.js`/`.min.css` signature). Guards against
/// a long prose paragraph or a single huge base64 string (few punctuation breaks) being
/// mistaken for code.
fn is_minified(text: &str) -> bool {
    text.lines().any(|line| {
        let len = line.chars().count();
        if len < MINIFIED_LINE_CHARS {
            return false;
        }
        let ws = line.chars().filter(|c| c.is_whitespace()).count();
        let punct = line
            .chars()
            .filter(|c| matches!(c, ';' | '{' | '}' | '(' | ')' | ',' | '=' | ':' | '.'))
            .count();
        // Sparse whitespace (minifiers strip it) AND dense code punctuation.
        ws * 100 < len * 8 && punct * 100 >= len * 4
    })
}

/// A source-map / embedded-asset body: a single base64-ish run longer than
/// [`BASE64_RUN_CHARS`] (an inline data URI, a `"mappings"` blob, an embedded font).
fn is_base64_blob(text: &str) -> bool {
    BASE64_RUN
        .find_iter(text)
        .any(|m| m.as_str().chars().count() >= BASE64_RUN_CHARS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_tokens(s: &str) -> usize {
        s.split_whitespace().count()
    }

    fn assert_big_savings(input: &str, out: &str) {
        let before = count_tokens(input);
        let after = count_tokens(out);
        let savings = 100.0 - (after as f64 / before as f64 * 100.0);
        assert!(
            savings >= 60.0,
            "expected ≥60% token savings, got {savings:.1}%"
        );
    }

    #[test]
    fn cargo_lock_body_elides() {
        let mut body = String::from("# This file is automatically @generated by Cargo.\n");
        body.push_str("# It is not intended for manual editing.\n");
        for i in 0..40 {
            body.push_str(&format!(
                "[[package]]\nname = \"crate-{i}\"\nversion = \"1.{i}.0\"\nsource = \"registry+https://github.com/rust-lang/crates.io-index\"\nchecksum = \"a1b2c3d4e5f6{i:040}\"\n\n"
            ));
        }
        let out = compress(&body).expect("a Cargo.lock body elides");
        assert!(
            out.starts_with("[generated file elided:"),
            "marker emitted: {out}"
        );
        assert!(out.lines().count() == 1, "collapsed to one line");
        assert_big_savings(&body, &out);
    }

    #[test]
    fn package_lock_excerpt_elides_without_header() {
        // A header-less excerpt of package-lock.json: judged by resolved/integrity density.
        let mut body = String::new();
        for i in 0..30 {
            body.push_str(&format!(
                "    \"node_modules/pkg-{i}\": {{\n      \"version\": \"2.{i}.1\",\n      \"resolved\": \"https://registry.npmjs.org/pkg-{i}/-/pkg-{i}-2.{i}.1.tgz\",\n      \"integrity\": \"sha512-{i:080}==\"\n    }},\n"
            ));
        }
        let out = compress(&body).expect("a package-lock excerpt elides by density");
        assert!(out.starts_with("[generated file elided:"));
        assert_big_savings(&body, &out);
    }

    #[test]
    fn minified_bundle_elides() {
        // One enormous line of dense, whitespace-sparse JS.
        let chunk = "function f(a,b){return a+b;}var x={k:1,v:2};g(x,f);";
        let line = chunk.repeat(80); // well over MINIFIED_LINE_CHARS
        let body = format!("//# sourceMappingURL=bundle.js.map\n{line}\n");
        let out = compress(&body).expect("a minified bundle elides");
        assert!(out.starts_with("[generated file elided:"));
        assert_big_savings(&body, &out);
    }

    #[test]
    fn base64_blob_elides() {
        let blob = "A".repeat(BASE64_RUN_CHARS + 200);
        let body = format!("data:font/woff2;base64,{blob}");
        let out = compress(&body).expect("a long base64 blob elides");
        assert!(out.starts_with("[generated file elided:"));
    }

    #[test]
    fn ordinary_rust_code_is_not_elided() {
        // Real, multi-line source with normal whitespace must never be elided.
        let code = (0..60)
            .map(|i| {
                format!(
                    "    let value_{i} = compute(input_{i}, factor); // step {i} of the pipeline"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(
            compress(&code),
            None,
            "ordinary source is left for the normal arm"
        );
    }

    #[test]
    fn ordinary_json_is_not_elided() {
        // Hand-authored config JSON has `version` but no resolved/integrity record density,
        // no minified line, no base64 blob.
        let json = (0..40)
            .map(|i| format!("  \"setting_{i}\": {{ \"enabled\": true, \"weight\": {i} }},"))
            .collect::<Vec<_>>()
            .join("\n");
        let json = format!("{{\n{json}\n  \"version\": \"1.0.0\"\n}}");
        assert_eq!(
            compress(&json),
            None,
            "ordinary JSON config is not a lockfile"
        );
    }

    #[test]
    fn prose_is_not_elided() {
        // A long prose paragraph on one wrapped-free line is not minified code (sparse
        // punctuation) and has no lockfile/base64 signal.
        let prose = "the quick brown fox jumps over the lazy dog ".repeat(60);
        assert_eq!(compress(&prose), None, "prose is never a generated file");
    }

    #[test]
    fn small_body_is_declined() {
        let small = "resolved \"x\"\nintegrity \"y\"\nresolved \"z\"";
        assert_eq!(compress(small), None, "below MIN_CHARS, declined");
    }
}
