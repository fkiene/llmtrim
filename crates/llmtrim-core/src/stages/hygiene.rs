//! Stage D — lossless data hygiene (+ optional base64 stripping).
//!
//! For each content text segment that parses as JSON, re-serialize it minified
//! (drops pretty-print whitespace; serde also normalizes redundant numeric forms
//! like trailing zeros). The JSON *value* is unchanged → no rehydration entry.
//!
//! When `strip_base64` is enabled (opt-in; lossy), long base64 runs and `data:`
//! URIs are replaced with a size placeholder — inside JSON string values when the
//! content is JSON, or in the raw text otherwise. Off by default because it removes
//! bytes the caller might actually need.

use anyhow::Result;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::gate::{GateKind, PlanEntry, Transform};
use crate::ir::Request;
use crate::provider::Provider;

pub struct HygieneStage {
    pub strip_base64: bool,
    /// Opt-in lossy: round float numbers to this many significant figures
    /// (CompactPrompt-style column quantization, arXiv:2510.18043).
    pub sig_figs: Option<u32>,
    /// Opt-in: tokenizer-aware text normalization — drop invisible/format waste,
    /// fold no-break spaces, then NFKC. Meaning-preserving but not byte-reversible.
    pub normalize_unicode: bool,
}

impl Transform for HygieneStage {
    fn name(&self) -> &str {
        "hygiene"
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
        for ptr in crate::cache_zone::compressible_pointers(req, provider) {
            let Some(s) = req.get_str(&ptr).map(str::to_string) else {
                continue;
            };
            if let Ok(mut value) = serde_json::from_str::<Value>(&s) {
                // JSON content: scrub base64 + normalize text + quantize floats, then minify.
                if self.strip_base64 {
                    scrub_base64_value(&mut value);
                }
                if self.normalize_unicode {
                    normalize_value(&mut value);
                }
                if let Some(sig) = self.sig_figs {
                    quantize_value(&mut value, sig);
                }
                let minified = serde_json::to_string(&value)?;
                if minified != s {
                    req.set(&ptr, Value::String(minified));
                }
            } else {
                // Non-JSON prose: optional base64 scrub + unicode normalization, then minify
                // any JSON embedded *inside* the prose (fenced ```json blocks and balanced
                // top-level {…}/[…] spans). The whole-segment JSON case is handled above.
                let mut text = s.clone();
                if self.strip_base64 {
                    text = scrub_base64_text(&text);
                }
                if self.normalize_unicode {
                    text = normalize_text(&text);
                }
                text = minify_embedded_json(&text);
                if text != s {
                    req.set(&ptr, Value::String(text));
                }
            }
        }
        Ok(())
    }
}

static DATA_URI: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"data:[A-Za-z0-9.+/-]+;base64,[A-Za-z0-9+/=]+").unwrap());
// Long bare base64 runs only (>=200 chars), to avoid clobbering ordinary words/ids.
static BASE64_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"[A-Za-z0-9+/]{200,}={0,2}").unwrap());

fn placeholder(n: usize) -> String {
    format!("[base64 elided: {n} chars]")
}

fn scrub_base64_text(s: &str) -> String {
    let s1 = DATA_URI.replace_all(s, |c: &regex::Captures| placeholder(c[0].len()));
    BASE64_RUN
        .replace_all(&s1, |c: &regex::Captures| placeholder(c[0].len()))
        .into_owned()
}

fn scrub_base64_value(v: &mut Value) {
    match v {
        Value::String(s) => *s = scrub_base64_text(s),
        Value::Array(arr) => arr.iter_mut().for_each(scrub_base64_value),
        Value::Object(map) => map.values_mut().for_each(scrub_base64_value),
        _ => {}
    }
}

/// Fenced ```json … ``` block (DOTALL, non-greedy). The language tag is matched
/// case-insensitively; the captured group is the inner content to minify.
static JSON_FENCE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)```json\b[^\r\n]*\r?\n(.*?)```").unwrap());

/// Losslessly minify JSON embedded inside prose: fenced ```json blocks and balanced
/// top-level `{…}` / `[…]` spans. Each span is only rewritten if it parses as valid JSON
/// and the minified form is strictly shorter — surrounding prose is never touched. With
/// serde_json's `arbitrary_precision`, numbers round-trip exactly, so this stays lossless.
fn minify_embedded_json(text: &str) -> String {
    // 1. Fenced ```json blocks: minify the inner JSON in place (keep the fence).
    let fenced = JSON_FENCE.replace_all(text, |caps: &regex::Captures| {
        let inner = &caps[1];
        match minify_json_str(inner) {
            Some(min) => format!("```json\n{min}\n```"),
            None => caps[0].to_string(),
        }
    });
    // 2. Balanced top-level {…}/[…] spans in the remaining prose (skipping code fences so
    //    `{` inside a ```rust block, or a json fence already handled above, is left alone).
    minify_balanced_spans(&fenced)
}

/// Minify a single JSON string if it parses and the result is strictly shorter; else `None`.
fn minify_json_str(s: &str) -> Option<String> {
    let value: Value = serde_json::from_str(s.trim()).ok()?;
    let min = serde_json::to_string(&value).ok()?;
    (min.len() < s.len()).then_some(min)
}

/// Per-attempt lookahead cap for the balanced-bracket scan. A span longer than this is
/// left unminified (rare; embedded JSON blobs that large dominate their request anyway).
const MAX_JSON_SCAN: usize = 256 * 1024;

/// Replace each balanced top-level `{…}`/`[…]` span that is valid JSON (and shrinks) with
/// its minified form, leaving prose and fenced code blocks untouched.
///
/// Pathology guard: a *truncated* JSON blob (open bracket, no close) makes every inner
/// bracket re-scan ahead and fail — O(n²) unguarded. Two bounds make the worst case
/// linear: each attempt scans at most [`MAX_JSON_SCAN`] bytes, and the total bytes spent
/// on *failed* attempts is budgeted at ~1× the input length, after which the remaining
/// text is copied verbatim (successful minifies advance past their span, so they are
/// amortized linear and never charged).
fn minify_balanced_spans(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    let mut failed_scan_budget = bytes.len().max(MAX_JSON_SCAN);
    while i < bytes.len() {
        // Skip fenced code blocks verbatim (``` … ```), so we don't parse code braces.
        if bytes[i..].starts_with(b"```") {
            let after = i + 3;
            let end = find_fence_close(bytes, after).unwrap_or(bytes.len());
            out.push_str(&text[i..end]);
            i = end;
            continue;
        }
        if (bytes[i] == b'{' || bytes[i] == b'[')
            && failed_scan_budget > 0
            && looks_like_json_open(bytes, i)
        {
            let limit = (i + MAX_JSON_SCAN).min(bytes.len());
            match balanced_json_end(&bytes[..limit], i) {
                Some(end) => {
                    if let Some(min) = minify_json_str(&text[i..end]) {
                        out.push_str(&min);
                        i = end;
                        continue;
                    }
                    // Balanced but not valid/smaller JSON: charge the wasted scan.
                    failed_scan_budget = failed_scan_budget.saturating_sub(end - i);
                }
                // Never balanced within the cap: charge the full lookahead.
                None => failed_scan_budget = failed_scan_budget.saturating_sub(limit - i),
            }
        }
        // Ordinary prose char: copy one whole UTF-8 char (structural bytes are ASCII, so the
        // branches above always land on char boundaries).
        let ch_len = text[i..].chars().next().map_or(1, char::len_utf8);
        out.push_str(&text[i..i + ch_len]);
        i += ch_len;
    }
    out
}

/// Byte index just past the closing ``` ``` `` starting the search at `from`, or `None`.
fn find_fence_close(bytes: &[u8], from: usize) -> Option<usize> {
    let mut j = from;
    while j + 3 <= bytes.len() {
        if bytes[j..].starts_with(b"```") {
            return Some(j + 3);
        }
        j += 1;
    }
    None
}

/// Cheap pre-check that the bracket at `open` plausibly begins JSON, so we don't run the
/// balanced scan (and a parse) on every `{` in prose like "the {placeholder} field". For an
/// object: the next non-space byte is `"` (a key) or `}` (empty). For an array: a JSON value
/// start. Avoids O(n²) rescans on brace-dense non-JSON prose.
fn looks_like_json_open(bytes: &[u8], open: usize) -> bool {
    let mut j = open + 1;
    while j < bytes.len() && (bytes[j] as char).is_whitespace() {
        j += 1;
    }
    let Some(&next) = bytes.get(j) else {
        return false;
    };
    if bytes[open] == b'{' {
        next == b'"' || next == b'}'
    } else {
        matches!(
            next,
            b'"' | b'{' | b'[' | b']' | b'-' | b't' | b'f' | b'n' | b'0'..=b'9'
        )
    }
}

/// Byte index one past the JSON value's matching close bracket starting at `start` (a `{`
/// or `[`), tracking nesting and string/escape state so a bracket inside a string doesn't
/// count. `None` if the brackets never balance (then it isn't a JSON span). Only ASCII
/// structural bytes are inspected, so returned indices are valid char boundaries.
fn balanced_json_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'"' => in_str = true,
                b'{' | b'[' => depth += 1,
                b'}' | b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                    if depth < 0 {
                        return None; // unbalanced close before open
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Tokenizer-aware text normalization (opt-in, meaning-preserving, not byte-reversible).
/// Three deterministic, language-agnostic passes:
///   1. fold no-break / exotic spaces to a plain space,
///   2. drop unambiguous invisible waste — but NOT ZWJ/ZWNJ (U+200D/200C), which carry
///      meaning in emoji sequences and Arabic/Persian/Indic scripts (keep it universal),
///   3. NFKC: fold compatibility characters to canonical form (ﬁ→fi, full-width→ASCII,
///      ②→2) — several of these cost 2–3 BPE tokens each.
///
/// Biggest wins on web/PDF-pasted and non-ASCII text; ~no-op on clean ASCII (gate drops it).
fn normalize_text(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    s.chars()
        .filter_map(|c| match c {
            // no-break / thin / figure spaces → plain space (meaning-preserving)
            '\u{00A0}' | '\u{2007}' | '\u{2009}' | '\u{200A}' | '\u{202F}' => Some(' '),
            // invisible waste: ZWSP, BOM/ZWNBSP, word joiner, soft hyphen
            '\u{200B}' | '\u{FEFF}' | '\u{2060}' | '\u{00AD}' => None,
            other => Some(other),
        })
        .nfkc()
        .collect()
}

fn normalize_value(v: &mut Value) {
    match v {
        Value::String(s) => *s = normalize_text(s),
        Value::Array(arr) => arr.iter_mut().for_each(normalize_value),
        Value::Object(map) => map.values_mut().for_each(normalize_value),
        _ => {}
    }
}

/// Round a float to `sig` significant figures.
fn round_sig(x: f64, sig: u32) -> f64 {
    if x == 0.0 || !x.is_finite() || sig == 0 {
        return x;
    }
    let d = sig as i32 - 1 - x.abs().log10().floor() as i32;
    let f = 10f64.powi(d);
    (x * f).round() / f
}

/// Recursively round float numbers in a JSON value to `sig` significant figures
/// (lossy). Integers are left exact.
fn quantize_value(v: &mut Value, sig: u32) {
    match v {
        Value::Number(n) if !n.is_i64() && !n.is_u64() => {
            // A raw repr with no '.'/'e' is a plain integer that merely overflows u64
            // (arbitrary_precision) — leave exact per the doc contract.
            let raw = n.to_string();
            if (raw.contains('.') || raw.contains('e') || raw.contains('E'))
                && let Some(x) = n.as_f64()
                && let Some(r) = serde_json::Number::from_f64(round_sig(x, sig))
            {
                *n = r;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|e| quantize_value(e, sig)),
        Value::Object(m) => m.values_mut().for_each(|e| quantize_value(e, sig)),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ProviderKind;
    use crate::pipeline;
    use crate::provider::OpenAiProvider;
    use crate::tokenizer::counter_for;
    use serde_json::json;

    fn run(body: Value, stage: HygieneStage) -> (Request, bool) {
        let mut req = Request::from_value(ProviderKind::OpenAi, body);
        let counter = counter_for(ProviderKind::OpenAi, Some("gpt-4o")).unwrap();
        let stages: Vec<Box<dyn Transform>> = vec![Box::new(stage)];
        let out = pipeline::run(&mut req, &OpenAiProvider, counter.as_ref(), &stages);
        let applied = out.stages[0].applied;
        (req, applied)
    }

    #[test]
    fn minifies_pretty_json_content_losslessly() {
        let pretty =
            serde_json::to_string_pretty(&json!({"a":1,"b":[1,2,3],"c":{"d":true}})).unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":pretty}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(applied, "minify should reduce tokens");
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(!now.contains("\n  "), "pretty whitespace removed");
        let before: Value = json!({"a":1,"b":[1,2,3],"c":{"d":true}});
        assert_eq!(serde_json::from_str::<Value>(now).unwrap(), before);
    }

    #[test]
    fn minifies_fenced_json_block_leaving_prose() {
        let pretty = "{\n  \"a\": 1,\n  \"b\": [\n    1,\n    2\n  ]\n}";
        let content = format!("Here is the response:\n```json\n{pretty}\n```\nThanks!");
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(applied, "embedded fenced JSON minifies → fewer tokens");
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(
            now.contains("Here is the response:"),
            "leading prose untouched"
        );
        assert!(now.contains("Thanks!"), "trailing prose untouched");
        assert!(
            now.contains(r#"{"a":1,"b":[1,2]}"#),
            "JSON minified losslessly: {now}"
        );
        assert!(!now.contains("\n  \"a\""), "pretty whitespace gone");
    }

    #[test]
    fn minifies_inline_json_span_leaving_prose() {
        // An inline balanced {…} span inside a sentence: minify only the JSON.
        let content = "The server returned { \"ok\" : true , \"count\" : 3 } as the body.";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(applied, "inline JSON span minifies");
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(
            now.starts_with("The server returned "),
            "prose before span intact"
        );
        assert!(now.ends_with(" as the body."), "prose after span intact");
        assert!(
            now.contains(r#"{"ok":true,"count":3}"#),
            "span minified: {now}"
        );
    }

    #[test]
    fn invalid_json_braces_are_left_alone() {
        // Braces that are NOT valid JSON (a code snippet / placeholder) must pass through
        // untouched — never corrupt surrounding prose.
        let content = "Use the { placeholder } token and call foo() { return 1; } please.";
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(!applied, "no valid JSON span → nothing to minify");
        assert_eq!(
            req.get_str("/messages/0/content").unwrap(),
            content,
            "invalid-JSON braces left exactly verbatim"
        );
    }

    #[test]
    fn embedded_json_scanner_handles_braces_inside_strings() {
        // A string value containing `}`, escaped `"`, and `[` must not close the span early
        // or corrupt the surrounding prose.
        let content = r#"prefix {"k": "a}b\"c[d", "n": 2} suffix"#;
        let out = minify_embedded_json(content);
        assert!(out.starts_with("prefix "), "prose before intact: {out}");
        assert!(out.ends_with(" suffix"), "prose after intact: {out}");
        assert!(
            out.contains(r#""a}b\"c[d""#),
            "string-with-braces preserved exactly: {out}"
        );
        // Re-parse the minified span to prove it's still valid JSON (lossless value).
        let lo = out.find('{').unwrap();
        let hi = out.rfind('}').unwrap();
        let v: Value = serde_json::from_str(&out[lo..=hi]).expect("still valid JSON");
        assert_eq!(v["k"], json!("a}b\"c[d"));
        assert_eq!(v["n"], json!(2));
    }

    #[test]
    fn embedded_json_minify_is_lossless_on_numbers() {
        // arbitrary_precision: a big/precise number must round-trip exactly through minify.
        let content = "data: {\"x\": 123456789012345678, \"y\": 0.1234567890123456789}";
        let out = minify_embedded_json(content);
        assert!(out.contains("123456789012345678"), "large int exact: {out}");
        assert!(
            out.contains("0.1234567890123456789"),
            "long decimal exact: {out}"
        );
    }

    #[test]
    fn leaves_prose_untouched_without_base64() {
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":"just some prose, not json"}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(!applied, "prose is not JSON; nothing to minify");
        assert_eq!(
            req.get_str("/messages/0/content"),
            Some("just some prose, not json")
        );
    }

    #[test]
    fn strips_base64_in_prose_when_enabled() {
        let blob = "A".repeat(300);
        let content = format!("attached blob {blob} thanks");
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: true,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(applied);
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(now.contains("base64 elided"));
        assert!(!now.contains(&"A".repeat(300)));
    }

    #[test]
    fn strips_base64_inside_json_string_values() {
        let blob = "Zm9v".repeat(60); // 240 base64 chars
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content": serde_json::to_string(&json!({"img": blob, "name": "ok"})).unwrap()}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: true,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(applied);
        let now = req.get_str("/messages/0/content").unwrap();
        let v: Value = serde_json::from_str(now).expect("still valid JSON");
        assert_eq!(v.get("name").and_then(Value::as_str), Some("ok"));
        assert!(
            v.get("img")
                .and_then(Value::as_str)
                .unwrap()
                .contains("elided"),
            "base64 value elided, JSON structure intact"
        );
    }

    #[test]
    fn base64_not_stripped_when_disabled() {
        let content = format!("blob {} end", "A".repeat(300));
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":content.clone()}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: false,
            },
        );
        assert!(!applied, "disabled => prose untouched");
        assert_eq!(req.get_str("/messages/0/content"), Some(content.as_str()));
    }

    #[test]
    fn quantizes_floats_when_enabled() {
        let inner =
            serde_json::to_string(&json!({"a":8.123456789,"b":2,"c":123.456789,"name":"keep"}))
                .unwrap();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":inner}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: Some(3),
                normalize_unicode: false,
            },
        );
        assert!(applied, "rounding long floats cuts tokens");
        let v: Value = serde_json::from_str(req.get_str("/messages/0/content").unwrap()).unwrap();
        assert_eq!(v["a"], json!(8.12));
        assert_eq!(v["b"], json!(2), "integers untouched");
        assert_eq!(v["name"], json!("keep"), "strings untouched");
    }

    #[test]
    fn normalize_text_strips_waste_folds_nfkc_keeps_joiners() {
        // BOM + ZWSP waste, NBSP, full-width "ＡＢ", ligature "ﬁ", and a ZWJ emoji seq.
        let out = normalize_text("\u{FEFF}ＡＢ\u{200B}x\u{00A0}y ﬁ 👨\u{200D}👩");
        assert!(!out.contains('\u{FEFF}'), "BOM stripped");
        assert!(!out.contains('\u{200B}'), "zero-width space stripped");
        assert!(out.contains("AB"), "full-width folded to ASCII");
        assert!(out.contains("x y"), "NBSP folded to a plain space");
        assert!(out.contains("fi"), "ﬁ ligature folded (NFKC)");
        assert!(
            out.contains('\u{200D}'),
            "ZWJ preserved — emoji/script meaning kept (universality)"
        );
    }

    #[test]
    fn normalize_unicode_wired_through_stage_for_prose() {
        let messy = "ＣＰＵ load high\u{200B}\u{200B}\u{200B}".to_string();
        let body = json!({"model":"gpt-4o","messages":[{"role":"user","content":messy}]});
        let (req, applied) = run(
            body,
            HygieneStage {
                strip_base64: false,
                sig_figs: None,
                normalize_unicode: true,
            },
        );
        assert!(applied, "folding full-width + dropping ZWSP cuts tokens");
        let now = req.get_str("/messages/0/content").unwrap();
        assert!(now.starts_with("CPU") && !now.contains('\u{200B}'));
    }

    #[test]
    fn quantize_preserves_large_integers() {
        let mut v = serde_json::from_str::<Value>(r#"{"id": 12345678901234567890}"#).unwrap();
        quantize_value(&mut v, 4);
        // Must not have been rounded to scientific notation
        assert_eq!(v["id"].to_string(), "12345678901234567890");
    }
}
