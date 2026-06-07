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
        for ptr in provider.content_text_pointers(req) {
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
            } else if self.strip_base64 || self.normalize_unicode {
                // Non-JSON prose: optional base64 scrub + unicode normalization.
                let mut text = s.clone();
                if self.strip_base64 {
                    text = scrub_base64_text(&text);
                }
                if self.normalize_unicode {
                    text = normalize_text(&text);
                }
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
            if let Some(x) = n.as_f64()
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
}
