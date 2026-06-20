//! Shared request-compression logic for llmtrim's proxy-wasm gateway plugins.
//!
//! Kong and Higress both buffer the LLM request body, hand it here, and replace the body
//! with what comes back. The only host-specific part (how the body is buffered and swapped)
//! lives in each gateway crate; the compression call and its rules live once, here.
//!
//! # Fail-open
//!
//! A gateway sits in the live request path, so a bug here must never break a user's request.
//! [`compress_body`] therefore never errors: on an oversized body, non-UTF-8 input, non-JSON
//! input, an undetectable provider, or any engine error, it returns the original bytes
//! unchanged and records why in [`Outcome::passthrough`]. The caller forwards
//! [`Outcome::body`] either way.

use llmtrim_core::ir::ProviderKind;

/// The default body-size ceiling: above this, the gateway forwards the request uncompressed
/// rather than parsing it. Bounds wasm linear memory, since `serde_json` builds a heap tree a
/// few times the raw size and an OOM in the wasm instance becomes a 500, not a passthrough.
pub const DEFAULT_MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

/// Per-route configuration a gateway reads from its own plugin config and passes in.
#[derive(Debug, Clone)]
pub struct Config {
    /// Target API shape (`openai` / `anthropic` / `google`), or `None` to auto-detect from
    /// the body. A gateway that fronts one provider should set this to skip detection.
    pub provider: Option<String>,
    /// Workload preset, or `None` to use the `auto` shape-routing default (the recommended
    /// setting for a gateway seeing mixed traffic).
    pub preset: Option<String>,
    /// Skip (pass through) bodies larger than this, to bound wasm memory. `None` disables the
    /// guard. Defaults to [`DEFAULT_MAX_BODY_BYTES`].
    pub max_body_bytes: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            provider: None,
            preset: None,
            max_body_bytes: Some(DEFAULT_MAX_BODY_BYTES),
        }
    }
}

#[derive(serde::Deserialize)]
struct RawConfig {
    provider: Option<String>,
    preset: Option<String>,
    max_body_bytes: Option<usize>,
}

impl Config {
    /// Build a [`Config`] from a gateway's JSON plugin-config bytes (`{"provider":…,
    /// "preset":…, "max_body_bytes":…}`, all optional). Fail-open: empty or malformed config
    /// yields the default (auto routing, default size limit) so a misconfigured plugin still
    /// runs and compresses rather than refusing traffic. An absent `max_body_bytes` keeps the
    /// default guard; a positive value sets it; `0` disables it (compress any size).
    pub fn from_json_bytes(bytes: &[u8]) -> Self {
        // Empty or whitespace-only config means "use defaults", not a parse error.
        if bytes.iter().all(u8::is_ascii_whitespace) {
            return Self::default();
        }
        match serde_json::from_slice::<RawConfig>(bytes) {
            Ok(raw) => Self {
                provider: raw.provider,
                preset: raw.preset,
                // Absent -> default guard; 0 -> disabled (None); positive -> that limit. This
                // is the only way an operator can opt out of the guard from JSON config.
                max_body_bytes: match raw.max_body_bytes {
                    None => Some(DEFAULT_MAX_BODY_BYTES),
                    Some(0) => None,
                    Some(n) => Some(n),
                },
            },
            Err(_) => Self::default(),
        }
    }
}

/// The result of a compression attempt. `body` is always present (original bytes on
/// passthrough); `passthrough` names the reason the engine was skipped, if any.
#[derive(Debug)]
pub struct Outcome {
    /// The body to forward: compressed on success, the untouched original on passthrough.
    pub body: Vec<u8>,
    /// `Some(reason)` when the engine was skipped and the original body is returned as-is.
    pub passthrough: Option<Passthrough>,
    /// Input token count before compression. Always `None` when [`Outcome::passthrough`] is
    /// `Some` (the engine did not run); an estimate, not exact, in wasm builds (no tiktoken).
    pub tokens_before: Option<u64>,
    /// Input token count after compression. Always `None` on passthrough.
    pub tokens_after: Option<u64>,
}

impl Outcome {
    /// Whether the engine was skipped and the original body forwarded unchanged. When `true`,
    /// the token counts are `None`; a metrics caller should record a zero-savings event.
    pub fn is_passthrough(&self) -> bool {
        self.passthrough.is_some()
    }
}

/// Why a request was forwarded uncompressed.
#[derive(Debug, PartialEq, Eq)]
pub enum Passthrough {
    /// The body exceeded [`Config::max_body_bytes`]; it was forwarded without parsing.
    TooLarge { len: usize, limit: usize },
    /// The body was not valid UTF-8 (so not a JSON request body we can read).
    NotUtf8,
    /// The configured preset name is unknown, or the engine rejected the request (invalid
    /// JSON, undetectable provider). The string is a structured engine message with no request
    /// body content (only static text, the preset name, or a serde structural position), so a
    /// host may log it verbatim. A test pins this; keep it true if engine errors change.
    Engine(String),
}

/// Compress a buffered request body. Never errors; see the module-level fail-open note.
///
/// Calling this twice on the same body is safe: the engine is lossless and deterministic, so a
/// second pass over an already-compressed body returns valid JSON without corrupting it (it may
/// simply find little left to compress). A gateway with a retry path need not guard against it.
pub fn compress_body(body: &[u8], config: &Config) -> Outcome {
    if let Some(limit) = config.max_body_bytes
        && body.len() > limit
    {
        return passthrough(
            body,
            Passthrough::TooLarge {
                len: body.len(),
                limit,
            },
        );
    }

    let Ok(input) = std::str::from_utf8(body) else {
        return passthrough(body, Passthrough::NotUtf8);
    };

    // An unparseable provider hint falls back to auto-detect rather than failing the request.
    let provider = config
        .provider
        .as_deref()
        .and_then(|p| p.parse::<ProviderKind>().ok());

    match llmtrim_core::rewrite_request(input, provider, config.preset.as_deref()) {
        Ok(result) => Outcome {
            body: result.request_json.into_bytes(),
            passthrough: None,
            tokens_before: Some(result.input_tokens_before.0 as u64),
            tokens_after: Some(result.input_tokens_after.0 as u64),
        },
        Err(e) => passthrough(body, Passthrough::Engine(format!("{e:#}"))),
    }
}

fn passthrough(body: &[u8], reason: Passthrough) -> Outcome {
    Outcome {
        body: body.to_vec(),
        passthrough: Some(reason),
        tokens_before: None,
        tokens_after: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A top-level `system` field makes this unambiguously Anthropic-shaped, so auto-detect
    // resolves it (a bare `{model, messages, max_tokens}` is ambiguous with OpenAI and
    // correctly fails detection; see `non_distinguishable_body_passes_through`).
    fn anthropic_agent_body() -> Vec<u8> {
        serde_json::json!({
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are a coding agent.",
            "messages": [{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1",
                "content": "ERROR boom\n".repeat(60)}]}],
            "max_tokens": 1024,
        })
        .to_string()
        .into_bytes()
    }

    #[test]
    fn compresses_a_tool_heavy_body_and_returns_valid_json() {
        let out = compress_body(&anthropic_agent_body(), &Config::default());
        assert!(
            out.passthrough.is_none(),
            "should not passthrough a valid body: {:?}",
            out.passthrough
        );
        assert!(out.tokens_after.unwrap() < out.tokens_before.unwrap());
        // The forwarded body must still parse as JSON for the upstream provider.
        let v: serde_json::Value = serde_json::from_slice(&out.body).expect("valid JSON out");
        assert_eq!(v["model"], "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn explicit_provider_hint_is_honored() {
        let cfg = Config {
            provider: Some("anthropic".into()),
            ..Config::default()
        };
        let out = compress_body(&anthropic_agent_body(), &cfg);
        assert!(out.passthrough.is_none());
    }

    #[test]
    fn unparseable_provider_hint_falls_back_to_autodetect() {
        // A bad hint must not fail the request; the body still auto-detects and compresses.
        let cfg = Config {
            provider: Some("not-a-provider".into()),
            ..Config::default()
        };
        let out = compress_body(&anthropic_agent_body(), &cfg);
        assert!(
            out.passthrough.is_none(),
            "bad provider hint must fall back, not fail"
        );
    }

    #[test]
    fn non_utf8_body_passes_through_unchanged() {
        let body = vec![0xff, 0xfe, 0x00, 0x01];
        let out = compress_body(&body, &Config::default());
        assert_eq!(out.passthrough, Some(Passthrough::NotUtf8));
        assert_eq!(out.body, body, "original bytes forwarded verbatim");
    }

    #[test]
    fn non_json_body_passes_through_unchanged() {
        let body = b"this is not json".to_vec();
        let out = compress_body(&body, &Config::default());
        assert!(matches!(out.passthrough, Some(Passthrough::Engine(_))));
        assert_eq!(out.body, body);
    }

    #[test]
    fn non_distinguishable_body_passes_through() {
        // `{model, messages, max_tokens}` is shared by OpenAI and Anthropic, so with no
        // provider configured the engine cannot detect a shape and the gateway forwards the
        // body unchanged rather than guessing. A gateway fronting one provider sets `provider`.
        let body = serde_json::json!({
            "model": "x",
            "messages": [{"role":"user","content":"hi"}],
            "max_tokens": 16,
        })
        .to_string()
        .into_bytes();
        let out = compress_body(&body, &Config::default());
        assert!(matches!(out.passthrough, Some(Passthrough::Engine(_))));
        assert_eq!(out.body, body);
    }

    #[test]
    fn empty_and_structural_json_bodies_pass_through() {
        // Real gateway traffic: empty health-probe bodies, array-shaped batches, bare objects.
        // None panic; all forward unchanged because there is no provider shape to detect.
        for body in [&b""[..], b"[]", b"{}", b"   "] {
            let out = compress_body(body, &Config::default());
            assert!(
                matches!(out.passthrough, Some(Passthrough::Engine(_))),
                "body {body:?} should pass through via the engine arm"
            );
            assert_eq!(out.body, body);
        }
    }

    #[test]
    fn oversized_body_passes_through_without_parsing() {
        let cfg = Config {
            max_body_bytes: Some(64),
            ..Config::default()
        };
        let big = vec![b'x'; 65];
        let out = compress_body(&big, &cfg);
        assert_eq!(
            out.passthrough,
            Some(Passthrough::TooLarge { len: 65, limit: 64 })
        );
        assert_eq!(out.body, big, "oversized body forwarded verbatim");
        assert!(out.is_passthrough());
    }

    #[test]
    fn double_compress_does_not_corrupt() {
        let body = anthropic_agent_body();
        let first = compress_body(&body, &Config::default());
        let second = compress_body(&first.body, &Config::default());
        let v: serde_json::Value = serde_json::from_slice(&second.body).expect("still valid JSON");
        assert_eq!(v["model"], "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn engine_passthrough_message_carries_no_request_content() {
        // The PII contract on Passthrough::Engine: a non-JSON body's reason must not echo the
        // body bytes back (a host may log this string).
        let body = b"SECRET_PROMPT_TEXT not json".to_vec();
        let out = compress_body(&body, &Config::default());
        let Some(Passthrough::Engine(msg)) = out.passthrough else {
            panic!("expected engine passthrough");
        };
        assert!(
            !msg.contains("SECRET_PROMPT_TEXT"),
            "engine message must not leak request content: {msg}"
        );
    }

    #[test]
    fn config_from_json_parses_all_fields() {
        let cfg = Config::from_json_bytes(
            br#"{"provider":"anthropic","preset":"agent","max_body_bytes":1024}"#,
        );
        assert_eq!(cfg.provider.as_deref(), Some("anthropic"));
        assert_eq!(cfg.preset.as_deref(), Some("agent"));
        assert_eq!(cfg.max_body_bytes, Some(1024));
    }

    #[test]
    fn config_from_json_fills_defaults_for_missing_fields() {
        let cfg = Config::from_json_bytes(br#"{"provider":"openai"}"#);
        assert_eq!(cfg.provider.as_deref(), Some("openai"));
        assert_eq!(
            cfg.preset, None,
            "absent preset stays None (rewrite_request applies auto)"
        );
        assert_eq!(
            cfg.max_body_bytes,
            Some(DEFAULT_MAX_BODY_BYTES),
            "absent size limit keeps the default guard"
        );
    }

    #[test]
    fn config_from_json_zero_max_body_disables_the_guard() {
        let cfg = Config::from_json_bytes(br#"{"max_body_bytes":0}"#);
        assert_eq!(cfg.max_body_bytes, None, "0 means unlimited");
        // And a disabled guard actually lets a large body through to the engine.
        let out = compress_body(&vec![b'x'; DEFAULT_MAX_BODY_BYTES + 1], &cfg);
        assert!(
            !matches!(out.passthrough, Some(Passthrough::TooLarge { .. })),
            "guard disabled, so size is not the passthrough reason"
        );
    }

    #[test]
    fn config_from_json_is_fail_open_on_empty_or_malformed() {
        // Empty, whitespace, and garbage all fall back to the safe default rather than
        // disabling the plugin.
        for raw in [&b""[..], b"   ", b"not json", b"{", br#"{"provider":42}"#] {
            let cfg = Config::from_json_bytes(raw);
            assert_eq!(cfg.provider, None, "raw {raw:?} should default");
            assert_eq!(cfg.max_body_bytes, Some(DEFAULT_MAX_BODY_BYTES));
        }
    }

    #[test]
    fn unknown_preset_passes_through_rather_than_failing() {
        let cfg = Config {
            preset: Some("nonsense".into()),
            ..Config::default()
        };
        let out = compress_body(&anthropic_agent_body(), &cfg);
        assert!(matches!(out.passthrough, Some(Passthrough::Engine(_))));
        assert_eq!(out.body, anthropic_agent_body());
    }
}
