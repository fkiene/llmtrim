//! Transport — send the compressed request to the LLM API and return the response.
//!
//! Blocking `ureq` (no async — a request/response round-trip needs no
//! concurrency). The pure parts (`url`, `headers`) are unit-tested; the live
//! `send` needs an API key + network, so it isn't exercised in the test suite.
//!
//! # Upstream proxy (`LLMTRIM_UPSTREAM_PROXY`)
//!
//! By default every outbound call sets `.proxy(None)` to suppress `HTTPS_PROXY` — without
//! this, a shell with `HTTPS_PROXY=127.0.0.1:PORT` pointing at the llmtrim daemon would
//! loop back through itself. If you need the daemon's own traffic to exit through a
//! corporate or gateway proxy, set `LLMTRIM_UPSTREAM_PROXY=http://host:port` (or
//! `http://user:pass@host:port`) in the daemon's launch environment **before** the daemon
//! starts. Changing it while running has no effect.
//!
//! The env var is honoured by: `Endpoint::send` (CLI `send` command),
//! `forward_post` (replay path), and the primary hudsucker MITM path
//! (via a `hyper-http-proxy` `ProxyConnector` wrapping hudsucker's outbound connector).
//! `forward_get` exists but currently has no call site — GET requests pass through
//! hudsucker's ProxyConnector directly and do not route through this function.
//!
//! **Security notes**: credentials in the URL are plaintext in the daemon's environment
//! (`/proc/<pid>/environ` on Linux); use a credential-free URL and OS-level auth where
//! possible. Only upstreams that point at llmtrim's own listen address (same loopback host
//! AND same port) are rejected — this prevents infinite recursion while allowing a companion
//! proxy like headroom running on a different loopback port (e.g. `127.0.0.1:9999`) to be
//! used as the upstream. `localhost`, `127.0.0.1`, `::1`, and `[::1]` are treated as
//! equivalent when comparing the host, so `localhost:8788` is caught if the daemon binds
//! `127.0.0.1:8788`.

use std::time::Duration;

use anyhow::{Context, Result};

use llmtrim_core::ir::ProviderKind;

/// Redact the `user:pass@` userinfo from a proxy URL before logging or surfacing in errors.
/// `http://alice:s3cr3t@proxy.corp:3128` → `http://proxy.corp:3128`.
/// URLs without userinfo are returned unchanged.
pub fn redact_proxy_url(url: &str) -> String {
    // Find `://` then look for `@` before the next `/` or end. Strip everything between.
    if let Some(scheme_end) = url.find("://") {
        let after_scheme = &url[scheme_end + 3..];
        // Only strip if there is a `@` before the path separator.
        let path_start = after_scheme.find('/').unwrap_or(after_scheme.len());
        let host_part = &after_scheme[..path_start];
        if let Some(at_pos) = host_part.rfind('@') {
            let scheme = &url[..scheme_end + 3];
            let rest = &after_scheme[at_pos + 1..];
            return format!("{scheme}{rest}");
        }
    }
    url.to_string()
}

/// Parse and validate `LLMTRIM_UPSTREAM_PROXY` once at startup. Returns `None` if the var is
/// unset, `Err` if it is set but invalid (points at llmtrim's own listen address, or
/// unparseable). The `ureq::Proxy` is `Clone` only in newer ureq; we keep a `String` and
/// reconstruct per-call (ureq 3 proxy construction is cheap — it just validates the URL).
///
/// `bind_addr` is llmtrim's own listen address (e.g. `127.0.0.1:8788`). When `Some`, an
/// upstream that resolves to the same loopback host AND the same port is rejected. Pass
/// `None` when the daemon bind address is unknown (e.g. the CLI `send` subcommand, which
/// does not run a proxy server).
pub fn upstream_proxy_url(bind_addr: Option<std::net::SocketAddr>) -> Result<Option<String>> {
    let url = match std::env::var("LLMTRIM_UPSTREAM_PROXY") {
        Ok(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };
    validate_proxy_url(&url, bind_addr)?;
    Ok(Some(url))
}

/// Returns true if `host` (already extracted, without brackets) is any loopback alias.
/// Note: only `127.0.0.1` is checked explicitly; `127.x.x.x` beyond `127.0.0.1` is also
/// technically loopback on Linux, but those addresses are an unguarded edge case here —
/// the risk is low because corporate proxies never live on loopback ranges other than
/// `127.0.0.1`, so a false-negative means a mis-typed proxy URL, not an exploit path.
fn is_loopback_host(host: &str) -> bool {
    host.eq_ignore_ascii_case("localhost") || host == "127.0.0.1" || host == "::1"
}

/// Reject upstreams that point at llmtrim's own listen address (same loopback host + same
/// port), which would create infinite recursion. A companion proxy on a different loopback
/// port is explicitly allowed.
fn validate_proxy_url(url: &str, bind_addr: Option<std::net::SocketAddr>) -> Result<()> {
    // Extract the host portion (between `://` and the first `:`, `/`, or end).
    let host = proxy_host(url).with_context(|| {
        format!(
            "LLMTRIM_UPSTREAM_PROXY: cannot parse host from `{}`",
            redact_proxy_url(url)
        )
    })?;

    // The raw host:port segment (before path), lowercased — needed to detect `[::1]:port`.
    let raw_host_segment = {
        let after_scheme = url.split_once("://").map(|(_, r)| r).unwrap_or("");
        let without_userinfo = after_scheme
            .rfind('@')
            .map(|i| &after_scheme[i + 1..])
            .unwrap_or(after_scheme);
        without_userinfo
            .split(['/', '?'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
    };

    // Resolve whether the upstream host is a loopback alias, handling bracketed IPv6.
    let upstream_is_loopback = is_loopback_host(host)
        || raw_host_segment == "[::1]"
        || raw_host_segment.starts_with("[::1]:");

    if upstream_is_loopback {
        // Extract the upstream port from the raw segment (e.g. "127.0.0.1:9999" → 9999).
        // `rsplit(':').next()` yields the last colon-separated token, which is the port
        // for both plain IPv4 (`127.0.0.1:9999` → `"9999"`) and bracketed IPv6
        // (`[::1]:9999` → `"9999"`). For bare `[::1]` with no port it yields `"[::1]"`,
        // which `parse::<u16>()` rejects, leaving `upstream_port` as `None`.
        let upstream_port: Option<u16> = raw_host_segment
            .rsplit(':')
            .next()
            .and_then(|p| p.parse().ok());

        if let (Some(bind), Some(up_port)) = (bind_addr, upstream_port) {
            // Only reject if the port matches llmtrim's own bind port. A different port on
            // the same loopback interface is a legitimate companion proxy (e.g. headroom).
            if up_port == bind.port() {
                anyhow::bail!(
                    "LLMTRIM_UPSTREAM_PROXY points at llmtrim's own listen address \
                     (`{host}:{up_port}`), which would cause infinite recursion. \
                     Use a different port or a non-loopback proxy address."
                );
            }
        } else if bind_addr.is_none() {
            // No bind address supplied (e.g. CLI `send` command) — loopback is allowed
            // since there is no daemon to recurse through.
        } else {
            // Bind address known but no port parsed from upstream URL (bare host, no port).
            // Reject conservatively: a loopback URL with no explicit port likely defaults
            // to 80 or is malformed; either way it cannot be a useful corporate proxy.
            anyhow::bail!(
                "LLMTRIM_UPSTREAM_PROXY points at a loopback address (`{host}`) without an \
                 explicit port — cannot verify it does not point at llmtrim itself. \
                 Use a non-loopback proxy address or include the port explicitly."
            );
        }
    }

    // Ask ureq to validate the full URL (scheme, port, etc.).
    ureq::Proxy::new(url).with_context(|| {
        format!(
            "LLMTRIM_UPSTREAM_PROXY: invalid proxy URL `{}`",
            redact_proxy_url(url)
        )
    })?;
    Ok(())
}

/// Extract just the host from a proxy URL like `http://user:pass@host:port/`.
fn proxy_host(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://")?.1;
    // Strip userinfo
    let without_userinfo = after_scheme
        .rfind('@')
        .map(|i| &after_scheme[i + 1..])
        .unwrap_or(after_scheme);
    // Strip port and path
    let host = without_userinfo.split([':', '/', '?']).next()?;
    (!host.is_empty()).then_some(host)
}

/// Build a `ureq::Proxy` from a URL string. Cheap; ureq 3 proxy construction just parses.
fn make_proxy(url: &str) -> Result<ureq::Proxy> {
    ureq::Proxy::new(url).with_context(|| {
        format!(
            "failed to construct upstream proxy `{}`",
            redact_proxy_url(url)
        )
    })
}

/// Hard ceiling on any single upstream round-trip. ureq has no default timeout, so a hung
/// upstream would pin a blocking thread forever (and on the replay path, leak the daemon's
/// connection pool). Generous enough for slow streamed generations.
const UPSTREAM_TIMEOUT: Duration = Duration::from_secs(600);

/// A configured provider endpoint.
pub struct Endpoint {
    pub provider: ProviderKind,
    pub base_url: String,
    pub api_key: String,
}

impl Endpoint {
    /// Build from the environment: `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` (required)
    /// and optional `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` overrides.
    pub fn from_env(provider: ProviderKind) -> Result<Self> {
        let (key_var, base_var, default_base) = match provider {
            ProviderKind::OpenAi => (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "https://api.openai.com",
            ),
            ProviderKind::Anthropic => (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "https://api.anthropic.com",
            ),
            ProviderKind::Google => (
                "GEMINI_API_KEY",
                "GEMINI_BASE_URL",
                "https://generativelanguage.googleapis.com",
            ),
        };
        let api_key = std::env::var(key_var).with_context(|| format!("{key_var} is not set"))?;
        let base_url = std::env::var(base_var).unwrap_or_else(|_| default_base.to_string());
        Ok(Self {
            provider,
            base_url,
            api_key,
        })
    }

    /// The chat/messages endpoint URL.
    pub fn url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        match self.provider {
            ProviderKind::OpenAi => format!("{base}/v1/chat/completions"),
            ProviderKind::Anthropic => format!("{base}/v1/messages"),
            // Gemini puts the model in the URL; the CLI `send` path has no model field, so
            // it targets a default model. The proxy preserves the client's real path.
            ProviderKind::Google => {
                format!("{base}/v1beta/models/gemini-2.0-flash:generateContent")
            }
        }
    }

    /// Provider auth + protocol headers. (`anthropic-version` is a required wire
    /// constant, not tunable data.)
    fn headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = vec![("content-type", "application/json".to_string())];
        match self.provider {
            ProviderKind::OpenAi => {
                headers.push(("authorization", format!("Bearer {}", self.api_key)));
            }
            ProviderKind::Anthropic => {
                headers.push(("x-api-key", self.api_key.clone()));
                headers.push(("anthropic-version", "2023-06-01".to_string()));
            }
            ProviderKind::Google => {
                headers.push(("x-goog-api-key", self.api_key.clone()));
            }
        }
        headers
    }

    /// POST the request body and return the raw response body. Blocking.
    pub fn send(&self, request_json: &str, proxy_url: Option<&str>) -> Result<String> {
        let url = self.url();
        // Suppress the ambient HTTPS_PROXY by default (it points at us — looping traffic back
        // through the daemon). If the caller supplies an explicit upstream proxy, use that instead.
        let proxy = match proxy_url {
            Some(u) => Some(make_proxy(u)?),
            None => None,
        };
        let mut request = ureq::post(&url)
            .config()
            .proxy(proxy)
            .timeout_global(Some(UPSTREAM_TIMEOUT))
            .build();
        for (name, value) in self.headers() {
            request = request.header(name, &value);
        }
        let mut response = request
            .send(request_json)
            .map_err(|e| anyhow::anyhow!("request to {url} failed: {e}"))?;
        response
            .body_mut()
            .read_to_string()
            .context("failed to read response body")
    }
}

/// A streamed upstream response for the reverse proxy: HTTP status, the upstream
/// content-type (echoed so SSE clients see `text/event-stream`), and a reader over the
/// body that yields bytes as they arrive — so streaming responses pass straight through.
pub struct Upstream {
    pub status: u16,
    pub content_type: Option<String>,
    pub reader: Box<dyn std::io::Read>,
}

/// Forward a POST of `body` to `url`, passing `headers` (the client's auth etc.) through
/// verbatim, and return the upstream response as a stream. Upstream HTTP errors (4xx/5xx)
/// are RELAYED, not turned into transport errors — a proxy must hand the client the
/// provider's real status and message.
pub fn forward_post(
    url: &str,
    headers: &[(String, String)],
    body: &str,
    proxy_url: Option<&str>,
) -> Result<Upstream> {
    // Suppress ambient HTTPS_PROXY (points at us) unless an explicit upstream proxy is set.
    let proxy = match proxy_url {
        Some(u) => Some(make_proxy(u)?),
        None => None,
    };
    let mut req = ureq::post(url)
        .config()
        .http_status_as_error(false)
        .proxy(proxy)
        .timeout_global(Some(UPSTREAM_TIMEOUT))
        .build();
    for (k, v) in headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let response = req
        .send(body)
        .map_err(|e| anyhow::anyhow!("upstream POST {url} failed: {e}"))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let reader = response.into_body().into_reader();
    Ok(Upstream {
        status,
        content_type,
        reader: Box::new(reader),
    })
}

/// Forward a GET (e.g. a client's `/v1/models` probe) — passthrough, no body, streamed.
pub fn forward_get(
    url: &str,
    headers: &[(String, String)],
    proxy_url: Option<&str>,
) -> Result<Upstream> {
    let proxy = match proxy_url {
        Some(u) => Some(make_proxy(u)?),
        None => None,
    };
    let mut req = ureq::get(url)
        .config()
        .http_status_as_error(false)
        .proxy(proxy)
        .timeout_global(Some(UPSTREAM_TIMEOUT))
        .build();
    for (k, v) in headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let response = req
        .call()
        .map_err(|e| anyhow::anyhow!("upstream GET {url} failed: {e}"))?;
    let status = response.status().as_u16();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let reader = response.into_body().into_reader();
    Ok(Upstream {
        status,
        content_type,
        reader: Box::new(reader),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(provider: ProviderKind) -> Endpoint {
        Endpoint {
            provider,
            base_url: "https://example.test/".to_string(), // trailing slash trimmed
            api_key: "secret".to_string(),
        }
    }

    #[test]
    fn openai_url_and_headers() {
        let e = endpoint(ProviderKind::OpenAi);
        assert_eq!(e.url(), "https://example.test/v1/chat/completions");
        let h = e.headers();
        assert!(
            h.iter()
                .any(|(k, v)| *k == "authorization" && v == "Bearer secret")
        );
        assert!(
            h.iter()
                .any(|(k, v)| *k == "content-type" && v == "application/json")
        );
    }

    #[test]
    fn anthropic_url_and_headers() {
        let e = endpoint(ProviderKind::Anthropic);
        assert_eq!(e.url(), "https://example.test/v1/messages");
        let h = e.headers();
        assert!(h.iter().any(|(k, v)| *k == "x-api-key" && v == "secret"));
        assert!(h.iter().any(|(k, _)| *k == "anthropic-version"));
        assert!(
            !h.iter().any(|(k, _)| *k == "authorization"),
            "no bearer for anthropic"
        );
    }

    // --- upstream proxy tests ---

    fn bind(addr: &str) -> Option<std::net::SocketAddr> {
        Some(addr.parse().expect("test bind addr"))
    }

    #[test]
    fn redact_strips_userinfo() {
        assert_eq!(
            redact_proxy_url("http://alice:s3cr3t@proxy.corp:3128"),
            "http://proxy.corp:3128"
        );
    }

    #[test]
    fn redact_no_userinfo_unchanged() {
        assert_eq!(
            redact_proxy_url("http://proxy.corp:3128"),
            "http://proxy.corp:3128"
        );
    }

    #[test]
    fn redact_preserves_path() {
        assert_eq!(
            redact_proxy_url("http://u:p@proxy.corp:3128/path"),
            "http://proxy.corp:3128/path"
        );
    }

    // Same host + same port as the daemon bind address — infinite recursion, must reject.
    #[test]
    fn loopback_127_rejected() {
        let err = validate_proxy_url("http://127.0.0.1:8788", bind("127.0.0.1:8788")).unwrap_err();
        assert!(
            err.to_string().contains("own listen address")
                || err.to_string().contains("infinite recursion"),
            "got: {err}"
        );
    }

    // IPv6 loopback, same port — must reject.
    #[test]
    fn loopback_ipv6_rejected() {
        let err = validate_proxy_url("http://[::1]:8788", bind("127.0.0.1:8788")).unwrap_err();
        assert!(
            err.to_string().contains("own listen address")
                || err.to_string().contains("infinite recursion")
                || err.to_string().contains("::1"),
            "got: {err}"
        );
    }

    // `localhost` spelling of the same bind port — must reject (loopback alias normalisation).
    #[test]
    fn loopback_localhost_rejected() {
        let err = validate_proxy_url("http://localhost:8788", bind("127.0.0.1:8788")).unwrap_err();
        assert!(
            err.to_string().contains("own listen address")
                || err.to_string().contains("infinite recursion"),
            "got: {err}"
        );
    }

    // Different port on the same loopback host — this is the headroom composition case, must pass.
    #[test]
    fn loopback_different_port_allowed() {
        assert!(
            validate_proxy_url("http://127.0.0.1:9999", bind("127.0.0.1:8788")).is_ok(),
            "companion proxy on a different loopback port must be allowed"
        );
    }

    // Non-loopback host — always allowed regardless of port.
    #[test]
    fn non_loopback_allowed() {
        assert!(validate_proxy_url("http://proxy.corp:8788", bind("127.0.0.1:8788")).is_ok());
    }

    #[test]
    fn valid_proxy_without_auth() {
        // ureq validates the URL; a non-loopback host should parse fine.
        assert!(validate_proxy_url("http://proxy.corp:3128", None).is_ok());
    }

    #[test]
    fn valid_proxy_with_auth() {
        assert!(validate_proxy_url("http://user:pass@proxy.corp:3128", None).is_ok());
    }

    #[test]
    fn proxy_host_extraction() {
        assert_eq!(
            proxy_host("http://user:pass@proxy.corp:3128/"),
            Some("proxy.corp")
        );
        assert_eq!(proxy_host("http://proxy.corp:3128"), Some("proxy.corp"));
        assert_eq!(proxy_host("http://proxy.corp/"), Some("proxy.corp"));
        assert_eq!(proxy_host("not-a-url"), None);
    }
}
