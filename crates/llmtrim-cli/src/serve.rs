//! `serve` — MITM HTTPS interceptor (the universal integration), modeled on
//! llm-interceptor but compressing instead of just logging.
//!
//! llmtrim runs as an `HTTPS_PROXY`. Point any tool's `HTTPS_PROXY` at it and trust
//! the local CA; llmtrim terminates TLS *only* for the LLM provider hosts (everything
//! else blind-tunnels), decrypts the request, compresses the body with the matching
//! provider adapter, and re-encrypts to the real API — streaming the response straight
//! back. One mechanism covers every tool and every provider in `bench/pricing.json`,
//! because they all speak HTTPS to a known set of API hosts.

#[cfg(not(feature = "intercept"))]
pub fn run(_port: u16, _force: bool) -> anyhow::Result<()> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

#[cfg(not(feature = "intercept"))]
pub fn run_supervised(_port: u16, _force: bool) -> anyhow::Result<()> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

#[cfg(not(feature = "intercept"))]
pub fn ca_cert_path() -> anyhow::Result<std::path::PathBuf> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

#[cfg(not(feature = "intercept"))]
pub fn ensure_ca() -> anyhow::Result<(String, String)> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

#[cfg(feature = "intercept")]
pub use imp::{ca_cert_path, ensure_ca, run, run_supervised};

#[cfg(feature = "intercept")]
mod imp {
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::mpsc::Sender;

    use serde_json::Value;
    use std::sync::{Arc, Mutex};

    use crate::reroute::sse::ReduceEvent;

    use anyhow::{Context, Result};
    use bytes::Bytes;
    use http_body_util::{BodyExt, BodyStream, Full};
    use hudsucker::certificate_authority::CertificateAuthority;
    use hudsucker::hyper::http::uri::Authority as HttpAuthority;
    use hudsucker::hyper::{Method, Request, Response, header};
    use hudsucker::rustls::ServerConfig;
    use hudsucker::rustls::crypto::CryptoProvider;
    use hudsucker::rustls::crypto::aws_lc_rs;
    use hudsucker::rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
    use hudsucker::{Body, HttpContext, HttpHandler, Proxy, RequestOrResponse};
    use hyper_http_proxy::{Intercept, Proxy as UpstreamProxy, ProxyConnector};
    use hyper_util::client::legacy::connect::HttpConnector;

    use crate::tracking::{Record, Tracker};
    use llmtrim_core::config::{DenseConfig, RuntimeConfig};
    use llmtrim_core::ir::ProviderKind;
    use llmtrim_core::memo::Memo;

    /// The exact endpoint hosts of every provider in the `llm_providers` registry — the
    /// maintained upstream source of truth. The CA is name-constrained to these (plus their
    /// subdomains, via [`host_covered`]) and only matching hosts are intercepted. We key on the
    /// *exact* host, NOT the registrable parent: collapsing `generativelanguage.googleapis.com`
    /// to `googleapis.com` (or `dashscope.aliyuncs.com` → `aliyuncs.com`) would let the MITM CA
    /// forge certs for — and intercept — all of Google Cloud / Alibaba Cloud, an enormous and
    /// unnecessary blast radius. Computed once.
    static LLM_DOMAINS: once_cell::sync::Lazy<std::collections::HashSet<String>> =
        once_cell::sync::Lazy::new(|| {
            let mut set = std::collections::HashSet::new();
            for provider in llm_providers::get_providers_data().values() {
                for endpoint in provider.endpoints.values() {
                    if let Some(host) = host_of_url(endpoint.base_url) {
                        set.insert(host.to_ascii_lowercase());
                    }
                }
            }
            // Vertex AI (`aiplatform`) isn't in the registry but serves Claude/Gemini/OpenAI
            // shapes by path; cover it explicitly so the exact-host switch doesn't drop it.
            for h in GOOGLE_API_HOSTS {
                set.insert((*h).to_string());
            }
            set
        });

    /// Google LLM API hosts needing CA coverage + interception. `generativelanguage` is also in
    /// the registry; `aiplatform` (Vertex) is not, so it must be listed here explicitly.
    static GOOGLE_API_HOSTS: &[&str] = &[
        "generativelanguage.googleapis.com",
        "aiplatform.googleapis.com",
    ];

    /// Reputable direct LLM API hosts beyond the `llm_providers` registry that speak one of
    /// the three wire shapes we adapt (OpenAI / Anthropic / Gemini). Interception + CA coverage
    /// only — the wire format is detected from the body, never assumed from the host. Kept small
    /// on purpose: each entry widens what the name-constrained MITM CA may forge. To request a
    /// host, open an issue: https://github.com/fkiene/llmtrim/issues
    static EXTRA_HOSTS: &[&str] = &[
        "opencode.ai", // opencode zen gateway (OpenAI shape)
        "api.groq.com",
        "api.together.xyz",
        "api.fireworks.ai",
        "api.deepinfra.com",
        "api.perplexity.ai",
        "api.sambanova.ai",
        "inference.baseten.co",
        "api.studio.nebius.ai",
        "ai-gateway.vercel.sh",
        "chatgpt.com", // Codex CLI with ChatGPT sign-in (OpenAI Responses shape)
    ];

    /// User-configured extra hosts (env `LLMTRIM_EXTRA_HOSTS` / file `extra_hosts`), already
    /// normalized + validated by [`RuntimeConfig`]. Beyond the curated `EXTRA_HOSTS`, so a
    /// self-hosted or gateway OpenAI-compatible endpoint can be intercepted without a code
    /// change. Each widens the name-constrained CA, exactly like `EXTRA_HOSTS`.
    ///
    /// Note one asymmetry: a user host `llm.acme.com` enters the CA's permitted subtrees as
    /// `DnsName("llm.acme.com")`, which by RFC 5280 also lets the CA sign for its *subdomains*
    /// (`api.llm.acme.com`). Interception is narrower — [`extra_host_match`] matches user hosts
    /// *exactly*, never their subdomains — so the proxy never MITMs a host it wasn't told to.
    /// The wider CA scope is unused and low-risk (the CA is local-only and regenerated when the
    /// host set changes), but list exact hosts, not bare apexes, to keep the CA narrow.
    fn user_extra_hosts() -> &'static [String] {
        &RuntimeConfig::get().extra_hosts
    }

    /// True if `host` is a curated `EXTRA_HOSTS` entry or a user-configured extra host.
    fn is_extra_host(host: &str) -> bool {
        extra_host_match(host, user_extra_hosts())
    }

    /// Pure host-match used by [`is_extra_host`]: a curated `EXTRA_HOSTS` entry matches exactly
    /// or as a parent of `host` (subdomain), but a user-configured host matches **exactly only**
    /// — never its subdomains. So a mistakenly broad user entry (e.g. an apex) can't silently
    /// widen what gets intercepted beyond the exact host named.
    fn extra_host_match(host: &str, user_hosts: &[String]) -> bool {
        EXTRA_HOSTS
            .iter()
            .any(|s| host == *s || host.ends_with(&format!(".{s}")))
            || user_hosts.iter().any(|s| host == s)
    }

    /// True if a request to `host` resolving to wire shape `provider` should bypass compression
    /// (user opt-out via `exclude_hosts` / `exclude_providers`). Excluded traffic is still
    /// intercepted; it is just forwarded verbatim. Host match is exact (mirrors a user
    /// `extra_hosts` entry); provider match is by canonical wire-shape name. `host` is expected
    /// lowercased (the entries are normalized at config load).
    fn exclusion_match(
        host: &str,
        provider: ProviderKind,
        exclude_hosts: &[String],
        exclude_providers: &[String],
    ) -> bool {
        exclude_hosts.iter().any(|h| host == h)
            || exclude_providers.iter().any(|p| p == provider.as_str())
    }

    /// Every name-constraint domain the CA should permit: the registry parent domains plus the
    /// curated extra hosts, sorted + deduped. The single source of truth for what we intend to
    /// intercept; compared byte-for-byte against the CA's sidecar to decide whether to regenerate.
    fn intercept_domains() -> Vec<String> {
        intercept_domains_with(user_extra_hosts())
    }

    /// Pure body of [`intercept_domains`]: the registry parents + curated extras + the given
    /// user hosts, sorted + deduped. Split out so the user-host → CA-sidecar flow is testable.
    fn intercept_domains_with(user_hosts: &[String]) -> Vec<String> {
        let mut v: Vec<String> = LLM_DOMAINS
            .iter()
            .cloned()
            .chain(EXTRA_HOSTS.iter().map(|h| (*h).to_string()))
            .chain(user_hosts.iter().cloned())
            .collect();
        v.sort();
        v.dedup();
        v
    }

    /// True if `host` (already lowercased) falls under one of `domains` — exact match or a
    /// subdomain. The interceptor gates on the *CA's* domain set, so it never tries to MITM a
    /// host the name-constrained CA can't actually sign for.
    fn host_covered(host: &str, domains: &std::collections::HashSet<String>) -> bool {
        domains
            .iter()
            .any(|d| host == d || host.ends_with(&format!(".{d}")))
    }

    /// Host component of a URL like `https://api.openai.com/v1` → `api.openai.com`.
    fn host_of_url(url: &str) -> Option<&str> {
        let after = url.split_once("://").map_or(url, |(_, rest)| rest);
        let host = after.split(['/', '?']).next()?.split(':').next()?;
        (!host.is_empty()).then_some(host)
    }

    /// True if `host` equals `domain` or is a subdomain of it (dot-anchored, so
    /// `notanthropic.com` does not match `anthropic.com`).
    fn host_in(host: &str, domain: &str) -> bool {
        host == domain || host.ends_with(&format!(".{domain}"))
    }

    /// The provider wire shape for a host, or `None` if it isn't an LLM API host (so it is
    /// not intercepted). Anthropic and Google have their own shapes; every other registry
    /// provider speaks the OpenAI `/v1/chat/completions` shape.
    fn provider_for_host(host: &str) -> Option<ProviderKind> {
        let h = host.to_ascii_lowercase();
        if host_in(&h, "anthropic.com") {
            return Some(ProviderKind::Anthropic);
        }
        if GOOGLE_API_HOSTS.iter().any(|d| host_in(&h, d)) {
            return Some(ProviderKind::Google);
        }
        // Registry endpoint hosts and the curated extra hosts default to the OpenAI shape; the
        // exact adapter is refined from the body at compress time (`provider::detect`), so a host
        // that serves a Claude- or Gemini-shaped body is still handled correctly.
        if host_covered(&h, &LLM_DOMAINS) || is_extra_host(&h) {
            return Some(ProviderKind::OpenAi);
        }
        None
    }

    /// Endpoints whose POST bodies we may compress: text-generation only. Embeddings,
    /// moderations, audio, image, files, batch, and token-count endpoints carry bodies that
    /// either aren't prompts or whose semantics our prompt stages would corrupt — they pass
    /// through verbatim. Allowlist by path suffix/marker (host already gated separately).
    fn is_compressible_path(path: &str) -> bool {
        // Token-count endpoints must pass through: we'd otherwise return the count of the
        // *compressed* prompt, silently skewing the client's budgeting.
        if path.contains("count_tokens") || path.contains(":countTokens") {
            return false;
        }
        path.ends_with("/chat/completions")
            || path.ends_with("/responses")
            || path.ends_with("/messages")
            || path.contains(":generateContent")
            || path.contains(":streamGenerateContent")
            || path.contains(":rawPredict")
            || path.contains(":streamRawPredict")
    }

    /// Vertex AI serves all three wire shapes on `aiplatform.googleapis.com`, distinguished by
    /// the request path: the OpenAI-compatible endpoint is `…/endpoints/openapi/chat/completions`,
    /// `:rawPredict`/`:streamRawPredict` carries an Anthropic (Claude-on-Vertex) body, and
    /// `:generateContent` (and its streaming form) a Gemini one. `None` for an unrecognized path.
    fn vertex_kind(path: &str) -> Option<ProviderKind> {
        if path.contains("openapi") || path.ends_with("/chat/completions") {
            Some(ProviderKind::OpenAi)
        } else if path.contains(":rawPredict") || path.contains(":streamRawPredict") {
            Some(ProviderKind::Anthropic)
        } else if path.contains(":generateContent") || path.contains(":streamGenerateContent") {
            Some(ProviderKind::Google)
        } else {
            None
        }
    }

    /// True if the request carries an AWS SigV4 signature (its body is signed, so we must
    /// not modify it).
    fn is_body_signed(headers: &header::HeaderMap) -> bool {
        headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.starts_with("AWS4-HMAC-SHA256"))
            .unwrap_or(false)
    }

    /// True if `req` is a WebSocket upgrade attempt — either an HTTP/1.1 `Upgrade: websocket`
    /// handshake or an HTTP/2 Extended CONNECT (RFC 8441, the `:protocol` pseudo-header set to
    /// `websocket`). We refuse these on intercepted LLM hosts (see `handle_request_inner`): a
    /// WebSocket carries the prompt as frames, not an HTTP body, so llmtrim can't compress it,
    /// and hudsucker can't forward an h2 Extended CONNECT anyway — it stalls until the client
    /// times out. Refusing fast makes the client fall back to the plain-HTTPS transport, which
    /// is a normal POST body llmtrim *does* compress. OpenAI's Codex is the motivating client.
    fn is_websocket_upgrade(req: &Request<Body>) -> bool {
        if req.method() == Method::CONNECT
            && let Some(p) = req.extensions().get::<hudsucker::hyper::ext::Protocol>()
        {
            return p.as_str().eq_ignore_ascii_case("websocket");
        }
        req.headers()
            .get(header::UPGRADE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.eq_ignore_ascii_case("websocket"))
    }

    /// Host of a request: the URI authority, else the `Host` header (port stripped).
    fn host_of<B>(req: &Request<B>) -> Option<String> {
        if let Some(h) = req.uri().host() {
            return Some(h.to_string());
        }
        req.headers()
            .get(header::HOST)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(':').next().unwrap_or(s).to_string())
    }

    /// Input-side data for a compressed request, held between `handle_request` and
    /// `handle_response` (the handler is cloned per request, so this is per request/response
    /// pair) so the response can attach the measured output tokens.
    #[derive(Clone)]
    struct Pending {
        provider: ProviderKind,
        model: Option<String>,
        tokenizer: String,
        exact: bool,
        input_before: i64,
        input_after: i64,
        /// Microseconds spent in `compress_with_config` — the latency we add before forwarding.
        compress_micros: i64,
        /// Serialized rehydration plan. Reserved for reversible output-side transforms;
        /// none ship today (Stage D is input-only), so it currently reverses nothing.
        plan: String,
        /// The original (uncompressed) request, replayed verbatim if the upstream rejects
        /// our compressed version. The safety net: compression never breaks the call.
        original: Option<OriginalRequest>,
        /// True when the *forwarded* body carried the output-shaping instruction (Stage F
        /// ran and the compressed body was kept, not a passthrough/replay).
        output_shaped: bool,
        /// Frozen (cache-controlled) prefix tokens the stages skipped — the new-content
        /// meter for the ledger.
        frozen_input_tokens: Option<i64>,
        /// Per-source attribution of the forwarded request (breakdown view). `None` on bodies we
        /// couldn't attribute; never affects proxying.
        breakdown: Option<BreakdownPending>,
        /// Set when this request was rerouted to a subscription backend (`sub = codex|kimi`): the
        /// target provider + resolved upstream model. `handle_response` uses it to translate the
        /// provider's SSE back into Anthropic SSE. `provider` above is kept `Anthropic` so the
        /// output-usage parser and ledger see the Anthropic-shaped reply we emit to the client.
        reroute: Option<RerouteInfo>,
        /// Set in `sub` fallback mode: the turn was forwarded to Anthropic normally, but if
        /// Anthropic cannot serve it, `handle_response` replays it to this
        /// subscription provider instead. Mutually exclusive with `reroute` (which reroutes up
        /// front). See [`FallbackInfo`].
        fallback: Option<FallbackInfo>,
    }

    /// Reroute marker attached to a [`Pending`] (see its `reroute` field).
    #[derive(Clone)]
    struct RerouteInfo {
        provider: crate::reroute::SubProvider,
        model: String,
        /// The model Claude Code selected. The synthesized Anthropic response must preserve it:
        /// returning the subscription model makes Claude Code reject the response as unknown.
        client_model: String,
        /// The rewritten upstream request, retained so `reroute_response` can re-issue it on a
        /// retryable failure (429/5xx). `None` on paths that never retry (the fallback path).
        replay: Option<RerouteReplay>,
        /// The full logical codex (or provider) request body *before* any continuation delta was
        /// applied. Used to record server continuation state.
        logical_body: Option<Value>,
        /// The CC session id (x-claude-code-session-id), used as key for continuation state.
        session_id: Option<String>,
    }

    /// Enough of the rewritten upstream request to re-issue it on a retryable failure. The body is
    /// shared (`Arc`) so retaining it for retries costs one copy of the compressed body, not one
    /// per attempt.
    ///
    /// Note: `body` here is the *wire* form (may already be a delta after continuation apply).
    /// Use `RerouteInfo.logical_body` when you need the full pre-delta version (e.g. on
    /// continuation errors).
    #[derive(Clone)]
    struct RerouteReplay {
        url: String,
        headers: Vec<(String, String)>,
        body: Arc<Vec<u8>>,
    }

    /// Fallback marker attached to a [`Pending`] (see its `fallback` field). Holds what
    /// `handle_response` needs to replay the turn to the subscription provider: the provider, the
    /// original Anthropic request body to translate, and the Claude Code session id (for the
    /// provider's session header).
    #[derive(Clone)]
    struct FallbackInfo {
        /// Providers to try, in order. Never empty (the arming site drops the fallback otherwise).
        providers: Vec<crate::reroute::SubProvider>,
        anthropic_body: Vec<u8>,
        session_id: Option<String>,
    }

    struct FallbackAttempt {
        provider: crate::reroute::SubProvider,
        model: String,
        logical_body: Option<Value>,
        body: Vec<u8>,
    }

    /// Anthropic statuses that mean "Anthropic can't serve this turn" and so trigger fallback.
    fn is_sub_fallback_status(status: u16) -> bool {
        // Quota/auth failures plus the transient classes used by the sibling proxy. Do not
        // fail over validation errors (400/422) or a missing Anthropic credential (401): those
        // are caller/configuration problems, not provider capacity failures.
        matches!(
            status,
            402 | 403 | 408 | 425 | 429 | 500 | 502 | 503 | 504 | 529
        )
    }

    /// Some upstreams report a failed turn *inside* an HTTP 200 response: an `error` frame in the
    /// SSE stream, or an error object in a non-streamed JSON reply. Only a top-level `{"type":
    /// "error"}` counts — an assistant message that merely *mentions* that shape parses as its own
    /// frame type (`content_block_delta`, …) and is not treated as a failure.
    fn is_sub_fallback_body(body: &[u8]) -> bool {
        let text = String::from_utf8_lossy(body);
        text.lines().any(|line| {
            let data = line
                .strip_prefix("data:")
                .map(str::trim)
                .unwrap_or(line.trim());
            serde_json::from_str::<Value>(data)
                .is_ok_and(|value| value.get("type").and_then(Value::as_str) == Some("error"))
        })
    }

    /// Whether the SSE seen so far has already committed content to the client — the first
    /// `message_start`/`content_block_*` frame. Past that point the turn cannot be failed over
    /// (the client has the beginning of an answer), so the fallback probe stops peeking and
    /// streams the rest through untouched.
    fn sse_committed_content(body: &[u8]) -> bool {
        let text = String::from_utf8_lossy(body);
        text.lines().any(|line| {
            let data = line
                .strip_prefix("data:")
                .map(str::trim)
                .unwrap_or(line.trim());
            serde_json::from_str::<Value>(data).is_ok_and(|value| {
                matches!(
                    value.get("type").and_then(Value::as_str),
                    Some("message_start" | "content_block_start" | "content_block_delta")
                )
            })
        })
    }

    /// A verbatim copy of the client's request, for replay-on-error.
    #[derive(Clone)]
    struct OriginalRequest {
        url: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    /// Client request headers to forward on replay — everything except `host` and
    /// `content-length` (the HTTP client sets those from the URL and body) and `accept-encoding`
    /// (we don't bundle a brotli decoder, so force identity: replaying the client's
    /// `accept-encoding: br` could hand the client an undecodable body — the replay path's whole
    /// job is to not break the call).
    fn forward_headers(headers: &header::HeaderMap) -> Vec<(String, String)> {
        headers
            .iter()
            .filter(|(n, _)| !matches!(n.as_str(), "host" | "content-length" | "accept-encoding"))
            .filter_map(|(n, v)| v.to_str().ok().map(|v| (n.to_string(), v.to_string())))
            .collect()
    }

    /// Whether an upstream status code is a *validation* failure class that our body edits can
    /// cause (400 Bad Request / 422 Unprocessable Entity). Used to gate the replay-on-error path.
    /// We deliberately exclude 429/5xx: those are unrelated to our mutation.
    fn should_replay(status: u16) -> bool {
        matches!(status, 400 | 422)
    }

    /// Send the original (uncompressed) request to the upstream and buffer the reply. `None` if
    /// the round-trip itself fails.
    fn fetch_original(
        orig: &OriginalRequest,
        proxy_url: Option<&str>,
    ) -> Option<(u16, Option<String>, Vec<u8>)> {
        use std::io::Read;
        let body = std::str::from_utf8(&orig.body).ok()?;
        let mut up =
            crate::transport::forward_post(&orig.url, &orig.headers, body, proxy_url).ok()?;
        let mut buf = Vec::new();
        up.reader.read_to_end(&mut buf).ok()?;
        Some((up.status, up.content_type, buf))
    }

    /// Build a client response from a buffered upstream reply.
    fn buffered_response(
        status: u16,
        content_type: Option<String>,
        body: Vec<u8>,
    ) -> Response<Body> {
        let mut builder = Response::builder().status(status);
        if let Some(ct) = content_type {
            builder = builder.header(header::CONTENT_TYPE, ct);
        }
        builder
            .body(Body::from(Full::new(Bytes::from(body))))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }

    /// Replay the original (uncompressed) request to the upstream — direct, all statuses
    /// relayed — and build a response for the client. `None` if the replay itself fails (in
    /// which case the caller keeps the compressed response's error).
    fn replay_original(orig: &OriginalRequest, proxy_url: Option<&str>) -> Option<Response<Body>> {
        let (status, content_type, body) = fetch_original(orig, proxy_url)?;
        Some(buffered_response(status, content_type, body))
    }

    /// Peek the head of an Anthropic SSE stream in fallback mode. Stops at the first frame that
    /// either reports an error (`Err`: the turn has committed nothing to the client, so it can
    /// still be failed over) or commits content (`message_start`/`content_block_*` — from there on
    /// the client owns a partial answer and the rest streams through untouched). A transport
    /// failure before any content is treated as a failed turn too: a dropped Anthropic connection
    /// is exactly what the chain exists for.
    async fn probe_sse_head(body: Body) -> Result<Body, ()> {
        use hudsucker::futures::StreamExt;
        let mut stream = BodyStream::new(body);
        let mut head: Vec<Bytes> = Vec::new();
        let mut seen = Vec::<u8>::new();
        while let Some(frame) = stream.next().await {
            let Ok(frame) = frame else {
                return Err(());
            };
            let Ok(bytes) = frame.into_data() else {
                continue;
            };
            seen.extend_from_slice(&bytes);
            head.push(bytes);
            if is_sub_fallback_body(&seen) {
                return Err(());
            }
            if sse_committed_content(&seen) {
                break;
            }
        }
        let head = hudsucker::futures::stream::iter(head.into_iter().map(Ok));
        let rest = stream.filter_map(|frame| {
            hudsucker::futures::future::ready(match frame {
                Ok(frame) => frame.into_data().ok().map(Ok),
                Err(e) => Some(Err(e)),
            })
        });
        Ok(Body::from_stream(head.chain(rest)))
    }

    /// A JSON body response (used for the local `count_tokens` answer on the reroute path).
    fn json_response(status: u16, body: &str) -> Response<Body> {
        Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(Full::new(Bytes::from(body.to_string()))))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }

    /// A single-frame Anthropic SSE response (used for reroute error/short-circuit replies).
    fn sse_response(body: String) -> Response<Body> {
        Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .body(Body::from(Full::new(Bytes::from(body))))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }

    /// An Anthropic-shaped error the client (Claude Code) renders, for reroute pre-flight failures
    /// (no auth, unparseable body, translation error). `status` is the HTTP status.
    fn anthropic_error(status: u16, message: &str) -> Response<Body> {
        anthropic_error_typed(status, "api_error", message, None)
    }

    /// Like [`anthropic_error`] but with an explicit error `kind` (so a rate limit renders as
    /// `rate_limit_error`, not a generic `api_error`) and an optional `Retry-After` header
    /// (seconds) so the client backs off sensibly instead of tight-retrying.
    fn anthropic_error_typed(
        status: u16,
        kind: &str,
        message: &str,
        retry_after_secs: Option<u64>,
    ) -> Response<Body> {
        let body = serde_json::json!({
            "type": "error",
            "error": { "type": kind, "message": message },
        })
        .to_string();
        let mut builder = Response::builder()
            .status(status)
            .header(header::CONTENT_TYPE, "application/json");
        if let Some(secs) = retry_after_secs {
            builder = builder.header(header::RETRY_AFTER, secs.to_string());
        }
        builder
            .body(Body::from(Full::new(Bytes::from(body))))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }

    /// True if the client asked for a streamed reply (`Accept: text/event-stream`, which the
    /// Anthropic SDK — and so Claude Code — sets on every `stream: true` call). Read from the
    /// headers, not the body, so it also works on the passthrough paths that never buffer one.
    fn wants_event_stream(headers: &header::HeaderMap) -> bool {
        headers
            .get(header::ACCEPT)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("text/event-stream"))
    }

    /// The reply to a *transport* failure against the upstream: the TLS/TCP/HTTP-2 exchange
    /// never produced a response, so there is no status to relay.
    ///
    /// hudsucker's default `handle_error` answers these with a bodiless 502 and logs through
    /// `tracing`, for which llmtrim installs no subscriber — so the client got an unparseable
    /// error and the user got no way to see the cause (issue #157). We answer in the shape the
    /// client is already parsing: an SSE `error` frame for a streaming call, JSON otherwise.
    ///
    /// `overloaded_error` (529), not `api_error` (502): a dropped connection is transient and
    /// this is the class Claude Code retries. A bodiless 502 is a hard failure to the SDK.
    fn upstream_transport_error(streaming: bool, message: &str) -> Response<Body> {
        let payload = serde_json::json!({
            "type": "error",
            "error": { "type": "overloaded_error", "message": message },
        });
        if streaming {
            // HTTP 200 + an SSE `error` frame: the stream has already been committed to by the
            // time the client is reading, so the error has to arrive *in* the stream.
            let mut out = String::new();
            out.push_str("event: error\ndata: ");
            out.push_str(&payload.to_string());
            out.push_str("\n\n");
            return sse_response(out);
        }
        anthropic_error_typed(529, "overloaded_error", message, Some(1))
    }

    /// Anthropic error `type` for an upstream status, so Claude Code renders rate limits and
    /// auth failures as their own error classes instead of a generic `api_error`.
    fn reroute_error_kind(status: u16) -> &'static str {
        match status {
            400 | 422 => "invalid_request_error",
            401 | 403 => "authentication_error",
            429 => "rate_limit_error",
            529 => "overloaded_error",
            _ => "api_error",
        }
    }

    /// Best-effort HTTP status for an error surfaced *inside* an HTTP 200 upstream stream (the
    /// reducer keeps only the message text, not the upstream code). Context-window overflow is a
    /// client input error (400); an explicit rate/usage limit is 429; anything else is treated as a
    /// bad gateway (502).
    fn reroute_stream_error_status(message: &str) -> u16 {
        let m = message.to_lowercase();
        if m.contains("context") && (m.contains("window") || m.contains("length")) {
            400
        } else if m.contains("rate limit") || m.contains("usage limit") {
            429
        } else {
            502
        }
    }

    /// Statuses worth re-issuing against the subscription backend: transient server/overload
    /// errors and rate limits. A rate limit is only actually retried when its backoff fits the
    /// budget (see [`reroute_backoff_ms`]); a multi-hour usage-limit reset is surfaced at once.
    fn reroute_should_retry(status: u16) -> bool {
        matches!(status, 429 | 500 | 502 | 503 | 504 | 529)
    }

    const REROUTE_RETRY_MAX: u32 = 3;
    /// Wall-clock budget for a whole fallback chain (all providers, all their retries). Bounds the
    /// worst case a client waits after Anthropic already failed.
    const FALLBACK_CHAIN_BUDGET: std::time::Duration = std::time::Duration::from_secs(90);
    const REROUTE_RETRY_BASE_MS: u64 = 1_000;
    const REROUTE_RETRY_CAP_MS: u64 = 20_000;

    /// Backoff before retry `attempt` (0-based). Honors an upstream reset hint (`retry_after_secs`)
    /// when present, otherwise jittered-free exponential backoff. Returns `None` when the wait
    /// would exceed the cap — e.g. a usage-limit reset hours away — so the caller stops retrying
    /// and surfaces the error immediately instead of stalling.
    fn reroute_backoff_ms(attempt: u32, retry_after_secs: Option<u64>) -> Option<u64> {
        if let Some(secs) = retry_after_secs {
            let ms = secs.saturating_mul(1_000);
            return (ms <= REROUTE_RETRY_CAP_MS).then_some(ms);
        }
        let ms = REROUTE_RETRY_BASE_MS
            .saturating_mul(1u64 << attempt.min(5))
            .min(REROUTE_RETRY_CAP_MS);
        Some(ms)
    }

    /// Extract a rate-limit reset hint (seconds) from an upstream response: the standard
    /// `Retry-After`, a Codex `x-codex-primary-reset-after-seconds` header, or the JSON body's
    /// `error.resets_in_seconds`. `None` when the response carries no usable hint.
    fn reroute_retry_after_secs(
        retry_after_header: Option<&str>,
        codex_reset_header: Option<&str>,
        body: &[u8],
    ) -> Option<u64> {
        let parse = |s: &str| s.trim().parse::<f64>().ok().filter(|v| *v >= 0.0);
        if let Some(v) = retry_after_header.and_then(parse) {
            return Some(v.ceil() as u64);
        }
        if let Some(v) = codex_reset_header.and_then(parse) {
            return Some(v.ceil() as u64);
        }
        serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|j| {
                j.get("error")
                    .and_then(|e| e.get("resets_in_seconds"))
                    .and_then(serde_json::Value::as_i64)
            })
            .filter(|s| *s > 0)
            .map(|s| s as u64)
    }

    /// [`reroute_retry_after_secs`] over a response header map (the first, hudsucker-fetched
    /// attempt, which still carries the provider's headers).
    fn reroute_retry_after_from_headers(headers: &header::HeaderMap, body: &[u8]) -> Option<u64> {
        let get = |n: &str| headers.get(n).and_then(|v| v.to_str().ok());
        reroute_retry_after_secs(
            get("retry-after"),
            get("x-codex-primary-reset-after-seconds"),
            body,
        )
    }

    fn is_continuation_error(body: &[u8]) -> bool {
        let lower = String::from_utf8_lossy(body).to_lowercase();
        lower.contains("previous_response_not_found")
            || lower.contains("previous response not found")
            || (lower.contains("continuation") && lower.contains("not found"))
    }

    /// Centralized recording for Codex continuation.
    /// Only records when eligible (completed non-incomplete) and we have logical/session.
    /// Uses take_codex_output_items() to get assistant outputs for the transcript.
    fn record_codex_continuation(
        ev: &ReduceEvent,
        reducer: &mut crate::reroute::StreamReducer,
        provider: crate::reroute::SubProvider,
        logical_body: Option<&Value>,
        session_id: Option<&str>,
    ) {
        if let ReduceEvent::Finish {
            response_id: Some(rid),
            continuation_eligible: true,
            ..
        } = ev
            && provider == crate::reroute::SubProvider::Codex
            && let Some(logical) = logical_body
        {
            let items = reducer.take_codex_output_items();
            crate::reroute::continuation::record_continuation(
                session_id,
                logical,
                Some(rid.as_str()),
                &items,
            );
        }
    }

    /// Build the client-facing message for a non-2xx subscription-backend response. Pulls the
    /// upstream `error.message` out of the JSON body when present (falling back to a raw snippet),
    /// and adds a hint for the common rate-limit case so the user sees "usage limit reached" and
    /// when it resets instead of an opaque HTTP code.
    fn reroute_upstream_error_message(
        provider: crate::reroute::SubProvider,
        status: u16,
        raw: &[u8],
    ) -> String {
        let json: Option<serde_json::Value> = serde_json::from_slice(raw).ok();
        let err = json.as_ref().and_then(|v| v.get("error"));
        let detail = err
            .and_then(|e| e.get("message"))
            .and_then(serde_json::Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                String::from_utf8_lossy(raw)
                    .trim()
                    .chars()
                    .take(200)
                    .collect()
            });

        let mut msg = format!(
            "llmtrim: {} subscription backend returned HTTP {status}: {detail}",
            provider.as_str(),
        );
        if let Some(secs) = err
            .and_then(|e| e.get("resets_in_seconds"))
            .and_then(serde_json::Value::as_i64)
            .filter(|s| *s > 0)
        {
            let mins = secs / 60;
            if mins >= 60 {
                msg.push_str(&format!(" (resets in ~{}h{:02}m)", mins / 60, mins % 60));
            } else {
                msg.push_str(&format!(" (resets in ~{mins}m)"));
            }
        }
        msg
    }

    /// Override the Codex reasoning effort on a to-be-translated Anthropic body by writing
    /// `output_config.effort`, the field the Codex translator reads. By default the reroute honors
    /// the client's own `output_config.effort` (Claude Code sets it per turn); this overwrites it
    /// with the operator's forced value, so it wins over what the client sent. No-op when `effort`
    /// is `None` (adaptive default) or the body isn't a JSON object. Kimi ignores it.
    fn apply_sub_effort(value: &mut serde_json::Value, effort: Option<&str>) {
        if let Some(effort) = effort
            && let Some(obj) = value.as_object_mut()
        {
            let oc = obj
                .entry("output_config")
                .or_insert_with(|| serde_json::json!({}));
            if let Some(oc) = oc.as_object_mut() {
                oc.insert("effort".to_string(), serde_json::json!(effort));
            }
        }
    }

    /// Does the plan carry a reversible *output* transform? No output-side transform ships
    /// today (Stage D is input-only; DSS was removed), so nothing is reversed.
    fn plan_reverses_output(_plan: &str) -> bool {
        false
    }

    /// Billing usage harvested from a captured response. All `None` when the response
    /// never arrived / carried no usage.
    #[derive(Default)]
    struct ResponseUsage {
        output_after: Option<i64>,
        cache_read: Option<i64>,
        fresh_input: Option<i64>,
        cache_write: Option<i64>,
    }

    fn record_from(p: Pending, usage: ResponseUsage) -> Record {
        Record {
            provider: p.provider.as_str().to_string(),
            model: p.model,
            tokenizer: p.tokenizer,
            exact: p.exact,
            input_before: p.input_before,
            input_after: p.input_after,
            output_before: None,
            output_after: usage.output_after,
            compress_micros: Some(p.compress_micros),
            cache_read_tokens: usage.cache_read,
            fresh_input_tokens: usage.fresh_input,
            cache_write_tokens: usage.cache_write,
            output_shaped: Some(p.output_shaped),
            frozen_input_tokens: p.frozen_input_tokens,
        }
    }

    /// Per-source attribution attached to a `Pending` for the breakdown view: the parsed
    /// content blocks of the forwarded request, the inferred identity, and the model's
    /// context window. `None` on bodies we couldn't attribute (still proxied normally).
    #[derive(Clone)]
    struct BreakdownPending {
        blocks: Vec<llmtrim_core::attribution::BlockAttribution>,
        identity: llmtrim_core::attribution::RequestIdentity,
        window: i64,
        /// Claude Code's own session id (`x-claude-code-session-id` header), when present.
        cc_session_id: Option<String>,
    }

    /// A completed breakdown turn (identity + frozen pricing) plus its reconciled source blocks,
    /// sent to the ledger thread on its own channel.
    struct BreakdownPayload {
        turn: crate::tracking::BreakdownTurn,
        blocks: Vec<crate::tracking::BreakdownBlock>,
    }

    /// Context window (tokens) for a model id, for occupancy %. Looks the model up in the curated
    /// models.dev snapshot (`llmtrim_core::context_window`) for a real per-model window, rather than
    /// guessing per family. Two cases the registry can't answer: `LLMTRIM_BREAKDOWN_WINDOW` overrides
    /// everything (unusual deployments), and the `[1m]` suffix is Anthropic's opt-in 1M-context beta
    /// (a per-request capability, not a registry property). Unknown models fall back to 128k; the
    /// caller clamps the window up to the real prompt size, so an under-shoot never shows over 100%.
    fn window_for(model: Option<&str>) -> i64 {
        if let Some(n) = RuntimeConfig::get().breakdown_window {
            return n;
        }
        let m = model.unwrap_or("");
        if m.to_ascii_lowercase().contains("[1m]") {
            return 1_000_000;
        }
        llmtrim_core::context_window(m).map_or(128_000, i64::from)
    }

    /// Parse a forwarded request body into its breakdown attribution (blocks + identity + window).
    /// Best-effort: returns `None` on a non-JSON body or when no blocks were found.
    fn attribute_for_breakdown(
        json: &str,
        kind: ProviderKind,
        model: Option<&str>,
        cc_session_id: Option<String>,
    ) -> Option<BreakdownPending> {
        let body: serde_json::Value = serde_json::from_str(json).ok()?;
        let counter = llmtrim_core::tokenizer::counter_for(kind, model).ok()?;
        let blocks = llmtrim_core::attribution::attribute(&body, kind, counter.as_ref());
        if blocks.is_empty() {
            return None;
        }
        let identity = llmtrim_core::attribution::extract_identity(&body, kind);
        Some(BreakdownPending {
            blocks,
            identity,
            window: window_for(model),
            cc_session_id,
        })
    }

    /// Reconcile the provider's real usage onto the attributed blocks and build the breakdown
    /// payload. Returns `None` when the request carried no attribution.
    ///
    /// Input billing is distributed by **calibration + prefix tape**: block token counts are
    /// scaled to the provider's real input total, then walked in wire order assigning the
    /// cached-read prefix first, the cache-write portion next, and the fresh tail last — the
    /// cache boundary is positional, so this is more faithful than a flat proportion. Output
    /// usage becomes a single synthetic `Output` block (responses aren't parsed per block).
    fn build_breakdown(p: &Pending, usage: &ResponseUsage) -> Option<BreakdownPayload> {
        let xp = p.breakdown.as_ref()?;
        let provider = p.provider.as_str();
        let rates = crate::monitor::rates_for(provider, p.model.as_deref());

        let fresh = usage.fresh_input.unwrap_or(0).max(0) as f64;
        let cache_read = usage.cache_read.unwrap_or(0).max(0) as f64;
        let cache_write = usage.cache_write.unwrap_or(0).max(0) as f64;
        let output = usage.output_after.unwrap_or(0).max(0) as f64;
        let total_in = fresh + cache_read + cache_write;

        let raw_sum: f64 = xp.blocks.iter().map(|b| b.tokens as f64).sum();
        let scale = if raw_sum > 0.0 && total_in > 0.0 {
            total_in / raw_sum
        } else {
            // No usage (e.g. streaming with no usage frame): keep raw counts, zero dollars.
            1.0
        };

        let mut remaining_read = cache_read;
        let mut remaining_write = cache_write;
        let mut out_blocks = Vec::with_capacity(xp.blocks.len() + 1);
        for b in &xp.blocks {
            let cal = b.tokens as f64 * scale;
            let mut amt = cal;
            let r = amt.min(remaining_read);
            remaining_read -= r;
            amt -= r;
            let w = amt.min(remaining_write);
            remaining_write -= w;
            amt -= w;
            let f = amt; // remainder is fresh input
            let (group, label) = b.category();
            out_blocks.push(crate::tracking::BreakdownBlock {
                zone: b.zone.as_str().to_string(),
                section: b.section.as_str().to_string(),
                bucket: b.bucket.as_str().to_string(),
                group_label: group.to_string(),
                label: label.to_string(),
                mcp_server: b.mcp_server.clone(),
                tool_name: b.tool_name.clone(),
                role: b.role.map(|r| r.as_str().to_string()),
                msg_index: b.msg_index.map(|i| i as i64),
                raw_tokens: b.tokens as i64,
                fresh_tok: f,
                cache_read_tok: r,
                cache_write_tok: w,
                output_tok: 0.0,
            });
        }
        // Floating-point dust: the tape consumes cache_read/cache_write exactly when the
        // calibrated block sum equals total_in, but rounding can leave a token or two
        // unassigned. Fold any remainder into the last input block so the block splits sum
        // back to the provider's reported usage. `out_blocks` is non-empty here (attribution
        // only attaches when it found ≥1 block), and the output block isn't pushed yet, so
        // `last` is always an input block.
        if (remaining_read.abs() > f64::EPSILON || remaining_write.abs() > f64::EPSILON)
            && let Some(last) = out_blocks.last_mut()
        {
            debug_assert!(
                last.fresh_tok + 1.0 >= remaining_read + remaining_write,
                "prefix-tape remainder exceeds last block's fresh tokens — splits would under-count"
            );
            last.cache_read_tok += remaining_read;
            last.cache_write_tok += remaining_write;
            last.fresh_tok = (last.fresh_tok - remaining_read - remaining_write).max(0.0);
        }
        if output > 0.0 {
            out_blocks.push(crate::tracking::BreakdownBlock {
                zone: "output".to_string(),
                section: "messages".to_string(),
                bucket: "text".to_string(),
                group_label: "Output".to_string(),
                label: "output".to_string(),
                raw_tokens: output as i64,
                output_tok: output,
                ..Default::default()
            });
        }

        let bill_micros = (fresh * rates.input
            + cache_read * rates.cache_read
            + cache_write * rates.cache_write
            + output * rates.output)
            .round() as i64;

        let id = &xp.identity;
        let turn = crate::tracking::BreakdownTurn {
            session_id: id
                .session_id
                .clone()
                .unwrap_or_else(|| "unknown".to_string()),
            cc_session_id: xp.cc_session_id.clone(),
            agent: id.agent.clone().unwrap_or_else(|| "unknown".to_string()),
            project: id.project.clone(),
            session_name: None,
            provider: provider.to_string(),
            model: p.model.clone(),
            // Who actually answered. `reroute` is set up front in `always` mode, and in
            // `fallback` mode only once an attempt has *succeeded* — so it is true per turn,
            // which config alone can never be (a fallback fires only when Anthropic fails).
            sub_provider: p.reroute.as_ref().map(|r| r.provider.as_str().to_string()),
            // The window heuristic (`window_for`) can under-shoot — it can't tell a 1M-context
            // beta from the model id — which would make occupancy exceed 100%. The window must
            // be at least the prompt actually sent, so clamp it to the real input size.
            window: xp.window.max(total_in as i64).max(raw_sum as i64),
            fresh_input: fresh as i64,
            cache_read: cache_read as i64,
            cache_write: cache_write as i64,
            output_tok: output as i64,
            input_rate: rates.input,
            output_rate: rates.output,
            cache_read_rate: rates.cache_read,
            cache_write_rate: rates.cache_write,
            bill_micros,
            input_before: p.input_before,
            input_after: p.input_after,
        };
        Some(BreakdownPayload {
            turn,
            blocks: out_blocks,
        })
    }

    /// Expand the model's shorthand answer back to normal output using the request's plan,
    /// rewrite it into the response JSON, and record the (shorthand-billed) output tokens.
    /// Returns the rewritten body, or the original on any parse failure.
    fn rehydrate_response(bytes: &[u8], p: &Pending, ledger: &Sender<Record>) -> Vec<u8> {
        let original = bytes.to_vec();
        let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            let _ = ledger.send(record_from(p.clone(), ResponseUsage::default()));
            return original;
        };
        let answer = llmtrim_core::provider::for_kind(p.provider).answer_text(&json);
        // The model billed the shorthand it emitted — count that for spend.
        let out_tok = answer.as_ref().and_then(|a| {
            llmtrim_core::tokenizer::counter_for(p.provider, p.model.as_deref())
                .ok()
                .map(|c| c.count(a) as i64)
        });
        if let Some(answer) = answer
            && let Ok(expanded) = llmtrim_core::rehydrate(&answer, &p.plan)
        {
            set_answer(&mut json, p.provider, &expanded);
        }
        let _ = ledger.send(record_from(
            p.clone(),
            ResponseUsage {
                output_after: out_tok,
                ..Default::default()
            },
        ));
        serde_json::to_vec(&json).unwrap_or(original)
    }

    /// Replace the answer text in a provider response with `text`.
    fn set_answer(json: &mut serde_json::Value, provider: ProviderKind, text: &str) {
        use serde_json::{Value, json};
        match provider {
            ProviderKind::OpenAi => {
                if let Some(c) = json.pointer_mut("/choices/0/message/content") {
                    *c = Value::String(text.to_string());
                }
            }
            ProviderKind::Anthropic => {
                if let Some(obj) = json.as_object_mut() {
                    obj.insert(
                        "content".to_string(),
                        json!([{"type": "text", "text": text}]),
                    );
                }
            }
            ProviderKind::Google => {
                if let Some(c) = json.pointer_mut("/candidates/0/content/parts/0/text") {
                    *c = Value::String(text.to_string());
                }
            }
        }
    }

    #[derive(Clone)]
    struct Interceptor {
        config: Arc<DenseConfig>,
        ledger: Sender<Record>,
        /// Companion channel for breakdown per-source attribution payloads, drained by the same
        /// ledger thread. Separate from `ledger` so the `compressions` write path is untouched.
        breakdown_ledger: Sender<BreakdownPayload>,
        /// Domains the live CA covers (its sidecar set). We intercept only hosts under these, so
        /// a stale CA — one generated before a host was added — blind-tunnels the new host rather
        /// than forging a cert its own name-constraints reject. Degrade safely, never break.
        domains: Arc<std::collections::HashSet<String>>,
        /// Turn-stability memo (see [`llmtrim_core::memo`]), shared across the per-request handler
        /// clones so a conversation's earlier-turn compressed prefix is reusable on its next
        /// turn — keeping the provider prefix cache warm on agent loops. In-memory only.
        memo: Arc<Memo>,
        /// The compressed request awaiting its response (set in `handle_request`).
        pending: Option<Pending>,
        /// Whether this request asked for a streamed reply (set in `handle_request`, read by
        /// `handle_error` so a transport failure is reported in the shape the client parses).
        /// Per-request: the handler is cloned per request, like `pending`.
        streaming: bool,
        /// Optional upstream proxy URL from `LLMTRIM_UPSTREAM_PROXY`. Used by the replay path
        /// (`forward_post`). The primary MITM interception path honours this setting via the
        /// `ProxyConnector` built at startup (see the `start` function in this module).
        upstream_proxy: Option<String>,
        /// User opt-out lists (`exclude_hosts` / `exclude_providers`), snapshotted from
        /// [`RuntimeConfig`] at construction like `domains` above: a request matching either is
        /// forwarded verbatim (still intercepted, just not compressed).
        exclude_hosts: Arc<Vec<String>>,
        exclude_providers: Arc<Vec<String>>,
        /// Subscription reroute target (`sub = codex|kimi`), snapshotted from [`RuntimeConfig`] at
        /// construction. `None` = normal transparent compress-and-forward. When set, intercepted
        /// Anthropic `/v1/messages` traffic is translated and sent to that subscription's backend.
        sub: Option<crate::reroute::SubProvider>,
        /// Ordered fallback providers. The active `sub` is used when this is empty.
        sub_chain: Arc<Vec<crate::reroute::SubProvider>>,
        /// Tier→model overrides for the active `sub` (from `[sub.<provider>.tiers]`); empty = use
        /// the built-in preset. Shared across per-request handler clones.
        sub_tiers: Arc<std::collections::BTreeMap<String, String>>,
        /// Fallback mode: when `true`, a set `sub` only takes over after Anthropic itself
        /// returns a quota/overload status; when `false`, `sub` reroutes every matching turn.
        sub_fallback: bool,
        /// Proxy-side Codex reasoning effort applied to every rerouted request (`None` = off).
        sub_effort: Option<String>,
    }

    impl Drop for Interceptor {
        fn drop(&mut self) {
            // A compressed request whose response we never saw (connection dropped): still
            // record the input savings, with output unknown.
            if let Some(p) = self.pending.take() {
                if let Some(x) = build_breakdown(&p, &ResponseUsage::default()) {
                    let _ = self.breakdown_ledger.send(x);
                }
                let _ = self.ledger.send(record_from(p, ResponseUsage::default()));
            }
        }
    }

    impl HttpHandler for Interceptor {
        /// Only MITM (forge a cert for) the LLM provider hosts; everything else is
        /// blind-tunneled, so the CA is never used outside its purpose.
        async fn should_intercept(&mut self, _ctx: &HttpContext, req: &Request<Body>) -> bool {
            host_of(req)
                .map(|h| host_covered(&h.to_ascii_lowercase(), &self.domains))
                .unwrap_or(false)
        }

        async fn handle_request(
            &mut self,
            _ctx: &HttpContext,
            req: Request<Body>,
        ) -> RequestOrResponse {
            self.streaming = wants_event_stream(req.headers());
            self.handle_request_inner(req).await
        }

        /// A request that never reached the upstream (TLS handshake, TCP reset, HTTP/2 GOAWAY,
        /// or — the common one on an agent loop — a pooled keep-alive connection the server had
        /// already closed). hudsucker's default hands the client a bodiless 502 and logs only
        /// through `tracing`, which llmtrim does not subscribe to: the user sees a dropped
        /// connection with no cause, and the SDK sees an error it cannot parse (issue #157).
        ///
        /// Report it instead: name the cause on stderr (where the rest of llmtrim's proxy
        /// diagnostics go, so `llmtrim serve` in the foreground shows it) and answer in the
        /// client's own error shape so it can back off and retry rather than fail the turn.
        async fn handle_error(
            &mut self,
            _ctx: &HttpContext,
            err: hyper_util::client::legacy::Error,
        ) -> Response<Body> {
            // `err` alone prints as "client error (Connect)"; the source chain carries the
            // underlying cause ("connection closed before message completed", the TLS error, …).
            let mut cause = err.to_string();
            let mut src: Option<&(dyn std::error::Error + 'static)> =
                std::error::Error::source(&err);
            while let Some(e) = src {
                cause.push_str(": ");
                cause.push_str(&e.to_string());
                src = std::error::Error::source(e);
            }
            eprintln!(
                "llmtrim: upstream request failed ({cause}) — reported to the client as a retryable error"
            );
            upstream_transport_error(self.streaming, &format!("upstream request failed: {cause}"))
        }

        /// Tee the response: forward it to the client unchanged while accumulating a copy,
        /// and once it finishes streaming, measure the output tokens and complete the ledger
        /// record. Non-compressed requests (no `pending`) pass straight through.
        async fn handle_response(
            &mut self,
            _ctx: &HttpContext,
            res: Response<Body>,
        ) -> Response<Body> {
            self.handle_response_inner(res).await
        }
    }

    impl Interceptor {
        /// Body of `handle_request`, factored out so tests can call it directly without
        /// constructing a `hudsucker::HttpContext` (which is `#[non_exhaustive]` and
        /// cannot be instantiated outside the hudsucker crate).
        async fn handle_request_inner(&mut self, req: Request<Body>) -> RequestOrResponse {
            // Refuse WebSocket upgrades on intercepted hosts so the client drops to the
            // compressible plain-HTTPS transport (see `is_websocket_upgrade`). 426 Upgrade
            // Required is a clean, immediate handshake failure — no body, no hang — so the
            // client falls back at once instead of retrying the dead upgrade for seconds.
            if is_websocket_upgrade(&req) {
                let res = Response::builder()
                    .status(hudsucker::hyper::StatusCode::UPGRADE_REQUIRED)
                    .body(Body::empty())
                    .expect("static 426 response is always valid");
                return RequestOrResponse::Response(res);
            }
            // Lowercase the host once: every host comparison below (Vertex suffix, provider
            // lookup, exclusion match) is case-insensitive.
            let host = host_of(&req).map(|h| h.to_ascii_lowercase());
            // Vertex AI serves all three wire shapes on one host, keyed by the path; every other
            // host is host-derived. Either way the kind here is only the fallback — the body
            // shape (`provider::detect`) refines it at compress time.
            let provider = host.as_deref().and_then(|h| {
                if h.ends_with("aiplatform.googleapis.com") {
                    Some(vertex_kind(req.uri().path()).unwrap_or(ProviderKind::Google))
                } else {
                    provider_for_host(h)
                }
            });
            // Subscription reroute: if `sub` is configured and this is Anthropic `/v1/messages`
            // traffic, translate it to the chosen provider's backend instead of forwarding to
            // Anthropic. `count_tokens` is answered locally (it would otherwise be billed against
            // the sub provider, so it can't be proxied to Anthropic). This runs before the normal
            // compress/exclude path so a `sub` selection always takes effect.
            // Fallback mode arms a fallback instead of rerouting up front: the turn goes to
            // Anthropic normally, and only replays to the sub provider if Anthropic fails (handled
            // in the response phase). Captured here so we can read the session header and body.
            let mut fallback_arm: Option<(Vec<crate::reroute::SubProvider>, Option<String>)> = None;
            if matches!(provider, Some(ProviderKind::Anthropic)) && req.method() == Method::POST {
                let path = req.uri().path();
                if !self.sub_fallback {
                    if let Some(sub) = self.sub {
                        if path.ends_with("/v1/messages/count_tokens") {
                            return self.reroute_count_tokens(req).await;
                        }
                        if path.ends_with("/v1/messages") {
                            return self.reroute_messages(req, sub).await;
                        }
                    }
                } else if path.ends_with("/v1/messages") {
                    // A chain is enough to arm the fallback: `sub` selects who serves *every* turn
                    // in `always` mode, but in fallback mode the chain is the whole answer, so
                    // `sub = off` + a chain is a valid configuration.
                    let mut providers = self.sub_chain.as_ref().clone();
                    if providers.is_empty() {
                        providers.extend(self.sub);
                    }
                    if !providers.is_empty() {
                        let session_id = req
                            .headers()
                            .get("x-claude-code-session-id")
                            .and_then(|v| v.to_str().ok())
                            .map(str::to_string);
                        fallback_arm = Some((providers, session_id));
                    }
                }
            }
            // Only compress POST bodies to a known provider; pass everything else through.
            let Some(provider) = provider.filter(|_| req.method() == Method::POST) else {
                return req.into();
            };
            // User opt-out: a host/provider on the exclude list is forwarded verbatim (still
            // MITM'd, just not compressed) — e.g. exclude Anthropic so Claude Code is untouched.
            if let Some(h) = host.as_deref()
                && exclusion_match(h, provider, &self.exclude_hosts, &self.exclude_providers)
            {
                return req.into();
            }
            // Compress text-generation endpoints only. Embeddings / moderations / token-count
            // bodies aren't prompts (or our edits would skew the result), so forward verbatim —
            // never silently mutate the input of a `/v1/embeddings` call.
            if !is_compressible_path(req.uri().path()) {
                return req.into();
            }
            // Body-signed requests (AWS SigV4) must pass through untouched — changing the
            // body would invalidate the signature and the provider would reject it.
            if is_body_signed(req.headers()) {
                return req.into();
            }
            let (parts, body) = req.into_parts();
            let cc_session_id = parts
                .headers
                .get("x-claude-code-session-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let bytes = match body.collect().await {
                Ok(c) => c.to_bytes(),
                Err(_) => return Request::from_parts(parts, Body::empty()).into(),
            };
            // Run the CPU-bound compression on the blocking pool so a burst of large requests
            // can't monopolize the async workers (which would stall response streaming for
            // everyone). Cheap to move in: `Bytes` is ref-counted, `config` is an `Arc`.
            let config = self.config.clone();
            let memo = self.memo.clone();
            let body_for_compress = bytes.clone();
            // Gemini/Vertex carry the model in the URL, not the body; capture it so the
            // capability gate can see it (the body-only lookup returns nothing for them).
            let model_override = model_from_path(parts.uri.path()).map(str::to_string);
            let compressed = tokio::task::spawn_blocking(move || {
                compress_blocking(
                    &config,
                    &body_for_compress,
                    provider,
                    model_override.as_deref(),
                    &memo,
                    cc_session_id,
                )
            })
            .await
            .ok()
            .flatten();

            // Capture the (uncompressed) body for the fallback before the match may move
            // `bytes`. The Anthropic call still goes out compressed; only a rare fallback replay
            // re-translates this original body to the provider.
            let fallback_body = fallback_arm.as_ref().map(|_| bytes.to_vec());
            let new_body = match compressed {
                Some((json, mut pending)) => {
                    // We changed the body — remember the original so we can replay it verbatim
                    // if the upstream rejects our compressed version (4xx/5xx). Compression must
                    // never break the user's call. Passthrough rows (after == before: the
                    // original was forwarded) get no replay copy — replaying an identical body
                    // can't fix anything.
                    if pending.input_after < pending.input_before
                        && let Some(host) = host.as_deref()
                    {
                        let path = parts.uri.path_and_query().map_or("/", |p| p.as_str());
                        pending.original = Some(OriginalRequest {
                            url: format!("https://{host}{path}"),
                            headers: forward_headers(&parts.headers),
                            body: bytes.to_vec(),
                        });
                    }
                    self.pending = Some(pending);
                    Body::from(json)
                }
                None => Body::from(Full::new(bytes)),
            };
            // Fallback mode: attach the provider fallback to the pending (creating a bare pending
            // when compression produced one), so the response phase can replay to the provider on
            // an Anthropic quota/overload status. We only arm the fallback on a *real* pending:
            // when `compress_blocking` declined (malformed body → no pending), we leave it None so
            // the request is forwarded verbatim and no phantom zero-token row is recorded. Such a
            // body would fail at Anthropic anyway, so losing the fallback there is harmless.
            if let Some((providers, session_id)) = fallback_arm
                && let Some(body) = fallback_body
                && let Some(pending) = self.pending.as_mut()
            {
                pending.fallback = Some(FallbackInfo {
                    providers,
                    anthropic_body: body,
                    session_id,
                });
            }
            let mut parts = parts;
            // Length changed; drop it so hyper recomputes (and never streams a stale value).
            parts.headers.remove(header::CONTENT_LENGTH);
            // When we'll tee + measure this response, force identity encoding so the body is
            // readable SSE/JSON for the usage/text parse — we don't bundle a decompressor.
            // Only when we actually compressed (`pending` set); passthroughs keep compression.
            if self.pending.is_some() {
                parts.headers.remove(header::ACCEPT_ENCODING);
            }
            Request::from_parts(parts, new_body).into()
        }

        /// Answer Claude Code's `/v1/messages/count_tokens` locally (see
        /// [`crate::reroute::count_tokens_json`]). The request can't be proxied to Anthropic once a
        /// `sub` is active (it would be billed against the sub), so short-circuit with a JSON reply.
        async fn reroute_count_tokens(&mut self, req: Request<Body>) -> RequestOrResponse {
            let (_parts, body) = req.into_parts();
            let bytes = body
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            let json = std::str::from_utf8(&bytes)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
                .map(|v| crate::reroute::count_tokens_json(&v))
                .unwrap_or_else(|| serde_json::json!({ "input_tokens": 0 }));
            json_response(200, &json.to_string()).into()
        }

        /// Translate an intercepted Anthropic `/v1/messages` request onto the `sub` provider's
        /// backend: optionally compress, translate the body, swap in the subscription auth, and
        /// rewrite the URI authority so hudsucker forwards to the provider. `handle_response` then
        /// translates the streamed reply back to Anthropic SSE. On any pre-flight failure (no auth,
        /// bad body, translation error) returns a synthetic Anthropic error the client renders —
        /// there is no Anthropic replay net here (replaying an Anthropic body to Codex is invalid).
        async fn reroute_messages(
            &mut self,
            req: Request<Body>,
            sub: crate::reroute::SubProvider,
        ) -> RequestOrResponse {
            let (mut parts, body) = req.into_parts();
            let session_id = parts
                .headers
                .get("x-claude-code-session-id")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let Ok(collected) = body.collect().await else {
                return anthropic_error(400, "llmtrim: could not read request body").into();
            };
            let bytes = collected.to_bytes();
            let Ok(anthropic) = std::str::from_utf8(&bytes)
                .ok()
                .and_then(|t| serde_json::from_str::<serde_json::Value>(t).ok())
                .ok_or(())
            else {
                return anthropic_error(400, "llmtrim: request body is not JSON").into();
            };

            // Compression stacks in front: compress the Anthropic body, then translate the
            // compressed form. On no win, translate the original. Runs on the blocking pool.
            let config = self.config.clone();
            let memo = self.memo.clone();
            let body_for_compress = bytes.clone();
            let started = std::time::Instant::now();
            let cc_session_id = session_id.clone();
            let compressed = tokio::task::spawn_blocking(move || {
                compress_blocking(
                    &config,
                    &body_for_compress,
                    ProviderKind::Anthropic,
                    None,
                    &memo,
                    cc_session_id,
                )
            })
            .await
            .ok()
            .flatten();
            let (mut translate_value, input_before, input_after, tokenizer, exact) =
                match &compressed {
                    Some((json, pending)) => (
                        serde_json::from_str::<serde_json::Value>(json)
                            .unwrap_or_else(|_| anthropic.clone()),
                        pending.input_before,
                        pending.input_after,
                        pending.tokenizer.clone(),
                        pending.exact,
                    ),
                    None => (anthropic.clone(), 0, 0, String::new(), false),
                };
            apply_sub_effort(&mut translate_value, self.sub_effort.as_deref());

            // Fetch the subscription token (OAuth refresh is blocking + single-flight).
            let token = match tokio::task::spawn_blocking(move || {
                crate::reroute::auth::get_token(sub)
            })
            .await
            {
                Ok(Ok(t)) => t,
                Ok(Err(e)) => {
                    return anthropic_error(
                        401,
                        &format!(
                            "llmtrim: {p} not authenticated ({e}). Run `llmtrim sub auth {p} login`.",
                            p = sub.as_str()
                        ),
                    )
                    .into();
                }
                Err(_) => return anthropic_error(500, "llmtrim: auth task failed").into(),
            };

            let rewrite = match crate::reroute::build_upstream(
                sub,
                &translate_value,
                &self.sub_tiers,
                &token,
                session_id.as_deref(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    return anthropic_error(
                        502,
                        &format!("llmtrim: reroute translation failed: {e}"),
                    )
                    .into();
                }
            };

            // Apply Codex continuation (previous_response_id + delta input) if possible.
            // `logical_body` is the full pre-delta codex request (used for recording the
            // transcript and for re-issuing on continuation errors). `sent_body` may be a
            // delta version. This matches the proxy's handling (except transport specifics).
            let mut sent_body = rewrite.body.clone();
            let mut logical_body: Option<Value> = None;
            if sub == crate::reroute::SubProvider::Codex
                && let Ok(mut bval) = serde_json::from_slice::<Value>(&rewrite.body)
            {
                logical_body = Some(bval.clone());
                let enabled =
                    llmtrim_core::config::RuntimeConfig::get().sub_codex_previous_response_id;
                let cand = crate::reroute::continuation::continuation_candidate(
                    session_id.as_deref(),
                    &bval,
                    enabled,
                );
                crate::reroute::continuation::apply_codex_continuation(&mut bval, &cand);
                if let Ok(serialized) = serde_json::to_vec(&bval) {
                    sent_body = serialized;
                }
            }

            // Record intent: provider stays Anthropic (we emit Anthropic SSE, which `Finalize`
            // measures); model is the resolved upstream model; `reroute` marks the response path.
            self.pending = Some(Pending {
                provider: ProviderKind::Anthropic,
                model: Some(rewrite.model.clone()),
                tokenizer,
                exact,
                input_before,
                input_after,
                compress_micros: started.elapsed().as_micros() as i64,
                plan: String::new(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: Some(RerouteInfo {
                    provider: sub,
                    model: rewrite.model.clone(),
                    client_model: anthropic
                        .get("model")
                        .and_then(|model| model.as_str())
                        .unwrap_or_default()
                        .to_string(),
                    replay: Some(RerouteReplay {
                        url: format!("https://{}{}", rewrite.host, rewrite.path),
                        headers: rewrite.headers.clone(),
                        body: Arc::new(sent_body.clone()),
                    }),
                    logical_body,
                    session_id: session_id.clone(),
                }),
                fallback: None,
            });

            // Retarget the request onto the provider host + path.
            if let Ok(uri) =
                format!("https://{}{}", rewrite.host, rewrite.path).parse::<hudsucker::hyper::Uri>()
            {
                parts.uri = uri;
            }
            // Strip the client's Anthropic auth + hop headers; hyper sets host/content-length.
            parts.headers.remove(header::HOST);
            parts.headers.remove(header::CONTENT_LENGTH);
            parts.headers.remove(header::ACCEPT_ENCODING);
            parts.headers.remove(header::AUTHORIZATION);
            parts.headers.remove("x-api-key");
            let anthropic_keys: Vec<_> = parts
                .headers
                .keys()
                .filter(|k| k.as_str().starts_with("anthropic-"))
                .cloned()
                .collect();
            for k in anthropic_keys {
                parts.headers.remove(k);
            }
            // Apply the provider headers.
            for (k, v) in &rewrite.headers {
                if let (Ok(name), Ok(val)) = (
                    header::HeaderName::from_bytes(k.as_bytes()),
                    header::HeaderValue::from_str(v),
                ) {
                    parts.headers.insert(name, val);
                }
            }
            Request::from_parts(parts, Body::from(sent_body)).into()
        }

        /// Stream a rerouted provider response back to the client as Anthropic SSE: feed each
        /// upstream chunk through the provider reducer + shared encoder incrementally (no
        /// buffering), accumulating the emitted Anthropic bytes so `Finalize` measures output. An
        /// upstream non-2xx is surfaced as an Anthropic SSE `error` frame.
        async fn reroute_response(
            &mut self,
            res: Response<Body>,
            pending: Pending,
            info: RerouteInfo,
        ) -> Response<Body> {
            use crate::reroute::sse::{AnthropicSseEncoder, ReduceEvent};
            let status = res.status();
            let (parts, body) = res.into_parts();

            if !status.is_success() {
                let mut cur_status = status.as_u16();
                let mut cur_body = body
                    .collect()
                    .await
                    .map(|c| c.to_bytes().to_vec())
                    .unwrap_or_default();
                let mut retry_after = reroute_retry_after_from_headers(&parts.headers, &cur_body);

                // Server-side retry for transient / rate-limit failures (mirrors what a native
                // Anthropic client does): re-issue the same upstream request with backoff, honoring
                // the reset hint. A multi-hour usage-limit reset exceeds the budget and is surfaced
                // at once rather than burning pointless retries.
                if let Some(replay) = info.replay.as_ref() {
                    let mut attempt = 0;
                    while attempt < REROUTE_RETRY_MAX && reroute_should_retry(cur_status) {
                        let Some(wait_ms) = reroute_backoff_ms(attempt, retry_after) else {
                            break;
                        };
                        tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
                        attempt += 1;

                        let body_to_send = if is_continuation_error(&cur_body) {
                            crate::reroute::continuation::clear_continuation(
                                info.session_id.as_deref(),
                            );
                            if let Some(lval) = &info.logical_body {
                                serde_json::to_vec(lval)
                                    .map(Arc::new)
                                    .unwrap_or_else(|_| replay.body.clone())
                            } else {
                                replay.body.clone()
                            }
                        } else {
                            replay.body.clone()
                        };

                        let Some((s, raw, ra)) = self
                            .reissue_reroute(&replay.url, replay.headers.clone(), body_to_send)
                            .await
                        else {
                            break;
                        };
                        if (200..300).contains(&s) {
                            return self.finish_buffered_reroute(pending, &info, raw);
                        }
                        cur_status = s;
                        cur_body = raw;
                        retry_after = ra;
                    }
                }

                // Surface the upstream failure as a real non-2xx Anthropic error (like the reroute
                // pre-flight failures), not a 200 SSE `error` frame: Claude Code renders a bare
                // in-stream error event that never had a `message_start` as "empty or malformed
                // response (HTTP 200)", hiding the actual cause (e.g. a rate-limited subscription).
                let message = reroute_upstream_error_message(info.provider, cur_status, &cur_body);
                let acc = Arc::new(Mutex::new(message.clone().into_bytes()));
                // Record the row (output 0) as this Finalize drops.
                let _finalize = Finalize {
                    acc,
                    pending: Some(pending),
                    ledger: self.ledger.clone(),
                    breakdown_ledger: self.breakdown_ledger.clone(),
                };
                // Emit `Retry-After` for a rate limit, capped so the client backs off without
                // stalling for the full (possibly multi-hour) reset — the true reset is in the
                // message. Type the error by status so it renders as a rate limit / auth failure.
                let retry_after_header = (cur_status == 429)
                    .then_some(retry_after)
                    .flatten()
                    .map(|s| s.min(60));
                return anthropic_error_typed(
                    cur_status,
                    reroute_error_kind(cur_status),
                    &message,
                    retry_after_header,
                );
            }

            // Peek the head of the 2xx stream: drive the reducer until its first content event or a
            // terminal. An upstream that fails the whole turn on an HTTP 200 stream (e.g.
            // `context_length_exceeded` as an `error`/`response.failed`) is surfaced as a real typed
            // error instead of a 200 stream Claude Code rejects as empty/malformed.
            use hudsucker::futures::StreamExt;
            let mut inner = BodyStream::new(body);
            let mut reducer = crate::reroute::StreamReducer::new(info.provider, &info.model);
            let mut encoder = AnthropicSseEncoder::new(&info.client_model);
            let mut prelude = String::new();
            let mut committed = false;
            let mut fatal: Option<String> = None;
            while !committed && fatal.is_none() {
                match inner.next().await {
                    Some(Ok(frame)) => {
                        let Ok(chunk) = frame.into_data() else {
                            continue;
                        };
                        for ev in reducer.push(&chunk) {
                            record_codex_continuation(
                                &ev,
                                &mut reducer,
                                info.provider,
                                info.logical_body.as_ref(),
                                info.session_id.as_deref(),
                            );
                            match &ev {
                                ReduceEvent::Error { message } => {
                                    fatal.get_or_insert_with(|| message.clone());
                                }
                                ReduceEvent::Finish { .. } => {}
                                _ => committed = true,
                            }
                            encoder.encode(&ev, &mut prelude);
                        }
                    }
                    Some(Err(_)) => {
                        fatal.get_or_insert_with(|| "llmtrim: upstream stream error".to_string());
                    }
                    None => break,
                }
            }

            if let Some(detail) = fatal {
                crate::reroute::continuation::clear_continuation(info.session_id.as_deref());
                let status = reroute_stream_error_status(&detail);
                let message = format!(
                    "llmtrim: {} subscription backend: {detail}",
                    info.provider.as_str()
                );
                let acc = Arc::new(Mutex::new(message.clone().into_bytes()));
                let _finalize = Finalize {
                    acc,
                    pending: Some(pending),
                    ledger: self.ledger.clone(),
                    breakdown_ledger: self.breakdown_ledger.clone(),
                };
                return anthropic_error_typed(status, reroute_error_kind(status), &message, None);
            }

            let acc = Arc::new(Mutex::new(Vec::<u8>::new()));
            let finalize = Finalize {
                acc: acc.clone(),
                pending: Some(pending),
                ledger: self.ledger.clone(),
                breakdown_ledger: self.breakdown_ledger.clone(),
            };

            enum Phase {
                Prelude,
                Body,
                Flush,
                Done,
            }
            struct St {
                inner: BodyStream<Body>,
                reducer: crate::reroute::StreamReducer,
                encoder: AnthropicSseEncoder,
                acc: Arc<Mutex<Vec<u8>>>,
                finalize: Finalize,
                phase: Phase,
                prelude: String,
                // For continuation recording on terminal Finish during live stream
                provider: crate::reroute::SubProvider,
                logical_body: Option<Value>,
                session_id: Option<String>,
            }
            let st = St {
                inner,
                reducer,
                encoder,
                acc: acc.clone(),
                finalize,
                phase: Phase::Prelude,
                prelude,
                provider: info.provider,
                logical_body: info.logical_body.clone(),
                session_id: info.session_id.clone(),
            };

            let stream = hudsucker::futures::stream::unfold(st, |mut st| async move {
                loop {
                    match st.phase {
                        Phase::Done => return None,
                        Phase::Prelude => {
                            // Emit the bytes buffered while peeking, then stream the rest.
                            st.phase = Phase::Body;
                            if st.prelude.is_empty() {
                                continue;
                            }
                            let out = std::mem::take(&mut st.prelude);
                            if let Ok(mut buf) = st.acc.lock() {
                                buf.extend_from_slice(out.as_bytes());
                            }
                            return Some((
                                Ok::<Bytes, std::io::Error>(Bytes::from(out.into_bytes())),
                                st,
                            ));
                        }
                        Phase::Flush => {
                            let mut out = String::new();
                            for ev in st.reducer.finish() {
                                record_codex_continuation(
                                    &ev,
                                    &mut st.reducer,
                                    st.provider,
                                    st.logical_body.as_ref(),
                                    st.session_id.as_deref(),
                                );
                                st.encoder.encode(&ev, &mut out);
                            }
                            st.encoder.finish_if_open(&mut out);
                            st.phase = Phase::Done;
                            if out.is_empty() {
                                // Keep `finalize` alive until the stream is fully drained.
                                let _ = &st.finalize;
                                return None;
                            }
                            if let Ok(mut buf) = st.acc.lock() {
                                buf.extend_from_slice(out.as_bytes());
                            }
                            let bytes = Bytes::from(out.into_bytes());
                            return Some((Ok::<Bytes, std::io::Error>(bytes), st));
                        }
                        Phase::Body => match st.inner.next().await {
                            Some(Ok(frame)) => {
                                let Ok(chunk) = frame.into_data() else {
                                    continue;
                                };
                                let mut out = String::new();
                                for ev in st.reducer.push(&chunk) {
                                    record_codex_continuation(
                                        &ev,
                                        &mut st.reducer,
                                        st.provider,
                                        st.logical_body.as_ref(),
                                        st.session_id.as_deref(),
                                    );
                                    st.encoder.encode(&ev, &mut out);
                                }
                                if out.is_empty() {
                                    continue;
                                }
                                if let Ok(mut buf) = st.acc.lock() {
                                    buf.extend_from_slice(out.as_bytes());
                                }
                                let bytes = Bytes::from(out.into_bytes());
                                return Some((Ok(bytes), st));
                            }
                            Some(Err(_)) => {
                                // Transport error mid-stream: surface it as an Anthropic `error`
                                // frame so the client can tell a dropped provider connection from a
                                // clean end-of-turn, instead of a silently truncated answer.
                                let mut out = String::new();
                                st.encoder.encode(
                                    &ReduceEvent::Error {
                                        message: "llmtrim: upstream stream error".to_string(),
                                    },
                                    &mut out,
                                );
                                st.phase = Phase::Flush;
                                if out.is_empty() {
                                    continue;
                                }
                                if let Ok(mut buf) = st.acc.lock() {
                                    buf.extend_from_slice(out.as_bytes());
                                }
                                return Some((Ok(Bytes::from(out.into_bytes())), st));
                            }
                            None => {
                                st.phase = Phase::Flush;
                                continue;
                            }
                        },
                    }
                }
            });

            let mut builder = Response::builder().status(200);
            builder = builder.header(header::CONTENT_TYPE, "text/event-stream");
            builder = builder.header(header::CACHE_CONTROL, "no-cache");
            builder
                .body(Body::from_stream(stream))
                .unwrap_or_else(|_| Response::new(Body::empty()))
        }

        /// Re-issue a rerouted upstream request (blocking, buffered `forward_post` like the replay
        /// net) for a retry attempt. Returns `(status, body, reset-hint-seconds)`, or `None` if the
        /// round-trip or its task failed (the caller stops retrying and surfaces the last error).
        async fn reissue_reroute(
            &self,
            url: &str,
            headers: Vec<(String, String)>,
            body: Arc<Vec<u8>>,
        ) -> Option<(u16, Vec<u8>, Option<u64>)> {
            let url = url.to_string();
            let proxy = self.upstream_proxy.clone();
            // `forward_post` exposes no response headers, so a retried attempt's reset hint comes
            // from the body (`resets_in_seconds`) — enough for the Codex/Kimi usage-limit shape.
            let (status, raw) = tokio::task::spawn_blocking(move || {
                use std::io::Read;
                let body_str = String::from_utf8_lossy(&body);
                let mut up =
                    crate::transport::forward_post(&url, &headers, &body_str, proxy.as_deref())
                        .map_err(|e| e.to_string())?;
                let mut buf = Vec::new();
                up.reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                Ok::<(u16, Vec<u8>), String>((up.status, buf))
            })
            .await
            .ok()?
            .ok()?;
            let retry_after = reroute_retry_after_secs(None, None, &raw);
            Some((status, raw, retry_after))
        }

        /// Translate a buffered, successful retry response into Anthropic SSE in one shot (the rare
        /// retry path, not the streaming hot path) and record the row on `Finalize` drop.
        fn finish_buffered_reroute(
            &self,
            pending: Pending,
            info: &RerouteInfo,
            raw: Vec<u8>,
        ) -> Response<Body> {
            use crate::reroute::sse::AnthropicSseEncoder;
            let mut enc = AnthropicSseEncoder::new(&info.client_model);
            let mut out = String::new();
            let mut reducer = crate::reroute::StreamReducer::new(info.provider, &info.model);
            for ev in reducer.push(&raw) {
                record_codex_continuation(
                    &ev,
                    &mut reducer,
                    info.provider,
                    info.logical_body.as_ref(),
                    info.session_id.as_deref(),
                );
                enc.encode(&ev, &mut out);
            }
            for ev in reducer.finish() {
                record_codex_continuation(
                    &ev,
                    &mut reducer,
                    info.provider,
                    info.logical_body.as_ref(),
                    info.session_id.as_deref(),
                );
                enc.encode(&ev, &mut out);
            }
            enc.finish_if_open(&mut out);

            let acc = Arc::new(Mutex::new(out.clone().into_bytes()));
            let _finalize = Finalize {
                acc,
                pending: Some(pending),
                ledger: self.ledger.clone(),
                breakdown_ledger: self.breakdown_ledger.clone(),
            };
            sse_response(out)
        }

        /// Provider-chain fallback. Each backend gets bounded retries for transient HTTP errors;
        /// a terminal response, auth failure, translation failure, or failed 200 stream advances
        /// to the next configured provider.
        async fn fallback_to_chain(
            &mut self,
            mut pending: Pending,
            fb: FallbackInfo,
        ) -> Response<Body> {
            use crate::reroute::sse::{AnthropicSseEncoder, ReduceEvent};
            let Some(mut anthropic) = std::str::from_utf8(&fb.anthropic_body)
                .ok()
                .and_then(|t| serde_json::from_str::<Value>(t).ok())
            else {
                return self.fallback_error(
                    pending,
                    "unknown",
                    "llmtrim: fallback body is not JSON",
                );
            };
            let client_model = anthropic
                .get("model")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();
            apply_sub_effort(&mut anthropic, self.sub_effort.as_deref());

            let mut providers = Vec::new();
            for provider in fb.providers.iter().copied() {
                if !providers.contains(&provider) {
                    providers.push(provider);
                }
            }
            // One wall-clock budget for the whole chain: per-provider backoff × N providers can
            // otherwise outlast the client's own timeout, and a client that has already given up
            // makes every further attempt pure waste (and pure spend).
            let deadline = std::time::Instant::now() + FALLBACK_CHAIN_BUDGET;
            let mut failures = Vec::new();
            for provider in providers {
                let attempt = match self
                    .fallback_attempt(provider, &anthropic, fb.session_id.as_deref(), deadline)
                    .await
                {
                    Ok(attempt) => attempt,
                    Err(error) => {
                        failures.push(format!("{}: {error}", provider.as_str()));
                        continue;
                    }
                };
                let mut reducer =
                    crate::reroute::StreamReducer::new(attempt.provider, &attempt.model);
                let mut encoder = AnthropicSseEncoder::new(&client_model);
                let mut output = String::new();
                let mut stream_error = None;
                for event in reducer.push(&attempt.body) {
                    record_codex_continuation(
                        &event,
                        &mut reducer,
                        attempt.provider,
                        attempt.logical_body.as_ref(),
                        fb.session_id.as_deref(),
                    );
                    if let ReduceEvent::Error { message } = &event {
                        stream_error = Some(message.clone());
                    }
                    encoder.encode(&event, &mut output);
                }
                if stream_error.is_none() {
                    for event in reducer.finish() {
                        record_codex_continuation(
                            &event,
                            &mut reducer,
                            attempt.provider,
                            attempt.logical_body.as_ref(),
                            fb.session_id.as_deref(),
                        );
                        if let ReduceEvent::Error { message } = &event {
                            stream_error = Some(message.clone());
                        }
                        encoder.encode(&event, &mut output);
                    }
                }
                if let Some(error) = stream_error {
                    crate::reroute::continuation::clear_continuation(fb.session_id.as_deref());
                    failures.push(format!("{}: {error}", attempt.provider.as_str()));
                    continue;
                }
                encoder.finish_if_open(&mut output);
                pending.model = Some(attempt.model.clone());
                pending.reroute = Some(RerouteInfo {
                    provider: attempt.provider,
                    model: attempt.model,
                    client_model: client_model.clone(),
                    replay: None,
                    logical_body: attempt.logical_body,
                    session_id: fb.session_id.clone(),
                });
                let acc = Arc::new(Mutex::new(output.clone().into_bytes()));
                let _finalize = Finalize {
                    acc,
                    pending: Some(pending),
                    ledger: self.ledger.clone(),
                    breakdown_ledger: self.breakdown_ledger.clone(),
                };
                return sse_response(output);
            }
            self.fallback_error(
                pending,
                &client_model,
                &format!(
                    "llmtrim: all fallback providers failed: {}",
                    failures.join("; ")
                ),
            )
        }

        async fn fallback_attempt(
            &self,
            provider: crate::reroute::SubProvider,
            anthropic: &Value,
            session_id: Option<&str>,
            deadline: std::time::Instant,
        ) -> Result<FallbackAttempt, String> {
            if std::time::Instant::now() >= deadline {
                return Err("fallback budget exhausted".to_string());
            }
            let token =
                tokio::task::spawn_blocking(move || crate::reroute::auth::get_token(provider))
                    .await
                    .map_err(|_| "auth task failed".to_string())?
                    .map_err(|e| format!("not authenticated ({e})"))?;
            let rewrite = crate::reroute::build_upstream(
                provider,
                anthropic,
                &self.sub_tiers,
                &token,
                session_id,
            )
            .map_err(|e| format!("translation failed: {e}"))?;
            // Same Codex continuation handling as the single-provider path: `logical_body` is the
            // full pre-delta request kept for the transcript, while the request actually sent may
            // carry `previous_response_id` plus only the new input. Without this the
            // `record_codex_continuation` call on this path would store state nothing ever reads.
            let mut sent_body = rewrite.body.clone();
            let mut logical_body = None;
            if provider == crate::reroute::SubProvider::Codex
                && let Ok(mut bval) = serde_json::from_slice::<Value>(&rewrite.body)
            {
                logical_body = Some(bval.clone());
                let enabled =
                    llmtrim_core::config::RuntimeConfig::get().sub_codex_previous_response_id;
                let cand = crate::reroute::continuation::continuation_candidate(
                    session_id, &bval, enabled,
                );
                crate::reroute::continuation::apply_codex_continuation(&mut bval, &cand);
                if let Ok(serialized) = serde_json::to_vec(&bval) {
                    sent_body = serialized;
                }
            }
            let url = format!("https://{}{}", rewrite.host, rewrite.path);
            let headers = rewrite.headers.clone();
            let body = String::from_utf8_lossy(&sent_body).into_owned();
            let proxy = self.upstream_proxy.clone();
            let mut attempt = 0;
            loop {
                let url = url.clone();
                let headers = headers.clone();
                let body = body.clone();
                let proxy = proxy.clone();
                let result = tokio::task::spawn_blocking(move || {
                    use std::io::Read;
                    let mut up =
                        crate::transport::forward_post(&url, &headers, &body, proxy.as_deref())
                            .map_err(|e| e.to_string())?;
                    let mut buf = Vec::new();
                    up.reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                    let retry_after =
                        reroute_retry_after_secs(up.retry_after.as_deref(), None, &buf);
                    Ok::<(u16, Vec<u8>, Option<u64>), String>((up.status, buf, retry_after))
                })
                .await
                .map_err(|_| "fallback task failed".to_string())?
                .map_err(|e| format!("request failed: {e}"))?;
                let (status, body, retry_after) = result;
                if (200..300).contains(&status) {
                    return Ok(FallbackAttempt {
                        provider,
                        model: rewrite.model,
                        logical_body,
                        body,
                    });
                }
                if !reroute_should_retry(status) || attempt >= REROUTE_RETRY_MAX {
                    let snippet: String =
                        String::from_utf8_lossy(&body).chars().take(300).collect();
                    return Err(format!("HTTP {status}: {snippet}"));
                }
                let Some(wait_ms) = reroute_backoff_ms(attempt, retry_after) else {
                    return Err(format!("HTTP {status}: retry window exceeds budget"));
                };
                // Don't burn the chain's whole budget waiting on one provider: if this backoff
                // would eat into the deadline, give the next provider its turn instead.
                let wait = std::time::Duration::from_millis(wait_ms);
                if std::time::Instant::now() + wait >= deadline {
                    return Err(format!(
                        "HTTP {status}: retry window exceeds fallback budget"
                    ));
                }
                tokio::time::sleep(wait).await;
                attempt += 1;
            }
        }

        /// One bounded retry of the turn against Anthropic before the chain takes over. A 503 or a
        /// short 429 is usually a blip, and a blip should not move a turn (and its spend) to a
        /// different provider. Returns the `Pending` back when no retry was possible or the retry
        /// failed, so the caller falls through to the chain. The retry sends the *original*
        /// (uncompressed) request — the compressed body isn't retained past the forward — so the
        /// row is recorded honestly as a no-savings turn.
        async fn retry_anthropic_once(
            &self,
            mut pending: Pending,
            retry_after_secs: Option<u64>,
        ) -> Result<Response<Body>, Pending> {
            let Some(original) = pending.original.clone() else {
                return Err(pending);
            };
            // A reset hint beyond the backoff cap (a usage limit hours away) means "don't wait" —
            // go straight to the chain, which is the whole point of fallback mode.
            let Some(wait_ms) = reroute_backoff_ms(0, retry_after_secs) else {
                return Err(pending);
            };
            tokio::time::sleep(std::time::Duration::from_millis(wait_ms)).await;
            let proxy = self.upstream_proxy.clone();
            let fetched =
                tokio::task::spawn_blocking(move || fetch_original(&original, proxy.as_deref()))
                    .await;
            let Ok(Some((status, content_type, body))) = fetched else {
                return Err(pending);
            };
            if !(200..300).contains(&status) || is_sub_fallback_body(&body) {
                return Err(pending);
            }
            pending.input_after = pending.input_before;
            pending.output_shaped = false;
            let _finalize = Finalize {
                acc: Arc::new(Mutex::new(body.clone())),
                pending: Some(pending),
                ledger: self.ledger.clone(),
                breakdown_ledger: self.breakdown_ledger.clone(),
            };
            Ok(buffered_response(status, content_type, body))
        }

        /// Emit an Anthropic SSE `error` frame for a fallback that failed before/at the
        /// provider round-trip, still recording the row (output 0) as the `Finalize` drops.
        fn fallback_error(&self, pending: Pending, model: &str, message: &str) -> Response<Body> {
            use crate::reroute::sse::{AnthropicSseEncoder, ReduceEvent};
            let mut enc = AnthropicSseEncoder::new(model);
            let mut out = String::new();
            enc.encode(
                &ReduceEvent::Error {
                    message: message.to_string(),
                },
                &mut out,
            );
            let acc = Arc::new(Mutex::new(out.clone().into_bytes()));
            let _finalize = Finalize {
                acc,
                pending: Some(pending),
                ledger: self.ledger.clone(),
                breakdown_ledger: self.breakdown_ledger.clone(),
            };
            sse_response(out)
        }

        /// Body of `handle_response`, factored out so tests can call it directly without
        /// a `hudsucker::HttpContext`.
        async fn handle_response_inner(&mut self, res: Response<Body>) -> Response<Body> {
            let Some(pending) = self.pending.take() else {
                return res;
            };
            // Subscription reroute: the upstream reply is the provider's SSE — translate it back to
            // Anthropic SSE (the normal compress/replay/tee path below is for verbatim-forwarded
            // Anthropic responses).
            if let Some(info) = pending.reroute.clone() {
                return self.reroute_response(res, pending, info).await;
            }
            // Fallback reroute: Anthropic could not serve the turn. On a *transient* status
            // (429/5xx) Anthropic gets one bounded retry first — a blip should not hand the turn,
            // and its spend, to another provider — and only then does the chain take over.
            if let Some(fb) = pending.fallback.clone()
                && is_sub_fallback_status(res.status().as_u16())
            {
                let status = res.status().as_u16();
                let (parts, body) = res.into_parts();
                let bytes = body
                    .collect()
                    .await
                    .map(|c| c.to_bytes().to_vec())
                    .unwrap_or_default();
                let mut pending = pending;
                if reroute_should_retry(status) {
                    let hint = reroute_retry_after_from_headers(&parts.headers, &bytes);
                    match self.retry_anthropic_once(pending, hint).await {
                        Ok(response) => return response,
                        Err(returned) => pending = returned,
                    }
                }
                return self.fallback_to_chain(pending, fb).await;
            }
            // An upstream can also report a failed turn inside an HTTP 200 body. In fallback mode
            // we peek the *head* of the stream — up to the first error frame or the first frame
            // that commits content to the client — and then forward the rest untouched. Buffering
            // the whole response here would cost every fallback-mode user their streaming.
            let mut res = res;
            if let Some(fb) = pending.fallback.clone()
                && (200..300).contains(&res.status().as_u16())
            {
                let content_type = res
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or_default()
                    .to_string();
                if content_type.contains("event-stream") {
                    let (parts, body) = res.into_parts();
                    match probe_sse_head(body).await {
                        Ok(body) => res = Response::from_parts(parts, body),
                        Err(()) => return self.fallback_to_chain(pending, fb).await,
                    }
                } else if content_type.contains("json") {
                    // Non-streamed JSON is a single small object: buffering it costs nothing.
                    let (parts, body) = res.into_parts();
                    let bytes = body
                        .collect()
                        .await
                        .map(|c| c.to_bytes().to_vec())
                        .unwrap_or_default();
                    if is_sub_fallback_body(&bytes) {
                        return self.fallback_to_chain(pending, fb).await;
                    }
                    res = Response::from_parts(parts, Body::from(Full::new(Bytes::from(bytes))));
                }
            }
            // Safety net: if the upstream rejected our compressed request with a *validation*
            // error (400 Bad Request / 422 Unprocessable) — the failure class our body edits can
            // cause — replay the original verbatim and hand the client THAT. We deliberately do
            // NOT replay 429/5xx (rate-limit, overload, server faults): those are unrelated to
            // our mutation, and re-sending would double provider load during an incident and
            // strip the client's `retry-after`. Only when we actually changed the body.
            let status = res.status();
            if should_replay(status.as_u16())
                && let Some(original) = pending.original.clone()
                && let Ok(Some(replayed)) = {
                    let proxy = self.upstream_proxy.clone();
                    tokio::task::spawn_blocking(move || {
                        replay_original(&original, proxy.as_deref())
                    })
                    .await
                }
            {
                eprintln!(
                    "llmtrim: upstream {} on compressed request — replayed original (no compression this call)",
                    status.as_u16()
                );
                // Record the fall-back honestly: the original was sent, so zero savings —
                // and the original carried no shaping instruction either.
                let mut rec = record_from(pending, ResponseUsage::default());
                rec.input_after = rec.input_before;
                rec.output_shaped = Some(false);
                let _ = self.ledger.send(rec);
                return replayed;
            }
            let is_sse = res
                .headers()
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .map(|c| c.contains("event-stream"))
                .unwrap_or(false);

            // Output-transform path (reserved): if a plan carried a reversible output
            // transform, we'd expand it here before the client sees it. None ship today, so
            // `plan_reverses_output` is always false and this branch is inert.
            if plan_reverses_output(&pending.plan) && !is_sse {
                let (parts, body) = res.into_parts();
                let bytes = body
                    .collect()
                    .await
                    .map(|c| c.to_bytes().to_vec())
                    .unwrap_or_default();
                let out = rehydrate_response(&bytes, &pending, &self.ledger);
                return Response::from_parts(parts, Body::from(out));
            }
            let (parts, body) = res.into_parts();
            let acc = Arc::new(Mutex::new(Vec::<u8>::new()));
            // `Finalize` records the full row (input + measured output) when the streamed
            // body is fully sent (or aborted) — i.e. when this closure is dropped.
            let finalize = Finalize {
                acc: acc.clone(),
                pending: Some(pending),
                ledger: self.ledger.clone(),
                breakdown_ledger: self.breakdown_ledger.clone(),
            };
            use hudsucker::futures::StreamExt;
            let stream = BodyStream::new(body).filter_map(move |frame| {
                let out = match frame {
                    Ok(frame) => match frame.into_data() {
                        Ok(bytes) => {
                            if let Ok(mut buf) = acc.lock() {
                                buf.extend_from_slice(&bytes);
                            }
                            Some(Ok(bytes))
                        }
                        Err(_non_data) => None,
                    },
                    Err(e) => Some(Err(e)),
                };
                let _keep_alive = &finalize;
                hudsucker::futures::future::ready(out)
            });
            Response::from_parts(parts, Body::from_stream(stream))
        }
    }

    /// The CPU-bound compression, run on the blocking pool (see `handle_request`). Pure w.r.t.
    /// I/O: takes the request body + config + the turn-stability memo, returns the compressed
    /// JSON paired with the per-request `Pending` (its `original` left unset — the caller fills
    /// it). `None` to forward verbatim (not UTF-8/JSON, errored, or no net token win).
    /// Extract the model id from a Vertex/Gemini request path (`.../models/{model}:{method}`),
    /// which — unlike OpenAI/Anthropic — carries the model in the URL, not the body. OpenAI
    /// (`/v1/chat/completions`) and Anthropic (`/v1/messages`) paths have no `/models/` segment
    /// and return `None`, so the body's `model` is used for them.
    fn model_from_path(path: &str) -> Option<&str> {
        let after = path.rsplit_once("/models/")?.1;
        let model = after.split(':').next().unwrap_or(after);
        (!model.is_empty()).then_some(model)
    }

    fn compress_blocking(
        config: &DenseConfig,
        body: &[u8],
        provider: ProviderKind,
        model_override: Option<&str>,
        memo: &Memo,
        cc_session_id: Option<String>,
    ) -> Option<(String, Pending)> {
        let text = std::str::from_utf8(body).ok()?;
        if !text.trim_start().starts_with('{') {
            return None;
        }
        // Pick the adapter from the body shape, falling back to the host-derived kind. A host
        // that serves more than one wire shape (Anthropic- and OpenAI-compatible paths on the
        // same domain) is then adapted by what the body actually is, not the host guess.
        let kind = serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .and_then(|v| llmtrim_core::provider::detect(&v))
            .unwrap_or(provider);
        let started = std::time::Instant::now();
        let mut result =
            llmtrim_core::compress_with_config_model(text, Some(kind), config, model_override)
                .ok()?;
        // Never forward a request larger than we received. On tiny or non-chat bodies (e.g.
        // token-count / auxiliary calls) the input-side stages can't offset the output-control
        // instruction's fixed cost, so the compressed form is a net token *increase*. Forward
        // the original verbatim — the "never a bigger bill" guarantee — but still record the
        // zero-savings row: the dashboard's request count and savings %s must describe ALL
        // proxied chat traffic, not just the wins (else the % is a self-selected best case).
        //
        // The turn-stability memo runs only on the compression-*success* path below, so its
        // invariant is clean — it stores and replays exactly the bytes we forward as the
        // compressed body. Passthrough requests (the original is forwarded) are left stateless,
        // so a later turn never replays a compressed prefix the provider never actually cached.
        if result.input_tokens_after >= result.input_tokens_before {
            let pending = Pending {
                provider: kind,
                model: result.model.clone(),
                tokenizer: result.tokenizer_label.clone(),
                exact: result.tokenizer_exact,
                input_before: result.input_tokens_before.0 as i64,
                input_after: result.input_tokens_before.0 as i64,
                compress_micros: started.elapsed().as_micros() as i64,
                plan: String::new(),
                original: None,
                // The forwarded original carries no shaping instruction.
                output_shaped: false,
                frozen_input_tokens: Some(result.frozen_input_tokens.0 as i64),
                // Passthrough forwards the original verbatim — attribute that body.
                breakdown: attribute_for_breakdown(
                    text,
                    kind,
                    result.model.as_deref(),
                    cc_session_id.clone(),
                ),
                reroute: None,
                fallback: None,
            };
            return Some((text.to_string(), pending));
        }
        // Compression won: replay an already-seen conversation prefix's compressed bytes verbatim
        // across turns so the provider prefix cache stays warm (the highest-traffic agent-loop
        // shape). The memo only reuses bytes it itself produced for a byte-identical earlier
        // message and keeps the result ≤ the original, so it can't flip this win into a loss;
        // it recomputes `input_after` over the rewritten body so the ledger stays honest.
        apply_turn_memo(config, memo, kind, text, &mut result);
        let compress_micros = started.elapsed().as_micros() as i64;
        let pending = Pending {
            // The detected kind, not the host fallback — `handle_response` reads the response
            // usage with this provider's field shapes, so it must match what we compressed as.
            provider: kind,
            model: result.model.clone(),
            tokenizer: result.tokenizer_label.clone(),
            exact: result.tokenizer_exact,
            input_before: result.input_tokens_before.0 as i64,
            input_after: result.input_tokens_after.0 as i64,
            compress_micros,
            plan: serde_json::to_string(&result.plan).unwrap_or_default(),
            original: None,
            output_shaped: result.output_shaped,
            frozen_input_tokens: Some(result.frozen_input_tokens.0 as i64),
            // Attribute the compressed body we actually forward — its tokens are what the
            // provider bills, so it matches the usage we reconcile against.
            breakdown: attribute_for_breakdown(
                &result.request_json,
                kind,
                result.model.as_deref(),
                cc_session_id,
            ),
            reroute: None,
            fallback: None,
        };
        capture_pair(text, &result.request_json, &pending, &result.stages);
        Some((result.request_json, pending))
    }

    /// Opt-in QA capture: when a capture dir is set (`LLMTRIM_CAPTURE_DIR` env or `capture_dir`
    /// in the config file), write the before/after request bodies of each compressed request as
    /// one JSON file so an external reviewer can audit compression quality. Off unless set; any
    /// write failure is logged and swallowed — capture must never break proxying.
    fn capture_pair(
        before: &str,
        after: &str,
        pending: &Pending,
        stages: &[llmtrim_core::pipeline::StageReport],
    ) {
        let Some(dir_path) = RuntimeConfig::get().capture_dir.clone() else {
            return;
        };
        // The names of the stages that actually rewrote this request — what an external auditor
        // needs to tell a lossless run that dropped content (a bug) from a lossy stage doing its
        // job. `plan` below is the output-rehydration plan (reversible response-side transforms),
        // a different axis, so both are recorded.
        let stages_applied: Vec<&str> = stages
            .iter()
            .filter(|s| s.applied)
            .map(|s| s.name.as_str())
            .collect();
        let record = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "provider": pending.provider.as_str(),
            "model": pending.model,
            "input_before": pending.input_before,
            "input_after": pending.input_after,
            "output_shaped": pending.output_shaped,
            "stages": stages_applied,
            "plan": pending.plan,
            "before": before,
            "after": after,
        });
        let name = format!(
            "{}-{:x}.json",
            chrono::Utc::now().timestamp_micros(),
            std::process::id()
        );
        let path = dir_path.join(name);
        if let Err(e) = std::fs::create_dir_all(&dir_path)
            .and_then(|_| std::fs::write(&path, record.to_string()))
        {
            eprintln!("llmtrim: capture failed ({}): {e}", path.display());
        }

        // Bound the corpus so it can't fill the disk (which starves the daemon's pidfile and
        // ledger writes). Checked every N captures — read_dir over the whole corpus is too
        // costly to run per request. `LLMTRIM_CAPTURE_MAX_MB` sets the ceiling (default 1024;
        // 0 disables the cap). The sweep runs on a detached thread so the read_dir + per-file
        // stat over a large corpus never blocks the request that triggered it.
        static SINCE_CHECK: AtomicU64 = AtomicU64::new(0);
        const CHECK_EVERY: u64 = 512;
        if SINCE_CHECK
            .fetch_add(1, Ordering::Relaxed)
            .is_multiple_of(CHECK_EVERY)
        {
            let max_bytes = RuntimeConfig::get()
                .capture_max_mb
                .unwrap_or(1024)
                .saturating_mul(1024 * 1024);
            if max_bytes > 0 {
                let dir = dir_path.to_path_buf();
                std::thread::spawn(move || enforce_capture_cap(&dir, max_bytes));
            }
        }
    }

    /// Evict oldest capture files until the directory's total size is within `max_bytes`.
    /// Only the top-level `*.json` captures count (other files a user may keep in the dir are
    /// left alone, and subdirectories are not recursed into). Capture filenames are
    /// `<timestamp_micros>-<pid>.json`, so a lexicographic sort is chronological — the oldest
    /// go first. Best-effort: any I/O error just leaves the file.
    fn enforce_capture_cap(dir: &std::path::Path, max_bytes: u64) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut files: Vec<(std::ffi::OsString, u64)> = entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name();
                let is_capture = std::path::Path::new(&name)
                    .extension()
                    .is_some_and(|x| x == "json");
                let meta = e.metadata().ok()?;
                (is_capture && meta.is_file()).then_some((name, meta.len()))
            })
            .collect();
        let total: u64 = files.iter().map(|(_, len)| len).sum();
        let Some(mut to_free) = total.checked_sub(max_bytes).filter(|&x| x > 0) else {
            return;
        };
        files.sort_by(|a, b| a.0.cmp(&b.0)); // oldest first
        for (name, len) in files {
            if to_free == 0 {
                break;
            }
            if std::fs::remove_file(dir.join(&name)).is_ok() {
                to_free = to_free.saturating_sub(len);
            }
        }
    }

    /// Apply the turn-stability memo to a freshly compressed `result`, in place. Replays an
    /// already-seen conversation prefix's compressed `content` verbatim (keeping the provider
    /// prefix cache warm) and records this turn for the next one. On reuse it recomputes
    /// `input_tokens_after` over the rewritten body so the recorded savings reflect what is
    /// actually sent. No-op (full stateless behavior) when the flag is off, the n-gram carve-out
    /// applies, or the JSON can't be re-parsed.
    fn apply_turn_memo(
        config: &DenseConfig,
        memo: &Memo,
        kind: ProviderKind,
        original_body: &str,
        result: &mut llmtrim_core::CompressResult,
    ) {
        if !config.memo {
            return;
        }
        // Carve-out: the n-gram stage rewrites content with whole-conversation-dependent
        // placeholders (`§1`…) backed by an injected legend — splicing an earlier turn's encoding
        // into a differently-numbered legend would corrupt it. Detect it from the *effective*
        // run (catches `auto` routing), and skip the memo entirely for this request when present.
        let ngram_ran = result.stages.iter().any(|s| s.applied && s.name == "ngram");
        if ngram_ran {
            return;
        }
        let (Ok(original), Ok(mut compressed)) = (
            serde_json::from_str::<serde_json::Value>(original_body),
            serde_json::from_str::<serde_json::Value>(&result.request_json),
        ) else {
            return;
        };
        // Scope the memo to this request's compression context: the same conversation under a
        // different provider kind or effective config produces different compressed bytes —
        // replaying across contexts would splice one preset's compression into another's output.
        // The top-level config alone is NOT enough: under `auto` it is identical every turn even
        // when per-request routing flips presets (a turn that adds tools routes `rag` → `agent`),
        // so the salt also folds in the effective STAGE LINEUP this run executed (routing-
        // determined, content-independent). Any context flip = cold start, never a cross-splice.
        let lineup: String = result
            .stages
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(",");
        let salt = format!(
            "{kind:?}|{lineup}|{}",
            serde_json::to_string(config).unwrap_or_default()
        );
        let reused = llmtrim_core::memo::apply(memo, salt.as_bytes(), &original, &mut compressed);
        if reused == 0 {
            // Either nothing matched (cold prefix) or an unmodelled shape: `compressed` is
            // unchanged from `result.request_json`, so leave the result (and its counts) as-is.
            return;
        }
        // Re-serialize the rewritten body and recompute the content+tools token count over it,
        // so `input_after` describes the bytes we actually forward. The counter is the same one
        // `compress_with_config` used (provider + model), so the measure is consistent.
        let Ok(rewritten) = serde_json::to_string(&compressed) else {
            return;
        };
        if let Ok(counter) = llmtrim_core::tokenizer::counter_for(kind, result.model.as_deref()) {
            let adapter = llmtrim_core::provider::for_kind(kind);
            let req = llmtrim_core::ir::Request::from_value(kind, compressed);
            let after =
                llmtrim_core::pipeline::content_tokens(&req, adapter.as_ref(), counter.as_ref());
            result.input_tokens_after = llmtrim_core::tokenizer::Tokens(after);
        }
        result.request_json = rewritten;
    }

    /// Owns the accumulated response bytes; on drop (stream complete/aborted) it measures
    /// the output tokens and writes the completed ledger record.
    struct Finalize {
        acc: Arc<Mutex<Vec<u8>>>,
        pending: Option<Pending>,
        ledger: Sender<Record>,
        breakdown_ledger: Sender<BreakdownPayload>,
    }

    impl Drop for Finalize {
        fn drop(&mut self) {
            let Some(p) = self.pending.take() else {
                return;
            };
            let buf = self
                .acc
                .lock()
                .map(|mut b| std::mem::take(&mut *b))
                .unwrap_or_default();
            // Prefer the provider's own output-token count (exact; includes tool-use and
            // thinking output). Fall back to tokenizing the answer text only when no usage is
            // present in the response.
            let output_after = extract_output_usage(p.provider, &buf).or_else(|| {
                extract_output_text(p.provider, &buf).and_then(|text| {
                    llmtrim_core::tokenizer::counter_for(p.provider, p.model.as_deref())
                        .ok()
                        .map(|c| c.count(&text) as i64)
                })
            });
            // Cached-prefix tokens the provider served from its prompt cache (the discounted
            // resent context); `None` when the provider reports none.
            let cache_read = extract_cache_read(p.provider, &buf);
            // Full-rate + cache-write input tokens — with `cache_read` these reconstruct the
            // request's real input bill, which the dashboard's net-$ figures are priced from.
            let (fresh_input, cache_write) = extract_input_usage(p.provider, &buf);
            let usage = ResponseUsage {
                output_after,
                cache_read,
                fresh_input,
                cache_write,
            };
            if let Some(x) = build_breakdown(&p, &usage) {
                let _ = self.breakdown_ledger.send(x);
            }
            let _ = self.ledger.send(record_from(p, usage));
        }
    }

    /// The model's answer text from a captured response body — a non-streaming JSON answer,
    /// or the concatenated text deltas of an SSE stream.
    fn extract_output_text(provider: ProviderKind, body: &[u8]) -> Option<String> {
        let text = std::str::from_utf8(body).ok()?;
        if text.trim_start().starts_with('{')
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim())
        {
            return llmtrim_core::provider::for_kind(provider).answer_text(&value);
        }
        let mut out = String::new();
        for line in text.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(data) {
                out.push_str(&sse_delta(provider, &value));
            }
        }
        (!out.is_empty()).then_some(out)
    }

    /// The provider-reported output-token count from a captured response — exact, and includes
    /// the tool-use / thinking output that re-tokenizing the visible text would miss. Reads the
    /// `usage` of a non-streaming JSON body, or the streaming usage event (Anthropic
    /// `message_delta`, OpenAI final chunk with `usage`, Gemini `usageMetadata`). `None` when no
    /// usage is present, so the caller can fall back to tokenizing the answer text.
    fn extract_output_usage(provider: ProviderKind, body: &[u8]) -> Option<i64> {
        use serde_json::Value;
        let text = std::str::from_utf8(body).ok()?;
        // Pull the output-token field from one JSON event, wherever the provider puts it.
        let from_value = |v: &Value| -> Option<i64> {
            match provider {
                ProviderKind::Anthropic => v
                    .pointer("/usage/output_tokens")
                    .or_else(|| v.pointer("/message/usage/output_tokens"))
                    .and_then(Value::as_i64),
                ProviderKind::OpenAi => v
                    .pointer("/usage/completion_tokens") // Chat Completions
                    .or_else(|| v.pointer("/usage/output_tokens")) // Responses (non-stream)
                    .or_else(|| v.pointer("/response/usage/output_tokens")) // Responses SSE done
                    .and_then(Value::as_i64),
                ProviderKind::Google => v
                    .pointer("/usageMetadata/candidatesTokenCount")
                    .and_then(Value::as_i64),
            }
        };
        // Non-streaming JSON body.
        if text.trim_start().starts_with('{')
            && let Ok(v) = serde_json::from_str::<Value>(text.trim())
        {
            return from_value(&v);
        }
        // SSE: the final usage wins (Anthropic reports a partial count in `message_start` and
        // the true total in `message_delta`), so take the max seen across events.
        let mut best: Option<i64> = None;
        for line in text.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
                && let Some(n) = from_value(&v)
            {
                best = Some(best.map_or(n, |b| b.max(n)));
            }
        }
        best
    }

    /// Provider-reported cached input tokens reused on this request (prompt-cache hits) — the
    /// discounted resent prefix. Anthropic `cache_read_input_tokens` (in `message_start`),
    /// OpenAI `prompt_tokens_details.cached_tokens`, Gemini `cachedContentTokenCount`. Mirrors
    /// `extract_output_usage`'s JSON/SSE walk; `None` when the provider reports none.
    fn extract_cache_read(provider: ProviderKind, body: &[u8]) -> Option<i64> {
        use serde_json::Value;
        let text = std::str::from_utf8(body).ok()?;
        let from_value = |v: &Value| -> Option<i64> {
            match provider {
                ProviderKind::Anthropic => v
                    .pointer("/usage/cache_read_input_tokens")
                    .or_else(|| v.pointer("/message/usage/cache_read_input_tokens"))
                    .and_then(Value::as_i64),
                ProviderKind::OpenAi => v
                    .pointer("/usage/prompt_tokens_details/cached_tokens") // Chat Completions
                    .or_else(|| v.pointer("/usage/input_tokens_details/cached_tokens")) // Responses
                    .or_else(|| v.pointer("/response/usage/input_tokens_details/cached_tokens"))
                    .and_then(Value::as_i64),
                ProviderKind::Google => v
                    .pointer("/usageMetadata/cachedContentTokenCount")
                    .and_then(Value::as_i64),
            }
        };
        if text.trim_start().starts_with('{')
            && let Ok(v) = serde_json::from_str::<Value>(text.trim())
        {
            return from_value(&v);
        }
        let mut best: Option<i64> = None;
        for line in text.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data)
                && let Some(n) = from_value(&v)
            {
                best = Some(best.map_or(n, |b| b.max(n)));
            }
        }
        best
    }

    /// Provider-reported `(fresh_input, cache_write)` tokens — the rest of the input bill
    /// alongside `extract_cache_read`. Fresh = input billed at the full rate: Anthropic
    /// `input_tokens` (already excludes cache read/write); OpenAI `prompt_tokens −
    /// cached_tokens` and Google `promptTokenCount − cachedContentTokenCount` (theirs
    /// include the cached share). Cache writes (surcharged 1.25×) only exist on Anthropic
    /// (`cache_creation_input_tokens`). Mirrors the JSON/SSE walk of the other extractors.
    fn extract_input_usage(provider: ProviderKind, body: &[u8]) -> (Option<i64>, Option<i64>) {
        use serde_json::Value;
        let Some(text) = std::str::from_utf8(body).ok() else {
            return (None, None);
        };
        let from_value = |v: &Value| -> (Option<i64>, Option<i64>) {
            match provider {
                ProviderKind::Anthropic => {
                    let at = |p: &str, q: &str| {
                        v.pointer(p)
                            .or_else(|| v.pointer(q))
                            .and_then(Value::as_i64)
                    };
                    (
                        at("/usage/input_tokens", "/message/usage/input_tokens"),
                        at(
                            "/usage/cache_creation_input_tokens",
                            "/message/usage/cache_creation_input_tokens",
                        ),
                    )
                }
                ProviderKind::OpenAi => {
                    let prompt = v
                        .pointer("/usage/prompt_tokens") // Chat Completions
                        .or_else(|| v.pointer("/usage/input_tokens")) // Responses
                        .or_else(|| v.pointer("/response/usage/input_tokens"))
                        .and_then(Value::as_i64);
                    let cached = v
                        .pointer("/usage/prompt_tokens_details/cached_tokens")
                        .or_else(|| v.pointer("/usage/input_tokens_details/cached_tokens"))
                        .or_else(|| v.pointer("/response/usage/input_tokens_details/cached_tokens"))
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                    (prompt.map(|p| (p - cached).max(0)), None)
                }
                ProviderKind::Google => {
                    let prompt = v
                        .pointer("/usageMetadata/promptTokenCount")
                        .and_then(Value::as_i64);
                    let cached = v
                        .pointer("/usageMetadata/cachedContentTokenCount")
                        .and_then(Value::as_i64)
                        .unwrap_or(0);
                    (prompt.map(|p| (p - cached).max(0)), None)
                }
            }
        };
        if text.trim_start().starts_with('{') {
            if let Ok(v) = serde_json::from_str::<Value>(text.trim()) {
                return from_value(&v);
            }
            return (None, None);
        }
        // SSE: the final usage wins (Anthropic's `message_start` carries the input usage;
        // later events may repeat it) — take the max seen, like the other extractors.
        let (mut fresh, mut write): (Option<i64>, Option<i64>) = (None, None);
        for line in text.lines() {
            let Some(data) = line.trim_start().strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                let (f, w) = from_value(&v);
                if let Some(n) = f {
                    fresh = Some(fresh.map_or(n, |b| b.max(n)));
                }
                if let Some(n) = w {
                    write = Some(write.map_or(n, |b| b.max(n)));
                }
            }
        }
        (fresh, write)
    }

    /// The text delta in one SSE event, per provider's streaming shape.
    fn sse_delta(provider: ProviderKind, v: &serde_json::Value) -> String {
        use serde_json::Value;
        match provider {
            ProviderKind::OpenAi => {
                // Chat Completions streams text under `/choices/0/delta/content`; the Responses
                // API streams it as the `delta` string of a `response.output_text.delta` event.
                if let Some(c) = v
                    .pointer("/choices/0/delta/content")
                    .and_then(Value::as_str)
                {
                    c.to_string()
                } else if v.get("type").and_then(Value::as_str)
                    == Some("response.output_text.delta")
                {
                    v.get("delta")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string()
                } else {
                    String::new()
                }
            }
            ProviderKind::Anthropic => v
                .pointer("/delta/text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            ProviderKind::Google => v
                .pointer("/candidates/0/content/parts")
                .and_then(Value::as_array)
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                        .collect::<String>()
                })
                .unwrap_or_default(),
        }
    }

    /// A port already in use. Carried as a typed error so the supervisor can recognize a
    /// permanent bind failure (don't retry) and the dispatcher can word the hint.
    #[derive(Debug)]
    pub struct PortInUse {
        port: u16,
        /// The llmtrim daemon already holding it, if the pidfile names a live one.
        by_llmtrim_pid: Option<u32>,
    }

    impl std::fmt::Display for PortInUse {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self.by_llmtrim_pid {
                Some(pid) => write!(
                    f,
                    "port {} is already served by llmtrim (pid {pid}) — `llmtrim stop` first, \
                     or rerun with --force to replace it",
                    self.port
                ),
                None => write!(
                    f,
                    "cannot bind port {} — already in use by another process; free it or pass \
                     a different --port",
                    self.port
                ),
            }
        }
    }

    impl std::error::Error for PortInUse {}

    /// Refuse (or, with `force`, clear) a port already held before we try to bind it. Turns the
    /// retry-looping `EADDRINUSE` crash into one actionable error, and lets `--force` take over a
    /// stale daemon the way `start`/`stop` already manage one. A daemon whose pidfile names *our
    /// own* pid (the supervised loop adopted it) is not a conflict — skip it.
    fn preflight_port(port: u16, force: bool) -> Result<()> {
        if let Some(state) = crate::daemon::running()
            && state.port == port
            && state.pid != std::process::id()
        {
            if !force {
                return Err(PortInUse {
                    port,
                    by_llmtrim_pid: Some(state.pid),
                }
                .into());
            }
            eprintln!(
                "llmtrim: --force: stopping existing daemon (pid {})",
                state.pid
            );
            // `stop` waits for the process to exit; wait_free closes the lingering-socket gap so
            // the bind below doesn't lose the race. A timeout here is its own error (the daemon
            // didn't release the port), not the foreign-process case below, so the message stays
            // accurate.
            if !crate::daemon::stop_and_wait_free(port)? {
                anyhow::bail!(
                    "--force stopped the daemon (pid {}) but port {port} was still held after 5s",
                    state.pid
                );
            }
            return Ok(());
        }
        // An untracked process on the port (no llmtrim pidfile, or one naming a different port):
        // `--force` can't safely kill a foreign process, so refuse immediately either way.
        if crate::daemon::probe_port(port) {
            return Err(PortInUse {
                port,
                by_llmtrim_pid: None,
            }
            .into());
        }
        Ok(())
    }

    /// Run the interceptor on `127.0.0.1:port`, blocking until Ctrl-C. Sets up its own
    /// Tokio runtime so the rest of the CLI stays synchronous.
    pub fn run(port: u16, force: bool) -> Result<()> {
        preflight_port(port, force)?;
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .context("failed to start async runtime")?;
        rt.block_on(run_async(port))
    }

    /// Supervised run: keep the interceptor alive across crashes (the daemon mode). Because
    /// `setup` points `HTTPS_PROXY` at us, a dead proxy breaks the client's HTTPS entirely —
    /// so on an unexpected exit/panic we restart. Gives up only if it fails fast 5× in a row
    /// (a real misconfig, not a transient), so it never spins forever.
    pub fn run_supervised(port: u16, force: bool) -> Result<()> {
        use std::time::{Duration, Instant};
        let mut fast_fails = 0u32;
        // `--force` is a one-time take-over: apply it only on the first start. On a crash-restart
        // the daemon we'd be replacing is ourselves (the pidfile now holds our own pid), so there
        // is nothing to force, and re-forcing could stop a freshly adopted sibling.
        let mut force = force;
        loop {
            // Adopt the pidfile if it's missing (a manually-launched supervised daemon, or
            // one whose pidfile was lost to a transient full disk). Best-effort: never let a
            // bookkeeping write stop the proxy from serving.
            let _ = crate::daemon::write_state_if_absent(std::process::id(), port);
            let started = Instant::now();
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(port, force)));
            force = false;
            match outcome {
                Ok(Ok(())) => return Ok(()), // graceful shutdown (Ctrl-C)
                // A bind failure is permanent (the port is taken or OS-reserved), not the
                // transient crash the restart loop exists for — retrying just spins. Surface
                // it once and give up so the operator sees the actionable message immediately.
                Ok(Err(e)) if e.downcast_ref::<PortInUse>().is_some() => return Err(e),
                Ok(Err(e)) => eprintln!("llmtrim: interceptor exited: {e}"),
                Err(_) => eprintln!("llmtrim: interceptor panicked"),
            }
            if started.elapsed() < Duration::from_secs(5) {
                fast_fails += 1;
                if fast_fails >= 5 {
                    anyhow::bail!(
                        "interceptor crashed 5× in a row — giving up (run `llmtrim serve` to see the error)"
                    );
                }
            } else {
                fast_fails = 0;
            }
            eprintln!("llmtrim: restarting interceptor in 2s…");
            crate::daemon::bump_restarts(); // surfaces as a `status` warning
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Build the lazy tokenizer BPE tables (the dominant one-time cost) + prime the stage code
    /// paths and the first tree-sitter grammar, before the proxy starts serving.
    fn warm_up(config: &DenseConfig) {
        use llmtrim_core::ir::ProviderKind;
        for (kind, model) in [
            (ProviderKind::OpenAi, Some("gpt-4o")), // o200k_base
            (ProviderKind::OpenAi, Some("gpt-4")),  // cl100k_base
            (ProviderKind::Anthropic, None),        // approximate counter
        ] {
            if let Ok(c) = llmtrim_core::tokenizer::counter_for(kind, model) {
                let _ = c.count("warm up the tokenizer table");
            }
        }
        // One real compression primes the stage code paths + the first grammar load.
        let sample = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"function f(){ return 1; }"}]}"#;
        let _ = llmtrim_core::compress_with_config(sample, Some(ProviderKind::OpenAi), config);
    }

    async fn run_async(port: u16) -> Result<()> {
        let _ = aws_lc_rs::default_provider().install_default();

        let (cert_pem, key_pem) = ensure_ca()?;
        let key = hudsucker::rcgen::KeyPair::from_pem(&key_pem)
            .map_err(|e| anyhow::anyhow!("failed to parse CA key: {e}"))?;
        let issuer = hudsucker::rcgen::Issuer::from_ca_cert_pem(&cert_pem, key)
            .map_err(|e| anyhow::anyhow!("failed to parse CA cert: {e}"))?;
        let ca = LeafCertAuthority::new(issuer, aws_lc_rs::default_provider());

        // Ledger writes go to a dedicated thread (rusqlite isn't async); the handler just
        // sends Records over the channel.
        let (tx, rx) = std::sync::mpsc::channel::<Record>();
        let (breakdown_tx, breakdown_rx) = std::sync::mpsc::channel::<BreakdownPayload>();
        std::thread::spawn(move || {
            if let Ok(tracker) = Tracker::open() {
                // The daemon opens the ledger once, so the open-time prune never re-runs on
                // its own. Re-prune every N writes to keep it bounded (row cap + any
                // configured age retention) across a long-running daemon's uptime.
                const PRUNE_EVERY: u64 = 1_000;
                let mut since_prune: u64 = 0;
                for rec in rx {
                    let _ = tracker.record(&rec);
                    since_prune += 1;
                    if since_prune >= PRUNE_EVERY {
                        since_prune = 0;
                        let _ = tracker.prune_default();
                    }
                }
            }
        });
        // breakdown attribution writes go to their own thread + ledger handle, so a burst of
        // per-source rows can't stall the primary `compressions` writer.
        std::thread::spawn(move || {
            if let Ok(tracker) = Tracker::open() {
                const PRUNE_EVERY: u64 = 1_000;
                let mut since_prune: u64 = 0;
                for payload in breakdown_rx {
                    let _ = tracker.record_breakdown(&payload.turn, &payload.blocks);
                    since_prune += 1;
                    if since_prune >= PRUNE_EVERY {
                        since_prune = 0;
                        let _ = tracker
                            .prune_breakdown(crate::tracking::Tracker::breakdown_turns_cap());
                    }
                }
            }
        });

        // Loopback by default — a MITM proxy must not be reachable off-host unless asked.
        // LLMTRIM_BIND=0.0.0.0 opts in (containers: port mapping can't reach loopback).
        let bind_ip: std::net::IpAddr = match RuntimeConfig::get().bind.clone() {
            Some(s) => s
                .parse()
                .with_context(|| format!("bind address is not a valid IP: {s}"))?,
            None => std::net::IpAddr::from([127, 0, 0, 1]),
        };
        let addr = SocketAddr::from((bind_ip, port));

        // Validate and read the upstream proxy setting once at startup so a bad URL is a hard
        // error before any traffic flows, not a surprise on the first replay. Pass the bind
        // address so the guard can distinguish "same port = recursion" from "different port =
        // legitimate companion proxy" (e.g. headroom on 127.0.0.1:9999).
        let upstream_proxy = crate::transport::upstream_proxy_url(Some(addr))?;
        if let Some(ref u) = upstream_proxy {
            eprintln!(
                "llmtrim: upstream proxy: {}",
                crate::transport::redact_proxy_url(u)
            );
        }

        // User opt-out lists, resolved once at startup (env + config file) like the other runtime
        // settings, then snapshotted onto the handler.
        let exclusions = llmtrim_core::config::exclusions();

        let handler = Interceptor {
            config: Arc::new(DenseConfig::load_for_interceptor()),
            ledger: tx,
            breakdown_ledger: breakdown_tx,
            // `ensure_ca` above just reconciled the CA + sidecar, so this is exactly what the
            // live CA can sign for.
            domains: Arc::new(covered_domains()),
            // One process-wide turn-stability memo, shared across the per-request handler clones.
            memo: Arc::new(Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY)),
            pending: None,
            streaming: false,
            upstream_proxy: upstream_proxy.clone(),
            exclude_providers: Arc::new(exclusions.providers.clone()),
            exclude_hosts: Arc::new(exclusions.hosts.clone()),
            sub: {
                // `sub` here is already non-`off`/non-empty (RuntimeConfig filters those). A value
                // that still doesn't parse is a typo (`sub = "codexx"`): warn once at startup so a
                // misconfigured opt-in doesn't silently run with reroute disabled.
                let raw = RuntimeConfig::get().sub.as_deref();
                let parsed = raw.and_then(crate::reroute::SubProvider::parse);
                if parsed.is_none()
                    && let Some(r) = raw
                {
                    eprintln!(
                        "llmtrim: unknown sub provider '{r}' — reroute disabled (expected codex|kimi)"
                    );
                }
                parsed
            },
            sub_chain: Arc::new(
                RuntimeConfig::get()
                    .sub_chain
                    .iter()
                    .filter_map(|p| {
                        let parsed = crate::reroute::SubProvider::parse(p);
                        if parsed.is_none() {
                            // Silently dropping it would leave a typo'd chain quietly shorter than
                            // the user configured — and the fallback they think they have missing.
                            eprintln!(
                                "llmtrim: unknown provider '{p}' in the sub fallback chain — ignored (expected codex|kimi)"
                            );
                        }
                        parsed
                    })
                    .collect(),
            ),
            sub_tiers: Arc::new(RuntimeConfig::get().sub_tiers.clone()),
            sub_fallback: RuntimeConfig::get().sub_fallback,
            sub_effort: RuntimeConfig::get().sub_effort.clone(),
        };
        // Pre-flight bind: hudsucker's `Proxy::start` collapses a bind failure to a bare
        // "io error", hiding whether the port is in use or OS-reserved. Bind once ourselves
        // first so the real cause reaches the log, then drop it microseconds before hudsucker
        // rebinds the same addr. Only an in-use port maps to PortInUse (so the supervisor fails
        // fast on it); a permission/transient error keeps its OS cause and stays retryable.
        drop(
            std::net::TcpListener::bind(addr).map_err(|e| -> anyhow::Error {
                if e.kind() == std::io::ErrorKind::AddrInUse {
                    PortInUse {
                        port,
                        by_llmtrim_pid: None,
                    }
                    .into()
                } else {
                    anyhow::Error::new(e).context(format!("cannot bind {addr}"))
                }
            })?,
        );
        eprintln!("llmtrim: MITM interceptor on http://{addr}");
        eprintln!("  export HTTPS_PROXY=http://{addr}");
        eprintln!(
            "  trust the CA: NODE_EXTRA_CA_CERTS={}",
            ca_cert_path()?.display()
        );

        // Prime the lazy tokenizer tables + stage machinery before serving, so the first real
        // request runs at steady state instead of paying ~150 ms of one-time init (which slowed
        // the first call and skewed the latency metric). One-time, at boot, off the request path.
        warm_up(&handler.config);

        // When LLMTRIM_UPSTREAM_PROXY is set, wrap hudsucker's outbound connector in a
        // hyper-http-proxy ProxyConnector that tunnels origin TLS connections via CONNECT
        // through the upstream proxy. When the var is unset, fall back to the standard
        // `.with_rustls_connector(...)` path — byte-identical behaviour to before this change.
        //
        // The two branches produce different generic Proxy<C, ...> types (rustls connector vs
        // proxy connector), so each branch calls .start().await directly rather than binding
        // a common variable.
        if let Some(ref upstream_url) = upstream_proxy {
            let upstream_uri =
                upstream_url
                    .parse::<hudsucker::hyper::Uri>()
                    .with_context(|| {
                        format!(
                            "failed to parse upstream proxy URI `{}`",
                            crate::transport::redact_proxy_url(upstream_url)
                        )
                    })?;
            let upstream_proxy_spec = UpstreamProxy::new(Intercept::All, upstream_uri);
            // ProxyConnector has two distinct connection roles:
            //  - Its INNER connector dials the PROXY itself. The upstream proxy is http://,
            //    so a plain HttpConnector is correct — no TLS to the proxy.
            //  - Its `tls` field wraps the ORIGIN connection that is tunnelled THROUGH the
            //    CONNECT. This must perform full verifying TLS against the real origin
            //    (openrouter.ai etc.) so the API key is never sent over a cleartext or
            //    unverified channel.
            //
            // `from_proxy` (with the `rustls-tls-native-roots` feature active) builds a
            // tokio-rustls TlsConnector for the origin leg using native roots and full cert
            // verification. The tokio-rustls ClientConfig uses whatever CryptoProvider is
            // installed at process start — we call aws_lc_rs::default_provider() at startup,
            // so origin TLS automatically uses aws-lc-rs throughout.
            let proxy_connector =
                ProxyConnector::from_proxy(HttpConnector::new(), upstream_proxy_spec)
                    .map_err(|e| anyhow::anyhow!("failed to build upstream ProxyConnector: {e}"))?;
            Proxy::builder()
                .with_addr(addr)
                .with_ca(ca)
                .with_http_connector(proxy_connector)
                .with_http_handler(handler)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build proxy (upstream-proxy path): {e}"))?
                .start()
                .await
                .map_err(|e| anyhow::anyhow!("proxy error: {e}"))?;
        } else {
            Proxy::builder()
                .with_addr(addr)
                .with_ca(ca)
                .with_rustls_connector(aws_lc_rs::default_provider())
                .with_http_handler(handler)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .build()
                .map_err(|e| anyhow::anyhow!("failed to build proxy: {e}"))?
                .start()
                .await
                .map_err(|e| anyhow::anyhow!("proxy error: {e}"))?;
        }
        Ok(())
    }

    pub fn ca_cert_path() -> Result<PathBuf> {
        Ok(crate::daemon::home_dir()?.join("ca.pem"))
    }

    fn ca_key_path() -> Result<PathBuf> {
        Ok(crate::daemon::home_dir()?.join("ca.key"))
    }

    /// Sidecar recording the domain set the persisted CA was built for — so we can tell, without
    /// parsing the certificate, when the intended host set changed and the CA must be regenerated.
    fn ca_hosts_path() -> Result<PathBuf> {
        Ok(crate::daemon::home_dir()?.join("ca.hosts"))
    }

    /// The domains the persisted CA was built for, from its sidecar (one per line). `None` when
    /// there is no sidecar (a CA from before sidecars existed, or no CA at all).
    fn read_ca_hosts() -> Option<Vec<String>> {
        let text = std::fs::read_to_string(ca_hosts_path().ok()?).ok()?;
        let v: Vec<String> = text
            .lines()
            .map(str::to_string)
            .filter(|l| !l.is_empty())
            .collect();
        (!v.is_empty()).then_some(v)
    }

    /// The domains the live CA actually covers (its sidecar set) — the interceptor's gate. Falls
    /// back to the static set when no sidecar is present, which is safe because `ensure_ca` always
    /// reconciles (writing the sidecar) before the daemon reads this.
    fn covered_domains() -> std::collections::HashSet<String> {
        read_ca_hosts()
            .unwrap_or_else(intercept_domains)
            .into_iter()
            .collect()
    }

    /// Whether the persisted CA already matches the intended host set (no regeneration needed):
    /// the cert+key exist and the sidecar equals `expected`. Pure, for testing.
    fn ca_is_current(have_ca: bool, sidecar: Option<&[String]>, expected: &[String]) -> bool {
        have_ca && sidecar == Some(expected)
    }

    /// Load the local CA, (re)generating a name-constrained root CA whenever it is missing or its
    /// recorded host set differs from the current one (a host added/removed across an update).
    /// Regeneration is in place: tools trusting it via `NODE_EXTRA_CA_CERTS` (the file) pick it up
    /// on relaunch — only OS trust-store *copies* go stale, so we say so. Returns `(cert, key)`.
    pub fn ensure_ca() -> Result<(String, String)> {
        let (cert_path, key_path) = (ca_cert_path()?, ca_key_path()?);
        let have_ca = cert_path.exists() && key_path.exists();
        let expected = intercept_domains();
        if ca_is_current(have_ca, read_ca_hosts().as_deref(), &expected) {
            return Ok((
                std::fs::read_to_string(&cert_path)?,
                std::fs::read_to_string(&key_path)?,
            ));
        }
        let (cert_pem, key_pem) = generate_ca(&expected)?;
        let dir = crate::daemon::home_dir()?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        std::fs::write(&cert_path, &cert_pem)?;
        #[cfg(unix)]
        {
            use std::io::Write as _;
            use std::os::unix::fs::OpenOptionsExt as _;
            // `mode` applies only when the file is *created* — remove any pre-existing key
            // (e.g. one written 0644 by an older build) so the 0600 always takes effect.
            let _ = std::fs::remove_file(&key_path);
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&key_path)
                .and_then(|mut f| f.write_all(key_pem.as_bytes()))
                .with_context(|| format!("failed to write CA key {}", key_path.display()))?;
        }
        #[cfg(not(unix))]
        std::fs::write(&key_path, &key_pem)?;
        std::fs::write(ca_hosts_path()?, expected.join("\n"))
            .with_context(|| "failed to write CA host sidecar")?;
        if have_ca {
            // Regenerated over an existing CA because the host set changed. Env-trusting tools
            // follow the file automatically; any OS trust-store copy is now stale.
            eprintln!(
                "llmtrim: CA updated for a changed provider-host set. Tools trusting it via \
                 NODE_EXTRA_CA_CERTS pick it up on relaunch; if you trusted it system-wide \
                 (GUI apps), re-trust it — see `llmtrim ca`."
            );
        }
        Ok((cert_pem, key_pem))
    }

    /// Generate a root CA whose signing power is name-constrained to `domains`.
    fn generate_ca(domains: &[String]) -> Result<(String, String)> {
        use hudsucker::rcgen::{
            BasicConstraints, CertificateParams, DistinguishedName, DnType, GeneralSubtree, IsCa,
            KeyPair, KeyUsagePurpose, NameConstraints,
        };
        let key = KeyPair::generate().map_err(|e| anyhow::anyhow!("CA keygen failed: {e}"))?;
        let mut params = CertificateParams::default();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::CrlSign,
            KeyUsagePurpose::DigitalSignature,
        ];
        let mut dn = DistinguishedName::new();
        dn.push(DnType::CommonName, "llmtrim local CA");
        dn.push(DnType::OrganizationName, "llmtrim");
        params.distinguished_name = dn;
        params.name_constraints = Some(NameConstraints {
            permitted_subtrees: domains
                .iter()
                .map(|d| GeneralSubtree::DnsName(d.clone()))
                .collect(),
            excluded_subtrees: vec![],
        });
        let cert = params
            .self_signed(&key)
            .map_err(|e| anyhow::anyhow!("CA self-sign failed: {e}"))?;
        Ok((cert.pem(), key.serialize_pem()))
    }

    /// MITM leaf-certificate authority: a drop-in for hudsucker's `RcgenAuthority` that adds the
    /// X.509 extensions strict TLS stacks require on a leaf but hudsucker 0.24 omits. Without the
    /// Authority Key Identifier, OpenSSL's strict verification (`VERIFY_X509_STRICT`, which Python
    /// 3.13 enables by default in `ssl.create_default_context()`) rejects the cert with "Missing
    /// Authority Key Identifier", which breaks every httpx / OpenAI-SDK Python client behind the
    /// proxy (curl and Node TLS skip the strict checks, so this went unnoticed). We also set Key
    /// Usage + Extended Key Usage (serverAuth). Otherwise identical to `RcgenAuthority`: a
    /// per-host leaf signed by our
    /// CA, cached in memory. `RcgenAuthority::new` exposes no hook for these extensions and 0.24
    /// is the latest published version, and llmtrim publishes to crates.io (so a git-patched
    /// hudsucker is not an option), hence the small in-tree copy.
    ///
    /// TODO: drop this and go back to `RcgenAuthority` once hudsucker mints leaves with an
    /// Authority Key Identifier (or exposes a hook to set leaf extensions). Tracking upstream at
    /// <https://github.com/omjadas/hudsucker/issues>.
    struct LeafCertAuthority {
        issuer: hudsucker::rcgen::Issuer<'static, hudsucker::rcgen::KeyPair>,
        private_key: PrivateKeyDer<'static>,
        provider: Arc<CryptoProvider>,
        /// Per-host leaf `ServerConfig`s. Intentionally unbounded and non-evicting: it is reached
        /// only after interception is decided, so it is keyed by the fixed `intercept_domains()`
        /// allowlist (a handful of LLM hosts), not arbitrary SNI, and the leaves are valid 365
        /// days. (hudsucker's `RcgenAuthority` used a moka TTL cache; that eviction is moot here.)
        cache: Mutex<std::collections::HashMap<String, Arc<ServerConfig>>>,
    }

    impl LeafCertAuthority {
        fn new(
            issuer: hudsucker::rcgen::Issuer<'static, hudsucker::rcgen::KeyPair>,
            provider: CryptoProvider,
        ) -> Self {
            let private_key =
                PrivateKeyDer::from(PrivatePkcs8KeyDer::from(issuer.key().serialize_der()));
            Self {
                issuer,
                private_key,
                provider: Arc::new(provider),
                cache: Mutex::new(std::collections::HashMap::new()),
            }
        }

        /// Mint a leaf cert for `host`, signed by our CA, carrying the extensions strict OpenSSL
        /// requires (the `RcgenAuthority::gen_cert` shape plus AKI / Key Usage / EKU).
        fn gen_cert(&self, host: &str) -> CertificateDer<'static> {
            use hudsucker::rcgen::string::Ia5String;
            use hudsucker::rcgen::{
                CertificateParams, DistinguishedName, DnType, ExtendedKeyUsagePurpose, IsCa,
                KeyUsagePurpose, SanType,
            };
            use time::{Duration, OffsetDateTime};

            let mut params = CertificateParams::default();
            // Unique serial without a `rand` dependency: a process counter mixed with the wall
            // clock. Uniqueness (not unpredictability) is all a leaf serial needs here.
            static SERIAL: AtomicU64 = AtomicU64::new(0);
            let serial = SERIAL.fetch_add(1, Ordering::Relaxed)
                ^ (OffsetDateTime::now_utc().unix_timestamp_nanos() as u64);
            params.serial_number = Some(serial.into());

            let not_before = OffsetDateTime::now_utc() - Duration::seconds(60);
            params.not_before = not_before;
            params.not_after = not_before + Duration::days(365);

            let mut dn = DistinguishedName::new();
            dn.push(DnType::CommonName, host);
            params.distinguished_name = dn;
            params.subject_alt_names.push(SanType::DnsName(
                Ia5String::try_from(host.to_string()).expect("host is a valid DNS name"),
            ));

            // The fix: extensions hudsucker 0.24 omits but strict OpenSSL 3.x requires on a leaf.
            params.use_authority_key_identifier_extension = true;
            params.key_usages = vec![
                KeyUsagePurpose::DigitalSignature,
                KeyUsagePurpose::KeyEncipherment,
            ];
            params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
            // Emit `BasicConstraints: CA:FALSE` explicitly (rcgen's default omits the extension
            // entirely). RFC 5280 §4.2.1.9 says it SHOULD appear on an end-entity cert, and some
            // stricter stacks (LibreSSL, corporate TLS inspectors) want it.
            params.is_ca = IsCa::ExplicitNoCa;

            params
                .signed_by(self.issuer.key(), &self.issuer)
                .expect("failed to sign leaf certificate")
                .into()
        }
    }

    impl CertificateAuthority for LeafCertAuthority {
        async fn gen_server_config(&self, authority: &HttpAuthority) -> Arc<ServerConfig> {
            let host = authority.host().to_string();
            // Recover from a poisoned lock rather than panicking the proxy hot path: the only
            // code run under the lock is a HashMap get/insert (can't panic), so the guarded data
            // is always consistent even if some other thread panicked elsewhere.
            if let Some(cfg) = self
                .cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get(&host)
                .cloned()
            {
                return cfg;
            }
            let certs = vec![self.gen_cert(&host)];
            let mut cfg = ServerConfig::builder_with_provider(Arc::clone(&self.provider))
                .with_safe_default_protocol_versions()
                .expect("rustls protocol versions")
                .with_no_client_auth()
                .with_single_cert(certs, self.private_key.clone_key())
                .expect("rustls ServerConfig from leaf cert");
            cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
            let cfg = Arc::new(cfg);
            self.cache
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(host, Arc::clone(&cfg));
            cfg
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn reroute_error_message_surfaces_rate_limit_and_reset() {
            // The ChatGPT/Codex backend returns this exact shape when the plan is exhausted.
            let raw = br#"{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"plus","resets_in_seconds":6922}}"#;
            let msg = reroute_upstream_error_message(crate::reroute::SubProvider::Codex, 429, raw);
            assert!(msg.contains("HTTP 429"), "names the status: {msg}");
            assert!(
                msg.contains("The usage limit has been reached"),
                "carries the upstream detail: {msg}"
            );
            assert!(msg.contains("resets in ~1h55m"), "shows reset time: {msg}");
        }

        #[test]
        fn window_for_uses_curated_registry_beta_and_default() {
            // Real per-model windows from the models.dev snapshot, not a per-family guess.
            assert_eq!(window_for(Some("gpt-5")), 400_000);
            assert_eq!(window_for(Some("gpt-5.6-terra")), 1_050_000);
            assert_eq!(window_for(Some("claude-opus-4-5")), 200_000);
            // `[1m]` is Anthropic's opt-in 1M beta, not a registry property.
            assert_eq!(window_for(Some("claude-opus-4-8[1m]")), 1_000_000);
            // Unknown model / no model -> generic fallback.
            assert_eq!(window_for(Some("model-shipped-tomorrow")), 128_000);
            assert_eq!(window_for(None), 128_000);
        }

        #[test]
        fn reroute_stream_error_status_infers_from_message() {
            assert_eq!(
                reroute_stream_error_status("Your input exceeds the context window of this model."),
                400
            );
            assert_eq!(reroute_stream_error_status("context_length_exceeded"), 400);
            assert_eq!(reroute_stream_error_status("rate limit reached"), 429);
            assert_eq!(
                reroute_stream_error_status("The usage limit has been reached"),
                429
            );
            assert_eq!(reroute_stream_error_status("something else broke"), 502);
        }

        /// `Accept: text/event-stream` is what the Anthropic SDK (so Claude Code) sends on a
        /// `stream: true` call, and it is on the headers of *every* request — including the
        /// passthrough paths that never buffer a body to look for `"stream": true`.
        #[test]
        fn wants_event_stream_reads_the_accept_header() {
            let mut h = header::HeaderMap::new();
            assert!(!wants_event_stream(&h), "no Accept header: not streaming");
            h.insert(header::ACCEPT, "application/json".parse().unwrap());
            assert!(!wants_event_stream(&h));
            h.insert(header::ACCEPT, "text/event-stream".parse().unwrap());
            assert!(wants_event_stream(&h));
        }

        /// A transport failure on a *streaming* call must arrive as an SSE `error` frame: the
        /// client is parsing an event stream, and hudsucker's default (a bodiless 502) is not
        /// something it can read — it just drops the connection (issue #157).
        #[tokio::test]
        async fn transport_error_on_a_stream_is_an_sse_error_frame() {
            let res = upstream_transport_error(true, "connection closed before message completed");
            assert_eq!(
                res.status(),
                200,
                "the stream itself is committed; the error rides in it"
            );
            assert_eq!(
                res.headers().get(header::CONTENT_TYPE).unwrap(),
                "text/event-stream"
            );
            let body =
                String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec())
                    .unwrap();
            assert!(
                body.starts_with("event: error\n"),
                "SSE error frame: {body}"
            );
            assert!(body.ends_with("\n\n"), "frame is terminated: {body:?}");
            let data = body
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .expect("data line");
            let v: serde_json::Value = serde_json::from_str(data).expect("parseable JSON");
            assert_eq!(v["type"], "error");
            assert_eq!(v["error"]["type"], "overloaded_error");
            assert!(
                v["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("connection closed"),
                "names the real cause so the user can report it: {v}"
            );
        }

        /// The non-streaming shape: a retryable Anthropic error the SDK backs off on, never the
        /// bodiless 502 that fails the turn outright.
        #[tokio::test]
        async fn transport_error_without_a_stream_is_a_retryable_json_error() {
            let res = upstream_transport_error(false, "tls handshake eof");
            assert_eq!(
                res.status(),
                529,
                "overloaded: transient, and Claude Code retries it"
            );
            assert_eq!(res.headers().get(header::RETRY_AFTER).unwrap(), "1");
            let body =
                String::from_utf8(res.into_body().collect().await.unwrap().to_bytes().to_vec())
                    .unwrap();
            let v: serde_json::Value = serde_json::from_str(&body).expect("parseable JSON");
            assert_eq!(v["error"]["type"], "overloaded_error");
            assert!(
                v["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("tls handshake eof")
            );
        }

        #[test]
        fn reroute_error_kind_maps_status_to_anthropic_type() {
            assert_eq!(reroute_error_kind(400), "invalid_request_error");
            assert_eq!(reroute_error_kind(422), "invalid_request_error");
            assert_eq!(reroute_error_kind(429), "rate_limit_error");
            assert_eq!(reroute_error_kind(401), "authentication_error");
            assert_eq!(reroute_error_kind(403), "authentication_error");
            assert_eq!(reroute_error_kind(529), "overloaded_error");
            assert_eq!(reroute_error_kind(500), "api_error");
        }

        #[test]
        fn reroute_should_retry_covers_transient_and_rate_limit() {
            for s in [429, 500, 502, 503, 504, 529] {
                assert!(reroute_should_retry(s), "{s} should retry");
            }
            for s in [200, 400, 401, 403, 404, 422] {
                assert!(!reroute_should_retry(s), "{s} should not retry");
            }
        }

        #[test]
        fn reroute_backoff_grows_and_respects_budget() {
            // No hint: exponential from the base, capped.
            assert_eq!(reroute_backoff_ms(0, None), Some(REROUTE_RETRY_BASE_MS));
            assert_eq!(reroute_backoff_ms(1, None), Some(REROUTE_RETRY_BASE_MS * 2));
            assert_eq!(reroute_backoff_ms(9, None), Some(REROUTE_RETRY_CAP_MS));
            // A short reset hint is honored; a multi-hour one exceeds the budget -> stop retrying.
            assert_eq!(reroute_backoff_ms(0, Some(3)), Some(3_000));
            assert_eq!(reroute_backoff_ms(0, Some(6_922)), None);
        }

        #[test]
        fn reroute_retry_after_prefers_header_then_body() {
            // Retry-After header wins.
            assert_eq!(
                reroute_retry_after_secs(Some("12"), Some("999"), b"{}"),
                Some(12)
            );
            // Codex reset header when no Retry-After.
            assert_eq!(
                reroute_retry_after_secs(None, Some("300"), b"{}"),
                Some(300)
            );
            // Body fallback.
            let body = br#"{"error":{"resets_in_seconds":6922}}"#;
            assert_eq!(reroute_retry_after_secs(None, None, body), Some(6922));
            // Nothing usable.
            assert_eq!(reroute_retry_after_secs(None, None, b"nope"), None);
        }

        #[test]
        fn reroute_error_message_formats_sub_hour_reset() {
            let raw = br#"{"error":{"message":"slow down","resets_in_seconds":300}}"#;
            let msg = reroute_upstream_error_message(crate::reroute::SubProvider::Kimi, 429, raw);
            assert!(msg.contains("resets in ~5m"), "shows minute reset: {msg}");
            assert!(msg.contains("kimi"), "names the provider: {msg}");
        }

        #[test]
        fn reroute_error_message_falls_back_to_body_snippet() {
            // Non-JSON / unexpected body: still surface something, never an empty message.
            let msg = reroute_upstream_error_message(
                crate::reroute::SubProvider::Codex,
                502,
                b"Bad Gateway",
            );
            assert!(msg.contains("HTTP 502"), "names the status: {msg}");
            assert!(
                msg.contains("Bad Gateway"),
                "carries the raw snippet: {msg}"
            );
        }

        #[test]
        fn port_in_use_message_points_to_force_when_llmtrim_owns_it() {
            let msg = PortInUse {
                port: 43117,
                by_llmtrim_pid: Some(616982),
            }
            .to_string();
            assert!(msg.contains("43117"), "names the port: {msg}");
            assert!(msg.contains("616982"), "names the pid: {msg}");
            assert!(msg.contains("--force"), "points to --force: {msg}");
        }

        #[test]
        fn port_in_use_message_for_foreign_process_does_not_promise_force() {
            // --force can't kill a foreign process, so the message must not suggest it.
            let msg = PortInUse {
                port: 9999,
                by_llmtrim_pid: None,
            }
            .to_string();
            assert!(msg.contains("9999"), "names the port: {msg}");
            assert!(
                msg.contains("another process"),
                "blames a foreign owner: {msg}"
            );
            assert!(!msg.contains("--force"), "must not promise --force: {msg}");
        }

        #[test]
        fn preflight_refuses_a_busy_port_immediately() {
            // Bind an ephemeral port so something is listening, then confirm preflight refuses
            // it (no daemon pidfile names this port, so it reads as a foreign process). The
            // ephemeral port avoids clashing with any real daemon's configured port.
            let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();

            let start = std::time::Instant::now();
            let err = preflight_port(port, false).expect_err("busy port must be refused");
            assert!(
                start.elapsed() < std::time::Duration::from_secs(2),
                "foreign-port refusal must be fast, took {:?}",
                start.elapsed()
            );
            assert!(
                err.downcast_ref::<PortInUse>().is_some(),
                "must surface PortInUse, got: {err}"
            );
            // It must be attributed to a foreign owner, not to llmtrim — otherwise the user is
            // told to `llmtrim stop` a process llmtrim can't stop.
            let msg = err.to_string();
            assert!(
                msg.contains("another process") && !msg.contains("--force"),
                "foreign port must read as foreign, got: {msg}"
            );
        }

        #[test]
        fn port_in_use_survives_anyhow_wrapping_for_the_supervisor() {
            // run_supervised fails fast only if it can downcast the error back to PortInUse.
            // Guard the core invariant: a regression that double-wraps or renames the type would
            // silently restore the 5× retry-spin this fix removes.
            let e: anyhow::Error = PortInUse {
                port: 1234,
                by_llmtrim_pid: None,
            }
            .into();
            assert!(e.downcast_ref::<PortInUse>().is_some());
        }

        #[test]
        fn capture_cap_evicts_oldest_until_under_limit() {
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-captest-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            // Ten 100-byte files, oldest (lowest timestamp prefix) first.
            for i in 0..10u32 {
                std::fs::write(dir.join(format!("{i:020}-abc.json")), vec![b'x'; 100]).unwrap();
            }
            // Cap at 450 bytes → must drop the 6 oldest to land at 4 files (400 bytes).
            enforce_capture_cap(&dir, 450);
            let mut left: Vec<String> = std::fs::read_dir(&dir)
                .unwrap()
                .flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            left.sort();
            assert_eq!(left.len(), 4, "should keep only what fits: {left:?}");
            assert!(
                left[0].starts_with(&format!("{:020}", 6)),
                "the 6 oldest go first: {left:?}"
            );
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn capture_cap_keeps_everything_under_limit() {
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-captest2-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            for i in 0..3u32 {
                std::fs::write(dir.join(format!("{i:020}-abc.json")), vec![b'x'; 100]).unwrap();
            }
            enforce_capture_cap(&dir, 10_000);
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn capture_cap_ignores_non_json_and_subdirs() {
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-captest3-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            // A README and a subdir that must survive even when the cap is breached, plus two
            // captures whose combined size already exceeds the limit.
            std::fs::write(dir.join("README.md"), vec![b'x'; 100]).unwrap();
            std::fs::create_dir(dir.join("0000-sub")).unwrap();
            std::fs::write(dir.join(format!("{:020}-abc.json", 1)), vec![b'x'; 100]).unwrap();
            std::fs::write(dir.join(format!("{:020}-abc.json", 2)), vec![b'x'; 100]).unwrap();
            // Cap of 50 bytes is below the 200 bytes of .json → both captures must go, but the
            // README and subdir (which sort first) must be untouched.
            enforce_capture_cap(&dir, 50);
            assert!(dir.join("README.md").exists(), "non-capture file kept");
            assert!(dir.join("0000-sub").is_dir(), "subdir kept");
            assert!(
                !dir.join(format!("{:020}-abc.json", 1)).exists(),
                "old capture evicted"
            );
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn capture_cap_handles_empty_dir() {
            let dir = std::env::temp_dir().join(format!(
                "llmtrim-captest4-{}-{}",
                std::process::id(),
                chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            enforce_capture_cap(&dir, 10); // no panic, nothing to do
            assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
            std::fs::remove_dir_all(&dir).ok();
        }

        #[test]
        fn extract_output_from_openai_json_and_sse() {
            // Non-streaming JSON answer.
            let json = br#"{"choices":[{"message":{"content":"hello world"}}]}"#;
            assert_eq!(
                extract_output_text(ProviderKind::OpenAi, json).as_deref(),
                Some("hello world")
            );
            // SSE stream: concatenate the content deltas, ignore [DONE].
            let sse = b"data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n\
                        data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n\
                        data: [DONE]\n\n";
            assert_eq!(
                extract_output_text(ProviderKind::OpenAi, sse).as_deref(),
                Some("hello")
            );
        }

        #[test]
        fn extract_output_anthropic_and_gemini_sse() {
            let anthropic = b"event: content_block_delta\n\
                              data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                              data: {\"type\":\"content_block_delta\",\"delta\":{\"text\":\" there\"}}\n\n";
            assert_eq!(
                extract_output_text(ProviderKind::Anthropic, anthropic).as_deref(),
                Some("hi there")
            );
            let gemini =
                b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"abc\"}]}}]}\n\n";
            assert_eq!(
                extract_output_text(ProviderKind::Google, gemini).as_deref(),
                Some("abc")
            );
        }

        #[test]
        fn extract_output_none_on_garbage() {
            assert_eq!(
                extract_output_text(ProviderKind::OpenAi, b"not json or sse"),
                None
            );
            assert_eq!(extract_output_text(ProviderKind::OpenAi, b""), None);
        }

        #[test]
        fn forward_headers_drops_host_and_length() {
            let mut h = header::HeaderMap::new();
            h.insert(header::AUTHORIZATION, "Bearer x".parse().unwrap());
            h.insert(header::HOST, "api.anthropic.com".parse().unwrap());
            h.insert(header::CONTENT_LENGTH, "123".parse().unwrap());
            let fwd = forward_headers(&h);
            assert!(fwd.iter().any(|(k, _)| k == "authorization"));
            assert!(
                !fwd.iter()
                    .any(|(k, _)| k == "host" || k == "content-length")
            );
        }

        #[test]
        fn set_answer_rewrites_each_provider() {
            let mut oa = serde_json::json!({"choices":[{"message":{"content":"old"}}]});
            set_answer(&mut oa, ProviderKind::OpenAi, "new");
            assert_eq!(
                oa.pointer("/choices/0/message/content")
                    .and_then(serde_json::Value::as_str),
                Some("new")
            );
            let mut an = serde_json::json!({"content":[{"type":"text","text":"old"}]});
            set_answer(&mut an, ProviderKind::Anthropic, "new");
            assert_eq!(
                an.pointer("/content/0/text")
                    .and_then(serde_json::Value::as_str),
                Some("new")
            );
        }

        #[test]
        fn output_usage_reads_provider_counts() {
            // Anthropic SSE: message_start has a partial count, message_delta the true total.
            let anthropic_sse = concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":50,\"output_tokens\":1}}}\n\n",
                "data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
                "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":42}}\n\n",
            );
            assert_eq!(
                extract_output_usage(ProviderKind::Anthropic, anthropic_sse.as_bytes()),
                Some(42)
            );
            // OpenAI streaming with include_usage: usage rides the final chunk.
            let openai_sse =
                "data: {\"choices\":[],\"usage\":{\"completion_tokens\":17}}\n\ndata: [DONE]\n\n";
            assert_eq!(
                extract_output_usage(ProviderKind::OpenAi, openai_sse.as_bytes()),
                Some(17)
            );
            // Non-streaming JSON body (Gemini's usageMetadata).
            let gemini_json = "{\"usageMetadata\":{\"candidatesTokenCount\":9}}";
            assert_eq!(
                extract_output_usage(ProviderKind::Google, gemini_json.as_bytes()),
                Some(9)
            );
            // Responses API: non-streaming body reports `output_tokens`…
            let responses_json = "{\"usage\":{\"input_tokens\":17,\"output_tokens\":42}}";
            assert_eq!(
                extract_output_usage(ProviderKind::OpenAi, responses_json.as_bytes()),
                Some(42)
            );
            // …and the streaming total rides the final `response.completed` event.
            let responses_sse = "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"output_tokens\":9}}}\n\n";
            assert_eq!(
                extract_output_usage(ProviderKind::OpenAi, responses_sse.as_bytes()),
                Some(9)
            );
            // No usage present → None (caller falls back to tokenizing the text).
            assert_eq!(
                extract_output_usage(ProviderKind::Anthropic, b"data: {\"type\":\"ping\"}\n\n"),
                None
            );
        }

        #[test]
        fn routes_known_hosts_to_providers() {
            assert_eq!(
                provider_for_host("api.anthropic.com"),
                Some(ProviderKind::Anthropic)
            );
            assert_eq!(
                provider_for_host("generativelanguage.googleapis.com"),
                Some(ProviderKind::Google)
            );
            assert_eq!(
                provider_for_host("api.openai.com"),
                Some(ProviderKind::OpenAi)
            );
            assert_eq!(
                provider_for_host("api.deepseek.com"),
                Some(ProviderKind::OpenAi)
            );
            assert_eq!(
                provider_for_host("openrouter.ai"),
                Some(ProviderKind::OpenAi)
            );
            // Curated extra hosts intercept too; the exact wire shape is refined from the body
            // at compress time, so the host-level kind is just the OpenAI-shaped default.
            assert_eq!(provider_for_host("opencode.ai"), Some(ProviderKind::OpenAi));
            assert_eq!(
                provider_for_host("api.groq.com"),
                Some(ProviderKind::OpenAi)
            );
            assert_eq!(
                provider_for_host("ai-gateway.vercel.sh"),
                Some(ProviderKind::OpenAi)
            );
            // Codex CLI with ChatGPT sign-in posts to chatgpt.com, not api.openai.com.
            assert_eq!(provider_for_host("chatgpt.com"), Some(ProviderKind::OpenAi));
            // Non-LLM hosts are not intercepted.
            assert_eq!(provider_for_host("github.com"), None);
            assert_eq!(provider_for_host("example.com"), None);
        }

        #[test]
        fn body_shape_picks_adapter_regardless_of_host() {
            // A Claude-shaped body (top-level `system`) detects as Anthropic even though it
            // arrived on a generic gateway host — the interceptor adapts by shape, not host.
            let claude_body = serde_json::json!({
                "model": "glm-4.6",
                "system": "you are a helpful assistant",
                "messages": [{"role": "user", "content": "hi"}],
            });
            assert_eq!(
                llmtrim_core::provider::detect(&claude_body),
                Some(ProviderKind::Anthropic)
            );
        }

        #[test]
        fn vertex_path_selects_wire_shape() {
            // OpenAI-compatible endpoint.
            assert_eq!(
                vertex_kind(
                    "/v1/projects/p/locations/us-central1/endpoints/openapi/chat/completions"
                ),
                Some(ProviderKind::OpenAi)
            );
            // Claude-on-Vertex (rawPredict).
            assert_eq!(
                vertex_kind(
                    "/v1/projects/p/locations/l/publishers/anthropic/models/claude-opus-4:rawPredict"
                ),
                Some(ProviderKind::Anthropic)
            );
            // Gemini-on-Vertex (streaming generateContent).
            assert_eq!(
                vertex_kind(
                    "/v1/projects/p/locations/l/publishers/google/models/gemini-2.5-pro:streamGenerateContent"
                ),
                Some(ProviderKind::Google)
            );
            assert_eq!(
                vertex_kind("/v1/projects/p/locations/l/operations/123"),
                None
            );
        }

        #[test]
        fn generated_ca_is_a_parseable_constrained_ca() {
            let (cert_pem, key_pem) = generate_ca(&intercept_domains()).unwrap();
            assert!(cert_pem.contains("BEGIN CERTIFICATE"));
            assert!(key_pem.contains("PRIVATE KEY"));
            // Round-trips through the same parser hudsucker uses.
            let key = hudsucker::rcgen::KeyPair::from_pem(&key_pem).unwrap();
            assert!(hudsucker::rcgen::Issuer::from_ca_cert_pem(&cert_pem, key).is_ok());
        }

        #[test]
        fn minted_leaf_carries_aki_keyusage_and_eku() {
            // Regression: hudsucker 0.24's RcgenAuthority mints a leaf with only a SAN, which
            // strict OpenSSL 3.x rejects ("Missing Authority Key Identifier"), breaking every
            // httpx / OpenAI-SDK (Python) client behind the proxy. LeafCertAuthority must add
            // the Authority Key Identifier plus Key Usage and Extended Key Usage.
            let (cert_pem, key_pem) = generate_ca(&intercept_domains()).unwrap();
            let key = hudsucker::rcgen::KeyPair::from_pem(&key_pem).unwrap();
            let issuer = hudsucker::rcgen::Issuer::from_ca_cert_pem(&cert_pem, key).unwrap();
            let ca = LeafCertAuthority::new(issuer, aws_lc_rs::default_provider());
            let der = ca.gen_cert("example.com");
            let bytes: &[u8] = der.as_ref();
            let has = |oid: &[u8]| bytes.windows(oid.len()).any(|w| w == oid);
            // X.509 extension OIDs, DER-encoded as `06 03 55 1D xx`.
            assert!(
                has(&[0x06, 0x03, 0x55, 0x1D, 0x23]),
                "leaf must carry the Authority Key Identifier (2.5.29.35)"
            );
            assert!(
                has(&[0x06, 0x03, 0x55, 0x1D, 0x0F]),
                "leaf must carry Key Usage (2.5.29.15)"
            );
            assert!(
                has(&[0x06, 0x03, 0x55, 0x1D, 0x25]),
                "leaf must carry Extended Key Usage (2.5.29.37)"
            );
        }

        #[test]
        fn intercept_domains_include_registry_and_extras_sorted() {
            let d = intercept_domains();
            assert!(d.contains(&"api.openai.com".to_string())); // exact registry endpoint host
            assert!(d.contains(&"opencode.ai".to_string())); // from EXTRA_HOSTS
            assert!(d.contains(&"chatgpt.com".to_string())); // Codex CLI, ChatGPT sign-in
            assert!(
                d.windows(2).all(|w| w[0] <= w[1]),
                "must be sorted for sidecar compare"
            );
        }

        #[test]
        fn codex_responses_path_is_compressible() {
            assert!(is_compressible_path("/backend-api/codex/responses"));
        }

        #[test]
        fn user_extra_hosts_flow_into_intercept_domains() {
            // A user-configured host must reach the CA sidecar set so it is actually intercepted.
            let user = vec!["llm.acme.com".to_string()];
            let d = intercept_domains_with(&user);
            assert!(d.contains(&"llm.acme.com".to_string()));
            assert!(
                d.contains(&"api.openai.com".to_string()),
                "registry still present"
            );
            assert!(
                d.windows(2).all(|w| w[0] <= w[1]),
                "stays sorted for sidecar compare"
            );
        }

        #[test]
        fn user_extra_hosts_match_exactly_not_by_subdomain() {
            // Curated EXTRA_HOSTS match a host and its subdomains; user hosts match ONLY the exact
            // host named, so a typo'd apex can't widen interception to its siblings/subdomains.
            let user = vec!["llm.acme.com".to_string()];
            assert!(
                extra_host_match("llm.acme.com", &user),
                "exact user host matches"
            );
            assert!(
                !extra_host_match("api.llm.acme.com", &user),
                "a subdomain of a user host is NOT intercepted"
            );
            assert!(!extra_host_match("acme.com", &user));
            assert!(!extra_host_match("notllm.acme.com", &user));
            // Curated extras keep subdomain matching.
            assert!(extra_host_match("api.groq.com", &[]));
            assert!(extra_host_match("foo.api.groq.com", &[]));
        }

        #[test]
        fn exclusion_match_by_host_and_provider() {
            // Provider exclusion is by canonical wire-shape name (coarse): excludes every host of
            // that shape.
            let excl_providers = vec!["anthropic".to_string()];
            assert!(exclusion_match(
                "api.anthropic.com",
                ProviderKind::Anthropic,
                &[],
                &excl_providers,
            ));
            assert!(
                !exclusion_match("api.openai.com", ProviderKind::OpenAi, &[], &excl_providers),
                "a different wire shape is not excluded"
            );
            // Host exclusion is exact — one OpenAI-shaped host opts out without affecting siblings.
            let excl_hosts = vec!["openrouter.ai".to_string()];
            assert!(exclusion_match(
                "openrouter.ai",
                ProviderKind::OpenAi,
                &excl_hosts,
                &[],
            ));
            assert!(
                !exclusion_match("api.groq.com", ProviderKind::OpenAi, &excl_hosts, &[]),
                "another OpenAI-shaped host stays compressed"
            );
            assert!(
                !exclusion_match("api.openrouter.ai", ProviderKind::OpenAi, &excl_hosts, &[]),
                "host match is exact, not by subdomain"
            );
            // Empty lists exclude nothing.
            assert!(!exclusion_match(
                "api.anthropic.com",
                ProviderKind::Anthropic,
                &[],
                &[],
            ));
        }

        #[test]
        fn intercept_set_does_not_widen_to_shared_cloud_parents() {
            // Security: we key on exact endpoint hosts, never the registrable parent — else the
            // name-constrained MITM CA could forge certs for, and intercept, all of Google
            // Cloud / Alibaba Cloud / Volcano Engine. Guard against a regression to parents.
            let d = intercept_domains();
            for bad in [
                "googleapis.com",
                "aliyuncs.com",
                "volces.com",
                "amazonaws.com",
            ] {
                assert!(
                    !d.contains(&bad.to_string()),
                    "must not cover shared infra {bad}"
                );
            }
            // But the specific LLM endpoints under those parents are still covered.
            let set: std::collections::HashSet<String> = d.into_iter().collect();
            assert!(host_covered("generativelanguage.googleapis.com", &set));
            assert!(host_covered("aiplatform.googleapis.com", &set));
            // A non-LLM Google host is NOT intercepted.
            assert!(!host_covered("storage.googleapis.com", &set));
        }

        #[test]
        fn compressible_path_allows_generation_blocks_embeddings_and_count() {
            for ok in [
                "/v1/chat/completions",
                "/v1/responses",
                "/v1/messages",
                "/v1beta/models/gemini-2.0-flash:generateContent",
                "/v1/projects/p/locations/l/publishers/anthropic/models/claude:rawPredict",
            ] {
                assert!(is_compressible_path(ok), "should compress {ok}");
            }
            for skip in [
                "/v1/embeddings",
                "/v1/moderations",
                "/v1/messages/count_tokens",
                "/v1beta/models/gemini-2.0-flash:countTokens",
                "/v1/audio/transcriptions",
            ] {
                assert!(!is_compressible_path(skip), "must NOT compress {skip}");
            }
        }

        #[test]
        fn model_from_path_extracts_gemini_and_vertex_only() {
            // Gemini/Vertex put the model in the URL; OpenAI/Anthropic paths have no
            // `/models/` segment so the body's `model` is used instead.
            assert_eq!(
                model_from_path("/v1beta/models/gemini-2.0-flash:generateContent"),
                Some("gemini-2.0-flash")
            );
            assert_eq!(
                model_from_path("/v1beta/models/gemini-2.5-pro:streamGenerateContent"),
                Some("gemini-2.5-pro")
            );
            assert_eq!(
                model_from_path(
                    "/v1/projects/p/locations/l/publishers/google/models/gemini-3-pro:streamGenerateContent"
                ),
                Some("gemini-3-pro")
            );
            assert_eq!(model_from_path("/v1/chat/completions"), None);
            assert_eq!(model_from_path("/v1/messages"), None);
        }

        #[test]
        fn host_covered_matches_exact_and_subdomains_only() {
            let domains: std::collections::HashSet<String> =
                ["openai.com".to_string(), "opencode.ai".to_string()]
                    .into_iter()
                    .collect();
            assert!(host_covered("api.openai.com", &domains)); // subdomain
            assert!(host_covered("opencode.ai", &domains)); // exact
            assert!(!host_covered("openai.com.evil.com", &domains)); // suffix spoof rejected
            assert!(!host_covered("github.com", &domains));
        }

        #[test]
        fn ca_is_current_only_when_present_and_sidecar_matches() {
            let want = vec!["a.com".to_string(), "b.com".to_string()];
            assert!(ca_is_current(true, Some(&want), &want));
            assert!(!ca_is_current(false, Some(&want), &want)); // no cert on disk
            assert!(!ca_is_current(true, None, &want)); // no sidecar → regenerate
            assert!(!ca_is_current(true, Some(&["a.com".to_string()]), &want)); // host added
        }

        #[test]
        fn compress_blocking_rejects_non_json() {
            use llmtrim_core::config::DenseConfig;
            use llmtrim_core::ir::ProviderKind;
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY);
            assert!(
                compress_blocking(&cfg, b"not json", ProviderKind::OpenAi, None, &memo, None)
                    .is_none()
            );
            assert!(
                compress_blocking(&cfg, b"", ProviderKind::OpenAi, None, &memo, None).is_none()
            );
        }

        #[test]
        fn compress_blocking_net_win_guard() {
            use llmtrim_core::config::DenseConfig;
            use llmtrim_core::ir::ProviderKind;
            // A tiny request: compression overhead exceeds savings → gate fires → None.
            let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY);
            // May return None (net loss) or Some (net win); both are valid.
            // The important assertion: when Some, the compressed form must have fewer input tokens.
            if let Some((compressed, pending)) = compress_blocking(
                &cfg,
                body.as_bytes(),
                ProviderKind::OpenAi,
                None,
                &memo,
                None,
            ) {
                assert!(
                    pending.input_after <= pending.input_before,
                    "net-win guard: compressed must not exceed original ({} vs {})",
                    pending.input_after,
                    pending.input_before
                );
                assert!(!compressed.is_empty(), "compressed body must not be empty");
            }
        }

        #[test]
        fn compress_blocking_compresses_repetitive_content() {
            use llmtrim_core::config::DenseConfig;
            use llmtrim_core::ir::ProviderKind;
            // 50 identical log lines: dedup should fire and compression should win.
            let lines = "ERROR database connection pool exhausted, retrying in 5s\n".repeat(50);
            let body = serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": lines}]
            })
            .to_string();
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY);
            // Dedup is default-on and the spam is contiguous: this must compress, so a
            // `None` here is a regression (the test must not pass vacuously).
            let (compressed, pending) = compress_blocking(
                &cfg,
                body.as_bytes(),
                ProviderKind::OpenAi,
                None,
                &memo,
                None,
            )
            .expect("50 identical log lines must produce a net token win");
            assert!(
                pending.input_after < pending.input_before,
                "50 identical log lines should compress: {} -> {}",
                pending.input_before,
                pending.input_after
            );
            assert!(
                serde_json::from_str::<serde_json::Value>(&compressed).is_ok(),
                "compressed body must remain valid JSON"
            );
        }

        // ── M0-2: fallback gate ─────────────────────────────────────────────

        #[test]
        fn replay_triggered_on_4xx() {
            // 400 Bad Request and 422 Unprocessable Entity trigger replay; other codes do not.
            assert!(should_replay(400), "400 must trigger replay");
            assert!(should_replay(422), "422 must trigger replay");
            assert!(!should_replay(200), "200 must not trigger replay");
            assert!(
                !should_replay(429),
                "429 (rate-limit) must not trigger replay"
            );
            assert!(!should_replay(500), "500 must not trigger replay");
            assert!(!should_replay(503), "503 must not trigger replay");
        }

        #[test]
        fn sub_fallback_triggers_on_capacity_and_transient_failures() {
            // Quota/forbidden, throttling, overload, and transient server failures reroute.
            for s in [402, 403, 408, 425, 429, 500, 502, 503, 504, 529] {
                assert!(is_sub_fallback_status(s), "{s} must trigger the fallback");
            }
            // A success, validation errors, and missing Anthropic auth stay with the client.
            for s in [200, 400, 401, 422] {
                assert!(
                    !is_sub_fallback_status(s),
                    "{s} must not trigger the fallback"
                );
            }
        }

        #[test]
        fn sub_fallback_detects_embedded_stream_errors() {
            assert!(is_sub_fallback_body(
                br#"data: {"type":"error","error":{"message":"overloaded"}}

"#
            ));
            // A non-streamed JSON error body.
            assert!(is_sub_fallback_body(
                br#"{"type":"error","error":{"type":"overloaded_error"}}"#
            ));
            assert!(!is_sub_fallback_body(
                br#"data: {"type":"message_stop","stop_reason":"end_turn"}

"#
            ));
            // An assistant message that merely *quotes* an error frame is not a failed turn.
            assert!(!is_sub_fallback_body(
                br#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"{\"type\":\"error\"}"}}

"#
            ));
        }

        #[test]
        fn sse_committed_content_marks_the_point_of_no_failover() {
            assert!(sse_committed_content(
                br#"data: {"type":"message_start","message":{"id":"m"}}

"#
            ));
            assert!(sse_committed_content(
                br#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}

"#
            ));
            // A ping alone commits nothing: the turn can still be failed over.
            assert!(!sse_committed_content(
                br#"data: {"type":"ping"}

"#
            ));
        }

        #[tokio::test]
        async fn probe_sse_head_streams_a_good_response_through_unbuffered() {
            let body = Body::from(Full::new(Bytes::from(
                "event: message_start\ndata: {\"type\":\"message_start\"}\n\n\
                 data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
                 data: {\"type\":\"message_stop\"}\n\n",
            )));
            let out = probe_sse_head(body)
                .await
                .expect("a good stream must not fail over");
            let bytes = out.collect().await.unwrap().to_bytes();
            let text = String::from_utf8_lossy(&bytes);
            assert!(text.contains("message_start"), "head is replayed");
            assert!(text.contains("message_stop"), "tail still streams");
        }

        #[tokio::test]
        async fn probe_sse_head_fails_over_on_an_error_frame() {
            let body = Body::from(Full::new(Bytes::from(
                "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\"}}\n\n",
            )));
            assert!(
                probe_sse_head(body).await.is_err(),
                "an error frame before any content must fail over"
            );
        }

        #[test]
        fn apply_sub_effort_injects_output_config_and_no_ops_safely() {
            // Injects into a fresh body.
            let mut v = serde_json::json!({ "model": "x" });
            apply_sub_effort(&mut v, Some("high"));
            assert_eq!(v["output_config"]["effort"], "high");
            // Overrides an existing output_config.effort (proxy-side override wins).
            let mut v = serde_json::json!({ "output_config": { "effort": "low", "format": {} } });
            apply_sub_effort(&mut v, Some("xhigh"));
            assert_eq!(v["output_config"]["effort"], "xhigh");
            assert!(
                v["output_config"]["format"].is_object(),
                "other keys preserved"
            );
            // None is a no-op.
            let mut v = serde_json::json!({ "model": "x" });
            apply_sub_effort(&mut v, None);
            assert!(v.get("output_config").is_none());
            // Non-object body is a safe no-op (no panic).
            let mut v = serde_json::json!("not an object");
            apply_sub_effort(&mut v, Some("high"));
            assert!(v.is_string());
        }

        // ── M0-2: compress_blocking actual-compression guarantee ─────────────

        #[test]
        fn compress_blocking_actual_compression_with_large_body() {
            use llmtrim_core::config::DenseConfig;
            use llmtrim_core::ir::ProviderKind;
            // A substantive system prompt + user turn — large enough that Stage A (truncation)
            // and Stage B (dedup) can fire and the net-win guard lets the result through.
            let system = "You are a senior software engineer reviewing pull requests. \
                          Be concise. Focus on correctness, performance, and maintainability. \
                          Do not repeat the same comment twice. \
                          Do not repeat the same comment twice. \
                          Do not repeat the same comment twice. \
                          Always explain *why*, not just *what*. \
                          Always explain *why*, not just *what*. \
                          Always explain *why*, not just *what*.";
            let body = serde_json::json!({
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": "Please review this diff:\n```\n-    x = x + 1\n+    x += 1\n```\n"}
                ]
            })
            .to_string();
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY);
            if let Some((compressed_json, pending)) = compress_blocking(
                &cfg,
                body.as_bytes(),
                ProviderKind::OpenAi,
                None,
                &memo,
                None,
            ) {
                // compress_blocking now always returns Some for valid JSON, even on zero-savings
                // inputs (the caller records the row; `input_after == input_before` means
                // passthrough). Assert that we never *increase* the token count.
                assert!(
                    pending.input_after <= pending.input_before,
                    "compress_blocking must not increase token count ({} -> {})",
                    pending.input_before,
                    pending.input_after
                );
                // The result must be valid JSON — upstream would reject garbage.
                serde_json::from_str::<serde_json::Value>(&compressed_json)
                    .expect("compressed body must be valid JSON");
                // The `original` field is NOT set by compress_blocking — the caller fills it.
                assert!(
                    pending.original.is_none(),
                    "compress_blocking must leave `original` unset for the caller to fill"
                );
            }
            // None is returned only for non-JSON / non-UTF-8 input — not for zero-savings bodies.
        }

        // ── async handler tests ──────────────────────────────────────────────
        //
        // These call `handle_request_inner` / `handle_response_inner` directly
        // (the thin wrappers introduced as testability seams) so we don't need to
        // construct `hudsucker::HttpContext`, which is `#[non_exhaustive]`.

        /// Build a minimal `Interceptor` wired to an in-process mpsc ledger.
        /// Returns the handler and the receiver for asserting emitted records.
        fn make_interceptor() -> (
            Interceptor,
            std::sync::mpsc::Receiver<crate::tracking::Record>,
        ) {
            let (tx, rx) = std::sync::mpsc::channel();
            // The breakdown ledger is drained into a channel the tests ignore.
            let (breakdown_tx, _breakdown_rx) = std::sync::mpsc::channel();
            let handler = Interceptor {
                config: Arc::new(llmtrim_core::config::DenseConfig::default()),
                ledger: tx,
                breakdown_ledger: breakdown_tx,
                domains: Arc::new(intercept_domains().into_iter().collect()),
                memo: Arc::new(Memo::with_capacity(llmtrim_core::memo::DEFAULT_CAPACITY)),
                pending: None,
                streaming: false,
                upstream_proxy: None,
                exclude_hosts: Arc::new(Vec::new()),
                exclude_providers: Arc::new(Vec::new()),
                sub: None,
                sub_tiers: Arc::new(std::collections::BTreeMap::new()),
                sub_chain: Arc::new(vec![]),
                sub_fallback: false,
                sub_effort: None,
            };
            (handler, rx)
        }

        #[test]
        fn build_breakdown_reconciles_usage_with_prefix_tape() {
            use llmtrim_core::attribution::{
                BlockAttribution, Bucket, RequestIdentity, Section, Zone,
            };
            // Three input blocks (system 100, schema 50, user text 50) → 200 raw tokens.
            let block = |bucket: Bucket, section: Section, tok: usize| BlockAttribution {
                zone: Zone::Input,
                section,
                bucket,
                mcp_server: None,
                tool_name: None,
                role: None,
                msg_index: None,
                tokens: tok,
            };
            let pending = Pending {
                provider: ProviderKind::Anthropic,
                model: Some("claude-sonnet-4".to_string()),
                tokenizer: "t".to_string(),
                exact: true,
                input_before: 200,
                input_after: 200,
                compress_micros: 0,
                plan: String::new(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: Some(BreakdownPending {
                    blocks: vec![
                        block(Bucket::System, Section::Static, 100),
                        block(Bucket::Schema, Section::Static, 50),
                        block(Bucket::Text, Section::Messages, 50),
                    ],
                    identity: RequestIdentity {
                        session_id: Some("s1".to_string()),
                        agent: Some("claude-code".to_string()),
                        project: None,
                    },
                    window: 200_000,
                    cc_session_id: None,
                }),
                reroute: None,
                fallback: None,
            };
            // Provider billed: 120 cached read, 30 cache write, 50 fresh (sum 200), 40 output.
            let usage = ResponseUsage {
                output_after: Some(40),
                cache_read: Some(120),
                fresh_input: Some(50),
                cache_write: Some(30),
            };
            let payload = build_breakdown(&pending, &usage).expect("breakdown built");
            // One synthetic output block is appended to the three input blocks.
            assert_eq!(payload.blocks.len(), 4);
            // Per-split sums match the provider usage exactly (calibration is lossless here).
            let sum = |f: fn(&crate::tracking::BreakdownBlock) -> f64| -> f64 {
                payload.blocks.iter().map(f).sum()
            };
            assert!((sum(|b| b.cache_read_tok) - 120.0).abs() < 1e-6);
            assert!((sum(|b| b.cache_write_tok) - 30.0).abs() < 1e-6);
            assert!((sum(|b| b.fresh_tok) - 50.0).abs() < 1e-6);
            assert!((sum(|b| b.output_tok) - 40.0).abs() < 1e-6);
            // Prefix tape: the 100-token system block is wholly within the 120 cached prefix.
            let sys = &payload.blocks[0];
            assert_eq!(sys.label, "System prompt");
            assert!((sys.cache_read_tok - 100.0).abs() < 1e-6);
            assert_eq!(payload.turn.session_id, "s1");
        }

        /// Build a POST `Request<Body>` with a JSON body directed at `uri`.
        fn post_request(uri: &str, body: &str) -> Request<Body> {
            Request::builder()
                .method(Method::POST)
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .expect("valid request")
        }

        /// Spin up a stub HTTP/1.1 server on an ephemeral loopback port. It accepts
        /// exactly one connection, reads (discards) the request, then writes the given
        /// status line and response body. Returns the bound port immediately.
        ///
        /// Because `replay_original` calls `transport::forward_post` (blocking ureq),
        /// the server is a plain `std::thread` — the OS TCP accept is the synchronization
        /// primitive, no sleeps needed.
        fn stub_http_server(status_line: &str, response_body: &str) -> u16 {
            use std::io::{Read, Write};
            use std::net::TcpListener;

            let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
            let port = listener.local_addr().expect("local addr").port();
            let status_line = status_line.to_string();
            let response_body = response_body.to_string();
            std::thread::spawn(move || {
                if let Ok((mut stream, _)) = listener.accept() {
                    // Drain the FULL request (headers + Content-Length body), not just the
                    // headers: closing a socket with unread data sends RST on Windows, and
                    // the client then errors before it can read the response.
                    let mut buf = [0u8; 4096];
                    let mut acc: Vec<u8> = Vec::new();
                    let mut want: Option<usize> = None;
                    loop {
                        match stream.read(&mut buf) {
                            Ok(0) | Err(_) => break,
                            Ok(n) => {
                                acc.extend_from_slice(&buf[..n]);
                                if want.is_none()
                                    && let Some(pos) = acc.windows(4).position(|w| w == b"\r\n\r\n")
                                {
                                    let headers = String::from_utf8_lossy(&acc[..pos]);
                                    let cl = headers
                                        .lines()
                                        .find_map(|l| {
                                            l.split_once(':').and_then(|(k, v)| {
                                                k.eq_ignore_ascii_case("content-length")
                                                    .then(|| v.trim().parse::<usize>().ok())
                                                    .flatten()
                                            })
                                        })
                                        .unwrap_or(0);
                                    want = Some(pos + 4 + cl);
                                }
                                if want.is_some_and(|w| acc.len() >= w) {
                                    break;
                                }
                            }
                        }
                    }
                    let cl = response_body.len();
                    let response = format!(
                        "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {cl}\r\nConnection: close\r\n\r\n{response_body}"
                    );
                    let _ = stream.write_all(response.as_bytes());
                }
            });
            port
        }

        /// Consume a `Body` to completion and return the collected bytes.
        async fn drain_body(body: Body) -> Vec<u8> {
            use http_body_util::BodyExt;
            body.collect()
                .await
                .map(|c| c.to_bytes().to_vec())
                .unwrap_or_default()
        }

        // Test 1: compression path ────────────────────────────────────────────

        /// `handle_request_inner` must compress a compressible body, set `pending` with
        /// `input_after < input_before`, stash `pending.original`, and return a smaller
        /// valid-JSON request body.
        #[tokio::test]
        async fn handle_request_inner_compresses_body_and_sets_pending() {
            let (mut handler, rx) = make_interceptor();

            // 50 duplicate log lines: dedup fires, net token win guaranteed.
            let log = "ERROR pool exhausted, retrying in 5s\n".repeat(50);
            let input_json = serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": log}]
            })
            .to_string();
            let input_len = input_json.len();

            let req = post_request("https://api.openai.com/v1/chat/completions", &input_json);
            let result = handler.handle_request_inner(req).await;

            // Must be a Request (forwarded compressed), not a short-circuit Response.
            let RequestOrResponse::Request(out_req) = result else {
                panic!("handle_request_inner returned a Response, expected a compressed Request");
            };

            // pending must be set: compression happened.
            let pending = handler
                .pending
                .as_ref()
                .expect("pending must be set after a compressible request");

            assert!(
                pending.input_after < pending.input_before,
                "input_after ({}) must be < input_before ({})",
                pending.input_after,
                pending.input_before
            );

            // The original body must be stashed for replay.
            let orig = pending
                .original
                .as_ref()
                .expect("original must be stashed in pending");
            assert_eq!(
                orig.body,
                input_json.as_bytes(),
                "stashed original must equal the pre-compression bytes"
            );

            // Outbound body must be valid JSON and smaller.
            let out_body = drain_body(out_req.into_body()).await;
            serde_json::from_slice::<serde_json::Value>(&out_body)
                .expect("compressed request body must be valid JSON");
            assert!(
                out_body.len() < input_len,
                "compressed body ({} bytes) must be smaller than original ({input_len} bytes)",
                out_body.len()
            );

            // No ledger record from handle_request_inner alone.
            assert!(
                rx.try_recv().is_err(),
                "no ledger record must be emitted from handle_request_inner alone"
            );
        }

        // Test 1b: WebSocket refusal ──────────────────────────────────────────

        #[test]
        fn is_websocket_upgrade_detects_h1_upgrade_header() {
            let ws = Request::builder()
                .method(Method::GET)
                .uri("https://chatgpt.com/backend-api/codex/responses")
                .header(header::UPGRADE, "websocket")
                .header(header::CONNECTION, "Upgrade")
                .body(Body::empty())
                .expect("valid request");
            assert!(is_websocket_upgrade(&ws));

            let plain = post_request("https://api.openai.com/v1/chat/completions", "{}");
            assert!(!is_websocket_upgrade(&plain));
        }

        /// A WebSocket upgrade on an intercepted host is short-circuited with 426 so the client
        /// falls back to the compressible HTTPS transport — never forwarded, never compressed.
        #[tokio::test]
        async fn handle_request_inner_refuses_websocket_upgrade_with_426() {
            let (mut handler, rx) = make_interceptor();
            let req = Request::builder()
                .method(Method::GET)
                .uri("https://chatgpt.com/backend-api/codex/responses")
                .header(header::UPGRADE, "websocket")
                .header(header::CONNECTION, "Upgrade")
                .body(Body::empty())
                .expect("valid request");

            let result = handler.handle_request_inner(req).await;

            let RequestOrResponse::Response(res) = result else {
                panic!("a WebSocket upgrade must be refused with a Response, not forwarded");
            };
            assert_eq!(res.status(), hudsucker::hyper::StatusCode::UPGRADE_REQUIRED);
            // Refusal is not a compressed request: no pending state, no ledger record.
            assert!(handler.pending.is_none());
            assert!(rx.try_recv().is_err());
        }

        // Test 2: non-compressible path ───────────────────────────────────────

        /// A request to `/v1/embeddings` must pass through verbatim; `pending` stays None.
        #[tokio::test]
        async fn handle_request_inner_passes_embeddings_verbatim() {
            let (mut handler, _rx) = make_interceptor();
            let body = r#"{"model":"text-embedding-3-small","input":"hello world"}"#;
            let req = post_request("https://api.openai.com/v1/embeddings", body);

            let result = handler.handle_request_inner(req).await;

            let RequestOrResponse::Request(out_req) = result else {
                panic!("expected passthrough Request, got Response");
            };
            assert!(
                handler.pending.is_none(),
                "pending must remain None for a non-compressible path"
            );
            let out_body = drain_body(out_req.into_body()).await;
            assert_eq!(out_body, body.as_bytes());
        }

        /// A compressible request whose host/provider is on the exclude list must pass through
        /// verbatim (still routed, just not compressed): `pending` stays None and the body is
        /// returned byte-identical. Covers the `exclusion_match` gate in `handle_request_inner`.
        #[tokio::test]
        async fn handle_request_inner_excludes_matching_host_and_provider() {
            let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}]}"#;
            let uri = "https://api.openai.com/v1/chat/completions";

            // Exclude by exact host.
            let (mut handler, _rx) = make_interceptor();
            handler.exclude_hosts = Arc::new(vec!["api.openai.com".to_string()]);
            let result = handler.handle_request_inner(post_request(uri, body)).await;
            let RequestOrResponse::Request(out_req) = result else {
                panic!("expected passthrough Request, got Response");
            };
            assert!(handler.pending.is_none(), "excluded host must not compress");
            assert_eq!(drain_body(out_req.into_body()).await, body.as_bytes());

            // Exclude by provider wire shape (api.openai.com → OpenAi).
            let (mut handler, _rx) = make_interceptor();
            handler.exclude_providers = Arc::new(vec!["openai".to_string()]);
            let result = handler.handle_request_inner(post_request(uri, body)).await;
            let RequestOrResponse::Request(out_req) = result else {
                panic!("expected passthrough Request, got Response");
            };
            assert!(
                handler.pending.is_none(),
                "excluded provider must not compress"
            );
            assert_eq!(drain_body(out_req.into_body()).await, body.as_bytes());
        }

        /// A mixed-case `Host` header still matches a (lowercased) exclude entry — the gate
        /// lowercases the host before comparing, like `provider_for_host` does.
        #[tokio::test]
        async fn handle_request_inner_exclude_host_is_case_insensitive() {
            let (mut handler, _rx) = make_interceptor();
            handler.exclude_hosts = Arc::new(vec!["api.openai.com".to_string()]);
            let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}]}"#;
            let req = post_request("https://API.OpenAI.COM/v1/chat/completions", body);

            let result = handler.handle_request_inner(req).await;

            let RequestOrResponse::Request(out_req) = result else {
                panic!("expected passthrough Request, got Response");
            };
            assert!(
                handler.pending.is_none(),
                "mixed-case host must still be excluded"
            );
            assert_eq!(drain_body(out_req.into_body()).await, body.as_bytes());
        }

        // Test 3: replay on 400 ───────────────────────────────────────────────

        /// When the upstream returns 400 on a compressed request, `handle_response_inner`
        /// must replay the original and return the replayed response. The ledger record
        /// must show zero savings (input_after == input_before).
        #[tokio::test]
        async fn handle_response_inner_replays_original_on_400() {
            let replay_body = r#"{"error":{"message":"replayed ok"}}"#;
            let port = stub_http_server("HTTP/1.1 200 OK", replay_body);
            let (mut handler, rx) = make_interceptor();

            let original_body =
                br#"{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}"#;
            handler.pending = Some(Pending {
                provider: ProviderKind::OpenAi,
                model: Some("gpt-4o".to_string()),
                tokenizer: "tiktoken".to_string(),
                exact: true,
                input_before: 100,
                input_after: 60,
                compress_micros: 1_000,
                plan: "[]".to_string(),
                original: Some(OriginalRequest {
                    url: format!("http://127.0.0.1:{port}/v1/chat/completions"),
                    headers: vec![("content-type".to_string(), "application/json".to_string())],
                    body: original_body.to_vec(),
                }),
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            let bad_resp = Response::builder()
                .status(400)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"error":"bad request"}"#))
                .expect("build 400");

            let out = handler.handle_response_inner(bad_resp).await;

            // Replay succeeded: status is what the stub server returned (200).
            assert_eq!(
                out.status().as_u16(),
                200,
                "replayed response must carry the stub's status, not the original 400"
            );
            let body_bytes = drain_body(out.into_body()).await;
            assert_eq!(
                body_bytes,
                replay_body.as_bytes(),
                "replayed body must be the stub server's response verbatim"
            );

            // Zero savings recorded.
            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("ledger must receive one record after replay");
            assert_eq!(
                rec.input_after, rec.input_before,
                "replay records zero savings (input_after == input_before)"
            );
            assert_eq!(rec.provider, "openai");
            assert!(rx.try_recv().is_err(), "exactly one record after replay");
        }

        // Test 4: 400 without original does not loop ──────────────────────────

        /// When `pending.original` is None, a 400 must NOT attempt replay. The response
        /// passes through to the SSE tee path; Finalize emits one record.
        #[tokio::test]
        async fn handle_response_inner_400_without_original_does_not_replay() {
            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::OpenAi,
                model: None,
                tokenizer: "tiktoken".to_string(),
                exact: true,
                input_before: 50,
                input_after: 30,
                compress_micros: 500,
                plan: "[]".to_string(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            let bad_resp = Response::builder()
                .status(400)
                .body(Body::from(r#"{"error":"bad"}"#))
                .expect("build 400");

            let out = handler.handle_response_inner(bad_resp).await;

            assert_eq!(
                out.status().as_u16(),
                400,
                "without original, 400 must be forwarded as-is"
            );

            // Drain so Finalize drops and emits the record.
            let _ = drain_body(out.into_body()).await;
            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("Finalize emits a record even on a 400 passthrough");
            assert_eq!(rec.provider, "openai");
        }

        // Test 5: 422 triggers replay ─────────────────────────────────────────

        /// 422 Unprocessable Entity must also trigger replay.
        #[tokio::test]
        async fn handle_response_inner_replays_original_on_422() {
            let replay_body = r#"{"id":"msg_ok"}"#;
            let port = stub_http_server("HTTP/1.1 201 Created", replay_body);
            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::Anthropic,
                model: Some("claude-opus-4-5".to_string()),
                tokenizer: "approx".to_string(),
                exact: false,
                input_before: 200,
                input_after: 120,
                compress_micros: 2_000,
                plan: "[]".to_string(),
                original: Some(OriginalRequest {
                    url: format!("http://127.0.0.1:{port}/v1/messages"),
                    headers: vec![],
                    body: b"{}".to_vec(),
                }),
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            let unprocessable = Response::builder()
                .status(422)
                .body(Body::from(r#"{"type":"error"}"#))
                .expect("build 422");

            let out = handler.handle_response_inner(unprocessable).await;

            assert_eq!(out.status().as_u16(), 201, "stub returned 201");
            let body_bytes = drain_body(out.into_body()).await;
            assert_eq!(body_bytes, replay_body.as_bytes());

            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("record after 422 replay");
            assert_eq!(
                rec.input_after, rec.input_before,
                "422 replay => zero savings"
            );
            assert!(rx.try_recv().is_err(), "exactly one record on 422 replay");
        }

        // Test 6: non-replay statuses pass through ────────────────────────────

        /// 429 and 5xx must NOT trigger replay. The response streams through and
        /// Finalize records real (non-zeroed) savings.
        #[tokio::test]
        async fn handle_response_inner_non_replay_statuses_pass_through() {
            for status in [429u16, 500, 503] {
                let (mut handler, rx) = make_interceptor();

                handler.pending = Some(Pending {
                    provider: ProviderKind::OpenAi,
                    model: None,
                    tokenizer: "tiktoken".to_string(),
                    exact: true,
                    input_before: 100,
                    input_after: 60,
                    compress_micros: 1_000,
                    plan: "[]".to_string(),
                    original: Some(OriginalRequest {
                        // Port 1 has nothing listening; replay would fail if attempted.
                        url: "http://127.0.0.1:1".to_string(),
                        headers: vec![],
                        body: b"{}".to_vec(),
                    }),
                    output_shaped: false,
                    frozen_input_tokens: None,
                    breakdown: None,
                    reroute: None,
                    fallback: None,
                });

                let resp = Response::builder()
                    .status(status)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(format!(r#"{{"error":{status}}}"#)))
                    .expect("build response");

                let out: Response<Body> = handler.handle_response_inner(resp).await;
                assert_eq!(
                    out.status().as_u16(),
                    status,
                    "status {status} must pass through without replay"
                );

                let _ = drain_body(out.into_body()).await;

                let rec = rx
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .expect("Finalize record");
                assert_eq!(
                    rec.input_before, 100,
                    "status {status}: input_before must be from compressed pending"
                );
                assert_eq!(
                    rec.input_after, 60,
                    "status {status}: input_after must be from compressed pending"
                );
                assert!(rx.try_recv().is_err(), "exactly one record for {status}");
            }
        }

        // Test 7: SSE tee — Finalize fires exactly once ───────────────────────

        /// A 200 SSE response must stream through byte-for-byte; Finalize must emit
        /// exactly one record (with measured output tokens) after body consumed.
        /// Synchronization is via body consumption — no sleeps.
        #[tokio::test]
        async fn handle_response_inner_sse_tee_finalize_runs_exactly_once() {
            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::OpenAi,
                model: Some("gpt-4o".to_string()),
                tokenizer: "tiktoken".to_string(),
                exact: true,
                input_before: 300,
                input_after: 180,
                compress_micros: 5_000,
                plan: "[]".to_string(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            let sse_chunks = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\" world\"}}]}\n\n",
                "data: {\"usage\":{\"completion_tokens\":7}}\n\n",
                "data: [DONE]\n\n",
            );

            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from(sse_chunks))
                .expect("build SSE response");

            let out: Response<Body> = handler.handle_response_inner(resp).await;

            assert_eq!(out.status().as_u16(), 200);
            assert!(
                handler.pending.is_none(),
                "pending must be taken by handle_response_inner"
            );

            // Consuming the body drops the tee stream's `Finalize`.
            let out_bytes = drain_body(out.into_body()).await;

            assert_eq!(
                out_bytes,
                sse_chunks.as_bytes(),
                "SSE body must pass through byte-for-byte"
            );

            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("Finalize must emit exactly one record after body consumed");

            assert_eq!(
                rec.output_after,
                Some(7),
                "Finalize must extract completion_tokens from the SSE usage event"
            );
            assert_eq!(rec.input_before, 300);
            assert_eq!(rec.input_after, 180);
            assert_eq!(rec.provider, "openai");
            assert!(rx.try_recv().is_err(), "Finalize emits exactly one record");
        }

        // Test 8: passthrough when no pending ─────────────────────────────────

        /// When `self.pending` is None, `handle_response_inner` must be a pure
        /// passthrough — no body modification, no ledger record.
        #[tokio::test]
        async fn handle_response_inner_passthrough_when_no_pending() {
            let (mut handler, rx) = make_interceptor();

            let resp = Response::builder()
                .status(200)
                .body(Body::from("some upstream body"))
                .expect("build response");

            let out: Response<Body> = handler.handle_response_inner(resp).await;

            assert_eq!(out.status().as_u16(), 200);
            let body = drain_body(out.into_body()).await;
            assert_eq!(body, b"some upstream body");
            assert!(
                rx.try_recv().is_err(),
                "no ledger record for a passthrough response"
            );
        }

        // Test 9: SSE events split across chunk boundaries ────────────────────

        /// The tee accumulates bytes across all frames into `acc` before extraction.
        /// Usage extraction must succeed even when a `data:` event is split across
        /// multiple network chunks (mid-field) and when a multi-byte UTF-8 scalar is
        /// split across frames. The client must receive every byte verbatim in order,
        /// and exactly one ledger record must be emitted.
        #[tokio::test]
        async fn handle_response_inner_sse_split_across_chunk_boundaries() {
            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::OpenAi,
                model: Some("gpt-4o".to_string()),
                tokenizer: "tiktoken".to_string(),
                exact: true,
                input_before: 200,
                input_after: 120,
                compress_micros: 3_000,
                plan: "[]".to_string(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            // The complete SSE payload.
            let full_sse = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
                "data: {\"usage\":{\"completion_tokens\":7}}\n\n",
                "data: [DONE]\n\n",
            );

            // Split the SSE bytes so the usage event straddles two chunks:
            //   chunk 0 ends at: …"completion_to
            //   chunk 1 starts:  kens":7}}\n\n…
            // The accumulator collects both before extraction, so the full `data:` line
            // is present and parseable — this is the contract we assert.
            let split_at = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n".len()
                + "data: {\"usage\":{\"completion_to".len();
            let (head, tail) = full_sse.as_bytes().split_at(split_at);

            // Two extra chunks carry a UTF-8 multibyte scalar (U+00E9 = é, 0xC3 0xA9)
            // split across frame boundaries — exercises the byte-stream path without
            // affecting usage extraction (these bytes sit outside any `data:` event).
            let extra_byte_1: &[u8] = &[0xC3]; // first byte of é
            let extra_byte_2: &[u8] = &[0xA9]; // second byte of é

            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
                Ok(bytes::Bytes::copy_from_slice(head)),
                Ok(bytes::Bytes::copy_from_slice(tail)),
                Ok(bytes::Bytes::copy_from_slice(extra_byte_1)),
                Ok(bytes::Bytes::copy_from_slice(extra_byte_2)),
            ];
            let stream = hudsucker::futures::stream::iter(chunks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .expect("build split-chunk SSE response");

            let out: Response<Body> = handler.handle_response_inner(resp).await;

            assert_eq!(out.status().as_u16(), 200);
            assert!(
                handler.pending.is_none(),
                "pending must be taken by handle_response_inner"
            );

            // Drain — triggers Finalize::drop once all frames are consumed.
            let out_bytes = drain_body(out.into_body()).await;

            // (a) Client receives every byte verbatim in order.
            let mut expected = Vec::new();
            expected.extend_from_slice(full_sse.as_bytes());
            expected.extend_from_slice(extra_byte_1);
            expected.extend_from_slice(extra_byte_2);
            assert_eq!(
                out_bytes, expected,
                "client must receive all bytes verbatim, in order, across chunk boundaries"
            );

            // (b) Finalize reassembles the full accumulator and finds completion_tokens.
            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("Finalize must emit one record after body consumed");
            assert_eq!(
                rec.output_after,
                Some(7),
                "completion_tokens must be extracted even when the usage event was split across chunks"
            );
            assert_eq!(rec.input_before, 200);
            assert_eq!(rec.input_after, 120);
            assert_eq!(rec.provider, "openai");

            // (c) Exactly one record.
            assert!(
                rx.try_recv().is_err(),
                "exactly one ledger record must be emitted"
            );
        }

        // Test 10: client abort mid-stream ────────────────────────────────────

        /// When the client drops the response body before the stream ends, Finalize
        /// must fire on drop and emit exactly one ledger record. The process must not
        /// hang. Synchronization: drop the partial body then recv_timeout — no sleeps.
        #[tokio::test]
        async fn handle_response_inner_client_abort_mid_stream_emits_one_record() {
            use http_body_util::BodyExt as _;

            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::Anthropic,
                model: None,
                tokenizer: "approx".to_string(),
                exact: false,
                input_before: 150,
                input_after: 90,
                compress_micros: 2_000,
                plan: "[]".to_string(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            // Three chunks; the client will consume only the first then drop the body.
            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![
                Ok(bytes::Bytes::from_static(
                    b"data: {\"delta\":{\"text\":\"hel\"}}\n\n",
                )),
                Ok(bytes::Bytes::from_static(
                    b"data: {\"delta\":{\"text\":\"lo\"}}\n\n",
                )),
                Ok(bytes::Bytes::from_static(
                    b"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":5}}\n\n",
                )),
            ];
            let stream = hudsucker::futures::stream::iter(chunks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .expect("build multi-chunk SSE response");

            let out: Response<Body> = handler.handle_response_inner(resp).await;
            assert_eq!(out.status().as_u16(), 200);

            // Consume one frame then drop the body to simulate a client abort.
            let mut body = out.into_body();
            let _first_frame = body
                .frame()
                .await
                .expect("stream must yield at least one frame")
                .expect("frame must be Ok");
            // Dropping the body here drops the stream closure, which drops Finalize.
            drop(body);

            // Finalize::drop is synchronous (plain mpsc send); recv_timeout is sufficient.
            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("Finalize must emit exactly one record on client abort");

            assert_eq!(
                rec.provider, "anthropic",
                "record must carry the pending provider"
            );
            assert_eq!(rec.input_before, 150, "input_before from pending");
            assert_eq!(rec.input_after, 90, "input_after from pending");

            // No second record.
            assert!(
                rx.try_recv().is_err(),
                "exactly one record must be emitted on abort — not two"
            );
        }

        // Test 11: empty SSE body / zero chunks ───────────────────────────────

        /// A 200 response with an immediately-ended stream must emit exactly one
        /// ledger record. `output_after` must be `None` — nothing to extract from an
        /// empty accumulator. No panic.
        #[tokio::test]
        async fn handle_response_inner_empty_body_emits_one_record_no_panic() {
            let (mut handler, rx) = make_interceptor();

            handler.pending = Some(Pending {
                provider: ProviderKind::OpenAi,
                model: Some("gpt-4o".to_string()),
                tokenizer: "tiktoken".to_string(),
                exact: true,
                input_before: 80,
                input_after: 50,
                compress_micros: 1_000,
                plan: "[]".to_string(),
                original: None,
                output_shaped: false,
                frozen_input_tokens: None,
                breakdown: None,
                reroute: None,
                fallback: None,
            });

            // Zero-chunk stream: body ends immediately.
            let chunks: Vec<Result<bytes::Bytes, std::io::Error>> = vec![];
            let stream = hudsucker::futures::stream::iter(chunks);
            let resp = Response::builder()
                .status(200)
                .header(header::CONTENT_TYPE, "text/event-stream")
                .body(Body::from_stream(stream))
                .expect("build zero-chunk SSE response");

            let out: Response<Body> = handler.handle_response_inner(resp).await;

            assert_eq!(out.status().as_u16(), 200);
            assert!(
                handler.pending.is_none(),
                "pending must be taken even for an empty body"
            );

            // Drain immediately — the body is already ended; Finalize fires on drop.
            let out_bytes = drain_body(out.into_body()).await;
            assert!(
                out_bytes.is_empty(),
                "empty stream yields empty client bytes"
            );

            let rec = rx
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("Finalize must emit one record even for an empty body");

            // Empty accumulator → no usage and no answer text → output_after is None.
            assert_eq!(
                rec.output_after, None,
                "empty body: output_after must be None (nothing to extract)"
            );
            assert_eq!(rec.input_before, 80);
            assert_eq!(rec.input_after, 50);
            assert_eq!(rec.provider, "openai");

            assert!(
                rx.try_recv().is_err(),
                "exactly one ledger record for an empty body"
            );
        }

        // --- C1 regression: origin TLS is verified through the CONNECT tunnel ---
        //
        // `ProxyConnector::from_proxy` (with the `rustls-tls-native-roots` feature active)
        // sets its internal `tls` field to `Some(TlsConnector)`, which is what hyper-http-proxy
        // uses to TLS-wrap the ORIGIN connection that is tunnelled through CONNECT.
        //
        // `from_proxy_unsecured` leaves `tls: None`, which means the tunnelled origin stream
        // would be returned as a PLAINTEXT, UNVERIFIED stream — the security bug this test
        // guards against.
        //
        // The Debug output from hyper-http-proxy's `ProxyConnector` encodes this distinction:
        //   - `tls: Some(_)` → `"ProxyConnector { proxies: ... }"` (no unsecured marker)
        //   - `tls: None`    → `"ProxyConnector (unsecured) { proxies: ... }"`
        //
        // Coverage boundary: this test verifies the *construction* path (TLS connector is
        // wired in), not a live TLS handshake. A live handshake against a self-signed cert
        // that should fail certificate verification would require a full local CONNECT+TLS
        // harness; that is considered integration-test territory and is not included here.
        // What the test guarantees: if someone reverts `from_proxy` to `from_proxy_unsecured`,
        // the test catches it immediately.
        #[test]
        fn proxy_connector_origin_tls_is_active() {
            // ProxyConnector::from_proxy internally builds a tokio-rustls ClientConfig.
            // tokio-rustls requires a CryptoProvider to be installed; we use the same
            // aws-lc-rs provider that production installs at daemon startup.
            let _ = aws_lc_rs::default_provider().install_default();

            let proxy_uri: hudsucker::hyper::Uri = "http://proxy.example.test:3128"
                .parse()
                .expect("test proxy URI");
            let upstream_proxy_spec = UpstreamProxy::new(Intercept::All, proxy_uri);
            let connector = ProxyConnector::from_proxy(HttpConnector::new(), upstream_proxy_spec)
                .expect("ProxyConnector::from_proxy must succeed");

            // hyper-http-proxy's Debug impl emits "(unsecured)" only when `tls` is None.
            // Assert its absence to confirm the origin leg has a verifying TLS connector.
            let debug_str = format!("{connector:?}");
            assert!(
                !debug_str.contains("(unsecured)"),
                "ProxyConnector must have a TLS connector for the origin leg \
                 (built with from_proxy, not from_proxy_unsecured). \
                 Debug output: {debug_str}"
            );
        }
    }
}
