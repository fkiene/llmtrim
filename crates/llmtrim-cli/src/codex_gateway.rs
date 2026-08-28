//! A loopback HTTP gateway that Codex can be pointed at with its own native configuration.
//!
//! Codex supports a user-level `[model_providers.<name>]` block with a `base_url`. When that
//! provider carries no `env_key` and no `auth` block, `auth_manager_for_provider` hands it the
//! session's own auth manager, and `BearerAuthProvider::add_auth_headers` attaches
//! `Authorization: Bearer <chatgpt token>` plus `ChatGPT-Account-ID` — regardless of where
//! `base_url` points. So Codex will happily send its normal, ChatGPT-authenticated Responses
//! request to `http://127.0.0.1:<port>` in plaintext.
//!
//! That removes the entire reason the legacy path needs a MITM proxy: no `HTTPS_PROXY`, no
//! local CA, no leaf certificates, no trust-store or `NODE_EXTRA_CA_CERTS` wiring. The gateway
//! reads a plain local request, compresses the body with the same `llmtrim_core` the proxy
//! uses, and forwards it upstream over ordinary HTTPS.
//!
//! Everything in this module is a pure function or sits behind the [`GatewayUpstream`] seam, so
//! the whole pipeline is exercised offline without opening a socket or reaching the network.

use std::collections::BTreeMap;

use llmtrim_core::config::DenseConfig;
use llmtrim_core::ir::ProviderKind;

pub mod socket;
pub mod upstream;

/// The only origin this gateway will ever talk to. Not configurable, not derived from the
/// request: a client that can reach the loopback port must not be able to choose a
/// destination, or the gateway becomes an open SSRF relay.
pub const UPSTREAM_ORIGIN: &str = "https://chatgpt.com";

/// The only route the gateway serves, matching what Codex posts for `wire_api = "responses"`.
pub const GATEWAY_ROUTE: &str = "/backend-api/codex/responses";

/// The compression profile the first gateway version pins: llmtrim's lossless input-only
/// preset. Not configurable here — a silently promoted profile would change what the model
/// sees without anyone asking for it.
pub const GATEWAY_PRESET: &str = "safe";

/// Headers the gateway forwards verbatim. `Authorization` and `ChatGPT-Account-ID` are the
/// credentials Codex attached; they are passed through in memory and never inspected for
/// meaning, logged or written down.
pub const FORWARDED_HEADERS: [&str; 5] = [
    "authorization",
    "chatgpt-account-id",
    "content-type",
    "accept",
    "originator",
];

/// Hop-by-hop headers that must not be relayed, plus the ones the upstream leg recomputes.
pub const DROPPED_HEADERS: [&str; 7] = [
    "host",
    "content-length",
    "connection",
    "proxy-authorization",
    "proxy-connection",
    "transfer-encoding",
    "upgrade",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    /// The request did not use the one method the route accepts.
    MethodNotAllowed,
    /// The request targeted something other than the single served route.
    RouteNotFound,
    /// Codex did not attach the credentials it normally attaches.
    MissingAuthorization,
    /// The `Authorization` header was present but not in the expected bearer form.
    MalformedAuthorization,
    /// Compression failed. The gateway fails closed rather than quietly forwarding the
    /// original body, so a session never silently runs without llmtrim.
    CompressionFailed,
    /// The upstream leg failed. No fallback, no alternate provider, no retry.
    UpstreamFailed,
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::MethodNotAllowed => "method not allowed",
            Self::RouteNotFound => "route not served by the codex gateway",
            Self::MissingAuthorization => "request carried no authorization header",
            Self::MalformedAuthorization => "authorization header was not a bearer token",
            Self::CompressionFailed => "compression failed; refusing to forward uncompressed",
            Self::UpstreamFailed => "upstream request failed",
        })
    }
}

impl std::error::Error for GatewayError {}

impl GatewayError {
    /// The status the gateway answers Codex with. Kept coarse on purpose.
    #[must_use]
    pub fn status(&self) -> u16 {
        match self {
            Self::MethodNotAllowed => 405,
            Self::RouteNotFound => 404,
            Self::MissingAuthorization | Self::MalformedAuthorization => 401,
            Self::CompressionFailed => 500,
            Self::UpstreamFailed => 502,
        }
    }
}

/// A request as it arrived on the loopback listener. Header names are lowercase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayRequest {
    pub method: String,
    pub path: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

/// What the gateway sends upstream, and what a test asserts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamRequest {
    pub url: String,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

/// The upstream answer, streamed back to Codex chunk by chunk and otherwise untouched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    /// Response body in the order the upstream produced it. For an SSE stream each element is
    /// one wire chunk; the gateway must not merge, reorder or rewrite them.
    pub chunks: Vec<Vec<u8>>,
}

/// The network seam. Production implements it with an ordinary HTTPS client; tests implement
/// it with a fake, which is what keeps the whole pipeline offline.
pub trait GatewayUpstream {
    fn send(&mut self, request: UpstreamRequest) -> Result<UpstreamResponse, GatewayError>;
}

/// Accept only the exact method and route Codex uses.
pub fn validate_route(method: &str, path: &str) -> Result<(), GatewayError> {
    // Compare the path only, never a caller-supplied absolute URL: an absolute-form target
    // would carry a destination, which this gateway must never honour.
    let path = path.split('?').next().unwrap_or(path);
    if path != GATEWAY_ROUTE {
        return Err(GatewayError::RouteNotFound);
    }
    if !method.eq_ignore_ascii_case("POST") {
        return Err(GatewayError::MethodNotAllowed);
    }
    Ok(())
}

/// Require the credentials Codex attaches, by shape only.
///
/// The value is never parsed for meaning, decoded or compared against anything. This exists so
/// the loopback port is not an anonymous relay: without the caller's own bearer, no
/// authenticated upstream request happens.
pub fn auth_gate(headers: &BTreeMap<String, String>) -> Result<(), GatewayError> {
    let value = headers
        .get("authorization")
        .ok_or(GatewayError::MissingAuthorization)?;
    let rest = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .ok_or(GatewayError::MalformedAuthorization)?;
    if rest.trim().is_empty() {
        return Err(GatewayError::MalformedAuthorization);
    }
    Ok(())
}

/// The upstream URL. Built from two constants — nothing from the request contributes.
#[must_use]
pub fn upstream_url() -> String {
    format!("{UPSTREAM_ORIGIN}{GATEWAY_ROUTE}")
}

/// Select the headers to relay. Anything not on the forward list is dropped, so a client
/// cannot smuggle routing or proxy directives through the gateway.
#[must_use]
pub fn forward_headers(incoming: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    incoming
        .iter()
        .filter(|(name, _)| {
            let lowered = name.to_ascii_lowercase();
            FORWARDED_HEADERS.contains(&lowered.as_str())
                && !DROPPED_HEADERS.contains(&lowered.as_str())
        })
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect()
}

/// Compress the request body with the product's real transform.
///
/// Fail-closed by design, and deliberately different from the legacy proxy: there, a body that
/// does not shrink is forwarded verbatim so a user never pays more. Here the point is to know
/// the gateway is doing its job, so a failure surfaces instead of silently passing through.
pub fn compress_body(body: &[u8]) -> Result<Vec<u8>, GatewayError> {
    let text = std::str::from_utf8(body).map_err(|_| GatewayError::CompressionFailed)?;
    let config = DenseConfig::preset(GATEWAY_PRESET).ok_or(GatewayError::CompressionFailed)?;
    let result = llmtrim_core::compress_with_config(text, Some(ProviderKind::OpenAi), &config)
        .map_err(|_| GatewayError::CompressionFailed)?;
    Ok(result.request_json.into_bytes())
}

/// Byte counts for one exchange. Metadata only — no body, no header, no credential.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GatewayMetrics {
    pub local_request_bytes: u64,
    pub upstream_request_bytes: u64,
    pub response_chunks: u64,
    pub status: u16,
}

impl GatewayMetrics {
    /// Reduction in hundredths of a percent. Local estimate, not official Codex usage.
    #[must_use]
    pub fn reduction_basis_points(&self) -> i128 {
        if self.local_request_bytes == 0 {
            return 0;
        }
        let saved = i128::from(self.local_request_bytes) - i128::from(self.upstream_request_bytes);
        saved * 10_000 / i128::from(self.local_request_bytes)
    }
}

/// The whole pipeline: validate, gate, compress, forward, relay.
pub fn handle<U: GatewayUpstream>(
    request: &GatewayRequest,
    upstream: &mut U,
) -> Result<(UpstreamResponse, GatewayMetrics), GatewayError> {
    validate_route(&request.method, &request.path)?;
    auth_gate(&request.headers)?;

    let compressed = compress_body(&request.body)?;
    let outgoing = UpstreamRequest {
        url: upstream_url(),
        headers: forward_headers(&request.headers),
        body: compressed,
    };
    let metrics = GatewayMetrics {
        local_request_bytes: request.body.len() as u64,
        upstream_request_bytes: outgoing.body.len() as u64,
        ..GatewayMetrics::default()
    };

    let response = upstream.send(outgoing)?;
    let metrics = GatewayMetrics {
        response_chunks: response.chunks.len() as u64,
        status: response.status,
        ..metrics
    };
    Ok((response, metrics))
}

/// The loopback listener. Separated from [`handle`] so the pipeline stays testable without a
/// socket, and so the bind policy is a value that can be asserted rather than a literal buried
/// in a server call.
pub mod listener {
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    /// The only address family and host the gateway will bind. A MITM-free local gateway that
    /// answered on a routable address would hand every machine on the network a way to spend
    /// the user's ChatGPT quota.
    pub const BIND_HOST: Ipv4Addr = Ipv4Addr::LOCALHOST;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum BindError {
        /// Anything that is not IPv4 loopback.
        NotLoopback,
    }

    impl std::fmt::Display for BindError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("the codex gateway only binds 127.0.0.1")
        }
    }

    impl std::error::Error for BindError {}

    /// Resolve the address the gateway may listen on. The host is not configurable: only the
    /// port is, and an explicitly requested non-loopback host is refused rather than silently
    /// corrected, so a misconfiguration is visible.
    pub fn bind_address(
        requested_host: Option<IpAddr>,
        port: u16,
    ) -> Result<SocketAddr, BindError> {
        match requested_host {
            None => Ok(SocketAddr::from((BIND_HOST, port))),
            Some(IpAddr::V4(host)) if host == Ipv4Addr::LOCALHOST => {
                Ok(SocketAddr::from((BIND_HOST, port)))
            }
            Some(_) => Err(BindError::NotLoopback),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{BIND_HOST, BindError, bind_address};
        use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

        #[test]
        fn the_default_bind_is_ipv4_loopback() {
            let address = bind_address(None, 43117).expect("default binds");
            assert_eq!(address.ip(), IpAddr::V4(BIND_HOST));
            assert!(address.ip().is_loopback());
            assert_eq!(address.port(), 43117);
        }

        #[test]
        fn an_all_interfaces_bind_is_refused() {
            assert_eq!(
                bind_address(Some(IpAddr::V4(Ipv4Addr::UNSPECIFIED)), 43117),
                Err(BindError::NotLoopback)
            );
        }

        #[test]
        fn a_public_ipv6_bind_is_refused() {
            for host in [
                IpAddr::V6(Ipv6Addr::UNSPECIFIED),
                IpAddr::V6(Ipv6Addr::LOCALHOST),
                IpAddr::V6("::1".parse().expect("v6")),
            ] {
                assert_eq!(
                    bind_address(Some(host), 43117),
                    Err(BindError::NotLoopback),
                    "{host}"
                );
            }
        }

        #[test]
        fn a_routable_address_is_refused() {
            for host in ["192.168.1.10", "10.0.0.5", "8.8.8.8"] {
                assert_eq!(
                    bind_address(Some(host.parse().expect("ip")), 43117),
                    Err(BindError::NotLoopback),
                    "{host}"
                );
            }
        }

        #[test]
        fn explicit_loopback_is_accepted_and_normalized() {
            let address = bind_address(Some(IpAddr::V4(Ipv4Addr::LOCALHOST)), 0).expect("loopback");
            assert!(address.ip().is_loopback());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DROPPED_HEADERS, GATEWAY_PRESET, GATEWAY_ROUTE, GatewayError, GatewayRequest,
        GatewayUpstream, UPSTREAM_ORIGIN, UpstreamRequest, UpstreamResponse, auth_gate,
        compress_body, forward_headers, handle, upstream_url, validate_route,
    };
    use std::collections::BTreeMap;

    /// The production half of the whole gateway — this file and every submodule under it,
    /// each cut at its own test module. The scans below must cover the entire call path: a
    /// forbidden mechanism moved into `socket` or `upstream` would otherwise slip past them.
    fn production_source() -> String {
        [
            include_str!("codex_gateway.rs"),
            include_str!("codex_gateway/socket.rs"),
            include_str!("codex_gateway/upstream.rs"),
        ]
        .into_iter()
        .map(production_half)
        .collect::<Vec<_>>()
        .join("\n")
    }

    /// Everything before the test module, minus the comments. The scans must not read their
    /// own assertion lists, and comments are prose: the module docs name the very mechanisms
    /// this architecture avoids, so scanning them would forbid explaining the design.
    fn production_half(whole: &str) -> String {
        let marker = concat!("#[cfg(", "test)]");
        whole
            .split(marker)
            .next()
            .expect("production half")
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Fictitious sentinels. Any real credential would be a bug in the test, not a secret.
    const FAKE_BEARER: &str = "Bearer SUPER_SECRET_DO_NOT_LEAK";
    const FAKE_ACCOUNT: &str = "FAKE_ACCOUNT_DO_NOT_LEAK";

    /// Stands in for chatgpt.com. Records exactly what the gateway tried to send, so the test
    /// can assert on the wire without a socket or a network.
    #[derive(Default)]
    struct FakeUpstream {
        seen: Vec<UpstreamRequest>,
        reply: Option<UpstreamResponse>,
        fail: bool,
    }

    impl FakeUpstream {
        fn streaming() -> Self {
            Self {
                reply: Some(UpstreamResponse {
                    status: 200,
                    headers: BTreeMap::from([(
                        "content-type".to_string(),
                        "text/event-stream".to_string(),
                    )]),
                    chunks: vec![
                        b"event: response.created\ndata: {\"a\":1}\n\n".to_vec(),
                        b"event: response.output_text.delta\ndata: {\"d\":\"x\"}\n\n".to_vec(),
                        b"event: response.completed\ndata: {\"done\":true}\n\n".to_vec(),
                    ],
                }),
                ..Self::default()
            }
        }

        fn failing() -> Self {
            Self {
                fail: true,
                ..Self::default()
            }
        }
    }

    impl GatewayUpstream for FakeUpstream {
        fn send(&mut self, request: UpstreamRequest) -> Result<UpstreamResponse, GatewayError> {
            self.seen.push(request);
            if self.fail {
                return Err(GatewayError::UpstreamFailed);
            }
            Ok(self.reply.clone().unwrap_or(UpstreamResponse {
                status: 200,
                headers: BTreeMap::new(),
                chunks: vec![b"{}".to_vec()],
            }))
        }
    }

    /// A Responses-shaped body carrying repetitive tool output — the shape llmtrim shrinks.
    fn codex_body() -> Vec<u8> {
        let records: Vec<String> = (0..80)
            .map(|i| {
                format!(
                    r#"{{"requestId":"REQ-{i:06}","service":"billing","level":"INFO","latencyMs":{},"route":"/api/v1/billing/items","region":"sa-east-1","message":"request completed"}}"#,
                    12 + i
                )
            })
            .collect();
        // Pretty-printed, like real tool output: whitespace and repeated keys are exactly
        // what the lossless `safe` preset removes. A pre-minified fixture would leave it
        // nothing to do and prove nothing.
        let parsed: Vec<serde_json::Value> = records
            .iter()
            .map(|r| serde_json::from_str(r).expect("record"))
            .collect();
        let tool_output = serde_json::to_string_pretty(&parsed).expect("pretty tool output");
        let content = format!(
            "PRIVATE_PROMPT_DO_NOT_LEAK Read the log tool output and reply with the requestId of REQ-000057.\n\n{tool_output}"
        );
        serde_json::to_vec(&serde_json::json!({
            "model": "gpt-5.6-sol",
            "messages": [{ "role": "user", "content": content }],
        }))
        .expect("serialize body")
    }

    fn codex_request() -> GatewayRequest {
        GatewayRequest {
            method: "POST".to_string(),
            path: GATEWAY_ROUTE.to_string(),
            headers: BTreeMap::from([
                ("authorization".to_string(), FAKE_BEARER.to_string()),
                ("chatgpt-account-id".to_string(), FAKE_ACCOUNT.to_string()),
                ("content-type".to_string(), "application/json".to_string()),
                ("host".to_string(), "127.0.0.1:45999".to_string()),
                ("content-length".to_string(), "999".to_string()),
            ]),
            body: codex_body(),
        }
    }

    // ----- route and method -------------------------------------------------------------

    #[test]
    fn the_gateway_serves_exactly_one_route() {
        validate_route("POST", GATEWAY_ROUTE).expect("the codex route is served");
        assert_eq!(
            validate_route("POST", "/v1/responses"),
            Err(GatewayError::RouteNotFound)
        );
        assert_eq!(
            validate_route("POST", "/"),
            Err(GatewayError::RouteNotFound)
        );
        assert_eq!(
            validate_route("GET", GATEWAY_ROUTE),
            Err(GatewayError::MethodNotAllowed)
        );
    }

    #[test]
    fn an_absolute_form_target_is_not_a_destination() {
        // A proxy would treat this as "go to example.com". The gateway must not: it only ever
        // compares the path, and an absolute URL is simply not the served route.
        for hostile in [
            "http://example.com/backend-api/codex/responses",
            "https://api.openai.com/v1/responses",
            "//evil.example/backend-api/codex/responses",
        ] {
            assert_eq!(
                validate_route("POST", hostile),
                Err(GatewayError::RouteNotFound),
                "{hostile}"
            );
        }
    }

    // ----- auth -------------------------------------------------------------------------

    #[test]
    fn a_request_without_the_callers_own_bearer_never_reaches_upstream() {
        let mut request = codex_request();
        request.headers.remove("authorization");
        let mut upstream = FakeUpstream::default();
        assert_eq!(
            handle(&request, &mut upstream),
            Err(GatewayError::MissingAuthorization)
        );
        assert!(
            upstream.seen.is_empty(),
            "the loopback port must not be an anonymous relay"
        );
    }

    #[test]
    fn a_malformed_authorization_is_rejected_by_shape_alone() {
        for value in ["", "Token abc", "Bearer ", "Bearer    "] {
            let mut headers = BTreeMap::new();
            headers.insert("authorization".to_string(), value.to_string());
            assert!(auth_gate(&headers).is_err(), "{value:?}");
        }
        // Shape only: any non-empty bearer passes, because the gateway never interprets it.
        let mut ok = BTreeMap::new();
        ok.insert("authorization".to_string(), FAKE_BEARER.to_string());
        auth_gate(&ok).expect("a well-formed bearer passes");
    }

    // ----- upstream is fixed ------------------------------------------------------------

    #[test]
    fn the_upstream_is_two_constants_and_nothing_from_the_request() {
        assert_eq!(upstream_url(), format!("{UPSTREAM_ORIGIN}{GATEWAY_ROUTE}"));
        assert!(upstream_url().starts_with("https://chatgpt.com"));
    }

    #[test]
    fn no_header_can_redirect_the_gateway() {
        let mut request = codex_request();
        for (name, value) in [
            ("x-upstream", "https://evil.example"),
            ("host", "evil.example"),
            ("x-forwarded-host", "evil.example"),
            ("forwarded", "host=evil.example"),
        ] {
            request.headers.insert(name.to_string(), value.to_string());
        }
        let mut upstream = FakeUpstream::default();
        handle(&request, &mut upstream).expect("still handled");
        let sent = &upstream.seen[0];
        assert_eq!(
            sent.url,
            upstream_url(),
            "the destination is not negotiable"
        );
        for leaked in ["x-upstream", "x-forwarded-host", "forwarded", "host"] {
            assert!(!sent.headers.contains_key(leaked), "{leaked} was relayed");
        }
    }

    #[test]
    fn hop_by_hop_headers_are_not_relayed() {
        let mut request = codex_request();
        for name in DROPPED_HEADERS {
            request.headers.insert(name.to_string(), "x".to_string());
        }
        let forwarded = forward_headers(&request.headers);
        for name in DROPPED_HEADERS {
            assert!(!forwarded.contains_key(name), "{name}");
        }
    }

    // ----- credentials pass through, untouched ------------------------------------------

    #[test]
    fn the_callers_credentials_reach_upstream_byte_for_byte() {
        let request = codex_request();
        let mut upstream = FakeUpstream::default();
        handle(&request, &mut upstream).expect("handled");
        let sent = &upstream.seen[0];
        assert_eq!(
            sent.headers.get("authorization").map(String::as_str),
            Some(FAKE_BEARER)
        );
        assert_eq!(
            sent.headers.get("chatgpt-account-id").map(String::as_str),
            Some(FAKE_ACCOUNT)
        );
    }

    // ----- compression ------------------------------------------------------------------

    #[test]
    fn the_body_is_compressed_by_the_products_real_transform() {
        let request = codex_request();
        let mut upstream = FakeUpstream::default();
        let (_, metrics) = handle(&request, &mut upstream).expect("handled");
        let sent = &upstream.seen[0];
        assert!(
            sent.body.len() < request.body.len(),
            "upstream body {} must be smaller than the local body {}",
            sent.body.len(),
            request.body.len()
        );
        assert_eq!(metrics.local_request_bytes, request.body.len() as u64);
        assert_eq!(metrics.upstream_request_bytes, sent.body.len() as u64);
        assert!(metrics.reduction_basis_points() > 0);
    }

    #[test]
    fn the_compressed_body_keeps_what_the_task_needs() {
        let request = codex_request();
        let mut upstream = FakeUpstream::default();
        handle(&request, &mut upstream).expect("handled");
        let sent = String::from_utf8(upstream.seen[0].body.clone()).expect("utf-8");
        for marker in ["REQ-000057", "gpt-5.6-sol", "billing"] {
            assert!(sent.contains(marker), "transform dropped {marker}");
        }
    }

    #[test]
    fn the_gateway_pins_the_safe_profile() {
        assert_eq!(GATEWAY_PRESET, "safe");
        assert!(llmtrim_core::config::DenseConfig::preset(GATEWAY_PRESET).is_some());
    }

    #[test]
    fn compression_failure_is_closed_not_open() {
        // Invalid UTF-8 cannot be compressed. The legacy proxy would forward the original;
        // this gateway must refuse, so a session never silently runs without llmtrim.
        let mut request = codex_request();
        request.body = vec![0xff, 0xfe, 0xfd];
        let mut upstream = FakeUpstream::default();
        assert_eq!(
            handle(&request, &mut upstream),
            Err(GatewayError::CompressionFailed)
        );
        assert!(
            upstream.seen.is_empty(),
            "nothing may be forwarded when compression fails"
        );
        assert!(compress_body(&[0xff, 0xfe]).is_err());
    }

    // ----- response ---------------------------------------------------------------------

    #[test]
    fn a_streamed_response_is_relayed_chunk_for_chunk_in_order() {
        let request = codex_request();
        let mut upstream = FakeUpstream::streaming();
        let expected = upstream.reply.clone().expect("reply");
        let (response, metrics) = handle(&request, &mut upstream).expect("handled");
        assert_eq!(
            response.chunks, expected.chunks,
            "SSE order must be preserved"
        );
        assert_eq!(response.status, 200);
        assert_eq!(
            response.headers.get("content-type").map(String::as_str),
            Some("text/event-stream")
        );
        assert_eq!(metrics.response_chunks, 3);
    }

    #[test]
    fn an_upstream_failure_is_reported_not_worked_around() {
        let request = codex_request();
        let mut upstream = FakeUpstream::failing();
        assert_eq!(
            handle(&request, &mut upstream),
            Err(GatewayError::UpstreamFailed)
        );
        assert_eq!(upstream.seen.len(), 1, "no retry, no second provider");
    }

    // ----- what the gateway must not do -------------------------------------------------

    #[test]
    fn errors_never_render_a_credential() {
        for error in [
            GatewayError::MissingAuthorization,
            GatewayError::MalformedAuthorization,
            GatewayError::CompressionFailed,
            GatewayError::UpstreamFailed,
            GatewayError::RouteNotFound,
            GatewayError::MethodNotAllowed,
        ] {
            let rendered = format!("{error} {error:?}");
            for sentinel in [
                "SUPER_SECRET_DO_NOT_LEAK",
                "FAKE_ACCOUNT_DO_NOT_LEAK",
                "Bearer ",
            ] {
                assert!(!rendered.contains(sentinel), "{error:?} leaked {sentinel}");
            }
        }
    }

    #[test]
    fn metrics_carry_no_content() {
        let request = codex_request();
        let mut upstream = FakeUpstream::streaming();
        let (_, metrics) = handle(&request, &mut upstream).expect("handled");
        let rendered = format!("{metrics:?}");
        for sentinel in [
            "SUPER_SECRET_DO_NOT_LEAK",
            "FAKE_ACCOUNT_DO_NOT_LEAK",
            "REQ-",
            "billing",
        ] {
            assert!(!rendered.contains(sentinel), "metrics leaked {sentinel}");
        }
    }

    // ----- synthetic end-to-end ---------------------------------------------------------

    #[test]
    fn synthetic_end_to_end_fake_codex_to_gateway_to_fake_upstream() {
        let request = codex_request();
        let mut upstream = FakeUpstream::streaming();
        let expected_stream = upstream.reply.clone().expect("reply").chunks;

        let (response, metrics) = handle(&request, &mut upstream).expect("the exchange completes");

        // A: the client sent a controlled payload; B: the gateway received it.
        assert_eq!(metrics.local_request_bytes, request.body.len() as u64);
        // C + D: real compression happened and the upstream body is smaller.
        let sent = &upstream.seen[0];
        assert!(sent.body.len() < request.body.len());
        // E: the semantic marker survived.
        let sent_text = String::from_utf8(sent.body.clone()).expect("utf-8");
        assert!(sent_text.contains("REQ-000057"));
        // F + G: credentials arrived identical.
        assert_eq!(
            sent.headers.get("authorization").map(String::as_str),
            Some(FAKE_BEARER)
        );
        assert_eq!(
            sent.headers.get("chatgpt-account-id").map(String::as_str),
            Some(FAKE_ACCOUNT)
        );
        // H: nothing observable carries the sentinel.
        let observable = format!("{metrics:?} {:?} {}", response.status, upstream_url());
        assert!(!observable.contains("SUPER_SECRET_DO_NOT_LEAK"));
        // J: the stream came back unchanged, in order.
        assert_eq!(response.chunks, expected_stream);

        // Reported for the delivery; a local estimate, never official Codex usage.
        println!("LOCAL_REQUEST_BYTES={}", metrics.local_request_bytes);
        println!("UPSTREAM_REQUEST_BYTES={}", metrics.upstream_request_bytes);
        println!(
            "LOCAL_PAYLOAD_REDUCTION_PERCENT={:.2}",
            metrics.reduction_basis_points() as f64 / 100.0
        );
    }

    #[test]
    fn the_gateway_writes_nothing_to_disk() {
        // I: no capture file, no ledger, no log. The module has no filesystem surface at all,
        // which this test pins by construction: a future `fs::write` in the call path would
        // have to be added here to be reachable.
        let source = production_source();
        let source = source.as_str();
        for forbidden in [
            "fs::write",
            "File::create",
            "OpenOptions",
            "capture_dir",
            "remove_dir_all",
        ] {
            assert!(
                !source.contains(forbidden),
                "the gateway call path must not touch the filesystem: {forbidden}"
            );
        }
    }

    #[test]
    fn the_gateway_call_path_has_no_mitm_no_ca_and_no_proxy_env() {
        let source = production_source();
        let source = source.as_str();
        for forbidden in [
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NODE_EXTRA_CA_CERTS",
            "SSL_CERT_FILE",
            "CURL_CA_BUNDLE",
            "rcgen",
            "hudsucker",
            "ca.key",
            "CONNECT",
        ] {
            assert!(
                !source.contains(forbidden),
                "the new architecture must not reintroduce {forbidden}"
            );
        }
    }

    #[test]
    fn the_gateway_needs_no_api_key() {
        let source = production_source();
        let source = source.as_str();
        for forbidden in [
            "OPENAI_API_KEY",
            "CODEX_API_KEY",
            "ANTHROPIC_API_KEY",
            "env_key",
        ] {
            assert!(!source.contains(forbidden), "{forbidden} must not appear");
        }
    }
}
