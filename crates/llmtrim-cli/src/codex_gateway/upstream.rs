//! The upstream leg: one fixed HTTPS destination, no proxy, no redirects.
//!
//! Kept apart from the policy module so the one place that opens a network connection is a
//! small file a reviewer can read in a minute. Every setting on the client below is a
//! security property rather than a tuning knob, and each one is pinned by a test.

use std::collections::BTreeMap;
use std::io::Read;
use std::time::Duration;

use super::{GatewayError, UpstreamRequest, upstream_url};

/// Time allowed to resolve, connect and finish the TLS handshake.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Time allowed for the upstream to return its response *headers*. Only the first byte is
/// bounded here — a generation may take far longer than this to finish.
pub const RESPONSE_TIMEOUT: Duration = Duration::from_secs(120);

/// Ceiling on receiving the whole response body: long enough for a slow streamed generation,
/// short enough that a wedged connection cannot hold a thread forever.
pub const BODY_TIMEOUT: Duration = Duration::from_secs(1800);

/// Redirects followed: none. The destination is a constant, and a `Location` header is the
/// network asking to change it.
pub const MAX_REDIRECTS: u32 = 0;

/// Response headers relayed back to Codex. An allowlist, so the default for anything new is
/// to be dropped: cookies, framing the local leg recomputes, and the upstream's own transfer
/// encoding all stay out by construction.
pub const RELAYED_RESPONSE_HEADERS: [&str; 1] = ["content-type"];

/// An upstream response whose body is still arriving. The reader yields bytes as the upstream
/// produces them, which is what lets the gateway relay a stream instead of collecting one.
pub struct UpstreamStream {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Box<dyn Read>,
}

/// The streaming network seam. Production dials the fixed origin; a test hands back a scripted
/// reader, which is what keeps the socket suite offline.
pub trait StreamingUpstream: Send + Sync + 'static {
    /// Blocking. Returns once the response head is in, with the body still on the wire.
    ///
    /// # Errors
    ///
    /// [`GatewayError::UpstreamFailed`] for anything that went wrong on the network. There is
    /// no retry and no alternate provider: one attempt, one destination.
    fn send(&self, request: UpstreamRequest) -> Result<UpstreamStream, GatewayError>;
}

/// The one HTTP client the gateway uses.
///
/// `proxy(None)` is the important line: ureq's default is to adopt the ambient proxy
/// variables, which would route the user's ChatGPT traffic through whatever the environment
/// names. `https_only` and ureq's ordinary rustls verification keep the upstream leg a normal,
/// verified TLS connection — there is no custom root, no pinning bypass and nothing to install.
#[must_use]
pub fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .proxy(None)
        .https_only(true)
        .max_redirects(MAX_REDIRECTS)
        .max_redirects_will_error(false)
        .http_status_as_error(false)
        .timeout_connect(Some(HANDSHAKE_TIMEOUT))
        .timeout_recv_response(Some(RESPONSE_TIMEOUT))
        .timeout_recv_body(Some(BODY_TIMEOUT))
        .build()
        .into()
}

/// The gateway's real upstream: HTTPS to one constant URL.
pub struct HttpsUpstream {
    agent: ureq::Agent,
}

impl HttpsUpstream {
    #[must_use]
    pub fn new() -> Self {
        Self { agent: agent() }
    }
}

impl Default for HttpsUpstream {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingUpstream for HttpsUpstream {
    fn send(&self, request: UpstreamRequest) -> Result<UpstreamStream, GatewayError> {
        // The caller does not get to pick a destination. `prepare` builds this URL out of two
        // constants; re-checking it here means a later refactor cannot quietly make it a
        // parameter without a test noticing.
        if request.url != upstream_url() {
            return Err(GatewayError::UpstreamFailed);
        }
        let mut outgoing = self.agent.post(upstream_url());
        for (name, value) in &request.headers {
            outgoing = outgoing.header(name.as_str(), value.as_str());
        }
        let response = outgoing
            .send(&request.body[..])
            .map_err(|_| GatewayError::UpstreamFailed)?;
        let (parts, body) = response.into_parts();
        let headers = parts
            .headers
            .iter()
            .filter(|(name, _)| RELAYED_RESPONSE_HEADERS.contains(&name.as_str()))
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_string(), value.to_string()))
            })
            .collect();
        Ok(UpstreamStream {
            status: parts.status.as_u16(),
            headers,
            body: Box::new(body.into_reader()),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use super::{
        BODY_TIMEOUT, HANDSHAKE_TIMEOUT, HttpsUpstream, MAX_REDIRECTS, RELAYED_RESPONSE_HEADERS,
        RESPONSE_TIMEOUT, StreamingUpstream, agent,
    };

    use crate::codex_gateway::{GatewayError, UpstreamRequest, upstream_url};

    /// The production half of this file. The scans below must not read their own lists, and
    /// the prose necessarily names the mechanisms this leg avoids.
    fn production_source() -> String {
        let whole = include_str!("upstream.rs");
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

    #[test]
    fn the_upstream_url_is_the_one_https_destination() {
        assert_eq!(
            upstream_url(),
            "https://chatgpt.com/backend-api/codex/responses"
        );
    }

    #[test]
    fn the_production_upstream_refuses_every_url_but_the_constant() {
        let upstream = HttpsUpstream::new();
        for hostile in [
            "https://api.openai.com/v1/responses",
            "http://127.0.0.1:43117/backend-api/codex/responses",
            "https://chatgpt.example.com/backend-api/codex/responses",
        ] {
            let request = UpstreamRequest {
                url: hostile.to_string(),
                headers: BTreeMap::new(),
                body: Vec::new(),
            };
            assert_eq!(
                upstream.send(request).err(),
                Some(GatewayError::UpstreamFailed),
                "{hostile}"
            );
        }
    }

    #[test]
    fn the_upstream_client_never_adopts_an_ambient_proxy() {
        // ureq's own default is to take the proxy from the environment. That default would
        // route the user's ChatGPT traffic wherever the environment points, so the gateway
        // turns it off explicitly rather than hoping the variables are unset.
        assert!(agent().config().proxy().is_none());
    }

    #[test]
    fn the_upstream_client_does_not_follow_redirects() {
        assert_eq!(MAX_REDIRECTS, 0);
        let agent = agent();
        assert_eq!(agent.config().max_redirects(), 0);
        // A 3xx comes back as a response to relay, not as a hop the client takes on its own.
        assert!(!agent.config().max_redirects_will_error());
    }

    #[test]
    fn the_upstream_client_keeps_ordinary_tls_verification() {
        assert!(agent().config().https_only());
        let source = production_source();
        for forbidden in [
            "danger",
            "accept_invalid",
            "insecure",
            "tls_config",
            "add_root",
            "root_store",
        ] {
            assert!(!source.contains(forbidden), "{forbidden} must not appear");
        }
    }

    #[test]
    fn every_upstream_timeout_is_bounded() {
        let agent = agent();
        let timeouts = agent.config().timeouts();
        assert_eq!(timeouts.connect, Some(HANDSHAKE_TIMEOUT));
        assert_eq!(timeouts.recv_response, Some(RESPONSE_TIMEOUT));
        assert_eq!(timeouts.recv_body, Some(BODY_TIMEOUT));
        for bound in [HANDSHAKE_TIMEOUT, RESPONSE_TIMEOUT, BODY_TIMEOUT] {
            assert!(bound > Duration::ZERO, "{bound:?}");
            assert!(bound <= Duration::from_secs(3600), "{bound:?}");
        }
    }

    #[test]
    fn only_the_content_type_comes_back_to_codex() {
        assert_eq!(RELAYED_RESPONSE_HEADERS, ["content-type"]);
        for never in [
            "set-cookie",
            "content-length",
            "transfer-encoding",
            "content-encoding",
        ] {
            assert!(!RELAYED_RESPONSE_HEADERS.contains(&never), "{never}");
        }
    }

    #[test]
    fn the_upstream_leg_writes_nothing_and_needs_no_key() {
        let source = production_source();
        for forbidden in [
            "fs::write",
            "File::create",
            "OpenOptions",
            "capture_dir",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CODEX_API_KEY",
            "env_key",
            "HTTPS_PROXY",
            "HTTP_PROXY",
            "ALL_PROXY",
            "NODE_EXTRA_CA_CERTS",
            "SSL_CERT_FILE",
            "CURL_CA_BUNDLE",
            "rcgen",
            "hudsucker",
        ] {
            assert!(!source.contains(forbidden), "{forbidden} must not appear");
        }
    }
}
