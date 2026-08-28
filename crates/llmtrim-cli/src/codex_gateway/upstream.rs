//! The upstream leg: one fixed HTTPS destination, no proxy, no redirects.
//!
//! Kept apart from the policy module so the one place that opens a network connection is a
//! small file a reviewer can read in a minute. Every setting on the client below is a
//! security property rather than a tuning knob, and each one is pinned by a test.

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
