//! Transport — send the compressed request to the LLM API and return the response.
//!
//! Blocking `ureq` (no async — RTK ethos; a request/response round-trip needs no
//! concurrency). The pure parts (`url`, `headers`) are unit-tested; the live
//! `send` needs an API key + network, so it isn't exercised in the test suite.

use anyhow::{Context, Result};

use crate::ir::ProviderKind;

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
    pub fn send(&self, request_json: &str) -> Result<String> {
        let url = self.url();
        let mut request = ureq::post(&url);
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
pub fn forward_post(url: &str, headers: &[(String, String)], body: &str) -> Result<Upstream> {
    let mut req = ureq::post(url).config().http_status_as_error(false).build();
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
pub fn forward_get(url: &str, headers: &[(String, String)]) -> Result<Upstream> {
    let mut req = ureq::get(url).config().http_status_as_error(false).build();
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
}
