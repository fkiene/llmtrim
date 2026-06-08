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
pub fn run(_port: u16) -> anyhow::Result<()> {
    anyhow::bail!("this build has no interceptor; rebuild with `--features intercept`")
}

#[cfg(not(feature = "intercept"))]
pub fn run_supervised(_port: u16) -> anyhow::Result<()> {
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
    use std::sync::mpsc::Sender;
    use std::sync::{Arc, Mutex};

    use anyhow::{Context, Result};
    use bytes::Bytes;
    use http_body_util::{BodyExt, BodyStream, Full};
    use hudsucker::certificate_authority::RcgenAuthority;
    use hudsucker::hyper::{Method, Request, Response, header};
    use hudsucker::rustls::crypto::aws_lc_rs;
    use hudsucker::{Body, HttpContext, HttpHandler, Proxy, RequestOrResponse};

    use crate::config::DenseConfig;
    use crate::ir::ProviderKind;
    use crate::tracking::{Record, Tracker};

    /// Parent domains of every endpoint in the `llm_providers` registry — the maintained
    /// upstream source of truth. The CA is name-constrained to these, and only hosts under
    /// them are intercepted. Computed once.
    static LLM_DOMAINS: once_cell::sync::Lazy<std::collections::HashSet<String>> =
        once_cell::sync::Lazy::new(|| {
            let mut set = std::collections::HashSet::new();
            for provider in llm_providers::get_providers_data().values() {
                for endpoint in provider.endpoints.values() {
                    if let Some(host) = host_of_url(endpoint.base_url) {
                        set.insert(parent_domain(host));
                    }
                }
            }
            set
        });

    /// Reputable direct LLM API hosts beyond the `llm_providers` registry that speak one of
    /// the three wire shapes we adapt (OpenAI / Anthropic / Gemini). Interception + CA coverage
    /// only — the wire format is detected from the body, never assumed from the host. Kept small
    /// on purpose: each entry widens what the name-constrained MITM CA may forge.
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
    ];

    /// True if `host` is one of `EXTRA_HOSTS` (exact, or a subdomain of one).
    fn is_extra_host(host: &str) -> bool {
        EXTRA_HOSTS
            .iter()
            .any(|s| host == *s || host.ends_with(&format!(".{s}")))
    }

    /// Host component of a URL like `https://api.openai.com/v1` → `api.openai.com`.
    fn host_of_url(url: &str) -> Option<&str> {
        let after = url.split_once("://").map_or(url, |(_, rest)| rest);
        let host = after.split(['/', '?']).next()?.split(':').next()?;
        (!host.is_empty()).then_some(host)
    }

    /// The registrable parent domain (last two labels): `ark.cn-beijing.volces.com` →
    /// `volces.com`, `api.openai.com` → `openai.com`.
    fn parent_domain(host: &str) -> String {
        let mut it = host.rsplit('.');
        match (it.next(), it.next()) {
            (Some(tld), Some(sld)) => format!("{sld}.{tld}"),
            _ => host.to_string(),
        }
    }

    /// The provider wire shape for a host, or `None` if it isn't an LLM API host (so it is
    /// not intercepted). Anthropic and Google have their own shapes; every other registry
    /// provider speaks the OpenAI `/v1/chat/completions` shape.
    fn provider_for_host(host: &str) -> Option<ProviderKind> {
        let h = host.to_ascii_lowercase();
        if h.ends_with("anthropic.com") {
            return Some(ProviderKind::Anthropic);
        }
        if h.ends_with("generativelanguage.googleapis.com")
            || h.ends_with("aiplatform.googleapis.com")
        {
            return Some(ProviderKind::Google);
        }
        // Registry domains and the curated extra hosts default to the OpenAI shape; the exact
        // adapter is refined from the body at compress time (`provider::detect`), so a host that
        // serves a Claude- or Gemini-shaped body is still handled correctly.
        if LLM_DOMAINS.contains(&parent_domain(&h)) || is_extra_host(&h) {
            return Some(ProviderKind::OpenAi);
        }
        None
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
    }

    /// A verbatim copy of the client's request, for replay-on-error.
    #[derive(Clone)]
    struct OriginalRequest {
        url: String,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    }

    /// Client request headers to forward on replay — everything except `host` and
    /// `content-length` (the HTTP client sets those from the URL and body).
    fn forward_headers(headers: &header::HeaderMap) -> Vec<(String, String)> {
        headers
            .iter()
            .filter(|(n, _)| !matches!(n.as_str(), "host" | "content-length"))
            .filter_map(|(n, v)| v.to_str().ok().map(|v| (n.to_string(), v.to_string())))
            .collect()
    }

    /// Replay the original (uncompressed) request to the upstream — direct, all statuses
    /// relayed — and build a response for the client. `None` if the replay itself fails (in
    /// which case the caller keeps the compressed response's error).
    fn replay_original(orig: &OriginalRequest) -> Option<Response<Body>> {
        use std::io::Read;
        let body = std::str::from_utf8(&orig.body).ok()?;
        let mut up = crate::transport::forward_post(&orig.url, &orig.headers, body).ok()?;
        let mut buf = Vec::new();
        up.reader.read_to_end(&mut buf).ok()?;
        let mut builder = Response::builder().status(up.status);
        if let Some(ct) = up.content_type {
            builder = builder.header(header::CONTENT_TYPE, ct);
        }
        builder.body(Body::from(Full::new(Bytes::from(buf)))).ok()
    }

    /// Does the plan carry a reversible *output* transform? No output-side transform ships
    /// today (Stage D is input-only; DSS was removed), so nothing is reversed.
    fn plan_reverses_output(_plan: &str) -> bool {
        false
    }

    fn record_from(p: Pending, output_after: Option<i64>, cache_read: Option<i64>) -> Record {
        Record {
            provider: p.provider.as_str().to_string(),
            model: p.model,
            tokenizer: p.tokenizer,
            exact: p.exact,
            input_before: p.input_before,
            input_after: p.input_after,
            output_before: None,
            output_after,
            compress_micros: Some(p.compress_micros),
            cache_read_tokens: cache_read,
        }
    }

    /// Expand the model's shorthand answer back to normal output using the request's plan,
    /// rewrite it into the response JSON, and record the (shorthand-billed) output tokens.
    /// Returns the rewritten body, or the original on any parse failure.
    fn rehydrate_response(bytes: &[u8], p: &Pending, ledger: &Sender<Record>) -> Vec<u8> {
        let original = bytes.to_vec();
        let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            let _ = ledger.send(record_from(p.clone(), None, None));
            return original;
        };
        let answer = crate::provider::for_kind(p.provider).answer_text(&json);
        // The model billed the shorthand it emitted — count that for spend.
        let out_tok = answer.as_ref().and_then(|a| {
            crate::tokenizer::counter_for(p.provider, p.model.as_deref())
                .ok()
                .map(|c| c.count(a) as i64)
        });
        if let Some(answer) = answer
            && let Ok(expanded) = crate::rehydrate(&answer, &p.plan)
        {
            set_answer(&mut json, p.provider, &expanded);
        }
        let _ = ledger.send(record_from(p.clone(), out_tok, None));
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
        /// The compressed request awaiting its response (set in `handle_request`).
        pending: Option<Pending>,
    }

    impl Drop for Interceptor {
        fn drop(&mut self) {
            // A compressed request whose response we never saw (connection dropped): still
            // record the input savings, with output unknown.
            if let Some(p) = self.pending.take() {
                let _ = self.ledger.send(record_from(p, None, None));
            }
        }
    }

    impl HttpHandler for Interceptor {
        /// Only MITM (forge a cert for) the LLM provider hosts; everything else is
        /// blind-tunneled, so the CA is never used outside its purpose.
        async fn should_intercept(&mut self, _ctx: &HttpContext, req: &Request<Body>) -> bool {
            host_of(req)
                .map(|h| provider_for_host(&h).is_some())
                .unwrap_or(false)
        }

        async fn handle_request(
            &mut self,
            _ctx: &HttpContext,
            req: Request<Body>,
        ) -> RequestOrResponse {
            let host = host_of(&req);
            let provider = host.as_deref().and_then(provider_for_host);
            // Only compress POST bodies to a known provider; pass everything else through.
            let Some(provider) = provider.filter(|_| req.method() == Method::POST) else {
                return req.into();
            };
            // Body-signed requests (AWS SigV4) must pass through untouched — changing the
            // body would invalidate the signature and the provider would reject it.
            if is_body_signed(req.headers()) {
                return req.into();
            }
            let (parts, body) = req.into_parts();
            let bytes = match body.collect().await {
                Ok(c) => c.to_bytes(),
                Err(_) => return Request::from_parts(parts, Body::empty()).into(),
            };
            // Run the CPU-bound compression on the blocking pool so a burst of large requests
            // can't monopolize the async workers (which would stall response streaming for
            // everyone). Cheap to move in: `Bytes` is ref-counted, `config` is an `Arc`.
            let config = self.config.clone();
            let body_for_compress = bytes.clone();
            let compressed = tokio::task::spawn_blocking(move || {
                compress_blocking(&config, &body_for_compress, provider)
            })
            .await
            .ok()
            .flatten();

            let new_body = match compressed {
                Some((json, mut pending)) => {
                    // We changed the body — remember the original so we can replay it verbatim
                    // if the upstream rejects our compressed version (4xx/5xx). Compression must
                    // never break the user's call.
                    if let Some(host) = host.as_deref() {
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

        /// Tee the response: forward it to the client unchanged while accumulating a copy,
        /// and once it finishes streaming, measure the output tokens and complete the ledger
        /// record. Non-compressed requests (no `pending`) pass straight through.
        async fn handle_response(
            &mut self,
            _ctx: &HttpContext,
            res: Response<Body>,
        ) -> Response<Body> {
            let Some(pending) = self.pending.take() else {
                return res;
            };
            // Safety net: if the upstream rejected our compressed request (4xx/5xx), replay
            // the original verbatim and hand the client THAT — compression must never break
            // the call. Only when we actually changed the body (original captured).
            let status = res.status();
            if (status.is_client_error() || status.is_server_error())
                && let Some(original) = pending.original.clone()
                && let Ok(Some(replayed)) =
                    tokio::task::spawn_blocking(move || replay_original(&original)).await
            {
                eprintln!(
                    "llmtrim: upstream {} on compressed request — replayed original (no compression this call)",
                    status.as_u16()
                );
                // Record the fall-back honestly: the original was sent, so zero savings.
                let mut rec = record_from(pending, None, None);
                rec.input_after = rec.input_before;
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

    /// The CPU-bound compression, run on the blocking pool (see `handle_request`). Pure: takes
    /// the request body + config, returns the compressed JSON paired with the per-request
    /// `Pending` (its `original` left unset — the caller fills it). `None` to forward verbatim
    /// (not UTF-8/JSON, errored, or no net token win).
    fn compress_blocking(
        config: &DenseConfig,
        body: &[u8],
        provider: ProviderKind,
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
            .and_then(|v| crate::provider::detect(&v))
            .unwrap_or(provider);
        let started = std::time::Instant::now();
        let result = crate::compress_with_config(text, Some(kind), config).ok()?;
        let compress_micros = started.elapsed().as_micros() as i64;
        // Never forward a request larger than we received. On tiny or non-chat bodies (e.g.
        // token-count / auxiliary calls) the input-side stages can't offset the output-control
        // instruction's fixed cost, so the compressed form is a net token *increase*. Forward
        // the original verbatim and record nothing — the "never a bigger bill" guarantee, and
        // keeps non-wins out of the savings ledger.
        if result.input_tokens_after >= result.input_tokens_before {
            return None;
        }
        let pending = Pending {
            provider,
            model: result.model.clone(),
            tokenizer: result.tokenizer_label.clone(),
            exact: result.tokenizer_exact,
            input_before: result.input_tokens_before.0 as i64,
            input_after: result.input_tokens_after.0 as i64,
            compress_micros,
            plan: serde_json::to_string(&result.plan).unwrap_or_default(),
            original: None,
        };
        Some((result.request_json, pending))
    }

    /// Owns the accumulated response bytes; on drop (stream complete/aborted) it measures
    /// the output tokens and writes the completed ledger record.
    struct Finalize {
        acc: Arc<Mutex<Vec<u8>>>,
        pending: Option<Pending>,
        ledger: Sender<Record>,
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
                    crate::tokenizer::counter_for(p.provider, p.model.as_deref())
                        .ok()
                        .map(|c| c.count(&text) as i64)
                })
            });
            // Cached-prefix tokens the provider served from its prompt cache (the discounted
            // resent context); `None` when the provider reports none.
            let cache_read = extract_cache_read(p.provider, &buf);
            let _ = self.ledger.send(record_from(p, output_after, cache_read));
        }
    }

    /// The model's answer text from a captured response body — a non-streaming JSON answer,
    /// or the concatenated text deltas of an SSE stream.
    fn extract_output_text(provider: ProviderKind, body: &[u8]) -> Option<String> {
        let text = std::str::from_utf8(body).ok()?;
        if text.trim_start().starts_with('{')
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(text.trim())
        {
            return crate::provider::for_kind(provider).answer_text(&value);
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
            if let Ok(v) = serde_json::from_str::<Value>(data)
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
            if let Ok(v) = serde_json::from_str::<Value>(data)
                && let Some(n) = from_value(&v)
            {
                best = Some(best.map_or(n, |b| b.max(n)));
            }
        }
        best
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

    /// Run the interceptor on `127.0.0.1:port`, blocking until Ctrl-C. Sets up its own
    /// Tokio runtime so the rest of the CLI stays synchronous.
    pub fn run(port: u16) -> Result<()> {
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
    pub fn run_supervised(port: u16) -> Result<()> {
        use std::time::{Duration, Instant};
        let mut fast_fails = 0u32;
        loop {
            let started = Instant::now();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(port)));
            match outcome {
                Ok(Ok(())) => return Ok(()), // graceful shutdown (Ctrl-C)
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
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    /// Build the lazy tokenizer BPE tables (the dominant one-time cost) + prime the stage code
    /// paths and the first tree-sitter grammar, before the proxy starts serving.
    fn warm_up(config: &DenseConfig) {
        use crate::ir::ProviderKind;
        for (kind, model) in [
            (ProviderKind::OpenAi, Some("gpt-4o")), // o200k_base
            (ProviderKind::OpenAi, Some("gpt-4")),  // cl100k_base
            (ProviderKind::Anthropic, None),        // approximate counter
        ] {
            if let Ok(c) = crate::tokenizer::counter_for(kind, model) {
                let _ = c.count("warm up the tokenizer table");
            }
        }
        // One real compression primes the stage code paths + the first grammar load.
        let sample = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"function f(){ return 1; }"}]}"#;
        let _ = crate::compress_with_config(sample, Some(ProviderKind::OpenAi), config);
    }

    async fn run_async(port: u16) -> Result<()> {
        let _ = aws_lc_rs::default_provider().install_default();

        let (cert_pem, key_pem) = ensure_ca()?;
        let key = hudsucker::rcgen::KeyPair::from_pem(&key_pem)
            .map_err(|e| anyhow::anyhow!("failed to parse CA key: {e}"))?;
        let issuer = hudsucker::rcgen::Issuer::from_ca_cert_pem(&cert_pem, key)
            .map_err(|e| anyhow::anyhow!("failed to parse CA cert: {e}"))?;
        let ca = RcgenAuthority::new(issuer, 1_000, aws_lc_rs::default_provider());

        // Ledger writes go to a dedicated thread (rusqlite isn't async); the handler just
        // sends Records over the channel.
        let (tx, rx) = std::sync::mpsc::channel::<Record>();
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

        let handler = Interceptor {
            config: Arc::new(DenseConfig::load_for_interceptor()),
            ledger: tx,
            pending: None,
        };

        let addr = SocketAddr::from(([127, 0, 0, 1], port));
        // Pre-flight bind: hudsucker's `Proxy::start` collapses a bind failure to a bare
        // "io error", hiding whether the port is in use or OS-reserved. Bind once ourselves
        // first so the real cause reaches the log, then drop it microseconds before hudsucker
        // rebinds the same addr.
        drop(
            std::net::TcpListener::bind(addr)
                .with_context(|| format!("cannot bind {addr} — port in use or reserved"))?,
        );
        eprintln!("llmtrim: MITM interceptor on http://127.0.0.1:{port}");
        eprintln!("  export HTTPS_PROXY=http://127.0.0.1:{port}");
        eprintln!(
            "  trust the CA: NODE_EXTRA_CA_CERTS={}",
            ca_cert_path()?.display()
        );

        // Prime the lazy tokenizer tables + stage machinery before serving, so the first real
        // request runs at steady state instead of paying ~150 ms of one-time init (which slowed
        // the first call and skewed the latency metric). One-time, at boot, off the request path.
        warm_up(&handler.config);

        let proxy = Proxy::builder()
            .with_addr(addr)
            .with_ca(ca)
            .with_rustls_connector(aws_lc_rs::default_provider())
            .with_http_handler(handler)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build proxy: {e}"))?;
        proxy
            .start()
            .await
            .map_err(|e| anyhow::anyhow!("proxy error: {e}"))?;
        Ok(())
    }

    pub fn ca_cert_path() -> Result<PathBuf> {
        Ok(crate::daemon::home_dir()?.join("ca.pem"))
    }

    fn ca_key_path() -> Result<PathBuf> {
        Ok(crate::daemon::home_dir()?.join("ca.key"))
    }

    /// Load the local CA, generating a name-constrained root CA on first use. Returns
    /// `(cert_pem, key_pem)`.
    pub fn ensure_ca() -> Result<(String, String)> {
        let (cert_path, key_path) = (ca_cert_path()?, ca_key_path()?);
        if cert_path.exists() && key_path.exists() {
            return Ok((
                std::fs::read_to_string(&cert_path)?,
                std::fs::read_to_string(&key_path)?,
            ));
        }
        let (cert_pem, key_pem) = generate_ca()?;
        let dir = crate::daemon::home_dir()?;
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
        std::fs::write(&cert_path, &cert_pem)?;
        std::fs::write(&key_path, &key_pem)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600));
        }
        Ok((cert_pem, key_pem))
    }

    /// Generate a root CA whose signing power is name-constrained to the LLM API domains.
    fn generate_ca() -> Result<(String, String)> {
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
            permitted_subtrees: LLM_DOMAINS
                .iter()
                .map(|d| GeneralSubtree::DnsName(d.clone()))
                .chain(
                    EXTRA_HOSTS
                        .iter()
                        .map(|h| GeneralSubtree::DnsName((*h).to_string())),
                )
                .collect(),
            excluded_subtrees: vec![],
        });
        let cert = params
            .self_signed(&key)
            .map_err(|e| anyhow::anyhow!("CA self-sign failed: {e}"))?;
        Ok((cert.pem(), key.serialize_pem()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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
                crate::provider::detect(&claude_body),
                Some(ProviderKind::Anthropic)
            );
        }

        #[test]
        fn generated_ca_is_a_parseable_constrained_ca() {
            let (cert_pem, key_pem) = generate_ca().unwrap();
            assert!(cert_pem.contains("BEGIN CERTIFICATE"));
            assert!(key_pem.contains("PRIVATE KEY"));
            // Round-trips through the same parser hudsucker uses.
            let key = hudsucker::rcgen::KeyPair::from_pem(&key_pem).unwrap();
            assert!(hudsucker::rcgen::Issuer::from_ca_cert_pem(&cert_pem, key).is_ok());
        }
    }
}
