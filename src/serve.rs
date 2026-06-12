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
    use crate::memo::Memo;
    use crate::tracking::{Record, Tracker};

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
    ];

    /// True if `host` is one of `EXTRA_HOSTS` (exact, or a subdomain of one).
    fn is_extra_host(host: &str) -> bool {
        EXTRA_HOSTS
            .iter()
            .any(|s| host == *s || host.ends_with(&format!(".{s}")))
    }

    /// Every name-constraint domain the CA should permit: the registry parent domains plus the
    /// curated extra hosts, sorted + deduped. The single source of truth for what we intend to
    /// intercept; compared byte-for-byte against the CA's sidecar to decide whether to regenerate.
    fn intercept_domains() -> Vec<String> {
        let mut v: Vec<String> = LLM_DOMAINS
            .iter()
            .cloned()
            .chain(EXTRA_HOSTS.iter().map(|h| (*h).to_string()))
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

    /// Expand the model's shorthand answer back to normal output using the request's plan,
    /// rewrite it into the response JSON, and record the (shorthand-billed) output tokens.
    /// Returns the rewritten body, or the original on any parse failure.
    fn rehydrate_response(bytes: &[u8], p: &Pending, ledger: &Sender<Record>) -> Vec<u8> {
        let original = bytes.to_vec();
        let Ok(mut json) = serde_json::from_slice::<serde_json::Value>(bytes) else {
            let _ = ledger.send(record_from(p.clone(), ResponseUsage::default()));
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
        /// Domains the live CA covers (its sidecar set). We intercept only hosts under these, so
        /// a stale CA — one generated before a host was added — blind-tunnels the new host rather
        /// than forging a cert its own name-constraints reject. Degrade safely, never break.
        domains: Arc<std::collections::HashSet<String>>,
        /// Turn-stability memo (see [`crate::memo`]), shared across the per-request handler
        /// clones so a conversation's earlier-turn compressed prefix is reusable on its next
        /// turn — keeping the provider prefix cache warm on agent loops. In-memory only.
        memo: Arc<Memo>,
        /// The compressed request awaiting its response (set in `handle_request`).
        pending: Option<Pending>,
    }

    impl Drop for Interceptor {
        fn drop(&mut self) {
            // A compressed request whose response we never saw (connection dropped): still
            // record the input savings, with output unknown.
            if let Some(p) = self.pending.take() {
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
            self.handle_request_inner(req).await
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
            let host = host_of(&req);
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
            // Only compress POST bodies to a known provider; pass everything else through.
            let Some(provider) = provider.filter(|_| req.method() == Method::POST) else {
                return req.into();
            };
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
            let compressed = tokio::task::spawn_blocking(move || {
                compress_blocking(&config, &body_for_compress, provider, &memo)
            })
            .await
            .ok()
            .flatten();

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

        /// Body of `handle_response`, factored out so tests can call it directly without
        /// a `hudsucker::HttpContext`.
        async fn handle_response_inner(&mut self, res: Response<Body>) -> Response<Body> {
            let Some(pending) = self.pending.take() else {
                return res;
            };
            // Safety net: if the upstream rejected our compressed request with a *validation*
            // error (400 Bad Request / 422 Unprocessable) — the failure class our body edits can
            // cause — replay the original verbatim and hand the client THAT. We deliberately do
            // NOT replay 429/5xx (rate-limit, overload, server faults): those are unrelated to
            // our mutation, and re-sending would double provider load during an incident and
            // strip the client's `retry-after`. Only when we actually changed the body.
            let status = res.status();
            if should_replay(status.as_u16())
                && let Some(original) = pending.original.clone()
                && let Ok(Some(replayed)) =
                    tokio::task::spawn_blocking(move || replay_original(&original)).await
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
    fn compress_blocking(
        config: &DenseConfig,
        body: &[u8],
        provider: ProviderKind,
        memo: &Memo,
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
        let mut result = crate::compress_with_config(text, Some(kind), config).ok()?;
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
        };
        capture_pair(text, &result.request_json, &pending);
        Some((result.request_json, pending))
    }

    /// Opt-in QA capture: when `LLMTRIM_CAPTURE_DIR` is set, write the before/after request
    /// bodies of each compressed request as one JSON file so an external reviewer can audit
    /// compression quality. Off (zero work beyond one env read) unless the var is set; any
    /// write failure is logged and swallowed — capture must never break proxying.
    fn capture_pair(before: &str, after: &str, pending: &Pending) {
        let Ok(dir) = std::env::var("LLMTRIM_CAPTURE_DIR") else {
            return;
        };
        if dir.is_empty() {
            return;
        }
        let record = serde_json::json!({
            "ts": chrono::Utc::now().to_rfc3339(),
            "provider": pending.provider.as_str(),
            "model": pending.model,
            "input_before": pending.input_before,
            "input_after": pending.input_after,
            "output_shaped": pending.output_shaped,
            "plan": pending.plan,
            "before": before,
            "after": after,
        });
        let name = format!(
            "{}-{:x}.json",
            chrono::Utc::now().timestamp_micros(),
            std::process::id()
        );
        let path = std::path::Path::new(&dir).join(name);
        if let Err(e) =
            std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, record.to_string()))
        {
            eprintln!("llmtrim: capture failed ({}): {e}", path.display());
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
        result: &mut crate::CompressResult,
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
        let reused = crate::memo::apply(memo, salt.as_bytes(), &original, &mut compressed);
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
        if let Ok(counter) = crate::tokenizer::counter_for(kind, result.model.as_deref()) {
            let adapter = crate::provider::for_kind(kind);
            let req = crate::ir::Request::from_value(kind, compressed);
            let after = crate::pipeline::content_tokens(&req, adapter.as_ref(), counter.as_ref());
            result.input_tokens_after = crate::tokenizer::Tokens(after);
        }
        result.request_json = rewritten;
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
            // Full-rate + cache-write input tokens — with `cache_read` these reconstruct the
            // request's real input bill, which the dashboard's net-$ figures are priced from.
            let (fresh_input, cache_write) = extract_input_usage(p.provider, &buf);
            let _ = self.ledger.send(record_from(
                p,
                ResponseUsage {
                    output_after,
                    cache_read,
                    fresh_input,
                    cache_write,
                },
            ));
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
            if let Ok(v) = serde_json::from_str::<Value>(data) {
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
            crate::daemon::bump_restarts(); // surfaces as a `status` warning
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
            // `ensure_ca` above just reconciled the CA + sidecar, so this is exactly what the
            // live CA can sign for.
            domains: Arc::new(covered_domains()),
            // One process-wide turn-stability memo, shared across the per-request handler clones.
            memo: Arc::new(Memo::with_capacity(crate::memo::DEFAULT_CAPACITY)),
            pending: None,
        };

        // Loopback by default — a MITM proxy must not be reachable off-host unless asked.
        // LLMTRIM_BIND=0.0.0.0 opts in (containers: port mapping can't reach loopback).
        let bind_ip: std::net::IpAddr = match std::env::var("LLMTRIM_BIND") {
            Ok(s) => s
                .parse()
                .with_context(|| format!("LLMTRIM_BIND is not a valid IP address: {s}"))?,
            Err(_) => std::net::IpAddr::from([127, 0, 0, 1]),
        };
        let addr = SocketAddr::from((bind_ip, port));
        // Pre-flight bind: hudsucker's `Proxy::start` collapses a bind failure to a bare
        // "io error", hiding whether the port is in use or OS-reserved. Bind once ourselves
        // first so the real cause reaches the log, then drop it microseconds before hudsucker
        // rebinds the same addr.
        drop(
            std::net::TcpListener::bind(addr)
                .with_context(|| format!("cannot bind {addr} — port in use or reserved"))?,
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
        fn intercept_domains_include_registry_and_extras_sorted() {
            let d = intercept_domains();
            assert!(d.contains(&"api.openai.com".to_string())); // exact registry endpoint host
            assert!(d.contains(&"opencode.ai".to_string())); // from EXTRA_HOSTS
            assert!(
                d.windows(2).all(|w| w[0] <= w[1]),
                "must be sorted for sidecar compare"
            );
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
            use crate::config::DenseConfig;
            use crate::ir::ProviderKind;
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(crate::memo::DEFAULT_CAPACITY);
            assert!(compress_blocking(&cfg, b"not json", ProviderKind::OpenAi, &memo).is_none());
            assert!(compress_blocking(&cfg, b"", ProviderKind::OpenAi, &memo).is_none());
        }

        #[test]
        fn compress_blocking_net_win_guard() {
            use crate::config::DenseConfig;
            use crate::ir::ProviderKind;
            // A tiny request: compression overhead exceeds savings → gate fires → None.
            let body = r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#;
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(crate::memo::DEFAULT_CAPACITY);
            // May return None (net loss) or Some (net win); both are valid.
            // The important assertion: when Some, the compressed form must have fewer input tokens.
            if let Some((compressed, pending)) =
                compress_blocking(&cfg, body.as_bytes(), ProviderKind::OpenAi, &memo)
            {
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
            use crate::config::DenseConfig;
            use crate::ir::ProviderKind;
            // 50 identical log lines: dedup should fire and compression should win.
            let lines = "ERROR database connection pool exhausted, retrying in 5s\n".repeat(50);
            let body = serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": lines}]
            })
            .to_string();
            let cfg = DenseConfig::default();
            let memo = Memo::with_capacity(crate::memo::DEFAULT_CAPACITY);
            // Dedup is default-on and the spam is contiguous: this must compress, so a
            // `None` here is a regression (the test must not pass vacuously).
            let (compressed, pending) =
                compress_blocking(&cfg, body.as_bytes(), ProviderKind::OpenAi, &memo)
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

        // ── M0-2: replay-on-error gate ──────────────────────────────────────

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

        // ── M0-2: compress_blocking actual-compression guarantee ─────────────

        #[test]
        fn compress_blocking_actual_compression_with_large_body() {
            use crate::config::DenseConfig;
            use crate::ir::ProviderKind;
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
            let memo = Memo::with_capacity(crate::memo::DEFAULT_CAPACITY);
            if let Some((compressed_json, pending)) =
                compress_blocking(&cfg, body.as_bytes(), ProviderKind::OpenAi, &memo)
            {
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
            let handler = Interceptor {
                config: Arc::new(crate::config::DenseConfig::default()),
                ledger: tx,
                domains: Arc::new(intercept_domains().into_iter().collect()),
                memo: Arc::new(Memo::with_capacity(crate::memo::DEFAULT_CAPACITY)),
                pending: None,
            };
            (handler, rx)
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
    }
}
