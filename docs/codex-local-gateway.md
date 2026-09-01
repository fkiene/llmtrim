# Codex local gateway

Run llmtrim in front of Codex without a proxy, without a local certificate authority and
without touching the machine's trust configuration. Codex is pointed at a loopback address
using its own native configuration, and llmtrim serves that address as an ordinary HTTP
endpoint.

```
Codex  ──HTTP──▶  127.0.0.1:43200  (llmtrim gateway)  ──HTTPS──▶  chatgpt.com
```

## Motivation

The interceptor is a capable, general solution: it can compress traffic from any client that
honours `HTTPS_PROXY`. The cost is a fair amount of machinery — a proxy variable exported into
the environment, a locally generated CA, leaf certificates minted per host, and
`NODE_EXTRA_CA_CERTS` / `SSL_CERT_FILE` / `CURL_CA_BUNDLE` wired up so clients trust it.

For Codex specifically, none of that is necessary. Codex already supports pointing at a custom
endpoint, and it keeps its ChatGPT session authentication when it does. A dedicated gateway is
therefore a much smaller thing to reason about: one route, one fixed destination, no
certificates and no environment-wide side effects.

## Why Codex accepts it

Codex supports a user-level `[model_providers.<name>]` block with a `base_url`. When that
provider defines no `env_key` and no `auth` block, it inherits the session's own auth manager:

- `model-provider/src/auth.rs` — `auth_manager_for_provider` returns the session auth manager
  when the provider carries no custom `auth`.
- `model-provider/src/auth.rs` — `auth_provider_from_auth` maps a ChatGPT auth snapshot to a
  bearer provider.
- `model-provider/src/bearer_auth_provider.rs` — that provider inserts
  `Authorization: Bearer <token>` and `ChatGPT-Account-ID`.

None of this depends on where `base_url` points, so Codex sends its normal,
ChatGPT-authenticated Responses request to the loopback address with the subscription
credentials attached. No API key is involved at any point.

(References are to the Codex Rust sources at the version this was checked against, 0.146.0.)

## Running it

```
llmtrim codex-gateway            # 127.0.0.1:43200
llmtrim codex-gateway --port 8123
```

The default is **43200**, deliberately not 43117. That one belongs to the interceptor, which
also scans the 64 ports above it when the default is busy — so a gateway default inside
`43117..=43181` would race the interceptor for a port on the same machine.

Foreground only. There is no daemon, no autostart, no background service and no supervision:
the gateway lives exactly as long as the process, and Ctrl-C stops accepting, ends the
connections still open and releases the port before returning.

`--port` is the whole configuration surface. The host is not a flag — it is always
`127.0.0.1`, and a non-loopback bind is refused rather than silently corrected. There is no
`--upstream`, no `--proxy`, no `--ca`, no `--api-key` and no `--capture`.

On startup it prints its address and the compression preset. It never prints a header, a
token, a prompt or a response; per request it prints two byte counts and a status.

## Configuration

In the **user-level** `~/.codex/config.toml`. Codex deliberately ignores `model_provider`,
`model_providers` and `*_base_url` when they come from a project-local `.codex/config.toml`,
and warns about it at startup — a sensible guard against a hostile repository.

```toml
[model_providers.llmtrim_local]
name = "LLMTrim Local"
base_url = "http://127.0.0.1:43200/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true

model_provider = "llmtrim_local"
```

Nothing here is secret. llmtrim does not write this file and does not read `auth.json`;
configuring Codex stays the user's decision.

Two details in that snippet are easy to get wrong:

- **`base_url` stops at `/backend-api/codex`.** Codex appends the operation itself, so
  `wire_api = "responses"` turns it into `POST /backend-api/codex/responses` — which is the
  one route the gateway serves. Writing the full path with `/responses` already on it yields
  `/backend-api/codex/responses/responses` and a `404` from the gateway.
- **`chatgpt_base_url` is not this setting.** That one points at the ChatGPT *login* host
  used during authentication. Pointing it at the gateway breaks sign-in and does not route a
  single turn through llmtrim.

### Provider name and first-party capabilities

Codex gates some first-party features on `provider.name` being exactly `"OpenAI"` —
`ModelProviderInfo::is_openai()` compares that string, not the endpoint or the credentials.
With any other name the turn works and the ChatGPT session is used normally, but the built-in
web-search tool is not offered to the model. Measured against Codex 0.146.0, that tool is
worth about 2.4k input tokens per turn; dropping it is a capability change, not a saving.

Naming the provider `"OpenAI"` restores it, and also lets Codex compress its request bodies
with zstd. This version does not decode that, so such a request is forwarded exactly as it
arrived — original body, matching `Content-Encoding` — and the turn completes normally; it
simply gets no llmtrim reduction. The per-request line says which happened.

To let the gateway see readable JSON and try the `safe` preset on it:

```toml
[features]
enable_request_compression = false
```

All three combinations were exercised against a live session and none of them costs you a
turn. What changes is how often llmtrim has anything to work with — and even when it does,
the reduction depends on the payload: structured JSON shrinks, prose and diffs may not shrink
at all.

## Security properties

| | |
|---|---|
| Listener | `127.0.0.1` only. A non-loopback bind is refused, not silently corrected. |
| Routes | `POST /backend-api/codex/responses` and nothing else. |
| Request framing | hyper parses the local request; nothing is framed by hand. |
| Upstream | `https://chatgpt.com` + that route, built from two constants. |
| SSRF | No input reaches the destination — no `?url=`, no `X-Upstream`, no `Host` routing. |
| Redirects | Not followed. A `3xx` is relayed as a status, never as a new destination. |
| Upstream proxy | Explicitly disabled, so no ambient `*_PROXY` variable can reroute the leg. |
| TLS | Ordinary certificate verification. No custom root, no pinning bypass, nothing installed. |
| Credentials | `Authorization` and `ChatGPT-Account-ID` relayed byte for byte, in memory. |
| Credential handling | Never parsed for meaning, never logged, never written to disk. |
| Auth gate | A well-formed bearer must be present, checked by shape only, before any body is read. |
| Request size | 64 MiB, the ceiling the rest of the product already uses. Over it: `413`, and no upstream request. |
| Compression | `llmtrim_core::compress_with_config`, `safe` preset. |
| Failure mode | Security gates fail closed. Compression fails **open**: a body llmtrim cannot transform is forwarded unchanged — same bytes, and the `Content-Encoding` that describes them — and the log line says so. |
| Response | Relayed incrementally, byte for byte and in order (SSE). Nothing is parsed or rewritten. |
| Memory | Bounded by a fixed relay queue, not by the length of the generation. |
| Timeouts | Connect 30 s, response head 120 s, response body 30 min. Nothing unbounded. |
| Retry | None. One attempt, one destination, no provider fallback. |
| Persistence | None. No capture directory, no ledger, no prompt or response written anywhere. |
| Keys | No API key, no paid API. |

The auth gate deserves a note: it exists so the loopback port is not an anonymous relay. The
gateway holds no credential of its own, so a caller that does not present a bearer cannot
cause an authenticated upstream request.

The relay deserves one too. The upstream leg runs on a blocking thread that hands back the
response head as soon as it arrives and then pushes the body into a fixed-depth queue. When
Codex reads more slowly than the upstream sends, that thread parks — so memory tracks the
queue, never the size of the answer, and the first token reaches Codex without waiting for
the last.

## Limitations

- **No separate gateway authentication.** The gateway has no credential of its own and
  persists none: `Authorization` and `ChatGPT-Account-ID` exist in memory for the length of one
  request and are never parsed, logged or written down. It requires a non-empty bearer before
  forwarding, but that check is on the header's shape — it is not authentication of the local
  process. Reaching the port does not by itself yield a valid ChatGPT credential, and a bearer
  that is merely well-formed is still rejected upstream.
- **The local hop is plaintext.** Credentials transit the loopback hop unencrypted. Binding to
  loopback reduces network exposure; it does not isolate the port from other processes on the
  same machine. Confirm the gateway started and bound the port you configured before pointing
  Codex at it: if another process already holds that port, Codex is talking to that process,
  not to the gateway.
- **Compression fails open.** A body llmtrim cannot transform is forwarded unchanged rather
  than refused: a failed optimisation should not cost the user their turn. It is not silent —
  the per-request line ends with `(compression failed; original body forwarded)`. The security
  gates around it stay fail-closed.
- **No zstd decoding.** Codex compresses request bodies when it considers the provider
  first-party. This gateway does not decode them, so the transform cannot run and the request
  is relayed exactly as it arrived: the turn works, llmtrim simply does nothing for it. The
  interceptor *does* decode zstd and gzip; to have those turns compressed here instead, use
  the `enable_request_compression = false` configuration above.
- **`llmtrim status` does not count these turns.** The gateway writes no ledger entry — it has
  no filesystem surface at all, which is a property the security-contract test enforces. Its
  only accounting is the two byte counts printed per request, and those are local estimates of
  request bytes, not billed usage.
- **The README's savings figures do not describe this path.** They come from the interceptor
  across mixed traffic. Here the preset is `safe` (lossless, input-only) and the measured
  effect depends entirely on payload shape: structured JSON pasted into a prompt shrinks;
  prose, diffs and logs with no exact repetition do not shrink at all. No output-token claim
  applies to this path.
- **Codex only.** One route, one upstream. This is not a general reverse proxy and is not
  meant to become one.
- **Smart App Control** still applies to a locally compiled, unsigned binary on Windows, and
  can block it from running. That is a property of the build, not of this gateway.

As a side benefit on Windows, the absence of a local CA and of machine-wide proxy variables
avoids the certificate and environment plumbing that modern application-control policies tend
to interact badly with.

## Testing

The policy layer is pure functions; the socket and upstream layers sit behind a streaming
seam. So the whole exchange is **validated with synthetic socket end-to-end tests**: a real
HTTP client, a real loopback socket, the product's real compression, and a scripted fake
upstream. No external host is contacted at any point.

Those tests are necessary but not sufficient, so the path is also **exercised against a live
ChatGPT-signed-in session**: `codex exec` posting to the loopback listener, llmtrim
compressing, the request reaching `https://chatgpt.com/backend-api/codex/responses`, and the
turn completing with the expected answer. Pinned to **Codex 0.146.0**
(`rust-v0.146.0`, `e363b08c9175ac1cbe5893615dd2cb9ddf95043b`); a different Codex version can
change what it sends and is a different claim.

The suite covers the bind policy and the refusal of any non-loopback host, the route and
method allowlist, the auth gate and the fact that a refused request never reaches the upstream
leg, the request-size ceiling, real compression through `llmtrim_core` with the task's
identifier surviving it, fail-closed behaviour, SSE ordering, incremental relay (the upstream
withholds its second chunk until the client confirms the first one arrived), graceful shutdown
with the port released and no listener left behind, and the static contract on the production
upstream client: fixed URL, no proxy, no redirects, standard TLS verification and bounded
timeouts. Fictitious sentinels are asserted absent from every error and every metric.

CI runs the suite on `windows-latest` and `ubuntu-latest`, plus a security-contract job that
fails if any file in the gateway module tree ever gains a proxy variable, a CA dependency, a
capture path, a filesystem write or an API key.
