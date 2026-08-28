# Codex local gateway

Run llmtrim in front of Codex without a proxy, without a local certificate authority and
without touching the machine's trust configuration. Codex is pointed at a loopback address
using its own native configuration, and llmtrim serves that address as an ordinary HTTP
endpoint.

```
Codex  ──HTTP──▶  127.0.0.1:43117  (llmtrim gateway)  ──HTTPS──▶  chatgpt.com
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
llmtrim codex-gateway            # 127.0.0.1:43117
llmtrim codex-gateway --port 8123
```

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
base_url = "http://127.0.0.1:43117/backend-api/codex"
wire_api = "responses"
requires_openai_auth = true

model_provider = "llmtrim_local"
```

Nothing here is secret. llmtrim does not write this file and does not read `auth.json`;
configuring Codex stays the user's decision.

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
| Failure mode | Fail closed: a body that cannot be compressed is refused, not forwarded. |
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

- **No gateway authentication.** Any process on the machine can reach the port. The mitigation
  is structural rather than added: the gateway stores no credential, and it refuses a request
  that does not carry the caller's own bearer.
- **The local hop is plaintext.** That is why it is loopback-only.
- **Fail-closed compression** differs from the interceptor, which forwards the original body
  when compression does not help. Here a failure surfaces, so a session is never silently
  running without llmtrim.
- **Codex only.** One route, one upstream. This is not a general reverse proxy and is not
  meant to become one.
- **Not yet exercised against a live Codex session.** Everything below is validated with a
  synthetic socket end-to-end test; the first real Codex run has not happened.
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
