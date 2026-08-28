# Codex local gateway

Run llmtrim in front of Codex without a proxy, without a local certificate authority and
without touching the machine's trust configuration. Codex is pointed at a loopback address
using its own native configuration, and llmtrim serves that address as an ordinary HTTP
endpoint.

```
Codex  ──HTTP──▶  127.0.0.1:PORT  (llmtrim gateway)  ──HTTPS──▶  chatgpt.com
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

Nothing here is secret. llmtrim does not write this file; configuring Codex stays the user's
decision.

## Security properties

| | |
|---|---|
| Listener | `127.0.0.1` only. A non-loopback bind is refused, not silently corrected. |
| Routes | `POST /backend-api/codex/responses` and nothing else. |
| Upstream | `https://chatgpt.com` + that route, built from two constants. |
| SSRF | No input reaches the destination — no `?url=`, no `X-Upstream`, no `Host` routing. |
| Credentials | `Authorization` and `ChatGPT-Account-ID` relayed byte for byte, in memory. |
| Credential handling | Never parsed for meaning, never logged, never written to disk. |
| Auth gate | A well-formed bearer must be present, checked by shape only. |
| Compression | `llmtrim_core::compress_with_config`, `safe` preset. |
| Failure mode | Fail closed: a body that cannot be compressed is refused, not forwarded. |
| Response | Status, headers and chunks relayed unchanged and in order (SSE). |
| Persistence | None. No capture directory, no prompt or response written anywhere. |
| Keys | No API key, no paid API, no provider fallback. |

The auth gate deserves a note: it exists so the loopback port is not an anonymous relay. The
gateway holds no credential of its own, so a caller that does not present a bearer cannot
cause an authenticated upstream request.

## Limitations

- **No gateway authentication.** Any process on the machine can reach the port. The mitigation
  is structural rather than added: the gateway stores no credential.
- **The local hop is plaintext.** That is why it is loopback-only.
- **Fail-closed compression** differs from the interceptor, which forwards the original body
  when compression does not help. Here a failure surfaces, so a session is never silently
  running without llmtrim.
- **Codex only.** This is not a general reverse proxy and is not meant to become one.

As a side benefit on Windows, the absence of a local CA and of machine-wide proxy variables
avoids the certificate and environment plumbing that modern application-control policies tend
to interact badly with.

## Testing

Everything in the module is either a pure function or sits behind an upstream seam, so the
pipeline is exercised offline with a fake upstream — no socket, no external host. The suite
covers the bind policy, the route allowlist, the fixed upstream, the absence of any
client-controlled destination, the auth gate, credential passthrough, real compression through
`llmtrim_core`, fail-closed behaviour, SSE ordering, and redaction: fictitious sentinels are
asserted absent from every error and every metric.

CI runs the suite on `windows-latest` and `ubuntu-latest`, plus a security-contract job that
fails if the gateway module ever gains a proxy variable, a CA dependency, a capture path or an
API key.
