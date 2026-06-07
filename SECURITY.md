# Security Policy

## Reporting a vulnerability

Please report vulnerabilities **privately** through a [GitHub security advisory](https://github.com/fkiene/llmtrim/security/advisories/new) — not a public issue, which would disclose the flaw before a fix is available.

We aim to acknowledge within 72 hours and to ship a fix or mitigation as quickly as the
severity warrants. Please include a minimal reproduction and the affected version.

## Supported versions

llmtrim is pre-1.0; security fixes land on the latest release. Pin a version in
production and update promptly.

## What llmtrim touches

llmtrim sits between your tool and the LLM provider, so it sees sensitive data — review
this when threat-modeling your deployment:

- **API keys** are read from the environment (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` /
  `GEMINI_API_KEY`) and used to authenticate the upstream request. They are **never written
  to disk or logged**. (The `serve` proxy needs no key — it forwards the client's own auth.)
- **Prompt content** passes through the compressor and the `serve` proxy in memory. It is
  **not persisted**.
- **The savings ledger** (a local SQLite file) stores only aggregate metadata — token
  counts, provider, model id, tokenizer label. It does **not** store prompt content,
  responses, or keys.
- **The proxy binds to `127.0.0.1` only.** Do not expose it on a public interface; it
  performs no client authentication and forwards using the client's own auth header.
- **The local CA** (`~/.llmtrim/ca.pem`, key `0600`) is generated locally and **X.509
  name-constrained to LLM API domains** — it cannot mint a valid certificate for any other
  host, even if the key were stolen. It is trusted per-tool via `NODE_EXTRA_CA_CERTS`, not
  installed system-wide. Only the LLM API hosts are TLS-intercepted; all other HTTPS
  blind-tunnels untouched.

## Scope

In scope: the compression core, the interceptor (`serve`) + its CA, the CLI, and the `init`
config writers. Out of scope: vulnerabilities in upstream provider APIs.
