# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **MITM HTTPS interceptor** (`serve`): a man-in-the-middle proxy (hudsucker) that compresses
  every tool's LLM API calls in flight, with **streaming (SSE) pass-through**. Provider host
  allowlist + name-constrained CA derived from the `llm_providers` registry. `llmtrim ca`
  manages the local CA.
- **Interceptor preset tuned on real agent traffic** (A/B'd through `claude -p`): the live
  default is `hygiene` + exact `dedup` + **tool-description trimming (300 chars)**, which cuts
  **~35% of input on Claude Code with tool use intact** (verified end-to-end). The `cache`
  stage is **off by default and guarded** — it no longer touches a request that already has
  `cache_control` (fixes a real 400 against Anthropic). `tool_select` / n-gram / TOON are off
  (≈0 gain on agent/prose traffic); opt in via a config file.
- **Background daemon + live status**: `serve --daemon`, `stop`, `status` (tokens saved, cost
  saved, total spend priced per-model via `llm_providers`, per-provider), `autostart`
  (cross-platform via `auto-launch`).
- **Response capture**: output tokens measured by teeing the streamed response (JSON + SSE),
  recorded to the ledger for total-spend reporting.
- **Google Gemini** provider adapter (`contents`/`parts` wire shape, `generationConfig`).
- `llmtrim setup` — one-command bootstrap: ensure the CA, write `HTTPS_PROXY` +
  `NODE_EXTRA_CA_CERTS` to the shell profile (no IDE config touched), enable autostart, start
  the daemon. `install.sh` runs it for a true one-liner.
- `llmtrim uninstall` — transparent one-command inverse: stop the daemon, disable
  autostart, strip the shell-profile block, remove the CA + state + binary (`--purge` also
  removes the savings ledger).
- Release scaffolding: `install.sh`, Homebrew formula, cross-platform release workflow, CI.

## [0.1.0] - Unreleased

Initial public release. Static, deterministic prompt/payload compression for the OpenAI and
Anthropic APIs — no auxiliary model calls, every transform measured with the real target
tokenizer and reverted if it doesn't earn its tokens.

### Added
- **Compression stages**: lexical retrieval (BM25/TextRank/MMR + DSLR sentence pruning),
  code skeletonization + minification (tree-sitter), data hygiene (JSON minify, numeric
  quantization, base64 stripping), columnar/TOON + CSV serialization, exact + SimHash
  near-duplicate collapse, reversible n-gram abbreviation, tool-schema selection/trimming,
  output-control shaping, DSS output shorthand, and provider prefix-cache breakpoints.
- **Workload presets** (`auto`, `safe`, `rag`, `agent`, `code`, `aggressive`, `cache`,
  `reasoning`) with structural auto-routing — zero model, picked from request shape.
- **Universal text handling**: language detection (whatlang) drives stopwords and the BM25
  tokenizer; Unicode-aware (UAX#29) tokenization works across scripts including CJK.
- **CLI**: `compress`, `send`, `serve` (MITM interceptor), `batch`, `eval`, `bench`, and
  `monitor` (savings dashboard; `status`/`gain` aliases). Response rehydration is internal.
- **Library surface**: `compress`, `compress_with_config` (the deterministic, tokio-free
  core; `default-features = false` for embedders).
- **Savings ledger** (SQLite) and live A/B benchmark harness (`--features live`).

[Unreleased]: https://github.com/fkiene/llmtrim/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fkiene/llmtrim/releases/tag/v0.1.0
