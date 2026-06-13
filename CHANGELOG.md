# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **UniFFI bindings (`llmtrim-uniffi`) + Python wheel.** A new binding crate exposes
  `llmtrim-core` to Python, Ruby, Swift and Kotlin from one Rust definition: a flat
  `compress(input, provider, preset) -> CompressOutput` surface with mapped error
  variants. Compression runs natively in-process (no server, no async). A reproducible
  `scripts/build-wheel.sh` produces a self-contained Python wheel, and
  `scripts/generate-bindings.sh` emits Python/Ruby/Swift/Kotlin glue from the same Rust
  definition. Python and Ruby are verified end-to-end (compress + error mapping); Swift
  and Kotlin generate cleanly (packaging as gem/SwiftPM/Gradle is the next step). See
  `crates/llmtrim-uniffi/README.md`.

### Changed
- **Split into a Cargo workspace: `llmtrim-core` (engine) + `llmtrim` (CLI/proxy).**
  The deterministic compression engine — `compress`/`compress_with_config`/`route`/
  `rehydrate`/`CompressResult` plus the pipeline, stage, provider, tokenizer, gate and
  config modules — now lives in a standalone `llmtrim-core` crate with no async/tokio
  in its dependency tree, so it can be embedded as a library. The `llmtrim` binary,
  MITM interceptor, daemon, token ledger, live benchmark and terminal UI move to the
  `llmtrim` CLI crate, which depends on `llmtrim-core`. No behavior change; the `llmtrim`
  command and its install paths are unchanged. `rehydrate` is now `pub` (the CLI's
  interceptor calls it across the crate boundary).

## [0.1.6] - 2026-06-12

### Added
- **Range-fold for regular sequences in tool-output template folds**: when a folded
  log's parameter column is a regular sequence — constant values, arithmetic integers,
  or constant-step ISO-8601-like timestamps — the explicit value list collapses to a
  lossless range (`[×30: (10:02:00Z..10:02:29Z step 1s; 0..29)]`). Every value stays
  byte-exactly reconstructible (a round-trip check gates each fold); irregular columns
  keep the explicit list, and a range is emitted only when strictly shorter. On the
  README's build-log example the same request now compresses −71% instead of −62%.
- **Missed-fold telemetry in the capture loop**: with `LLMTRIM_CAPTURE_DIR` set,
  datetime-ish columns that fall back to the explicit list are logged to
  `missed_folds.jsonl` (reason + 5-value sample), so real traffic — not guesswork —
  decides which timestamp shapes the range fold learns next. Zero overhead when
  capture is off; a write failure can never break a fold.

### Fixed
- **Re-run → passthrough rail now survives non-deterministic output**: the rail that
  ships a re-invoked tool's output in full used raw-text equality, so any run-to-run
  noise — TAP's `duration_ms` timings, log timestamps, ports, PIDs — defeated it and
  the retry was windowed identically. Repeat detection now compares a
  volatile-value-masked fingerprint (the template stage's variable masking), so a
  re-run that differs only in such values passes through in full, while a real result
  change (a test flipping `ok` ↔ `not ok`) still compresses fresh.
- **TAP test failures no longer elided** (reported against v0.1.5): a `node --test` /
  `prove` TAP log could lose its only failing test — `not ok N`, the YAML diagnostic,
  even the `# fail 1` summary — because the failure-signal regex didn't know TAP's
  `not ok` marker (nor camelCase tokens like `failureType: 'testCodeFailure'`), and
  the retrieve stage ranked chunks purely by query relevance with no failure
  protection at all. Failure-signal lines and their continuation blocks (indented
  traceback frames; for TAP, the whole diagnostic up to the next test point) now
  survive pruning in both the tool-output and retrieve stages, regardless of query
  overlap.

## [0.1.5] - 2026-06-12

### Added
- **`setup` reclaims orphaned daemons**: when the default port is busy, setup now
  identifies the holder (native OS tools); an old llmtrim daemon — e.g. left running
  after `npm uninstall`, which can't stop it — is killed and the default port reclaimed
  instead of silently drifting to the next port. Foreign holders are named in the note
  ("busy (chrome.exe, pid 123)").

### Fixed
- **`uninstall` no longer deletes package-manager-owned binaries**: under an npm /
  cargo / Homebrew install it keeps the file and prints the manager's uninstall command
  (deleting it out from under the manager left broken bookkeeping). INSTALL.md documents
  the order: `llmtrim uninstall` first, then the package manager.
- npm packages now ship a README (npmjs renders the tarball readme, not the repo's).

## [0.1.4] - 2026-06-12

### Fixed
- **crates.io publish (for real this time)**: excluding `.cargo/` from the package
  wasn't enough — `cargo publish`'s verify build runs under `target/package/` and
  cargo's config discovery walks up into the repo, still picking up the committed
  mold-linker config. The config now lives outside the repo (developer-local
  `~/.cargo/config.toml`) and the publish job defensively removes `.cargo/` before
  publishing. v0.1.3 never reached crates.io.

## [0.1.3] - 2026-06-12

### Fixed
- **`cargo install llmtrim` no longer requires mold**: the published crate accidentally
  shipped `.cargo/config.toml` with the local mold-linker setting, breaking the install
  build on machines without mold (caught by `cargo publish`'s verify on v0.1.2 — that
  version never reached crates.io). Now excluded from the package.
- **npm publish**: package directories are passed with a `./` prefix (a bare `a/b`
  argument is npm's GitHub-repo shorthand and made publish try to clone from GitHub).

### Changed
- **npm packages are scoped: `@llmtrim/cli`** (+ `@llmtrim/<os>-<arch>` platform
  packages). The unscoped `llmtrim` npm name belongs to an unrelated 2025 package —
  installing it does not get you this tool.

## [0.1.2] - 2026-06-12

### Added
- **Four new install channels**, each self-verified by the release pipeline:
  `cargo binstall llmtrim` (prebuilt, seconds instead of a source build), a multi-arch
  Docker image (`ghcr.io/fkiene/llmtrim`, distroless, built from the attested release
  binaries), Scoop (`scoop bucket add llmtrim https://github.com/fkiene/scoop-bucket`),
  and npm (`npm i -g @llmtrim/cli` — meta package + per-platform prebuilt binaries; the unscoped `llmtrim` name belongs to an unrelated package).
- **`LLMTRIM_BIND`**: `serve` binds an explicit IP (default stays loopback — a MITM
  proxy must not be reachable off-host unless asked). The Docker image sets `0.0.0.0`
  so port mapping works.
- **`ca --pem`**: print the CA certificate PEM to stdout — pipe it out of a container
  straight into `NODE_EXTRA_CA_CERTS`, no volume spelunking.

### Fixed
- **Windows uninstall self-delete**: a single delete attempt 2 s after exit lost the
  race when the shell or Defender still held the fresh exe; now retried for ~60 s.
- **`install.ps1` checksum verification**: GitHub serves the `.sha256` sidecar as
  octet-stream, making PowerShell's `.Content` a byte array; now downloaded to a file.
  Errors in that block also report the underlying exception.

## [0.1.1] - 2026-06-12

### Fixed
- **Binary installers verify checksums again**: `install.sh` / `install.ps1` requested
  `<archive>.tar.gz.sha256` while CI uploads `<name>.sha256` — every prebuilt install
  failed (404) at the verification step.
- **`cargo install llmtrim` compiles again**: capped transitive `time` below 0.3.48,
  whose new blanket trait impl collides with `rcgen 0.14` (E0119) on a fresh, un-locked
  resolve. Install docs and the `update` hint now recommend `cargo install --locked`.

### Changed
- Release pipeline hardening: crates.io publishing via Trusted Publishing (OIDC, no
  stored token), Homebrew tap bump as a plain script, least-privilege CI permissions.

## [0.1.0] - 2026-06-12

Initial public release. llmtrim is a static, deterministic LLM prompt/payload compressor —
no auxiliary model calls, every transform measured with the real target tokenizer and
auto-reverted if it doesn't earn its tokens. Worst case is zero savings: never a bigger
bill, never a broken call.

### Compression engine
- **Input stages**: lexical retrieval (BM25/TextRank/MMR + DSLR sentence pruning), code
  skeletonization + minification (tree-sitter), data hygiene (JSON minify — including
  lossless minification of JSON embedded in prose with exact numeric round-trip — numeric
  quantization, base64 stripping), columnar/TOON + CSV serialization, exact + SimHash
  near-duplicate collapse, reversible n-gram abbreviation, tool-schema
  selection/trimming, output-control shaping, DSS output shorthand, and provider
  prefix-cache breakpoints.
- **Read-path tool-output compression (Stage T)**: adaptive compression of `tool_result`
  content — log windowing with severity-aware keeps (errors-only mode under pressure),
  diff/grep/plaintext handling, repeated-template masking. Generated content (lockfiles,
  minified bundles, base64 blobs) is detected and skipped; ANSI/CR sequences are
  normalized before detection. Cache discipline guarantees provider prefix caching is
  never broken mid-conversation.
- **Workload presets** (`auto`, `safe`, `rag`, `agent`, `code`, `aggressive`, `cache`,
  `reasoning`) with structural auto-routing — zero model calls, picked from request shape.
- **Universal text handling**: language detection (whatlang) drives stopwords and the
  BM25 tokenizer; Unicode-aware (UAX#29) tokenization works across scripts including CJK.

### HTTPS interceptor & daemon
- **MITM interceptor** (`serve`): compresses every tool's LLM API calls in flight, with
  streaming (SSE) pass-through. Provider host allowlist + name-constrained CA derived
  from the `llm_providers` registry (OpenAI, Anthropic, Google Gemini adapters; every
  non-LLM connection is blind-tunneled untouched). `llmtrim ca` manages the local CA.
- **Live default preset tuned on real agent traffic** (A/B'd through `claude -p`):
  `hygiene` + exact `dedup` + tool-description trimming (300 chars) cuts ~35% of input on
  Claude Code with tool use intact, verified end-to-end. The `cache` stage is off by
  default and never touches a request that already carries `cache_control`;
  `tool_select` / n-gram / TOON are off (≈0 gain on agent/prose traffic) and opt-in via
  config.
- **Background daemon**: `serve --daemon`, `start`/`stop` (with proper wait-for-exit and
  wait-for-port so restart cycles never race), `autostart` (cross-platform via
  `auto-launch`), crash-restart supervision with restart counting.
- **Response capture**: output tokens measured by teeing the streamed response
  (JSON + SSE), recorded to the ledger for total-spend reporting.

### Observability
- **`monitor`** — live savings dashboard (tokens saved, cost saved, total spend priced
  per-model via `llm_providers`, per-provider breakdown, today anchor next to the
  all-time saving).
- **`status`** — health chain in the header (daemon alive → port accepting → env wired →
  CA present → age of last request), one calm line when healthy and one warning per
  broken link naming its fix. Exit codes follow the `systemctl is-active` convention
  (0 healthy / 1 stopped / 2 degraded); `-q/--quiet` prints one word for scripts;
  `--json/--csv` exports carry daemon state, uptime, restarts, and version skew.
- **`doctor`** — read-only end-to-end diagnosis: binary, daemon, port, persisted env and
  this shell's env, CA, autostart, ledger, daemon-vs-binary version skew — one pass/fail
  row each, non-zero exit when something needs fixing.

### Install & lifecycle
- **`setup`** — one-command bootstrap: ensure the CA, write `HTTPS_PROXY` +
  `NODE_EXTRA_CA_CERTS` to the shell profile (env-level only — no IDE config touched),
  enable autostart, start the daemon. `install.sh` runs it for a true one-liner.
- **`uninstall`** — transparent inverse: stop the daemon, disable autostart, strip the
  shell-profile block, remove CA + state + binary (`--purge` also removes the ledger).
- **`update`** — channel-aware: binary installs self-update via the installer;
  cargo/Homebrew print their command. `setup` restarts the daemon so updates go live.
  A cached (≤once/day), opt-out (`LLMTRIM_NO_UPDATE_CHECK`) release check surfaces a
  "vX.Y available" notice in `monitor`.

### CLI & library
- **CLI**: `compress`, `send`, `serve`, `batch`, `eval`, `bench`, `monitor`
  (`status`/`gain` aliases). Response rehydration is internal.
- **Library surface**: `compress`, `compress_with_config` — the deterministic, tokio-free
  core (`default-features = false` for embedders).
- **Savings ledger** (SQLite) backing all spend/savings reporting.

### Benchmarks & release infrastructure
- Live A/B benchmark harness (`--features live`): every case sent twice (original and
  compressed), answered, scored, and billed at real rates; judge-verdict parsing,
  transient-error skip, and cache-bust nonces for fair runs. Tool-output corpus +
  Headroom head-to-head harness included.
- `install.sh` / `install.ps1`, Homebrew formula, cross-platform release workflow
  (6 targets with SLSA build provenance), CI on Linux/macOS/Windows with secret
  scanning, license compliance, and MSRV gates.

[Unreleased]: https://github.com/fkiene/llmtrim/compare/v0.1.6...HEAD
[0.1.6]: https://github.com/fkiene/llmtrim/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/fkiene/llmtrim/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/fkiene/llmtrim/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/fkiene/llmtrim/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/fkiene/llmtrim/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/fkiene/llmtrim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fkiene/llmtrim/releases/tag/v0.1.0
