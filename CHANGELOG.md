# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **`setup`'s caveman warning no longer claims llmtrim shapes output the same way caveman
  does.** caveman users run coding agents, which route to the `agent` preset where `auto`
  deliberately leaves output unshaped, so the old "llmtrim already does this (Stage F)" reason
  was wrong for exactly the people who saw it. The warning now explains that `auto` already
  shapes output where it pays (code, long context, plain prose) and skips tool-call traffic
  because terse shaping saves no tokens on short replies (bench: quality neutral), so caveman
  is redundant either way.

## [0.1.10] - 2026-06-14

### Added
- **Language bindings now expose the per-stage compression breakdown.** `CompressOutput`
  carries a `stages` list (one `StageReport` per pipeline stage: `name`, `applied`,
  `tokens_before`, `tokens_after`, `note`), so embedders in Python, Ruby, Swift and Kotlin
  can attribute the input-token reduction to each stage instead of only seeing the total.

### Fixed
- **Windows autostart no longer leaves a console window open.** The login Run-key entry
  launched `serve --supervised` as a foreground console app, so Explorer opened a terminal
  that stayed visible for the daemon's whole life. The entry now passes `--hide-console`,
  which hides the process's own console at startup, so the interceptor runs windowless at
  login. Re-run `llmtrim setup` (or `llmtrim autostart`) to rewrite the entry.
- **Python package now carries its README on PyPI.** The wheel set a summary but no long
  description, so the PyPI project page showed "no project description". `pyproject.toml`
  now points `readme` at the binding README.
- **Intel-mac Ruby gem (`x86_64-darwin`) now publishes.** The cross-compiled gem inherited
  the build host's platform (`Gem::Platform.local` -> `arm64-darwin`) and collided with the
  native arm64 gem, so it never shipped. The gem platform is derived from the build target
  instead.
- **Package metadata reads cleanly.** Removed the em-dash from the shared description string
  used by the PyPI summary, the Maven Central description, and the gem summary.
- **Release no longer stalls on a flaky provenance step.** On the native arm64-Windows
  runner the binary can land in `target/release` instead of `target/<triple>/release`, so
  the attestation intermittently failed and cascaded skips onto npm/Docker/Scoop. The step
  now resolves whichever path holds the binary.

## [0.1.9] - 2026-06-13

### Added
- **Swift package.** llmtrim is now installable from Swift Package Manager via
  [`fkiene/llmtrim-swift`](https://github.com/fkiene/llmtrim-swift):
  `.package(url: "https://github.com/fkiene/llmtrim-swift", from: "0.1.9")`. It wraps the
  prebuilt `llmtrimFFI.xcframework` attached to each release, so `import Llmtrim` needs no
  Rust toolchain. This replaces the previous "build the XCFramework yourself" step.

## [0.1.8] - 2026-06-13

### Fixed
- **Language-binding publishes (PyPI / RubyGems / Maven Central) now build their
  `x86_64-apple-darwin` artifacts by cross-compiling on an arm64 macOS runner** instead of
  natively on a `macos-13` Intel runner. Intel macOS hosted runners are scarce and the
  `v0.1.7` binding jobs stalled in the queue, blocking those publishes. The CLI/crate
  release was unaffected. The build scripts now honor an optional `LLMTRIM_TARGET`.

## [0.1.7] - 2026-06-13

### Added
- **`LLMTRIM_CAPTURE_DIR` records the applied stages.** Each capture JSON now carries a
  `stages` array — the names of the compression stages that actually rewrote the request.
  Previously only `plan` (the output-rehydration plan, a different axis and usually empty)
  was recorded, so an external auditor could not tell a lossless run that dropped content
  (a bug) from a lossy stage doing its job.
- **`bench-agent` — agent-loop token benchmark (#14).** A new dev command drives a
  tool-calling loop with deterministic tool stubs over a small golden task set and records,
  per iteration, the input / cached / output tokens and tool-call count, plus totals and cost.
  It compares conditions (`baseline` vs presets, applying the proxy's own compress + turn-memo
  transform) so a preset's agent-loop value can be measured on a set instead of one noisy live
  session. A provider-agnostic data contract and loop driver back a single reference provider
  (OpenAI Chat); `--dry-run` (default) uses a synthetic transport with no API calls, `--live`
  (needs `--features live`) calls the model.
- **UniFFI bindings (`llmtrim-uniffi`) + Python wheel.** A new binding crate exposes
  `llmtrim-core` to Python, Ruby, Swift and Kotlin from one Rust definition: a flat
  `compress(input, provider, preset) -> CompressOutput` call with errors mapped to native
  exceptions, running natively in-process (no server, no extra model calls). Each language
  ships as a published package with the compiled engine bundled (no Rust toolchain needed
  by consumers): a Python wheel (PyPI), a Ruby gem (RubyGems), a Kotlin/JVM jar (Maven
  Central) and a Swift package (SwiftPM/XCFramework), built for Linux, macOS and Windows.
  All four are exercised in CI. See `crates/llmtrim-uniffi/README.md`.

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

### Fixed
- **Tool selection no longer churns the cached prompt prefix on agent loops** (#9): tool
  *selection* keeps only the tools its relevance ranking scores against the conversation,
  so the kept subset changes from turn to turn. Providers fold the `tools[]` block into the
  cached prompt prefix, so a changing block invalidated the prefix on every turn of an agent
  loop — provider prompt-cache reads dropped and the prefix was rebilled as fresh input,
  which on a cache-warm loop can cost *more* than not compressing at all. Selection now runs
  **only on the first turn** of a conversation (where there is no prior prefix to bust and the
  saving is free); from the second turn on the tool set is left intact, and only the
  deterministic description-trim and schema-minify stages shrink the block — they are pure
  functions of the toolset, so the block stays byte-identical turn to turn (regression-tested).
  Applies to every preset that selects tools (`agent`, `aggressive`). A single-shot request with
  a large toolset still gets the full pruning saving. On a cache-warm multi-turn loop this keeps
  the tool prefix reusable instead of rebilling it each turn (an exploratory `gpt-4o-mini` run
  showed it roughly halving freshly-billed input once the prefix is warm — indicative, not a
  committed benchmark). The first turn ships the pruned set and turn two the full set, so there is
  a one-time prefix change at that boundary (a single extra cache write, ~25% on Anthropic) before
  it stays warm. This stabilizes the *tool block* on its own; keeping earlier-turn *message
  content* byte-stable across turns still relies on the turn-stability memo (`memo = true`, default).

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

[Unreleased]: https://github.com/fkiene/llmtrim/compare/v0.1.10...HEAD
[0.1.10]: https://github.com/fkiene/llmtrim/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/fkiene/llmtrim/compare/v0.1.8...v0.1.9
[0.1.8]: https://github.com/fkiene/llmtrim/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/fkiene/llmtrim/compare/v0.1.6...v0.1.7
[0.1.6]: https://github.com/fkiene/llmtrim/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/fkiene/llmtrim/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/fkiene/llmtrim/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/fkiene/llmtrim/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/fkiene/llmtrim/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/fkiene/llmtrim/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/fkiene/llmtrim/releases/tag/v0.1.0
