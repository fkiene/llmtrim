# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Codex CLI with ChatGPT sign-in is now compressed and tracked.** It posts to `chatgpt.com`, which was missing from the intercept set, so its traffic passed through the proxy untouched and never showed up in `llmtrim status` (#101).

## [0.5.0] - 2026-06-29

### Added
- **Ledger retention is now configurable.** Two new settings cap how much history llmtrim keeps: `max_rows` (env `LLMTRIM_MAX_ROWS`) for the compression event ledger and `max_breakdown_turns` (env `LLMTRIM_MAX_BREAKDOWN_TURNS`) for the per-source cost breakdown. Both take a positive integer and fall back to the built-in defaults when unset. Raise `max_breakdown_turns` to keep a deeper breakdown history at the cost of a larger database file.

### Changed
- **`llmtrim status` opens the live dashboard directly on a TTY.** The `--watch` flag is now a hidden no-op, kept so existing scripts keep working.

### Fixed
- **`status --json` no longer scans the whole ledger by default.** The per-source/per-session `breakdown` object aggregated the entire history on every call (a full `breakdown_blocks` scan that pinned a CPU core and used over a gigabyte of RAM on a large database). It is now opt-in via `status --json --breakdown`; the default export carries the same health and savings fields it always did.
- **The per-source breakdown table is now bounded.** It was capped at 100,000 *turns*, but each turn fans out into hundreds of source rows, so the table reached millions of rows and its write-ahead log grew past 700 MB without a checkpoint. Breakdown retention now defaults to 50,000 turns (tunable via `max_breakdown_turns`) and is enforced both as the daemon records turns and when the ledger is next opened, with a WAL checkpoint after pruning so the file shrinks once the daemon is stopped. Note: on first run after upgrading, breakdown history beyond the cap is deleted permanently.
- **Restart-to-update hints now point at `llmtrim start --force` everywhere.** The stale-version banner in `status`/`doctor` and the Homebrew/Cargo note in INSTALL.md previously suggested `llmtrim stop && llmtrim start` or `llmtrim setup`; the latter does not restart a healthy daemon already on the right port, so it could leave the old binary running after an update. All three now match what `llmtrim update` actually runs.

## [0.4.0] - 2026-06-28

### Added
- **Exclude providers or hosts from compression.** Two new config keys let a request pass
  through verbatim while still being proxied: `exclude_providers` (env
  `LLMTRIM_EXCLUDE_PROVIDERS`) by wire shape — `openai`, `anthropic`, `google`; coarse, covers
  every host of that shape — and `exclude_hosts` (env `LLMTRIM_EXCLUDE_HOSTS`) by exact hostname,
  precise. Each env var is comma-separated and replaces the file array. Set
  `exclude_providers = ["anthropic"]` (or `LLMTRIM_EXCLUDE_PROVIDERS=anthropic`) to leave Claude
  Code traffic untouched while still compressing everything else.
- **Cost breakdown now labels 18 coding agents, up from 3.** The per-source view recognized
  only Claude Code, Codex, and Cursor; every other tool's traffic landed under `unknown`. It
  now fingerprints cline, roo, kilo, goose, opencode, crush, gemini, qwen, grok, kimi,
  mistral-vibe, mux, pi, forge, and openclaw from their system prompts as well, so their token
  and cost stats show up under the right name.
- **`status` dashboard: press `c` to switch what the size gauge measures.** On the Overview tab,
  the "SMALLER REQUESTS" gauge toggles between the reduction of new content (the default, what
  llmtrim can actually trim) and the reduction across the whole prompt (diluted by the reused
  cached prefix). The caption names the active basis. Dollar figures are unaffected.

### Fixed
- **The SAVINGS TREND chart no longer prints the dollar amount on top of the bar.** ratatui's
  `BarChart` overlays its value label on the bar's base, so the `$/day` figure was painted over
  the green bar. The amount now sits on its own row between the bar and the weekday label, so the
  bar, amount, and day read as three clean rows.
- **Replies now stay in the user's language.** The output-shaping instructions are written in
  English and land last in the request, which biased the model to answer in English even when
  the user wrote in another language. When the prompt is detected as non-English, the primary
  shaping instruction now carries a short `Reply in the user's language.` clause; reliably
  English prompts add nothing, so the fix costs no tokens on the common case (prompts too short
  to detect get the clause as a cheap, safe fallback). Only applies when output shaping is
  already active (never on tool-call requests).
- **`status` dashboard: "SAVED TODAY" now reconciles with the totals.** It was priced at list
  rates while the all-time and weekly figures used the real, cache-discounted bill, so on
  cache-heavy traffic (e.g. Claude with prompt caching) today could read higher than the total
  ever saved. Every dollar figure is now the same net-of-cache basis.

## [0.3.2] - 2026-06-22

### Added
- **llmtrim is registered in the official MCP Registry.** A `server.json` declares the
  `io.github.fkiene/llmtrim` server, `@llmtrim/cli` carries the `mcpName` ownership marker, and a
  release workflow publishes to `registry.modelcontextprotocol.io` over GitHub OIDC, so MCP
  clients can discover the server (`llmtrim_compress`, `llmtrim_compress_text`, `llmtrim_stats`).

### Fixed
- **`update` now restarts the daemon itself instead of leaving it stale.** On the binary
  channel (the installer path), `llmtrim update` laid down the new binary but left the running
  daemon on the old build, so `status` still showed a `u  Update` nudge and you had to restart
  by hand. It now runs `llmtrim start --force` after a successful install, finishing the job in
  one command.
- **A failed `update` now shows how to finish it by hand.** When the installer can't launch or
  exits non-zero, `update` prints the manual command for that channel (the `curl … | sh`
  one-liner, version pin included) plus the daemon restart, instead of just reporting the error.

## [0.3.1] - 2026-06-21

### Fixed
- **`status` dashboard: a stale daemon now offers a working restart, not a broken command.**
  When the running daemon was an older build than the installed binary (after an update), the
  dashboard treated it as a fault: a Repair button and a suggested `llmtrim restart`, which is
  not a real command. It now shows a `u  Update` action that restarts the daemon
  (`llmtrim start --force`, the supported "pick up a new binary after an update" path) to load
  the new build. A newer release still takes precedence over a restart, and a failed restart
  surfaces its exit code instead of exiting cleanly.

### Removed
- **The Proxy-Wasm gateway plugin (Kong / Higress) has been removed.** Shipped in v0.3.0, it is
  dropped along with its crates (`llmtrim-gateway`, `llmtrim-gateway-plugin`) and its release
  publishing. The v0.3.0 OCI image and release asset stay where they are; no new versions
  publish. Gateway-side integration may return later through a different surface.

## [0.3.0] - 2026-06-21

### Added
- **Interactive cost dashboard on `llmtrim status`.** On a terminal, `status` now opens a
  tabbed TUI (Overview / Sessions / Detail): what you saved, what you paid, the daily trend,
  per-model and per-session breakdowns, and where each session's tokens went. Piped or
  redirected output keeps the plain text, so scripts are unaffected. `t` switches theme
  (Catppuccin Mocha / Macchiato / Frappé / Latte, persisted), `q` quits.
- **`net_spend_usd` in `status --json`.** The JSON now also carries the cache-discounted
  bill, so the machine-readable output matches the dashboard's "you paid" figure.
- **Proxy-Wasm gateway plugin: compress LLM request bodies at Kong or Higress.** A single
  wasm module buffers the request body, runs it through llmtrim, and forwards the smaller body
  upstream. The same artifact runs on both gateways (both are Proxy-Wasm 0.2 ABI hosts); they
  differ only in deployment config. It is fail-open: an oversized, non-JSON, or
  undetectable-provider body is forwarded unchanged. Configurable per route with `provider`,
  `preset`, and `max_body_bytes`. Each release publishes the module to both channels: an OCI
  artifact at `ghcr.io/fkiene/llmtrim-gateway-plugin` (for Higress) and a `.wasm` Release asset
  (for Kong). See `crates/adapters/llmtrim-gateway-plugin/README.md`.

- **`llmtrim_core::rewrite_request(input, provider, preset)`**: a stable entrypoint for
  integration adapters. It defaults the preset to `auto` (per-request structural routing) and
  never reads the environment or a config file, so the same request compresses identically
  across the native, WASM, and UniFFI bindings. A cross-binding conformance suite
  (`crates/llmtrim-core/tests/conformance/`) pins its output so a core change cannot silently
  alter what an adapter forwards.
- **`serve --force` / `start --force` / `setup --force`** replace an llmtrim daemon that's
  already holding the port (stops it first, waits for the port to free, then takes over)
  instead of refusing, no-opping, or leaving a healthy same-port daemon untouched.
- **Custom intercept hosts.** Point llmtrim at a self-hosted or gateway OpenAI-compatible
  endpoint with `LLMTRIM_EXTRA_HOSTS=host1,host2` (or an `extra_hosts` array in the config
  file). The name-constrained CA regenerates automatically to cover them. Entries must be
  exact hostnames; malformed or overly broad ones are dropped.
- **Every runtime setting is now settable in the config file, not just via env.** The
  proxy knobs (`upstream_proxy`, `db_path`, `capture_dir`, `capture_max_mb`, `bind`,
  `breakdown_window`, `no_update_check`) each now resolve env-first then from a top-level key
  in `config.toml`, matching the existing `preset`/`retention_days` behavior. The matching
  `LLMTRIM_*` env vars are unchanged and still win when set.

### Changed
- **`@llmtrim/js`: calling `compress()` with no preset now compresses with `auto`
  (shape-routing) instead of the lossless-only baseline.** The npm package barely compressed
  out of the box before, because the no-preset path fell back to lossless; it now matches the
  native CLI and the live proxy. This changes the body forwarded to your provider when you call
  `compress(body, provider)` with no third argument. To keep the previous lossless-only
  behavior, pass `"safe"` as the preset.

### Performance
- **Lower per-request compression cost on tool-heavy (agent) prompts.** The pipeline now
  re-serializes and re-tokenizes the tools array only when a stage actually changed it, instead
  of after every content-rewriting stage. Per-language stopword sets are built once and reused
  (behind a read-mostly lock) rather than rebuilt per segment. Output is unchanged.

### Fixed
- **Strict-TLS clients can now use the proxy.** The MITM leaf certificates llmtrim minted
  lacked the Authority Key Identifier (and Key Usage / Extended Key Usage) extensions. Python
  3.13 turns on OpenSSL's `VERIFY_X509_STRICT` mode by default, which enforces RFC 5280 and
  rejects such a leaf with "Missing Authority Key Identifier", breaking every Python `httpx` /
  OpenAI-SDK client behind the proxy (e.g. Hermes Agent, litellm). `curl` and Node TLS don't run
  the strict checks, so this went unnoticed. Minted leaves now include the Authority Key
  Identifier plus Key Usage and Extended Key Usage (serverAuth). The local CA is unchanged, so
  no re-trust is needed.
- **`serve` fails fast when the port is taken.** A bind-in-use error no longer spins through
  the supervised restart loop five times; it reports once, naming the daemon already on the
  port (`llmtrim stop` first, or `--force` to replace it) or the foreign process holding it.
- **A recycled pid no longer reads as a running daemon.** Daemon liveness now verifies the
  pid is actually llmtrim, not just that *some* process has that pid, so a stale pidfile
  pointing at an unrelated reused pid is cleared instead of reported as running.
- **`@llmtrim/js` / `@llmtrim/wasm` now publish to npm.** The v0.2.1 release built the
  package but its `wasm-pack` publish step failed (`wasm-opt` rejected the post-MVP wasm
  features recent rustc emits), so the packages never landed on npm. The redundant
  `wasm-opt` pass is now skipped (rustc already optimizes under `--release`), so the
  bindings publish.

## [0.2.1] - 2026-06-19

### Added
- **`LLMTRIM_UPSTREAM_PROXY` env var** — set this in the daemon's launch environment
  (launchd plist / systemd unit, not `.profile`) to route the daemon's own outbound
  requests through a corporate or gateway proxy, or a local companion proxy such as
  headroom on a different port (e.g. `http://127.0.0.1:9999`).
  Accepts `http://host:port` or `http://user:pass@host:port`. Only upstreams that match
  llmtrim's own listen address (same loopback host AND same port) are rejected at startup
  to prevent infinite recursion; a different port on the same loopback interface is
  allowed. Credentials in the URL are plaintext in the process environment — use a
  credential-free URL with OS-level auth where possible.
  Requested by @gkgoat1 ([#62](https://github.com/fkiene/llmtrim/issues/62)).
- **`llmtrim-core` builds for C-toolchain-free targets (including WebAssembly).** The
  tree-sitter grammars, the tiktoken BPE vocabs, and the `image` decoders are now optional
  cargo features (`skeleton`, `tiktoken`, `multimodal`), all on by default, so default
  builds are unchanged. A `--no-default-features` build drops them; without `tiktoken` the
  estimate tokenizer is used (token counts become approximate, savings percentages
  unchanged).
- **WebAssembly/JS bindings, published to npm as `@llmtrim/js`.** A single
  `compress(input, provider, preset)` call that returns the compressed body, per-stage
  report, and token counts as a fully-typed object (TypeScript definitions are generated
  via `tsify`, so consumers get a typed `CompressOutput`). It runs the engine in a browser,
  Node, or a Cloudflare Worker with no network or filesystem access. (`@llmtrim/wasm` is an
  alias for the same package.)

### Fixed
- **`json-crush` now preserves the original row count of sampled arrays** so count and
  aggregate questions stay answerable after down-sampling. Previously a sampled array lost
  its original length. A bare top-level array now reports it in the sample note text; a
  nested array field leaves a `_sampled_from_<field>: N` sibling on its parent object.
  Neither injects a sentinel row into the array nor wraps a bare array in an object, so
  exact aggregation over the kept rows is unaffected.

## [0.2.0] - 2026-06-18

### Changed
- **Relicensed from AGPL-3.0-only to MPL-2.0.** llmtrim is now under the Mozilla Public
  License 2.0. You can use it freely, including commercially and as a network service, with
  no source-disclosure obligation; the file-level copyleft applies only to modifications you
  make to llmtrim's own source files. This removes the AGPL network-use trigger.

### Added
- **`llmtrim discover` splits residual into live vs cache-frozen.** Each bucket gains `live
  tok` / `live%` columns and the header shows a corpus-wide addressable total. Only residual
  after the last `cache_control` marker can still be compressed; the frozen cached prefix is
  reported separately so the ranking reflects reachable headroom.

### Fixed
- **`llmtrim discover --by-tool` now attributes OpenAI tool results to their tool.** A
  `role: "tool"` message with string content was never matched to its `tool_call_id`, so its
  residual landed in the `unknown` bucket instead of the tool that produced it. Anthropic
  `tool_result` blocks were already attributed correctly.

## [0.1.13] - 2026-06-17

### Added
- **`llmtrim discover` finds where compressible tokens still escape compression.** A
  read-only scan over the before/after capture corpus (written when `LLMTRIM_CAPTURE_DIR`
  is set) that re-buckets each request's token surface by block kind (system, user,
  assistant, tool_result, tool_call_args, document, tool_schema) and, with `--by-tool`, by
  the tool behind each tool_result. Each row reports the residual still in the compressed
  request, its share of the corpus-wide residual, and how much compression already removed
  (before→after) — so the next compression target is chosen from real traffic instead of
  guesswork. `--json` for the machine-readable report, `--dir`/`--limit` to scope the scan.
- **`llmtrim wrap <agent>` convenience launcher.** A one-command way to run a coding agent
  through the interceptor: `llmtrim wrap claude`, `llmtrim wrap codex -- --model …`, or any
  binary on PATH. It is sugar over `setup` plus a subprocess launch (no per-agent config and
  no base-URL rewriting) and it refuses to launch when `HTTPS_PROXY` isn't pointing at
  llmtrim in the current shell, so a wrapped agent can't silently bypass compression. Starts
  the daemon for you if the environment is wired but the interceptor is down.

### Fixed
- **`setup` now sets `NO_PROXY` so localhost and LAN traffic bypasses the interceptor.** The
  managed shell-profile block (and `HKCU\Environment` on Windows) wired `HTTPS_PROXY` for the
  whole user, which made *every* proxy-aware program — not just LLM tools — funnel its local
  and LAN calls at `127.0.0.1:<port>`, where they failed with "couldn't connect" whenever the
  interceptor was down or on a stale port (e.g. Plex device discovery, localhost dev servers).
  The block now also exports `NO_PROXY`/`no_proxy` (both casings — curl and Go only read the
  lowercase form): `localhost`, `127.0.0.1`, `::1`, `*.local`, plus the private LAN CIDR ranges.
  The literal hosts and `*.local` are honored by nearly every client; CIDR-based LAN bypass is
  best-effort (curl and Node/undici match only exact host or domain suffix, not CIDR). Existing
  installs self-heal: the daemon rewrites a pre-`NO_PROXY` block in place when it starts at
  login (reusing the wired port), so no `setup` re-run is needed. Already-running apps still
  need a one-time restart to pick up the new environment.
- **Token-F1 bench scorer now char-tokenizes CJK, fixing degenerate scores on
  Chinese/Japanese/Korean.** The scorer split on whitespace, which CJK text doesn't use, so a
  whole CJK answer collapsed to one or two "tokens" and the reported F1 was meaningless. It now
  character-tokenizes CJK runs, so quality benchmarks over CJK corpora report a real F1.
- **`tool_trim` stage now appears as `"tool_trim"` in capture `stages` lists** (was `"tools"`).
  When description trimming is active (`agent`/`aggressive` presets), the `ToolStage` stage name
  is now `"tool_trim"` instead of `"tools"`. This lets QA auditors correctly identify
  description-trimming runs as lossy and not flag them as category-4 bugs (lossless-only runs
  that dropped content). Lossless-only uses (selection + schema minification without trimming)
  continue to appear as `"tools"`.

## [0.1.12] - 2026-06-15

### Added
- **Logo-faithful terminal motifs in the shared UI.** The `ui` design system speaks the
  logo's "trim the wool, same sheep" story in one voice: a `wordmark` banner
  (`‹‹ llmtrim ››`), the shear before→after metaphor on the savings axes
  (`108.8M ─✂─▶ 34.8M`), a `hero` accent style, and a `sparkline`. `monitor`'s hero carries
  the `✓ same answers, smaller bill` promise, and `monitor --daily/--weekly/--monthly` and
  `update` lead with the wordmark.
- **Redesigned `status` dashboard.** A single dominant hero figure in a clean box — the
  real, cache-discounted dollars that came off the bill — with the promise
  (`✓ same answers, smaller bill`) beneath it, the input savings drawn with the shear metaphor
  (`108.8M ─✂─▶ 34.8M`), and a new `7-DAY TREND` sparkline of daily tokens saved. The header
  collapses to one calm strip (`‹‹ llmtrim ›› ● running · … ✓ healthy`) when healthy, expanding
  to per-link warnings only when degraded. The savings bars fill with the accent (the win grows
  the solid block), and the `BY MODEL` table is sorted by `$` saved with a light header rule.
  The added-latency footer and every honesty caveat are preserved.

### Changed
- **The `LLMTRIM_CAPTURE_DIR` corpus is now size-capped.** Capture wrote one JSON per
  request with no ceiling, so a long-lived daemon could fill the disk (which then starves
  the daemon's own pidfile and ledger writes). It now evicts the oldest `*.json` captures
  once they exceed `LLMTRIM_CAPTURE_MAX_MB` (default 1024; set 0 to disable). The sweep
  counts only top-level capture files (any other files you keep in the dir are left alone)
  and runs on a background thread, so it never blocks request handling.

### Fixed
- **`status --watch` no longer drifts on terminals narrower than its longest line.** The
  in-place repaint assumed one logical line per screen row, so a soft-wrapped line (e.g. a
  daemon warning) left stale rows. Lines are now truncated to the terminal width (ANSI-aware)
  before the repaint; the full text still prints in the one-shot `status`.
- **`llmtrim update` now prints the correct npm upgrade command.** It printed
  `npm update -g @llmtrim/cli`, which npm treats as a no-op for a globally installed package
  already on a satisfying version; it now prints `npm install -g @llmtrim/cli@latest`.
- **`llmtrim update` on a Homebrew install now prints a command that works.** The Homebrew
  arm told you to run `brew upgrade llmtrim`, but the formula is tapped as
  `fkiene/tap/llmtrim`. On a machine that never added the tap, that errors with "no
  available formula named llmtrim" and you stay on the old binary. It now prints the
  tap-qualified, idempotent form (`brew tap fkiene/tap` then
  `brew upgrade fkiene/tap/llmtrim`).
- **`status` no longer reports "stopped" while the proxy is serving.** Health was decided
  from the pidfile alone, so a daemon whose pidfile went missing (e.g. lost to a full disk)
  showed the loud "stopped — LLM calls will fail" banner even though the proxy was still
  live on the wired port. `status` now probes that port directly: a proxy answering with no
  pidfile reads as running-but-degraded (llmtrim can't confirm it owns the listener, so it
  flags "no pidfile … re-run `llmtrim setup`") instead of the false "stopped". The
  supervised daemon also re-records its own pidfile on restart, so a transient loss
  self-heals.
- **`llmtrim autostart` no longer hardcodes the default port.** Run with no `--port`, the
  command wrote the default port into the login entry regardless of the port your daemon
  and `HTTPS_PROXY` were actually on, so a reboot could bring the interceptor up on a port
  the environment wasn't wired to (LLM calls then fail until re-fixed). It now resolves the
  port the same way `setup`/`start` do — explicit `--port`, else the running daemon, else
  the configured env — and only falls back to the default when nothing is pinned.
- **`uninstall`'s closing message now names the leftover env vars and gives a remedy that
  works.** It said only "open a new shell", which never told you what was left behind and
  read as optional. It now spells out that the current shell still has `HTTPS_PROXY`,
  `HTTP_PROXY`, and `NODE_EXTRA_CA_CERTS` exported (the exact set `setup` writes) and that
  clearing them means a new shell or `unset HTTPS_PROXY HTTP_PROXY NODE_EXTRA_CA_CERTS` —
  not re-sourcing the profile, which leaves an already-exported var set.

## [0.1.11] - 2026-06-14

### Added
- **Named academic benchmarks: TruthfulQA, SQuAD v2, BFCL.** The quality A/B now ships
  three more standard suites alongside GSM8K, so the accuracy-preservation results name
  the benchmarks a reader already knows. `bench/scripts/download.py` fetches them
  reproducibly (`download.py 40 truthfulqa,squad2,bfcl`, sha256-pinned in the manifest),
  `bench suite` runs them at a conservative shape-matched preset, and the results table is
  in the README. BFCL uses the multi-tool `live_multiple` slice (2 to 37 candidate
  functions per call), where tool selection cuts 33% of input by dropping the schemas the
  query doesn't need, at unchanged tool-call accuracy. SQuAD v2's unanswerable questions
  are handled correctly: a right "no answer" scores as a hit. A new `choice` (MC1) scorer
  grades TruthfulQA by the selected option letter, not by any letter the model mentions in
  passing.
- **`llmtrim mcp` runs an MCP server over stdio.** Any MCP client (Claude Code, Cursor,
  custom agents) can spawn `llmtrim mcp` and call the engine as tools: `llmtrim_compress`
  (compress a full request body and report the token deltas, honoring your `~/.llmtrim`
  config like the proxy and CLI), `llmtrim_compress_text` (shrink a single text blob with
  the lossless `safe` preset, independent of config), and `llmtrim_stats` (read the savings
  ledger, the same data `llmtrim status --json` shows). Every call records to the same
  ledger, so MCP traffic shows up in `llmtrim status`. Behind the `mcp` feature, which ships
  in the default build. `llmtrim mcp install` registers the server with Claude Code via its
  `claude mcp add` CLI (idempotent); `llmtrim mcp install --print` emits the config block to
  paste into any other client.

### Changed
- **The benchmark commands are now one `bench` subcommand group.** `llmtrim bench` and
  `llmtrim bench-agent` are replaced by `llmtrim bench quality` and `llmtrim bench agent`,
  joined by three new axes under the same dispatcher: `bench suite` (the full corpus matrix
  in one process, replacing the `run_all.sh` shell script and its per-corpus `cargo run`
  spawns), `bench latency` (the warm compress-path micro-bench, folded in from the loose
  `latency.rs`), and `bench compare <headroom|caveman>` (a thin dispatcher over the Python
  head-to-head comparators). `bench suite` refuses to run live while an `*_PROXY` var is set,
  so the llmtrim proxy can no longer silently contaminate the A/B baseline.
- **Benchmark result JSON now carries a shared envelope.** Every `--json-out` (quality,
  suite, agent) wraps its body in `{ schema, produced_at, commit, llmtrim_version, meta,
  result }`, so any consumer can identify the schema and the code that produced it. The
  README/chart synthesizers unwrap it transparently and still read pre-envelope files.
- **`bench quality --offline --json-out` now writes its results.** Previously `--json-out`
  was honored only on live runs, so the free offline savings pass produced nothing on disk.
  It now writes a `quality-offline-v1` envelope (per-case input-token before/after plus the
  totals), which makes `bench suite --offline` reproducible without an API key.

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

[Unreleased]: https://github.com/fkiene/llmtrim/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/fkiene/llmtrim/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/fkiene/llmtrim/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/fkiene/llmtrim/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/fkiene/llmtrim/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/fkiene/llmtrim/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/fkiene/llmtrim/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/fkiene/llmtrim/compare/v0.1.13...v0.2.0
[0.1.13]: https://github.com/fkiene/llmtrim/compare/v0.1.12...v0.1.13
[0.1.12]: https://github.com/fkiene/llmtrim/compare/v0.1.11...v0.1.12
[0.1.11]: https://github.com/fkiene/llmtrim/compare/v0.1.10...v0.1.11
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
