# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Grok reroute pins `prompt_cache_key` to the Claude Code session id.** Codex and Kimi already
  set Responses `prompt_cache_key` from `x-claude-code-session-id`; Grok did not, so multi-session
  concurrency (and any server-side routing change) could drop `cached_tokens` to 0/128 on an
  otherwise append-only conversation. cli-chat-proxy accepts the field (HTTP 200) and reports
  `input_tokens_details.cached_tokens` on hits.

- **Grok upstream usage capture under `LLMTRIM_CAPTURE_DIR`.** When the capture corpus is on, each
  completed Grok turn also writes `*-upstream-usage.json` with the raw `response.completed.usage`
  object and the mapped ledger fields, so a cache-collapse audit can compare Grok's wire usage
  against the Anthropic-shaped ledger without re-running the turn.

### Fixed

- **Upstream transport resets are retried and recorded instead of becoming silent zero-output rows.**
  Stale keep-alive connections to subscription backends (especially Grok) used to surface as hyper
  `SendRequest: connection reset` and land in the ledger as anonymous `out=0` rows. The outbound
  client now expires idle pool sockets after a short timeout (with hyper-util's `pool_timer` wired
  so the timeout actually fires), retries once when the request body is still available to replay,
  logs session/model on every transport failure, and records `outcome=transport_reset` rather than
  a silent zero-output row.

## [0.11.5] - 2026-07-20

### Added

- **`llmtrim setup --env` creates CA files (if needed) and prints envvars** When this option is
  used, setup ensures the local CA (and, on POSIX, the combined OS-trust bundle) exist, then
  prints an evaluatable snippet — `export` lines on POSIX, `$env:` assignments on Windows — to
  wire the interceptor into your current shell.

### Fixed

- **The managed shell-profile block now single-quotes its values.** `setup` interpolated the proxy
  URL and CA paths into bare double quotes, so a home directory or CA path containing `"`, `$`, a
  backtick or a backslash produced a profile line that broke or expanded at shell start. Values
  are now emitted as inert single-quoted literals on both the POSIX and PowerShell paths, matching
  the quoting `setup --env` already used. Re-run `llmtrim setup` to rewrite an existing block.

- **A legacy `sub` value keeps its formatting when migrated to a `[sub]` table.** Coercing
  `sub = "codex"` or an inline `sub = { … }` emitted the header as `[sub ]`, leaking the space
  that sat before the old `=`, and dropped a trailing comment on the line. The header is now
  clean and the comment moves to its own line above it.

- **The config writers now preserve comments and key order, and no longer corrupt the file.**
  All three (`compact.models`, `theme`, and the `sub` reroute editors) go through `toml_edit`,
  a format-preserving TOML editor, instead of hand-rolled line edits and a full re-serialize.
  Three bugs go with it: `compact.models = [...]` written in TOML's dotted top-level form was
  invisible to the writer, which appended a second `[compact]` section and left the key defined
  twice so the file stopped parsing; `llmtrim sub` re-serialized the whole document and silently
  deleted every comment while reordering keys; and `theme` was matched anywhere in the file
  rather than only at the top level. A writer handed a file that doesn't parse now reports it
  and leaves the file untouched, rather than editing it blind and producing plausible-looking
  output from an already-broken config.

- **`compact.models` no longer corrupts the config when it was written as a multi-line array.**
  The writer replaced the single `models = [` line and left the array's continuation lines
  behind, stranding `"haiku",` / `]` after the new value and producing a `config.toml` that
  llmtrim could no longer parse. It now consumes the whole array. A file already damaged this
  way still needs a manual edit: once the tail is orphaned nothing distinguishes it from real
  config, so repairing it automatically would risk deleting keys.

## [0.11.4] - 2026-07-18

### Added

- **Status line shows the active `/sub` plan's rate limits (Codex first).** Under a Codex
  reroute the `◔` segment now reads the ChatGPT plan's windows (weekly always; 5h when the plan
  reports one) from `GET /backend-api/wham/usage`, cached at `~/.llmtrim/sub-rate-limits.json` by
  the proxy on each throttled turn. A matching snapshot replaces Claude Code's Anthropic blob
  only while the proxy is healthy; until the first poll lands (or if the proxy is down) the
  Claude numbers stay so the segment isn't blank. Kimi/Grok share the cache schema; their usage
  endpoints are not wired yet.

### Fixed

- **Codex traffic is compressed and tracked again.** Codex CLI 0.144+ sends its request bodies
  with `Content-Encoding: zstd`, which the interceptor could not parse, so every Codex turn was
  forwarded verbatim and never reached the ledger — `llmtrim status` showed no Codex activity at
  all (#193). The interceptor now decodes zstd- and gzip-encoded request bodies (capped at
  256 MB) before compression and attribution, and forwards the body as plain JSON. A body it
  cannot decode still passes through verbatim, so a decode problem can never break the call.
- **Codex sessions are labeled `codex` in the cost breakdown again.** Two attribution gaps hid
  them: the fingerprint markers predated Codex 0.144's system prompt (which opens with "You are
  Codex" and no longer says "Codex CLI"), and identity extraction read only the *first*
  system/developer item of a Responses body — Codex now leads with a non-message
  `additional_tools` item, so the identity message behind it was skipped. The fingerprint gained
  the current marker and identity extraction concatenates every system/developer *message*,
  skipping tool registrations (whose schemas would also destabilize the session hash).

## [0.11.3] - 2026-07-17

### Fixed

- **`/compact` redirects to a cheaper model only when the prompt cache is cold.** The redirect
  used to run on every `/compact`. But `/compact` re-sends the conversation Claude Code has been
  caching against your selected model, so while that cache is warm a cache-read there is cheaper
  than a cold read on a smaller model, so redirecting a warm compact cost more, not less. The
  redirect now stays on your selected model when the session's last turn is within the cache TTL.
  Unknown freshness (no session row yet, or a build without the `breakdown` feature) still
  redirects, so the feature is never silently disabled.

## [0.11.2] - 2026-07-16

### Fixed

- **Window `/sub` survives Claude Code `/clear` (and `/compact`) without the env token.**
  Retention used to depend only on `LLMTRIM_CLAUDE_WINDOW_TOKEN` surviving into the next
  `SessionStart` hook. Claude Code often drops that env on clear, so `/sub on grok` was lost and
  the next turn fell back to Anthropic. SessionEnd now keeps the live session map and a
  short-lived `cleared` backup; SessionStart reattaches via session map, that backup, or the
  env token last (so a stale token from another window cannot override this session).
- **`/sub` skill bash no longer trips Claude Code permission checks.** Session id is read from
  process env (not shell-expanded on the skill line), and `Bash(<llmtrim> window-sub slash *)` is
  pre-approved via skill `allowed-tools` + `settings.permissions.allow`. `ensure` treats a
  missing/stale allow rule as stale so upgrades self-heal.

## [0.11.1] - 2026-07-16

### Changed

- **Overview and non-TTY status dollars match Sessions (frozen proxy bills).** KPI "You paid" /
  "Total saved" / today / week / trend / top models, and the piped `status` hero, read
  `breakdown_turns.bill_micros` plus a per-turn frozen-rate input counterfactual
  (`llmtrim-ledger::money`) — not live re-pricing of `compressions`. Lifetime $ may drop vs
  older builds when breakdown retention is shorter than compressions, or for CLI/MCP-only
  traffic (no session bill). Token % gauge is unchanged. When there is no proxy bill yet, the
  UI shows a billing notice (not `$0.00`).
- **Sessions columns:** `saved` → **`saved$`** (money) + **`trim`** (token size %). Detail bill
  footer uses `SUM(bill_micros)` (blocks allocate; `*` when they residual).
- **`status --json` / MCP `llmtrim_stats`:** additive **`money`** object with `unavailable` +
  null dollars when there is no proxy attribution. Legacy **`cost`** remains compressions +
  live list prices with `"source": "compressions_live_prices"` — prefer `money` when
  `unavailable` is false.

### Added

- **Money coverage UX** on Overview (and doctor empty-case only) when compressions exist but no
  breakdown bills — banner instead of fake `$0.00`.
- **`llmtrim-ledger::money`:** shared `MoneyTotals` / coverage / per-day / per-model queries;
  shared `saved_micros` SQL for Overview and Sessions.
- **Grok remote auth:** paste the callback URL or xAI one-time code into
  `llmtrim sub auth grok login`, or use RFC 8628 `llmtrim sub auth grok device` for SSH/headless
  hosts (no localhost callback).

## [0.11.0] - 2026-07-15

### Breaking

- **`SubProvider` / `StreamReducer` gain a `Grok` variant** and are marked `#[non_exhaustive]` so
  further backends can land without a major bump. External exhaustive `match`es need a `_` arm.
  Workspace version is **0.11.0** (0.x minor = semver-breaking).

### Added

- **`llmtrim ensure`:** bring this machine to the recommended state. Installs or refreshes owned
  Claude Code integrations (statusline, cold-cache guard, window-local `/sub`, default compact
  model chain), tray login when the GUI binary is present, and restarts a version-skewed daemon.
  Opt-outs live in `~/.llmtrim/integrations.json` so ensure does not re-enable after an explicit no.
- **`llmtrim doctor --fix`** and status **`f`** run the same ensure path.
- **Quiet auto-heal on `start` and status open** when the binary version advanced or owned pieces
  look stale (no prompts; no network tray download; no first-time installs).
- **Linux desktop tray one-shot download** during interactive ensure when the tray binary is
  missing (optional; default off when non-interactive).
- **One-time subscription onboarding in the status TUI** (`s` for login steps, `n` to dismiss)
  when Claude Code is present and `sub` is not configured.
- **Subscription reroute: Grok (SuperGrok / Grok Build).** `llmtrim sub auth grok login` signs in
  via OAuth against `auth.x.ai` (browser PKCE); `llmtrim sub use grok` maps Claude tiers to
  `grok-4.5` (opus/sonnet/fable) and `grok-composer-2.5-fast` (haiku) and routes Claude Code's
  Anthropic `/v1/messages` traffic to `cli-chat-proxy.grok.com/v1/responses`. Tokens live at
  `~/.llmtrim/grok/auth.json`. Works with `sub mode fallback` chains (`codex,kimi,grok`) and the
  window `/sub` command. Device-code login is not available yet (browser only).
- **Window `/sub on <provider>`.** The Claude Code slash command accepts an explicit backend:
  `/sub on codex`, `/sub on kimi`, or `/sub on grok`. Bare `/sub on` still re-enables the last
  window provider or the global `sub` setting.
- **Status line shows window `/sub` overrides immediately.** Mid-session `/sub on grok` (or any
  provider) now paints `→grok-4.5` even when earlier turns were Anthropic and global `sub` is
  `off`. Window overrides are treated as always-mode for the arrow (matching the proxy), and a
  policy switch overrides a stale ledger backend until the new one serves a turn.
- **Window `/sub` no longer inherits another provider's tier map.** With global `llmtrim sub on
  codex` and a Claude Code `/sub on grok`, the proxy was already routing to Grok but still applied
  `[sub.codex.tiers]` (e.g. `opus → gpt-5.6-terra`) on the Grok request. Tiers are now loaded per
  serving provider, and foreign model ids in overrides are ignored. After upgrading, run
  **`llmtrim ensure`** (or **`f`** in `status`) so owned `/sub` hooks refresh and a version-skewed
  daemon restarts onto the new binary.

### Changed

- **`setup` and binary-channel `update` call ensure.** First install wires
  statusline/guard/`/sub`/compact. Binary `update` restarts the daemon and runs `ensure -q` on the
  new binary (never in-process after a self-replace). Package-manager channels print the package
  command then a single `llmtrim ensure` follow-up.
- **Owned Claude Code hooks and statusline rewrite in place** when the binary path or payload
  (e.g. `refreshInterval`) changes. Quiet auto-heal only refreshes stale owned pieces and daemon
  skew; it never first-installs integrations or enables tray login.
- **Top-level help** leads with `setup` / `update` / `ensure` / `doctor`. Power-user
  `statusline` / `guard` / `compact` stay available but are hidden from the main command list.
- **Opt-outs stick:** `statusline` / `guard` / `window-sub` / `compact off` write
  `integrations.json` opt-outs; corrupt state refuses auto-apply instead of resetting them.
- **Safer hooks and tray install:** shell-quoted hook paths, atomic `settings.json` writes,
  shared `CLAUDE_CONFIG_DIR`, arch-aware tray download via path-filtered tar into a temp dir,
  then promote.

## [0.10.2] - 2026-07-15

### Added

- **Ordered model fallback for Claude Code `/compact`.** Configure `[compact] models = ["haiku",
  "sonnet"]` or run `llmtrim compact models haiku sonnet` to try cheaper models for compaction.
  Each candidate is admitted against its known context window after candidate-specific compression;
  failures before any output advance to the next candidate, and Claude Code's selected model remains
  the implicit final fallback. The same chain works through subscription tier mappings, while
  preserving the original model identity in the response. `llmtrim setup` proposes the default
  chain on Claude Code installations that have not made a choice.
- **`llmtrim guard` — stop paying for a cold cache without being told.** Resume a session that has
  been idle past the prompt-cache TTL and the next turn silently re-writes the entire context,
  billed at the cache-write rate (2× base input for the 1h TTL Claude Code asks for). On a 225k
  context — the median across my own idle gaps — that is a few dollars before any work happens, and
  nothing at the prompt says so. `guard` is a Claude Code `UserPromptSubmit` hook: on the first
  submit after a long gap on a big context it prints the idle time, the context size and what the
  cold write costs, then exits 2, which blocks the prompt with no API call. A resend goes straight
  through, and the blocked message is saved to disk since Claude Code does not restore it. `llmtrim
  setup` offers to wire it in; `guard install` / `guard uninstall` do it directly, merging into
  `~/.claude/settings.json` rather than clobbering hooks you already have. It reads Claude Code's
  transcript, not llmtrim's ledger — the ledger keys sessions by a hash of the system prompt, so a
  subagent turn would mask a stale main conversation — and every failure path exits 0: a bug in the
  guard must never block a prompt.

### Changed

- **The status line now refreshes while a session sits idle.** It already reddened the cache segment
  once the prompt cache expired, but Claude Code only re-renders on conversation events, so an
  abandoned session kept the line drawn at its last turn — green and warm — straight through the TTL
  expiring. The warning only appeared *after* the turn that paid for it. It now sets
  `refreshInterval`, so the state is on screen before you type. Re-run `llmtrim statusline install`
  to pick it up.
- **The cold-cache segment no longer suggests `/compact`.** It read `♻ cold · /compact`, which sold
  compaction as the thrifty move. It isn't: `/compact` re-reads the same cold context to summarise
  it — paying the charge it appears to avoid — and the next turn then writes a fresh cache for the
  summary. Resending pays once. The segment now just reports the state: `♻ cache cold`.

## [0.10.1] - 2026-07-13

### Fixed

- **Codex reroute failed every turn after the first.** The ChatGPT backend accepts
  `previous_response_id` only over its WebSocket transport; on the HTTP `/responses` path llmtrim
  uses, it answers `400 Unsupported parameter: previous_response_id`. Server-side continuation
  defaulted to on, so from the second turn onward every rerouted request died — and it was
  incoherent with the `store: false` llmtrim sends anyway, since there is no stored response to
  continue from. Continuation now defaults to off (still available via
  `LLMTRIM_CODEX_PREVIOUS_RESPONSE_ID` or `[sub.codex] previous_response_id`, for a transport that
  accepts it), and prefix caching via `prompt_cache_key` carries the cache instead. A rejected
  continuation now also heals itself: the request is re-issued in full rather than surfacing the
  400, so an enabled-by-config continuation degrades instead of breaking the session.

- **Compression could rewrite the system prompt.** The compressible set came solely from the cache
  zone, so a request with no `cache_control` markers had nothing frozen and the text-mutating
  stages were free to fold n-grams through the instructions. On Claude Code's title-generation
  call — which carries no markers — this rewrote the few-shot examples, deleting the conditional
  that gave them meaning. Instructions are what the model conditions on rather than reads as data,
  so a fold that is harmless in a tool result can invert a directive. `system`, `instructions` and
  system-role messages are now excluded from compression unconditionally, cached or not. This
  costs nothing on real traffic (593 of 594 captured requests already had `cache_control` on
  `system`; replaying 58 of them produces byte-identical output) and closes the gap for the small
  utility calls that carry no markers.

- **Status line blanked out mid-turn.** The cache percentage and context gauge were read only from
  Claude Code's blob, which fills `current_usage` and `total_input_tokens` from the *last*
  response. Claude Code re-runs the status line on every render, so while a turn was in flight both
  were absent: the cache segment vanished and the gauge emptied, then both snapped back when the
  turn landed — which reads like the prompt cache collapsing on every message. The proxy measures
  the same quantities on the wire, so the last completed turn now stands in across the gap. The
  blob still wins whenever it carries a figure, and a fresh session with no recorded turn renders
  empty rather than inventing a value.

### Added

- **Reroute request capture.** With `capture_dir` set, the body actually sent to a reroute backend
  is recorded alongside the existing before/after pair. The capture previously stopped at llmtrim's
  own pipeline output, leaving the translated payload — the thing the upstream prompt cache keys
  on — unobservable, which made cache behaviour under `sub` impossible to diagnose from evidence.

## [0.10.0] - 2026-07-12

### Fixed

- **Status line claimed a reroute that never happened in `sub` fallback mode.** The reroute arrow
  and the context gauge were derived from config: if `sub` was set, the line rendered
  `◆ Opus→gpt-5.6-terra` on every turn. But in `fallback` mode Anthropic serves the turn and the
  chain fires only when Anthropic *fails* — so the arrow named a backend that hadn't answered, and
  the gauge rescaled to that backend's context window, under-reporting how full the real (Claude)
  window was. The serving backend is now recorded per turn in the ledger (new `sub_provider`
  column, added to existing ledgers on open) and the status line reads *that*: the arrow appears
  only on a turn a backend genuinely served, and shows the model it ran. In `always` mode, where
  every turn reroutes, config still seeds the arrow before the first turn lands, so nothing
  changes there.

### Added

- **Ordered subscription fallback chains.** `llmtrim sub mode fallback` now tries a list of
  providers in order instead of a single one: `llmtrim sub chain codex,kimi` (or
  `LLMTRIM_SUB_CHAIN=codex,kimi`, or `[sub] chain = ["codex", "kimi"]`). When Anthropic cannot
  serve a turn — quota, payment, throttling, overload, or a transient server failure — Anthropic is
  retried once, and if it still fails each provider in the chain is tried in turn (with bounded
  backoff, honoring `Retry-After`) until one answers. A failure reported *inside* an HTTP 200
  stream counts as a failure too, and is caught before any of it reaches the client. With no chain
  configured, the active `sub` provider is the chain, so existing setups behave as before.

### Changed

- **Breaking (config): `sub mode on-error` is now `sub mode fallback`.** The old name still works —
  `llmtrim sub mode on-error`, `LLMTRIM_SUB_MODE=on_error`, and `[sub] mode = "on_error"` all keep
  meaning fallback, and print a one-line notice — but the new spelling is what gets persisted.
- **Breaking (library): `llmtrim-core`'s `RuntimeConfig::sub_on_error` is now `sub_fallback`**, and
  `RuntimeConfig` gained a `sub_chain` field. Hence the 0.10 minor bump.

### Fixed

- **Upstream connection failures are reported instead of swallowed ([#157](https://github.com/fkiene/llmtrim/issues/157)).**
  A request that failed *before* the upstream answered (a TLS error, a reset, a dropped
  keep-alive) fell through to the proxy library's default handler: an empty-bodied `502`, with
  the cause logged only through a `tracing` subscriber llmtrim never installs. Claude Code could
  not parse the reply, so the turn died — and nothing anywhere recorded why. A transport failure
  is now reported in the shape the client is reading (an SSE `error` frame on a streaming call, a
  retryable `overloaded_error` otherwise), and the underlying cause is named on stderr, so
  `llmtrim serve` in the foreground shows what actually broke.

- **Subscription reroute: the Codex prompt cache no longer goes cold every turn.** Claude Code
  prepends a billing-header system block whose hash changes on every turn, and the Codex translator
  forwarded it at the head of the Responses `instructions` — the very front of the cached prefix —
  so each turn re-sent a different prefix and paid a full cold cache. The block is now dropped for
  both Codex and Kimi. On a 4-turn session, cached input read back rose from 4.6k to 34.3k tokens.

- **Subscription reroute: the status line shows trim again.** Rerouted turns never recorded their
  Claude Code session, so the trim segment sat at `✂ –` even though compression was running
  normally. It now reports the session's real savings (`✂ 39.8%` on the run above).

- **GPT-5.6 Codex reroutes.** Claude model requests keep the standard Responses shape and use the
  official Codex originator and user-agent identity required by the backend.

- **Subscription reroute: Claude Code thinking no longer 502s on Codex.** The Codex
  reducer streamed `thinking_delta` summaries but never forwarded the upstream
  `encrypted_content` as Anthropic `signature_delta`, so Claude Code rejected the SSE
  stream (502 / dropped connection) and follow-up turns could not replay thinking blocks.
  Signatures now map both ways (`encrypted_content` ↔ `signature`), text waits for a late
  signature when needed, and adaptive `thinking` without an explicit `output_config.effort`
  still enables Codex reasoning.

- **Claude Code statuslines survive package upgrades.** `llmtrim statusline install` now keeps
  a stable executable path when the active binary is also reachable through a symlink on `PATH`
  (as with Homebrew's versioned `Cellar` layout), and `llmtrim update` re-points an existing
  llmtrim-owned entry before the upgrade runs, so the statusline no longer ends up pointing at
  a removed binary.

## [0.9.4] - 2026-07-12

### Fixed

- **Low cache reuse on `sub codex` / `sub kimi`.** Follow-up turns now reuse backend state via `previous_response_id` + output transcript, producing the expected high cache hit rates instead of full resends.

- **Status line stayed red after `/compact`.** The cold-cache segment (`♻ cold · /compact`) and
  context gauge keyed only on the interceptor ledger's `last_ts`. `/compact` runs inside Claude
  Code and does not update that timestamp, so the line kept warning after a successful compact.
  The renderer now clears the cold state when Claude Code reports a post-compact reset (empty
  context, no `current_usage` yet).

### Changed

- **Status line shows the resolved upstream model on `sub` reroutes.** Codex reroutes now render
  the concrete backend model (e.g. `→gpt-5.6-terra`) instead of the provider shortname (`→codex`);
  Kimi still shows `→kimi`.

## [0.9.3] - 2026-07-11

### Fixed

- **The desktop tray popover rendered blank.** Every installed tray (macOS, Linux, Windows) showed
  an empty white box under the tray icon instead of the stats UI. The shipped binaries were built
  with plain `cargo` rather than `cargo tauri build`, which left Tauri's `custom-protocol` feature
  off, so the webview ran in dev mode and tried to load a Vite dev server that does not exist on a
  user's machine. `custom-protocol` is now a default feature, so release builds embed and serve the
  real UI. ([#149](https://github.com/fkiene/llmtrim/issues/149))

## [0.9.2] - 2026-07-10

### Added

- **`llmtrim statusline`: a custom status line for Claude Code.** Reads Claude Code's JSON
  session blob on stdin and prints one width-adaptive line: `◆ model→backend`, a context gauge,
  compression saved for the current session (`✂ –` until it has trimmed anything), and (when
  present) 5-hour and 7-day rate-limit usage (`◔ 5h·24% · 7d·12%`) and this turn's prompt-cache
  reuse. The context gauge fills and colours against the *real* window of the model serving the
  turn — the rerouted backend's window from the model registry under a `sub` reroute, not
  Claude's — green below 40%, orange 40–65%, red above. Rate-limit percentages escalate green →
  amber (70%) → orange (80%) → red (90%). The cache segment turns into `♻ cold · /compact` once
  the session has been idle past the cache TTL, since the next message then pays a cold cache
  write. The reroute arrow (e.g. `→gpt-5.6-terra` or `→kimi`) shows the backend actually serving the turn,
  and a `⚠` replaces the savings segment when the interceptor is degraded. Extras shed
  right-to-left on narrow terminals. `llmtrim statusline install` wires it into
  `~/.claude/settings.json` (`--print` emits the snippet instead); `uninstall` removes it.
  `setup` points Claude Code users to it in its next-steps, but never writes the file itself
  (setup stays client-agnostic and touches no IDE config).

### Changed

- **`llmtrim sub` config changes apply immediately.** The interceptor reads its config once at
  startup, so a reroute change used to sit inert until you manually restarted the daemon. `sub on`,
  `off`, `mode`, `effort`, `map`, and `unmap` now restart a running interceptor in place to apply
  the change (and say so); pass `--no-restart` to keep the old behavior and just print the restart
  hint. Nothing running means nothing is restarted. The restart is skipped when it can't matter
  yet: `sub on` skips it while signed out, and `effort`/`map`/`unmap` skip it when the edit targets
  a provider reroute isn't currently pointed at.

- **`llmtrim sub codex`: updated default tier models.** The balanced preset now maps Opus to
  `gpt-5.6-terra`, Sonnet to `gpt-5.6-luna`, and Fable to `gpt-5.6-sol`; Haiku stays on the
  cheap `gpt-5.4-mini`. Custom `[sub.codex.tiers]` overrides are unaffected.

- **`llmtrim sub`: clearer on/off verbs.** Enabling a reroute is now `sub on <provider>`
  (with `sub use` and `sub start` accepted as aliases); disabling stays `sub off` (with
  `sub stop` as an alias). A bare `sub on` re-enables the last provider you used without
  retyping it, and keeps your customized tier mapping intact across an off/on cycle. `sub`
  also now appears in the top-level `--help`.

## [0.9.1] - 2026-07-09

### Fixed

- **Subscription reroute now surfaces the real upstream error.** When the ChatGPT/Codex or
  Kimi backend rejected a rerouted turn, llmtrim wrapped the failure in a `200 OK` SSE `error`
  frame, which Claude Code showed as the opaque "API returned an empty or malformed response
  (HTTP 200)". This covered both a non-2xx response (e.g. the plan hit its usage limit, HTTP
  429) and a turn that failed mid-stream on an HTTP 200 response (e.g.
  `context_length_exceeded` when a resumed transcript is larger than the target model's
  context window). The failure now comes back with a real status code and the provider's
  message, including when a rate-limited plan resets, so `--resume` and normal turns report
  what actually went wrong.
- **Breakdown occupancy uses each model's real context window.** The context-occupancy view
  guessed the window by model family (1M for anything gpt-5/codex, 200k for Claude), which was
  wrong for many models. It now reads the per-model window from a curated models.dev snapshot
  (`tools/refresh_context.py`, refreshed at release), so the occupancy percentage is accurate.
  `LLMTRIM_BREAKDOWN_WINDOW` still overrides, and the Claude `[1m]` beta still reports 1M.

### Added

- **Subscription reroute retries transient failures.** A rerouted turn that hits a transient
  upstream failure (HTTP 500/502/503/504/529) or a short rate limit is now re-issued with
  exponential backoff (up to 3 attempts), honoring the provider's reset hint. A usage-limit
  reset that is hours away is surfaced at once rather than burning pointless retries. The
  surfaced error is also typed by status (`rate_limit_error` / `authentication_error` /
  `overloaded_error`) and, for rate limits, carries a capped `Retry-After` header so the
  client backs off instead of tight-retrying a large resumed request.

## [0.9.0] - 2026-07-09

### Added

- **Subscription reroute (`llmtrim sub`).** The proxy can now serve Claude Code (and any
  Anthropic `/v1/messages` client) from a different subscription's backend instead of
  Anthropic: a ChatGPT plan through the Codex Responses API, or a Kimi coding plan. llmtrim
  rewrites each request to the provider's wire format, streams the reply back as Anthropic
  SSE, and maps the Claude model tiers (opus/sonnet/haiku/fable) to provider models per a
  saved config. `llmtrim sub auth codex login` / `llmtrim sub auth kimi login` sign in (OAuth PKCE or
  device code, tokens stored at `~/.llmtrim/<provider>/auth.json`, mode 0600); `llmtrim sub
  setup` opens an interactive editor for the tier mapping; `llmtrim sub use codex` applies the
  built-in preset; `llmtrim sub off` disables it. Using a subscription this way may violate
  the provider's terms of service, so the login command warns before it stores a token.
- **Web search works on the Codex reroute.** The hosted web-search tool was dropped from the
  translated request, so a Claude Code turn that wanted to search silently couldn't. It now maps to
  the Codex `web_search` tool (with any domain filters), and the results reach the answer.
- **Hallucinated `Read` tool calls are sanitized on the Codex reroute.** A Codex model sometimes
  emits a `Read` with an absurd `offset` (millions of lines) or an empty `pages`, which made Claude
  Code's Read fail. Those args are now stripped before the call reaches the client, and the next
  turn is told the offset was ignored so the model stops re-issuing it.
- **Codex reasoning effort.** Codex reasoning on the reroute now follows the effort Claude Code
  requests per turn (its `output_config.effort`), streamed back as Anthropic `thinking` blocks, so
  reasoning is adaptive by default. `llmtrim sub effort <none|low|medium|high|xhigh>` (env
  `LLMTRIM_CODEX_EFFORT` or `sub.codex.effort`) is an optional override that forces one effort on
  every rerouted turn. Kimi ignores it.
- **On-error reroute mode (`llmtrim sub mode on-error`).** Instead of rerouting every turn, the
  proxy can forward to Anthropic as usual and only fall back to the subscription provider when
  Anthropic answers with a usage-limit or overload status (402/403/429/529). `always` (the
  default) keeps rerouting everything. Set it with `llmtrim sub mode <always|on-error>` or the
  env var `LLMTRIM_SUB_MODE`.
- **Non-interactive reroute controls for scripts and the tray.** `llmtrim sub map <provider>
  <from> <to>` sets one mapping entry without the editor, where `from` is a Claude tier
  (`opus`/`sonnet`/`haiku`/`fable`) or an exact incoming model id, so any Anthropic-speaking IDE
  can be remapped model by model; `llmtrim sub unmap` removes one. `llmtrim sub models <provider>
  --json` lists the candidate provider models, `llmtrim sub status --json` reports the full
  reroute state (provider, mode, mapping, auth), and `llmtrim sub auth <codex|kimi> status --json`
  reports login state. `llmtrim status` now shows the active reroute provider and mode.

### Changed

- `llmtrim stop` no longer breaks tools in new terminals. On POSIX the shell-profile block now exports `HTTPS_PROXY` only while the interceptor is running, so a terminal opened after `stop` talks to LLM hosts directly instead of failing at a dead proxy, and new terminals re-wire on their own once you `start` again. On Windows, `stop` clears the interceptor values from `HKCU\Environment` and `start` re-sets them, so newly-launched terminals and apps stay in sync. The shell you run `stop` in keeps the vars it already inherited (a process can't rewrite its parent shell), so `stop` prints the one-line `unset` for it. Existing installs pick up the new behavior the next time the daemon starts, with no `setup` re-run.

### Fixed

- The Python package build was broken on Linux (`No UniFFI metadata found`), so `pip install llmtrim` had no Linux wheel.
- The reply-language clause no longer misfires when a non-English prompt carries pasted code. Code is stripped from the language-detection sample, and the sample now favors the most recent question over an earlier pasted context, so a large code block ahead of a short non-English question no longer causes the model to answer in English.
- Native TLS tools that ignore `NODE_EXTRA_CA_CERTS` (curl, git, Python, and rustls-based agents like OpenAI Codex) failed with `invalid peer certificate: UnknownIssuer` against the proxy. `setup` now also builds `~/.llmtrim/ca-bundle.pem` (the OS root bundle plus the llmtrim CA) and exports `SSL_CERT_FILE`/`CURL_CA_BUNDLE` on POSIX, so those clients trust the interceptor out of the box.
- OpenAI Codex hung for several seconds retrying a WebSocket connection through the proxy before falling back to plain HTTPS. The proxy now refuses WebSocket upgrades to intercepted hosts with an immediate `426`, so the client drops straight to the HTTPS transport — which, unlike a WebSocket, carries the prompt as a request body llmtrim can compress.

## [0.8.1] - 2026-07-07

### Changed

- The anti-overthinking directive now reaches Claude Code and other agent harnesses. It fires for any model the models.dev registry marks as a reasoning model, detected from the model id so it works even when the client sends no `thinking`/`reasoning` field on the wire. It also rides tool-call-shaped agent turns now, not just prose, since a thinking model runs a chain of thought before its tool calls too. On the agent path it injects first-turn-only, so it shares the loop's cached prefix without churning it.
- Terse output shaping now applies to the first turn of an agent loop (Claude Code and other tool-using harnesses), where the model still emits real prose to shape. Later tool-call turns stay unshaped, so the cached prefix is untouched. The reply-language clause that keeps answers in the user's language now rides any injected directive, not only terse, so a non-English conversation is preserved even when only the frugality or anti-overthink directive fires.

## [0.8.0] - 2026-07-06

### Added
- **Anti-overthinking directive (`output_anti_overthink`).** On prose requests that explicitly
  declare both a quantized serving tier (OpenRouter `provider.quantizations`) and a reasoning
  pass, `auto` now asks the model to commit to its answer instead of restarting mid-reasoning
  ("Wait", "But", "Actually…") — the failure mode quantization amplifies in reasoning models
  (arXiv:2606.00206). On in `aggressive`/`rag`/`code`/`agent`; `safe`/`lossless`/`frugal` stay
  untouched. Bench (`llmtrim bench quality`, this lever isolated, `openai/gpt-oss-20b` via
  OpenRouter, n=40 real GSM8K problems): −62.0% output tokens, quality retention +0.0pp
  (unchanged).

### Fixed
- **`RetrieveStage` no longer mistakes a tool result for the live question.** On
  Anthropic and Gemini requests, tool output (`tool_result` / `functionResponse`)
  rides on a synthetic `role: "user"` message (neither provider has a distinct tool
  role). When it landed on the newest turn — e.g. right after a large Skill
  invocation or function call — retrieval treated it as the live question: folded
  into the BM25 query (self-referential, near-zero pruning) and pinned verbatim
  instead of pruned. It's now excluded from that classification on both providers, so
  large tool results are pruned like any other bulk context.

## [0.7.0] - 2026-07-05

### Added
- **Agent-loop frugality directive (`output_frugal_tools`).** On tool-call-shaped requests —
  the one shape prose output-shaping skips — it injects a short directive steering the agent
  toward the fewest tool-use turns (batch independent calls into one turn, don't repeat a call).
  On in the `agent` preset, so `auto` applies it to tool-call traffic; `safe`/`lossless` stay
  untouched. It fires only on the first tool-call turn of a task (first-turn-only, idempotent),
  so it costs about one short injection with no per-turn cache churn. It is also model-gated:
  only models capable enough to act on the steer get it (measured against an embedded LMArena
  leaderboard snapshot, threshold just below Claude Sonnet 5); cheaper models that ignore it are
  skipped so they don't pay the directive's cost, and an unknown model still gets it. The effect
  is tail insurance, not a median saver: the agent A/B is a median wash but it caps the worst-case
  exploration blow-ups, with no task-success regression. Set `output_frugal_tools = false` to
  opt out. Preset `frugal` isolates the directive alone (no compression) for benchmarking.

## [0.6.3] - 2026-07-04

### Fixed
- **The `llmtrim` and `llmtrim-ledger` crates on crates.io catch up to the current
  version.** The 0.6.2 crates.io publish stopped after `llmtrim-core` because
  `llmtrim-ledger` was missing its Trusted Publisher, leaving `llmtrim` and
  `llmtrim-ledger` stuck at 0.6.1 on crates.io (the binaries, installers, npm, Homebrew,
  Scoop, and Docker image all shipped 0.6.2 normally). This release completes the
  crates.io publish so `cargo install llmtrim` serves the current version again.

## [0.6.2] - 2026-07-04

### Added
- **The Windows tray now ships as an installer.** The release attaches a `-setup.exe`
  NSIS installer for the desktop tray that bootstraps the Edge WebView2 runtime on
  install (preinstalled on Windows 11 and recent Windows 10, absent on older setups).

### Fixed
- **The Windows tray no longer shows a black window.** Vite stamped a `crossorigin`
  attribute on the tray's bundled `<script>` and `<link>` tags. On Windows the webview
  serves the app from `http://tauri.localhost` with no `Access-Control-Allow-Origin`, so
  the CORS check blocked the JS and CSS and nothing rendered (macOS/Linux use the
  `tauri://` scheme, which is unaffected). The build now strips the attribute.
- **The Windows tray no longer flashes a console window every few seconds.** The tray
  polls the `llmtrim` CLI (`status --json`) on each refresh; on Windows those subprocess
  spawns opened a console window each tick. The tray now spawns the CLI with
  `CREATE_NO_WINDOW`, so no terminal appears.

## [0.6.1] - 2026-07-04

### Fixed
- **Prebuilt binaries and the desktop tray downloads ship again.** The 0.6.0 release
  built each archive with `--bin llmtrim-tray` but no matching `--package`, and the tray
  crate sits outside the workspace `default-members`, so every tray target failed. No
  prebuilt binaries, npm / Homebrew / Scoop packages, Docker image, or tray downloads
  were published for 0.6.0 (installing from crates.io with `cargo install llmtrim` still
  worked). This restores the full binary and tray distribution.

## [0.6.0] - 2026-07-04

### Added
- **A desktop tray app shows per-agent savings at a glance.** A menu-bar (macOS),
  system-tray (Windows), or AppIndicator (Linux) popover reports llmtrim's aggregate
  compression savings, a per-agent breakdown, and a savings trend, refreshed on a
  configurable interval. It reads the same ledger the proxy writes and exposes only
  aggregate numbers. Start and stop the proxy and enable launch-at-login from the
  tray or from the CLI (`llmtrim tray`, `llmtrim setup`). Launching it again (from the
  dashboard, the CLI, or autostart) focuses the running instance instead of opening a
  second icon. It ships in the Homebrew,
  Scoop, and npm packages alongside the CLI; the Linux build is a separate download
  on the GitHub Release and needs `libwebkit2gtk-4.1` and `libayatana-appindicator3`.
- **`llmtrim setup` offers to install the tray** and to enable launch-at-login for it.
- **`llmtrim start` points you at the tray once.** After an update lands the tray next to
  the CLI, the first start prints a one-time hint to run `llmtrim tray`. Installs without
  the tray binary (a plain `cargo install`) stay silent.
- **The `llmtrim status` dashboard opens the tray with `y`.** When the tray is installed, the
  live dashboard shows a `y tray` action (in the footer and the `?` help) that launches it in
  the background while the dashboard stays open.

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

[Unreleased]: https://github.com/fkiene/llmtrim/compare/v0.11.5...HEAD
[0.11.5]: https://github.com/fkiene/llmtrim/compare/v0.11.4...v0.11.5
[0.11.4]: https://github.com/fkiene/llmtrim/compare/v0.11.3...v0.11.4
[0.11.3]: https://github.com/fkiene/llmtrim/compare/v0.11.2...v0.11.3
[0.11.2]: https://github.com/fkiene/llmtrim/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/fkiene/llmtrim/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/fkiene/llmtrim/compare/v0.10.2...v0.11.0
[0.10.2]: https://github.com/fkiene/llmtrim/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/fkiene/llmtrim/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/fkiene/llmtrim/compare/v0.9.4...v0.10.0
[0.9.4]: https://github.com/fkiene/llmtrim/compare/v0.9.3...v0.9.4
[0.9.3]: https://github.com/fkiene/llmtrim/compare/v0.9.2...v0.9.3
[0.9.2]: https://github.com/fkiene/llmtrim/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/fkiene/llmtrim/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/fkiene/llmtrim/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/fkiene/llmtrim/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/fkiene/llmtrim/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/fkiene/llmtrim/compare/v0.6.3...v0.7.0
[0.6.3]: https://github.com/fkiene/llmtrim/compare/v0.6.2...v0.6.3
[0.6.2]: https://github.com/fkiene/llmtrim/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/fkiene/llmtrim/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/fkiene/llmtrim/compare/v0.5.0...v0.6.0
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
