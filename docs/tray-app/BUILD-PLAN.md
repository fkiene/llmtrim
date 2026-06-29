# llmtrim Tray App — Multi-Agent Build Plan (v3, post expert review)

**Branch:** `feat/tray-app`
**Scope (phase 1):** macOS + Windows only. Linux deferred (tray DE quirks).
**Orchestration:** Claude (this session) is *chef d'orchestre* — dispatches each step to a
sub-agent, reviews the result against the step's exit gate, and only then unblocks the next
step. Build agents are **Sonnet (high effort)**. The **UI/UX designer is the only Opus
agent** and runs through **`/impeccable`**.

The target is a system-tray (menu-bar) desktop app that shows **compression per agent**
(Claude Code, Codex, Gemini, ...) — the same visual caliber as the OpenUsage reference in
`docs/image copy 3.png`: rounded card per agent, gradient progress bars, a usage-trend
sparkline, a settings footer, and a live "next update in Ns" line.

> **v2 changelog.** This revision incorporates four expert Sonnet reviews (architect,
> Tauri/desktop, data/backend, CI/release). The material changes from v1, with the reason:
> - **New `llmtrim-ledger` crate** instead of the tray depending on `llmtrim-core` or
>   `llmtrim-cli`. `Tracker`/`BreakdownDb` live in `llmtrim-cli`; depending on it would drag
>   the whole MITM proxy stack into a tray app. (architect §Risk-1, data §Finding-5)
> - **`saved_pct` is the gross ratio**, not "compressible surface." `breakdown_turns` has no
>   `frozen_input_tokens` column, so the honest compressible-surface figure is *undeliverable*
>   query-only — it would need a schema + proxy-write change. v1 overpromised. (data §Finding-1,
>   reconciling architect §Risk-4 which assumed the data was present)
> - **Query lives in `BreakdownDb` (the breakdown layer), not `Tracker`.** Per-agent data is
>   in `breakdown_turns`; `Tracker::by_period` reads `compressions` which has no `agent`
>   column. (data §Finding-4/5)
> - **`bill_micros` (integer) crosses the wire, USD formatted in the frontend** — drift-free
>   sums. (data §Finding-2)
> - **`default-members` excludes the tray; separate `tray.yml` workflow** so ubuntu CI never
>   compiles Tauri and a tray build failure never blocks the crates.io release. (CI §Risk-1/3)
> - **`[lints]` override** in the tray crate (drop `unsafe_code = "forbid"`), mirroring
>   `llmtrim-uniffi`. (CI §Risk-2)
> - **macOS menubar-app behaviors** are now explicit Step-B gates: hide Dock icon, blur
>   auto-hide, tray-click debounce, Windows `skip_taskbar`. (Tauri §gap-1/2)
> - **Unsigned-artifact reality** (Gatekeeper / SmartScreen) documented in Step E. (Tauri §4,
>   CI §Risk-6)
>
> **v3 adds two more reviews (security/privacy, QA/test-strategy):**
> - **Locked Tauri capability allowlist + strict CSP** — no `shell`/`fs`/`http`/`updater`
>   permissions; `connect-src 'none'`. The default scaffold is permissive enough to be an RCE
>   path. (security §Risk-1/2)
> - **`AgentAggregate` is structurally incapable of carrying `project`/`session_name`** — the
>   ledger holds absolute project paths + human session names; aggregate-by-agent already drops
>   them, now enforced at the type level + grep gate. No raw prompt text, no API keys exist in
>   the schema (confirmed). (security §Risk-3/6)
> - **Read-only ledger open that skips `migrate()`** (`SQLITE_OPEN_READ_ONLY`); proxy starts
>   first and owns DDL, so the tray never takes an exclusive lock. (security §Risk-4)
> - **`build_dashboard()` is a pure function in `llmtrim-ledger`, not the tray** — otherwise the
>   app's core mapping logic sits in the coverage-excluded crate and is tested nowhere. The
>   Tauri command becomes a one-line real-IO wrapper (legitimately uncovered). (QA §Gap-3)
> - **Real edge-case + `insta` snapshot tests** for the queries (empty / pre-meter NULL /
>   trend bucket boundaries / multi-agent / i64 overflow / Unicode title-case / zero-before),
>   and A0 proves behavior preservation by **moving the tests with the code** + a pre-extraction
>   snapshot anchor. (QA §Gap-1/2/7)
> - **Honest gate labeling** — the Step C "matches reference" check is human sign-off, and the
>   Step E E2E becomes an `assert_snapshot!` over `build_dashboard` JSON + a manually committed
>   screenshot (tauri-driver automation is out of phase-1 scope). (QA §Gap-5/6)

---

## 0. Non-negotiables (apply to every agent)

- **Reuse the data layer; do not reinvent aggregation.** Numbers come from the existing
  `breakdown_turns` ledger via the new `llmtrim-ledger` crate (extracted from
  `crates/llmtrim-cli/src/tracking.rs` + `src/breakdown/db.rs`). The new per-agent query
  reuses the `sessions()` query shape (`breakdown/db.rs`) with `GROUP BY agent`.
- **New crate, not a filter.** GUI app. The CLI's "no async / <10 ms startup" rules do **not**
  apply, but the tray must never block or crash the proxy. Tray is **read-only** on the
  ledger; the proxy stays the sole writer (SQLite WAL makes concurrent reads safe).
- **Every outward string / `.md` runs through `/avoid-ai-writing:avoid-ai-writing`.**
- **Each step has a machine-checkable exit gate.** No step is "done" until its gate passes
  and the orchestrator has reviewed the diff.
- **Honest metrics.** `saved_pct` is the gross input ratio and must be labeled as such — the
  term "compressible surface" is reserved in this codebase for the frozen-zone-aware figure
  and must not be used for the tray number.

---

## 1. Stack & crate topology (locked)

**Tauri v2** + a web frontend (Vite + TS).

### Crate changes

```
crates/
  llmtrim-core/      (unchanged, published)
  llmtrim-cli/       (depends on llmtrim-ledger; proxy + CLI, published)
  llmtrim-ledger/    NEW — Tracker, BreakdownDb, the row structs, the per-agent query.
                     Pure Rust + rusqlite. Published? -> publish = false for now (phase 1);
                     promote later if stable. Both cli and tray depend on it.
  llmtrim-tray/      NEW — Tauri app. publish = false. Depends on llmtrim-ledger only.
  llmtrim-uniffi/    (unchanged, publish = false)
  llmtrim-wasm/      (unchanged)
```

Extracting `llmtrim-ledger` is the clean reconciliation of the architect's "don't depend on
cli" with the data reviewer's "the query belongs in `BreakdownDb`." `BreakdownDb` moves into
the new crate; `llmtrim-cli` keeps working by depending on it. This is a **prerequisite
refactor (Step A0)** before any tray code.

### Root `Cargo.toml`

```toml
[workspace]
members = [
  "crates/llmtrim-core", "crates/llmtrim-cli", "crates/llmtrim-ledger",
  "crates/llmtrim-uniffi", "crates/llmtrim-wasm", "crates/llmtrim-tray",
]
default-members = [   # bare `cargo build/check/test` skips the Tauri crate
  "crates/llmtrim-core", "crates/llmtrim-cli", "crates/llmtrim-ledger",
  "crates/llmtrim-uniffi", "crates/llmtrim-wasm",
]
```

`llmtrim-tray/Cargo.toml` re-declares `[lints]` dropping `unsafe_code = "forbid"` (mirror
`llmtrim-uniffi`), since Tauri macro-generated code uses `unsafe`.

Rejected: egui/eframe (harder to hit the rounded-card/gradient look), Electron (no Rust reuse).

---

## 2. Data contract (designed first, before any UI)

One read model. Backend returns integers/raw values; the frontend formats.

```ts
interface AgentCard {
  agent: string;            // "claude-code" | "codex" | "gemini" | ...
  display_name: string;     // "Claude Code"; unknown ids -> Unicode-aware title-case
  // Compression headline (OUR metric; gross input savings ratio):
  input_before: number;
  input_after: number;
  saved_pct: number;        // gross: max(0, before-after)/before*100; 0 when before==0
  has_savings_data: boolean;// false when all turns predate input_before/after columns
  // Spend / cache (already reconciled in the ledger):
  bill_micros: number;      // integer micro-USD; frontend divides by 1_000_000
  cache_read_tokens: number;
  // Sparkline: raw saved_pct per Period bucket (NOT pre-normalized; frontend scales).
  trend: number[];
  last_event_ts: string | null;
}

interface Dashboard {
  cards: AgentCard[];
  totals: { input_before: number; input_after: number; saved_pct: number; bill_micros: number };
  generated_at: string;     // rfc3339
  next_update_secs: number; // drives "Next update in Ns"
}
```

Dropped from v1: `plan_label` (no source in the ledger; `BreakdownTurn` has only
`provider`/`model`). If desired later, derive it from `(provider, model)` in the display-name
map. `bill_usd` → `bill_micros`. `saved_pct` reworded to gross.

Backend (Tauri commands): `get_dashboard() -> Dashboard`, `get_agent_trend(agent, period)
-> number[]`, `open_settings()`, `set_poll_interval(secs)`, `quit()`.

Ledger surface (in `llmtrim-ledger`, on `BreakdownDb`):
- `agent_aggregates() -> Vec<AgentAggregate>` — `sessions()` query shape, `GROUP BY agent`:
  `SUM(input_before)`, `SUM(input_after)`, `SUM(bill_micros)`, `SUM(cache_read)`, `MAX(ts)`,
  with `COALESCE(..., 0)` for nullable `input_before/after`. **`AgentAggregate` must NOT have
  `project` or `session_name` fields** — they are absolute filesystem paths + human session
  names (`breakdown_turns`, `tracking.rs:57-58`); the type is structurally incapable of
  leaking them into the webview. (security §Risk-3)
- `agent_trend(agent, period, buckets) -> Vec<PeriodSaved>` — **new** query over
  `breakdown_turns` (NOT `Tracker::by_period`, which reads `compressions` and has no `agent`),
  using `Period::sql_bucket()` with `WHERE agent = ?`, group by bucket. `idx_breakdown_turns_agent`
  keeps it efficient.
- `build_dashboard(rows, generated_at, poll_secs) -> Dashboard` — **pure function**, no DB, no
  Tauri. Owns the contract mapping, `display_name` title-casing, `has_savings_data`, and totals.
  Lives here (not in the tray) so the coverage gate actually exercises it. The Tauri
  `get_dashboard` command is then a one-liner wrapper (real-IO, legitimately uncovered).
  (QA §Gap-3)
- **Read path:** a `open_readonly()` that opens with `OpenFlags::SQLITE_OPEN_READ_ONLY` and
  **skips `migrate()`** (skip if expected tables exist). The proxy starts first and owns DDL;
  the tray must never be the first opener or take an exclusive lock. (security §Risk-4)

---

## 3. Agent roster

| Agent | Model / mode | Responsibility |
|---|---|---|
| **A0 — Ledger extraction** | Sonnet (high) | Extract `llmtrim-ledger` crate (Tracker, BreakdownDb, structs); repoint `llmtrim-cli`; zero behavior change. |
| **A — Per-agent query** | Sonnet (high) | `agent_aggregates` + `agent_trend` + pure `build_dashboard` on `BreakdownDb`; `open_readonly`; full edge + snapshot tests. |
| **B — Tauri shell** | Sonnet (high) | Scaffold `llmtrim-tray`; tray icon + popover; macOS menubar text + menubar-app behaviors; capability allowlist + CSP; command wiring; `[lints]` override. |
| **C — UI/UX designer** | **Opus, via `/impeccable`** | Frontend to reference caliber: cards, gradient bars, SVG sparkline, settings footer, light/dark, motion, empty/loading states. The ONLY Opus agent. |
| **D — Packaging/CI** | Sonnet (high) | `tauri.conf.json` bundle; **new `tray.yml`** workflow (mac+Windows, no `needs:` on release.yml); `default-members` + docs `--exclude`; signing as optional secrets. |
| **E — Verification** | Sonnet (high) | E2E smoke + screenshot; unsigned-artifact docs (Gatekeeper/SmartScreen); changelog; gate enforcement. |

---

## 4. Execution graph (with exit gates)

Dependencies: **A0 → A → B → C → E**; **D** runs parallel after B; **E** closes.

### Step A0 — Ledger extraction (Sonnet high)  *(prerequisite refactor)*
- Move `Tracker`, `BreakdownDb`, and the row structs into `crates/llmtrim-ledger`. **Move the
  `#[cfg(test)]` blocks with the code** (the cli suite shrinks by exactly the moved tests).
  Repoint `llmtrim-cli` imports. No new behavior.
- **Before extraction:** add an `insta::assert_snapshot!` over `BreakdownDb::sessions()` on a
  seeded `:memory:` DB as the regression anchor. (QA §Gap-1)
- **Gate:** full workspace `cargo nextest run --features intercept,mcp` green; the
  pre-extraction `sessions()` snapshot still passes (byte-identical proof); total test count
  unchanged (relocated, not lost); `cargo build -p llmtrim-cli` unchanged; `cargo llvm-cov
  --features intercept,mcp -p llmtrim-ledger --show-missing-lines` reviewed per-file, gaps
  justified.

### Step A — Per-agent query (Sonnet high)
- Implement `agent_aggregates`, `agent_trend`, and the pure `build_dashboard`; `display_name`
  map (unknown agents → Unicode-aware title-case, per CLAUDE.md §5). Handle pre-meter NULLs →
  `has_savings_data=false`, empty ledger → `cards: []` + zero totals (no divide-by-zero).
- **Fixtures:** seeded `:memory:` DBs (real shapes, not synthetic stubs — cli-testing.md), in
  `crates/llmtrim-ledger/tests/fixtures/`.
- **Gate (fixture-based tests for each, + `insta` snapshot over serialized output):**
  `saved_pct` == gross formula; **empty ledger** → `[]`; **pre-meter NULL** rows →
  `has_savings_data=false` & `saved_pct=0`; **`input_before==0`** guard; **multi-agent** GROUP BY
  no cross-contamination; **`agent_trend` Day/Week/Month bucket boundaries** (week Monday edge);
  **i64 `bill_micros` sum** near-overflow fixture or documented headroom; **Unicode** agent id
  title-cased; `bill_micros` summed in SQL (not rate-recomputed). `cargo llvm-cov -p
  llmtrim-ledger` per-file reviewed. **Security:** `grep -r 'project\|session_name'
  crates/llmtrim-tray crates/llmtrim-ledger/src` shows no leak of those columns into the contract.

### Step B — Tauri shell (Sonnet high)
- New crate `llmtrim-tray` (member, NOT in `default-members`, `publish = false`, `[lints]`
  override). Tray icon + click-to-toggle popover via `TrayIconBuilder` +
  `tauri-plugin-positioner` (`TrayCenter`). macOS menubar title shows aggregate `% saved` via
  `set_title()`; Windows = icon + tooltip (no inline text).
- **macOS menubar-app behaviors (explicit gates):** `setDockVisibility(false)` in `setup()`;
  `onFocusChanged` blur → `hide()`; tray-click debounce (don't re-open while closing).
  **Windows:** `skip_taskbar: true` on the popover window.
- Commands from §2 wired to Step A (each a thin wrapper over `build_dashboard`/`agent_trend`);
  ledger opened via `open_readonly()`; poll timer emits a `dashboard` event.
- **Security (non-negotiable outputs, not "shoulds"):**
  - `src-tauri/capabilities/main.json` with an explicit allowlist — only the five
    `llmtrim-tray:*` commands + `core:default`. **No** `shell`/`fs`/`http`/`os`/`process`/
    `clipboard`/`global-shortcut`/`updater` permissions. (security §Risk-1)
  - `tauri.conf.json` `security.csp`: `default-src 'self'; script-src 'self'; style-src 'self'
    'unsafe-inline'; img-src 'self' data:; connect-src 'none'; font-src 'self' data:; object-src
    'none'; frame-src 'none'` — set **before** Step C so the UI agent builds within it.
    (security §Risk-2)
- macOS menubar-app behaviors (as before): `setDockVisibility(false)`, blur→`hide()`,
  tray-click debounce. Windows `skip_taskbar: true`.
- Ship `scripts/seed-dev-ledger.sh` (seed a temp/`:memory:` ledger) + documented manual
  verification steps, so the launch gate is reproducible. (QA §Gap-4)
- **Gate:** `cargo build -p llmtrim-tray` green on `macos-latest` + `windows-latest`; app
  launches, no Dock icon (mac), tray appears, popover toggles + auto-hides on blur,
  `get_dashboard` returns the seeded data; `cargo deny check` clean on tray deps;
  `grep -r '"shell"\|"fs:"\|"http"\|"updater"' src-tauri/capabilities/` empty;
  `tauri-plugin-updater` absent from `Cargo.toml`.

### Step C — UI/UX designer (Opus, `/impeccable`)
- Reproduce `docs/image copy 3.png` caliber, re-themed for **compression**: per-agent card,
  **"% saved" hero** with gradient bar (our metric), secondary rows (bill from
  `bill_micros/1e6`, cache reuse), **SVG-path sparkline** drawn in vanilla TS from `trend[]`
  (no chart framework; React only if justified), draggable card handle, agent glyph, settings
  footer with version + "Next update in Ns".
- Light + dark, reduced-motion fallback, **empty state** ("No traffic yet — start the proxy"),
  `has_savings_data=false` → show "—" not "0% saved", loading shimmer, responsive height.
- Binds only to the §2 contract via Tauri `invoke`/events, **within the Step-B CSP** (no remote
  fonts/CDN/inline `<script>`; data-URI SVG only).
- Minimal TS test story: a `tsc --strict` / `zod` check over a sample JSON blob matching the Rust
  `serde` output (catches contract drift at build time); unit tests for the sparkline's
  degenerate inputs (empty / single-point / all-zero `trend[]` → no NaN path). (QA §Gap-5)
- **Gate:** CSP lint passes (no `script src="http"` / `@import url(http` in the bundle); TS
  contract + sparkline tests green; dark mode + reduced-motion verified; no layout shift.
- **Human sign-off (not a machine gate):** side-by-side screenshot vs `docs/image copy 3.png`
  against the design checklist (spacing scale, radii, gradient, type ramp, contrast AA).
  (QA §Gap-5c)

### Step D — Packaging/CI (Sonnet high, parallel after B)
- `tauri.conf.json` targets: `.dmg` + `.app` (mac), `.msi` + NSIS (Windows). Add
  `default-members` to root `Cargo.toml`; add `--exclude llmtrim-tray` to the `docs` job's
  `--workspace` `cargo doc`.
- **New `tray.yml`** workflow on `push: tags: ['v*']`, jobs `build-mac` + `build-windows`, **no
  `needs:` to/from release.yml** (mirrors `publish-wasm` independence). Sign/notarize only when
  `APPLE_*` / `WINDOWS_CERTIFICATE` secrets present; else log "unsigned, dev use only" and
  continue. No change to `release.yml`'s `publish-crate` (already names only core + cli).
- `cargo deny check` runs in `tray.yml` (not just locally). Confirm `tauri-plugin-updater` is
  absent (phase 1). (security §Risk-5)
- **Gate:** `tray.yml` produces `.dmg` + `.msi` artifacts on both OSes; ubuntu CI jobs
  unaffected (verify coverage/test/docs still green); `cargo deny` clean workspace-wide.

### Step E — Verification + docs (Sonnet high)
- Automatable smoke: a Rust integration test calling `build_dashboard(seeded_rows, ...)` with
  `insta::assert_snapshot!` over the JSON output (fast, honest, no Tauri). A manually committed
  popover screenshot in `docs/tray-app/screenshots/` is the visual reference — reviewed once by a
  human, **not** CI-diffed. (tauri-driver E2E is out of phase-1 scope.) (QA §Gap-6)
- `docs/tray-app/README.md` (run, build, package) **including the unsigned-artifact steps**:
  macOS Gatekeeper (`xattr -dr com.apple.quarantine` / right-click → Open), Windows SmartScreen
  ("More info → Run anyway"), and the Windows hidden-tray-overflow note. `CHANGELOG.md`
  `[Unreleased]` entry.
- **Gate:** all prior gates green on a clean checkout; screenshot committed; docs + changelog
  pass `/avoid-ai-writing`.

---

## 5. Orchestration protocol (how I run it)

1. Dispatch one step's agent with a self-contained brief (contract excerpt + exit gate).
2. On return: read the diff, run the step's gate command myself, reject + redispatch with
   specific feedback if it fails. Never advance a dependent step on a red gate.
3. A0→A→B→C strictly serial. Launch D right after B, parallel with C. Run E last.
4. All work on `feat/tray-app` in this worktree. **No push, no PR** without explicit user
   go-ahead. One coherent commit per step; reshape before any eventual push.

---

## 6. Risks & calls (post-review)

- **Ledger extraction is the first real risk**, not the tray. A0 must be behavior-preserving;
  gate on identical test outcomes.
- **`saved_pct` is gross, by design** — honest compressible-surface needs `frozen_input_tokens`
  on `breakdown_turns` (schema + proxy write). Out of phase-1 scope; tracked as a follow-up.
- **Windows menubar text:** none (tooltip + popover only). macOS-only headline.
- **Unsigned artifacts** are dev-distribution only until signing secrets exist; Gatekeeper /
  SmartScreen block double-click launch. Documented, not hidden.
- **CI cache:** adding Tauri busts shared rust-cache keys once; `default-members` confines tray
  compilation to tray-only jobs thereafter.
- **Linux later:** Tauri abstracts the tray, but GNOME needs an extension; deferring costs
  nothing structurally.
- **Don't over-build:** phase 1 = popover + tray + per-agent cards + sparkline + settings. No
  history window, no export UI, no notifications (CLI already exports).
- **Security posture (phase 1):** locked capability allowlist + strict CSP (`connect-src
  'none'`) are the primary defense against a supply-chain inject in the Vite bundle reaching
  `shell:execute`. Ledger holds no secrets/prompt text; aggregate-by-agent + `AgentAggregate`
  type drop the only sensitive columns (`project` paths, `session_name`). Read-only,
  migrate-skipping open keeps the proxy the sole writer. No auto-updater this phase.

---

## 7. Definition of done (phase 1)

- `tray.yml` builds macOS `.dmg` + Windows `.msi`; ubuntu CI and the crates.io release path
  remain untouched and green.
- Tray icon → popover with one card per active agent, live gross `% saved` + SVG sparkline,
  matching the reference's visual caliber; macOS has no Dock icon and auto-hides on blur.
- Numbers verifiably equal the existing `breakdown_turns` aggregates.
- Docs (incl. unsigned-launch steps) + changelog landed; all gates green; branch ready for the
  user to review and push.
