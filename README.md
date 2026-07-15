<p align="center">
  <img src="assets/logo.png" alt="llmtrim" width="450">
</p>

<h1 align="center">llmtrim</h1>

<p align="center">
  <strong>Local proxy that compresses LLM API traffic so you pay less. Same answers, smaller bill.</strong>
</p>

<p align="center">
  <sub>
    <b>−31% input · −74% output · −66% round-trip cost</b>
    · 112 live A/B cases · ~5&nbsp;ms/call · no model to load
  </sub>
</p>

<p align="center">
  <sub>Proxy · CLI · MCP · library (Python · Ruby · Swift · Kotlin · JS/WASM)</sub>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="assets/status-watch-light.svg">
    <img src="assets/status-watch-dark.svg" alt="llmtrim status: a live dashboard showing tokens trimmed, dollars saved off your real bill, input/output savings bars, and a per-model breakdown" width="760">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml"><img src="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MPL--2.0-blue" alt="License: MPL 2.0"></a>
  <a href="https://crates.io/crates/llmtrim"><img src="https://img.shields.io/crates/v/llmtrim?logo=rust&cacheSeconds=3600" alt="crates.io"></a>
  <a href="https://www.npmjs.com/package/@llmtrim/cli"><img src="https://img.shields.io/npm/v/@llmtrim/cli?logo=npm&cacheSeconds=3600" alt="npm"></a>
  <a href="https://www.npmjs.com/package/@llmtrim/cli"><img src="https://img.shields.io/npm/dm/@llmtrim/cli?cacheSeconds=3600" alt="npm downloads"></a>
  <img src="https://img.shields.io/badge/rust-1.88%2B-orange" alt="Rust 1.88+">
</p>

<p align="center">
  <a href="#get-started">Install</a> &bull;
  <a href="#day-to-day">Day to day</a> &bull;
  <a href="#what-it-does">How it works</a> &bull;
  <a href="#works-with">Works with</a> &bull;
  <a href="#claude-code">Claude Code</a> &bull;
  <a href="#the-numbers">Numbers</a> &bull;
  <a href="#configuration">Config</a> &bull;
  <a href="#use-it-as-a-cli-mcp-or-library">CLI &amp; library</a>
</p>

---

## Get started

```bash
npm install -g @llmtrim/cli@latest && llmtrim setup
# open a new terminal, then keep working
llmtrim status
```

That's it. `setup` starts a local proxy, wires your shell, and (when Claude Code is present) turns on the status line, cold-cache guard, `/sub`, and cheaper `/compact`. You do not run a separate install for each of those.

| You want | Run |
|---|---|
| First install | `llmtrim setup` |
| New version | `llmtrim update` (then `llmtrim ensure` after npm/brew/cargo) |
| Something broken | `llmtrim ensure` · `llmtrim doctor --fix` · or **`f`** in `status` |

> Any tool that honors `HTTPS_PROXY` works (Claude Code, Codex, Cursor, Aider, your SDK). GitHub Copilot does not (certificate pinning). [Full list →](#works-with)

<details>
<summary><b>Other installers</b> (Homebrew, curl, Scoop, Cargo, Docker)</summary>

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex

# Package managers
brew install fkiene/tap/llmtrim
cargo binstall llmtrim
scoop install llmtrim
docker run -d -p 43117:43117 -v llmtrim-state:/data ghcr.io/fkiene/llmtrim
```

Full options: [INSTALL.md](INSTALL.md).

</details>

<details>
<summary><b>Desktop tray</b> (menu bar / system tray)</summary>

Menu-bar / system-tray popover with the same savings numbers. Bundled in Homebrew, Scoop, and npm; `setup` can enable open-at-login. Open with `llmtrim tray`. On Linux desktops, interactive `ensure` can fetch the tray binary from the [latest release](https://github.com/fkiene/llmtrim/releases) (needs `libwebkit2gtk-4.1` and `libayatana-appindicator3`).

<p align="center"><img src="crates/llmtrim-tray/docs/popover.svg" alt="llmtrim tray popover" width="320"></p>

</details>

<details>
<summary><b>Is this safe?</b></summary>

Same technique as [mitmproxy](https://mitmproxy.org), scoped to LLM API hosts only. `setup` changes three things; `llmtrim uninstall` reverses all three:

1. Private CA in `~/.llmtrim/` (name-constrained; cannot intercept your bank or email)
2. Shell env: `HTTPS_PROXY` + CA trust
3. Login service: daemon at login

No API keys stored (your tool's auth is forwarded). Prompts never touch disk; only anonymous token counts. Full threat model: [SECURITY.md](SECURITY.md).

```bash
llmtrim ca
openssl x509 -in ~/.llmtrim/ca.pem -noout -text | grep -A3 "Name Constraints"
```

</details>

---

## Day to day

```bash
llmtrim status     # savings + health  (aliases: monitor, gain)
llmtrim update     # new release, restart daemon, refresh integrations
llmtrim ensure     # match the recommended install state on this machine
```

| Situation | Command |
|---|---|
| Watch savings | `llmtrim status` |
| After `npm` / `brew` / `cargo` upgrade | `llmtrim ensure` (or **`f`** in status) |
| Diagnose | `llmtrim doctor` · repair with `doctor --fix` |
| Pause / resume proxy | `llmtrim stop` · `llmtrim start` |
| Force one session through llmtrim | `llmtrim wrap claude` |
| Remove everything | `llmtrim uninstall` |

After `setup`, `update`, or `ensure`, owned Claude Code pieces (status line, guard, `/sub`, compact defaults) stay in sync with the binary. You should not need `statusline install` or similar after an upgrade.

Time series: `llmtrim status --daily` · `--weekly` · `--monthly` · `--json` · `--csv`.

---

## What it does

You run Claude Code, Codex, Cursor, or your own app. Each request carries a large blob: system prompt, tools, history, and raw tool output. You pay for that on every turn.

A lot of it is noise. A 200-line build log with two errors. Tool schemas resent fifty times. JSON with hundreds of near-identical rows. The model does not need the bulk to answer well.

llmtrim strips that noise on your machine before the request hits the provider. Your tool stays the same, the reply stays the same, the bill shrinks.

```
  before:  your tool ───── full request ─────▶  OpenAI / Anthropic / …
                    ◀──────── reply ──────────

  after:   your tool ──▶ llmtrim ──smaller──▶  OpenAI / Anthropic / …
                            (on your machine)
                    ◀──────── reply ──────────  (same answer)
```

> [!IMPORTANT]
> Compression cannot raise your bill or break a request. Each step is re-measured with the provider's real tokenizer and undone if it does not save tokens. If the provider rejects the compressed body, the original is resent. Worst case is zero savings.

Everything runs locally. Nothing is sent to us.

### In action

Real agent build log: 58 lines, two of them errors. Keep the errors, drop the rest.

Before (4,662 chars) → after (978 chars, −79%):

```text
# before (noise + signal)
[2026-06-13T10:02:00Z] INFO  compiling module core::worker::task_0 (incremental)
… 28 more near-identical INFO lines …
[2026-06-13T10:02:31Z] ERROR src/worker/pool.rs:214: mismatched types: expected `usize`, found `i64`
… 25 more INFO lines …
[2026-06-13T10:03:01Z] ERROR src/net/conn.rs:88: cannot borrow `buf` as mutable more than once
[2026-06-13T10:03:02Z] INFO  build failed, 2 errors

# after (errors verbatim; INFO folded losslessly)
[{}] INFO compiling module core::worker::task_{} (incremental) [×30: (10:02:00Z..10:02:29Z step 1s; 0..29)]
[2026-06-13T10:02:31Z] ERROR src/worker/pool.rs:214: mismatched types: expected `usize`, found `i64`
[{}] INFO compiling module core::net::conn_{} (incremental) [×25: 10:02:32Z..10:02:56Z; 0..24]
[2026-06-13T10:03:01Z] ERROR src/net/conn.rs:88: cannot borrow `buf` as mutable more than once
[2026-06-13T10:03:02Z] INFO  build failed, 2 errors
```

Try any request body:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai
```

| Waste | What happens |
|---|---|
| Build logs, diffs, grep dumps | Keep errors / changes / matches; fold the rest |
| Long pasted context | Keep chunks relevant to the question |
| Source code | Keep useful bodies; rest → signatures |
| Tool schemas every turn | Trim + keep the cache prefix stable |
| Huge JSON arrays | Compact table (TOON) or sample |
| Verbose model replies | Ask for terser output where safe |

<details>
<summary><b>All 10 compressors</b></summary>

Stages run in savings order. Nothing under a `cache_control` marker is ever rewritten.

| Stage | What it does | When it runs |
|---|---|---|
| **tool-output** | Lossless template fold first, then window logs · diffs · grep · dumps down to errors / changes / matches | tool results |
| **cache discipline** | Mark + stabilize the invariant prefix (sort tools/schema · OpenAI `prompt_cache_key`) so it stays cached | tools |
| **lexical retrieval** | BM25+ ranking with RM3 feedback · TextTiling topic cuts · budgeted non-redundant selection; question protected | long context |
| **skeletonization** | tree-sitter keeps relevant function bodies, drops the rest to signatures (14 languages) | code |
| **serialize + hygiene** | Minify JSON, encode record arrays to [TOON](https://crates.io/crates/toon-format) or CSV, Unicode-normalize | always · lossless |
| **json sample** | Down-sample huge record arrays: first/last + outliers + a query-biased diverse sample | big JSON |
| **dedup** | Collapse duplicate + near-duplicate lines (prose only) | always |
| **output control** | Terse instruction · Chain-of-Draft · token budget · native JSON schema · anti-overthink directive (quantized reasoning) · agent-loop frugality directive | auto |
| **tool layer** | Static tool selection + description trimming | tools |
| **multimodal** | Downscale images to the provider's resolution cap | images |

Default `auto` enables each stage only where it pays. `safe` is lossless-only. [Config →](#configuration)

</details>

---

## Claude Code

When `~/.claude` exists, `setup`, `update`, and `ensure` wire these. No separate install commands.

| Feature | What you get |
|---|---|
| Status line | Model, context gauge, trim %, rate limits, cache warm/cold |
| Guard | Blocks one turn if a cold-cache resume would rewrite a huge context (and bill for it) |
| `/compact` models | Prefer Haiku → Sonnet before your selected model |
| `/sub` | Per-window: `/sub on [codex\|kimi\|grok]` · `/sub off` · `/sub status` |

```text
◆ Opus→gpt-5.6-terra   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 3h·24% · 4d·12%   ♻ 63% cached
```

<details>
<summary><b>Status line details</b></summary>

Claude Code [custom status line](https://code.claude.com/docs/en/statusline). The arrow is the backend that answered the last turn (not merely what is configured). In `sub` fallback mode it stays off while Anthropic serves and shows up when a chain provider does.

- `✂`: trim for this session (`✂ –` until something is saved)
- `◔`: Claude.ai rate-limit windows (time left · % used)
- Context gauge: fill of the serving model's real window (green under 40%, orange 40-65%, red above)
- `♻`: prompt-cache reuse; becomes `♻ cache cold` after the cache TTL

Owned settings rewrite themselves when the binary path or payload changes. To opt out, leave your own status line in place, or uninstall ours (`llmtrim statusline uninstall`).

</details>

<details>
<summary><b>Cold-cache guard</b></summary>

Resuming a large session after the prompt-cache TTL rewrites the whole context at cache-write rates (often a few dollars) with no warning at the prompt.

Guard is a free `UserPromptSubmit` hook. It blocks one turn, prints idle time, context size, and estimated cost, then lets a resend through. `/compact` pays that cold write too, because it has to read the full context to summarize.

Opt out: `llmtrim guard uninstall`. `ensure` remembers that choice.

```text
Idle 6h 19m, 347k tokens of context. The prompt cache has expired, so the next turn
rewrites the whole context (about $3.47 before any work happens).
```

</details>

<details>
<summary><b>Cheaper `/compact`</b></summary>

```bash
llmtrim compact models haiku sonnet   # setup already sets this by default
llmtrim compact status
llmtrim compact off
```

```toml
[compact]
models = ["haiku", "sonnet"]
```

Candidates run in order when they fit the compressed request. Claude's selected model is always the last fallback (do not put it in the list). Empty `models = []` records opt-out.

</details>

<details>
<summary><b>Subscription reroute (`sub`)</b> (opt-in; may conflict with provider ToS)</summary>

Serve Claude Code from a ChatGPT/Codex, Kimi, or SuperGrok plan instead of Anthropic, or as fallback when Anthropic fails. Login prints a warning; decide for yourself.

```bash
llmtrim sub auth codex login    # or: kimi / grok
llmtrim sub on codex            # or: grok
llmtrim sub status
llmtrim sub mode fallback       # only when Anthropic fails
llmtrim sub chain codex,kimi,grok
llmtrim sub off
```

Grok maps Claude tiers to `grok-4.5` (opus/sonnet/fable) and `grok-composer-2.5-fast` (haiku).

This window only (installed with ensure; includes subagents; survives `/clear`):

```text
/sub on                  # last provider for this window, or the global sub
/sub on codex
/sub on kimi
/sub on grok
/sub off
/sub status
```

Tokens: `~/.llmtrim/<provider>/auth.json` (mode 0600). Env: `LLMTRIM_SUB`, `LLMTRIM_SUB_MODE`, `LLMTRIM_SUB_CHAIN`. After upgrading, run **`llmtrim ensure`** (or **`f`** in `status`) so owned `/sub` hooks refresh and the daemon picks up the new binary (including `grok`).

</details>

---

## Use it as a CLI, MCP, or library

Same engine, no proxy required. No extra model calls; compress runs in-process.

| Language | Install |
|---|---|
| Rust | `cargo add llmtrim-core` |
| Python | `pip install llmtrim` |
| Ruby | `gem install llmtrim` |
| Kotlin | `implementation("io.github.fkiene:llmtrim:0.10.2")` |
| Swift | SwiftPM `fkiene/llmtrim-swift` ≥ 0.1.8 |
| JS / TS | `@llmtrim/js` (WASM) |

<details>
<summary><b>CLI pipe</b></summary>

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai > out.json
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim send --provider openai
```

</details>

<details>
<summary><b>Rust · Python · JS</b></summary>

```rust
use llmtrim_core::{compress, ir::ProviderKind};
let out = compress(request_json, Some(ProviderKind::OpenAi))?;
```

```python
import llmtrim
out = llmtrim.compress(request_json, llmtrim.Provider.OPEN_AI, "aggressive")
```

```ts
import { compress } from "@llmtrim/js";
const out = compress(requestJson, "openai", "aggressive");
```

Bindings and WASM notes: [`crates/llmtrim-uniffi`](crates/llmtrim-uniffi) · [`crates/llmtrim-wasm`](crates/llmtrim-wasm).

</details>

<details>
<summary><b>MCP server</b></summary>

```bash
llmtrim mcp install          # Claude Code
llmtrim mcp install --print  # paste into any client
```

```json
{
  "mcpServers": {
    "llmtrim": { "command": "llmtrim", "args": ["mcp"] }
  }
}
```

Tools: `llmtrim_compress`, `llmtrim_compress_text`, `llmtrim_stats` (same ledger as `status`).

</details>

## Works with

Any tool that honors `HTTPS_PROXY` and an env-provided CA:

| Tool | Works | Notes |
|---|:---:|---|
| Claude Code | ✅ | Prompt-cache discount stays intact |
| Codex CLI | ✅ | |
| Gemini CLI | ✅ | |
| Cursor (IDE), Cline, Roo, Kilo Code | ✅ | VS Code extensions; set `NODE_EXTRA_CA_CERTS` for the Node host process |
| Goose, OpenCode, Crush, Mux, Forge, OpenClaw, Pi/OMP | ✅ | CLI agents on standard provider hosts |
| Qwen Code, Grok CLI, Kimi Code, Mistral Vibe | ✅ | Provider hosts ship in the `llm_providers` registry, intercepted out of the box |
| Aider, any other `HTTPS_PROXY`-aware CLI | ✅ | |
| Hermes, Droid (BYOK mode) | ✅ | Interceptable only when a direct provider key is configured; see [guide](HERMES.md) for Hermes |
| Your own app / SDK | ✅ | Or call the [CLI / library](#use-it-as-a-cli-mcp-or-library) directly |
| GitHub Copilot | ❌ | Certificate pinning blocks interception |
| Warp, Devin | ❌ | Provider call is server-side; a local proxy never sees it |
| Cursor Agent, Kiro | ❌ | Routes through a vendor gateway, not a standard provider host |

No proxy: any MCP client can call llmtrim as tools (`llmtrim mcp install`), or use the [CLI / library](#use-it-as-a-cli-mcp-or-library).

Providers come from the [`llm_providers`](https://crates.io/crates/llm_providers) registry (OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI, Moonshot, Zhipu, Qwen, OpenRouter, …) and update with it. Non-LLM connections pass through untouched.

## Configuration

Default is fine for most traffic. `auto` inspects each request and picks compressors by shape (tools → `agent`, code → `code`, long Q&A → `rag`, else `aggressive`).

Override with `LLMTRIM_PRESET=<name>` or `preset = "<name>"` in `$XDG_CONFIG_HOME/llmtrim/config.toml`:

| preset | When to use |
| --- | --- |
| **`auto`** *(default)* | Let llmtrim choose per request |
| **`safe`** | Lossless input only |
| **`aggressive`** | Max squeeze, quality-gated |

<details>
<summary><b>Advanced presets</b></summary>

`auto` composes these per request shape, so most users never set them directly. Pick one when you know your traffic and want to skip shape detection:

| preset | for |
| --- | --- |
| `agent` | tool-calling loops: prunes the tool block first-turn-only so the prompt cache stays warm |
| `code` | coding turns: skeletonize and minify code, compress pasted logs and diffs |
| `rag` | long context with a question: sentence-level retrieval |
| `cache` | a fixed prefix reused across many calls |
| `reasoning` | math and step-by-step workloads |
| `frugal` | isolates the agent-loop frugality directive alone, for clean benchmarking |

</details>

<details>
<summary><b>Per-flag overrides (power users)</b></summary>

Every stage is individually tunable via config flags; `preset` wins over individual flags. The full table is long; see the field list in [`config.rs`](crates/llmtrim-core/src/config.rs) or run `llmtrim compress --help`. The most useful knobs:

| field | default | meaning |
| --- | --- | --- |
| `toolout` | on in `agent`/`aggressive` | tool-output compression (logs / diffs / grep / dumps) |
| `retrieve` | `false` | lexical retrieval for long context (lossy) |
| `skeletonize` | `false` | drop non-relevant function bodies to signatures |
| `serialize` | `true` | TOON / CSV encoding of record arrays |
| `json_crush` | on in `agent`/`aggressive` | sample huge record arrays |
| `output_control` | `false` | terse-output instruction + cap |
| `output_anti_overthink` | on in `aggressive`/`rag`/`code`/`agent` | commit-to-answer directive for quantized reasoning traffic |
| `output_frugal_tools` | on in `agent` | steers agent loops toward fewer tool-call turns (batch, don't repeat) |
| `cache` | `false` | `cache_control` breakpoints (lossless) |
| `dedup` | `true` | collapse duplicate lines (lossless) |
| `quality_gate` | `true` | revert any lossy cut whose query-relevant coverage drops too far |

Env: `LLMTRIM_PRESET` (preset), `LLMTRIM_CONFIG` (config-file path).

</details>

<details>
<summary><b>Runtime settings (env or config file)</b></summary>

These knobs are orthogonal to compression. Each resolves env-first, then from the config file, so set whichever fits. The env var wins when both are present.

| env var | config key | meaning |
| --- | --- | --- |
| `LLMTRIM_EXTRA_HOSTS` | `extra_hosts` | extra exact LLM-API hosts to intercept (comma-separated env / array in file), e.g. a self-hosted OpenAI-compatible endpoint |
| `LLMTRIM_EXCLUDE_PROVIDERS` | `exclude_providers` | wire shapes to skip compressing: `openai` / `anthropic` / `google` (e.g. `anthropic` to leave Claude Code untouched); coarse, covers every host of that shape |
| `LLMTRIM_EXCLUDE_HOSTS` | `exclude_hosts` | exact hostnames to skip compressing (e.g. `openrouter.ai`); precise, leaves other hosts of the same shape compressed |
| `LLMTRIM_UPSTREAM_PROXY` | `upstream_proxy` | route egress through another proxy (see below) |
| `LLMTRIM_DB_PATH` | `db_path` | ledger location |
| `LLMTRIM_CAPTURE_DIR` | `capture_dir` | before/after QA capture directory |
| `LLMTRIM_CAPTURE_MAX_MB` | `capture_max_mb` | capture corpus size ceiling (`0` disables) |
| `LLMTRIM_BIND` | `bind` | listen IP (default loopback; `0.0.0.0` for containers) |
| `LLMTRIM_BREAKDOWN_WINDOW` | `breakdown_window` | context-window override for the cost breakdown |
| `LLMTRIM_RETENTION_DAYS` | `retention_days` | ledger age-retention in days |
| `LLMTRIM_NO_UPDATE_CHECK` | `no_update_check` | disable the passive update check |

`extra_hosts` entries must be exact hostnames (`llm.acme.com`, never a bare `acme.com`): each one widens the name-constrained MITM CA, which regenerates automatically on the next launch to cover them.

</details>

Claude Code options (compact models, subscription reroute) are under
[Claude Code](#claude-code).

<details>
<summary><b>Upstream proxy</b> (corporate egress or chaining local tools)</summary>

```bash
export LLMTRIM_UPSTREAM_PROXY=http://host:port
# or with auth: http://user:pass@host:port  (redacted in logs)
```

Outbound calls use `CONNECT` + verifying TLS; the upstream only sees the encrypted stream.
Looping to llmtrim's own listen address is rejected. Put the variable in the **daemon's**
launch environment (launchd / systemd), not only your interactive shell. Profile secrets
sit in plaintext.

Companion tools on another port (e.g. [headroom](https://github.com/chopratejas/headroom)) are fine.

</details>

## The numbers

Every case is sent twice, once original and once compressed, then both answers are scored and billed at real rates. Cost and quality are measured together, not estimated, across 112 cases:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="assets/frontier-light.svg">
    <img src="assets/frontier-dark.svg" alt="llmtrim cuts the round-trip bill on both ends: $0.0365 original vs $0.0126 compressed, −66% cost, across 112 live A/B cases" width="840">
  </picture>
</p>

| | original | compressed | saved |
|---|--:|--:|--:|
| input tokens | 71,031 | 49,062 | **−31%** |
| output tokens | 25,843 | 6,628 | **−74%** |
| **round-trip cost** | **$0.0365** | **$0.0126** | **−66%** |
| answer quality | 78.9% | 82.2% | no measured degradation |

The token cuts are model-independent (−31% input, −74% output). The dollar saving tracks the model's output-to-input price ratio: −66% here, projecting to −57% at GPT-4o rates and −59% at Claude Sonnet rates. The proxy compresses only the new-content surface and never rewrites the cache-controlled prefix, so your prompt-cache discount survives.

<details>
<summary><b>Accuracy preserved on standard benchmarks</b></summary>

The same A/B on the standard academic suites, at a conservative shape-matched preset (`qwen3-next-80b`, paired 95% CI). Quality is the score on the original request vs the compressed one. GSM8K comes from the frontier above (n=12); the other three are the named benchmarks readers compare against (n=20 each):

| benchmark | task | scorer | input saved | quality (orig → comp) | retention |
|---|---|---|--:|:--:|--:|
| GSM8K | grade-school math | numeric-exact | −47%¹ | 100% → 92% | −8pp |
| TruthfulQA (MC1) | factual truthfulness | choice-exact | 0% | 75% → 75% | +0.0±0.0pp |
| SQuAD v2 | extractive QA | token-F1 / EM | 11% | 84% → 84% | −0.0±15.2pp |
| BFCL (live_multiple) | function calling | tool-call match | 33% | 95% → 95% | +0.0±15.2pp |

Three rows compress with no quality loss; GSM8K is the one dip:

- **BFCL** drops the tool schemas the query doesn't need (a menu of 2 to 37 candidates per call).
- **SQuAD v2** still answers its unanswerable questions correctly.
- **TruthfulQA** holds factual accuracy exactly: its ~75-token prompts are almost all answer text, so the safe preset finds nothing to cut.
- **GSM8K** trades −8pp of accuracy for −71% cost, so measure per workload before enabling its reasoning preset. ¹Its input goes negative because that preset injects a Chain-of-Draft instruction whose payoff is output-side (see the frontier table).

Evidence and a one-line reproduce ([named-benchmark snapshot](crates/llmtrim-cli/bench/snapshots/named-benchmarks/README.md)):

```bash
make -C crates/llmtrim-cli/bench data
(cd crates/llmtrim-cli && cargo run -q --features live -- bench quality \
   --corpus bench/data/squad2.jsonl --preset rag \
   --model qwen/qwen3-next-80b-a3b-instruct --route "" --n 20)
```

</details>

Methodology, per-corpus frontier, and confidence intervals: [crates/llmtrim-cli/bench/README.md](crates/llmtrim-cli/bench/README.md). Reproduce it:

```bash
make -C crates/llmtrim-cli/bench data   # pull real corpora (gsm8k, humaneval, dolly, hotpotqa, …)
(cd crates/llmtrim-cli && cargo run -q --features live -- bench suite)  # live A/B across all corpora (needs OPENROUTER_API_KEY)
(cd crates/llmtrim-cli/bench/scripts && PYTHONPATH=. python3 -m benchkit.tools.chart)  # regenerate the chart + table
```

## How it compares

Each tool compresses one slice of the request. llmtrim compresses input and output, leaves the cached prefix untouched to keep the prompt cache stable, and scores on whether the answer survives the cut, not on tokens removed. Both axes below use the `o200k_base` encoder and reproduce from this repo.

| | **llmtrim** | Headroom | RTK | caveman |
|---|:---:|:---:|:---:|:---:|
| Compresses | input · output | input | tool/CLI output | model output |
| Skips no-op transforms | ✅ | ❌ | ❌ | n/a |
| One static binary | ✅ | Python + models | ✅ | ✅ |

### Input

Input reduction (deterministic) next to answer quality from a live A/B. Quality is the drop vs llmtrim at each tool's compared setting (✅ held, a statistical tie; ❌ significantly lower), so a big reduction with a ❌ means the tool bought tokens by losing answers:

| Tool | Reduction | Quality vs llmtrim | Overhead |
|---|---:|---:|---:|
| llmtrim `auto` | 25% | ✅ ref | ~5 ms |
| llmtrim `aggressive` | 28% | ✅ ref | ~5 ms |
| Headroom (ML on) | 24% | ✅ tie | ~0.9 s |
| leanctx / LLMLingua-2 | 52-81% | ❌ 18% lower | ~6 s |
| entroly | 80-89% | ❌ 42% lower | <1 ms |

Overhead is the median per-call compress time (Python wall-clock, not like-for-like CPU): Headroom and leanctx run ML on CPU here (faster on a GPU) and pay a one-time model load on top (~3 s and ~4 s); llmtrim is Rust and entroly is lexical, so neither does.

- `auto` is the quality-gated default; `aggressive` accepts lossy cuts where the gate holds.
- Headroom drops to 0% with its ML disabled (its routers no-op on prose).
- leanctx and entroly are lossy with no quality gate; entroly has no low-reduction mode.

Headroom ties at matched reduction (24-25%, n=30, not significant) but its longer answers hit the model's output-token limit and get truncated 12 times to llmtrim's 2, the output inflation behind its higher cost. leanctx (measured at 26%) and entroly (at 69%, its mildest) score significantly lower than llmtrim (n=20), and fall further at their headline reductions ([vs-leanctx](crates/llmtrim-cli/bench/snapshots/vs-leanctx/README.md), [vs-entroly](crates/llmtrim-cli/bench/snapshots/vs-entroly/README.md)).

### Output

Output reduction by asking for terser responses, on a paid live call over 9 coding prompts:

| | output cut | overhead / request |
|---|---:|---:|
| caveman | 80% | 949 tokens |
| llmtrim `output_terse` | 69% | 19 tokens |

The cost is the 949-token system prompt caveman resends on every request (right column); llmtrim's is 19 for nearly the same cut. Both still net-save here, so caveman's deeper cut comes out ahead only when the output it removes is worth more than the 949 tokens it adds back ([vs-caveman artifact](crates/llmtrim-cli/bench/snapshots/vs-caveman/README.md)).

The tools stack: RTK shrinks CLI output, then llmtrim compresses the tool schemas on top. Full head-to-heads: [crates/llmtrim-cli/bench/README.md](crates/llmtrim-cli/bench/README.md).

## Known limits

These are surfaced by the same A/B that proves the savings:

- **Anthropic / Gemini token counts are approximate.** There's no public exact tokenizer, so a BPE proxy is used and flagged in `status`. OpenAI is exact.
- **Output savings aren't measured live.** The proxy compresses input; an output *saving* needs the A/B counterfactual, which only the offline benchmark runs. `status` "saved" is input-side.
- **The default is quality-gated, not lossless.** Lossy stages run only where the eval shows quality holds. Want a byte-faithful round-trip? Use the `safe` preset.
- **"Lossless" is input-side, not response restoration.** A lossless stage preserves the information the model reads (a folded log run, a TOON-encoded array, an abbreviation legend the model decodes in-prompt), and the token gate reverts any input cut that doesn't pay off. The engine does not transform the model's response back to an original form.

## Acknowledgments

Every compressor is a deterministic implementation of published research: the ideas are theirs, the engineering and the token gate are ours.

<details>
<summary><b>Papers + crates behind each stage</b></summary>

**Retrieval & context:** BM25 (Robertson & Zaragoza 2009, [`bm25`](https://crates.io/crates/bm25)); BM25+ (Lv & Zhai, CIKM 2011); RM3 (Lavrenko & Croft, SIGIR 2001); TextTiling (Hearst, CL 1997); TextRank (Mihalcea & Tarau, EMNLP 2004); MMR (Carbonell & Goldstein, SIGIR 1998); Submodular objective (Lin & Bilmes, ACL 2011); modified-greedy knapsack maximizer (Tang et al., SIGMETRICS 2021, [arXiv:2008.05391](https://arxiv.org/abs/2008.05391)); DPP diverse sampling (Chen et al., NeurIPS 2018); Lost in the Middle ([arXiv:2307.03172](https://arxiv.org/abs/2307.03172)); DSLR ([arXiv:2407.03627](https://arxiv.org/abs/2407.03627)).

**Code:** RepoCoder ([arXiv:2303.12570](https://arxiv.org/abs/2303.12570)); Hierarchical Context Pruning ([arXiv:2406.18294](https://arxiv.org/abs/2406.18294)); The Hidden Cost of Readability ([arXiv:2508.13666](https://arxiv.org/abs/2508.13666)); Minification token accounting ([arXiv:2606.01326](https://arxiv.org/abs/2606.01326)).

**Tool output:** Drain (He et al., ICWS 2017); Brain (Yu et al., IEEE TSC 2023); LogLSHD ([arXiv:2504.02172](https://arxiv.org/abs/2504.02172)).

**Dedup & abbreviation:** SimHash (Charikar, STOC 2002, [`gaoya`](https://crates.io/crates/gaoya)); CompactPrompt ([arXiv:2510.18043](https://arxiv.org/abs/2510.18043)); Maximal repeats ([arXiv:1304.0528](https://arxiv.org/abs/1304.0528)) + Re-Pair (Larsson & Moffat, DCC 1999).

**Output control:** Chain-of-Draft ([arXiv:2502.18600](https://arxiv.org/abs/2502.18600)); TALE ([arXiv:2412.18547](https://arxiv.org/abs/2412.18547)).

**Serialization:** [TOON](https://crates.io/crates/toon-format) (Token-Oriented Object Notation), Johann Schopplich.

Built on [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs), [`tree-sitter`](https://crates.io/crates/tree-sitter), [`image`](https://crates.io/crates/image), [`whatlang`](https://crates.io/crates/whatlang), [`hudsucker`](https://crates.io/crates/hudsucker), [`rusqlite`](https://crates.io/crates/rusqlite), and more.

</details>

## Found a problem?

```bash
llmtrim doctor          # diagnose
llmtrim doctor --fix     # diagnose + apply repairs
llmtrim ensure          # same repair path
```

Each failing check names its fix. If a request was mangled, set `LLMTRIM_CAPTURE_DIR` and
[open an issue](https://github.com/fkiene/llmtrim/issues) with the before/after pair.

If llmtrim saved you money, a ⭐ helps others find it.

## Star history

<a href="https://www.star-history.com/?repos=fkiene%2Fllmtrim&type=date&legend=top-left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=fkiene/llmtrim&type=date&theme=dark&legend=top-left" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=fkiene/llmtrim&type=date&legend=top-left" />
    <img alt="Star history chart for fkiene/llmtrim" src="https://api.star-history.com/chart?repos=fkiene/llmtrim&type=date&legend=top-left" />
  </picture>
</a>

---

<sub>Licensed under [MPL-2.0](LICENSE). Use llmtrim freely in your stack, including commercially, with no source-disclosure obligation for your own code; the file-level copyleft applies only to modifications you make to llmtrim's own source files. Contributions via [DCO](CONTRIBUTING.md#sign-your-commits-dco) sign-off.</sub>
</content>
</invoke>
