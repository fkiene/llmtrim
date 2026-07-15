<p align="center">
  <img src="assets/logo.png" alt="llmtrim" width="450">
</p>

<h1 align="center">llmtrim</h1>

<p align="center">
  <strong>llmtrim is a local proxy that compresses your LLM API requests so you pay less, with no change to the answers.</strong><br>
  It sits between your AI tools and the provider, strips the wasted tokens out of every request, and forwards it on. You get the same answers for a smaller bill.
</p>

<p align="center">
  <sub><b>−31% input and −74% output tokens</b>, measured live across 112 A/B cases, with no change in answer quality.</sub>
</p>

<p align="center">
  <sub>One static binary. <b>~5 ms per call</b>, no model to load.</sub>
</p>

<p align="center">
  <sub>Use it as a <b>proxy</b>, a <b>CLI</b>, an <b>MCP server</b>, or a <b>library</b> (Python · Ruby · Swift · Kotlin · JS · TS · WASM).</sub>
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
  <a href="#what-it-actually-does">What it does</a> &bull;
  <a href="#see-it-on-real-output">In action</a> &bull;
  <a href="#get-started">Get started</a> &bull;
  <a href="#use-it-as-a-cli-mcp-or-library">CLI &amp; library</a> &bull;
  <a href="#works-with">Works with</a> &bull;
  <a href="#the-numbers">Numbers</a> &bull;
  <a href="#how-it-compares">How it compares</a>
</p>

---

## What it actually does

You run Claude Code, Codex, Cursor, or your own app. Every time it talks to an LLM, it sends a big blob of text: your system prompt, the tool definitions, the whole conversation history, and the raw output of every command it ran. You pay for every one of those tokens, on every single turn.

**A lot of that text is waste.** A 200-line build log where only 2 lines are errors. A tool schema resent identically 50 times. A JSON array with 500 near-identical rows. The model doesn't need the bulk of it to answer well, but you're billed for all of it.

**llmtrim removes the waste before it's sent.** It installs as a local proxy that sits between your tool and the LLM provider. Requests pass through it, get compressed, and continue to the provider. The reply comes back unchanged. Your tool doesn't know it's there; you just get a smaller bill.

```
  before:  your tool ───── full request ─────▶  OpenAI / Anthropic / …
                    ◀──────── reply ──────────

  after:   your tool ──▶ llmtrim ──smaller──▶  OpenAI / Anthropic / …
                            (on your machine)
                    ◀──────── reply ──────────  (same answer)
```

> [!IMPORTANT]
> **It can never make your bill bigger or break a request.** Every compression step is re-measured with the provider's real tokenizer; if a step doesn't actually save tokens, it's reverted. If the provider rejects the compressed request, the original is resent verbatim. Worst case is zero savings, never a worse outcome.

Everything runs locally. Nothing is ever sent to us.

## See it on real output

Here's one real thing llmtrim does, end to end. An AI agent ran a build, and the `bash` tool returned a 58-line log. Only two lines matter (the errors), but all 58 get sent to the model and billed.

**Before**, what the model would receive (58 lines, 4,662 chars):

```text
[2026-06-13T10:02:00Z] INFO  compiling module core::worker::task_0 (incremental)
[2026-06-13T10:02:01Z] INFO  compiling module core::worker::task_1 (incremental)
[2026-06-13T10:02:02Z] INFO  compiling module core::worker::task_2 (incremental)
... 27 more near-identical INFO lines ...
[2026-06-13T10:02:31Z] ERROR src/worker/pool.rs:214: mismatched types: expected `usize`, found `i64`
... 25 more INFO lines ...
[2026-06-13T10:03:01Z] ERROR src/net/conn.rs:88: cannot borrow `buf` as mutable more than once
[2026-06-13T10:03:02Z] INFO  build failed, 2 errors
```

**After**, what llmtrim sends instead (5 lines, 978 chars, **−79%**):

```text
[{}] INFO compiling module core::worker::task_{} (incremental) [×30: (10:02:00Z..10:02:29Z step 1s; 0..29)]
[2026-06-13T10:02:31Z] ERROR src/worker/pool.rs:214: mismatched types: expected `usize`, found `i64`
[{}] INFO compiling module core::net::conn_{} (incremental) [×25: 10:02:32Z..10:02:56Z; 0..24]
[2026-06-13T10:03:01Z] ERROR src/net/conn.rs:88: cannot borrow `buf` as mutable more than once
[2026-06-13T10:03:02Z] INFO  build failed, 2 errors
```

Both errors and the summary survive **verbatim**. The repetitive INFO lines fold into a template plus their values, losslessly, because the range is regular (`task_0..task_29`). The model still sees exactly what happened; it just costs a fifth as much.

> If that's useful to you, a ⭐ helps other people find it.

Try it yourself on any request body:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai
```

Log-folding is just one of ten compressors. A different one re-encodes bulky JSON arrays into a compact table, with the same data in a third of the tokens:

```text
before:  [{"id":1,"city":"Paris","ok":true},{"id":2,"city":"Lyon","ok":false}, … 200 rows]
after:   [200]{id,city,ok}: 1,Paris,true; 2,Lyon,false; …          (TOON encoding, lossless)
```

Each compressor fires only where it pays:

| Where the waste is | What llmtrim does |
|---|---|
| **Tool output** (build logs, diffs, grep dumps, big JSON) | Keep the signal (errors, changes, matches), fold the noise |
| **Long context** (pasted docs, history) | Rank and keep the chunks relevant to the question; drop the rest |
| **Source code** | Keep the bodies of relevant functions, reduce the rest to signatures |
| **Tool schemas** (resent every turn) | Trim descriptions, drop unused tools, keep the cache prefix stable |
| **JSON / record arrays** | Re-encode to a compact table format, sample huge arrays |
| **The model's reply** | Ask for terser output where it won't hurt the answer |

<details>
<summary><b>Full stage reference (all 10 compressors)</b></summary>

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

Default `auto` switches each stage on only where it pays. `safe` runs the lossless stages only. [Full config →](#configuration)

</details>

## Get started

> [!NOTE]
> Works with any tool that routes through `HTTPS_PROXY`: Claude Code, Codex, Cursor, Aider, your own app. GitHub Copilot pins its certificates and can't be intercepted ([full list](#works-with)).

```bash
# 1. Install (any OS, prebuilt binary, no Rust needed)
npm install -g @llmtrim/cli@latest && llmtrim setup

# 2. Open a new shell. Your AI tools now route through llmtrim automatically.

# 3. Watch the savings add up as you work
llmtrim status
```

Prefer a GUI? The desktop tray app puts per-agent savings in your menu bar (macOS),
system tray (Windows), or AppIndicator (Linux). `setup` offers to install it, or run
`llmtrim tray` yourself. It ships in the Homebrew, Scoop, and npm packages; on Linux
download `llmtrim-tray` from the [latest release](https://github.com/fkiene/llmtrim/releases)
(needs `libwebkit2gtk-4.1` and `libayatana-appindicator3`).

<p align="center"><img src="crates/llmtrim-tray/docs/popover.svg" alt="llmtrim tray popover" width="320"></p>

**No Node?** Use an installer instead:

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh

# Windows (PowerShell)
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Or your own package manager, same binary everywhere: `brew install fkiene/tap/llmtrim` · `cargo binstall llmtrim` · `scoop install llmtrim` · `docker run ghcr.io/fkiene/llmtrim`. Full options in [INSTALL.md](INSTALL.md).

### Is this safe to install?

`setup` is a local HTTPS proxy, the same technique as [mitmproxy](https://mitmproxy.org), scoped to LLM APIs. It changes exactly three things (a CA certificate in `~/.llmtrim/`, a proxy setting in your shell profile, a login service), and `llmtrim uninstall` reverses all three. No API keys are stored (it forwards your tool's own auth), and your prompts never touch disk; only an anonymous count of tokens saved is kept. Full threat model: [SECURITY.md](SECURITY.md).

<details>
<summary><b>What gets installed, and how to verify the cert is harmless</b></summary>

1. **A private certificate** in `~/.llmtrim/`, cryptographically constrained to LLM API domains. It *cannot* read your bank, email, or any other traffic, even if the key were stolen. Check that yourself:
   ```bash
   llmtrim ca   # prints the cert path
   openssl x509 -in ~/.llmtrim/ca.pem -noout -text | grep -A3 "Name Constraints"
   # those domains are the only ones it can ever touch
   ```
2. **A proxy setting** in your shell profile (`HTTPS_PROXY` + `NODE_EXTRA_CA_CERTS`).
3. **A background service** that starts at login.

If the service stops, your tools fail fast with a connection error rather than silently bypassing it.

</details>

<details>
<summary><b>Day-to-day commands</b></summary>

```bash
llmtrim status              # health + savings dashboard (aliases: monitor, gain)
llmtrim statusline install  # add a live status line to Claude Code
llmtrim doctor              # something off? end-to-end diagnosis; each check names its fix
llmtrim start               # start the background proxy
llmtrim stop                # stop it
llmtrim serve               # run in the foreground instead (Ctrl-C to quit)
llmtrim wrap claude         # run an agent, guaranteeing this session routes through llmtrim (fails fast if it can't)
llmtrim update              # update to the latest release + restart
llmtrim uninstall           # exact inverse of setup: removes all three changes
```

`llmtrim status --daily` (or `--weekly` / `--monthly`) gives a time-series report; add `--json` or `--csv` to export.

</details>

## Use it as a CLI, MCP, or library

The same compression runs with no proxy and no setup, as a one-shot CLI, an embeddable Rust crate, native bindings for **Python, Ruby, Swift and Kotlin**, or a **WebAssembly** module for JavaScript (browser, Node, Cloudflare Worker). No extra model calls, no network: the deterministic engine runs in your process.

| Language | Install |
|---|---|
| Rust | `cargo add llmtrim-core` |
| Python | `pip install llmtrim` |
| Ruby | `gem install llmtrim` |
| Kotlin | `implementation("io.github.fkiene:llmtrim:0.10.2")` (Maven Central) |
| Swift | `.package(url: "https://github.com/fkiene/llmtrim-swift", from: "0.1.8")` (SwiftPM) |

**CLI.** Pipe a request in, get a compressed one out:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai > out.json
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim send     --provider openai   # compress, call, print
```

**Rust.** The engine is the [`llmtrim-core`](https://crates.io/crates/llmtrim-core) crate (no `tokio`, no network in its dependency tree):

```rust
use llmtrim_core::{compress, compress_with_config, config::DenseConfig, ir::ProviderKind};

// None auto-detects the provider from the request shape.
let out = compress(request_json, Some(ProviderKind::OpenAi))?;
println!("{} -> {} input tokens", out.input_tokens_before, out.input_tokens_after);

// …or pass an explicit preset/config:
let out = compress_with_config(request_json, Some(ProviderKind::OpenAi), &DenseConfig::preset("agent").unwrap())?;
```

**Python / Ruby / Swift / Kotlin.** One flat `compress(input, provider, preset)` call, generated from the same Rust engine via [UniFFI](https://mozilla.github.io/uniffi-rs/). The compiled engine is bundled in each package, so there's no Rust toolchain to install:

```python
import llmtrim

out = llmtrim.compress(request_json, llmtrim.Provider.OPEN_AI, "aggressive")
print(out.input_tokens_before, "->", out.input_tokens_after)
```

> [!NOTE]
> Every binding returns the compressed `request_json` plus the before/after token counts, and maps errors to native exceptions. Per-language install and usage live in [`crates/llmtrim-uniffi`](crates/llmtrim-uniffi).

**JavaScript / TypeScript (WebAssembly).** The same `compress(input, provider, preset)` call, compiled to WebAssembly, runs in the browser, Node, Bun, Deno, or a Cloudflare Worker with no network or filesystem access. The output type is fully typed in TypeScript:

```ts
import { compress } from "@llmtrim/js"; // @llmtrim/wasm is an alias for the same package

const out = compress(requestJson, "openai", "aggressive");
console.log(out.input_tokens_before, "->", out.input_tokens_after);
```

> [!NOTE]
> To stay small, the WASM build uses the estimate tokenizer: the absolute token counts are approximate, but the savings percentage is unaffected. Build and usage live in [`crates/llmtrim-wasm`](crates/llmtrim-wasm).

**MCP server.** `llmtrim mcp` speaks the [Model Context Protocol](https://modelcontextprotocol.io) over stdin/stdout, so any MCP client can compress payloads and read your savings without the proxy. It exposes three tools: `llmtrim_compress` (compress a full request body, honoring your `~/.llmtrim` config like the proxy), `llmtrim_compress_text` (shrink one text blob, lossless), and `llmtrim_stats` (your savings ledger). Every call records to the same ledger, so MCP traffic shows up in `llmtrim status`.

```bash
llmtrim mcp install          # register with Claude Code (one command)
llmtrim mcp install --print  # or print the config to paste into any other client
```

The printed block is the standard MCP config; for a client you edit by hand it looks like:

```json
{
  "mcpServers": {
    "llmtrim": { "command": "llmtrim", "args": ["mcp"] }
  }
}
```

**Status line for Claude Code.** `llmtrim statusline` renders a single line for Claude Code's
[custom status line](https://code.claude.com/docs/en/statusline): the model, the backend that
actually served the turn when you reroute (e.g. `→gpt-5.6-terra` or `→kimi`), a context-health
gauge, and how much llmtrim is trimming, plus rate-limit and prompt-cache reuse when Claude Code
reports them. The arrow reflects what answered, not what's configured — so in `sub` fallback mode
it stays off while Anthropic is serving, and appears on the turns the chain actually fires.

```bash
llmtrim statusline install          # wire it into ~/.claude/settings.json
llmtrim statusline install --print  # or print the settings snippet to paste yourself
```

```text
◆ Opus→gpt-5.6-terra   ▓▓▓▓▓░░░ 142k   ✂ 6.8%   ◔ 3h·24% · 4d·12%   ♻ 63% cached
```

The `✂` trim figure is scoped to the current Claude Code session; it reads `✂ –` until llmtrim
has saved something this session. `◔ 3h·24% · 4d·12%` shows the time remaining until each
Claude.ai rate-limit window resets and the share of that window used. The context gauge fills
against the real window of the model serving the turn — the rerouted backend's window under `sub`,
not Claude's — green below 40%, orange
40–65%, red above. `♻` shows this turn's prompt-cache reuse, and turns into `♻ cache cold`
once the session has been idle past the cache TTL, since the next message then pays a cold cache
write. Segments drop right-to-left on narrow terminals, and anything Claude Code doesn't report
(no reroute, no rate limits) is simply left out.

### Guard

Resuming a session whose prompt cache has expired re-writes the whole context on the very next
turn, billed at the cache-write rate — a few dollars on a large session, spent before any work
happens, with nothing at the prompt to say so.

`llmtrim guard` is a Claude Code `UserPromptSubmit` hook that stops that one turn:

```text
Idle 6h 19m, 347k tokens of context. The prompt cache has expired, so the next turn
re-writes the whole context — about $3.47 before any work happens.

You pay that whichever way you go: /compact pays it too (it reads the full context to
summarise it), and only comes out ahead if you are staying for several more turns. To
avoid the charge entirely, ask in a fresh session.
```

The prompt is not sent and no API call is made, so the warning itself is free. Resend to continue;
you are only warned once per idle gap. `llmtrim setup` offers to wire it in, or run `llmtrim guard
install` (`guard uninstall` to remove it — it merges into `~/.claude/settings.json` and leaves your
other hooks alone).

## Works with

Any tool that honors `HTTPS_PROXY` and an env-provided CA. That covers all 18 agents below plus any HTTPS_PROXY-aware CLI or SDK you build yourself:

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

Prefer no proxy? Any MCP client (Claude Code, Cursor, custom agents) can call llmtrim directly as tools instead: run `llmtrim mcp install`, or see [CLI, MCP, or library](#use-it-as-a-cli-mcp-or-library).

Providers come from the [`llm_providers`](https://crates.io/crates/llm_providers) registry (OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI, Moonshot, Zhipu, Qwen, OpenRouter, …) and update with it. Every non-LLM connection passes through untouched.

## Configuration

**Zero config needed.** The default (`auto`) inspects each request and picks the right compressors for its shape: tool-heavy → `agent`, code → `code`, long context with a question → `rag`, otherwise → `aggressive`.

Three tiers cover almost everyone. To force one, set `LLMTRIM_PRESET=<name>` or `preset = "<name>"` in `$XDG_CONFIG_HOME/llmtrim/config.toml`:

| preset | for |
| --- | --- |
| **`auto`** *(default)* | routes each request to the right compressors; right for almost everyone |
| **`safe`** | lossless input only: byte-faithful round-trip, no lossy stages |
| **`aggressive`** | squeeze everything, accept lossy cuts (quality-gated) |

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
| `LLMTRIM_EXCLUDE_PROVIDERS` | `exclude_providers` | wire shapes to skip compressing — `openai` / `anthropic` / `google` (e.g. `anthropic` to leave Claude Code untouched); coarse, covers every host of that shape |
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

<details>
<summary><b>Use cheaper models for Claude Code compaction</b></summary>

Claude Code runs `/compact` by sending an internal summarization request. llmtrim can recognize
that request and try an ordered list of cheaper models before the model selected in Claude Code:

```bash
llmtrim compact models haiku sonnet
llmtrim compact status
```

The equivalent config is:

```toml
[compact]
models = ["haiku", "sonnet"]
```

For each entry, llmtrim rebuilds and compresses the original request for that candidate, then checks
the embedded model registry. A configured model is skipped when its context window is unknown or
cannot hold the compressed input plus the requested output. If an eligible model fails before any
answer event reaches Claude Code, llmtrim advances to the next entry. The model originally selected
in Claude Code is always the implicit final attempt; do not add it to the list. Subscription reroutes
use the same order after applying the active provider's tier mapping. Run `llmtrim compact off` to
disable the override.

`llmtrim setup` proposes `haiku, sonnet` on Claude Code installations that have not made a choice.
An explicitly empty list records that the feature is disabled and prevents repeated setup prompts.

</details>

<details>
<summary><b>Route Claude Code to another subscription</b> (opt-in; a ChatGPT/Codex or Kimi plan, gray-area on the provider's terms)</summary>

llmtrim can serve Claude Code from a different subscription's backend instead of Anthropic.
It intercepts the Anthropic `/v1/messages` call, rewrites it to the provider's wire format,
streams the reply back as Anthropic SSE, and maps the Claude model tiers to provider models.
Two providers are supported: a ChatGPT plan through the Codex Responses API, and a Kimi
coding plan.

> **Warning:** using a subscription this way may violate the provider's terms of service. The
> login command prints this warning before it stores a token. Decide for yourself whether your
> plan permits it.

Sign in once, then turn it on:

```bash
llmtrim sub auth codex login   # or: llmtrim sub auth kimi login
llmtrim sub use codex         # apply the built-in tier mapping
```

Tokens are stored under `~/.llmtrim/<provider>/auth.json` (mode 0600) and refreshed
automatically. `llmtrim sub setup codex` opens an interactive editor to change which provider model
each Claude tier (opus / sonnet / haiku / fable) maps to; `llmtrim sub status` shows the
current mapping; `llmtrim sub off` disables rerouting and sends traffic back to Anthropic.

`llmtrim setup` also installs a user-wide Claude Code command for changing only the current
Claude Code window:

```text
/sub on
/sub off
/sub status
```

The window override includes its subagents and survives `/clear` and compaction. It does not affect
other open windows or newly started Claude Code processes. A resumed or forked conversation starts
with a fresh window policy; `/exit` removes the override, while an uncleanly terminated window
expires after 30 minutes. `/sub off` forces Anthropic with compression for that window, even when
the global reroute policy uses `always` or `fallback`. Run `llmtrim window-sub install` or
`llmtrim window-sub uninstall` explicitly to manage the Claude Code integration.

By default a set provider reroutes every turn. To use subscriptions only as a backup, switch to
fallback mode: `llmtrim sub mode fallback` forwards to Anthropic as usual and tries the configured
provider chain when Anthropic hits a quota, overload, transient server error, or transport failure.
Set the order with `llmtrim sub chain codex,kimi`; `llmtrim sub mode always` restores the default.
A transient failure (429/5xx) gets one bounded retry against Anthropic before the chain takes over,
so a blip doesn't move the turn — and its spend — to another provider. (`on-error`, the pre-0.10
name for this mode, still works.)

The mapping is not limited to the four Claude tiers. `llmtrim sub map codex <from> <to>` remaps
one model, where `from` is a tier name or an exact incoming model id, so any Anthropic-speaking
tool can be routed model by model; `llmtrim sub unmap` removes an entry, and `llmtrim sub models
codex` lists the candidate provider models. `llmtrim sub status --json` and `llmtrim sub auth
<codex|kimi> status --json` expose the state for scripts and the tray.

| env var | config key | meaning |
| --- | --- | --- |
| `LLMTRIM_SUB` | `sub` | active reroute provider (`codex`, `kimi`, or `off`) |
| `LLMTRIM_SUB_MODE` | `sub.mode` | `always` (default) or `fallback` |
| `LLMTRIM_SUB_CHAIN` | `sub.chain` | ordered fallback providers, e.g. `codex,kimi` |

Codex reasoning is adaptive by default; `llmtrim sub effort <none|low|medium|high|xhigh>` (env
`LLMTRIM_CODEX_EFFORT`) pins one effort on every rerouted turn instead.

</details>

### Chaining through an upstream proxy

Set `LLMTRIM_UPSTREAM_PROXY=http://host:port` to make llmtrim send its outbound
calls through another proxy instead of dialing the API directly. This covers two
cases: a corporate or auth proxy that all egress must pass through, and running
llmtrim alongside a second local tool such as [headroom](https://github.com/chopratejas/headroom)
on a different port.

The connection to the API is tunneled through the upstream with `CONNECT` and stays
under verifying TLS, so the upstream sees only the encrypted stream. An upstream that
points at llmtrim's own listen address (any spelling of localhost on the same port) is
rejected, because it would loop traffic back into the daemon; a companion proxy on a
different port is allowed. `http://user:pass@host:port` works for proxy auth, and those
credentials are redacted from logs.

Put the variable in the daemon's own launch environment (the launchd plist or systemd
unit), not only your shell profile. Credentials written into a shell profile are stored
there in plaintext.

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

Run `llmtrim doctor` for an end-to-end diagnosis; each failing check names its fix. Found a request it mangled? Set `LLMTRIM_CAPTURE_DIR` and [open an issue](https://github.com/fkiene/llmtrim/issues) with the before/after capture, since a repro is a fix. And if it saved you money, a ⭐ helps others find it.

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
