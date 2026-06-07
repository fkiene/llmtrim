<h1 align="center">llmtrim</h1>

<p align="center">
  <strong>Cut the whole LLM bill ~46% — input, output, and cache — with zero extra model calls.</strong>
</p>

<p align="center">
  <a href="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml"><img src="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL v3"></a>
  <img src="https://img.shields.io/badge/rust-1.85%2B-orange" alt="Rust 1.85+">
  <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/fkiene/llmtrim/main/.github/badges/tests.json" alt="tests">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/round--trip_cost-%E2%88%9246%25-2ea043" alt="round-trip cost saved">
  <img src="https://img.shields.io/badge/output_tokens-%E2%88%9273%25-2ea043" alt="output tokens saved">
  <img src="https://img.shields.io/badge/added_model_calls-0-0969da" alt="zero added model calls">
  <img src="https://img.shields.io/badge/deterministic-yes-0969da" alt="deterministic">
  <img src="https://img.shields.io/badge/any_provider-yes-0969da" alt="any provider">
</p>

<p align="center">
  <a href="#why-llmtrim">Why llmtrim</a> &bull;
  <a href="#install">Install</a> &bull;
  <a href="#run-it-and-forget-it">How it works</a> &bull;
  <a href="#what-it-does-to-your-prompt">Stages</a> &bull;
  <a href="#benchmark">Benchmark</a> &bull;
  <a href="#security">Security</a> &bull;
  <a href="#license">License</a>
</p>

---

**llmtrim cuts your whole LLM bill ~46% — one install, any provider, zero extra model calls.** Output tokens **−73%**, input **−24%**, the cached prefix billed once, the answer unchanged — **87 live A/B cases, billed at real rates.**

## −46% of the bill — measured live, not estimated

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="bench/frontier-light.svg">
    <img src="bench/frontier-dark.svg" alt="llmtrim cuts the LLM round-trip bill both ends: original $0.0167 vs llmtrim $0.0090, −46% cost (output −73%, input −24%) across 87 live A/B cases" width="840">
  </picture>
</p>

| 87 live A/B cases | original | compressed | saved |
|---|--:|--:|--:|
| input tokens | 58,518 | 44,379 | **−24%** |
| output tokens | 27,588 | 7,504 | **−73%** |
| total tokens | 86,106 | 51,883 | **−40%** |
| **round-trip cost** | **$0.0167** | **$0.0090** | **−46%** |

Every case is sent **twice** — once original, once compressed — both answers scored, the round-trip priced at real provider rates (`openai/gpt-oss-20b` via Groq). **Not estimated. Billed.** Pooled across 87 cases (generation, chat, reasoning, code, RAG, agents, summary, cache), the win is **both ends**: output collapses **−73%** (the expensive half), input trims **−24%** — most tools cut only one side. At n≈12/corpus the *per-corpus* deltas are noisy (the LLM-judge baseline alone swings ±30pp between runs), so read the pooled figure. **[Full methodology, per-corpus frontier + CIs →](bench/README.md)**

llmtrim is a **static, deterministic, zero-LLM-call** compressor for closed LLM APIs — a transparent `HTTPS_PROXY` that intercepts the request your tools send to OpenAI / Anthropic / Gemini / any provider, shrinks it with deterministic algorithms only (**no auxiliary model, no embeddings, no neural scoring**), forwards it, and reverses the lossless transforms on the response. A per-stage tokenizer gate means it **can never make your bill bigger or break a call**. One install, every tool, every provider.

## Why llmtrim

There are three places a request bleeds tokens. Most tools fix exactly one.

```
   you ─▶ [ REQUEST ] ─▶ LLM ─▶ [ RESPONSE ] ─▶ you
            ▲                       ▲
            └─────── llmtrim ───────┘   input + output + cache · any provider
   rtk ........ tool output only — local CLI, before it re-enters the REQUEST
   caveman .... model output only — English-prose instruction
```

rtk and caveman are both good at their one layer. llmtrim covers the layer they can't — **the API round-trip itself** — and does it deterministically, with a gate that means it **can never make your bill bigger**.

| | [rtk](https://github.com/rtk-ai/rtk) | [caveman](https://github.com/JuliusBrussee/caveman) | **llmtrim** |
|---|---|---|---|
| Compresses | local CLI tool output | model output (prose) | **the whole round-trip: input + output + cache + batch** |
| Sits at | Bash-tool hook | system-prompt skill | **network layer (HTTPS proxy)** |
| Touches the actual API request | no | no — *adds* to it | **yes** |
| Coverage | 60 known commands | English caveman prose | **any payload · any language · any provider** |
| Per-call instruction cost | n/a | a **528-word** skill prompt | **one 12-word sentence** (rides the cached prefix) |
| Can it increase your bill? | no (passthrough) | possible — skill prompt is added input | **no — per-stage tokenizer gate auto-reverts** |
| Can it break a call? | no | no | **no — replays the original on any upstream reject** |
| Quality measured? | no | tokens only | **yes — live A/B (saved *and* retained)** |
| Their own headline | −80% CLI output | −65% output | **−46% round-trip cost** |
| Extra model calls | 0 | 0 | **0** |

**vs caveman — output compression is its whole game, and llmtrim matches it.** Output tokens **−73%** on our 87-case live A/B vs caveman's reported **−65%** on theirs — different models and corpora, so *same ballpark*, not a knockout. The difference is everything *around* the savings. llmtrim also cuts **input**; caveman touches output only (its own README is candid the win is "readability and speed, cost a bonus"). llmtrim's terse lever is a single neutral sentence on the cacheable prefix — when we A/B'd a *forceful telegraphic style* (caveman's approach) it **backfired**: empty replies and hallucinated `because X` padding, so we kept the neutral instruction. And llmtrim works in any language, where caveman is English caveman-speak. Same output savings, fewer ways to lose — measured, not asserted.

**vs rtk — different layer, run both.** rtk is excellent at what it does: command-aware filtering of 60 CLI tools, −80% on their output. But it only fires on the Bash path, only for commands it has a filter for, and it never sees the API request. llmtrim compresses the request that actually hits the API — the system prompt, the **tool schemas resent every turn** (−35% input on Claude Code with tools intact), the message history, the structured payloads — generically, with no per-command filters, for any tool and any provider. rtk shrinks what your shell hands back; llmtrim shrinks what leaves the machine. They stack.

> **The guarantee neither competitor has:** every input stage is measured with the **real tokenizer** before it ships. If a transform doesn't net fewer tokens — counting the legend/instruction it injects — it auto-reverts to the original. If the upstream rejects the compressed request, the proxy replays the original verbatim. Worst case is *no savings*. Never a bigger bill, never a broken call.

## Install

```bash
# Prebuilt binary (Linux / macOS) — installs and runs `setup` for you
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh

# or with Cargo
cargo install --git https://github.com/fkiene/llmtrim

# or Homebrew
brew install fkiene/tap/llmtrim
```

**Windows:** native — `cargo install --git https://github.com/fkiene/llmtrim`, then `llmtrim setup` wires the PowerShell `$PROFILE` (`$env:HTTPS_PROXY` + `NODE_EXTRA_CA_CERTS`), trusts the local CA (`certutil -addstore -user Root`), enables autostart, and runs the daemon (`status` / `stop` via `tasklist` / `taskkill`); or run [`install.ps1`](install.ps1). **WSL** is the simplest, most-tested path if you already use it.

Full options (PATH, pinned versions, build-from-source) in [INSTALL.md](INSTALL.md).

## Run it and forget it

llmtrim is a **man-in-the-middle HTTPS proxy** — like mitmproxy, but compressing. It never touches a single IDE setting. One command wires everything:

```bash
llmtrim setup     # local CA + HTTPS_PROXY/NODE_EXTRA_CA_CERTS in your shell profile + autostart + start
```

```
  without llmtrim:                              with llmtrim:

  tool ──request──▶ LLM API                     tool ──request──▶ llmtrim ──compressed──▶ LLM API
   ▲                  │                           ▲                  │  (gate · stream)     │
   └──── response ────┘                           └──── response ────┴── pass-through ───────┘
        full bill                                          −46% bill, answer unchanged
```

Open a new shell and your tools route through it. Then:

```bash
llmtrim monitor       # savings dashboard: ● running · $ saved · −% round-trip · by-model
llmtrim monitor --watch   # live, refreshing — watch the bill shrink in real time
llmtrim uninstall     # one command back out — reverses everything, transparently
```

The proxy needs **no API key** (it forwards your tool's own auth). The CA is generated locally and **name-constrained to LLM API domains only** — it cannot forge a cert for anything else, and every other HTTPS connection blind-tunnels untouched. `llmtrim uninstall` is the exact inverse of `setup`: stops the daemon, disables autostart, strips the shell-profile block, removes the CA + binary — printing each step.

**What it stores:** the only thing llmtrim persists locally is a **metadata-only** savings ledger (`~/.local/share/llmtrim/tracking.db`) — provider, model, and token *counts*, **never prompt or response text**. It's bounded to the most recent 100k events; add age-based pruning with `retention_days = N` in the config TOML (or `LLMTRIM_RETENTION_DAYS=N`). `llmtrim uninstall --purge` deletes it.

<details>
<summary><strong>More proxy commands</strong></summary>

```bash
llmtrim serve --daemon   # start the interceptor in the background (setup does this)
llmtrim serve            # or foreground (Ctrl-C to stop)
llmtrim stop             # stop the daemon
llmtrim update           # update to the latest release + restart the daemon (channel-aware)
llmtrim autostart        # run at login (--off to disable)
llmtrim ca               # print the CA path + how to trust it system-wide (for GUI apps)
llmtrim monitor --daily  # time-series report (--weekly/--monthly); --json/--csv to export
```

`monitor` is the one savings view — snapshot, `--watch` live dashboard, `--daily/--weekly/--monthly` reports, `--json/--csv` export. (`status` and `gain` are kept as aliases.)

**It works with any tool** that respects `HTTPS_PROXY` + a custom CA from the environment — essentially every CLI coding agent and Node/VS Code tool. The intercepted host list is derived from the [`llm_providers`](https://crates.io/crates/llm_providers) registry — OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI, Moonshot, Zhipu, Qwen, MiniMax, Cerebras, OpenRouter, … — and updates with the crate. Tools that hardcode their endpoint or pin certs (e.g. Copilot) can't be intercepted.

**The interceptor defaults to `auto`** — shape-routing: each request goes to the preset matching its shape (tools → `agent`, fenced code → `code`, long context + a question → `rag`, else → `aggressive`), so the per-shape wins land without misfiring. Safe on real client traffic because the breakers are in place — the `cache` stage skips a client managing its own `cache_control` (no 400s), `retrieve` protects directive blocks, and `tool_select` never drops a tool already invoked in the conversation. On agent traffic the `agent` preset's **tool-description trimming** is the big lever: clients resend long tool schemas every call, so trimming them cuts ~35% of input with tool use intact. Override any of it with a config file.

</details>

## What it does to your prompt

Nine stages, ordered by the savings hierarchy `retrieve > cache > output control > serialization > structural skeleton > dedup > micro-text`. Each runs only if the **real-tokenizer gate** confirms a net win; anything that errors reverts. Compression uses vetted crates (`tiktoken-rs`, `toon-format`, `bm25`, `gaoya`, `tree-sitter`, `pest`, `image`, `unicode-normalization`, `whatlang`) — no hand-rolled parsers.

| Stage | Lever | What it does | When it runs |
|---|---|---|---|
| **A** cache discipline | cache | marks the invariant prefix with Anthropic `cache_control` so it's billed once | **auto · tools** |
| **B** lexical retrieval | retrieve | BM25 / TextRank keeps only query-relevant chunks; role-aware (never drops the question); MMR + lost-in-the-middle reordering | **auto · long context** |
| **C** skeletonization | skeleton | tree-sitter drops function bodies (keeps signatures) across 14 languages — Rust, JS, TS(X), Python, Go, Java, C, C++, C#, Kotlin, Swift, Zig, Ruby, PHP — with per-language stubs | **auto · code** |
| **D** serialize + hygiene | serialization | minify JSON; encode uniform record arrays to [TOON](https://crates.io/crates/toon-format)/CSV; strip base64; **Unicode normalize** (NFKC, strip invisible/format waste — keeps ZWJ for emoji + Arabic/Indic) | **always** · lossless |
| **E** dedup | dedup | collapse repeated lines as `[×N]`; near-dup (SimHash) + n-gram glossary on **prose only** — structured data (JSON/CSV/tables/config/code) left verbatim | **always** · exact |
| **F** output control | output | terse instruction · Chain-of-Draft · soft "answer within N tokens" budget · provider-native JSON schema | **auto** |
| **G** tool layer | tool | static tool selection + description trimming (schemas resend every call → big agent saving) | **auto · tools** |
| **H** multimodal | multimodal | downscale images to the provider's effective resolution cap (quality-neutral by construction); tile-snap | **auto · images** |
| **I** batch | transport | stacks on the provider's ~50% Batch-API discount | `batch` cmd |

*The **When it runs** column is what the default (`auto`) does, with zero config. **always** = every request (the lossless stages D · E). **auto · &lt;shape&gt;** = the default switches it on for the matching request — long context, code, tool/agent calls, or an embedded image. **auto** (F) = on broadly. Nothing is ever "off" under the default — each stage fires exactly where it pays. The bar is **cost with no measured quality loss**, not losslessness; the **`safe`** preset runs only the lossless stages (byte-faithful round-trip).*

**The default is `auto`** — it reads the request shape and applies the proven preset (tools → `agent`, fenced code → `code`, long context + a question → `rag`, else → `aggressive`), all zero-model, falling back to lossless `safe` when nothing fits. `safe`/`DenseConfig::default()` is the opt-in **lossless** mode (guaranteed byte-faithful). <a href="#configuration">Full config →</a>

## One-shot & library

The pure transform core (no I/O) underlies the proxy and the CLI:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai > out.json
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim send --provider openai   # compress + call + print
llmtrim batch --provider openai < requests.jsonl > out.jsonl     # stacks on the ~50% Batch discount
```

```rust
use llmtrim::{compress, compress_with_config};
use llmtrim::config::DenseConfig;
use llmtrim::ir::ProviderKind;

let result = compress(request_json, Some(ProviderKind::OpenAi))?;   // env/file config, auto-detect with None
println!("{} -> {} input tokens", result.input_tokens_before, result.input_tokens_after);

let result = compress_with_config(request_json, Some(ProviderKind::OpenAi), &DenseConfig::default())?;
```

## Benchmark

Two axes, measured live: **tokens saved** (real tokenizer, at compress time) and **quality retained** (the A/B delta between the model's answer on the *original* vs the *compressed* request). A preset is only honest if quality holds at its saving — the frontier of (saved, retained) is the benchmark, not the saving alone.

The per-corpus frontier (cost saved + quality retention, with confidence intervals) lives in **[bench/README.md](bench/README.md)**. At **n≈12/corpus the per-corpus deltas are noisy** — the LLM-judge baseline alone swings ±30pp between runs — so the **pooled** figure (−46% cost / −73% output) is the trustworthy one; the per-corpus view shows the *shape* of where compression pays (output-heavy generation/chat/reasoning) vs where it can't (cache, short extractive RAG).

Scored ground-truth where possible — numeric-exact (math), **pass@1 that runs the unit tests** (code) — plus token-F1 (QA), tool-call match (agents), LLM judge (open-ended). The per-stage token gate *guarantees fewer tokens, not preserved quality* — only this A/B axis catches the difference, which is why every lossy stage is gated on it.

```bash
python3 bench/scripts/download.py 40   # pull + hash real corpora (gsm8k, humaneval, dolly, hotpotqa, glaive, ultrachat, cnn)
bash    bench/scripts/run_all.sh       # live A/B (needs OPENROUTER_API_KEY; builds --features live)
python3 bench/scripts/chart.py         # regenerate the chart + table
```

## Configuration

**Zero config needed** — the default is `auto` (shape-routing). To pick a profile, set **one line**: `preset = "<name>"` in the config TOML (`$LLMTRIM_CONFIG` or `$XDG_CONFIG_HOME/llmtrim/config.toml`), or `LLMTRIM_PRESET=<name>` in the env.

| preset | for |
| --- | --- |
| **`auto`** *(default)* | shape-routes each request to the proven profile — the right answer for almost everyone |
| **`safe`** | lossless only — byte-faithful round-trip (turns the lossy stages off) |

**For a known workload**, name a specialized profile: `reasoning` (math / step-by-step — Chain-of-Draft) · `cache` (a fixed system prefix reused across calls — Stage A prefix caching).

`auto` already routes to `rag` / `agent` / `code` / `aggressive` per request shape, so naming those yourself rarely helps — **`aggressive` in particular just forces *every* lever onto *every* request**: identical to `auto` on plain prose, *riskier* on tools/code/RAG (which is exactly why `auto` routes RAG to `rag`, not `aggressive`). They stay nameable for experts but aren't recommended. A `preset` and raw flags are alternatives — `preset` wins; omit it to hand-tune the flags below.

<details>
<summary><strong>Advanced — per-flag overrides (alternative to a preset)</strong></summary>

| field | default | meaning |
| --- | --- | --- |
| `hygiene` | `true` | Stage D minify (+ base64 strip if enabled) |
| `normalize_unicode` | `false` | NFKC fold + strip invisible/format waste (lossy; in `aggressive`) |
| `serialize` | `true` | Stage D TOON encoding |
| `serialize_nested` | `true` | also encode arrays nested in content JSON |
| `serialize_min_rows` | `2` | min array rows before encoding |
| `serialize_csv` | `false` | encode flat arrays as both TOON and CSV, keep the smaller |
| `strip_base64` | `false` → **on in `auto`** | elide base64/data-URI blobs (≥200 chars) to a `[elided]` marker; lossy but **measured +0.0pp** (`bench/data/base64.jsonl`) |
| `numeric_sig_figs` | _(none)_ | round floats to N significant figures (lossy) |
| `output_control` | `false` | Stage F terse instruction + cap |
| `output_level` | `"terse"` | `terse` (clean) or `draft` (Chain-of-Draft) |
| `output_max_tokens` | _(none)_ | impose a hard cap when the request has none |
| `output_token_budget` | _(none)_ | inject a soft "answer within N tokens" budget |
| `output_compact_code` | `false` | instruct minified-code output (model-gated) |
| `retrieve` | `false` | Stage B lexical retrieval (lossy) |
| `retrieve_keep_ratio` | `0.5` | fraction of chunks to keep |
| `retrieve_reorder` | `false` | head+tail U-shape (lost-in-the-middle; lossless) |
| `retrieve_mmr` | `false` | MMR diversity-aware selection |
| `retrieve_sentence` | `false` | training-free DSLR sentence pruning (answer + boundary protected) |
| `cache` / `cache_max_breakpoints` | `false` / `4` | Stage A `cache_control` breakpoints (lossless) |
| `dedup` | `true` | collapse exact-duplicate lines (lossless) |
| `dedup_near` | `false` | also collapse near-duplicate lines (SimHash) |
| `ngram` / `ngram_max_entries` | `false` / `32` | reversible n-gram abbreviation (lossless) |
| `tool_select` / `tool_trim_desc` | `false` | Stage G keep relevant tools / trim descriptions |
| `skeletonize` / `minify_code` | `false` | Stage C drop bodies / strip indentation (lossless) |
| `multimodal` / `image_detail` | `false` | Stage H downscale to the provider's cap |

Env: `LLMTRIM_PRESET` (select a preset by name), `LLMTRIM_CONFIG` (config-file path), `LLMTRIM_DB_PATH` (ledger location).

</details>

## Security

llmtrim sits between your tool and the provider, so its trust model *is* the product. Full policy + threat model in **[SECURITY.md](SECURITY.md)**; the essentials:

- **Local CA, name-constrained.** Generated on your machine (`~/.llmtrim/ca.pem`, key `0600`) and X.509-constrained to LLM API domains — it **cannot mint a cert for any other host**, even if the key were stolen. Trusted per-tool via `NODE_EXTRA_CA_CERTS`, not system-wide; every non-LLM HTTPS connection blind-tunnels untouched.
- **No keys, no prompts on disk.** The proxy forwards your tool's own auth (needs no key of its own); prompt and response text stay in memory — never logged, never persisted.
- **Binds `127.0.0.1` only** — no client authentication; never expose it on a public interface.
- **Metadata-only ledger** — token counts, provider, model id. Never prompt or response text. `llmtrim uninstall --purge` deletes it.

Report vulnerabilities **privately** via a [GitHub security advisory](https://github.com/fkiene/llmtrim/security/advisories/new), not a public issue.

## Known limits

Honesty is the product. The same A/B that proves the savings is the one that surfaces these:

- **Anthropic / Gemini token counts are approximate** — no public exact tokenizer, so an o200k BPE proxy is used and flagged (`is_exact() == false`, surfaced in `gain`). OpenAI is exact (tiktoken).
- **Output savings aren't measured live** — the proxy compresses the input prompt; an output *saving* needs the A/B counterfactual, which only the offline `bench` has. `status` shows captured output + total spend, but "saved" is input-side.
- **The default is quality-gated, not lossless.** The bar is *cost with no measured quality regression* — so the shipped default (`auto`) runs lossy stages (output control everywhere; retrieval / skeleton / dedup per shape) wherever the [eval](bench/README.md) shows quality holds. The per-stage **token** gate guarantees fewer tokens, **not** quality — quality is proven offline and baked into the routing. Need a guaranteed byte-faithful round-trip? Use the **`safe`** preset (lossless only). When you raise a drop ratio yourself, read the frontier first.
- **`rusqlite` is pinned to 0.31** — newer versions need the unstable `cfg_select` on stable Rust.

## License

llmtrim is licensed under [**AGPL-3.0-only**](LICENSE). You may use, modify, and self-host it freely; if you run a **modified** version as a network service, the AGPL requires you to release your source under the same license.

Contributions are accepted under the AGPL with a [DCO](CONTRIBUTING.md#sign-your-commits-dco) sign-off.
