<p align="center">
  <img src="logo.svg" alt="llmtrim" width="140">
</p>

<h1 align="center">llmtrim</h1>

<p align="center">
  <strong>Cut ~66% off your LLM bill: input, output, and cache, with zero extra model calls.</strong>
</p>

<p align="center">
  <a href="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml"><img src="https://github.com/fkiene/llmtrim/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-blue" alt="License: AGPL v3"></a>
  <img src="https://img.shields.io/badge/rust-1.88%2B-orange" alt="Rust 1.88+">
</p>

<p align="center">
  <a href="#-the-numbers">Numbers</a> &bull;
  <a href="#-what-compression-looks-like">Before / after</a> &bull;
  <a href="#-get-started-60-seconds">Get started</a> &bull;
  <a href="#-works-with">Works with</a> &bull;
  <a href="#-compared-to">Compared to</a> &bull;
  <a href="#-benchmark">Benchmark</a> &bull;
  <a href="#-security">Security</a> &bull;
  <a href="https://github.com/fkiene/llmtrim/issues">Issues</a>
</p>

---

A drop-in HTTPS proxy that compresses every LLM request and reply. Works with any provider, with no model in the loop. Quality holds, A/B-checked live on every benchmark case.

A request bleeds tokens in three places. llmtrim fixes all three:

- **Input**: system prompt, tool schemas, history. Resent every turn.
- **Output**: the model's reply. The expensive half.
- **Cache**: the invariant prefix. Re-billed in full when busted.

Every cut passes the **token gate**: a check that re-counts the result with the provider's real tokenizer and reverts any stage that doesn't save.

> **The guarantee:** no net token win → auto-revert. Upstream rejects the request → the original is replayed verbatim. Worst case is zero savings - never a bigger bill, never a broken call.

## 💸 The numbers

Measured live, not estimated. Every one of the 112 A/B cases is sent twice (original and compressed), then answered, scored, and billed at real rates.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="bench/frontier-light.svg">
    <img src="bench/frontier-dark.svg" alt="llmtrim cuts the LLM round-trip bill both ends: original $0.0365 vs llmtrim $0.0126, −66% cost (output −74%, input −31%) across 112 live A/B cases" width="840">
  </picture>
</p>

- **Quality holds.** Answers scored 78.9% original vs 82.2% compressed. The +3.3pp delta sits within the per-corpus confidence intervals (±5–15pp at these sample sizes), so read it as *no degradation*, not a bonus - per-corpus CIs in [bench/README.md](bench/README.md).
- **The token cuts travel; the price tag varies.** The cuts are model-independent: −31% input, −74% output. The cost saving depends on the model's output:input price ratio - −66% on the benchmark model (`qwen/qwen3-next-80b-a3b-instruct`, ≈12:1 ratio), projected −57–59% at GPT-4o / Claude Sonnet rates, less on reasoning models whose hidden thinking tokens can't be cut from the prompt side.
- **Your prompt cache survives.** On live Claude Code traffic, llmtrim cuts **−68% of compressible input** without ever touching the cached prefix. Your ~90% prompt-cache discount stays intact; `llmtrim status` shows yours.

<details>
<summary><b>Exact numbers (112 live A/B cases)</b></summary>

| | original | compressed | saved |
|---|--:|--:|--:|
| input tokens | 71,031 | 49,062 | **−31%** |
| output tokens | 25,843 | 6,628 | **−74%** |
| total tokens | 96,874 | 55,690 | **−43%** |
| **round-trip cost** | **$0.0365** | **$0.0126** | **−66%** |
| **answer quality** | 78.9% | 82.2% | Δ within CI (no measured degradation) |

[Methodology + per-corpus frontier →](bench/README.md)

</details>

## 🔍 What compression looks like

Each stage fires only where it pays, and only if the token gate nets a win:

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="pipeline-light.svg">
    <img src="pipeline-dark.svg" alt="One agent request through llmtrim's five biggest levers: tool-output folding, lexical retrieval, cache discipline, TOON serialization, and output control - all ten stages take it from 11,240 to 3,914 input tokens, −65%" width="880">
  </picture>
</p>

<details>
<summary><b>Full stage reference</b></summary>

Stages run in savings order: `tool-output > retrieve > cache > output > json-sample > serialization > skeleton > dedup > micro-text`. Nothing under a `cache_control` marker is ever rewritten.

| Stage | Lever | What it does | When it runs |
|---|---|---|---|
| **T** tool-output | toolout | lossless template fold first (consecutive runs *and* interleaved parallel-build lines), then window logs · diffs · grep · repetitive dumps to the signal: errors, changes, matches | auto · tool results |
| **A** cache discipline | cache | mark + stabilize the invariant prefix (sort tools/schema · OpenAI `prompt_cache_key`) so it stays cached across calls | auto · tools |
| **B** lexical retrieval | retrieve | BM25+ ranking with RM3 feedback (TextRank when query-less) · TextTiling cuts prose at topic shifts · budgeted submodular selection keeps the relevant *non-redundant* chunks; question protected | auto · long context |
| **C** skeletonization | skeleton | tree-sitter keeps the bodies of the query-relevant functions, drops the rest to signatures - 14 languages | auto · code |
| **D** serialize + hygiene | serialization | minify JSON, encode record arrays to [TOON](https://crates.io/crates/toon-format) (a compact table encoding for JSON arrays) or CSV, Unicode-normalize | always · lossless |
| **D₊** json sample | json_crush | down-sample huge record arrays: keep first/last + outliers (errors, rare values) + a query-biased *diverse* sample | auto · big JSON |
| **E** dedup | dedup | collapse duplicate + near-duplicate lines (prose only; data untouched) | always · exact |
| **F** output control | output | terse instruction · Chain-of-Draft · token budget · native JSON schema | auto |
| **G** tool layer | tool | static tool selection + description trimming (schemas resent each call) | auto · tools |
| **H** multimodal | multimodal | downscale images to the provider's resolution cap | auto · images |

Default `auto` switches each stage on only where it pays. `safe` runs the lossless stages only. [Full config →](#-configuration)

</details>

## ⚡ Get started (60 seconds)

> **Is this safe?** Everything runs locally - nothing is ever sent to us. llmtrim sees your LLM traffic only; every other connection passes through untouched. `setup` changes three things (a certificate in `~/.llmtrim/`, a proxy block in your shell profile, a background service) and `llmtrim uninstall` removes all three. Anything that can't be compressed safely is sent through unmodified. Full threat model: [SECURITY.md](SECURITY.md).

```bash
# 1 - Install (Linux / macOS; runs `setup` for you)
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh

# 2 - Open a new shell. Your tools now route through llmtrim.

# 3 - Watch the bill shrink
llmtrim status --watch
```

```powershell
# Windows (PowerShell) - prebuilt, installs and runs setup
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Prefer to read what you run? `cargo install --git https://github.com/fkiene/llmtrim` or `brew install fkiene/tap/llmtrim` - same `setup`, no script. Prebuilt for x64 and ARM64; WSL uses the Linux line. Full options in [INSTALL.md](INSTALL.md).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="status-watch-light.svg">
    <img src="status-watch-dark.svg" alt="llmtrim status --watch: live dashboard showing tokens trimmed, dollars off your real bill, input/output savings bars, and per-model breakdown, refreshing in real time" width="760">
  </picture>
</p>

### How it works

llmtrim is a local HTTPS proxy: it decrypts, compresses, and re-encrypts your LLM traffic on your own machine - the same technique as mitmproxy, scoped to LLM APIs. `setup` creates a private certificate that lets llmtrim read *this one kind of traffic*: it is technically restricted to LLM domains and cannot read your bank, email, or anything else. It also wires `HTTPS_PROXY`/`NODE_EXTRA_CA_CERTS` into your shell profile and starts the daemon at login. No IDE settings are touched.

Don't take the README's word for the "LLM domains only" part - check it yourself:

```bash
llmtrim ca   # prints the certificate path, then:
openssl x509 -in ~/.llmtrim/ca.pem -noout -text | grep -A3 "Name Constraints"
# the domains listed there are the only ones it can ever intercept
```

```
  without llmtrim:                              with llmtrim:

  tool ──request──▶ LLM API                     tool ──request──▶ llmtrim ──compressed──▶ LLM API
   ▲                  │                           ▲                  │  (gate · stream)      │
   └──── response ────┘                           └──── response ────┴── pass-through ───────┘
        full bill                                          −66% bill, answer unchanged
```

There's no API key to manage - it forwards your tool's own auth. The CA is name-constrained to LLM domains, and only a metadata-only counts ledger touches disk ([Security →](#-security)).

```bash
llmtrim status      # health + savings: ● running · ✓ port ✓ env ✓ ca · $ saved · by-model
llmtrim doctor      # something off? end-to-end diagnosis; each failing check names its fix
llmtrim uninstall   # exact inverse of setup: daemon, profile block, CA, binary - all reversed
```

If the daemon ever stops, your tools fail fast with a connection error rather than silently bypassing compression. `llmtrim doctor` names the problem; `llmtrim start` fixes it.

<details>
<summary><b>More proxy commands</b></summary>

```bash
llmtrim start            # start the interceptor in the background (setup does this)
llmtrim serve            # or foreground (Ctrl-C to stop)
llmtrim stop             # stop the daemon
llmtrim update           # update to the latest release + restart the daemon (channel-aware)
llmtrim autostart        # run at login (--off to disable)
llmtrim ca               # print the CA path + how to trust it system-wide (for GUI apps)
llmtrim status --daily   # time-series report (--weekly/--monthly); --json/--csv to export
```

`status` doubles as a health check. It verifies the whole chain (daemon → port → env → CA → traffic) and exits 0 healthy / 1 stopped / 2 degraded. `status -q` prints just `healthy|degraded|stopped` for scripts; the JSON export carries the same under `daemon.health`.

Default `auto` routes each request to its shape's preset, with breakers that keep it safe on live traffic:

- `cache` skips a client managing its own `cache_control` (no 400s)
- `retrieve` protects directive blocks
- `tool_select` never drops an already-invoked tool

On agent traffic, tool-description trimming is the big lever - clients resend long tool schemas on every call.

</details>

## 🔌 Works with

Any tool that honors `HTTPS_PROXY` and an env-provided CA - which is every CLI agent and most Node apps:

| Tool | Works | Notes |
|---|:---:|---|
| Claude Code | ✅ | −68% of compressible input on live traffic, cache discount intact |
| Codex CLI | ✅ | |
| Gemini CLI | ✅ | |
| Cursor / VS Code extensions | ✅ | Node-based: picks up `NODE_EXTRA_CA_CERTS` |
| Aider, OpenCode, any HTTPS_PROXY-aware CLI | ✅ | |
| Your own app / SDK | ✅ | or call the [library / one-shot CLI](#-one-shot--library) directly |
| GitHub Copilot | ❌ | certificate pinning - can't be intercepted |

Providers come from the [`llm_providers`](https://crates.io/crates/llm_providers) registry (OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI, Moonshot, Zhipu, Qwen, MiniMax, Cerebras, OpenRouter, …) and update with the crate. Every non-LLM connection is blind-tunneled untouched.

## 🛠️ One-shot & library

Use the same compression without the proxy - from the CLI or as a Rust library:

```bash
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim compress --provider openai > out.json
echo '{"model":"gpt-4o","messages":[...]}' | llmtrim send --provider openai   # compress + call + print
```

```rust
use llmtrim::{compress, compress_with_config};
use llmtrim::config::DenseConfig;
use llmtrim::ir::ProviderKind;

let result = compress(request_json, Some(ProviderKind::OpenAi))?;   // env/file config, auto-detect with None
println!("{} -> {} input tokens", result.input_tokens_before, result.input_tokens_after);

let result = compress_with_config(request_json, Some(ProviderKind::OpenAi), &DenseConfig::default())?;
```

## 🤝 Compared to

Three neighbors solve parts of the same problem - good company to be in. [RTK](https://github.com/rtk-ai/rtk) pioneered CLI-output filtering, [caveman](https://github.com/JuliusBrussee/caveman) the terse-output skill, and [Headroom](https://github.com/chopratejas/headroom) is the closest peer on the input side. Each compresses one layer; llmtrim does the whole round-trip.

| | **llmtrim** | Headroom | RTK | caveman |
|---|:---:|:---:|:---:|:---:|
| Whole round-trip (input · output · cache) | ✅ | input only | CLI only | output only |
| **Can't increase your bill** (auto-revert gate) | ✅ | ❌ | ✅ | ❌ |
| **Live A/B**: savings *and* answer quality | ✅ | offline evals | ❌ | tokens only |
| Install: one static binary | ✅ | Python + GB models | ✅ | ✅ |
| **Overhead it adds / request** | **<10 ms** | 52 ms median\* | <10 ms | n/a |
| Deterministic: same request → same result | ✅ | ❌ | ✅ | ✅ |

<sub>\* Headroom's own production telemetry (161 ms mean, 4.2 s P99) - sources in the feature comparison below.</sub>

**They stack.** llmtrim removes another 35% from Claude Code's resent tool schemas *on top of* RTK. On agentic tool output it saves **93–98%**, with the bill measured both ways. (Our A/B of telegraphic output styles found they hurt quality, so Stage F injects a neutral terse instruction instead.)

<details>
<summary><b>llmtrim vs Headroom: feature by feature</b></summary>

The trade is **pure-Rust simplicity + cache-correctness** vs **ML reach**:

| | llmtrim | Headroom |
|---|---|---|
| Runtime | single 47 MB static binary, 0 deps | Python + numpy / onnxruntime / transformers / magika / fastembed (100s MB – GB) |
| Latency it adds | **<10 ms per request**, measured here: 0.5 ms at 5 KB, 7 ms at 49k tokens. ~110 ms one-time startup. The smaller prompt often makes the call *faster* overall | **52 ms median / 161 ms mean**, P99 4.2 s - self-reported production telemetry\* |
| Models | none (deterministic) | ONNX detection (magika) + learned text compressor (Kompress) + embeddings |
| Tool output | log / diff / grep + repetitive fallback, adaptive↔aggressive auto-split | SmartCrusher / log / diff / search (ML-assisted) |
| Cache discipline | never rewrites the `cache_control` prefix + tool/schema sort + OpenAI `prompt_cache_key` | live-zone byte-range surgery + cache stabilization |
| Output side | terse / Chain-of-Draft / token-budget shaping | input-side only |

**Where Headroom leads** (honest): ML content detection, semantic relevance, a learned text compressor, cross-agent memory, an MCP server, more providers (Bedrock / Vertex). Savings are in the same league (llmtrim 93–98%; Headroom ~92%).

<sub>\* llmtrim's latencies are measured here (`cargo bench --bench latency`). Headroom's numbers are self-reported on its [benchmarks page](https://headroom-docs.vercel.app/docs/benchmarks).</sub>

</details>

## 🔬 Benchmark

The benchmark measures two things per request, both live:

- **tokens saved**: real tokenizer, at compress time
- **quality retained**: A/B delta between the answer on the *original* vs the *compressed* request

A preset only counts if quality holds at its saving, so the (saved, retained) frontier is the benchmark, not the saving alone. It also shows where compression pays (output-heavy generation, chat, reasoning) and where it can't (cache workloads, short extractive RAG). Full per-corpus frontier + CIs in [bench/README.md](bench/README.md).

Scoring uses ground truth where possible: numeric-exact (math), pass@1 running the unit tests (code), token-F1 (QA), tool-call match (agents), LLM judge (open-ended).

```bash
python3 bench/scripts/download.py 40   # pull + hash real corpora (gsm8k, humaneval, dolly, hotpotqa, glaive, ultrachat, cnn)
bash    bench/scripts/run_all.sh       # live A/B (needs OPENROUTER_API_KEY; builds --features live)
python3 bench/scripts/chart.py         # regenerate the chart + table
```

## 📊 Configuration

**Zero config needed.** Default `auto` routes every request by its shape: tools → `agent`, code → `code`, long-context + question → `rag`, else → `aggressive`. To force a preset, set one line - `preset = "<name>"` in the config TOML (`$LLMTRIM_CONFIG` or `$XDG_CONFIG_HOME/llmtrim/config.toml`) or `LLMTRIM_PRESET=<name>`.

| preset | for |
| --- | --- |
| **`auto`** *(default)* | routes each request to the proven preset for its shape - right for almost everyone |
| **`safe`** | lossless only - byte-faithful round-trip (lossy stages off) |

Known workload? Name a preset: `reasoning` (math / step-by-step) or `cache` (a fixed prefix reused across calls). Naming one yourself rarely beats `auto`. Power users can hand-tune raw flags (`preset` wins over flags).

<details>
<summary><strong>Advanced - per-flag overrides (alternative to a preset)</strong></summary>

| field | default | meaning |
| --- | --- | --- |
| `hygiene` | `true` | Stage D minify (+ base64 strip if enabled) |
| `normalize_unicode` | `false` | NFKC fold + strip invisible/format waste (lossy; in `aggressive`) |
| `serialize` | `true` | Stage D TOON encoding |
| `serialize_nested` | `true` | also encode arrays nested in content JSON |
| `serialize_min_rows` | `2` | min array rows before encoding |
| `serialize_csv` | `false` | encode flat arrays as both TOON and CSV, keep the smaller |
| `serialize_flatten` | `false` → on in `agent`/`aggressive` | flatten nested-uniform records to dotted columns (`meta.region`) |
| `serialize_buckets` | `false` → on in `agent`/`aggressive` | partition heterogeneous record arrays into per-shape TOON tables |
| `json_crush` / `json_crush_max_rows` | `false` / `50` → on in `agent`/`aggressive` | sample record arrays longer than the cap (keep first/last + outliers + a query-biased sample); lossy |
| `strip_base64` | `false` → on in `auto` | elide base64/data-URI blobs (≥200 chars) to a `[elided]` marker; lossy but measured +0.0pp (`bench/data/base64.jsonl`) |
| `numeric_sig_figs` | _(none)_ | round floats to N significant figures (lossy) |
| `output_control` | `false` | Stage F terse instruction + cap |
| `output_level` | `"terse"` | `terse` (clean) or `draft` (Chain-of-Draft) |
| `output_max_tokens` | _(none)_ | impose a hard cap when the request has none |
| `output_token_budget` | _(none)_ | inject a soft "answer within N tokens" budget |
| `output_compact_code` | `false` | instruct minified-code output (model-gated) |
| `retrieve` | `false` | Stage B lexical retrieval (lossy) |
| `retrieve_keep_ratio` | `0.5` | fraction of the segment's tokens kept (the selection budget) |
| `retrieve_reorder` | `false` | head+tail U-shape (lost-in-the-middle; lossless) |
| `retrieve_mmr` | `false` | MMR diversity-aware selection |
| `retrieve_sentence` | `false` | training-free DSLR sentence pruning (answer + boundary protected) |
| `cache` / `cache_max_breakpoints` | `false` / `4` | Stage A `cache_control` breakpoints (lossless) |
| `dedup` | `true` | collapse exact-duplicate lines (lossless) |
| `dedup_near` | `false` | also collapse near-duplicate lines (SimHash) |
| `ngram` / `ngram_max_entries` | `false` / `32` | reversible n-gram abbreviation (lossless) |
| `tool_select` / `tool_trim_desc` | `false` | Stage G keep relevant tools / trim descriptions |
| `toolout` | `false` → on in `agent`/`aggressive` | Stage T tool-output compression (log / diff / grep + repetitive fallback); positional elision |
| `toolout_mode` | `"auto"` | Stage T split: `adaptive` · `aggressive` · `auto` (per-segment by noise density) |
| `toolout_max_lines` / `toolout_min_lines` | `40` / `20` | keep-budget ceiling / skip segments shorter than this |
| `toolout_template` | `true` | lossless template fold before windowing: consecutive runs (Drain) + interleaved lines (LSH grouping) |
| `skeletonize` / `minify_code` | `false` | Stage C drop bodies / strip indentation (lossless) |
| `skeleton_keep_full_top_k` | `5` | bodies kept for the top-k functions overlapping the conversation (Hierarchical Context Pruning) |
| `skeleton_drop_unmatched` / `skeleton_drop_min_body_lines` | `false` / `8` | also drop zero-overlap functions ≥ N lines entirely (on in `aggressive`) |
| `multimodal` / `image_detail` | `false` | Stage H downscale to the provider's cap |
| `tool_minify_schema` | `false` → on in `agent`/`aggressive` | minify tool JSON-Schemas in place (drop `title`/`$schema`/`examples`, dedup boilerplate descriptions): stays valid JSON Schema |
| `quality_gate` | `true` | after the token gate, revert a lossy cut whose query-relevant coverage drops below the calibrated threshold ("saved tokens by deleting the answer") |
| `memo` | `true` | proxy-only memo: a conversation prefix seen last turn reuses its compressed bytes verbatim, so the provider's prefix cache stays warm on agent loops (in-memory only) |

Env: `LLMTRIM_PRESET` (preset by name), `LLMTRIM_CONFIG` (config-file path), `LLMTRIM_DB_PATH` (ledger location).

</details>

## 🔒 Security

llmtrim sits between your tool and the provider - its trust model *is* the product. Full threat model in **[SECURITY.md](SECURITY.md)**:

- **Local CA, name-constrained.** Generated on your machine (`~/.llmtrim/ca.pem`, key `0600`), X.509-constrained to LLM API domains. Even a stolen key can't mint a cert for any other host. Trusted per-tool via `NODE_EXTRA_CA_CERTS`; every non-LLM connection blind-tunnels untouched.
- **No keys, no prompts on disk.** Forwards your tool's own auth. Prompt/response text stays in memory - never logged, never persisted.
- **Binds `127.0.0.1` only.** No client auth; never expose it on a public interface.
- **Metadata-only ledger** (`~/.local/share/llmtrim/tracking.db`) - provider, model, token *counts*, never content. Cap 100k events; `retention_days = N` to age-prune; `uninstall --purge` wipes it.

Report vulnerabilities **privately** via a [security advisory](https://github.com/fkiene/llmtrim/security/advisories/new), not a public issue.

## ⚠️ Known limits

These are the current limits, surfaced by the same A/B that proves the savings:

- **Anthropic / Gemini counts are approximate.** No public exact tokenizer, so an o200k BPE proxy is used and flagged (`is_exact() == false`, surfaced in `status`). OpenAI is exact (tiktoken).
- **Output savings aren't measured live.** The proxy compresses input; an output *saving* needs the A/B counterfactual, which only offline `bench` has. `status` "saved" is input-side.
- **Default is quality-gated, not lossless.** Lossy stages run where the [eval](bench/README.md) shows quality holds; the token gate ensures fewer tokens, not quality. Want a byte-faithful round-trip? Use `safe`.

## 🙏 Acknowledgments

Every lever is a deterministic implementation of published research - the ideas are theirs, the engineering and the token gate are ours.

<details>
<summary><strong>Papers + crates behind each stage</strong></summary>

**Retrieval & context (Stage B)**
- **BM25**: Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009) · [`bm25`](https://crates.io/crates/bm25)
- **BM25+**: Lv & Zhai, *Lower-Bounding Term Frequency Normalization* (CIKM 2011) - δ floor so an occurrence always beats absence
- **RM3**: Lavrenko & Croft, *Relevance-Based Language Models* (SIGIR 2001) - pseudo-relevance feedback for sparse queries
- **TextTiling**: Hearst, *TextTiling: Segmenting Text into Multi-paragraph Subtopic Passages* (CL 1997) - prose chunk boundaries at lexical-cohesion valleys
- **TextRank**: Mihalcea & Tarau, *TextRank: Bringing Order into Texts* (EMNLP 2004)
- **MMR**: Carbonell & Goldstein, *The Use of MMR, Diversity-Based Reranking…* (SIGIR 1998)
- **Submodular selection**: Lin & Bilmes, *A Class of Submodular Functions for Document Summarization* (ACL 2011) + cost-ratio knapsack greedy, [arXiv:2008.05391](https://arxiv.org/abs/2008.05391) - token-budgeted chunk/row selection (CELF lazy greedy)
- **Diverse sampling**: Chen et al., *Fast Greedy MAP Inference for DPP* (NeurIPS 2018) - the json-sample diversity fill
- **Lost in the Middle**: Liu et al. (2023), [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) - head+tail reordering
- **DSLR**: Hwang et al. (2024), [arXiv:2407.03627](https://arxiv.org/abs/2407.03627) - sentence-level pruning

**Code (Stages C, F)**
- **RepoCoder**: Zhang et al. (2023), [arXiv:2303.12570](https://arxiv.org/abs/2303.12570) - AST skeletons beat raw source for non-focus code
- **Hierarchical Context Pruning**: Zhang et al. (2024), [arXiv:2406.18294](https://arxiv.org/abs/2406.18294) - keep full bodies only for the completion-relevant functions (our ranking is lexical, not embeddings)
- **The Hidden Cost of Readability**: Pan et al. (2025), [arXiv:2508.13666](https://arxiv.org/abs/2508.13666) - code minification
- **Reducing Token Usage … via Minification**: Hrubec & Cito (2026), [arXiv:2606.01326](https://arxiv.org/abs/2606.01326) - per-transformation token accounting

**Tool output (Stage T)**
- **Drain**: He et al., *Drain: An Online Log Parsing Approach with Fixed Depth Tree* (ICWS 2017) - the consecutive template fold
- **Brain**: Yu et al., *Brain: Log Parsing with Bidirectional Parallel Tree* (IEEE TSC 2023) - positional-voting template extraction
- **LogLSHD**: Huang et al. (2025), [arXiv:2504.02172](https://arxiv.org/abs/2504.02172) - MinHash-LSH grouping of interleaved same-template lines (ours is deterministic: first-N voting, alphanumeric tokens kept)

**Dedup & abbreviation (Stages E, E+)**
- **SimHash**: Charikar, *Similarity Estimation Techniques from Rounding Algorithms* (STOC 2002) · [`gaoya`](https://crates.io/crates/gaoya)
- **CompactPrompt**: Choi et al. (2025), [arXiv:2510.18043](https://arxiv.org/abs/2510.18043) - n-gram abbreviation
- **Maximal repeats**: Becher et al., *Efficient Repeat Finding via Suffix Arrays* ([arXiv:1304.0528](https://arxiv.org/abs/1304.0528)) + **Re-Pair**, Larsson & Moffat (DCC 1999) - the dictionary miner: all maximal repeated phrases, selected by real token gain

**Output control (Stage F)**
- **Chain-of-Draft**: Xu et al. (2025), [arXiv:2502.18600](https://arxiv.org/abs/2502.18600) - terse reasoning steps
- **TALE**: Han et al. (2024), [arXiv:2412.18547](https://arxiv.org/abs/2412.18547) - soft "answer within N tokens" budget

**Serialization (Stage D)**
- **TOON** (Token-Oriented Object Notation) - Johann Schopplich · [`toon-format`](https://crates.io/crates/toon-format)

Built on the Rust ecosystem: [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs), [`toon-format`](https://crates.io/crates/toon-format), [`bm25`](https://crates.io/crates/bm25), [`gaoya`](https://crates.io/crates/gaoya), [`tree-sitter`](https://crates.io/crates/tree-sitter), [`pest`](https://crates.io/crates/pest), [`image`](https://crates.io/crates/image), [`unicode-normalization`](https://crates.io/crates/unicode-normalization), [`whatlang`](https://crates.io/crates/whatlang), [`hudsucker`](https://crates.io/crates/hudsucker), [`rusqlite`](https://crates.io/crates/rusqlite).

</details>

## 🚀 Try it on one real session

Install, open a new shell, and leave `llmtrim status --watch` running while you work. If the dollars column doesn't move, `llmtrim uninstall` reverses everything. Found a request it mangled? Set `LLMTRIM_CAPTURE_DIR` and [open an issue](https://github.com/fkiene/llmtrim/issues) with the before/after capture - a repro is a fix. And if it saved you money, a ⭐ helps others find it.

## 📄 License

[**AGPL-3.0-only**](LICENSE): use, modify, and self-host freely. **Running llmtrim locally to compress your own traffic triggers no obligations** - the AGPL only applies if you offer a modified llmtrim as a network service to others, in which case you must release your source under the same license. Contributions via [DCO](CONTRIBUTING.md#sign-your-commits-dco) sign-off.
