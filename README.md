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
  <img src="https://img.shields.io/badge/round--trip_cost-%E2%88%9266%25-2ea043" alt="round-trip cost saved">
</p>

<p align="center">
  <a href="#-why-llmtrim">Why llmtrim</a> &bull;
  <a href="#-install">Install</a> &bull;
  <a href="#-run-it-and-forget-it">How it works</a> &bull;
  <a href="#-what-it-does-to-your-prompt">Stages</a> &bull;
  <a href="#-benchmark">Benchmark</a> &bull;
  <a href="#-security">Security</a> &bull;
  <a href="#-acknowledgments">Acknowledgments</a> &bull;
  <a href="#-license">License</a>
</p>

---

A drop-in HTTPS proxy that compresses every LLM request and reply. **Any provider, answers unchanged, no model in the loop.**

## 💸 −66% of the bill - measured live, not estimated

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="bench/frontier-light.svg">
    <img src="bench/frontier-dark.svg" alt="llmtrim cuts the LLM round-trip bill both ends: original $0.0365 vs llmtrim $0.0126, −66% cost (output −74%, input −31%) across 112 live A/B cases" width="840">
  </picture>
</p>

<table align="center">
  <thead>
    <tr>
      <th align="left">112 live A/B cases</th>
      <th align="right">original</th>
      <th align="right">compressed</th>
      <th align="right">saved</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>input tokens</td><td align="right">71,031</td><td align="right">49,062</td><td align="right"><b>−31%</b></td></tr>
    <tr><td>output tokens</td><td align="right">25,843</td><td align="right">6,628</td><td align="right"><b>−74%</b></td></tr>
    <tr><td>total tokens</td><td align="right">96,874</td><td align="right">55,690</td><td align="right"><b>−43%</b></td></tr>
    <tr><td><b>round-trip cost</b></td><td align="right"><b>$0.0365</b></td><td align="right"><b>$0.0126</b></td><td align="right"><b>−66%</b></td></tr>
    <tr><td><b>answer quality</b></td><td align="right"><b>78.9%</b></td><td align="right"><b>82.2%</b></td><td align="right"><b>+3.3pp</b></td></tr>
  </tbody>
</table>

**The savings don't cost quality:** it's *up* +3.3pp. Every case is A/B'd live: sent twice, answered, scored, and billed at real rates, never estimated.

**Beyond the benchmark:** on live Claude Code traffic, llmtrim cuts **−68% of compressible input** while never touching the cached prefix, so your ~90% prompt-cache discount stays intact (`llmtrim status` shows yours).

<sub>Measured on `qwen/qwen3-next-80b-a3b-instruct`. Cost % scales with a model's output:input pricing: −66% here, −44–59% on others, less on reasoning models. [Methodology + per-corpus frontier →](bench/README.md)</sub>

## 🎯 Why llmtrim

A request bleeds tokens in three places. Most tools fix one; llmtrim fixes all three:

- **Input**: system prompt, tool schemas (resent every turn), history
- **Output**: the model's reply, the expensive half
- **Cache**: the invariant prefix, else re-billed in full

rtk and caveman each compress one layer; [Headroom](https://github.com/chopratejas/headroom) is the closest peer (input / tool-output, but Python + ML). llmtrim does the whole round-trip, pure-Rust, behind a gate that **can't make your bill bigger**.

| | **llmtrim** | Headroom | rtk | caveman |
|---|:---:|:---:|:---:|:---:|
| Whole round-trip (input · output · cache) | ✅ | input only | CLI only | output only |
| **Can't increase your bill** (auto-revert gate) | ✅ | ❌ | ✅ | ❌ |
| Quality measured **live** (A/B: saved *and* kept) | ✅ | offline evals | ❌ | tokens only |
| No Python · no models · single binary | ✅ | ❌ | ✅ | ✅ |
| **Overhead it adds / request** | **<10 ms** | 52 ms median\* | <10 ms | n/a |
| Deterministic (no ML variance, no downloads) | ✅ | ❌ | ✅ | ✅ |

<sub>\* Headroom's own production telemetry (161 ms mean, 4.2 s P99) — sources in the comparison below.</sub>

**Stack them:** llmtrim adds −35% on Claude Code's resent tool schemas *on top of* rtk, and hits **93–98%** on agentic tool-output with the bill measured both ways.

<details>
<summary><b>llmtrim vs Headroom: feature by feature</b></summary>

The trade is **pure-Rust simplicity + cache-correctness** vs **ML reach**:

| | llmtrim | Headroom |
|---|---|---|
| Runtime | single 47 MB static binary, 0 deps | Python + numpy / onnxruntime / transformers / magika / fastembed (100s MB – GB) |
| Latency it adds | **<10 ms per request** (0.5 ms at 5 KB → 7 ms at 49k tokens; tokenizer-bound, faster on Anthropic), negligible next to the model round-trip, and the smaller prompt often makes the call *faster* overall. ~110 ms one-time startup. Measured | **52 ms median / 161 ms mean** proxy overhead, P99 4.2 s (its own production telemetry\*); on its 10.2K-token search-results scenario, compression alone is **189 ms p50** — llmtrim does a comparable ~11k-token request in **2.7 ms**, ~**70× faster** |
| Models | none (deterministic) | ONNX detection (magika) + learned text compressor (Kompress) + embeddings |
| Tool output | log / diff / grep + repetitive fallback, adaptive↔aggressive auto-split | SmartCrusher / log / diff / search (ML-assisted) |
| Cache discipline | frozen-zone guard (never busts the `cache_control` prefix) + tool/schema sort + OpenAI `prompt_cache_key` | live-zone byte-range surgery + cache stabilization |
| Output side | terse / Chain-of-Draft / token-budget shaping | input-side only |

**Where Headroom leads** (honest): ML content detection, semantic relevance, a learned text compressor, cross-agent memory, an MCP server, more providers (Bedrock / Vertex). Savings are in the same league (llmtrim 93–98%; Headroom ~92%).

<sub>\* llmtrim's latencies are measured here (`cargo bench --bench latency`). Headroom's numbers are self-reported in its [benchmarks page](https://headroom-docs.vercel.app/docs/benchmarks): 52 ms median / 161 ms mean / 4.2 s P99 proxy overhead from its production telemetry, and 189 ms p50 to compress its 10.2K-token search-results scenario; llmtrim compresses a comparable ~11k-token request in 2.7 ms (measured).</sub>

</details>

*(We A/B'd caveman's telegraphic style: it backfired with empty replies + hallucinated padding; we ship a neutral one-liner instead.)*

> **The guarantee neither has:** every stage is checked by the real tokenizer before it ships. No net token win → auto-revert. Upstream rejects it → replay the original verbatim. Worst case is no savings: never a bigger bill, never a broken call.

## ⚡ Install

```bash
# Prebuilt binary (Linux / macOS) - installs and runs `setup` for you
curl -fsSL https://raw.githubusercontent.com/fkiene/llmtrim/main/install.sh | sh

# or with Cargo
cargo install --git https://github.com/fkiene/llmtrim

# or Homebrew
brew install fkiene/tap/llmtrim
```

```powershell
# Windows (PowerShell) - prebuilt, installs and runs setup
irm https://raw.githubusercontent.com/fkiene/llmtrim/main/install.ps1 | iex
```

Prebuilt for x64 and ARM64. WSL: run the Linux line above. Full options (PATH, pinned versions, build-from-source) in [INSTALL.md](INSTALL.md).

## 🔧 Run it and forget it

A man-in-the-middle HTTPS proxy, like mitmproxy but compressing. No IDE settings touched; one command wires it:

```bash
llmtrim setup     # local CA + HTTPS_PROXY/NODE_EXTRA_CA_CERTS in your shell profile + autostart + start
```

```
  without llmtrim:                              with llmtrim:

  tool ──request──▶ LLM API                     tool ──request──▶ llmtrim ──compressed──▶ LLM API
   ▲                  │                           ▲                  │  (gate · stream)      │
   └──── response ────┘                           └──── response ────┴── pass-through ───────┘
        full bill                                          −66% bill, answer unchanged
```

Open a new shell and your tools route through it. Then:

```bash
llmtrim status        # health + savings: ● running · ✓ port ✓ env ✓ ca · $ saved · by-model
llmtrim status --watch    # live, refreshing - watch the bill shrink in real time
llmtrim doctor        # something off? end-to-end diagnosis, each failing check names its fix
llmtrim uninstall     # one command back out - reverses everything, transparently
```

`uninstall` is the exact inverse of `setup`: it stops the daemon, strips the shell-profile block, and removes the CA and binary, printing each step. There's no API key to manage (it forwards your tool's own auth). Safe by construction: a local name-constrained CA, with only a metadata-only counts ledger on disk ([Security →](#-security)).

<details>
<summary><strong>More proxy commands</strong></summary>

```bash
llmtrim start            # start the interceptor in the background (setup does this)
llmtrim serve            # or foreground (Ctrl-C to stop)
llmtrim stop             # stop the daemon
llmtrim update           # update to the latest release + restart the daemon (channel-aware)
llmtrim autostart        # run at login (--off to disable)
llmtrim ca               # print the CA path + how to trust it system-wide (for GUI apps)
llmtrim status --daily   # time-series report (--weekly/--monthly); --json/--csv to export
```

`monitor` is the one savings view: snapshot, `--watch`, `--daily/--weekly/--monthly`, and `--json/--csv` export (`status`/`gain` are aliases). The snapshot doubles as a health check: it verifies the whole chain (daemon → port → env → CA → traffic) and exits 0 healthy / 1 stopped / 2 degraded; `status -q` prints just `healthy|degraded|stopped` for scripts, and the JSON export carries the same under `daemon.health`.

Any tool honoring `HTTPS_PROXY` + an env CA works (every CLI agent, Node/VS Code). The host list comes from the [`llm_providers`](https://crates.io/crates/llm_providers) registry - OpenAI, Anthropic, Google, DeepSeek, Mistral, xAI, Moonshot, Zhipu, Qwen, MiniMax, Cerebras, OpenRouter, … - and updates with the crate. Pinned-cert tools (e.g. Copilot) can't be intercepted.

Default `auto` [routes each request to its shape's preset](#-what-it-does-to-your-prompt), safe on live traffic via breakers:

- `cache` skips a client managing its own `cache_control` (no 400s)
- `retrieve` protects directive blocks
- `tool_select` never drops an already-invoked tool

On agent traffic, `agent`'s tool-description trimming is the big lever, since clients resend long tool schemas every call.

</details>

## 🧩 What it does to your prompt

Ten stages, ordered by the savings hierarchy `tool-output > retrieve > cache > output > json-sample > serialization > skeleton > dedup > micro-text`. Each fires only if the real-tokenizer gate nets a win, and **never rewrites content under a `cache_control` marker**, so compression can't bust the prompt cache.

| Stage | Lever | What it does | When it runs |
|---|---|---|---|
| **T** tool-output | toolout | lossless template fold first: consecutive runs *and* interleaved parallel-build lines, then window logs · diffs · grep · repetitive dumps to the signal (errors, changes, matches); adaptive↔aggressive auto-split | auto · tool results |
| **A** cache discipline | cache | mark + stabilize the invariant prefix (sort tools/schema · OpenAI `prompt_cache_key`) so it stays cached across calls | auto · tools |
| **B** lexical retrieval | retrieve | BM25+ ranking with RM3 feedback (TextRank when query-less) · TextTiling cuts prose at topic shifts · budgeted submodular selection keeps the relevant *non-redundant* chunks; question protected | auto · long context |
| **C** skeletonization | skeleton | tree-sitter keeps the bodies of the query-relevant functions, drops the rest to signatures - 14 languages | auto · code |
| **D** serialize + hygiene | serialization | minify JSON, encode record arrays to [TOON](https://crates.io/crates/toon-format)/CSV, Unicode-normalize | always · lossless |
| **D₊** json sample | json_crush | down-sample huge record arrays: keep first/last + outliers (errors, rare values) + a query-biased *diverse* sample | auto · big JSON |
| **E** dedup | dedup | collapse duplicate + near-duplicate lines (prose only; data untouched) | always · exact |
| **F** output control | output | terse instruction · Chain-of-Draft · token budget · native JSON schema | auto |
| **G** tool layer | tool | static tool selection + description trimming (schemas resent each call) | auto · tools |
| **H** multimodal | multimodal | downscale images to the provider's resolution cap | auto · images |

*Default `auto` switches each stage on only where it pays (the "When it runs" column). `safe` runs the lossless stages only. [Full config →](#-configuration)*

## 🛠️ One-shot & library

Same transform core, no proxy:

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

## 🔬 Benchmark

Two axes, both measured live:

- **tokens saved**: real tokenizer, at compress time
- **quality retained**: A/B delta between the answer on the *original* vs *compressed* request

A preset is honest only if quality holds at its saving: the (saved, retained) frontier is the benchmark, not the saving alone. Full per-corpus frontier + CIs in [bench/README.md](bench/README.md). It shows where compression pays (output-heavy generation/chat/reasoning) and where it can't (cache, short extractive RAG).

Scored on ground truth where possible - numeric-exact (math), pass@1 running the unit tests (code) - plus token-F1 (QA), tool-call match (agents), LLM judge (open-ended).

```bash
python3 bench/scripts/download.py 40   # pull + hash real corpora (gsm8k, humaneval, dolly, hotpotqa, glaive, ultrachat, cnn)
bash    bench/scripts/run_all.sh       # live A/B (needs OPENROUTER_API_KEY; builds --features live)
python3 bench/scripts/chart.py         # regenerate the chart + table
```

## 📊 Configuration

**Zero config needed**: default `auto` shape-routes every request. To force a profile, set one line: `preset = "<name>"` (config TOML at `$LLMTRIM_CONFIG` or `$XDG_CONFIG_HOME/llmtrim/config.toml`) or `LLMTRIM_PRESET=<name>`.

| preset | for |
| --- | --- |
| **`auto`** *(default)* | shape-routes each request to the proven profile - right for almost everyone |
| **`safe`** | lossless only - byte-faithful round-trip (lossy stages off) |

Known workload? Name a profile: `reasoning` (math / step-by-step) · `cache` (a fixed prefix reused across calls).

Under the hood `auto` routes by shape: tools → `agent`, code → `code`, long-context + question → `rag`, else → `aggressive`. Naming one yourself rarely helps; `aggressive` just forces every lever onto every request, same as `auto` on prose but riskier on tools/code/RAG. Power users can still hand-tune raw flags (`preset` wins over flags).

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
| `skeleton_keep_full_top_k` | `5` | bodies kept for the top-k functions overlapping the conversation (HCP-graded) |
| `skeleton_drop_unmatched` / `skeleton_drop_min_body_lines` | `false` / `8` | also drop zero-overlap functions ≥ N lines entirely (on in `aggressive`) |
| `multimodal` / `image_detail` | `false` | Stage H downscale to the provider's cap |
| `tool_minify_schema` | `false` → on in `agent`/`aggressive` | minify tool JSON-Schemas in place (drop `title`/`$schema`/`examples`, dedup boilerplate descriptions): stays valid JSON Schema |
| `quality_gate` | `true` | after the token gate, revert a lossy cut whose query-relevant coverage drops below the calibrated threshold ("saved tokens by deleting the answer") |
| `memo` | `true` | proxy-only turn-stability memo: an already-seen conversation prefix reuses last turn's compressed bytes verbatim, keeping the provider prefix cache warm on agent loops (in-memory only) |

Env: `LLMTRIM_PRESET` (preset by name), `LLMTRIM_CONFIG` (config-file path), `LLMTRIM_DB_PATH` (ledger location).

</details>

## 🔒 Security

llmtrim sits between your tool and the provider - its trust model *is* the product. Full threat model in **[SECURITY.md](SECURITY.md)**:

- **Local CA, name-constrained.** Generated on your machine (`~/.llmtrim/ca.pem`, key `0600`), X.509-constrained to LLM API domains, so it can't mint a cert for any other host even if the key were stolen. Trusted per-tool via `NODE_EXTRA_CA_CERTS`; every non-LLM connection blind-tunnels untouched.
- **No keys, no prompts on disk.** Forwards your tool's own auth; prompt/response text stays in memory - never logged, never persisted.
- **Binds `127.0.0.1` only**: no client auth; never expose it on a public interface.
- **Metadata-only ledger** (`~/.local/share/llmtrim/tracking.db`) - provider, model, token *counts*, never content. Cap 100k events; `retention_days = N` to age-prune; `uninstall --purge` wipes it.

Report vulnerabilities **privately** via a [security advisory](https://github.com/fkiene/llmtrim/security/advisories/new), not a public issue.

## ⚠️ Known limits

Honesty is the product: the same A/B that proves the savings surfaces these:

- **Anthropic / Gemini counts are approximate**: no public exact tokenizer, so an o200k BPE proxy is used and flagged (`is_exact() == false`, surfaced in `gain`). OpenAI is exact (tiktoken).
- **Output savings aren't measured live**: the proxy compresses input; an output *saving* needs the A/B counterfactual, which only offline `bench` has. `status` "saved" is input-side.
- **Default is quality-gated, not lossless**: lossy stages run where the [eval](bench/README.md) shows quality holds; the token gate ensures fewer tokens, not quality. Want a byte-faithful round-trip? Use `safe`.
- **`rusqlite` pinned at 0.39**: 0.40+ pulls `libsqlite3-sys` 0.38, whose build script needs the still-unstable `cfg_select` ([rust#115585](https://github.com/rust-lang/rust/issues/115585)) and won't build on stable.

## 🙏 Acknowledgments

Every lever is a deterministic implementation of published research - the ideas are theirs, the engineering and the real-tokenizer gate are ours.

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

## 📄 License

[**AGPL-3.0-only**](LICENSE): use, modify, and self-host freely. Run a modified version as a network service and the AGPL requires you to release your source under the same license. Contributions via [DCO](CONTRIBUTING.md#sign-your-commits-dco) sign-off.
