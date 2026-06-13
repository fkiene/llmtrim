# llmtrim-core

<strong>The static, deterministic compression engine behind [llmtrim](https://github.com/fkiene/llmtrim) — as an embeddable Rust library.</strong>

It takes a provider-shaped LLM request (OpenAI, Anthropic or Google JSON), strips the wasted tokens out of it with deterministic algorithms only — **no auxiliary model, no embeddings, no network, no `tokio`** — and hands you back a smaller request plus the measured token delta. Typical savings are **30–90% of input tokens**, with no change to the answer.

[![crates.io](https://img.shields.io/crates/v/llmtrim-core)](https://crates.io/crates/llmtrim-core)
[![docs.rs](https://img.shields.io/docsrs/llmtrim-core)](https://docs.rs/llmtrim-core)
[![license](https://img.shields.io/badge/license-AGPL--3.0-blue)](https://www.gnu.org/licenses/agpl-3.0.txt)

```toml
[dependencies]
llmtrim-core = "0.1"
```

```rust
use llmtrim_core::{compress, compress_with_config, config::DenseConfig, ir::ProviderKind};

// Auto-detect the provider from the request shape (pass Some(ProviderKind::…) to force it).
let out = compress(&request_json, None)?;
println!("{} -> {} input tokens", out.input_tokens_before, out.input_tokens_after);
// `out.request_json` is the compressed body — send it to the provider unchanged.

// Or compress with an explicit workload preset:
let cfg = DenseConfig::preset("agent").unwrap();
let out = compress_with_config(&request_json, Some(ProviderKind::Anthropic), &cfg)?;
```

> [!IMPORTANT]
> **It can never make the request bigger or break it.** Every step is re-measured with the provider's real tokenizer; a step that doesn't actually save tokens is reverted. Anything under a `cache_control` marker is left byte-identical so the provider's prompt cache stays warm. Worst case is zero savings, never a worse outcome.

## What it compresses

Each compressor fires only where it pays — `auto` (the default) picks the right ones from the request's shape:

| Where the waste is | What the engine does |
|---|---|
| **Tool output** (build logs, diffs, grep dumps, big JSON) | Keep the signal (errors, changes, matches), losslessly fold the noise |
| **Long context** (pasted docs, history) | Rank + keep the chunks relevant to the question; drop the rest |
| **Source code** | Keep relevant function bodies, reduce the rest to signatures (14 languages) |
| **Tool schemas** (resent every turn) | Trim descriptions, drop unused tools, keep the cache prefix stable |
| **JSON / record arrays** | Re-encode to a compact table ([TOON](https://crates.io/crates/toon-format) / CSV), sample huge arrays |
| **The model's reply** | Ask for terser output where it won't hurt the answer |

Presets: `auto` (shape-routed, default), `aggressive`, `agent`, `code`, `rag`, `safe` (lossless stages only).

## API

- `compress` — load configuration from the environment / config file, optionally auto-detect the provider.
- `compress_with_config` — compress with an explicit `config::DenseConfig`; no environment access, fully deterministic (used by tests and embedders).
- `route` — pick the workload preset for a request from its structure alone.
- `CompressResult` — the compressed body, the per-stage report, and the before/after token counts.

Full reference on [docs.rs](https://docs.rs/llmtrim-core).

## Other languages

The same engine is exposed to **Python, Ruby, Swift and Kotlin** via [UniFFI](https://mozilla.github.io/uniffi-rs/) — see [`llmtrim-uniffi`](../llmtrim-uniffi). The [`llmtrim`](https://crates.io/crates/llmtrim) crate is the CLI and drop-in compressing proxy built on this engine.

## License

AGPL-3.0-only.
