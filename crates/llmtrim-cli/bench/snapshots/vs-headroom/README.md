# llmtrim vs Headroom

Both libraries driven through their Python APIs (`llmtrim.compress` and `headroom.compress`). Before/after token counts use the **same** `o200k_base` encoder over the **same** message-content span. Latency is the median compress time over 9 runs (model load excluded). llmtrim preset: `agent`.

## tool-output (n=5)

| tool | input tokens before→after | input saved | median ms |
|---|--:|--:|--:|
| **llmtrim** | 41,214 → 6,746 | **84%** | 4.0 |
| Headroom | 41,214 → 26,201 | 36% | 13.8 |

## general (n=64)

| tool | input tokens before→after | input saved | median ms |
|---|--:|--:|--:|
| **llmtrim** | 34,735 → 31,629 | **9%** | 0.5 |
| Headroom | 34,735 → 34,659 | 0% | 0.3 |

## llmtrim per-stage attribution (tool-output group)

Each stage's own token delta, the breakdown the binding now exposes and Headroom does not.

| stage | applied | tokens removed |
|---|--:|--:|
| toolout | 3/5 | 6,369 |
| image | 5/5 | 0 |
| hygiene | 2/5 | 7,206 |
| json-crush | 2/5 | 20,892 |
| serialize-toon | 0/5 | 0 |
| dedup | 0/5 | 0 |
| tools | 0/5 | 0 |
| cache | 5/5 | 0 |

## Image compression (head-to-head)

Verified against both codebases. On the bench corpus the image stage removed 0 tokens (the images were already under each provider's cap), so this is a capability comparison, not a savings claim on this set.

| capability | llmtrim | Headroom |
|---|---|---|
| image-pixel compression | yes (Stage H) | none |
| decode + Lanczos3 downscale | yes | none |
| tile optimizer | yes (`snap_tile`) | none |

- llmtrim ships image-pixel compression (Stage H): decode plus Lanczos3 downscale to per-provider effective-resolution caps (OpenAI 2048/768 with 512px tiles, Anthropic 1568/1.15MP, Gemini 3072), a tile optimizer (`snap_tile` for OpenAI's 512px tile pricing), EXIF-orientation pass-through, decode-bomb limits, JPEG q90, and a size-regression guard. Files: `crates/llmtrim-core/src/media.rs`, `stages/image.rs`.
- Headroom (Rust port in `../headroom`) ships no image-pixel compression: no `image` crate or codec dependency, no resize/downscale code. It skips `image_url` blocks as non-compressible and only redacts base64 image bytes from logs.
- Headroom's ONNX runtime is a text router, not an image router. The Rust port uses `fastembed` (bge-small-en text embeddings) plus `magika` (content classifier); the Python `headroom` package uses a ModernBERT encoder instead (see Method notes). The port enables fastembed's `image-models` feature flag but never calls it (dead flag).
- The gap runs the other way: Headroom has ML/embedding semantic relevance routing; llmtrim is lexical-only (BM25 plus TextRank, `crates/llmtrim-core/src/stages/retrieve.rs`, no embeddings by spec). That gap is text routing, not images, and is a separate line item.

## Method notes

- Latency is the median compress call, with a warm-up first so neither library is charged for one-time setup. llmtrim must run on the **release** wheel (`build-wheel.sh --release`); the debug build is several times slower and not representative.
- Headroom's `compress` runs a **local ModernBERT encoder** (`answerdotai/ModernBERT-base`, the multi-hundred-MB model it downloads on first use) for its semantic smart-crusher. It makes no generative LLM API call; verified by running compress with all network sockets blocked. llmtrim is purely algorithmic (BPE counting plus deterministic stages), which is why its warm latency is lower despite removing more tokens.
- Headroom protects user and system messages, so on the `general` corpora (natural request shapes, no tool results) it mostly no-ops; the `tool-output` group is its home turf.
- Output tokens are out of this head-to-head. Headroom is input-only, and llmtrim's output shaping is a preset feature (e.g. `aggressive`, `reasoning`) measured in the main benchmark on a non-reasoning model. The `--live` arm exists for spot checks, but gpt-oss-20b bills hidden reasoning as output, so it is not a fair output denominator (see the main bench README).

