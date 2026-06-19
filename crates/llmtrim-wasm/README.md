# llmtrim-wasm

WebAssembly/JS bindings for the [`llmtrim-core`](../llmtrim-core) compression engine, for
running the static prompt/payload compressor in a browser, Node, Bun, Deno, or a Cloudflare
Worker. No network or filesystem access.

Published to npm as **`@llmtrim/js`** (with **`@llmtrim/wasm`** as an alias that re-exports
the same package).

## API

```ts
import { compress } from "@llmtrim/js";

const out = compress(requestBodyJson, "openai", "agent");
// out: CompressOutput { request_json, provider, model, tokenizer_label,
//   tokenizer_exact, input_tokens_before, input_tokens_after,
//   frozen_input_tokens, output_shaped, stages: StageReport[] }
```

- `provider`: `"openai" | "anthropic" | "google"`, or `undefined`/`null` to auto-detect from
  the body.
- `preset`: a named workload preset (`aggressive`, `agent`, `code`, `rag`, `safe`, …), or
  `undefined`/`null` for the built-in defaults. This binding never reads the environment or a
  config file.

TypeScript types for `CompressOutput` and `StageReport` are generated via `tsify`, so the
`.d.ts` is fully typed (no `any`).

This build links `llmtrim-core` with `default-features = false`: the estimate tokenizer is
used (counts are approximate; savings percentages are unchanged), and code skeletonization
and image downscaling are no-ops. That keeps the bundle small (~1 MB gzipped, under the
Cloudflare Workers 3 MB free-tier cap).

## Building

The npm package is built in CI with `wasm-pack build --target bundler` (it manages a
matching `wasm-bindgen` internally). The manual `cargo` + `wasm-bindgen` recipe below is the
equivalent for local development:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli

# The JS-backed getrandom backend needs a rustc cfg in the environment (it cannot live in a
# repo .cargo/config.toml, which would break `cargo publish`).
RUSTFLAGS='--cfg getrandom_backend="wasm_js"' \
  cargo build -p llmtrim-wasm --release --target wasm32-unknown-unknown

wasm-bindgen target/wasm32-unknown-unknown/release/llmtrim_wasm.wasm \
  --out-dir pkg --target bundler   # or: nodejs | web

# optional size pass
wasm-opt -O3 --enable-reference-types --enable-bulk-memory --enable-mutable-globals \
  --enable-nontrapping-float-to-int --enable-sign-ext --enable-multivalue \
  pkg/llmtrim_wasm_bg.wasm -o pkg/llmtrim_wasm_bg.wasm
```

## Smoke test

After building with `--target nodejs` into `pkg/`, run `node smoke.mjs` (it imports
`./pkg/llmtrim_wasm.js`). It exercises OpenAI and Anthropic requests, a named preset, an
error path, and non-ASCII (CJK) input.

## License

MPL-2.0.
