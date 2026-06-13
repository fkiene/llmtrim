# llmtrim-uniffi

[UniFFI](https://mozilla.github.io/uniffi-rs/) bindings over [`llmtrim-core`] — one Rust
definition, idiomatic in-process bindings for **Python, Ruby, Swift and Kotlin**. The
compression runs natively in the caller's process (no server, no async).

## API

A deliberately flat surface over the engine:

```rust
fn compress(
    input: String,                 // a provider-shaped request body (JSON)
    provider: Option<Provider>,    // OpenAi | Anthropic | Google, or None to auto-detect
    preset: Option<String>,        // "aggressive" | "agent" | "code" | "rag" | "safe" | …
                                   // None = config from the environment / config file
) -> Result<CompressOutput, LlmtrimError>
```

`CompressOutput` carries the compressed `request_json`, the resolved `provider`/`model`,
the tokenizer label/exactness, and the before/after/frozen input-token counts. Embedders
that need the full rehydration plan or per-stage reports should depend on [`llmtrim-core`]
directly in Rust.

## Python

```bash
# Build a self-contained wheel (cdylib + generated glue):
crates/llmtrim-uniffi/scripts/build-wheel.sh --release
pip install target/wheels/llmtrim-*.whl
```

```python
import llmtrim, json

req = json.dumps({"model": "gpt-4o",
                  "messages": [{"role": "user", "content": "…"}]})
out = llmtrim.compress(req, llmtrim.Provider.OPEN_AI, "aggressive")
print(out.input_tokens_before, "->", out.input_tokens_after)
# send out.request_json to the provider
```

> **Why `build-wheel.sh` and not plain `maturin build`:** maturin's `bindings = "uniffi"`
> auto-packaging is sensitive to the maturin↔uniffi version pair. With maturin 1.14 +
> uniffi 0.31 it builds the native library into the wheel but omits the generated Python
> glue (empty package `__init__.py`). The script runs maturin, then injects the freshly
> generated bindings and repacks the wheel with valid RECORD hashes. Remove it once the
> auto path packages cleanly.

## Ruby / Swift / Kotlin

All targets generate from the same built library — no extra Rust. The generated glue is a
build artifact (its checksums are pinned to the library ABI), so it is regenerated per
release rather than committed:

```bash
crates/llmtrim-uniffi/scripts/generate-bindings.sh out/   # python, ruby, swift, kotlin
```

> **Generation needs an unstripped library.** Library-mode `uniffi-bindgen` reads metadata
> symbols from the cdylib, but the workspace release profile sets `strip = true`. The script
> therefore generates from the (unstripped) debug build; the native library you *ship* can be
> a stripped `cargo build --release -p llmtrim-uniffi` cdylib — the glue loads it by name.

Ruby (verified) — put `libllmtrim_ffi.so` on the load path and:

```ruby
require_relative "llmtrim_ffi"
require "json"
out = LlmtrimFfi.compress(
  JSON.generate({model: "gpt-4o", messages: [{role: "user", content: "…"}]}),
  LlmtrimFfi::Provider::OPEN_AI, "aggressive")
puts "#{out.input_tokens_before} -> #{out.input_tokens_after}"
```

Swift emits `llmtrim_ffi.swift` + an FFI header and modulemap; Kotlin emits
`uniffi/.../llmtrim_ffi.kt` (which loads the cdylib via JNA). CI compiles and runs a smoke
for both — Swift on macOS (`swiftc` against the modulemap), Kotlin on a JVM (`kotlinc` +
JNA) — so a binding break is caught in all four languages (see `tests/swift`, `tests/kotlin`
and the `bindings*` jobs in `.github/workflows/ci.yml`).

## Publishable packages

Each ships the compiled engine bundled, so consumers need no Rust toolchain:

| Target | Build | Package | Verified |
|--------|-------|---------|----------|
| Python (PyPI) | `scripts/build-wheel.sh` | wheel | locally |
| Ruby (gem) | `scripts/build-gem.sh` | `packaging/ruby/` | locally |
| Kotlin/JVM (Maven) | `scripts/build-maven.sh` | `packaging/kotlin/` | locally |
| Swift (SwiftPM) | `scripts/build-xcframework.sh` | `packaging/swift/` | macOS CI only |

Each `packaging/<lang>/README.md` has the usage + publish details.

[`llmtrim-core`]: ../llmtrim-core
