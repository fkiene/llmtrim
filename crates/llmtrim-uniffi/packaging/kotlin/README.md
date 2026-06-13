# llmtrim (Kotlin / JVM)

Native, in-process bindings to the [llmtrim](https://github.com/fkiene/llmtrim)
compression engine for the JVM — cut LLM input tokens 30–90%, no network, no extra model
calls. The compiled engine is bundled in the jar (loaded via JNA from the classpath), so
no Rust toolchain is needed at runtime.

```kotlin
import uniffi.llmtrim_ffi.Provider
import uniffi.llmtrim_ffi.compress

val req = """{"model":"gpt-4o","messages":[{"role":"user","content":"…"}]}"""
val out = compress(req, Provider.OPEN_AI, "aggressive")
println("${out.inputTokensBefore} -> ${out.inputTokensAfter}")
// send out.requestJson to the provider
```

`compress(input, provider, preset)` — `provider` is `Provider.OPEN_AI`/`ANTHROPIC`/`GOOGLE`
or `null` to auto-detect; `preset` is a workload name or `null` for the environment config.
Throws `LlmtrimException.Compress` / `UnknownPreset`.

## Build

```bash
crates/llmtrim-uniffi/scripts/build-maven.sh build            # compile + jar
crates/llmtrim-uniffi/scripts/build-maven.sh publishToMavenLocal
```

The script generates the UniFFI Kotlin glue into `src/main/kotlin/`, places the optimized
cdylib under `src/main/resources/<os-arch>/` (where JNA resolves it on the classpath), and
runs Gradle. A release jar bundles every platform's library. Depends on
`net.java.dev.jna:jna`. License: AGPL-3.0-only.
