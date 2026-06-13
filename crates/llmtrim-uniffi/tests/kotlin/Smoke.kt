// Kotlin binding smoke test — compiled with the generated llmtrim_ffi.kt (which uses JNA)
// and run against the cdylib in CI (JVM on Ubuntu). Proves the bindings load and
// `compress` works from Kotlin. See ci.yml `bindings-kotlin`. Not run locally (no JVM here).

import uniffi.llmtrim_ffi.LlmtrimException
import uniffi.llmtrim_ffi.Provider
import uniffi.llmtrim_ffi.compress

fun main() {
    val req = """{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}],"max_tokens":5}"""

    val out = compress(req, Provider.OPEN_AI, "safe")
    require(out.provider == "openai") { "expected openai, got ${out.provider}" }
    require(out.model == "gpt-4o") { "model" }
    require(out.inputTokensBefore > 0uL) { "tokens before should be > 0" }
    println("kotlin OK: ${out.inputTokensBefore} -> ${out.inputTokensAfter}")

    try {
        compress(req, Provider.OPEN_AI, "no-such-preset")
        error("expected UnknownPreset to throw")
    } catch (e: LlmtrimException.UnknownPreset) {
        println("kotlin error mapping OK")
    }
}
