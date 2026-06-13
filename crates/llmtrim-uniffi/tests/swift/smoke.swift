// Swift binding smoke test — compiled together with the generated llmtrim_ffi.swift and
// run against the cdylib in CI (macOS). Proves the bindings load and `compress` works
// from Swift. See ci.yml `bindings-swift`. Not run locally (no Swift toolchain here).

import Foundation

@main
struct Smoke {
    static func main() throws {
        let req = #"{"model":"gpt-4o","messages":[{"role":"user","content":"hello world"}],"max_tokens":5}"#

        let out = try compress(input: req, provider: .openAi, preset: "safe")
        precondition(out.provider == "openai", "expected openai, got \(out.provider)")
        precondition(out.model == "gpt-4o", "model")
        precondition(out.inputTokensBefore > 0, "tokens before should be > 0")
        print("swift OK: \(out.inputTokensBefore) -> \(out.inputTokensAfter)")

        do {
            _ = try compress(input: req, provider: .openAi, preset: "no-such-preset")
            fatalError("expected UnknownPreset to throw")
        } catch LlmtrimError.UnknownPreset {
            print("swift error mapping OK")
        }
    }
}
