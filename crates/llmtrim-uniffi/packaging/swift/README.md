# llmtrim (Swift / SwiftPM)

Native, in-process bindings to the [llmtrim](https://github.com/fkiene/llmtrim)
compression engine for Apple platforms (macOS, iOS), cutting LLM input tokens 30–90%, no
network, no extra model calls. The engine ships as a binary XCFramework, so consumers
build no Rust.

```swift
import Llmtrim

let req = #"{"model":"gpt-4o","messages":[{"role":"user","content":"…"}]}"#
let out = try compress(input: req, provider: .openAi, preset: "aggressive")
print("\(out.inputTokensBefore) -> \(out.inputTokensAfter)")
// send out.requestJson to the provider
```

`compress(input:provider:preset:)`: `provider` is `.openAi`/`.anthropic`/`.google` or
`nil` to auto-detect; `preset` is a workload name or `nil` for the environment config.
Throws `LlmtrimError.Compress` / `.unknownPreset`.

## Build

```bash
crates/llmtrim-uniffi/scripts/build-xcframework.sh   # macOS + Xcode only
swift build --package-path crates/llmtrim-uniffi/packaging/swift
```

The script builds a release static lib per Apple target (macOS arm64/x86_64, iOS device,
iOS simulator), generates the Swift API + FFI header/modulemap, and assembles
`llmtrimFFI.xcframework`. For release distribution, attach `llmtrimFFI.xcframework.zip` to
the GitHub release and switch `Package.swift` to the remote `binaryTarget` (url + checksum;
see the comment in `Package.swift`). License: MPL-2.0.
