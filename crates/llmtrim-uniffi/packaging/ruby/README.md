# llmtrim (Ruby)

Native, in-process bindings to the [llmtrim](https://github.com/fkiene/llmtrim)
compression engine, cutting LLM input tokens 30–90% with zero extra model calls, no network,
no server. The compiled engine is bundled in the gem, so no Rust toolchain is needed.

```ruby
require "llmtrim"
require "json"

req = JSON.generate("model" => "gpt-4o",
                    "messages" => [{ "role" => "user", "content" => "…" }])
out = Llmtrim.compress(req, Llmtrim::Provider::OPEN_AI, "aggressive")
puts "#{out.input_tokens_before} -> #{out.input_tokens_after}"
# send out.request_json to the provider
```

`compress(input, provider, preset)`: `provider` is `Llmtrim::Provider::OPEN_AI` /
`ANTHROPIC` / `GOOGLE` or `nil` to auto-detect; `preset` is a workload name
(`"aggressive"`, `"agent"`, `"code"`, `"rag"`, `"safe"`, …) or `nil` for the environment
config. Raises `Llmtrim::LlmtrimError::Compress` / `UnknownPreset` on error.

Built with `crates/llmtrim-uniffi/scripts/build-gem.sh` (platform-specific gem with the
bundled native library). License: MPL-2.0.
