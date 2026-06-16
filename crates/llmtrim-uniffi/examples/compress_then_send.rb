# Compress a request in-process, then send it yourself: no proxy, no CA trust.
#
# This is the integration route for environments where the HTTPS_PROXY interception
# can't run: sandboxed / serverless functions, and certificate-pinned HTTP clients that
# reject a MITM CA. You call compress() natively, then POST the shaped body with your
# own HTTP client. Nothing intercepts the connection.
#
# Run after installing the gem (built by scripts/build-gem.sh):
#   OPENAI_API_KEY=sk-... ruby crates/llmtrim-uniffi/examples/compress_then_send.rb

require "llmtrim"
require "json"
require "net/http"
require "uri"

# 1. A normal provider-shaped request body (here: OpenAI chat completions).
request = JSON.generate(
  "model" => "gpt-4o",
  "messages" => [
    { "role" => "system", "content" => "You are a terse assistant." },
    { "role" => "user", "content" => "Summarize the build log above." },
  ]
)

# 2. Compress in-process. The provider is Llmtrim::Provider::OPEN_AI (or nil to
#    auto-detect from the body); the preset is a workload name such as "aggressive" /
#    "agent" / "code" / "rag" / "safe" (nil uses the environment / config-file defaults).
out = Llmtrim.compress(request, Llmtrim::Provider::OPEN_AI, "aggressive")

puts "input tokens #{out.input_tokens_before} -> #{out.input_tokens_after}"
puts "provider=#{out.provider} model=#{out.model} tokenizer=#{out.tokenizer_label}"

# 3. Send the compressed body yourself. No proxy, no CA trust, no env-var setup.
#    This works inside a sandbox and against a pinned TLS client.
uri = URI("https://api.openai.com/v1/chat/completions")
http = Net::HTTP.new(uri.host, uri.port)
http.use_ssl = true
req = Net::HTTP::Post.new(uri)
req["Authorization"] = "Bearer #{ENV.fetch('OPENAI_API_KEY')}"
req["Content-Type"] = "application/json"
req.body = out.request_json

body = JSON.parse(http.request(req).body)
puts body.dig("choices", 0, "message", "content")
