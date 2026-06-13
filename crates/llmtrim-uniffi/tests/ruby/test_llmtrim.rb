# End-to-end tests for the llmtrim Ruby bindings.
#
# Run after generating the Ruby glue and building the cdylib:
#   crates/llmtrim-uniffi/scripts/generate-bindings.sh /tmp/gen   # emits ruby/llmtrim_ffi.rb
#   ruby -I/tmp/gen/ruby -r./llmtrim_ffi \
#     crates/llmtrim-uniffi/tests/ruby/test_llmtrim.rb
# with LD_LIBRARY_PATH (or DYLD_LIBRARY_PATH) pointing at the dir holding libllmtrim_ffi.
#
# Uses presets so it doesn't depend on ambient LLMTRIM_* environment config. Plain
# minitest (ships with Ruby) + the `ffi` gem that the generated glue requires.

require "minitest/autorun"
require "json"
require "llmtrim_ffi"

class TestLlmtrim < Minitest::Test
  def openai(content = "hello world")
    JSON.generate("model" => "gpt-4o",
                  "messages" => [{ "role" => "user", "content" => content }],
                  "max_tokens" => 5)
  end

  def test_compress_returns_projected_fields
    out = LlmtrimFfi.compress(openai, LlmtrimFfi::Provider::OPEN_AI, "safe")
    assert_equal "openai", out.provider
    assert_equal "gpt-4o", out.model
    assert out.tokenizer_exact
    assert out.input_tokens_before > 0
    refute_empty out.request_json
    JSON.parse(out.request_json) # still valid JSON
  end

  def test_agent_preset_compresses_a_tool_result
    diff = (0...40).map { |i| "diff --git a/f#{i}.rs b/f#{i}.rs\n@@ -1,2 +1,2 @@\n-old #{i}\n+new #{i}\n" }.join
    req = JSON.generate("model" => "claude-3-5-sonnet-20241022",
                        "messages" => [{ "role" => "user",
                                         "content" => [{ "type" => "tool_result", "tool_use_id" => "t", "content" => diff }] }],
                        "max_tokens" => 1024)
    out = LlmtrimFfi.compress(req, LlmtrimFfi::Provider::ANTHROPIC, "agent")
    assert_equal "anthropic", out.provider
    assert out.input_tokens_after < out.input_tokens_before
  end

  def test_auto_detect_provider_when_nil
    req = JSON.generate("system" => "s",
                        "messages" => [{ "role" => "user", "content" => "hi" }],
                        "max_tokens" => 5)
    out = LlmtrimFfi.compress(req, nil, "safe")
    assert_equal "anthropic", out.provider
  end

  def test_unknown_preset_raises
    assert_raises(LlmtrimFfi::LlmtrimError::UnknownPreset) do
      LlmtrimFfi.compress(openai, LlmtrimFfi::Provider::OPEN_AI, "no-such-preset")
    end
  end

  def test_invalid_json_raises
    assert_raises(LlmtrimFfi::LlmtrimError::Compress) do
      LlmtrimFfi.compress("not json", LlmtrimFfi::Provider::OPEN_AI, "safe")
    end
  end
end
