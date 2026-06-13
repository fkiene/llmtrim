# frozen_string_literal: true

# llmtrim — static, deterministic LLM prompt/payload compression.
#
# Thin entry point: load the UniFFI-generated bindings (which the build step patched to
# load the native library bundled inside this gem) and re-expose them under `Llmtrim`.
#
#   require "llmtrim"
#   out = Llmtrim.compress(request_json, Llmtrim::Provider::OPEN_AI, "aggressive")
#   out.input_tokens_before # => Integer
#
# The compression runs natively in-process via the bundled `llmtrim-core` engine.

require_relative "llmtrim/llmtrim_ffi"

# `LlmtrimFfi` is the module name UniFFI derives from the crate; alias it to the friendlier
# `Llmtrim` without breaking the generated internals.
Llmtrim = LlmtrimFfi
