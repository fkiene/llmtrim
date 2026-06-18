# frozen_string_literal: true

# Gemspec for the llmtrim Ruby bindings. Assembled by scripts/build-gem.sh, which
# generates the UniFFI glue, patches it to load the bundled native library, drops the
# compiled cdylib into lib/llmtrim/, and runs `gem build`. The version comes from
# LLMTRIM_VERSION (set by the release pipeline) and defaults to the dev version.
#
# This builds a platform-specific (precompiled) gem: the native library is bundled, so
# users need no Rust toolchain. `gem build` is invoked with `--platform` by the script.

Gem::Specification.new do |s|
  s.name = "llmtrim"
  s.version = ENV.fetch("LLMTRIM_VERSION", "0.1.7.dev")
  # Platform-specific gem (bundled native lib). The build script sets this to the target
  # platform; unset means a generic gem (used only for local `gem build` smoke).
  s.platform = ENV["LLMTRIM_GEM_PLATFORM"] if ENV["LLMTRIM_GEM_PLATFORM"]
  s.summary = "Static, deterministic LLM prompt/payload compression that cuts input tokens 30-90%."
  s.description = "Native in-process bindings to the llmtrim-core compression engine " \
                  "(no network, no extra model calls), generated via UniFFI."
  s.authors = ["François Kiene"]
  s.homepage = "https://github.com/fkiene/llmtrim"
  s.license = "MPL-2.0"
  s.required_ruby_version = ">= 3.0"

  s.files = Dir["lib/**/*"] + ["README.md"]
  s.require_paths = ["lib"]

  # The generated glue is built on the ffi gem.
  s.add_dependency "ffi", "~> 1.15"

  s.metadata = {
    "source_code_uri" => "https://github.com/fkiene/llmtrim",
    "rubygems_mfa_required" => "true",
  }
end
