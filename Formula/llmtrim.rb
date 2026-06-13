# Homebrew formula for llmtrim (build-from-source).
class Llmtrim < Formula
  desc "Static, deterministic LLM prompt/payload compressor"
  homepage "https://github.com/fkiene/llmtrim"
  url "https://github.com/fkiene/llmtrim/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "79fc349b3cc5c15716432c50392e6fe4562b6f13ad998fac34c346aafb846be4"
  license "AGPL-3.0-only"
  head "https://github.com/fkiene/llmtrim.git", branch: "main"

  depends_on "rust" => :build

  def install
    # The workspace root is a virtual manifest (no [package]); point cargo at the CLI
    # member crate that owns the `llmtrim` binary.
    system "cargo", "install", *std_cargo_args(path: "crates/llmtrim-cli")
  end

  test do
    assert_match "llmtrim", shell_output("#{bin}/llmtrim --version")
  end
end
