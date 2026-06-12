# Homebrew formula for llmtrim (build-from-source).
class Llmtrim < Formula
  desc "Static, deterministic LLM prompt/payload compressor"
  homepage "https://github.com/fkiene/llmtrim"
  url "https://github.com/fkiene/llmtrim/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "50c00c2ee3ed60f7f7491e28a94c2680afcd6d0be3e2a20d1e77d952f284f417"
  license "AGPL-3.0-only"
  head "https://github.com/fkiene/llmtrim.git", branch: "main"

  depends_on "rust" => :build

  def install
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "llmtrim", shell_output("#{bin}/llmtrim --version")
  end
end
