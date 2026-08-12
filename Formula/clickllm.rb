# Homebrew formula.
#   brew tap dshakes/tap https://github.com/dshakes/clickllm
#   brew install dshakes/tap/clickllm
class Clickllm < Formula
  include Language::Python::Virtualenv

  desc "Prove which open model can replace your closed one, then migrate safely"
  homepage "https://dshakes.github.io/clickllm/"
  url "https://github.com/dshakes/clickllm/archive/refs/tags/v1.0.0.tar.gz"
  # Filled in by the release workflow from the tarball GitHub actually published,
  # never from one built locally — the point of the checksum is that they match.
  sha256 "341c0cd859991d536044b389e7bb4dba8d1535646467ce88a7fe4a54b824df72"
  license "Apache-2.0"
  head "https://github.com/dshakes/clickllm.git", branch: "main"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    require "json"
    # --json is the contract other tools consume. If it stops being valid JSON,
    # or loses a key, the formula must fail rather than install something broken.
    parsed = JSON.parse(shell_output("#{bin}/clickllm fit --json"))
    assert parsed.key?("hardware"), "fit --json must report hardware"
    assert parsed.key?("feasible"), "fit --json must report feasibility"
    assert parsed.key?("runtime"),  "fit --json must recommend a runtime"

    # An unknown model is a clean error, not a traceback.
    assert_match(/error/, shell_output("#{bin}/clickllm fit --explain nope 2>&1", 2))
  end
end
