# Homebrew formula.
#   brew tap dshakes/tap https://github.com/dshakes/onpar
#   brew install dshakes/tap/onpar
class Onpar < Formula
  include Language::Python::Virtualenv

  desc "Prove which open model can replace your closed one, then migrate safely"
  homepage "https://dshakes.github.io/onpar/"
  url "https://github.com/dshakes/onpar/archive/refs/tags/v1.3.4.tar.gz"
  # Filled in by the release workflow from the tarball GitHub actually published,
  # never from one built locally — the point of the checksum is that they match.
  sha256 "f2fcbc672562ab315b266687d1f10f5c14a353c9e678cd85b1be154506323142"
  license "Apache-2.0"
  head "https://github.com/dshakes/onpar.git", branch: "main"

  depends_on "python@3.13"

  def install
    virtualenv_install_with_resources
  end

  test do
    require "json"
    # --json is the contract other tools consume. If it stops being valid JSON,
    # or loses a key, the formula must fail rather than install something broken.
    parsed = JSON.parse(shell_output("#{bin}/onpar fit --json"))
    assert parsed.key?("hardware"), "fit --json must report hardware"
    assert parsed.key?("feasible"), "fit --json must report feasibility"
    assert parsed.key?("runtime"),  "fit --json must recommend a runtime"

    # An unknown model is a clean error, not a traceback.
    assert_match(/error/, shell_output("#{bin}/onpar fit --explain nope 2>&1", 2))
  end
end
