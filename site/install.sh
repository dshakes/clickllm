#!/usr/bin/env sh
# clickllm installer.
#   curl -fsSL https://dshakes.github.io/clickllm/install.sh | sh
#
# Picks the best channel already present rather than installing a package manager
# in order to install a package. POSIX sh, because this runs before anything is
# set up.
set -eu

say()  { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

say "clickllm installer"

if have uv; then
  say "-> uv detected; installing as a tool"
  uv tool install clickllm
elif have brew; then
  say "-> Homebrew detected"
  brew install dshakes/tap/clickllm
elif have pipx; then
  say "-> pipx detected"
  pipx install clickllm
elif have python3; then
  ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  case "$ver" in
    3.1[1-9]|3.[2-9][0-9]) : ;;
    *) die "python3 is $ver; clickllm needs 3.11 or newer. Install uv instead: https://astral.sh/uv" ;;
  esac
  say "-> falling back to pip --user (python $ver)"
  python3 -m pip install --user clickllm
else
  die "no supported installer found. Install uv (https://astral.sh/uv), Homebrew, pipx, or Python 3.11+."
fi

if ! have clickllm; then
  say ""
  say "Installed, but 'clickllm' is not on PATH yet."
  say "Add your user bin directory to PATH, or run it without installing:"
  say "  uvx clickllm fit"
  exit 0
fi

say ""
say "Installed. Try:"
say "  clickllm fit --context 32k --concurrency 8"
say "  clickllm fit --explain qwen3-30b-a3b"
