#!/usr/bin/env sh
# onpar installer.
#   curl -fsSL https://dshakes.github.io/onpar/install.sh | sh
#
# Picks the best channel already present rather than installing a package manager
# in order to install a package. POSIX sh, because this runs before anything is
# set up.
#
# Every channel installs the distribution `onpar`, which provides the
# `onpar` command. The two names differ because PyPI refused `onpar` as too
# similar to the existing `click-llm` — so `pip install onpar` will never work,
# and the `-cli` suffix below is load-bearing rather than a typo. There is still no
# Homebrew formula, so there is no brew branch.
#
# No npm branch either, and that is a decision rather than an omission. `npx
# onpar` is a shim that execs uvx, then `uv tool run`, then `pipx run` — every
# channel this script already tries first. A branch for it could only ever be
# reached on a machine where all three are missing, which is exactly where the
# shim itself gives up. It would add node as a dependency and install nothing.
set -eu

say()  { printf '%s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

say "onpar installer"

if have uv; then
  say "-> uv detected; installing as a tool"
  uv tool install onpar
elif have pipx; then
  say "-> pipx detected"
  pipx install onpar
elif have python3; then
  ver=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
  case "$ver" in
    3.1[1-9]|3.[2-9][0-9]) : ;;
    *) die "python3 is $ver; onpar needs 3.11 or newer. Install uv instead: https://astral.sh/uv" ;;
  esac
  say "-> falling back to pip --user (python $ver)"
  # PEP 668: a Homebrew or distro python refuses `pip install` outright, and its
  # own error suggests `brew install onpar`, which does not exist. Catch it
  # and point at the channel that actually works rather than letting the reader
  # follow that advice into a second dead end.
  python3 -m pip install --user onpar || die \
    "pip refused to install into this python (see above; PEP 668 marks Homebrew and
distro pythons externally managed). Install uv and re-run: https://astral.sh/uv"
else
  die "no supported installer found. Install uv (https://astral.sh/uv), pipx, or Python 3.11+."
fi

if ! have onpar; then
  say ""
  say "Installed, but 'onpar' is not on PATH yet."
  say "Add your user bin directory to PATH, or run it without installing:"
  say "  uvx --from onpar onpar fit"
  exit 0
fi

say ""
say "Installed. Try:"
say "  onpar fit --context 32k --concurrency 8"
say "  onpar fit --explain qwen3-30b-a3b"
