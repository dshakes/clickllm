#!/usr/bin/env python3
"""Set the release version everywhere it is published, in one command.

The version lives in eighteen places across eight files: two manifests, four
install-pin examples on three doc surfaces, and one "currently X" line. Cutting
a release by hand means six careful edits, and this repo has already proved what
that costs — `CLAUDE.md` advertised 61 tests against a real 227 because the one
surface nobody remembered was the one nothing checked.

    python3 tools/bump.py            # report the current state
    python3 tools/bump.py 0.1.6      # set it everywhere

A global find-and-replace is wrong here, and that is the whole reason this file
exists rather than a sed one-liner. Some occurrences are *pins* — an example the
reader copies, which must name the release being cut. Others are *history*:
"Fixed in 0.1.5, so a receipt now names the build that produced it" is a
statement about a past release and is falsified by bumping it. Only the patterns
below are touched, and each one names which kind it is.

`tests/test_docs_lab.py::test_version_pins_in_the_docs_match_the_version_we_ship`
is the other half: it fails the build when a pin drifts from the manifest, so
forgetting to run this is caught rather than shipped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_V = r"\d+\.\d+\.\d+"

#: (file, what it is, pattern). Group 2 is the version; groups 1 and 3 are kept
#: verbatim, so surrounding markup and prose are never disturbed.
SURFACES: tuple[tuple[str, str, str], ...] = (
    ("pyproject.toml", "python manifest", rf'(^version = ")({_V})(")'),
    ("npm/package.json", "npm manifest", rf'(^  "version": ")({_V})(")'),
    # The Rust workspace was NOT here, so it sat at 0.1.0 while the package
    # shipped 0.1.9. It works only because `core.COMPATIBLE` also said 0.1.0 —
    # two frozen numbers agreeing by accident. The moment anyone bumped one, the
    # compiled extension would refuse to load against its own package.
    ("Cargo.toml", "rust workspace", rf'(^version = ")({_V})(")'),
    # Intra-workspace dependency pins. These were NOT here either, and the
    # omission was invisible for the whole 0.1.x line: a pin of `0.1.0` is a
    # caret requirement, `>=0.1.0, <0.2.0`, so it matched every release this
    # project ever cut. It cannot match 1.0.0. The first version bump that
    # crossed a major boundary broke `cargo build` on a repo whose tests were
    # all green — which is to say, this project could never have released a
    # 1.0.0 without discovering it the hard way.
    (
        "onpar-gateway/Cargo.toml",
        "core dep pin",
        rf'(onpar-core = \{{ path = "\.\./onpar-core", version = ")({_V})(")',
    ),
    (
        "onpar-py/Cargo.toml",
        "core dep pin",
        rf'(onpar-core = \{{ path = "\.\./onpar-core", version = ")({_V})(")',
    ),
    (
        "onpar-py/Cargo.toml",
        "gateway dep pin",
        rf'(onpar-gateway = \{{ path = "\.\./onpar-gateway", version = ")({_V})(")',
    ),
    # Install pins — examples a reader copies. These must name the new release.
    ("README.md", "uvx pin", rf"(uvx --from onpar==)({_V})( onpar fit)"),
    ("README.md", "npx pin", rf"(npx onpar-cli@)({_V})( fit)"),
    ("README.md", "npx prose pin", rf"(`npx onpar-cli@)({_V})(` runs)"),
    ("README.md", "runs-exactly", rf"(runs `onpar` )({_V})( and nothing)"),
    ("README.md", "currently", rf"(fetch the newest release — currently \*\*)({_V})(\*\*)"),
    ("site/index.html", "uvx pin", rf"(uvx --from onpar==)({_V})( onpar fit)"),
    ("site/index.html", "npx pin", rf"(npx onpar-cli@)({_V})( fit)"),
    ("site/index.html", "npx note", rf"(runs onpar==)({_V})( exactly)"),
    ("site/docs/index.html", "uvx pin", rf"(uvx --from onpar==)({_V})( onpar fit)"),
    ("site/docs/index.html", "npx pin", rf"(npx onpar-cli@)({_V})( fit)"),
    ("site/docs/index.html", "npx note", rf"(<code>npx onpar-cli@)({_V})(</code> runs)"),
    ("site/docs/index.html", "runs-exactly", rf"(<code>onpar</code> )({_V})( and nothing)"),
)

#: Occurrences that must NOT move, asserted so a future pattern cannot start
#: matching them. Each is a claim about a release that already happened.
HISTORY: tuple[tuple[str, str], ...] = (
    ("README.md", r"Fixed in \d+\.\d+\.\d+, so a receipt now names"),
)


def current() -> dict[tuple[str, str], str]:
    """What every surface says right now."""
    out = {}
    for rel, what, pattern in SURFACES:
        m = re.search(pattern, (ROOT / rel).read_text(), re.M)
        if not m:
            raise RuntimeError(f"{rel}: the {what} pattern matched nothing ({pattern!r})")
        out[rel, what] = m.group(2)
    return out


def _history_snapshot() -> list[str]:
    return [
        m.group(0)
        for rel, pattern in HISTORY
        for m in [re.search(pattern, (ROOT / rel).read_text())]
        if m
    ]


def write(version: str) -> list[str]:
    """Set every surface. Returns what changed.

    Every rewrite is computed in memory and the history check runs against the
    result BEFORE a single file is touched. Checking afterwards — which is what
    this did first — meant a refusal left the tree half-bumped: some files on
    the new version, some on the old, and the release only discovering it later.
    A guard that fires after the damage is a report, not a guard.
    """
    # Accumulated per FILE, not per pattern. README.md alone has five patterns,
    # and computing each rewrite from the original text made them siblings that
    # overwrite each other — the last one written wins and the other four are
    # lost. The sequential read-modify-write this replaced did not have that
    # problem; making the write atomic reintroduced it, and only the post-write
    # verification below caught it (`INCONSISTENT: ['0.1.6', '0.1.7']`).
    proposed: dict[Path, str] = {}
    labels: list[str] = []
    for rel, what, pattern in SURFACES:
        path = ROOT / rel
        text = proposed.get(path, path.read_text())
        new_text = re.sub(
            pattern, lambda m, v=version: f"{m.group(1)}{v}{m.group(3)}", text, count=1, flags=re.M
        )
        if new_text != text:
            proposed[path] = new_text
            labels.append(f"{rel}:{what}")
    for rel, pattern in HISTORY:
        path = ROOT / rel
        original = path.read_text()
        was = re.search(pattern, original)
        now = re.search(pattern, proposed.get(path, original))
        if (was is None) != (now is None) or (was and now and was.group(0) != now.group(0)):
            raise RuntimeError(
                f"a pattern would rewrite a historical claim in {rel}, making it false:\n"
                f"  was: {was and was.group(0)}\n  now: {now and now.group(0)}\n"
                f"Nothing was written."
            )

    for path, text in proposed.items():
        path.write_text(text)
    return labels


def main() -> int:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and not re.fullmatch(_V, sys.argv[1])):
        print(__doc__)
        return 2

    state = current()
    for (rel, what), v in state.items():
        print(f"  {rel + ':' + what:<44} {v}")

    versions = set(state.values())
    if len(sys.argv) == 1:
        if len(versions) == 1:
            print(f"\nconsistent at {versions.pop()}")
            return 0
        print(f"\nINCONSISTENT: {sorted(versions)} — run: python3 tools/bump.py <version>")
        return 1

    target = sys.argv[1]
    changed = write(target)
    print(f"\nset to {target}: " + (", ".join(changed) if changed else "nothing to change"))
    now = set(current().values())
    if now != {target}:
        print(f"FAILED: surfaces still disagree — {sorted(now)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
