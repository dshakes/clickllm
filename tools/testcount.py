#!/usr/bin/env python3
"""Count the suite, and write that count onto every surface that publishes it.

The test count appears in three places — the README badge, the README prose, and
the landing page's stat strip. They drifted to 478 / 668 / 675 without anything
noticing, because no check compared them to each other or to the suite.

`tests/test_docs_lab.py` now compares all three. This script is the other half:
the one command that makes them right, so keeping them honest costs a command
rather than three careful edits.

    python3 tools/testcount.py            # report; exit 1 if stale
    python3 tools/testcount.py --write    # update every surface

Counting is delegated to the test runners themselves — `cargo test -- --list`
and `pytest --collect-only` — rather than to a regex over source files. Grepping
for `#[test]` undercounts this repo by 161, because parameterised and generated
cases have no attribute to find.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Where the number is published, and how to find it. The replacement rewrites
#: only the digits, so surrounding markup is never disturbed.
SURFACES: tuple[tuple[str, str], ...] = (
    ("README.md", r"(badge/tests-)(\d+)(-)"),
    ("README.md", r"(\*\*)(\d[\d,]*)( tests\.\*\*)"),
    ("site/index.html", r'(<div class="n grad">)(\d+)(</div><div class="l">tests)'),
)

#: The per-runner counts, which live in the Verification block's comments and
#: are NOT the total. They drifted too — the README said 249 Rust and 419 Python
#: after the weights crate was retired and four modules were added, so it was
#: wrong in both directions at once. Written from the same measurement as the
#: total, so all five numbers move together or the check fails.
SPLIT_SURFACES: tuple[tuple[str, str, str], ...] = (
    ("README.md", "rust", r"(cargo test --all\s+# )(\d+)( Rust)"),
    ("README.md", "python", r"(pytest -q\s+# )(\d+)( Python)"),
    # CLAUDE.md publishes the same two numbers in its gate block and was not
    # listed here, so it drifted furthest of all: it advertised 61 Rust and 27
    # Python against a real 227 and 702. It is the file every agent reads first,
    # which makes a wrong number there the most expensive one in the repo.
    ("CLAUDE.md", "rust", r"(# Rust gate \()(\d+)( tests\))"),
    ("CLAUDE.md", "python", r"(# Python gate \()(\d+)( tests\))"),
    # The README says the split a second time, in prose, three lines under the
    # gate block: "**1874 tests.** 1632 Python, 234 Rust." Nothing matched it,
    # so `--write` printed `rust count ok` while a stale 234 sat immediately
    # below the 242 it had just written. The comment above said five numbers
    # move together; there were seven, and the two nobody listed were the two
    # a reader meets first.
    ("README.md", "python", r"( tests\.\*\* )(\d[\d,]*)( Python, )"),
    ("README.md", "rust", r"( Python, )(\d[\d,]*)( Rust\.)"),
)


def rust_tests() -> int:
    """Rust tests, by asking each compiled test binary to list its own."""
    cargo = shutil.which("cargo")
    if cargo is None:
        raise RuntimeError("cargo not found; cannot count the Rust half")
    proc = subprocess.run(
        [cargo, "test", "--", "--list"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"cargo test -- --list failed:\n{proc.stderr[-400:]}")
    totals = [int(n) for n in re.findall(r"^(\d+) tests?, \d+ benchmark", proc.stdout, re.M)]
    if not totals:
        raise RuntimeError("cargo listed no test binaries; refusing to report zero")
    return sum(totals)


#: How to collect, in order of preference. The direct invocation first, because
#: it is fastest when pytest is installed — then the command CLAUDE.md actually
#: documents, because in this repo it usually is not.
#:
#: Without the fallback this script cannot run in the environment the repo tells
#: you to use, so keeping the counts honest went back to being three careful
#: edits — which is exactly the thing it exists to replace, and they drifted
#: again within a day.
_COLLECTORS: tuple[tuple[str, ...], ...] = (
    (sys.executable, "-m", "pytest", "-q", "--collect-only", "tests"),
    (
        "uv",
        "run",
        "--with",
        "pytest",
        "--with",
        "pyyaml",
        "--python",
        "3.13",
        "pytest",
        "-q",
        "--collect-only",
        "tests",
    ),
)


def python_tests() -> int:
    """Python tests, by collection — skips included, since they still exist."""
    attempts = []
    for cmd in _COLLECTORS:
        try:
            proc = subprocess.run(list(cmd), cwd=ROOT, capture_output=True, text=True, timeout=1800)
        except FileNotFoundError:
            attempts.append(f"{cmd[0]}: not on PATH")
            continue
        m = re.search(r"^(\d+) tests? collected", proc.stdout, re.M)
        if m:
            return int(m.group(1))
        attempts.append(f"{cmd[0]}: {(proc.stdout or proc.stderr)[-160:].strip()}")
    raise RuntimeError("could not read a collected count:\n  " + "\n  ".join(attempts))


def published_split() -> dict[tuple[str, str, str], int]:
    """What each per-runner comment currently claims, per site.

    Keyed by the whole site — file, runner *and* pattern — because each weaker
    key has already lost a number. Keyed by runner alone, a second file would
    overwrite the first. Keyed by `(file, runner)`, which is what this was, the
    README's prose sentence collided with the README's gate block: same file,
    same runner, different line. The surviving entry was checked and the other
    drifted, which is how a stale 234 sat three lines under a freshly written
    242 while the report said `ok`.
    """
    out = {}
    for rel, which, pattern in SPLIT_SURFACES:
        m = re.search(pattern, (ROOT / rel).read_text())
        if not m:
            raise RuntimeError(f"{rel}: no {which} count matched {pattern!r}")
        out[rel, which, pattern] = int(m.group(2).replace(",", ""))
    return out


def write_split(rust: int, py: int) -> list[str]:
    """Rewrite the per-runner comments. Returns what changed."""
    changed = []
    for rel, which, pattern in SPLIT_SURFACES:
        path = ROOT / rel
        text = path.read_text()
        n = rust if which == "rust" else py
        # `n` bound as a default, not captured: the closure outlives the loop
        # iteration that made it, and a late-bound `n` would write the last
        # runner's count onto every surface.
        new = re.sub(pattern, lambda m, n=n: f"{m.group(1)}{n}{m.group(3)}", text, count=1)
        if new != text:
            path.write_text(new)
            changed.append(f"{rel}:{which}")
    return changed


def published() -> dict[str, int]:
    """What each surface currently claims."""
    out = {}
    for rel, pattern in SURFACES:
        text = (ROOT / rel).read_text()
        m = re.search(pattern, text)
        if not m:
            raise RuntimeError(f"{rel}: no test count matched {pattern!r}")
        out[f"{rel}:{m.group(1)[:24]}"] = int(m.group(2).replace(",", ""))
    return out


def write(total: int) -> list[str]:
    """Rewrite the digits on every surface. Returns what changed."""
    changed = []
    for rel, pattern in SURFACES:
        path = ROOT / rel
        text = path.read_text()
        new = re.sub(pattern, lambda m: f"{m.group(1)}{total}{m.group(3)}", text, count=1)
        if new != text:
            path.write_text(new)
            changed.append(rel)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="update the surfaces in place")
    args = ap.parse_args()

    rust, py = rust_tests(), python_tests()
    total = rust + py
    print(f"suite: {rust} Rust + {py} Python = {total}")

    current = published()
    for where, n in current.items():
        mark = "ok" if n == total else f"STALE (says {n})"
        print(f"  {where:<52} {mark}")

    split = published_split()
    want = {k: (rust if k[1] == "rust" else py) for k in split}
    seen: dict[tuple[str, str], int] = {}
    for key, n in split.items():
        rel, which, _ = key
        seen[rel, which] = seen.get((rel, which), 0) + 1
        nth = f" #{seen[rel, which]}" if seen[rel, which] > 1 else ""
        label = f"{which} count{nth}"
        mark = "ok" if n == want[key] else f"STALE (says {n}, is {want[key]})"
        print(f"  {rel}:{label:<{52 - len(rel) - 1}} {mark}")

    if all(n == total for n in current.values()) and split == want:
        return 0
    if not args.write:
        print("\nstale. run: python3 tools/testcount.py --write")
        return 1
    changed = write(total) + write_split(rust, py)
    print("\nupdated: " + ", ".join(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
