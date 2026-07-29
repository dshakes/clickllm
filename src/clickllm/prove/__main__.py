"""`python -m clickllm.prove` — the package's own self-check.

Every module here answers to `python -m clickllm.<mod>`; the package did not,
because a package needs this file to be executable. Same `demo()` the tests run.
"""

from __future__ import annotations

from . import demo

if __name__ == "__main__":
    demo()
