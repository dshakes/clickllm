"""What the CLI prints today, captured before anything moves underneath it.

These are approval tests, and their whole value is that they were recorded
*before* the engine refactor rather than after. A conformance test proves the
surfaces agree with each other; it cannot prove they still agree with what
users saw last week. Only a golden recorded beforehand does that.

The machine is synthetic on purpose. `clickllm fit` reads the host, so a golden
captured on a laptop would encode that laptop and fail on every CI runner —
which would be a test of the runner, the mistake this repo has now made twice
(a latency assertion about the fixture's speed, a refusal test about the
runner's load).

To re-record after an intentional change:

    CLICKLLM_REGENERATE_GOLDEN=1 \
        uv run --with pytest --with pyyaml pytest -q tests/test_cli_golden.py

Read the diff before committing it. A golden updated without being read is a
test that agrees with whatever the code now does, which is no test at all.
"""

from __future__ import annotations

import io
import os
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from clickllm import cli, hardware
from clickllm.hardware import Hardware

GOLDEN = Path(__file__).parent / "golden"
GB = 1024**3

#: One fixed machine for every golden. Big enough that interesting models fit
#: and the NOT FEASIBLE section is still populated — a machine where everything
#: fits would golden a code path nobody hits.
MACHINE = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * GB,
    usable_bytes=96 * GB,
    bandwidth_gbps=546.0,
    cores=16,
)

#: (name, argv). Hardware-dependent and hardware-independent commands both, so
#: a refactor that changes how hardware reaches the solver shows up here.
CASES: tuple[tuple[str, list[str]], ...] = (
    ("fit", ["fit", "--context", "32k", "--concurrency", "8"]),
    ("fit_json", ["fit", "--context", "32k", "--concurrency", "8", "--json"]),
    ("fit_explain", ["fit", "--explain", "llama-3.1-8b"]),
    ("where", ["where", "llama-3.1-8b", "--context", "32k"]),
    ("where_json", ["where", "llama-3.1-8b", "--context", "32k", "--json"]),
    ("advise", ["advise", "--context", "128k", "--concurrency", "16"]),
    ("build", ["build", "a support chatbot for 20 agents"]),
    ("models", ["models"]),
    ("catalog_sources", ["catalog-sources"]),
)


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> str:
    """Run one command in-process against the synthetic machine."""
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    # `cli` imports `hardware` as a module and calls `hardware.detect()`, so
    # patching the module attribute reaches every caller. Patching
    # `cli.hardware.detect` would be the same object; this is the clearer spelling.
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cli.main(argv)
        except SystemExit as e:  # argparse exits on --help and usage errors
            buf.write(f"\n[exit {e.code}]\n")
    return buf.getvalue()


@pytest.mark.parametrize("name,argv", CASES, ids=[n for n, _ in CASES])
def test_cli_output_is_unchanged(name: str, argv: list[str], monkeypatch):
    """Byte-for-byte, against a golden recorded before the engine refactor."""
    got = _run(argv, monkeypatch)
    path = GOLDEN / f"{name}.txt"

    if os.environ.get("CLICKLLM_REGENERATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(got)
        pytest.skip(f"regenerated {path.name}")

    assert path.exists(), (
        f"no golden for {name}. Record it with CLICKLLM_REGENERATE_GOLDEN=1, "
        "and read the diff before committing."
    )
    want = path.read_text()
    assert got == want, (
        f"`clickllm {' '.join(argv)}` no longer prints what it printed before "
        "the engine refactor. If the change is intended, re-record with "
        "CLICKLLM_REGENERATE_GOLDEN=1 — after reading the diff."
    )


def test_the_goldens_are_not_empty():
    """A recorder that captured nothing would make every case above pass.

    This is the control for the harness itself: an empty or missing golden is
    the failure mode that looks exactly like success.
    """
    if os.environ.get("CLICKLLM_REGENERATE_GOLDEN"):
        pytest.skip("regenerating")
    for name, _ in CASES:
        path = GOLDEN / f"{name}.txt"
        assert path.exists(), f"{name} was never recorded"
        assert len(path.read_text().strip()) > 40, (
            f"{path.name} is {len(path.read_text())} bytes — a golden that "
            "captured nothing agrees with anything"
        )


def test_the_synthetic_machine_is_actually_used(monkeypatch):
    """If `detect` were not patched, these goldens would encode the developer's
    laptop and fail on every runner — the mistake this repo has made twice."""
    out = _run(["fit", "--context", "32k", "--concurrency", "8"], monkeypatch)
    assert "M4 Max" in out
    assert "128 GB" in out or "96 GB" in out
