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

from clickllm import catalog, cli, hardware
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


#: MCP tool calls, goldened for the same reason the CLI is — and added after a
#: refactor changed what agents receive without a single test noticing.
#:
#: `mcp._fit` moved onto the shared engine, its field names were preserved
#: carefully, and its *values* silently changed: `tokens_per_sec_estimate` 15 →
#: 14.5, `total_gb` 84.1 → 84.06. The CLI was protected by its golden; the agent
#: surface had nothing, and the conformance test compared field presence rather
#: than values. This is the missing half.
MCP_CASES: tuple[tuple[str, str, dict], ...] = (
    ("mcp_fit", "clickllm_fit", {"context": "32k", "concurrency": 8}),
    ("mcp_where", "clickllm_where", {"model": "llama-3.1-8b", "context": "32k"}),
    ("mcp_explain", "clickllm_explain", {"model_id": "llama-3.1-8b"}),
    ("mcp_catalog", "clickllm_catalog", {}),
)


def _run(argv: list[str], monkeypatch: pytest.MonkeyPatch, config_home: Path) -> str:
    """Run one command in-process against the synthetic machine."""
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    # `cli` imports `hardware` as a module and calls `hardware.detect()`, so
    # patching the module attribute reaches every caller. Patching
    # `cli.hardware.detect` would be the same object; this is the clearer spelling.
    #
    # catalog.load() also reads $XDG_CONFIG_HOME/clickllm/models.d — pointed at
    # a fresh empty tmp_path so a developer's real drop-in catalogue (or lack
    # of one) can't change what a golden records. Without this, `catalog
    # sources` and every command that loads the catalogue would encode
    # whatever happens to be in ~/.config on the machine that ran the test.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.delenv("CLICKLLM_CATALOG", raising=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            cli.main(argv)
        except SystemExit as e:  # argparse exits on --help and usage errors
            buf.write(f"\n[exit {e.code}]\n")
    return buf.getvalue()


def _normalize_host_paths(text: str, config_home: Path) -> str:
    """Replace the absolute paths `catalog-sources` prints with fixed tokens.

    The built-in catalogue path is wherever this checkout lives
    (`.../src/clickllm/models.json`) and the drop-in dir is under
    `$XDG_CONFIG_HOME`, which here is a per-test tmp_path. Both are real,
    correct, host-specific paths — exactly what makes them unfit for a golden
    that must match on every machine.
    """
    text = text.replace(str(catalog.CATALOG_PATH), "<repo>/src/clickllm/models.json")
    text = text.replace(str(config_home), "<config-home>")
    return text


@pytest.mark.parametrize("name,argv", CASES, ids=[n for n, _ in CASES])
def test_cli_output_is_unchanged(name: str, argv: list[str], monkeypatch, tmp_path):
    """Byte-for-byte, against a golden recorded before the engine refactor."""
    got = _run(argv, monkeypatch, tmp_path)
    if name == "catalog_sources":
        got = _normalize_host_paths(got, tmp_path)
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


def test_the_synthetic_machine_is_actually_used(monkeypatch, tmp_path):
    """If `detect` were not patched, these goldens would encode the developer's
    laptop and fail on every runner — the mistake this repo has made twice."""
    out = _run(["fit", "--context", "32k", "--concurrency", "8"], monkeypatch, tmp_path)
    assert "M4 Max" in out
    assert "128 GB" in out or "96 GB" in out


@pytest.mark.parametrize("name,tool,args", MCP_CASES, ids=[n for n, _, _ in MCP_CASES])
def test_mcp_tool_output_is_unchanged(name, tool, args, monkeypatch, tmp_path):
    """The agent surface, byte-for-byte.

    An agent cannot notice that a number moved. It has no golden of its own to
    compare against and no human reading the diff — which is exactly how
    `tokens_per_sec_estimate` went from 15 to 14.5 in a change whose commit
    message said the wire format was unchanged. It was: the *names* were.
    """
    import json

    from clickllm import mcp

    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("CLICKLLM_CATALOG", raising=False)

    reply = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
    )
    got = reply["result"]["content"][0]["text"] + "\n"
    path = GOLDEN / f"{name}.txt"

    if os.environ.get("CLICKLLM_REGENERATE_GOLDEN"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(got)
        pytest.skip(f"regenerated {path.name}")

    assert path.exists(), f"no golden for {name}; record with CLICKLLM_REGENERATE_GOLDEN=1"
    assert got == path.read_text(), (
        f"the {tool} tool no longer returns what it returned before. If intended, "
        "re-record — after reading the diff, because an agent will not."
    )
    json.loads(got)  # and it must still be JSON an agent can parse
