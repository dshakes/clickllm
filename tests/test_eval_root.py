"""ADR-0014 — the one MCP tool where an agent names a path.

`clickllm_prove` reads an eval set the caller names and the contents land in the
agent's context. That is a file-read primitive addressable by whatever is
steering the agent — which invariant 7 says may itself have come out of a
customer's request log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clickllm import mcp
from clickllm.mcp import _within_eval_root, eval_root


def _call(path: str):
    r = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "clickllm_prove", "arguments": {"eval_set": path}},
        }
    )
    return r.get("result", {})


@pytest.mark.parametrize(
    "path",
    [
        "/etc/hosts",
        "/etc/passwd",
        "../../../../etc/hosts",
        "~/.ssh/known_hosts",
        "~/.aws/credentials",
    ],
)
def test_a_path_outside_the_root_is_refused(path, tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    with pytest.raises(ValueError, match="outside the eval root"):
        _within_eval_root(path)


def test_a_symlink_cannot_walk_out_of_the_root(tmp_path, monkeypatch):
    """Resolved *before* the comparison. Checking the literal path would let a
    symlink inside the root point at anything outside it."""
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}")
    link = tmp_path / "innocent.json"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="outside the eval root"):
        _within_eval_root(str(link))


def test_the_refusal_names_the_root_and_the_variable(tmp_path, monkeypatch):
    """A refusal nobody can act on is only marginally better than the read."""
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    with pytest.raises(ValueError) as caught:
        _within_eval_root("/etc/hosts")
    msg = str(caught.value)
    assert str(tmp_path) in msg
    assert "CLICKLLM_EVAL_ROOT" in msg
    assert "CLI is unrestricted" in msg, "the asymmetry has to be discoverable"


def test_a_path_inside_the_root_is_read(tmp_path, monkeypatch):
    """The negative control, and the one that matters: a guard that refuses
    everything satisfies every test above."""
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    target = tmp_path / "evalset.json"
    target.write_text("{}")

    assert _within_eval_root(str(target)) == target.resolve()
    assert _within_eval_root("evalset.json") == target.resolve()


def test_a_nested_path_inside_the_root_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    nested = tmp_path / "sets" / "q3" / "evalset.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}")
    assert _within_eval_root(str(nested)) == nested.resolve()


def test_the_default_root_is_the_working_directory(monkeypatch):
    """Absent the variable, the behaviour is the safe default rather than the
    permissive one — the whole reason this is opt-in rather than opt-out."""
    monkeypatch.delenv("CLICKLLM_EVAL_ROOT", raising=False)
    assert eval_root() == Path.cwd().resolve()


def test_the_tool_refuses_rather_than_returning_file_contents(tmp_path, monkeypatch):
    """End to end through the MCP surface, because the guard being in a helper
    is not the same as the guard being on the path the agent takes."""
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path))
    result = _call("/etc/hosts")
    body = json.dumps(result)
    assert result.get("isError"), body
    assert "outside the eval root" in body
    assert "localhost" not in body, "no file content may appear in the response"


def test_the_cli_is_deliberately_unrestricted(tmp_path, monkeypatch):
    """ADR-0014's asymmetry, asserted so nobody 'fixes' the inconsistency.

    The person typing `clickllm prove <path>` is the person the confinement
    exists to protect. Confining a command a human typed would be theatre; the
    two surfaces have genuinely different threat models.
    """
    monkeypatch.setenv("CLICKLLM_EVAL_ROOT", str(tmp_path / "elsewhere"))
    outside = tmp_path / "evalset.json"
    outside.write_text(json.dumps({"items": []}))

    from clickllm import cli

    # It must fail on the *contents* (no items), not on the path.
    rc = cli.main(["prove", str(outside)])
    assert rc != 0
