"""ADR-0014, extended to `onpar_distill`.

`onpar_prove` was the one MCP tool where an agent names a filesystem path
and the contents land in its context — until `onpar_distill` grew a
`captures` argument that decrypts whatever capture log it is pointed at and
returns the prompts and baselines inside it. Same shape, same guard: confined
to the capture root (`ONPAR_HOME`), not the whole filesystem.
"""

from __future__ import annotations

import json

import pytest

from onpar import mcp
from onpar.mcp import _within_capture_root


def _call(captures: str):
    r = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "onpar_distill", "arguments": {"captures": captures}},
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
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    with pytest.raises(ValueError, match="outside the capture root"):
        _within_capture_root(path)


def test_a_symlink_cannot_walk_out_of_the_root(tmp_path, monkeypatch):
    """Resolved *before* the comparison. Checking the literal path would let a
    symlink inside the root point at anything outside it."""
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    outside = tmp_path.parent / "outside.log"
    outside.write_bytes(b"not a real capture log")
    link = tmp_path / "innocent.log"
    link.symlink_to(outside)

    with pytest.raises(ValueError, match="outside the capture root"):
        _within_capture_root(str(link))


def test_the_refusal_names_the_root_and_the_variable(tmp_path, monkeypatch):
    """A refusal nobody can act on is only marginally better than the read."""
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    with pytest.raises(ValueError) as caught:
        _within_capture_root("/etc/hosts")
    msg = str(caught.value)
    assert str(tmp_path) in msg
    assert "ONPAR_HOME" in msg
    assert "CLI is unrestricted" in msg, "the asymmetry has to be discoverable"


def test_a_path_inside_the_root_is_read(tmp_path, monkeypatch):
    """The negative control, and the one that matters: a guard that refuses
    everything satisfies every test above."""
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    target = tmp_path / "captures.log"
    target.write_bytes(b"not a real capture log")

    assert _within_capture_root(str(target)) == target.resolve()
    assert _within_capture_root("captures.log") == target.resolve()


def test_a_nested_path_inside_the_root_is_read(tmp_path, monkeypatch):
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    nested = tmp_path / "archive" / "2026-q1" / "captures.log"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"not a real capture log")
    assert _within_capture_root(str(nested)) == nested.resolve()


def test_the_tool_refuses_rather_than_returning_file_contents(tmp_path, monkeypatch):
    """End to end through the MCP surface, because the guard being in a helper
    is not the same as the guard being on the path the agent takes."""
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    result = _call("/etc/hosts")
    body = json.dumps(result)
    assert result.get("isError"), body
    assert "outside the capture root" in body
    assert "localhost" not in body, "no file content may appear in the response"


def test_the_default_captures_argument_is_unaffected(tmp_path, monkeypatch):
    """Absent a `captures` argument, `_distill` still falls back to the standard
    log under the capture root rather than routing through the guard — the
    guard exists for caller-named paths, not the default one."""
    monkeypatch.setenv("ONPAR_HOME", str(tmp_path))
    result = _call("")
    body = json.dumps(result)
    assert result.get("isError"), body
    assert "outside the capture root" not in body
    assert "no capture log at" in body
