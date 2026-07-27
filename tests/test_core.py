"""The Rust seam, from the Python side.

Two audiences here, and both matter:

- **Without the extension** — the common case for someone who ran
  `uvx clickllm fit`. Nothing may explode at import time, and asking for a Rust
  feature must produce one readable sentence naming what to install.
- **With the extension** — skipped when it is absent, so CI without a compiled
  wheel still runs green rather than silently not testing this.

The second set reads a log this test did not write by hand: the bytes come from
the Rust store, through the binding, into Python. A test that constructed the
file in Python would prove the two agree on a format Python invented.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from clickllm import core

HAS_EXT = core.available()
needs_ext = pytest.mark.skipif(not HAS_EXT, reason="_clickllm_core is not installed")


# --- the no-extension contract -------------------------------------------------


def test_importing_core_never_fails_even_without_the_extension():
    # `uvx clickllm fit` imports the package. If this module raised on import,
    # the zero-dependency promise would be broken by a feature nobody used.
    assert core.available() in (True, False)


def test_availability_and_reason_are_consistent():
    if core.available():
        assert core.why_unavailable() is None
    else:
        reason = core.why_unavailable()
        assert reason and "clickllm[core]" in reason


@pytest.mark.skipif(HAS_EXT, reason="only meaningful without the extension")
def test_a_missing_extension_names_what_to_install():
    with pytest.raises(core.SeamError) as e:
        core.redact("anything")
    msg = str(e.value)
    assert "pip install 'clickllm[core]'" in msg
    assert "clickllm fit" in msg, "must say what still works without it"


def test_version_is_none_or_a_known_one():
    v = core.version()
    assert v is None or v in core.COMPATIBLE


# --- with the extension --------------------------------------------------------


@needs_ext
def test_redaction_is_the_same_policy_the_gateway_applies():
    clean, counts = core.redact("mail ada@example.com about card 4539578763621486")
    assert "ada@example.com" not in clean
    assert clean == "mail <EMAIL> about card <CARD>"
    assert counts == {"email": 1, "card_number": 1}


@needs_ext
def test_an_order_id_survives_here_too():
    # The Luhn check is the reason. A second Python implementation would be the
    # thing that eventually forgets it and hollows out the eval set.
    clean, counts = core.redact("order 1234567890123456 shipped")
    assert "1234567890123456" in clean
    assert counts == {}


@needs_ext
def test_oversized_text_is_refused_rather_than_partially_scanned():
    with pytest.raises(ValueError, match="dropped"):
        core.redact("a" * (4 * 1024 * 1024 + 1))


@needs_ext
def test_an_absent_log_is_an_empty_list_and_creates_nothing(tmp_path: Path):
    log = tmp_path / "never" / "written"
    assert core.read_captures(str(log), b"k" * 32) == []
    assert not (tmp_path / "never").exists(), "reading must not touch the filesystem"


@needs_ext
def test_a_short_key_is_rejected_with_the_length(tmp_path: Path):
    with pytest.raises(ValueError, match="32 bytes, got 5"):
        core.read_captures(str(tmp_path / "log"), b"short")


@needs_ext
def test_a_generated_key_is_stable_and_owner_only(tmp_path: Path):
    p = tmp_path / "capture.key"
    first = core.load_or_create_key(str(p))
    assert len(first) == 32
    assert core.load_or_create_key(str(p)) == first, "regenerating would orphan the log"
    if sys.platform != "win32":
        assert oct(p.stat().st_mode)[-3:] == "600"


@needs_ext
def test_the_wrong_key_raises_capture_error_not_a_bare_runtime_error(tmp_path: Path):
    log = _written_by_rust(tmp_path)
    with pytest.raises(core.CaptureError, match="key"):
        core.read_captures(str(log), b"\x09" * 32)


@needs_ext
def test_a_log_written_by_rust_is_read_by_python_with_types_intact(tmp_path: Path):
    log = _written_by_rust(tmp_path)
    rows = core.read_captures(str(log), b"\x07" * 32)

    assert len(rows) == 2
    first, second = rows

    assert first["request_id"] == "req-a"
    assert first["model"] == "gpt-5"
    assert first["response"] == "Sent to <EMAIL>"
    assert first["prompt_tokens"] == 11
    assert first["completion_tokens"] == 4
    assert first["redacted"] == {"email": 2}

    # The message tree arrives as native Python, not a JSON string — clustering
    # reads it directly and should not have to parse it back.
    assert isinstance(first["messages"], list)
    assert first["messages"][0]["role"] == "user"
    assert "<EMAIL>" in first["messages"][0]["content"]
    assert "the invoice" in first["messages"][0]["content"]

    # Unreported usage stays None. A 0 here would read as "this request was free",
    # which is the one direction a cost figure must never err in.
    assert second["prompt_tokens"] is None
    assert second["completion_tokens"] is None


# --- fixture -------------------------------------------------------------------


def _written_by_rust(tmp_path: Path) -> Path:
    """Produce a capture log using the real Rust writer.

    Building a fixture in Python would only prove the binding agrees with a
    format Python made up. These bytes come from the code that runs in the
    gateway.
    """
    root = Path(__file__).resolve().parents[1]
    example = root / "clickllm-gateway" / "examples" / "write_captures.rs"
    if not example.exists():  # pragma: no cover - repo layout guard
        pytest.skip("writer example missing")
    log = tmp_path / "captures"
    r = subprocess.run(
        [
            "cargo",
            "run",
            "-q",
            "-p",
            "clickllm-gateway",
            "--example",
            "write_captures",
            "--",
            str(log),
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:  # pragma: no cover
        pytest.skip(f"cargo unavailable or failed: {r.stderr[-400:]}")
    return log
