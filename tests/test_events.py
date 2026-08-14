"""Diagnostics that stay on the machine, and stay out of the way.

The Rust datapath runs every fallible operation in a `tracing` span. The Python
control plane — the half that solves, collects, judges and issues receipts —
emitted nothing at all, so a forty-minute prove run that refused at the end left
no record of which part was slow.

Most of what is asserted here is about restraint rather than emission: silent
unless asked, no network under any configuration, and never changing what the
program does.
"""

from __future__ import annotations

import io
import logging
import pathlib
import socket

import pytest

from clickllm import events
from clickllm.events import LOGGER, event, span


@pytest.fixture
def captured(monkeypatch):
    """Route events into a buffer, leaving the real logger untouched."""
    log = logging.getLogger(LOGGER)
    saved, saved_level, saved_prop = list(log.handlers), log.level, log.propagate
    log.handlers.clear()
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    yield buf
    log.handlers.clear()
    log.handlers.extend(saved)
    log.setLevel(saved_level)
    log.propagate = saved_prop


# --- restraint -------------------------------------------------------------------


def test_nothing_is_emitted_until_asked(monkeypatch, capsys):
    """A tool that starts printing spans at people is a tool they turn off."""
    log = logging.getLogger(LOGGER)
    saved = list(log.handlers)
    log.handlers.clear()
    try:
        monkeypatch.delenv(events.ENV, raising=False)
        assert events.configure() is False
        with span("fit", model="x") as extra:
            extra["chosen"] = "y"
        event("prove", items=10)
        out = capsys.readouterr()
        assert out.out == "" and out.err == ""
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)


def test_a_disabled_span_still_yields_a_usable_dict(monkeypatch):
    """Otherwise every call site needs a branch, and one of them gets it wrong."""
    log = logging.getLogger(LOGGER)
    saved = list(log.handlers)
    log.handlers.clear()
    try:
        with span("fit") as extra:
            extra["chosen"] = "qwen3-32b"  # must not raise
        assert extra == {"chosen": "qwen3-32b"}
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)


def test_configure_is_idempotent(monkeypatch, capsys):
    """The classic way a CLI ends up printing everything twice."""
    log = logging.getLogger(LOGGER)
    saved = list(log.handlers)
    log.handlers.clear()
    try:
        monkeypatch.setenv(events.ENV, "debug")
        assert events.configure() is True
        assert events.configure() is True
        assert len(log.handlers) == 1
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)


def test_a_typo_in_the_environment_variable_does_not_break_a_run(monkeypatch, tmp_path):
    """This is diagnostics. A bad value means no diagnostics, never no answer."""
    log = logging.getLogger(LOGGER)
    saved = list(log.handlers)
    log.handlers.clear()
    try:
        # An unwritable path: no crash, no events.
        assert events.configure(str(tmp_path / "no" / "such" / "dir" / "x.log")) is False
        # A writable one works.
        assert events.configure(str(tmp_path / "x.log")) is True
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)


def test_it_does_not_reconfigure_logging_for_the_whole_process(monkeypatch, tmp_path):
    """A library that touches the root logger is one people vendor around."""
    log = logging.getLogger(LOGGER)
    saved, root_before = list(log.handlers), list(logging.getLogger().handlers)
    log.handlers.clear()
    try:
        events.configure(str(tmp_path / "x.log"))
        assert logging.getLogger().handlers == root_before
        assert log.propagate is False
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)


# --- zero egress (NFR-2) ---------------------------------------------------------


def test_emitting_an_event_never_opens_a_socket(monkeypatch, captured):
    """Captured traffic is the most sensitive data a customer has, and an event
    carrying a cluster name or a model id is derived from it. There is no
    exporter here and no configuration that adds one — this is the test that
    stops one being added by accident.
    """
    opened = []

    class Tripwire(socket.socket):
        def __init__(self, *a, **kw):
            opened.append(a)
            raise AssertionError("an event tried to open a socket")

    monkeypatch.setattr(socket, "socket", Tripwire)
    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **kw: pytest.fail("event opened a connection")
    )

    events.configure("debug")
    with span("prove", cluster="refunds") as extra:
        extra["items"] = 400
    event("receipt", digest="abc123")
    assert not opened
    assert "prove.ok" in captured.getvalue()


def test_no_exporter_or_endpoint_setting_exists():
    """The module has no way to be pointed at a collector. Asserted rather than
    assumed, because "we would never add one" is not a property.

    Read from the parsed imports, not from the source text. The first version of
    this searched the file for substrings after splitting on the docstring
    quotes — which excluded the import block entirely, so adding
    `import urllib.request` left it passing. It found nothing because it was
    looking nowhere.
    """
    import ast

    src = pathlib.Path(events.__file__ or "")
    tree = ast.parse(src.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    networked = {"urllib", "requests", "http", "socket", "httpx", "ssl", "smtplib", "ftplib"}
    assert not (imported & networked), (
        f"{sorted(imported & networked)} reached the events module — there is no "
        "destination for an event but this machine (NFR-2)"
    )
    # And nothing that would carry one in indirectly.
    assert "opentelemetry" not in imported and "structlog" not in imported, (
        "clickllm fit must work under uvx with zero runtime dependencies"
    )


# --- what it records -------------------------------------------------------------


def test_a_span_records_its_fields_its_result_and_its_duration(captured):
    with span("engine.fit", context="32k", concurrency=8) as extra:
        extra["chosen"] = "qwen3-32b"
    out = captured.getvalue()
    assert "engine.fit.start" in out and "engine.fit.ok" in out
    assert "context=32k" in out and "concurrency=8" in out
    assert "chosen=qwen3-32b" in out, "a field added inside the body was lost"
    assert "ms=" in out


def test_a_failure_is_recorded_and_re_raised(captured):
    """A span that swallowed would turn a diagnostic into a behaviour change,
    which is the one thing observability must never do."""
    with pytest.raises(ValueError, match="nope"), span("prove", set_id="abc"):
        raise ValueError("nope")
    out = captured.getvalue()
    assert "prove.failed" in out and "error=ValueError" in out and "set_id=abc" in out


def test_fields_are_ordered_so_two_runs_diff_cleanly(captured):
    event("x", zebra=1, apple=2, mango=3)
    line = captured.getvalue().strip()
    assert line.endswith("apple=2 mango=3 zebra=1"), line


def test_a_missing_value_is_absent_not_the_word_none(captured):
    """ "The judge was not used" and "the judge was None" read differently to a
    human scanning a log, and only the first is a fact."""
    event("prove", judge=None, items=12)
    out = captured.getvalue()
    assert "judge" not in out and "None" not in out
    assert "items=12" in out


# --- the operations actually emit ------------------------------------------------


def test_the_engine_emits_what_it_decided(captured):
    from clickllm import engine
    from clickllm.hardware import Hardware

    gb = 1024**3
    hw = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * gb,
        usable_bytes=96 * gb,
        bandwidth_gbps=546.0,
        cores=16,
    )
    engine.fit(context="32k", concurrency=8, hw=hw)
    out = captured.getvalue()
    assert "engine.fit.ok" in out
    assert "machine=M4 Max" in out, "the answer's identifiers must reach the log"
    assert "chosen=" in out and "runtime=" in out


def test_a_prove_run_emits_its_shape(captured):
    from clickllm.prove import EvalItem, suite

    items = [
        EvalItem(item_id=str(i), cluster="c", prompt=f"p{i}", baseline="a", candidate="a")
        for i in range(20)
    ]
    suite(items, shares={"c": 1.0}, issued="2026-08-12")
    out = captured.getvalue()
    assert "prove.suite" in out
    assert "items=20" in out and "clusters=1" in out and "movable=" in out


# --- discoverability -------------------------------------------------------------


def test_a_failure_points_at_the_diagnostics(monkeypatch, capsys):
    """The one moment a user needs to know these exist.

    There is no env-var section in the docs to find `CLICKLLM_LOG` in, so a
    feature nobody can discover is a feature nobody has. Pointing at it from the
    failure path costs a line and lands exactly when it is useful.
    """
    from clickllm import cli

    monkeypatch.delenv(events.ENV, raising=False)
    assert cli.main(["prove", "/definitely/not/here.json"]) == 1
    err = capsys.readouterr().err
    assert "error:" in err
    assert events.ENV in err and "debug" in err
    assert "Nothing leaves this machine" in err, (
        "the advice has to say where the trace goes, or it reads like telemetry"
    )


def test_the_hint_is_suppressed_when_the_user_already_has_it_on(monkeypatch, capsys):
    """Otherwise the advice is 'do what you are already doing'."""
    from clickllm import cli

    monkeypatch.setenv(events.ENV, "debug")
    assert cli.main(["prove", "/definitely/not/here.json"]) == 1
    assert events.ENV not in capsys.readouterr().err
