"""Flags that must do what they say, on every branch that reads them.

`fit` has three output paths — the ranked table, the JSON dump, and
`--explain` — and its flags were wired to two of them.
"""

from __future__ import annotations

import json

import pytest

from clickllm.cli import main


def test_explain_with_quiet_still_explains(capsys):
    """`--quiet` is documented as "hide the NOT FEASIBLE section", and
    `--explain` has no such section — it is one model.

    The gate meant `fit --explain X --quiet` printed nothing and exited 0, which
    reads as "no answer" rather than as a suppressed list.
    """
    rc = main(["fit", "--explain", "llama-3.1-8b", "--quiet"])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "Llama 3.1 8B" in out, out
    assert "verdict:" in out


def test_quiet_still_hides_the_section_it_names(capsys):
    """The negative control: removing the gate must not make `--quiet` a no-op
    on the path it was written for."""
    main(["fit"])
    loud = capsys.readouterr().out
    main(["fit", "--quiet"])
    quiet = capsys.readouterr().out

    assert "NOT FEASIBLE" in loud
    assert "NOT FEASIBLE" not in quiet


def test_explain_with_json_emits_json(capsys):
    """`--explain` returned before the `--json` branch, so a script consuming
    this parsed prose as JSON."""
    main(["fit", "--explain", "llama-3.1-8b", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "llama-3.1-8b"
    assert "explain" in payload and payload["explain"]
    assert isinstance(payload["feasible"], bool)


def test_explain_with_both_flags_prefers_json(capsys):
    """`--json` and `--quiet` together is a script asking for machine output and
    less noise; emitting both blocks would corrupt the parse."""
    main(["fit", "--explain", "llama-3.1-8b", "--json", "--quiet"])
    out = capsys.readouterr().out
    json.loads(out)  # must parse — no prose prepended


@pytest.mark.parametrize("platform", ["sunos5", "aix7", "riscos"])
def test_desktop_on_an_unsupported_platform_is_a_sentence(platform, capsys, monkeypatch):
    """Already fixed; kept as a guard. `desktop.install()` raises `RuntimeError`
    naming the platform, and `main()`'s handler lists it — the convention is a
    sentence and a nonzero exit, never a traceback."""
    monkeypatch.setattr("sys.platform", platform)
    rc = main(["desktop"])
    assert rc == 2
    assert platform in capsys.readouterr().err
