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


#: Commands whose entire job is answering a question an agent asks. Each must
#: emit parseable JSON, not a table an agent has to re-parse by column offset.
#: `version` and `models` are the two an agent calls first — "what am I talking
#: to" and "what can it run" — and neither had `--json` until now, on a product
#: whose README calls itself agent-first.
_AGENT_JSON_COMMANDS = (
    ("version", dict),
    ("models", list),
)


@pytest.mark.parametrize(("command", "shape"), _AGENT_JSON_COMMANDS)
def test_the_agent_facing_commands_emit_parseable_json(command, shape, capsys):
    """`--json` has to produce JSON, and only JSON.

    The failure this guards is not "the flag is missing" — it is a flag that
    exists and prints a human line first, so `json.loads` on the output dies at
    character 1. That is worse than no flag, because it looks supported.
    """
    assert main([command, "--json"]) == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)  # raises, loudly, if anything non-JSON was printed
    assert isinstance(parsed, shape), f"{command} --json returned {type(parsed).__name__}"
    assert parsed, f"{command} --json returned an empty {shape.__name__}"


def test_the_json_flag_does_not_change_what_is_true(capsys):
    """The machine answer and the human answer must be the same answer.

    Two renderings of one fact drift the moment they are computed separately —
    the shape this repo has been bitten by repeatedly (the README table, the
    site's throughput figures, the teaching table). An agent reading `--json`
    and a human reading the table must not be told different versions.
    """
    from clickllm import __version__

    assert main(["version", "--json"]) == 0
    machine = json.loads(capsys.readouterr().out)

    assert main(["version"]) == 0
    human = capsys.readouterr().out

    assert machine["version"] == __version__, (
        f"--json reports {machine['version']!r}, the package is {__version__!r}"
    )
    assert __version__ in human, "the human output does not name the installed version"
    assert machine["distribution"] in human, (
        "--json and the table disagree about the distribution name"
    )


def test_a_receipt_can_be_read_by_the_machine_it_is_addressed_to(tmp_path, capsys):
    """A receipt exists to be checked by someone who was not there.

    That audience is increasingly an agent, and until now the only way to read a
    verdict was to parse the rendered block — so the one artefact whose entire
    purpose is machine-checkable evidence could not be machine-read. The digest
    has to survive the round trip too, or "verify this yourself" is a slogan.
    """
    import sys

    sys.path.insert(0, "tests")
    from test_agent_surfaces import _good_receipt

    written = _good_receipt()
    path = tmp_path / "receipt.json"
    path.write_text(written.to_json())

    assert main(["receipt", str(path), "--json"]) == 0
    emitted = json.loads(capsys.readouterr().out)

    assert emitted["digest"] == json.loads(written.to_json())["digest"], (
        "the digest changed passing through --json, so a receipt round-tripped "
        "through the CLI would fail its own verification"
    )
