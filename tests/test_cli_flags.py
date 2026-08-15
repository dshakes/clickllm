"""Flags that must do what they say, on every branch that reads them.

`fit` has three output paths — the ranked table, the JSON dump, and
`--explain` — and its flags were wired to two of them.
"""

from __future__ import annotations

import json

import pytest

from onpar.cli import main


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
    assert rc == 1
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
    from onpar import __version__

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


def test_the_exit_code_says_which_kind_of_failure_it_was():
    """0 success, 1 the world did not cooperate, 2 you called me wrong.

    Every nonzero exit used to be `2` — argparse's *usage error* — so a caller
    could not tell "fix the invocation, do not retry" from "the call was fine,
    retry or report". An agent that cannot separate those either retries a
    malformed command forever or abandons a transient network failure.

    The two failing cases below must not collapse back to one number. A test
    asserting only `!= 0` would pass on exactly the behaviour this replaced,
    which is why both are pinned to their own value.
    """
    assert main(["fit"]) == 0, "a working invocation must be 0"

    # Runtime: the call is well-formed, the endpoint is not listening.
    assert main(["measure", "--endpoint", "http://127.0.0.1:1/v1", "--samples", "1"]) == 1, (
        "a connection failure is a runtime failure, not a usage error"
    )

    # Usage: no subcommand, which is what argparse itself reports as 2.
    assert main([]) == 2, "a missing required argument stays a usage error"


def test_every_surface_that_prints_throughput_says_it_is_an_estimate(monkeypatch, capsys):
    """A number nobody has measured must not be the one with no qualifier.

    `--explain` has carried "(roofline estimate, not measured)" all along and
    `--json` carries `estimate_basis`, but the default table — the surface
    nearly everyone reads — printed `~tok/s` bare. Two of three obeying a
    convention is the drift shape this repo keeps finding: a fact stated in
    several places and maintained in some of them.

    It matters more than a label usually would. `BANDWIDTH_EFFICIENCY` scales
    every figure in that column and has never been checked against a
    measurement (#222).
    """
    # A synthetic machine, because `fit` reads the host. Without this the
    # assertion below failed on macos-latest with "no feasible rows" — a fact
    # about the runner, not about onpar. `test_cli_golden.py` pins a machine
    # for exactly this reason and this test should have from the start; a test
    # that skips on CI is a test that does not run where it matters.
    from onpar import hardware
    from onpar.hardware import Hardware

    monkeypatch.setattr(
        hardware,
        "detect",
        lambda: Hardware(
            kind="apple",
            name="M4 Max",
            total_bytes=128 * 1024**3,
            usable_bytes=96 * 1024**3,
            bandwidth_gbps=546.0,
            cores=16,
        ),
    )

    assert main(["fit"]) == 0
    table = capsys.readouterr().out
    assert "roofline" in table.lower(), (
        "the default fit table prints ~tok/s with no indication it is unmeasured"
    )

    assert main(["fit", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    feasible = rows.get("feasible") or []
    assert feasible, "the synthetic machine below should fit something"
    assert "estimate_basis" in feasible[0], "--json dropped the estimate basis"


def test_measure_does_not_compare_a_measurement_to_a_different_model():
    """The roofline must describe the model the endpoint actually serves.

    `measure` picked the best-fitting quantisation for the machine and compared
    the measurement against that. Serving qwen2.5-coder:32b at q4 (16.4 GB)
    while the roofline described q8 (32.8 GB) produced a measured/roofline ratio
    of 1.79 — nothing decodes faster than its memory bandwidth, so the ratio was
    not merely wrong, it was impossible. That impossibility was the only signal
    that the two numbers described different models.

    `--quant` states what is served. Without it the basis is still a guess, and
    the point of this test is that the guess and the declaration disagree — so a
    caller who knows the quant has a reason to pass it.
    """
    from onpar import catalog, fit
    from onpar.hardware import Hardware

    gb = 1024**3
    machine = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * gb,
        usable_bytes=96 * gb,
        bandwidth_gbps=546.0,
        cores=16,
    )
    spec = next(m for m in catalog.load() if m.id == "qwen3-32b")

    guessed = fit.best_quant(spec, machine, 32768, 1)
    declared = fit.solve(spec, "q4", machine, 32768, 1)

    assert guessed is not None
    assert guessed.quant != "q4", (
        "this machine now best-fits q4, so the mismatch this guards is no longer "
        "reachable here — re-pick a model where the guess and a common served "
        "quant differ, or the test is watching nothing"
    )
    assert declared.tokens_per_sec > guessed.tokens_per_sec, (
        "a smaller quantisation must predict a higher roofline; if not, the "
        "comparison below proves nothing"
    )


def test_measure_cli_still_prints_the_measurement_alongside_the_roofline_basis(capsys, monkeypatch):
    """`cmd_measure` must render the actual measurement — samples, median,
    spread, usable/not-usable — on every path, `--quant` included.

    A prior version printed only the "roofline basis" line whenever a roofline
    was available, via an `elif` that made the basis line and the measurement
    report mutually exclusive. It also referenced `quant_basis` outside the
    `if args.model:` block that defined it, so a bare `measure --endpoint ...`
    with no `--model` raised `UnboundLocalError` instead of returning 1.
    """
    from onpar import measure as M
    from onpar.hardware import Hardware

    def fake_decode_once(endpoint, model, prompt, *, max_tokens, timeout, api_key=""):
        return M.Sample(tokens=20, decode_seconds=1.0, ttft_seconds=0.05)

    monkeypatch.setattr(M, "_decode_once", fake_decode_once)

    gb = 1024**3
    machine = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * gb,
        usable_bytes=96 * gb,
        bandwidth_gbps=546.0,
        cores=16,
    )
    monkeypatch.setattr("onpar.cli.hardware.detect", lambda: machine)

    rc = main(
        [
            "measure",
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--model",
            "qwen3-32b",
            "--quant",
            "q4",
            "--samples",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc in (0, 1), "well-formed and measured, not a usage error"
    assert "median" in out, "the measurement report itself must still print"
    assert "roofline basis: --quant q4" in out, "the basis line must still say which model"

    # The bare invocation with no --model must not crash on an undefined
    # `quant_basis`, and must still render the measurement.
    rc = main(["measure", "--endpoint", "http://127.0.0.1:9/v1", "--samples", "2"])
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "median" in out


def test_measure_rejects_a_quant_the_model_does_not_publish(capsys, monkeypatch):
    """`--quant` went straight to `fit.solve`, which indexes `QUANT_BITS[quant]`
    and raises a bare `KeyError` for anything not in that global table — a
    typo like `q4_k_m` printed `error: 'q4_k_m'` with no hint of what was
    expected. `launch.plan` already validates a declared quant against the
    model's own `spec.quants` (`{model.id} does not publish ...`); `measure`
    must reject the same way, naming the offending value and what qwen3-32b
    actually publishes.
    """
    from onpar.hardware import Hardware

    monkeypatch.setattr(
        "onpar.cli.hardware.detect",
        lambda: Hardware(
            kind="apple",
            name="M4 Max",
            total_bytes=128 * 1024**3,
            usable_bytes=96 * 1024**3,
            bandwidth_gbps=546.0,
            cores=16,
        ),
    )

    rc = main(
        [
            "measure",
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--model",
            "qwen3-32b",
            "--quant",
            "q4_k_m",
            "--samples",
            "2",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 1, "an unpublished quant is well-formed input that failed, not a usage error"
    assert "q4_k_m" in err, "the offending value must be in the message"
    assert "q4" in err and "q8" in err, "what the model actually publishes must be in the message"


def test_measure_json_carries_the_roofline_basis(monkeypatch, capsys):
    """`--json` and `--out` printed `roofline_tokens_per_sec` and
    `ratio_to_roofline` with no way to tell whether the roofline came from a
    declared `--quant` or a guessed best fit — the human render grew a
    "roofline basis" line but the machine-readable payload stayed silent, so a
    script reading the JSON still had the exact mismatch this feature exists
    to prevent.
    """
    from onpar import measure as M
    from onpar.hardware import Hardware

    def fake_decode_once(endpoint, model, prompt, *, max_tokens, timeout, api_key=""):
        return M.Sample(tokens=20, decode_seconds=1.0, ttft_seconds=0.05)

    monkeypatch.setattr(M, "_decode_once", fake_decode_once)
    monkeypatch.setattr(
        "onpar.cli.hardware.detect",
        lambda: Hardware(
            kind="apple",
            name="M4 Max",
            total_bytes=128 * 1024**3,
            usable_bytes=96 * 1024**3,
            bandwidth_gbps=546.0,
            cores=16,
        ),
    )

    rc = main(
        [
            "measure",
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--model",
            "qwen3-32b",
            "--quant",
            "q4",
            "--samples",
            "2",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert rc in (0, 1)
    assert payload["roofline_basis"] == "--quant q4", (
        "the JSON payload must say which quant the roofline describes, same as the "
        "human-readable render"
    )


def test_measure_discloses_a_declared_quant_that_does_not_fit_this_machine(monkeypatch, capsys):
    """A prior version treated `fit.solve`'s non-`None` return as proof the
    config fits, unlike the old `best_quant` path it replaced, which returned
    `None` when nothing did. `fit.solve` returns a `Fit` — with a computed
    `tokens_per_sec` — even when `fit_result.feasible` is `False`, so a
    declared `--quant` too large for the detected machine produced a roofline
    with no sign it described a config the machine could not hold.

    The fix disclosed rather than refused, matching `Fit.beyond_published_context`:
    the bandwidth arithmetic is still correct, so the number stays, but the
    basis line must say the config does not fit.
    """
    from onpar import measure as M
    from onpar.hardware import Hardware

    def fake_decode_once(endpoint, model, prompt, *, max_tokens, timeout, api_key=""):
        return M.Sample(tokens=20, decode_seconds=1.0, ttft_seconds=0.05)

    monkeypatch.setattr(M, "_decode_once", fake_decode_once)
    # A machine too small to hold qwen3-32b at fp16, so the declared quant
    # below is a published one that nonetheless does not fit here.
    monkeypatch.setattr(
        "onpar.cli.hardware.detect",
        lambda: Hardware(
            kind="apple",
            name="M1",
            total_bytes=8 * 1024**3,
            usable_bytes=6 * 1024**3,
            bandwidth_gbps=68.0,
            cores=8,
        ),
    )

    rc = main(
        [
            "measure",
            "--endpoint",
            "http://127.0.0.1:9/v1",
            "--model",
            "qwen3-32b",
            "--quant",
            "fp16",
            "--samples",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert rc in (0, 1)
    assert "roofline basis: --quant fp16" in out
    assert "does not fit this machine" in out, (
        "an infeasible declared quant must say so, not print a bare roofline as if "
        "the config were real"
    )
