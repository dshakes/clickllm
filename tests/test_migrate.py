"""The loop, and the one thing it must never do.

A scheduled job that walks the chain is only safe to leave running because it
cannot promote itself. Most of these tests are about that: whatever the evidence
says, the stage on disk does not move, and the thing that moves it is a person
going through the gateway's control surface (invariant 8).

The rest are about resuming, since a loop that forgets where it was every run is
a loop that never gets past the first rung.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from clickllm import migrate
from clickllm.prove import EvalItem, suite

ROOT = Path(__file__).resolve().parents[1]


def _rows(n_text: int = 60, n_tool: int = 0) -> list[dict]:
    rows = [
        {
            "request_id": f"r{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"summarise {i}"}],
            "response": "a summary",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [],
            "tool_calls": [],
            "response_format": None,
        }
        for i in range(n_text)
    ]
    rows += [
        {
            "request_id": f"t{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"refund {i}"}],
            "response": "",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [{"function": {"name": "refund"}}],
            "tool_calls": ["refund"],
            "response_format": "json_object",
        }
        for i in range(n_tool)
    ]
    return rows


def _judge(comparison):
    """A judge that always agrees. Its verdict does not matter here — what
    matters is that one *ran*, which is what puts scores in the JUDGE tier."""
    from clickllm.prove.judge import Reply

    return Reply(winner="tie", reason="identical")


def _prover(*, matching: bool = True, judged: bool = False):
    """A candidate that answers exactly like the incumbent, or does not."""

    def prove(doc: dict):
        items = [
            EvalItem(
                item_id=str(i["item_id"]),
                cluster=str(i["cluster"]),
                prompt=str(i["prompt"]),
                baseline=str(i["baseline"]),
                candidate=str(i["baseline"]) if matching else "something else entirely",
            )
            for i in doc["items"]
        ]
        return suite(
            items,
            shares=doc["shares"],
            names=doc["names"],
            candidate="cand",
            incumbent="inc",
            issued="2026-08-11",
            judge=_judge if judged else None,
            judge_model="test-judge" if judged else "",
        )

    return prove


# --- the thing it must never do --------------------------------------------------


def test_a_perfect_run_does_not_advance_the_stage():
    """The whole reason this is safe to leave running on a schedule."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    st, run, decision = migrate.step(st, rows=_rows(), prove=_prover(), today=date(2026, 8, 11))
    assert st.stage_phase == "shadow" and st.stage_percent == 0, "the loop moved traffic"
    assert run.stage == "shadow"


def test_when_the_gate_permits_an_advance_it_is_printed_as_a_proposal():
    """`ADVANCE` is a recommendation. The text has to say it was not applied,
    or a reader skimming a cron log will assume it was."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    st, run, decision = migrate.step(
        st, rows=_rows(n_text=400), prove=_prover(), today=date(2026, 8, 11)
    )
    text = migrate.render(st, run, decision)
    if decision.action.value == "advance":
        assert "has not been applied" in text
        assert "invariant 8" in text
        assert st.stage_percent == 0, "printed a proposal and took it"
    else:
        # Not a failure — a small eval set legitimately holds. Assert the thing
        # that must be true either way rather than skipping the check.
        assert st.stage_percent == 0


def test_nothing_in_the_module_writes_a_nonzero_stage():
    """Asserted over the source, because the behavioural test above can only
    cover the paths it happens to reach.

    A future edit that sets `stage_percent` from a decision would be the one
    change here that matters, and it would look entirely reasonable in a diff.
    """
    src = (ROOT / "src" / "clickllm" / "migrate.py").read_text()
    # Assignments only. The lookaheads exclude the dataclass's own
    # `stage_percent: int = 0` annotation and the `== 0` comparisons in this
    # module's `demo()` — the first two versions of this regex reported each of
    # those as the violation, which is a check that cries wolf about itself.
    assignments = re.findall(r"stage_percent(?!\s*:)\s*=(?!=)\s*([^,\n)]+)", src)
    for value in assignments:
        v = value.strip()
        assert v == "0", f"migrate.py assigns stage_percent = {v}"
    assert "control" not in src.lower() or "control surface" in src, (
        "if this module ever imports the control surface, it can move traffic"
    )
    for forbidden in ("import requests", "urllib.request", "httpx"):
        assert forbidden not in src, f"the loop opened a network client: {forbidden}"


def test_a_judged_suite_is_refused_rather_than_mismodelled():
    """`Reading.judge_only` decides whether a cluster may advance a migration,
    and a `SuiteResult` does not carry per-item tiers to derive it from.

    Defaulting it to False would let exactly the cluster the flag exists to stop
    be the one that advances, so the loop refuses instead.
    """
    st = migrate.State(candidate="cand", started="2026-08-11")
    with pytest.raises(ValueError, match="judged suite"):
        migrate.step(st, rows=_rows(), prove=_prover(judged=True), today=date(2026, 8, 11))


# --- resuming --------------------------------------------------------------------


def test_state_round_trips_including_its_tuples(tmp_path):
    """JSON has no tuples. A load that returns lists where the dataclass says
    tuple is a bug waiting for the first comparison."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    st, _, _ = migrate.step(st, rows=_rows(), prove=_prover(), today=date(2026, 8, 11))
    p = tmp_path / "m.json"
    migrate.save_state(st, p)

    back = migrate.load_state(p, "cand")
    assert back.candidate == st.candidate
    assert len(back.runs) == len(st.runs)
    assert isinstance(back.runs[0].uncovered, tuple)
    assert back.stage().render() == st.stage().render()


def test_a_state_file_belongs_to_one_candidate(tmp_path):
    """Sharing one would interleave two histories and compare a rung reached by
    one candidate against evidence gathered by the other."""
    p = tmp_path / "m.json"
    migrate.save_state(migrate.State(candidate="cand", started="2026-08-11"), p)
    with pytest.raises(ValueError, match="not 'other'"):
        migrate.load_state(p, "other")
    assert migrate.load_state(p, "cand").candidate == "cand"
    # No candidate named: reading the state to *report* it is always allowed.
    assert migrate.load_state(p).candidate == "cand"


def test_a_missing_state_is_a_new_migration_not_an_error(tmp_path):
    st = migrate.load_state(tmp_path / "absent.json", "cand")
    assert st.candidate == "cand" and st.runs == []


def test_history_is_trimmed_so_the_file_does_not_become_a_log(tmp_path):
    st = migrate.State(candidate="cand", started="2026-08-11")
    st.runs = [
        migrate.Run(
            when="2026-08-11",
            captures=1,
            clusters=1,
            items=1,
            action="hold",
            reason="x",
            stage="shadow",
        )
        for _ in range(migrate.HISTORY + 10)
    ]
    p = tmp_path / "m.json"
    migrate.save_state(st, p)
    assert len(json.loads(p.read_text())["runs"]) == migrate.HISTORY


def test_an_unknown_field_in_the_state_is_refused(tmp_path):
    """A state file is a file; something else may have written it."""
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"candidate": "cand", "stage_percent": 100, "sneaky": True}))
    with pytest.raises(ValueError, match="unknown fields"):
        migrate.load_state(p)


# --- what it reports -------------------------------------------------------------


def test_captures_that_produce_no_items_are_an_error_not_an_empty_verdict():
    """A decision over nothing would render as a clean hold, which reads as
    "checked, and fine" rather than "there was nothing to check"."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    with pytest.raises(ValueError, match="no eval items"):
        migrate.step(st, rows=[], prove=_prover(), today=date(2026, 8, 11))


def test_a_narrowing_proof_is_pointed_out(tmp_path):
    """Fewer items than last time and a steadier-looking verdict are the same
    picture. The loop says so rather than letting a shrinking eval set read as
    an improving one."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    st, _, _ = migrate.step(st, rows=_rows(n_text=60), prove=_prover(), today=date(2026, 8, 11))
    st, run, decision = migrate.step(
        st, rows=_rows(n_text=10), prove=_prover(), today=date(2026, 8, 12)
    )
    text = migrate.render(st, run, decision)
    assert "fewer items than last run" in text


def test_uncovered_clusters_are_named_in_the_run_record():
    """Carried into the history because "the proof got greener" and "the proof
    got narrower" look identical in a single number."""
    st = migrate.State(candidate="cand", started="2026-08-11")
    st, run, decision = migrate.step(
        st,
        rows=_rows(n_text=40, n_tool=6),
        prove=_prover(),
        budget=2,
        min_per_cluster=1,
        today=date(2026, 8, 11),
    )
    if run.uncovered:
        assert "says nothing about that traffic" in migrate.render(st, run, decision)


def test_the_schedule_is_printed_and_not_installed():
    """Same rule as `watch`: a recurring job that reads your captured traffic is
    your decision."""
    fragment, where = migrate.install_schedule(interval_hours=6)
    assert "clickllm migrate --step" in fragment
    assert "*/6" in fragment
    assert "crontab" in where


def test_the_cli_exposes_the_loop_and_no_way_to_apply_its_proposal():
    src = (ROOT / "src" / "clickllm" / "cli.py").read_text()
    block = re.search(r'mg = sub\.add_parser\("migrate".*?mg\.set_defaults', src, re.S)
    assert block, "the migrate parser moved"
    for verb in ("--apply", "--advance", "--promote", "--percent", "--cutover"):
        assert verb not in block.group(0), f"migrate must not expose {verb}"
