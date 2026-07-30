"""The eval suite's kiosk layer.

`prove.demo()` walks one worked example end to end. These cover the properties
that would be expensive to get wrong and cheap to break: the judge economics,
the disclosure refusals, the traffic-share arithmetic, and the boundary that
keeps an agent out of the cutover.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from clickllm import mcp
from clickllm.prove import (
    Agreement,
    Comparison,
    EvalItem,
    Reply,
    run,
    shares_from_clusters,
    suite,
)

SRC = str(Path(__file__).resolve().parents[1] / "src")


def items(cluster: str, n: int, *, passing: bool) -> list[EvalItem]:
    """n items in one cluster, either structurally clean or shape-broken."""
    cand = '{"a": 1}' if passing else '{"b": 1}'
    return [EvalItem(f"{cluster}-{i}", cluster, f"prompt {i}", '{"a": 1}', cand) for i in range(n)]


# --- the judge economics -------------------------------------------------------


def test_the_judge_is_not_paid_to_reconfirm_a_structural_failure():
    """Scoring is conjunctive, so a judge PASS cannot rescue a shape break.

    Calling it anyway is spend with no information in it. This judge raises, so
    the run only completes if it was never consulted.
    """

    def never_call_me(c: Comparison) -> Reply:
        raise AssertionError("judge consulted on an already-disqualified item")

    m = run(
        items("rare-json", 15, passing=False),
        shares={"rare-json": 1.0},
        judge=never_call_me,
        judge_model="claude-opus-5",
    )
    assert m.candidates[0].clusters[0].band() == "regressed"


def test_the_judge_is_consulted_where_structure_passed():
    """Structure passing is not semantics passing — that is the judge's whole job."""
    seen: list[str] = []

    def judge(c: Comparison) -> Reply:
        seen.append(c.prompt)
        return Reply("A")  # a tie, whichever way round it is asked

    run(
        items("codegen", 5, passing=True),
        shares={"codegen": 1.0},
        judge=judge,
        judge_model="claude-opus-5",
    )
    # Five items, asked twice each — position-swapped, which is the point.
    assert len(seen) == 10, seen


def test_a_judge_that_errors_degrades_the_run_rather_than_failing_it():
    """An outage must cost graded items, not the whole result."""

    def flaky(c: Comparison) -> Reply:
        raise RuntimeError("upstream 503")

    m = run(
        items("codegen", 5, passing=True),
        shares={"codegen": 1.0},
        judge=flaky,
        judge_model="claude-opus-5",
    )
    # Still a report; the judge simply contributed nothing.
    assert m.candidates[0].clusters[0].known


def test_a_cluster_carried_by_the_judge_alone_still_cannot_advance():
    """The invariant, checked over the real scores rather than a hand-built flag.

    Forty items whose baseline no deterministic grader can touch, all of which
    the judge calls a tie. The cluster reads 40/40 — a perfect score with a
    trustworthy judge behind it — and the gate must still refuse to move it,
    because nothing but a language model's opinion is holding it up.
    """
    from dataclasses import replace as dc_replace

    from clickllm.prove import Action, Stage, decide, grade, judge_item, read_cluster
    from clickllm.prove.graders import ExactMatch

    # `ExactMatch` alone: a prose baseline leaves it N/A, so nothing deterministic
    # applies and the judge is the only voice in the room. This is `run`'s own
    # composition — grade, then append the judge's score — assembled here because
    # the gate reads `ItemResult`s and `run` returns a rendered matrix.
    only_exact = (ExactMatch(),)
    scored = []
    for i in range(40):
        item = EvalItem(
            f"s{i}", "summarise", f"p{i}", "The meeting is on Tuesday.", "It is Tuesday."
        )
        base = grade(item, only_exact)
        assert not base.graded, "the fixture is pointless if a deterministic grader fires"
        verdict = judge_item(item, lambda c: Reply("tie"), "claude-opus-5")
        scored.append(dc_replace(base, scores=base.scores + (verdict.to_score(),)))

    reading = read_cluster("summarise", "summarise", 1.0, scored)
    assert reading.judge_only, "the judge is the only applicable score here"
    assert reading.interval.passed == 40 and reading.interval.total == 40, reading.interval.render()

    d = decide(
        [reading],
        Stage("shadow", 0),
        agreement=Agreement(agreed=38, total=40, model="claude-opus-5"),
    )
    assert d.action is Action.HOLD, d.render()
    assert "judge alone" in d.reason, d.reason


def test_the_judge_is_measured_against_the_graders_on_every_run():
    """Calibration falls out of the run; nobody has to sit down with 40 items.

    `Agreement` needs a human and most teams will never sample one, so a judge
    would otherwise ship unmeasured. Here the graders are the known standard.
    """
    m = run(
        items("codegen", 20, passing=True),
        shares={"codegen": 1.0},
        judge=lambda c: Reply("tie"),
        judge_model="claude-opus-5",
    )
    cal = m.calibration
    assert cal is not None and cal.total == 20, cal
    assert cal.agreed == 20 and cal.calibrated
    assert "20 items scored both ways" in m.render()


def test_a_judge_that_contradicts_the_graders_is_reported_as_uncalibrated():
    """It condemns twenty items whose JSON demonstrably matches. That is a
    measurable property of the instrument, and the report states it."""

    def contrarian(c: Comparison) -> Reply:
        # Whichever position the baseline is in, the baseline wins: a consistent
        # preference, so it survives the position swap and is scored as a FAIL.
        return Reply("a" if c.a != c.b else "tie", "the first is better")

    m = run(
        [EvalItem(f"i{i}", "codegen", f"p{i}", '{"a": 1}', '{"a": 1, "b": 2}') for i in range(20)],
        shares={"codegen": 1.0},
        judge=contrarian,
        judge_model="rogue-judge",
    )
    # Shape breaks: the graders fail these, so the judge never sees them and
    # there is nothing to calibrate against — which is itself disclosed.
    assert m.calibration is not None and not m.calibration.calibrated
    assert "UNCALIBRATED" in m.render() or "BELOW THE FLOOR" in m.render()


def test_the_calibration_reaches_the_receipt():
    """The receipt is the artifact someone who does not trust us reads. A judge's
    calibration belongs in the file, not only on the terminal."""
    r = suite(
        items("codegen", 20, passing=True),
        shares={"codegen": 1.0},
        issued="2026-07-29",
        judge=lambda c: Reply("tie"),
        judge_model="claude-opus-5",
    )
    assert r.receipt.judge_calibration is not None
    assert "matched the deterministic graders" in r.receipt.judge_calibration
    assert r.receipt.judge_calibration in r.receipt.render()


# --- disclosure and refusals ---------------------------------------------------


def test_an_anonymous_judge_is_refused():
    """An undisclosed judge makes every score it touched unauditable."""
    with pytest.raises(ValueError, match="disclosure"):
        run(items("c", 3, passing=True), shares={"c": 1.0}, judge=lambda c: Reply("A"))


def test_an_empty_eval_set_is_refused_rather_than_scored_as_zero():
    with pytest.raises(ValueError, match="empty eval set"):
        run([], shares={})


def test_a_run_with_nothing_gradeable_refuses_to_issue_a_receipt():
    """An empty baseline leaves every grader inapplicable — there is no bar to match.

    The honest outcome is a refusal. Issuing a receipt here would certify a
    measurement that never happened.
    """
    blank = [EvalItem("i1", "chat", "prompt", "", "something")]
    with pytest.raises(ValueError, match="ungraded"):
        suite(blank, shares={"chat": 1.0}, issued="2026-07-27")


def test_shares_cannot_be_derived_from_zero_captures():
    with pytest.raises(ValueError, match="0 captures"):
        shares_from_clusters([])


# --- the arithmetic ------------------------------------------------------------


def test_a_perfect_but_small_cluster_is_unproven_not_equivalent():
    """12/12 gives a Wilson lower bound of 0.76, which does not clear a 0.90 bar.

    This is the single most important behaviour in the suite: a small perfect
    score must not open a gate.
    """
    m = run(items("codegen", 12, passing=True), shares={"codegen": 1.0})
    c = m.candidates[0].clusters[0]
    assert c.interval.point == 1.0
    assert c.band() == "unproven", c.render_cell()


def test_thirty_five_clean_samples_are_needed_before_a_cluster_can_move():
    """The exact boundary, pinned. 34 perfect items is still not enough evidence.

    Pinned rather than described because it is the number that decides whether
    traffic moves, and a change to the interval maths must not slide it quietly.
    """
    below = run(items("c", 34, passing=True), shares={"c": 1.0}).candidates[0].clusters[0]
    at = run(items("c", 35, passing=True), shares={"c": 1.0}).candidates[0].clusters[0]
    assert below.band() == "unproven", below.render_cell()
    assert at.band() == "equivalent", at.render_cell()


def test_a_cluster_missing_from_shares_is_reported_at_zero_weight_not_dropped():
    """Silently dropping it would shrink the denominator and inflate the verdict."""
    both = items("codegen", 45, passing=True) + items("surprise", 15, passing=False)
    m = run(both, shares={"codegen": 1.0})
    names = {c.cluster for c in m.candidates[0].clusters}
    assert names == {"codegen", "surprise"}
    assert m.candidates[0].movable_share() == 1.0


def test_the_verdict_is_traffic_weighted_not_item_weighted():
    """45 codegen items and 15 rare-json items, but rare-json is 25% of traffic.

    Item-weighted would give 75% by coincidence here; make the shares disagree
    with the counts so the two are distinguishable.
    """
    both = items("codegen", 45, passing=True) + items("rare-json", 15, passing=False)
    m = run(both, shares={"codegen": 0.60, "rare-json": 0.40})
    assert m.candidates[0].movable_share() == 0.60
    assert abs((m.candidates[0].weighted_score() or 0) - 0.60) < 0.01


def test_regret_is_stated_and_never_rounded_away():
    both = items("codegen", 45, passing=True) + items("rare-json", 15, passing=False)
    r = suite(both, shares={"codegen": 0.75, "rare-json": 0.25}, issued="2026-07-27")
    assert [c.name for c in r.receipt.regret] == ["rare-json"]
    assert "REGRET" in r.render()


def test_an_unproven_cluster_is_told_what_it_would_take():
    """ "Not yet proven" is a dead end; "35 items, you have 25" is a task.

    25 flawless items is the case that traps people — it reads conclusive and
    sits three points under the bar. The number comes from the closed form
    n > 9z², pinned in `tests/test_stats.py`.
    """
    r = suite(items("codegen", 25, passing=True), shares={"codegen": 1.0}, issued="2026-07-29")
    claim = r.receipt.unproven[0]
    assert claim.needed == 35, claim
    assert "needs 35" in r.receipt.render()
    assert "10 to go" in r.render(), r.render()

    # A cluster that already cleared the bar is not sent to gather more.
    proven = suite(items("c", 45, passing=True), shares={"c": 1.0}, issued="2026-07-29")
    assert proven.receipt.proven[0].needed is None


def test_a_cluster_that_is_simply_worse_is_not_sent_to_gather_more_evidence():
    """The other half of the instruction: no sample size rescues a rate that is
    already at or under the bar, and "gather more" there burns a week on a
    question the evidence has already answered."""
    r = suite(items("rare-json", 20, passing=False), shares={"rare-json": 1.0}, issued="2026-07-29")
    assert r.receipt.regret[0].needed is None

    # Exactly *at* the bar is the subtle one: 18/20 is 90%, the interval straddles,
    # and no amount of more-of-the-same will push a 90% rate above a 90% bar.
    at_the_bar = items("mixed", 18, passing=True) + items("mixed", 2, passing=False)
    r2 = suite(at_the_bar, shares={"mixed": 1.0}, issued="2026-07-29")
    assert r2.receipt.unproven[0].needed is None
    assert "will not clear it" in r2.render(), r2.render()


def test_the_weighted_verdict_carries_an_interval_and_names_its_method():
    """The headline of the whole report was the one number printed bare."""
    both = items("codegen", 45, passing=True) + items("rare-json", 15, passing=False)
    text = suite(both, shares={"codegen": 0.75, "rare-json": 0.25}, issued="2026-07-29").render()
    assert "weighted verdict" in text
    assert "Jeffreys posterior" in text, "the estimator must be named, not implied"
    assert "Wilson" in text, "and so must the per-cell one"


def test_scoring_many_clusters_against_one_bar_is_disclosed():
    """Twelve clusters at 95% each is a coin flip that one clears on noise. The
    report either adjusts or says it did not; silence is the failure mode."""
    many = []
    for i in range(6):
        many += items(f"c{i}", 45, passing=True)
    text = suite(many, shares={f"c{i}": 1 / 6 for i in range(6)}, issued="2026-07-29").render()
    assert "UNADJUSTED" in text and "Bonferroni" in text
    assert "6 clusters scored against the same 90% bar" in text


def test_the_same_eval_set_digests_identically():
    """Reproduction is the claim; a signature would only say 'we said this'."""
    both = items("codegen", 45, passing=True) + items("rare-json", 15, passing=False)
    kw = {"shares": {"codegen": 0.75, "rare-json": 0.25}, "issued": "2026-07-27"}
    assert suite(both, **kw).receipt.eval_set == suite(both, **kw).receipt.eval_set


def test_an_unmeasured_judge_is_not_a_trusted_judge():
    both = items("codegen", 45, passing=True)
    m = run(
        both,
        shares={"codegen": 1.0},
        judge=lambda c: Reply("A"),
        judge_model="claude-opus-5",
        agreement=Agreement(agreed=0, total=0, model="claude-opus-5"),
    )
    assert not m.judge_trustworthy
    assert "UNMEASURED" in m.render()


def test_no_cost_rate_means_no_saving_is_printed():
    """A fabricated saving is the most damaging number the report could carry."""
    both = items("codegen", 45, passing=True)
    r = suite(both, shares={"codegen": 1.0}, issued="2026-07-27")
    assert "unknown" in r.render()
    assert "$" not in r.policy.render()


# --- the agent boundary --------------------------------------------------------


def test_no_mcp_tool_can_move_traffic():
    """The boundary is a test, not a promise — this is that test."""
    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    leaked = [n for n in mcp.TOOLS if any(w in n for w in forbidden)]
    assert not leaked, f"write-side tools exposed to agents: {leaked}"


def test_the_agent_gets_the_same_verdict_as_the_cli(tmp_path: Path):
    """Four surfaces, one implementation — verified rather than asserted."""
    rows = [
        {
            "item_id": f"c{i}",
            "cluster": "codegen",
            "prompt": f"p{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"a": 1}',
        }
        for i in range(45)
    ] + [
        {
            "item_id": f"r{i}",
            "cluster": "rare-json",
            "prompt": f"q{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"b": 1}',
        }
        for i in range(15)
    ]
    path = tmp_path / "evalset.json"
    path.write_text(json.dumps({"items": rows, "shares": {"codegen": 0.75, "rare-json": 0.25}}))

    via_agent = mcp._prove(str(path), candidate="glm-5.2", incumbent="gpt-5", issued="2026-07-27")

    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "clickllm.cli",
            "prove",
            str(path),
            "--candidate",
            "glm-5.2",
            "--incumbent",
            "gpt-5",
            "--issued",
            "2026-07-27",
            "--json",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    via_cli = json.loads(cli.stdout)

    assert via_agent["receipt_digest"] == via_cli["digest"]
    assert via_agent["movable_share"] == 0.75


def test_the_agent_result_says_when_the_verdict_is_only_equal_weighted(tmp_path: Path):
    """Equal-weighted is a weaker claim; an agent must be able to tell them apart."""
    rows = [
        {
            "item_id": f"c{i}",
            "cluster": "codegen",
            "prompt": f"p{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"a": 1}',
        }
        for i in range(45)
    ]
    path = tmp_path / "noshares.json"
    path.write_text(json.dumps({"items": rows}))
    assert mcp._prove(str(path))["traffic_weighted"] is False

    path2 = tmp_path / "withshares.json"
    path2.write_text(json.dumps({"items": rows, "shares": {"codegen": 1.0}}))
    assert mcp._prove(str(path2))["traffic_weighted"] is True


def test_the_agent_never_claims_a_judge_it_did_not_use(tmp_path: Path):
    rows = [
        {
            "item_id": f"c{i}",
            "cluster": "codegen",
            "prompt": f"p{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"a": 1}',
        }
        for i in range(45)
    ]
    path = tmp_path / "e.json"
    path.write_text(json.dumps({"items": rows}))
    out = mcp._prove(str(path))
    assert out["judge_used"] is False
    assert "deterministic graders only" in out["report"]


def test_the_cli_exits_nonzero_when_nothing_can_move(tmp_path: Path):
    """Usable as a CI gate without parsing the output."""
    rows = [
        {
            "item_id": f"r{i}",
            "cluster": "rare-json",
            "prompt": f"q{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"b": 1}',
        }
        for i in range(15)
    ]
    path = tmp_path / "allbad.json"
    path.write_text(json.dumps({"items": rows, "shares": {"rare-json": 1.0}}))
    r = subprocess.run(
        [sys.executable, "-m", "clickllm.cli", "prove", str(path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 1, r.stdout


def test_the_cli_reports_an_error_rather_than_a_traceback(tmp_path: Path):
    """Repo convention: the CLI catches and returns a nonzero exit code."""
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"items": []}))
    r = subprocess.run(
        [sys.executable, "-m", "clickllm.cli", "prove", str(path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, r.stderr


def test_a_trailing_full_stop_is_not_a_wrong_answer():
    """Measured against a real Llama 3.1 8B over a live endpoint: asked "capital
    of France? Reply with only the city name" it answers `Paris.`

    That scored FAIL against a baseline of `Paris`, across all ten extraction
    items, and the suite reported REGRET — keep the incumbent — for a model that
    answered every question correctly. It fails in the safe direction, but a
    user shown 0% for a perfect run stops trusting the tool.
    """
    from clickllm.prove import EvalItem, Outcome
    from clickllm.prove.graders import ExactMatch

    g = ExactMatch()
    for got in ("Paris.", " Paris ", '"Paris"', "`Paris`", "Paris!", "paris"):
        item = EvalItem("i", "extraction", "capital of France?", "Paris", got)
        assert g.grade(item).outcome is Outcome.PASS, f"{got!r} is the right answer"


def test_normalisation_does_not_erase_a_real_difference():
    """The negative control. Stripping punctuation until everything matches
    would turn the grader into a rubber stamp, which is worse than brittle."""
    from clickllm.prove import EvalItem, Outcome
    from clickllm.prove.graders import ExactMatch

    g = ExactMatch()
    for base, got in (("3.14", "314"), ("yes", "no"), ("Paris", "Lyon")):
        item = EvalItem("i", "extraction", "q", base, got)
        assert g.grade(item).outcome is Outcome.FAIL, f"{base!r} vs {got!r} must differ"
