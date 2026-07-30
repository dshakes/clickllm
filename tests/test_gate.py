"""M9 — the quality gate.

`gate.demo()` covers the shape of each decision. These are the cases that need a
constructed world: exhaustive sweeps over the statistics, and the asymmetries
that only show up when you compare two decisions against each other.
"""

from __future__ import annotations

import pytest

from clickllm.prove.equivalence import DEFAULT_EQUIVALENCE_BAR, ClusterScore
from clickllm.prove.gate import (
    LADDER,
    PRECAUTION_MARGIN,
    PRECAUTION_MIN_SAMPLES,
    Action,
    Health,
    Reading,
    Stage,
    decide,
    read_cluster,
)
from clickllm.prove.graders import EvalItem, ItemResult, Outcome, Score, Tier, grade
from clickllm.prove.judge import Agreement, Calibration
from clickllm.prove.stats import wilson

BAR = DEFAULT_EQUIVALENCE_BAR


def reading(name="c", passed=40, total=40, *, judge=False, share=1.0) -> Reading:
    return Reading(
        score=ClusterScore(name, name, share, wilson(passed, total), 0),
        judge_only=judge,
    )


CALIBRATED = Calibration(agreed=19, total=20, model="claude-opus-5")

SHADOW = Stage("shadow", 0)
CANARY = Stage("canary", 25)
HEALTHY = Health(requests=100, errors=0)


# --- the property the two rollback rules exist to guarantee ---------------------


def test_a_materially_bad_cluster_with_enough_samples_always_acts():
    """No gap between 'proven' and 'precautionary'.

    This is the load-bearing property of having two rules instead of one. If any
    (passed, total) fell between them, a regression could sit on live traffic
    indefinitely — bad enough to worry about, never bad enough to trip anything.
    """
    unacted = []
    for total in range(PRECAUTION_MIN_SAMPLES, 200):
        for passed in range(total + 1):
            iv = wilson(passed, total)
            if iv.point >= BAR - PRECAUTION_MARGIN:
                continue  # not materially bad; not this test's business
            d = decide([reading(passed=passed, total=total)], CANARY, health=HEALTHY)
            if d.action is not Action.ROLL_BACK:
                unacted.append((passed, total, iv.render()))
    assert not unacted, f"materially bad but no rollback: {unacted[:5]}"


def test_proven_and_precautionary_are_both_reachable_and_labelled_apart():
    # If one rule shadowed the other entirely, the distinction would be a comment
    # rather than a behaviour.
    proven = decide([reading(passed=2, total=40)], CANARY, health=HEALTHY)
    assert proven.action is Action.ROLL_BACK and not proven.precautionary

    precautionary = decide([reading(passed=9, total=12)], CANARY, health=HEALTHY)
    assert precautionary.action is Action.ROLL_BACK and precautionary.precautionary
    assert "not yet conclusive" in precautionary.reason

    assert "precautionary" in precautionary.render().lower()
    assert "precautionary" not in proven.render().lower()


def test_a_healthy_candidate_at_the_bar_is_never_rolled_back():
    # The other direction of the sweep: passing clusters must not trip anything.
    for total in range(PRECAUTION_MIN_SAMPLES, 120):
        d = decide([reading(passed=total, total=total)], CANARY, health=HEALTHY)
        assert d.action is not Action.ROLL_BACK, f"{total}: {d.render()}"


# --- the asymmetry between advancing and rolling back --------------------------


def test_only_a_rollback_may_be_applied_without_a_human():
    assert decide([reading(passed=2, total=40)], CANARY, health=HEALTHY).automatic
    assert not decide([reading()], SHADOW).automatic
    assert not decide([reading(passed=3, total=3)], SHADOW).automatic


def test_the_same_evidence_means_different_things_in_shadow_and_canary():
    # A losing cluster in shadow is a pinning decision; live it is a rollback.
    bad = [reading(passed=2, total=40)]
    assert decide(bad, SHADOW).action is Action.HOLD
    assert decide(bad, CANARY, health=HEALTHY).action is Action.ROLL_BACK


def test_health_is_checked_before_quality_not_after():
    # A broken candidate with perfect grades must still roll back, and the reason
    # must name the breakage rather than the (fine) quality.
    d = decide(
        [reading(passed=40, total=40)],
        CANARY,
        health=Health(requests=100, errors=20, baseline_requests=100, baseline_errors=0),
    )
    assert d.action is Action.ROLL_BACK
    assert "failing" in d.reason and "20.0%" in d.reason


def test_an_incumbent_that_already_errors_does_not_condemn_a_matching_candidate():
    # Comparing against an absolute rate would make the candidate look broken for
    # merely matching what was already being tolerated.
    d = decide(
        [reading()],
        CANARY,
        health=Health(
            requests=100,
            errors=5,
            baseline_requests=100,
            baseline_errors=5,
            baseline_p95_ms=200,
            candidate_p95_ms=210,
        ),
    )
    assert d.action is not Action.ROLL_BACK, d.render()


# --- the judge rules -----------------------------------------------------------


def test_judge_only_evidence_can_block_but_never_advance():
    # Invariant 8, in both directions.
    blocked = decide([reading(judge=True)], SHADOW)
    assert blocked.action is Action.HOLD and "judge alone" in blocked.reason

    # ...but the same judge may still stop a migration, once it has been measured
    # against the deterministic graders. See the calibration test below for what
    # happens when it has not.
    stopped = decide(
        [reading(passed=2, total=40, judge=True)],
        CANARY,
        health=HEALTHY,
        calibration=CALIBRATED,
    )
    assert stopped.action is Action.ROLL_BACK


def test_an_uncalibrated_judge_cannot_veto_on_its_own():
    """The instrument has to be measured before it gets to stop anything.

    A judge that fails work the graders demonstrably passed would roll back a
    healthy migration on its own opinion, repeatedly, until the operator learned
    to ignore the alarm. The evidence is not discarded — it produces a HOLD that
    names what is missing, which is the difference between a gate and a shrug.
    """
    for calibration in (
        None,  # never measured
        Calibration(agreed=9, total=9, model="j"),  # perfect, but too few items
        Calibration(agreed=6, total=12, model="j"),  # measured, and bad at it
    ):
        d = decide(
            [reading(passed=2, total=40, judge=True)],
            CANARY,
            health=HEALTHY,
            calibration=calibration,
        )
        assert d.action is Action.HOLD, f"{calibration}: {d.render()}"
        assert "unmeasured instrument" in d.reason, d.reason
        assert "calibrate the judge" in d.reason, d.reason


def test_one_deterministic_trigger_is_enough_to_roll_back():
    """A judge alongside a grader is corroborating, not deciding — so the
    calibration floor must not hold up a rollback the graders asked for."""
    d = decide(
        [reading("summarise", passed=2, total=40, judge=True), reading("classify", 2, 40)],
        CANARY,
        health=HEALTHY,
    )
    assert d.action is Action.ROLL_BACK and d.automatic, d.render()


def test_a_calibrated_judge_still_cannot_advance_anything():
    """The invariant that survives the whole calibration story."""
    d = decide([reading(passed=40, total=40, judge=True)], SHADOW, calibration=CALIBRATED)
    assert d.action is Action.HOLD and "judge alone" in d.reason, d.render()


def test_an_untrustworthy_judge_blocks_advancing():
    for agreement in (
        Agreement(agreed=5, total=10, model="j"),  # rate too low
        Agreement(agreed=10, total=10, model="j"),  # too few samples
    ):
        d = decide([reading()], SHADOW, agreement=agreement)
        assert d.action is Action.HOLD, f"{agreement.render()}: {d.render()}"
        assert "not trustworthy" in d.reason


def test_a_trustworthy_judge_does_not_block():
    d = decide([reading()], SHADOW, agreement=Agreement(agreed=19, total=20, model="j"))
    assert d.action is Action.ADVANCE, d.render()


# --- the ladder ----------------------------------------------------------------


def test_the_ladder_never_jumps_straight_from_shadow_to_a_cut():
    d = decide([reading()], SHADOW)
    assert d.to == Stage("canary", 5), d.to
    assert d.to.percent < 100


def test_advancing_walks_every_rung_and_then_stops():
    seen = [SHADOW]
    stage = SHADOW
    for _ in range(len(LADDER) + 2):
        d = decide([reading()], stage, health=HEALTHY)
        if d.action is not Action.ADVANCE:
            break
        stage = d.to
        seen.append(stage)
    assert seen == list(LADDER), seen
    final = decide([reading()], LADDER[-1], health=HEALTHY)
    assert final.action is Action.HOLD and "top of the ladder" in final.reason


def test_an_unmodelled_stage_holds_rather_than_guessing():
    d = decide([reading()], Stage("canary", 37), health=HEALTHY)
    assert d.action is Action.HOLD


# --- evidence sufficiency ------------------------------------------------------


def test_an_underpowered_cluster_says_how_many_more_it_needs():
    d = decide([reading(passed=3, total=3)], SHADOW)
    assert d.action is Action.HOLD
    assert "5 more needed" in d.reason and "3 graded items" in d.reason


def test_a_cluster_with_no_graded_items_is_named_not_ignored():
    d = decide([reading("extract"), reading("weird", 0, 0)], SHADOW)
    assert d.action is Action.HOLD
    assert "no graded items" in d.reason and "weird" in d.reason


def test_live_traffic_without_a_health_signal_is_not_assumed_fine():
    assert decide([reading()], CANARY).action is Action.HOLD
    thin = decide([reading()], CANARY, health=Health(requests=3, errors=0))
    assert thin.action is Action.HOLD and "only 3 requests" in thin.reason


def test_no_evidence_at_all_holds():
    d = decide([], SHADOW)
    assert d.action is Action.HOLD and "no evidence" in d.reason


# --- pinning -------------------------------------------------------------------


def test_a_pinned_cluster_cannot_block_or_trip_what_it_does_not_serve():
    clusters = [reading("good", 40, 40, share=0.7), reading("bad", 2, 40, share=0.3)]
    assert decide(clusters, CANARY, health=HEALTHY).action is Action.ROLL_BACK
    pinned = decide(clusters, CANARY, health=HEALTHY, pinned=frozenset({"bad"}))
    assert pinned.action is Action.ADVANCE, pinned.render()
    assert "bad" not in pinned.clusters


def test_the_advance_reason_reports_movable_share_not_total_share():
    clusters = [reading("good", 40, 40, share=0.7), reading("bad", 2, 40, share=0.3)]
    d = decide(clusters, CANARY, health=HEALTHY, pinned=frozenset({"bad"}))
    assert "70% of traffic" in d.reason, d.reason


# --- tier provenance -----------------------------------------------------------


def _item(**kw) -> EvalItem:
    base = dict(item_id="i", cluster="c", prompt="p", baseline="same", candidate="same")
    return EvalItem(**{**base, **kw})


def test_read_cluster_detects_judge_only_evidence_from_real_scores():
    judged = ItemResult(
        item_id="i",
        cluster="c",
        scores=(Score("pairwise-judge", Tier.JUDGE, Outcome.PASS),),
    )
    assert read_cluster("c", "c", 1.0, [judged]).judge_only

    deterministic = grade(_item())
    assert not read_cluster("c", "c", 1.0, [deterministic]).judge_only

    # Mixed evidence is not judge-only: one deterministic check is enough.
    assert not read_cluster("c", "c", 1.0, [judged, deterministic]).judge_only


def test_read_cluster_reuses_the_equivalence_interval():
    from clickllm.prove.equivalence import score_cluster

    results = [grade(_item(item_id=f"i{i}")) for i in range(10)]
    assert (
        read_cluster("c", "n", 0.5, results).interval
        == score_cluster("c", "n", 0.5, results).interval
    )


# --- reasons are usable at 3am -------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(readings=[reading(passed=2, total=40)], stage=CANARY, health=HEALTHY),
        dict(readings=[reading(passed=9, total=12)], stage=CANARY, health=HEALTHY),
        dict(readings=[reading(passed=3, total=3)], stage=SHADOW),
        dict(readings=[reading(judge=True)], stage=SHADOW),
        dict(readings=[reading()], stage=SHADOW),
        dict(readings=[], stage=SHADOW),
    ],
)
def test_every_decision_carries_a_specific_reason(kwargs):
    d = decide(**kwargs)
    assert d.reason and len(d.reason) > 20, d.reason
    assert d.render()
    # No decision may report a bare verdict with nothing behind it.
    assert d.reason.strip() != d.action.value
