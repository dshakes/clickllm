"""What `run()` and `suite()` do with an eval set that does not line up.

`prove.demo()` covers the happy path and the refusals. These cover the seams
between the eval set and the traffic shares — where a cluster exists on one
side and not the other, and the report has to say so rather than quietly
redefining what it measured.
"""

from __future__ import annotations

import pytest

from clickllm.prove import run, suite
from clickllm.prove.graders import EvalItem
from clickllm.prove.judge import Comparison, Reply


def item(cluster: str, ok: bool = True, item_id: str | None = None) -> EvalItem:
    return EvalItem(
        item_id or f"{cluster}-{ok}",
        cluster,
        f"prompt for {cluster}",
        '{"a": 1}',
        '{"a": 1}' if ok else "not json",
    )


# --- coverage gaps ---------------------------------------------------------------


def test_a_cluster_with_traffic_share_but_no_eval_items_is_reported_as_unknown():
    # The gap that used to vanish. `run()` built its clusters strictly from
    # `items`, so a share the eval set never covered simply was not in the
    # report — and `movable_share` was then a fraction of a denominator the
    # report had silently redefined to exclude it.
    m = run([item("covered")], shares={"covered": 0.7, "never-sampled": 0.3})
    by_name = {c.cluster: c for c in m.candidates[0].clusters}
    assert set(by_name) == {"covered", "never-sampled"}
    gap = by_name["never-sampled"]
    assert not gap.known
    assert gap.band() == "unknown"
    assert gap.share == pytest.approx(0.3), "the share it holds must survive"


def test_an_uncovered_cluster_cannot_be_proven_or_move_traffic():
    m = run([item("covered")], shares={"covered": 0.7, "never-sampled": 0.3})
    gap = next(c for c in m.candidates[0].clusters if c.cluster == "never-sampled")
    assert gap not in m.best().regret(m.bar)
    assert gap.band() not in ("equivalent", "regressed")


def test_the_mirror_case_still_works_items_with_no_share():
    # The behaviour that was already correct, pinned so the new branch does
    # not disturb it.
    m = run([item("orphan")], shares={})
    only = m.candidates[0].clusters[0]
    assert only.cluster == "orphan"
    assert only.share == 0.0


def test_clusters_are_ordered_the_same_however_they_got_there():
    # Uncovered clusters are appended after the grouping loop, so the combined
    # list has to be re-sorted or the report's row order depends on which side
    # a cluster came from.
    m = run([item("m-mid")], shares={"z-last": 0.3, "a-first": 0.3, "m-mid": 0.4})
    order = [c.cluster for c in m.candidates[0].clusters]
    assert order == sorted(order), order


# --- the judge is not consulted on a disqualified item ----------------------------


def test_the_judge_is_never_consulted_on_a_structurally_failed_item():
    # Counted rather than raised. A judge that raises proves nothing here:
    # `judge_item` catches every exception from the injected callable by
    # design, converts it to UNCERTAIN, and UNCERTAIN scores NOT_APPLICABLE,
    # which the conjunctive `passed` ignores. The band reads "regressed"
    # either way, so an exception-based check passes with the guard deleted.
    consulted: list[Comparison] = []

    def spy(c: Comparison) -> Reply:
        consulted.append(c)
        return Reply("A")

    run(
        [item("bad", ok=False)],
        shares={"bad": 1.0},
        judge=spy,
        judge_model="claude-opus-5",
    )
    assert consulted == []


def test_the_judge_is_consulted_where_it_can_change_the_answer():
    # The other half, and the reason the guard is a guard rather than a
    # blanket skip: an item with no applicable grader is exactly the blind
    # spot the judge exists to cover.
    consulted: list[Comparison] = []

    def spy(c: Comparison) -> Reply:
        consulted.append(c)
        return Reply("A")

    run(
        [EvalItem("u1", "ungradeable", "p", "", "")],
        shares={"ungradeable": 1.0},
        judge=spy,
        judge_model="claude-opus-5",
    )
    # Two calls for one item, not a bug: `judge_item` asks twice with the
    # sides swapped so position bias shows up as a disagreement.
    assert len(consulted) == 2


# --- suite() says which of the two problems it hit --------------------------------


def test_a_share_map_that_names_different_clusters_says_so():
    # Both remedies in the old message were wrong for this cause: the
    # candidate answered perfectly and a judge would change nothing. The
    # actual fault is the caller's `shares` dict.
    with pytest.raises(ValueError) as e:
        suite([item("clusterA")], shares={}, issued="2026-08-04")
    msg = str(e.value)
    assert "zero traffic share" in msg
    assert "clusterA" in msg, "the scored cluster must be named"
    assert "ungraded" not in msg, "this is not the ungraded case"


def test_nothing_gradeable_still_reports_the_ungraded_cause():
    # The case the original message was written for must keep it.
    with pytest.raises(ValueError) as e:
        suite([EvalItem("u1", "c", "p", "", "")], shares={"c": 1.0}, issued="2026-08-04")
    assert "every item was ungraded" in str(e.value)


def test_the_share_map_is_named_in_the_mismatch_message():
    with pytest.raises(ValueError) as e:
        suite([item("actual")], shares={"expected": 1.0}, issued="2026-08-04")
    msg = str(e.value)
    assert "actual" in msg and "expected" in msg, msg


# --- an uncovered share reaches the policy and the headline ----------------------


def covered_plus_gap():
    # Distinct prompts: identical ones are duplicate-merged into a single
    # observation, which is correct and would collapse this to 1/1.
    items = [EvalItem(f"i{i}", "covered", f"prompt-{i}", '{"a": 1}', '{"a": 1}') for i in range(45)]
    return run(items, shares={"covered": 0.7, "never-sampled": 0.3})


def test_an_unmeasured_cluster_reaches_the_hybrid_policy():
    # Surfacing the gap in the report but not in the policy would have been
    # worse than not surfacing it: `SuiteResult.policy` is what the MCP tools
    # and the API hand to an agent, and traffic nobody measured cannot move
    # for exactly the same reason traffic straddling the bar cannot.
    m = covered_plus_gap()
    policy = m.hybrid_for(m.best())
    assert "never-sampled" in policy.unproven_clusters
    assert any("never-sampled" in n for n in policy.needs)


def test_an_unmeasured_cluster_is_not_counted_as_movable():
    m = covered_plus_gap()
    assert m.best().movable_share(m.bar) == pytest.approx(0.7)


def test_the_weighted_headline_says_what_share_it_covers():
    # The renormalisation is the only arithmetic available — an unmeasured
    # cluster has no rate to contribute — but doing it silently printed a flat
    # "100% weighted" over traffic that was 30% unlooked-at.
    text = covered_plus_gap().render()
    assert "weighted of 70%" in text, text
    assert "never measured and is not in this number" in text, text


def test_full_coverage_prints_no_coverage_caveat():
    # A caveat on every report is a caveat nobody reads.
    m = run(
        [EvalItem(f"i{i}", "covered", f"p-{i}", '{"a": 1}', '{"a": 1}') for i in range(45)],
        shares={"covered": 1.0},
    )
    text = m.render()
    assert "never measured" not in text
    assert "weighted of" not in text


def test_a_capture_count_is_not_invented_for_traffic_the_eval_set_never_saw():
    # `len(items)` stands in for the capture count, and it stops standing for
    # anything once `shares` names a cluster with no items: every item comes
    # from the covered clusters, so "drawn from 45 captured requests" would
    # attribute the whole traffic distribution to a sample that never saw part
    # of it. The receipt is the proof artifact; fabricated provenance in it is
    # worse than none.
    items = [EvalItem(f"i{i}", "covered", f"p-{i}", '{"a": 1}', '{"a": 1}') for i in range(45)]
    r = suite(items, shares={"covered": 0.7, "gap": 0.3}, issued="2026-08-07")
    assert r.receipt.traffic_captures == 0
    assert "captured requests" not in r.receipt.render()
    # ...and the gap itself is still disclosed, which is the point.
    assert "gap" in [c.cluster for c in r.receipt.unproven]


def test_a_capture_count_is_still_defaulted_when_the_shares_are_covered():
    items = [EvalItem(f"i{i}", "covered", f"p-{i}", '{"a": 1}', '{"a": 1}') for i in range(45)]
    r = suite(items, shares={"covered": 1.0}, issued="2026-08-07")
    assert r.receipt.traffic_captures == 45
    assert "Drawn from 45 captured requests" in r.receipt.render()


def test_an_explicit_capture_count_is_honoured_over_the_gap_rule():
    # The caller knows the real number; this must not second-guess it.
    items = [EvalItem(f"i{i}", "covered", f"p-{i}", '{"a": 1}', '{"a": 1}') for i in range(45)]
    r = suite(
        items, shares={"covered": 0.7, "gap": 0.3}, issued="2026-08-07", traffic_captures=12_400
    )
    assert r.receipt.traffic_captures == 12_400
    assert "Drawn from 12400 captured requests" in r.receipt.render()
