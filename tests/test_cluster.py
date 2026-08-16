"""Budget allocation across clusters.

`cluster.demo()` covers the regime where the budget comfortably seats every
cluster. These cover the one it cannot reach: more clusters than the budget can
pay for, where the per-cluster floor collapses. That regime used to return an
eval set with silent holes — and at the default budget, an eval set that was
entirely empty while every signal in the report said the clusters were covered.
"""

from __future__ import annotations

import pytest

from onpar.distill.cluster import MIN_CLUSTER_SIZE, cluster, sample
from onpar.distill.shape import Capture


def cap(i: int, system: str, resp: str = "x") -> Capture:
    return Capture(
        request_id=f"req-{i}",
        model="gpt-5",
        messages=({"role": "system", "content": system}, {"role": "user", "content": f"q{i}"}),
        response=resp,
        prompt_tokens=500,
    )


def make(shapes: int, per: int, tag: str = "task") -> list:
    """`shapes` distinct clusters of `per` captures each."""
    caps = [
        cap(hash((tag, k, i)) % 10**9, f"{tag}-{k}", "y" * (i % 11 + 1))
        for k in range(shapes)
        for i in range(per)
    ]
    cs = cluster(caps)
    assert len(cs) == shapes, f"fixture built {len(cs)} clusters, wanted {shapes}"
    return cs


def test_the_default_budget_does_not_return_an_empty_eval_set():
    # The headline failure. 250 distinct task shapes is plausible for real
    # enterprise traffic, and at the DEFAULT budget every single cluster was
    # sampled to zero: floor collapsed to 0, then each cluster's 1/250th share
    # truncated its allocation to 0 as well. total_sampled was 0 — an empty
    # eval suite — with nothing in the report saying so.
    rep = sample(make(250, 20))
    assert rep.total_sampled > 0, "the default budget produced no eval set at all"
    assert rep.total_sampled == 200, rep.total_sampled


def test_truncation_does_not_throw_the_budget_away():
    # Each cluster's exact share is fractional, and `int()` on each one
    # independently discards all of it. Largest-remainder spends it.
    for shapes, budget in ((250, 200), (40, 137), (7, 23), (99, 100)):
        rep = sample(make(shapes, 20), budget=budget)
        assert rep.total_sampled == budget, (
            f"{shapes} clusters, budget {budget}: spent {rep.total_sampled}"
        )


def test_a_cluster_sampled_to_nothing_is_named_rather_than_left_looking_covered():
    # A cluster with no samples still has a key in `sampled`, so a caller
    # checking `key in rep.sampled` sees it as covered. `uncovered` is the
    # field that contradicts that, and it must agree exactly with reality.
    rep = sample(make(12, 20), budget=8, min_per_cluster=3)
    empty = {k for k, v in rep.sampled.items() if not v}
    assert empty, "8 across 12 clusters must leave some uncovered"
    assert set(rep.uncovered) == empty
    # ...and the keys are still present, which is exactly why the field exists.
    assert all(k in rep.sampled for k in rep.uncovered)


def test_a_reduced_floor_is_reported_rather_than_silently_applied():
    # The docstring promises every cluster `min_per_cluster`. When the budget
    # cannot pay for that, the promise is arithmetically impossible — but
    # breaking it silently is not the same as breaking it.
    rep = sample(make(12, 20), budget=8, min_per_cluster=3)
    assert rep.floor_applied == 0

    affordable = sample(make(12, 20), budget=200, min_per_cluster=3)
    assert affordable.floor_applied == 3
    assert affordable.uncovered == ()


def test_every_cluster_is_covered_when_the_budget_can_afford_it():
    # The guarantee that does still hold: budget >= clusters means nobody
    # gets zero, whatever the share distribution looks like.
    for shapes, budget in ((250, 1000), (250, 250), (12, 12), (40, 200)):
        rep = sample(make(shapes, 20), budget=budget)
        assert rep.uncovered == (), f"{shapes} clusters, budget {budget}"
        assert all(rep.sampled.values())


def _bulk_plus_one_critical():
    """Nine 1000-capture clusters beside one 6-capture cluster.

    The module's stated reason for existing: "a 12-request cluster may be the
    one that blocks the migration". The small one holds 0.07% of traffic, so a
    pure share split rounds it away. It is also >= MIN_CLUSTER_SIZE, so nothing
    in the report warns about it on size grounds.
    """
    caps = [
        cap(1000 * k + i, f"bulk-{k}", "z" * (i % 9 + 1)) for k in range(9) for i in range(1000)
    ]
    caps += [cap(99_000 + i, "critical", "w" * (i + 1)) for i in range(6)]
    cs = cluster(caps)
    crit = next(c for c in cs if c.size == 6)
    assert crit.size >= MIN_CLUSTER_SIZE
    return cs, crit


def test_a_tiny_share_does_not_round_a_critical_cluster_away():
    # A regression guard, not a demonstration of the fix: at this budget the
    # floor was affordable before the change too. It pins the property the
    # change had to preserve — share-weighting must not override the floor.
    cs, crit = _bulk_plus_one_critical()
    rep = sample(cs, budget=200, min_per_cluster=3)
    assert len(rep.sampled[crit.key]) >= 3, "the floor must beat a 0.07% share"
    assert crit.key not in rep.uncovered
    assert crit.key not in rep.small_clusters


def test_when_the_budget_cannot_seat_every_cluster_the_loss_is_named():
    # The honest limit of the fix, written down so nobody reads more into it
    # than it does. With fewer budget units than clusters, some clusters
    # genuinely cannot be sampled, and share-weighting drops the smallest —
    # the critical one included. What the fix guarantees is not that it
    # survives, but that its absence lands in `uncovered` instead of looking
    # like a covered cluster that happened to return an empty list.
    cs, crit = _bulk_plus_one_critical()
    rep = sample(cs, budget=8, min_per_cluster=3)
    assert len(cs) == 10 and rep.total_sampled == 8
    assert not rep.sampled[crit.key], "a 0.07% share cannot win a seat here"
    assert crit.key in rep.uncovered, "and that must be reported, not silent"
    assert rep.floor_applied == 0


def test_allocation_never_exceeds_the_budget():
    for shapes, budget, mpc in ((250, 200, 3), (3, 30, 3), (12, 8, 3), (50, 999, 7)):
        rep = sample(make(shapes, 20), budget=budget, min_per_cluster=mpc)
        assert rep.total_sampled <= budget


def test_allocation_is_deterministic_across_runs():
    a = sample(make(50, 40), budget=200)
    b = sample(make(50, 40), budget=200)
    assert {k: [c.request_id for c in v] for k, v in a.sampled.items()} == {
        k: [c.request_id for c in v] for k, v in b.sampled.items()
    }


def test_a_cluster_is_never_sampled_beyond_its_own_size():
    # A tiny cluster in a rich budget must not ask for more captures than it
    # has; sample_cluster caps it, and total_sampled must reflect the cap.
    rep = sample(make(4, 2), budget=500)
    assert rep.total_sampled == 8
    assert all(len(v) == 2 for v in rep.sampled.values())


def test_units_freed_by_a_cluster_smaller_than_its_allocation_are_re_placed():
    # Raised by the Codex audit against the first version of this fix, and it
    # is the same defect one layer down: capping a cluster at its own size and
    # walking away loses the surplus silently. Sizes [1, 1000] with budget 100
    # spent 98, `uncovered` empty and nothing saying where the 2 went.
    caps = [cap(0, "tiny", "a")]
    caps += [cap(100 + i, "bulk", "b" * (i % 9 + 1)) for i in range(1000)]
    cs = cluster(caps)
    assert sorted(c.size for c in cs) == [1, 1000]

    rep = sample(cs, budget=100, min_per_cluster=3)
    assert rep.total_sampled == 100, "the surplus from the 1-capture cluster was lost"
    tiny = next(c for c in cs if c.size == 1)
    assert len(rep.sampled[tiny.key]) == 1, "and it must not be over-drawn to get there"


def test_under_spend_is_only_ever_the_corpus_being_smaller_than_the_budget():
    # After re-placing, the single honest reason to spend less than the budget
    # is that there are not that many captures — which the report already
    # states as total_captures, so it needs no extra field.
    for shapes, per, budget in ((4, 2, 500), (10, 3, 100), (2, 1, 50)):
        rep = sample(make(shapes, per), budget=budget)
        assert rep.total_sampled == min(budget, rep.total_captures)


@pytest.mark.parametrize("budget", [0, 1, 2])
def test_a_budget_at_or_near_zero_is_reported_not_hidden(budget):
    rep = sample(make(5, 20), budget=budget)
    assert rep.total_sampled == budget
    assert len(rep.uncovered) == 5 - budget


def test_a_negative_budget_still_refuses_with_the_offending_value():
    with pytest.raises(ValueError, match="-1"):
        sample(make(3, 10), budget=-1)


def test_a_negative_floor_refuses_rather_than_overspending_the_budget():
    # Raised by the Codex audit. A negative `min_per_cluster` was unvalidated,
    # and the surplus pass turned it into an over-spend: a negative floor makes
    # `want` negative for a small cluster, so `budget - sum(want)` reads a
    # deficit that was never spent and hands the phantom units out. With sizes
    # [1, 1000], `budget=10, min_per_cluster=-1` returned 11 samples.
    with pytest.raises(ValueError, match="-1"):
        sample(make(3, 10), min_per_cluster=-1)


def test_the_budget_holds_for_every_floor_a_caller_can_pass():
    # The property the guard exists to protect, checked across the range
    # rather than at the one value that broke.
    cs = make(6, 40)
    for budget in (0, 1, 5, 17, 200):
        for mpc in range(0, 8):
            rep = sample(cs, budget=budget, min_per_cluster=mpc)
            assert rep.total_sampled <= budget, (budget, mpc, rep.total_sampled)


def test_floor_applied_is_the_affordable_floor_not_the_smallest_sample():
    # Also from the audit. The field said "the minimum actually achieved",
    # which a cluster smaller than the floor contradicts — sizes [1, 1000] at
    # budget 100 report floor_applied 3 while one cluster yields its single
    # capture. That is the cluster's size, not a budget shortfall, so the
    # number is right and the claim about it was wrong.
    caps = [cap(0, "tiny", "a")]
    caps += [cap(100 + i, "bulk", "b" * (i % 9 + 1)) for i in range(1000)]
    cs = cluster(caps)
    rep = sample(cs, budget=100, min_per_cluster=3)
    assert rep.floor_applied == 3, "the budget could afford a floor of 3"
    assert min(len(v) for v in rep.sampled.values()) == 1, "and a 1-capture cluster gives 1"
    assert rep.uncovered == (), "which is not the same as being uncovered"


def test_no_clusters_is_an_empty_report_not_a_crash():
    rep = sample([], budget=100)
    assert rep.total_sampled == 0
    assert rep.uncovered == ()
    assert rep.floor_applied == 0
