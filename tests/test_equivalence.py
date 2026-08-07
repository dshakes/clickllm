"""The report's disclosures must describe the model it recommends.

`equivalence.demo()` renders a single-candidate matrix, where "the first
candidate" and "the recommended candidate" are the same object and any mix-up
between them is invisible. These use two candidates that differ in what was
excluded from their scores.
"""

from __future__ import annotations

import pytest

from clickllm.prove.equivalence import (
    CandidateReport,
    ClusterScore,
    Matrix,
    score_cluster,
)
from clickllm.prove.graders import EvalItem, grade
from clickllm.prove.stats import wilson


def results(cluster: str, n_pass: int, n_fail: int, n_ungraded: int = 0):
    out = []
    for i in range(n_pass):
        out.append(grade(EvalItem(f"{cluster}-p{i}", cluster, "p", '{"a":1}', '{"a":1}')))
    for i in range(n_fail):
        out.append(grade(EvalItem(f"{cluster}-f{i}", cluster, "p", '{"a":1}', "not json")))
    for i in range(n_ungraded):
        out.append(grade(EvalItem(f"{cluster}-u{i}", cluster, "p", "", "")))
    return out


def two_candidates(*, ungraded_on_best: int = 0, ungraded_on_first: int = 0):
    """A weak first candidate and a strong second, so `best` is never index 0."""
    weak = CandidateReport(
        "model-A", (score_cluster("c1", "x", 1.0, results("c1", 50, 50, ungraded_on_first)),)
    )
    strong = CandidateReport(
        "model-B", (score_cluster("c1", "x", 1.0, results("c1", 95, 0, ungraded_on_best)),)
    )
    return Matrix([weak, strong], incumbent_cost=1000.0)


def test_the_recommended_model_is_not_the_first_candidate_in_these_fixtures():
    # The premise every test below rests on. If `best` ever became candidates[0]
    # these would all pass for the wrong reason.
    m = two_candidates()
    assert m.candidates[0].model == "model-A"
    assert m.best().model == "model-B"


def test_ungraded_items_are_disclosed_for_the_model_being_recommended():
    # The defect: `ungraded` depends on the candidate's own output — a grader
    # returns NOT_APPLICABLE when the candidate produced nothing to compare —
    # so it genuinely differs per candidate. The report recommended model-B,
    # whose 5 items had no applicable grader, while printing model-A's zero.
    m = two_candidates(ungraded_on_best=5)
    text = m.render()
    assert "5 items had no applicable grader" in text, text


def test_the_first_candidates_exclusions_are_not_attributed_to_the_recommended_one():
    # The mirror direction, which a fix that merely swapped the index would
    # still get wrong: model-A has the excluded items, model-B is recommended
    # and has none, so there is nothing to disclose.
    m = two_candidates(ungraded_on_first=5)
    assert "no applicable grader" not in m.render()


def test_underpowered_is_reported_from_the_recommended_model():
    # Same root cause, different warning: `interval.total` shrinks with the
    # candidate's own ungraded count, so `underpowered` is per-candidate too.
    weak = CandidateReport("model-A", (score_cluster("c1", "x", 1.0, results("c1", 50, 50)),))
    thin = CandidateReport("model-B", (score_cluster("c1", "x", 1.0, results("c1", 3, 0)),))
    m = Matrix([weak, thin], incumbent_cost=1000.0)
    if m.best() is thin:
        assert "underpowered" in m.render()


def test_the_header_row_still_names_every_cluster():
    # The header takes names and shares, which every candidate carries
    # identically — the fix must not narrow the table to the winner's clusters.
    m = two_candidates(ungraded_on_best=5)
    text = m.render()
    assert "x" in text
    assert "model-A" in text and "model-B" in text, "both rows must still render"


def test_a_matrix_with_nothing_gradeable_still_renders():
    # `best` is None here, and the disclosures must fall back rather than crash.
    empty = CandidateReport("model-A", (score_cluster("c1", "x", 1.0, []),))
    m = Matrix([empty], incumbent_cost=1000.0)
    assert m.best() is None
    assert m.render()


# --- an impossible traffic share cannot be constructed ---------------------------


def test_a_share_above_one_is_refused_at_the_type():
    # The guard was first written in `stats`, on `weighted_point` and
    # `weighted_posterior` — the wrong layer. Shares arrive through
    # `run(shares=...)`, land here, and `movable_share()` sums them raw without
    # calling a weighted function at all, so `{"good": 10.0}` produced a policy
    # reading "Move 900% of traffic to candidate".
    with pytest.raises(ValueError, match="fraction of traffic"):
        ClusterScore("c", "c", 10.0, wilson(9, 10), 0)


def test_a_negative_share_is_refused_at_the_type():
    with pytest.raises(ValueError, match="fraction of traffic"):
        ClusterScore("c", "c", -1.0, wilson(9, 10), 0)


@pytest.mark.parametrize("share", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_share_is_refused_at_the_type(share):
    # NaN is the one that used to reach the receipt, because it fails every
    # ordinary comparison silently rather than raising.
    with pytest.raises(ValueError, match="finite"):
        ClusterScore("c", "c", share, wilson(9, 10), 0)


@pytest.mark.parametrize("share", [0.0, 0.5, 1.0])
def test_the_shares_a_caller_should_pass_are_accepted(share):
    # Zero is a legitimate "cannot move traffic", and 1.0 is a single-cluster
    # eval set. A guard that rejected either would break documented behaviour.
    assert ClusterScore("c", "c", share, wilson(9, 10), 0).share == share


def test_the_product_entrypoint_refuses_an_impossible_share():
    # The finding was that `suite()` reached `best()` and issued a receipt
    # before any weighted function validated anything. This is the end-to-end
    # check that the guard is now on the path the product actually takes.
    from clickllm.prove import suite
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"i{i}", "good", f"p{i}", '{"a": 1}', '{"a": 1}') for i in range(120)]
    with pytest.raises(ValueError, match="fraction of traffic"):
        suite(items, shares={"good": 10.0}, issued="2026-08-07")
