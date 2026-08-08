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


def test_shares_that_add_up_to_more_traffic_than_exists_are_refused():
    # Per-share validation was not enough, and missing that was the same
    # mistake twice: `movable_share()` *sums* shares, so the value reaching the
    # arithmetic is the total, not any one element. Two individually legal
    # shares rendered "Move 180% of traffic to candidate".
    good = ClusterScore("a", "a", 0.9, wilson(118, 120), 0)
    also = ClusterScore("b", "b", 0.9, wilson(118, 120), 0)
    with pytest.raises(ValueError, match="exceed all of the traffic"):
        CandidateReport("m", (good, also))


def test_shares_that_add_to_exactly_one_are_fine():
    parts = tuple(
        ClusterScore(k, k, s, wilson(118, 120), 0)
        for k, s in (("a", 0.6), ("b", 0.15), ("c", 0.25))
    )
    assert CandidateReport("m", parts).clusters == parts


def test_shares_that_add_to_less_than_one_are_fine():
    # Under-allocation is legitimate: a cluster scored at share 0 because it
    # was absent from the share map contributes nothing to the total.
    parts = (
        ClusterScore("a", "a", 0.4, wilson(9, 10), 0),
        ClusterScore("b", "b", 0.0, wilson(9, 10), 0),
    )
    assert CandidateReport("m", parts)


def test_a_split_landing_a_hair_over_one_is_tolerated_not_refused():
    # Shares are floats and arrive from division upstream, so a caller who
    # split traffic exactly can still land above 1.0. Refusing that would make
    # the guard fire on correct input, which is how a guard gets deleted.
    parts = (
        ClusterScore("a", "a", 0.5, wilson(9, 10), 0),
        ClusterScore("b", "b", 0.5000001, wilson(9, 10), 0),
    )
    assert sum(c.share for c in parts) > 1.0, "precondition: this really does overshoot"
    assert CandidateReport("m", parts)


def test_an_overshoot_past_the_tolerance_is_still_refused():
    # The other side of the same line: tolerance is for representation error,
    # not for a share map that is genuinely wrong.
    parts = (
        ClusterScore("a", "a", 0.5, wilson(9, 10), 0),
        ClusterScore("b", "b", 0.51, wilson(9, 10), 0),
    )
    with pytest.raises(ValueError, match="exceed all of the traffic"):
        CandidateReport("m", parts)


def test_the_product_entrypoint_refuses_over_allocated_shares():
    from clickllm.prove import suite
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"a{i}", "a", f"p{i}", '{"x":1}', '{"x":1}') for i in range(120)]
    items += [EvalItem(f"b{i}", "b", f"q{i}", '{"x":1}', '{"x":1}') for i in range(120)]
    with pytest.raises(ValueError, match="exceed all of the traffic"):
        suite(items, shares={"a": 0.9, "b": 0.9}, issued="2026-08-07")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.5, 2.0])
def test_an_impossible_share_is_refused_even_when_its_cluster_has_no_items(bad):
    # The gap the type-level guard could not close. `ClusterScore` validates
    # its own field, but a share whose cluster has no eval items never becomes
    # a ClusterScore: `_uncovered` keeps only `share > 0`, and NaN and
    # negatives both fail that test, so the value was dropped rather than
    # refused. `run()` therefore checks the raw map before anything filters.
    from clickllm.prove import suite
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"i{i}", "covered", f"p{i}", '{"a":1}', '{"a":1}') for i in range(45)]
    with pytest.raises(ValueError, match="traffic share"):
        suite(items, shares={"covered": 1.0, "uncovered": bad}, issued="2026-08-07")


def test_the_offending_cluster_is_named():
    from clickllm.prove import run
    from clickllm.prove.graders import EvalItem

    items = [EvalItem("i0", "covered", "p", '{"a":1}', '{"a":1}')]
    with pytest.raises(ValueError, match="deprecated"):
        run(items, shares={"covered": 1.0, "deprecated": float("nan")})


def test_a_valid_share_map_with_an_uncovered_cluster_still_works():
    # The guard must not fire on the case the previous commit added support
    # for: a legitimate positive share the eval set never covered.
    from clickllm.prove import run
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"i{i}", "covered", f"p{i}", '{"a":1}', '{"a":1}') for i in range(45)]
    m = run(items, shares={"covered": 0.7, "gap": 0.3})
    assert [c.cluster for c in m.candidates[0].clusters] == ["covered", "gap"]


# --- the door nobody has opened yet ----------------------------------------------

#: `samples_needed` takes a bar and deliberately does not check it. It is a
#: calculator, not a decision — "how many more items at the observed rate" — and
#: it is called in a loop from `render_need`. Every surface that acts on its
#: answer refuses a degenerate bar before it is ever reached.
_NOT_A_DECISION = {"samples_needed", "check_bar"}


def _call_with(bar: float):
    """One minimal, otherwise-valid call per surface that takes a `bar`.

    Calling with `bar` alone would raise `TypeError` on the missing positional
    arguments *before* the body runs, so every surface would look guarded. Each
    of these is a call that would otherwise succeed.
    """
    from clickllm.prove import issue, run, suite
    from clickllm.prove.equivalence import CandidateReport, ClusterScore
    from clickllm.prove.gate import Reading, Stage, decide
    from clickllm.prove.graders import EvalItem
    from clickllm.prove.stats import wilson

    # 41/80 — the candidate nobody should migrate to, at every surface.
    score = ClusterScore("x", "x", 1.0, wilson(41, 80), 0)
    report = CandidateReport("m", (score,))
    items = [EvalItem(f"i{i}", "x", f"p{i}", '{"a":1}', '{"a":2}') for i in range(80)]

    return {
        "clickllm.prove.gate.decide": lambda: decide(
            [Reading(score=score, judge_only=False)], Stage("shadow", 0), bar=bar
        ),
        "clickllm.prove.receipt.issue": lambda: issue(
            report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=bar
        ),
        "clickllm.prove.run": lambda: run(items, shares={"x": 1.0}, bar=bar),
        "clickllm.prove.suite": lambda: suite(
            items, shares={"x": 1.0}, issued="2026-08-07", bar=bar
        ),
    }


def _bar_taking_functions() -> set[str]:
    """Every public function in `clickllm.prove` that takes a `bar`."""
    import importlib
    import inspect
    import pkgutil

    import clickllm.prove as P

    out = set()
    # The package itself, not just its submodules: `run` and `suite` are defined
    # in `prove/__init__.py`, and `walk_packages` alone would walk straight past
    # the two widest entry points in the module.
    names = ["clickllm.prove"] + [
        m.name for m in pkgutil.walk_packages(P.__path__, "clickllm.prove.")
    ]
    for mod_name in names:
        mod = importlib.import_module(mod_name)
        for name, fn in vars(mod).items():
            if name.startswith("_") or not inspect.isfunction(fn):
                continue
            if fn.__module__ != mod_name or name in _NOT_A_DECISION:
                continue
            if "bar" in inspect.signature(fn).parameters:
                out.add(f"{mod_name}.{name}")
    return out


def test_no_surface_taking_a_bar_is_left_without_a_check():
    """A fifth `bar` parameter must fail here, not in an audit.

    The guard has been added four times, once per door, each time after a
    reviewer found the door: `Matrix.__post_init__`, then `issue()`, then
    `Receipt.__post_init__` when a receipt could still arrive from disk, then
    `gate.decide()` — which builds neither object and so inherited neither
    guard, while being the one surface that proposes moving production traffic.
    """
    missing = _bar_taking_functions() - set(_call_with(0.5))
    assert not missing, (
        "these take a `bar` and this test does not exercise them — add a call to "
        f"`_call_with`, or name them in `_NOT_A_DECISION` with a reason: {sorted(missing)}"
    )


@pytest.mark.parametrize(
    "bar",
    [
        # out of range
        0.0,
        -1.0,
        1.0,
        2.0,
        # not a number at all. These belong in the *same* sweep, because the
        # failure they catch is different in kind: a surface that guards the
        # range but touches the value first still raises for the four above —
        # from the range check, inside the object it eventually builds — and
        # raises `TypeError` from a comparison for these. `cli.main()` catches
        # `ValueError` and not `TypeError`, so the second is a traceback where
        # the repo promises a sentence. `issue()` failed exactly here, having
        # had its own guard removed one commit earlier as redundant.
        "0.9",
        None,
        [0.9],
        True,
        10**400,  # finite, and not convertible to a float
    ],
)
def test_every_surface_that_takes_a_bar_refuses_a_degenerate_one(bar):
    unguarded = []
    for name, call in sorted(_call_with(bar).items()):
        try:
            call()
        except ValueError as e:
            if "equivalence bar" in str(e):
                continue
            unguarded.append(f"{name}: rejected {bar!r}, but not as a bar — {e}")
        except Exception as e:  # noqa: BLE001 — the point is which type arrives
            unguarded.append(f"{name}: raised {type(e).__name__}, which the CLI does not catch")
        else:
            unguarded.append(f"{name}: accepted bar={bar!r}")
    assert not unguarded, "surfaces that take a bar without checking it:\n  " + "\n  ".join(
        unguarded
    )


def test_the_same_surfaces_all_accept_a_real_bar():
    """The negative control for the sweep above: these calls are valid apart
    from the bar, so a guard that refused everything would pass that test and
    fail this one."""
    for _name, call in sorted(_call_with(0.90).items()):
        call()  # must not raise


def test_the_guard_fires_before_the_eval_run_it_would_invalidate():
    """The sweep above proves a degenerate bar *raises*. It did before this too —
    from `Matrix.__post_init__`, after every item had been graded and, with a
    judge injected, after a paid call per item.

    The exception was correct and cost a full eval run to arrive at. A guard
    that fires late is a guard the caller pays for, so `run` checks the bar
    before it grades anything, and `suite` inherits that by calling `run` first.

    Free-text items on purpose: `_judged` skips the judge for an item the
    deterministic graders already failed, so a fixture of differing JSON never
    reaches it at *any* bar and "the judge was not called" would prove nothing.
    The paired test below is the control that this fixture does reach it.
    """
    from clickllm.prove import run, suite
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"i{i}", "x", f"p{i}", "the capital is Paris", "Paris") for i in range(80)]
    calls: list[int] = []

    def judge(*a, **k):
        calls.append(1)
        raise AssertionError("the judge must not be reached with an invalid bar")

    for call in (
        lambda: run(items, shares={"x": 1.0}, judge=judge, judge_model="m", bar=0.0),
        lambda: suite(
            items, shares={"x": 1.0}, issued="2026-08-07", judge=judge, judge_model="m", bar=0.0
        ),
    ):
        with pytest.raises(ValueError, match="equivalence bar"):
            call()
    assert calls == [], f"the judge was called {len(calls)} times before the bar was checked"


@pytest.mark.parametrize("value", ["0.9", None, [0.9], {"bar": 0.9}, True])
def test_a_bar_that_is_not_a_number_is_a_sentence_not_a_traceback(value):
    """Both checks reached for the value's *range* first, which assumes it has
    one. `0.0 < "0.9"` raises `TypeError`, and `cli.main()` catches `ValueError`,
    not `TypeError`.

    So a receipt file carrying `"bar": "0.9"` still failed closed — as a
    traceback, where the repo promises a sentence and exit 2 for an untrusted
    file. `True` is in here because it is an `int`: a bar of `True` is a bar of
    1.0 nobody typed.
    """
    from clickllm.prove.equivalence import check_bar

    with pytest.raises(ValueError, match="must be a number"):
        check_bar(value)


@pytest.mark.parametrize("value", ["0.9", None, [0.9], True])
def test_a_share_that_is_not_a_number_is_refused_the_same_way(value):
    """The sibling. `check_share` called `math.isfinite` first, which raises
    `TypeError` on a string exactly as the comparison does — the same defect,
    in the function next to it, and not in the finding that prompted this."""
    from clickllm.prove.equivalence import check_share

    with pytest.raises(ValueError, match="must be a number"):
        check_share(value)


def test_a_receipt_file_with_a_string_bar_reaches_the_cli_as_a_value_error():
    """The route that matters: `from_json` is what `clickllm receipt`, `guard`
    and the box read from disk, and `main()`'s handler lists `ValueError`."""
    import json as _json

    from clickllm.prove import Receipt, issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    blob = _json.loads(good.to_json())
    blob["receipt"]["bar"] = "0.9"
    with pytest.raises(ValueError, match="must be a number"):
        Receipt.from_json(_json.dumps(blob))


def test_the_numbers_that_are_numbers_still_pass():
    """The negative control for the four tests above."""
    from clickllm.prove.equivalence import check_bar, check_share

    for v in (0.9, 0.5, 1e-9):
        check_bar(v)
    for v in (0.0, 0.5, 1.0, 0, 1):
        check_share(v)


def test_a_real_bar_still_reaches_the_judge():
    """The negative control, and it earned its place: the first version of the
    test above used differing-JSON items, which `_judged` never sends to the
    judge because a graded failure cannot be rescued. `calls == []` held for a
    reason that had nothing to do with the guard, and this test is what caught
    it."""
    from clickllm.prove import run
    from clickllm.prove.graders import EvalItem

    items = [EvalItem(f"i{i}", "x", f"p{i}", "the capital is Paris", "Paris") for i in range(4)]
    calls: list[int] = []

    def judge(*a, **k):
        calls.append(1)
        return None

    run(items, shares={"x": 1.0}, judge=judge, judge_model="m", bar=0.90)
    assert calls, "a valid bar must still run the eval"


@pytest.mark.parametrize(
    "weight", [float("nan"), float("inf"), float("-inf"), 10**400, "0.5", None, True]
)
def test_a_weight_that_cannot_be_averaged_is_refused_before_the_arithmetic(weight):
    """The third site with the same flaw, and the one the review did not name.

    `_check_weights` called `math.isfinite` directly, so a string raised
    `TypeError` and an `int` too large to convert raised `OverflowError` — and
    when the check was first narrowed to let ints through as "finite by
    construction", `weighted_point` raised `OverflowError` on the next line
    instead. Guarding the value is not enough if the guard's idea of valid is
    wider than the arithmetic's.
    """
    from clickllm.prove.stats import weighted_point, weighted_posterior, wilson

    for fn in (weighted_point, weighted_posterior):
        with pytest.raises(ValueError, match="traffic share"):
            fn([(wilson(41, 80), weight)])


def test_weights_that_can_be_averaged_still_are():
    """The negative control for the sweep above."""
    from clickllm.prove.stats import weighted_point, wilson

    assert round(weighted_point([(wilson(41, 80), 1.0)]), 4) == 0.5125
    assert round(weighted_point([(wilson(40, 40), 1), (wilson(0, 40), 1)]), 4) == 0.5
