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
    and the box read from disk, and `main()`'s handler lists `ValueError`.

    The message moved once the receipt gained a whole-dataclass type sweep,
    which now runs before `check_bar` and reports the field path instead —
    "Receipt.bar must be int or float, got str". This asserts the two things the
    test is named for rather than a wording: the exception type the CLI catches,
    and that the sentence identifies the field and the value. `check_bar`'s own
    message is covered directly above, at the layer that produces it.
    """
    import json as _json

    from clickllm.prove import Receipt, issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    blob = _json.loads(good.to_json())
    blob["receipt"]["bar"] = "0.9"
    with pytest.raises(ValueError) as caught:
        Receipt.from_json(_json.dumps(blob))
    msg = str(caught.value)
    assert "bar" in msg and "0.9" in msg, msg


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


# --- what the coverage caveat is actually claiming --------------------------------


def _coverage_line(scores) -> str:
    m = Matrix([CandidateReport("m", tuple(scores))], incumbent_cost=1000.0)
    return next((line for line in m.render().splitlines() if "weighted verdict" in line), "")


def test_traffic_that_was_collected_but_ungraded_is_not_called_unmeasured():
    """`known` means *graded* (`interval.total > 0`), so a cluster whose items
    all came back with no applicable grader counted as unmeasured and the report
    said "was never measured".

    Those items were collected and run, and excluded at grading. The
    conservative arithmetic is right either way; the sentence was wrong in one
    of the two cases, and the distinction is free — a cluster carries its
    `ungraded` count.
    """
    line = _coverage_line(
        [
            ClusterScore("good", "good", 0.7, wilson(118, 120), 0),
            ClusterScore("silent", "silent", 0.3, wilson(0, 0), 12),
        ]
    )
    assert "never measured" not in line, line
    assert "could be graded" in line, line


def test_traffic_that_really_was_never_measured_still_says_so():
    """The negative control: softening the wording everywhere would be a
    different lie."""
    line = _coverage_line(
        [
            ClusterScore("good", "good", 0.7, wilson(118, 120), 0),
            ClusterScore("absent", "absent", 0.3, wilson(0, 0), 0),
        ]
    )
    assert "never measured" in line, line


def test_a_mixture_names_both_rather_than_picking_one():
    line = _coverage_line(
        [
            ClusterScore("good", "good", 0.4, wilson(118, 120), 0),
            ClusterScore("silent", "silent", 0.3, wilson(0, 0), 9),
            ClusterScore("absent", "absent", 0.3, wilson(0, 0), 0),
        ]
    )
    assert "some of it never measured" in line, line


def test_full_coverage_says_nothing_at_all():
    assert "of traffic;" not in _coverage_line(
        [ClusterScore("good", "good", 1.0, wilson(118, 120), 0)]
    )


def test_a_cluster_carrying_no_traffic_is_not_a_policy_item():
    """`unproven()` includes every unknown regardless of share, so a 0%-share
    cluster landed in `HybridPolicy.unproven_clusters` — printed under "Not yet
    proven" directly beneath "Move 100% of traffic", which reads as a
    contradiction and is really just noise."""
    m = Matrix(
        [
            CandidateReport(
                "m",
                (
                    ClusterScore("live", "live", 1.0, wilson(118, 120), 0),
                    ClusterScore("deprecated", "deprecated", 0.0, wilson(0, 0), 0),
                ),
            )
        ],
        incumbent_cost=1000.0,
    )
    policy = m.hybrid_for(m.candidates[0])
    assert "deprecated" not in policy.unproven_clusters, policy.unproven_clusters
    assert policy.moved_share == 1.0
    # `needs` must be filtered identically to `unproven_clusters`, or render()
    # prints a "→" remediation bullet for a cluster the header never named.
    assert not any("deprecated" in n for n in policy.needs), policy.needs
    assert "deprecated" not in policy.render()


def test_a_cluster_that_carries_traffic_is_still_named():
    """The negative control: filtering on share must not empty the list."""
    m = Matrix(
        [
            CandidateReport(
                "m",
                (
                    ClusterScore("live", "live", 0.7, wilson(118, 120), 0),
                    ClusterScore("thin", "thin", 0.3, wilson(3, 3), 0),
                ),
            )
        ],
        incumbent_cost=1000.0,
    )
    assert "thin" in m.hybrid_for(m.candidates[0]).unproven_clusters


# --- the check that could not run ------------------------------------------------


def _receipt_with(fingerprints=None):
    from clickllm.prove import EvalItem, suite

    items = [
        EvalItem(item_id=str(i), cluster="c", prompt="p", baseline="x", candidate="x")
        for i in range(40)
    ]
    return suite(
        items,
        shares={"c": 1.0},
        candidate="cand",
        incumbent="inc",
        issued="2026-08-11",
        fingerprints=fingerprints,
    ).receipt


def test_a_receipt_with_no_fingerprints_says_the_check_did_not_run():
    """The most important of guard's three checks was passing by doing nothing.

    `guard.check` iterates `receipt.fingerprints`, and `clickllm prove` had no
    flag to populate it — so *every* receipt the CLI produced recorded none, the
    loop ran zero times, and a caller supplying correct current fingerprints was
    told the receipt still holds. A provider could swap the model underneath a
    team gating deploys on this and nothing would say so.
    """
    from datetime import date

    from clickllm import guard

    p = guard.check(
        _receipt_with(None),
        today=date(2026, 8, 12),
        fingerprints={"cand": "sha256:whatever"},
    )
    kinds = {f.kind for f in p.findings}
    assert guard.Drift.UNCHECKABLE in kinds, "silence again"
    assert "--fingerprints" in p.action


def test_not_being_able_to_check_is_not_evidence_that_it_changed():
    """`UNCHECKABLE` must not void the receipt. "I could not tell" and "it
    changed" are different claims, and voiding on the first would make every
    receipt issued before this existed read as broken."""
    from datetime import date

    from clickllm import guard

    p = guard.check(
        _receipt_with(None),
        today=date(2026, 8, 12),
        fingerprints={"cand": "sha256:whatever"},
    )
    assert p.valid is True
    assert not any(f.invalidates for f in p.findings)


def test_a_receipt_that_records_fingerprints_detects_a_swap():
    """The capability the flag restores, end to end."""
    from datetime import date

    from clickllm import guard

    r = _receipt_with({"cand": "sha256:aaaa"})
    changed = guard.check(r, today=date(2026, 8, 12), fingerprints={"cand": "sha256:bbbb"})
    assert not changed.valid
    assert guard.Drift.MODEL_CHANGED in {f.kind for f in changed.findings}

    same = guard.check(r, today=date(2026, 8, 12), fingerprints={"cand": "sha256:aaaa"})
    assert same.valid and not same.findings


def test_no_finding_when_the_caller_supplied_nothing_to_compare():
    """The control that keeps the new finding from becoming noise: a caller who
    did not pass fingerprints is not asking the question, and answering it
    anyway would fire on every plain `clickllm guard`."""
    from datetime import date

    from clickllm import guard

    for r in (_receipt_with(None), _receipt_with({"cand": "sha256:aaaa"})):
        p = guard.check(r, today=date(2026, 8, 12))
        assert not p.findings, p.findings


def test_the_prove_command_actually_records_the_fingerprints_it_was_given(tmp_path):
    """Removing the wiring from `cmd_prove` left every unit test above green.

    They exercise `suite(fingerprints=...)`, which is the *library*. The defect
    being fixed was that no caller passed it — so the test has to be the command,
    not the function it eventually calls. Same shape as the capture guard whose
    call site could be deleted with its own unit tests still passing.
    """
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    evalset = tmp_path / "e.json"
    evalset.write_text(
        json.dumps(
            [
                {
                    "item_id": str(i),
                    "cluster": "c",
                    "prompt": "p",
                    "baseline": "x",
                    "candidate": "x",
                }
                for i in range(40)
            ]
        )
    )
    fp = tmp_path / "fp.json"
    fp.write_text(json.dumps({"cand": "sha256:aaaa", "inc": 12345}))
    out = tmp_path / "r.json"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "clickllm.cli",
            "prove",
            str(evalset),
            "--out",
            str(out),
            "--issued",
            "2026-08-11",
            "--fingerprints",
            str(fp),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    # Not `returncode == 0`: `cmd_prove` returns 0 only when it proposes moving
    # traffic, and a verdict of "move nothing" is a successful run with a no.
    # Asserting 0 here would be asserting that the candidate passed.
    assert out.exists(), proc.stdout + proc.stderr
    recorded = json.loads(out.read_text())["receipt"]["fingerprints"]
    assert recorded["cand"] == "sha256:aaaa"
    # Stringified, not trusted: a JSON number would be compared against a string
    # later and always differ, reporting a model change that never happened —
    # the one false alarm guaranteed to make someone stop believing the alert.
    assert recorded["inc"] == "12345"


def test_prove_without_the_flag_still_works_and_records_nothing(tmp_path):
    """The control for the above: the flag is optional, and a receipt without it
    is still a receipt — it just cannot answer the model-change question, which
    `guard` now says out loud rather than passing silently."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path as _P

    root = _P(__file__).resolve().parents[1]
    evalset = tmp_path / "e.json"
    evalset.write_text(
        json.dumps(
            [{"item_id": "1", "cluster": "c", "prompt": "p", "baseline": "x", "candidate": "x"}]
        )
    )
    out = tmp_path / "r.json"
    proc = subprocess.run(
        [sys.executable, "-m", "clickllm.cli", "prove", str(evalset), "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert out.exists(), proc.stdout + proc.stderr
    assert json.loads(out.read_text())["receipt"]["fingerprints"] == {}


def test_the_saving_does_not_claim_zero_quality_loss():
    """It said "at zero measured quality loss", on the one number a reader most
    wants to believe. It was false.

    A cluster moves when its whole interval clears the bar. At a 0.90 bar a
    cluster passing 98% [95–99] moves — and in this 200-item run that is four
    items whose answers measurably differed. "Zero loss" was the strongest
    possible claim attached to the strongest possible incentive to believe it.

    What is true, and what it says now: the moved traffic met the bar the reader
    chose, and the bar is named so they can judge it.
    """
    from clickllm.prove import EvalItem, suite

    items = [
        EvalItem(
            item_id=str(i),
            cluster="c",
            prompt=f"prompt {i}",  # distinct: identical prompts are deduplicated
            baseline="the answer",
            candidate="the answer" if i >= 4 else f"something else {i}",
        )
        for i in range(200)
    ]
    r = suite(
        items,
        shares={"c": 1.0},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        bar=0.90,
        traffic_captures=400,
        traffic_window="14 days",
    )

    score = r.matrix.candidates[0].clusters[0]
    differed = sum(1 for _id, ok in score.outcomes if not ok)
    assert differed == 4, f"the fixture must contain real differences, had {differed}"
    assert r.policy.moved_share == 1.0, "the fixture must actually clear the bar"

    saving = next(line for line in r.policy.render().splitlines() if "Saving" in line)
    assert "zero measured quality loss" not in saving, saving
    assert "90% bar" in saving, saving
    # And it is a range, not a point: the share it derives from was measured.
    assert "–" in saving and "~" in saving, saving


def test_the_saving_line_names_whichever_bar_was_set():
    """The claim has to track the bar, or it is decoration."""
    from clickllm.prove import EvalItem, suite

    items = [
        EvalItem(item_id=str(i), cluster="c", prompt=f"p{i}", baseline="a", candidate="a")
        for i in range(200)
    ]
    for bar in (0.90, 0.99):
        r = suite(
            items,
            shares={"c": 1.0},
            issued="2026-08-12",
            incumbent_cost=100.0,
            monthly_cost=10.0,
            bar=bar,
            traffic_captures=400,
            traffic_window="14 days",
        )
        saving = next(line for line in r.policy.render().splitlines() if "Saving" in line)
        assert f"{bar * 100:.0f}% bar" in saving, saving


def test_an_unknown_cost_is_still_refused_rather_than_guessed():
    """The control for both: the honest wording must not have made a saving
    appear where there is no rate to compute one from."""
    from clickllm.prove import EvalItem, suite

    items = [
        EvalItem(item_id=str(i), cluster="c", prompt=f"p{i}", baseline="a", candidate="a")
        for i in range(200)
    ]
    r = suite(items, shares={"c": 1.0}, issued="2026-08-12")
    saving = next(line for line in r.policy.render().splitlines() if "Saving" in line)
    assert "unknown" in saving and "no cost rate" in saving, saving
