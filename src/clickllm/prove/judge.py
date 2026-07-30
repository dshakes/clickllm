"""Tier 3 — pairwise LLM judging, used only when tiers 1 and 2 cannot decide.

This module implements the **protocol**, not a judge. The model is injected by
the caller as a plain callable. There is deliberately no built-in judge and no
default model: a hidden judge is an unauditable one, and every report must name
which model produced its verdicts.

Three properties the protocol enforces:

**Position swap.** LLM judges have a well-documented preference for whichever
response they see first. Every comparison is run twice with the order flipped. If
the two runs disagree once mirrored, that is *position bias*, and the result is
``UNCERTAIN`` — never the average of the two. Averaging a disagreement produces a
number with no meaning attached to it.

**Disclosure.** The judge model, the number of comparisons, and the measured
human-agreement rate travel with the result. A score whose provenance is missing
is reported as unknown.

**Subordinate.** A judge verdict never authorises a cutover on its own. Shadow
mode against live traffic is the real gate; this tier exists to explain *why* a
cluster differs, and to rank candidates that the deterministic tiers tie.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .graders import EvalItem, ItemResult, Score, Tier
from .graders import Outcome as GraderOutcome
from .stats import Interval, wilson


class Verdict(StrEnum):
    """The result of one pairwise comparison, from the candidate's point of view."""

    CANDIDATE_BETTER = "candidate_better"
    EQUIVALENT = "equivalent"
    BASELINE_BETTER = "baseline_better"
    #: The judge declined, errored, or contradicted itself across positions.
    UNCERTAIN = "uncertain"

    def mirrored(self) -> Verdict:
        """This verdict as it would read with the two responses swapped."""
        return {
            Verdict.CANDIDATE_BETTER: Verdict.BASELINE_BETTER,
            Verdict.BASELINE_BETTER: Verdict.CANDIDATE_BETTER,
        }.get(self, self)

    @property
    def acceptable(self) -> bool:
        """Whether this counts as the candidate holding the bar.

        Equivalent or better. "Uncertain" is not acceptable — an unresolved
        comparison must not be counted as a pass.
        """
        return self in (Verdict.EQUIVALENT, Verdict.CANDIDATE_BETTER)


@dataclass(frozen=True, slots=True)
class Comparison:
    """What the judge is asked. ``a`` and ``b`` are deliberately anonymous."""

    prompt: str
    a: str
    b: str


@dataclass(frozen=True, slots=True)
class Reply:
    """What a judge returns. ``winner`` is "a", "b", or "tie"."""

    winner: str
    reasoning: str = ""

    def as_verdict(self, candidate_is: str) -> Verdict:
        """Translate a positional answer into a candidate-relative verdict."""
        w = self.winner.strip().lower()
        if w == "tie":
            return Verdict.EQUIVALENT
        if w not in ("a", "b"):
            return Verdict.UNCERTAIN
        return Verdict.CANDIDATE_BETTER if w == candidate_is else Verdict.BASELINE_BETTER


#: A judge is any callable from a comparison to a reply. Sync on purpose: the
#: caller owns concurrency, retries, and rate limiting.
JudgeFn = Callable[[Comparison], Reply]


@dataclass(frozen=True, slots=True)
class JudgeResult:
    """A position-swapped comparison and its provenance."""

    verdict: Verdict
    #: True when the two orderings disagreed — the judge preferred a position,
    #: not a response.
    position_bias: bool
    first_pass: Verdict
    second_pass: Verdict
    model: str
    reasoning: str = ""

    def to_score(self) -> Score:
        """Fold into the common grader vocabulary.

        An uncertain verdict is NOT_APPLICABLE rather than FAIL: the judge failed
        to decide, which says nothing about the candidate, and scoring it as a
        failure would penalise a model for our own instrument's ambiguity.
        """
        if self.verdict is Verdict.UNCERTAIN:
            detail = "position bias" if self.position_bias else "judge could not decide"
            return Score("pairwise-judge", Tier.JUDGE, GraderOutcome.NOT_APPLICABLE, detail)
        outcome = GraderOutcome.PASS if self.verdict.acceptable else GraderOutcome.FAIL
        return Score(
            "pairwise-judge", Tier.JUDGE, outcome, f"{self.verdict.value}: {self.reasoning[:160]}"
        )


def judge_item(item: EvalItem, judge: JudgeFn, model: str) -> JudgeResult:
    """Compare candidate against baseline, both ways round.

    Args:
        judge: the model, supplied by the caller.
        model: its identifier, for disclosure. Required — an anonymous judge
            makes the resulting score unauditable.

    Raises:
        ValueError: if ``model`` is blank.
    """
    if not model.strip():
        raise ValueError("a judge model identifier is required for disclosure")

    # Pass 1: baseline first. Pass 2: candidate first.
    try:
        r1 = judge(Comparison(item.prompt, a=item.baseline, b=item.candidate))
        v1 = r1.as_verdict(candidate_is="b")
    except Exception as e:  # noqa: BLE001 — a judge failure must not abort the run
        return JudgeResult(
            Verdict.UNCERTAIN,
            False,
            Verdict.UNCERTAIN,
            Verdict.UNCERTAIN,
            model,
            f"judge error: {e}",
        )
    try:
        r2 = judge(Comparison(item.prompt, a=item.candidate, b=item.baseline))
        v2 = r2.as_verdict(candidate_is="a")
    except Exception as e:  # noqa: BLE001
        return JudgeResult(
            Verdict.UNCERTAIN, False, v1, Verdict.UNCERTAIN, model, f"judge error on swap: {e}"
        )

    if v1 is Verdict.UNCERTAIN or v2 is Verdict.UNCERTAIN:
        return JudgeResult(Verdict.UNCERTAIN, False, v1, v2, model, r1.reasoning)

    if v1 == v2:
        return JudgeResult(v1, False, v1, v2, model, r1.reasoning)

    # Disagreement across positions. Not averaged — averaged disagreement is a
    # number with nothing behind it.
    return JudgeResult(
        Verdict.UNCERTAIN,
        True,
        v1,
        v2,
        model,
        f"position bias: saw {v1.value} then {v2.value} when swapped",
    )


@dataclass(frozen=True, slots=True)
class Agreement:
    """How often the judge matched a human on a sampled subset."""

    agreed: int
    total: int
    model: str

    @property
    def rate(self) -> float | None:
        """``None`` when nothing was sampled — never a default of 1.0."""
        return self.agreed / self.total if self.total else None

    def render(self) -> str:
        r = self.rate
        if r is None:
            return f"judge: {self.model} · human agreement UNMEASURED"
        return f"judge: {self.model}, position-swapped · human agreement {r:.2f} (n={self.total})"

    @property
    def trustworthy(self) -> bool:
        """Whether judge scores should be presented as evidence at all.

        An unmeasured judge is not a trusted judge. The bar is deliberately
        conservative: below it, the report shows judge cells as unknown.
        """
        r = self.rate
        return r is not None and self.total >= 20 and r >= 0.75


# --- calibration ---------------------------------------------------------------
# `Agreement` needs a human, which most teams will never sample. Calibration
# needs nothing: it falls out of the run itself, from the items where a
# deterministic grader and the judge both had an opinion.

#: Fraction of deterministic verdicts the judge must reproduce before its own
#: verdicts count for anything. A calibration knob, not a truth — 0.70 says "wrong
#: about three items in ten is the most we will tolerate from an instrument", and
#: it is stated here so it can be argued with.
CALIBRATION_FLOOR = 0.70

#: Overlapping items below which the calibration rate is noise, not a measurement.
MIN_CALIBRATION_ITEMS = 10


@dataclass(frozen=True, slots=True)
class Calibration:
    """How often the judge reproduced a deterministic grader's verdict.

    An LLM judge is an instrument, and an instrument nobody checked against a
    known standard is not measuring anything. ``Agreement`` is the gold standard
    for that check and needs a human to sit down with 40 items; this is the one
    that comes free, because every run already scores some items both ways.

    **What it can and cannot see.** The suite does not pay a judge to re-confirm
    a structural failure, so the overlap is almost entirely items the
    deterministic graders *passed*. That makes this a measure of the judge's
    false-FAIL rate — how often it condemns work that is demonstrably fine — and
    it says nothing about false PASSes. That is the useful direction for the one
    thing judge-only evidence is allowed to do, which is veto.
    """

    agreed: int
    total: int
    model: str

    @property
    def rate(self) -> float | None:
        """``None`` when nothing overlapped — never a default of 1.0."""
        return self.agreed / self.total if self.total else None

    @property
    def interval(self) -> Interval:
        """Wilson interval on the agreement rate. Reported, not gated on."""
        return wilson(self.agreed, self.total)

    @property
    def calibrated(self) -> bool:
        """Whether this judge has earned the right to block a migration."""
        r = self.rate
        return r is not None and self.total >= MIN_CALIBRATION_ITEMS and r >= CALIBRATION_FLOOR

    def render(self) -> str:
        if self.total == 0:
            return f"judge calibration: {self.model} · UNCALIBRATED (no item scored both ways)"
        state = "calibrated" if self.calibrated else "BELOW THE FLOOR"
        return (
            f"judge calibration: {self.model} matched the deterministic graders on "
            f"{self.interval.render()} of {self.total} items scored both ways "
            f"({state}, floor {CALIBRATION_FLOOR:.0%} over "
            f"{MIN_CALIBRATION_ITEMS}+ items)"
        )


def calibrate(results: Sequence[ItemResult], model: str) -> Calibration:
    """Measure the judge against the deterministic graders on the same items.

    An item counts only when both sides were applicable. Agreement is on the
    item's verdict, not on individual graders: the deterministic side is
    conjunctive (see :class:`~.graders.ItemResult`), so the comparison is
    "did the judge reach the same conclusion", which is the claim being made.
    """
    agreed = total = 0
    for r in results:
        judged = [s for s in r.applicable if s.tier is Tier.JUDGE]
        determined = [s for s in r.applicable if s.tier is not Tier.JUDGE]
        if not judged or not determined:
            continue
        total += 1
        agreed += all(s.passed for s in judged) == all(s.passed for s in determined)
    return Calibration(agreed=agreed, total=total, model=model)


def demo() -> None:
    item = EvalItem(
        "i1", "c1", "Summarise this.", baseline="The cat sat.", candidate="A cat sat down."
    )

    # A consistent judge produces a confident verdict.
    def fair(c: Comparison) -> Reply:
        return Reply("tie", "both convey the same fact")

    r = judge_item(item, fair, model="claude-opus-5")
    assert r.verdict is Verdict.EQUIVALENT and not r.position_bias
    assert r.to_score().passed

    # A judge that always picks whatever it sees first is pure position bias.
    def first_wins(c: Comparison) -> Reply:
        return Reply("a", "the first one reads better")

    biased = judge_item(item, first_wins, model="m")
    assert biased.position_bias, "always-pick-A must be detected as positional"
    assert biased.verdict is Verdict.UNCERTAIN
    assert not biased.to_score().passed
    # ...and it must not be scored as a failure of the candidate either.
    assert not biased.to_score().applicable, "our instrument's ambiguity is not the model's fault"

    # A genuinely consistent preference survives the swap.
    def prefers_candidate(c: Comparison) -> Reply:
        return Reply("b" if c.b == item.candidate else "a", "more precise")

    good = judge_item(item, prefers_candidate, model="m")
    assert good.verdict is Verdict.CANDIDATE_BETTER and not good.position_bias

    # A judge that errors does not abort the run.
    def broken(c: Comparison) -> Reply:
        raise RuntimeError("rate limited")

    err = judge_item(item, broken, model="m")
    assert err.verdict is Verdict.UNCERTAIN and "rate limited" in err.reasoning

    # Garbage output is uncertain, not a coin flip.
    assert judge_item(item, lambda c: Reply("maybe?"), model="m").verdict is Verdict.UNCERTAIN

    # Disclosure is mandatory.
    try:
        judge_item(item, fair, model="  ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an anonymous judge must be rejected")

    # Agreement: unmeasured is unmeasured, never assumed perfect.
    assert Agreement(0, 0, "m").rate is None
    assert "UNMEASURED" in Agreement(0, 0, "m").render()
    assert not Agreement(0, 0, "m").trustworthy
    assert not Agreement(19, 19, "m").trustworthy, "too few samples to trust"
    assert not Agreement(14, 20, "m").trustworthy, "70% agreement is too low"
    assert Agreement(36, 40, "claude-opus-5").trustworthy

    assert Verdict.CANDIDATE_BETTER.mirrored() is Verdict.BASELINE_BETTER
    assert Verdict.EQUIVALENT.mirrored() is Verdict.EQUIVALENT

    # Calibration: measured against the deterministic graders, free, every run.
    def scored(det_pass: bool, judge_pass: bool) -> ItemResult:
        outcome = GraderOutcome.PASS if det_pass else GraderOutcome.FAIL
        j = GraderOutcome.PASS if judge_pass else GraderOutcome.FAIL
        return ItemResult(
            "i",
            "c",
            (
                Score("json-shape", Tier.ASSERT, outcome),
                Score("pairwise-judge", Tier.JUDGE, j),
            ),
        )

    # An instrument nobody checked is not calibrated, and cannot become so by
    # being asked politely.
    assert calibrate([], "m").rate is None
    assert not calibrate([], "m").calibrated
    assert "UNCALIBRATED" in calibrate([], "m").render()

    # Twelve items scored both ways, the judge matching on eleven.
    agreeing = [scored(True, True)] * 11 + [scored(True, False)]
    cal = calibrate(agreeing, "claude-opus-5")
    assert (cal.agreed, cal.total) == (11, 12), cal
    assert cal.calibrated and "calibrated" in cal.render()
    # ...and the rate carries its interval, like every other number here.
    assert cal.interval.render() in cal.render()

    # A judge that condemns everything the graders passed is measured as such.
    contrarian = calibrate([scored(True, False)] * 12, "m")
    assert contrarian.rate == 0.0 and not contrarian.calibrated
    assert "BELOW THE FLOOR" in contrarian.render()

    # Items where only one side had an opinion are not counted either way.
    only_judge = ItemResult("i", "c", (Score("pairwise-judge", Tier.JUDGE, GraderOutcome.PASS),))
    assert calibrate([only_judge] * 20, "m").total == 0

    print(Agreement(36, 40, "claude-opus-5").render())
    print(cal.render())
    print("ok")


if __name__ == "__main__":
    demo()
