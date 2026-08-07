"""M9 — the quality gate: what the evidence permits, and what it demands.

Everything before this milestone gathers evidence. This module is the thing that
*acts* on it, which makes it the most dangerous file in the product: it is what
stands between a model that quietly got worse and a customer's users.

## The asymmetry, which is the whole design

Advancing and rolling back are not two directions of one decision. They have
wildly different costs when wrong:

| | Cost of a false positive |
|---|---|
| **Roll back** | a little lost progress; re-advance tomorrow |
| **Advance** | production traffic served by a model that is worse |

So they get different rules:

- **Rollback is automatic, and deliberately easy to trigger.** Waiting for
  statistical certainty while serving regressions is the wrong trade. There are
  two grades of it — *proven* (the whole confidence interval sits below the bar)
  and *precautionary* (the point estimate is bad and we have enough samples to be
  worried, but not enough to be sure). Both act. They are labelled differently
  because a human reading the log deserves to know which one fired.
- **Advancing is a proposal, never an action.** The gate says "the evidence
  supports 25%". A human moves it. Anything else would make an automated system
  the last word on production traffic, and no confidence interval earns that.

## Three rules inherited from the product's invariants

1. **An LLM judge alone never moves traffic.** A cluster whose entire case rests
   on `Tier.JUDGE` scores cannot support an advance — [`Reading.judge_only`].
   It can still trigger a rollback, because the asymmetry above cuts that way —
   *provided the judge has been calibrated*. An instrument nobody checked against
   a known standard is not measuring anything, and a rollback fired on an
   unmeasured instrument is a coin toss wearing a lab coat. Uncalibrated
   judge-only evidence produces a HOLD that names the problem, never a silent
   pass. See [`Calibration`][clickllm.prove.judge.Calibration].
2. **Underpowered never advances.** Fewer than [`MIN_SAMPLES_FOR_CONFIDENCE`]
   samples is not evidence, and the decision says how many more are needed rather
   than reporting a number it does not have.
3. **Health beats quality.** A candidate that errors or crawls is not "slightly
   less equivalent" — it is broken, and no grading result rescues it. Health
   trips a rollback on its own, before quality is even considered.

## What the gate does not decide

Whether a *regressed* cluster means rollback depends on whether it is currently
being served. In shadow nothing is at risk, so a losing cluster is pinned to the
incumbent and the migration continues around it — that is the hybrid policy, not
a failure. Under canary or cut the same cluster is live, and the same evidence
means stop. The stage is an input for exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from clickllm.prove.equivalence import DEFAULT_EQUIVALENCE_BAR, ClusterScore, check_bar
from clickllm.prove.graders import ItemResult, Tier
from clickllm.prove.judge import Agreement, Calibration
from clickllm.prove.stats import MIN_SAMPLES_FOR_CONFIDENCE, Interval

__all__ = [
    "ERROR_RATE_MARGIN",
    "LADDER",
    "LATENCY_REGRESSION_RATIO",
    "MIN_HEALTH_SAMPLES",
    "PRECAUTION_MARGIN",
    "PRECAUTION_MIN_SAMPLES",
    "Action",
    "Decision",
    "Health",
    "Reading",
    "Stage",
    "decide",
    "demo",
    "read_cluster",
]

# --- policy knobs --------------------------------------------------------------
# Calibration, not truth. Each is a judgement call about how much worse is too
# much, and each is stated here rather than buried in a comparison so it can be
# argued with.

#: Candidate p95 above this multiple of the incumbent's is a rollback. A model
#: that is right but three times slower has not replaced anything.
LATENCY_REGRESSION_RATIO = 2.0

#: Percentage points of error rate above the incumbent's that trips a rollback.
#: Not an absolute rate: an incumbent that already 500s 3% of the time should not
#: make the candidate look broken for matching it.
ERROR_RATE_MARGIN = 0.02

#: Health signals below this many requests are noise, not a trend.
MIN_HEALTH_SAMPLES = 20

#: How far a point estimate must sit below the bar to trigger a *precautionary*
#: rollback — one taken before the interval proves anything. Wide enough that
#: ordinary sampling wobble does not fire it.
PRECAUTION_MARGIN = 0.10

#: Minimum graded items before a precautionary rollback is allowed. Below this a
#: bad point estimate really is just noise.
PRECAUTION_MIN_SAMPLES = 12


class Action(StrEnum):
    """What the gate concluded.

    Three, not two: "the evidence is not yet sufficient" is a real answer and
    collapsing it into either of the others would be a lie in one direction or
    the other.
    """

    HOLD = "hold"
    ADVANCE = "advance"
    ROLL_BACK = "roll_back"


@dataclass(frozen=True, slots=True)
class Stage:
    """Where a migration currently sits.

    `percent` is the share of live traffic on the candidate. Shadow is 0 by
    definition — the candidate is scored, never served.
    """

    phase: str
    percent: int = 0

    @property
    def live(self) -> bool:
        """Whether real users are currently seeing the candidate's output."""
        return self.percent > 0

    def render(self) -> str:
        """Human-readable, e.g. `canary 25%`."""
        return self.phase if not self.live else f"{self.phase} {self.percent}%"


#: The promotion ladder. Deliberately gradual at the start, where the evidence is
#: thinnest, and never jumping straight from shadow to a cut.
LADDER: tuple[Stage, ...] = (
    Stage("shadow", 0),
    Stage("canary", 5),
    Stage("canary", 25),
    Stage("canary", 50),
    Stage("cut", 100),
)


def _next_stage(current: Stage) -> Stage | None:
    """The rung above `current`, or None at the top.

    An unrecognised stage returns None rather than guessing. A migration in a
    state this module does not model is one it should not be advancing.
    """
    for i, rung in enumerate(LADDER):
        if rung.phase == current.phase and rung.percent == current.percent:
            return LADDER[i + 1] if i + 1 < len(LADDER) else None
    return None


@dataclass(frozen=True, slots=True)
class Health:
    """Operational signals, independent of output quality.

    Latencies are optional because an incumbent that was never measured is a real
    situation, and inventing a baseline to compare against would be worse than
    declining to compare.
    """

    requests: int
    errors: int
    baseline_p95_ms: int | None = None
    candidate_p95_ms: int | None = None
    baseline_requests: int = 0
    baseline_errors: int = 0

    @property
    def measured(self) -> bool:
        """Whether there is enough traffic for these numbers to mean anything."""
        return self.requests >= MIN_HEALTH_SAMPLES

    @property
    def error_rate(self) -> float | None:
        """Candidate error rate, or None when there is nothing to divide by."""
        return None if self.requests == 0 else self.errors / self.requests

    @property
    def baseline_error_rate(self) -> float | None:
        """Incumbent error rate, or None when it was not observed."""
        if self.baseline_requests == 0:
            return None
        return self.baseline_errors / self.baseline_requests

    def latency_ratio(self) -> float | None:
        """Candidate p95 as a multiple of the incumbent's, when both are known."""
        b, c = self.baseline_p95_ms, self.candidate_p95_ms
        if b is None or c is None or b <= 0:
            return None
        return c / b


@dataclass(frozen=True, slots=True)
class Reading:
    """One cluster's evidence, plus where that evidence came from.

    [`ClusterScore`] knows how many items passed. It does not know whether they
    passed because a deterministic grader checked the JSON parsed or because a
    language model was asked for an opinion. That distinction decides whether
    this cluster may move traffic, so it is carried here.
    """

    score: ClusterScore
    #: Every applicable score came from [`Tier.JUDGE`]. Such a cluster may block
    #: a migration but never advance one.
    judge_only: bool

    @property
    def cluster(self) -> str:
        """Cluster key."""
        return self.score.cluster

    @property
    def name(self) -> str:
        """Human-readable cluster name."""
        return self.score.name

    @property
    def share(self) -> float:
        """Fraction of traffic this cluster represents."""
        return self.score.share

    @property
    def interval(self) -> Interval:
        """Equivalence rate with its confidence bounds."""
        return self.score.interval

    def proven_regression(self, bar: float) -> bool:
        """Even the optimistic reading is below the bar."""
        return self.interval.total > 0 and self.interval.clearly_below(bar)

    def suspected_regression(self, bar: float) -> bool:
        """Point estimate is materially bad, with enough samples to worry.

        Deliberately weaker than [`proven_regression`][Reading.proven_regression]:
        on live traffic, waiting for significance means continuing to serve the
        thing we suspect is broken.
        """
        return (
            self.interval.total >= PRECAUTION_MIN_SAMPLES
            and self.interval.point < bar - PRECAUTION_MARGIN
        )

    def supports_advance(self, bar: float) -> bool:
        """Whether this cluster's evidence permits moving more traffic."""
        return (
            self.interval.total > 0
            and not self.interval.underpowered
            and self.interval.clearly_above(bar)
            and not self.judge_only
        )


def read_cluster(cluster: str, name: str, share: float, results: list[ItemResult]) -> Reading:
    """Score a cluster and record what tier its evidence came from.

    Wraps [`score_cluster`][clickllm.prove.equivalence.score_cluster] rather than
    recomputing the interval — one implementation of the arithmetic.
    """
    from clickllm.prove.equivalence import score_cluster

    score = score_cluster(cluster, name, share, results)
    applicable = [s for r in results for s in r.applicable]
    judge_only = bool(applicable) and all(s.tier is Tier.JUDGE for s in applicable)
    return Reading(score=score, judge_only=judge_only)


@dataclass(frozen=True, slots=True)
class Decision:
    """What to do, why, and what it was based on.

    The reason is not decoration. Someone will read it at 3am while deciding
    whether to trust an automatic rollback, and "quality regression detected"
    would not help them. Every reason names the cluster and the number.
    """

    action: Action
    reason: str
    #: Clusters responsible for the action.
    clusters: tuple[str, ...] = ()
    #: Where the migration should move, for `ADVANCE`. `None` otherwise.
    to: Stage | None = None
    #: True when a rollback fired on suspicion rather than proof. Acted on
    #: identically; disclosed differently, because a precautionary rollback is a
    #: judgement call and the human deserves to know one was made.
    precautionary: bool = False

    @property
    def automatic(self) -> bool:
        """Whether this may be applied without a human.

        Only rollback. An advance is a proposal — see the module docstring.
        """
        return self.action is Action.ROLL_BACK

    def render(self) -> str:
        """One line, suitable for a log or a terminal."""
        head = {
            Action.ROLL_BACK: "ROLL BACK",
            Action.ADVANCE: "ADVANCE",
            Action.HOLD: "HOLD",
        }[self.action]
        if self.action is Action.ROLL_BACK and self.precautionary:
            head = "ROLL BACK (precautionary)"
        if self.to is not None:
            head = f"{head} → {self.to.render()}"
        return f"{head}: {self.reason}"


def decide(
    readings: list[Reading],
    stage: Stage,
    health: Health | None = None,
    agreement: Agreement | None = None,
    pinned: frozenset[str] = frozenset(),
    bar: float = DEFAULT_EQUIVALENCE_BAR,
    calibration: Calibration | None = None,
) -> Decision:
    """Decide what the evidence permits.

    Args:
        readings: per-cluster evidence.
        stage: where the migration currently sits.
        health: operational signals. `None` means unmeasured, which blocks an
            advance on live traffic — absence of evidence is not health.
        agreement: judge-vs-human agreement, when a judge was used.
        pinned: clusters already routed to the incumbent. They cannot regress
            what they do not serve, so they are excluded from rollback triggers.
        bar: equivalence threshold.
        calibration: judge-vs-deterministic-grader agreement, from the run
            itself. Required before judge-only evidence may veto; `None` reads
            as uncalibrated, which is what it is.

    Returns:
        A [`Decision`]. `ROLL_BACK` may be applied automatically; `ADVANCE` is a
        proposal for a human.
    """
    # The fourth door to the same number. `Matrix` and `Receipt` guard the bar
    # in their constructors, and `run`/`suite`/`issue` reach it through them —
    # but this takes a bare float and builds neither, so it inherited nothing.
    # It is also the one that matters most: at bar=0.0 a 41/80 cluster returns
    # ADVANCE reading "100% of traffic is proven at or above the 0% bar", which
    # is the rollout surface invariant 8 exists to protect.
    check_bar(bar)
    live = [r for r in readings if r.cluster not in pinned]

    # 1. Health first. A broken candidate is not a quality question, and running
    #    the grading maths on it would only delay the obvious.
    if stage.live and health is not None and (broken := _health_failure(health)):
        return Decision(Action.ROLL_BACK, broken)

    # 2. Proven regression on traffic that is actually being served.
    if stage.live:
        proven = [r for r in live if r.proven_regression(bar)]
        if proven:
            if blocked := _uncalibrated_veto(proven, calibration):
                return Decision(Action.HOLD, blocked, clusters=tuple(r.cluster for r in proven))
            worst = min(proven, key=lambda r: r.interval.point)
            return Decision(
                Action.ROLL_BACK,
                f"{worst.name} is at {worst.interval.render()} against a "
                f"{bar:.0%} bar — the whole interval is below it, on "
                f"{stage.render()} traffic",
                clusters=tuple(r.cluster for r in proven),
            )

        # 3. Suspicion, acted on before proof. See the module docstring on why
        #    this exists and why it is labelled.
        suspect = [r for r in live if r.suspected_regression(bar)]
        if suspect:
            if blocked := _uncalibrated_veto(suspect, calibration):
                return Decision(Action.HOLD, blocked, clusters=tuple(r.cluster for r in suspect))
            worst = min(suspect, key=lambda r: r.interval.point)
            return Decision(
                Action.ROLL_BACK,
                f"{worst.name} is running at {worst.interval.point:.0%} over "
                f"{worst.interval.total} items, more than "
                f"{PRECAUTION_MARGIN:.0%} below the {bar:.0%} bar — not yet "
                f"conclusive, and not worth serving while we find out",
                clusters=tuple(r.cluster for r in suspect),
                precautionary=True,
            )

    # 4. Nothing is wrong. Is anything proven enough to move?
    if not readings:
        return Decision(Action.HOLD, "no evidence gathered yet")
    # `live`, not `readings`. Everything below decides on `live`, so a guard on
    # `readings` passes while the set actually being judged is empty — which is
    # the ordinary outcome once every scored cluster is pinned to the incumbent.
    # `_advance_blocked` then iterates over nothing, every check is vacuously
    # satisfied, and the gate proposed ADVANCE with a self-refuting reason:
    #
    #   "0% of traffic is proven at or above the 90% bar;
    #    shadow -> canary 5% is supported."
    #
    # Reproduced on one pinned cluster scoring 2/40 against a 0.90 bar — a
    # cluster we had already refused to trust was enough to clear the ladder.
    # The gate exists to say no; an empty evidence set is the one input it must
    # never read as yes.
    if not live:
        return Decision(
            Action.HOLD,
            f"every scored cluster is pinned to the incumbent ({len(readings)} "
            f"of {len(readings)}) — nothing is proven that could move traffic",
            clusters=tuple(r.cluster for r in readings),
        )

    nxt = _next_stage(stage)
    if nxt is None:
        return Decision(
            Action.HOLD,
            f"{stage.render()} is the top of the ladder — nothing left to advance to",
        )

    if hold := _advance_blocked(live, stage, health, agreement, bar):
        return Decision(Action.HOLD, hold[0], clusters=hold[1])

    movable = sum(r.share for r in live if r.supports_advance(bar))
    return Decision(
        Action.ADVANCE,
        f"{movable:.0%} of traffic is proven at or above the {bar:.0%} bar; "
        f"{stage.render()} → {nxt.render()} is supported. A human applies this.",
        clusters=tuple(r.cluster for r in live if r.supports_advance(bar)),
        to=nxt,
    )


def _uncalibrated_veto(triggers: list[Reading], calibration: Calibration | None) -> str | None:
    """Why a rollback carried only by an unmeasured judge must not fire.

    Returns a reason when *every* cluster asking for the rollback rests on judge
    scores alone and that judge has not been calibrated against the deterministic
    graders. One deterministic trigger anywhere in the set is enough to act on —
    the judge is then corroborating, not deciding.

    This is the one place where the rollback asymmetry is deliberately not
    followed to its end. A judge that fails work the graders demonstrably passed
    would roll back a healthy migration on its own opinion, repeatedly, and the
    operator would learn to ignore the alarm. The answer is to measure the
    instrument, and the decision says exactly that.
    """
    if any(not r.judge_only for r in triggers):
        return None
    if calibration is not None and calibration.calibrated:
        return None
    state = calibration.render() if calibration is not None else "no calibration was measured"
    names = ", ".join(r.name for r in triggers[:3])
    return (
        f"{names} looks regressed, but every score there is the judge's own — and "
        f"{state}. An unmeasured instrument does not get to stop a migration: "
        f"calibrate the judge, or add a deterministic grader for this cluster"
    )


def _health_failure(h: Health) -> str | None:
    """Describe a health-based rollback trigger, or None if healthy."""
    if not h.measured:
        return None

    rate, base = h.error_rate, h.baseline_error_rate
    if rate is not None:
        # Against the incumbent when we have it, against zero when we do not.
        # Comparing to zero is the conservative reading, and it is stated.
        floor = base if base is not None else 0.0
        if rate > floor + ERROR_RATE_MARGIN:
            against = f"the incumbent's {base:.1%}" if base is not None else "no baseline"
            return (
                f"candidate is failing {rate:.1%} of {h.requests} requests, "
                f"against {against} — beyond the {ERROR_RATE_MARGIN:.0%} margin"
            )

    ratio = h.latency_ratio()
    if ratio is not None and ratio > LATENCY_REGRESSION_RATIO:
        return (
            f"candidate p95 is {h.candidate_p95_ms}ms against the incumbent's "
            f"{h.baseline_p95_ms}ms — {ratio:.1f}× slower, past the "
            f"{LATENCY_REGRESSION_RATIO:.0f}× limit"
        )
    return None


def _advance_blocked(
    live: list[Reading],
    stage: Stage,
    health: Health | None,
    agreement: Agreement | None,
    bar: float,
) -> tuple[str, tuple[str, ...]] | None:
    """Why an advance cannot proceed, or None if it can.

    Every branch returns a specific, actionable sentence. "Not enough evidence"
    without saying how much more would leave the operator with nothing to do.
    """
    # An unmeasured candidate already serving traffic is not "probably fine".
    if stage.live and health is None:
        return ("no health signal for live traffic — cannot confirm it is serving", ())
    if stage.live and health is not None and not health.measured:
        return (
            f"only {health.requests} requests observed; need "
            f"{MIN_HEALTH_SAMPLES} before health means anything",
            (),
        )

    # A judge that disagrees with humans cannot be the reason traffic moved.
    if agreement is not None and not agreement.trustworthy:
        return (
            f"judge agreement is {agreement.render()} — not trustworthy enough to move traffic on",
            (),
        )

    unknown = [r for r in live if r.interval.total == 0]
    if unknown:
        return (
            f"{len(unknown)} cluster(s) have no graded items: "
            f"{', '.join(r.name for r in unknown[:3])}",
            tuple(r.cluster for r in unknown),
        )

    thin = [r for r in live if r.interval.underpowered]
    if thin:
        worst = min(thin, key=lambda r: r.interval.total)
        need = MIN_SAMPLES_FOR_CONFIDENCE - worst.interval.total
        return (
            f"{worst.name} has {worst.interval.total} graded items; "
            f"{need} more needed before its interval means anything",
            tuple(r.cluster for r in thin),
        )

    judged = [r for r in live if r.judge_only and not r.interval.clearly_below(bar)]
    if judged:
        # A judge-only cluster that is *clearly below* the bar is excluded from
        # `judged` — it is a regression, not merely unproven — and returning
        # here meant the later `unproven` check that would have named it never
        # ran. So a straddling cluster hid a worse one behind it, and the
        # Decision named only the milder of the two though both blocked the
        # advance. Every reason names every cluster responsible for it.
        also_regressed = [r for r in live if r.judge_only and r.interval.clearly_below(bar)]
        blocking = judged + also_regressed
        named = ", ".join(r.name for r in blocking[:3])
        return (
            f"{named} rests entirely on judge scores — a judge alone never moves traffic",
            tuple(r.cluster for r in blocking),
        )

    unproven = [r for r in live if not r.supports_advance(bar)]
    if unproven:
        worst = min(unproven, key=lambda r: r.interval.low)
        # How much more evidence, not just "more evidence". An operator told the
        # number can go and get it; an operator told "not yet" can only wait.
        n = worst.score.needed(bar)
        todo = (
            f"{n - worst.interval.total} more graded items would clear it at the "
            f"observed {worst.interval.point:.0%}"
            if n is not None
            else "and the observed rate is at or below the bar, so more items will not clear it"
        )
        return (
            f"{worst.name} is at {worst.interval.render()}; its lower bound has "
            f"not cleared the {bar:.0%} bar — {todo}",
            tuple(r.cluster for r in unproven),
        )
    return None


def demo() -> None:
    """Self-check. Run with `python -m clickllm.prove.gate`."""
    from clickllm.prove.equivalence import ClusterScore
    from clickllm.prove.stats import wilson

    def reading(name: str, passed: int, total: int, *, judge=False, share=1.0):
        return Reading(
            score=ClusterScore(name, name, share, wilson(passed, total), 0),
            judge_only=judge,
        )

    shadow, canary = Stage("shadow", 0), Stage("canary", 25)

    # A clean candidate in shadow proposes the first rung, and only proposes it.
    d = decide([reading("extract", 40, 40)], shadow)
    assert d.action is Action.ADVANCE, d.render()
    assert d.to == Stage("canary", 5), d.to
    assert not d.automatic, "an advance is never applied automatically"

    # The same evidence from a judge alone cannot move anything.
    d = decide([reading("summarise", 40, 40, judge=True)], shadow)
    assert d.action is Action.HOLD and "judge alone" in d.reason, d.render()

    # Thin evidence holds, and says how much more is needed.
    d = decide([reading("rare", 3, 3)], shadow)
    assert d.action is Action.HOLD and "5 more needed" in d.reason, d.render()

    # A proven regression on live traffic rolls back, automatically.
    d = decide([reading("classify", 2, 40)], canary, health=Health(100, 0))
    assert d.action is Action.ROLL_BACK and d.automatic, d.render()
    assert not d.precautionary and "whole interval" in d.reason, d.render()

    # The same regression in shadow is not a rollback: nothing is being served.
    d = decide([reading("classify", 2, 40)], shadow)
    assert d.action is Action.HOLD, d.render()

    # Suspicion acts before proof, and says so. 9/12 straddles the bar (interval
    # 47–91%) so nothing is proven, but a 75% point estimate is not worth serving
    # while we find out.
    d = decide([reading("classify", 9, 12)], canary, health=Health(100, 0))
    assert d.action is Action.ROLL_BACK and d.precautionary, d.render()
    assert "not yet conclusive" in d.reason, d.render()

    # Health overrides quality entirely.
    d = decide(
        [reading("extract", 40, 40)],
        canary,
        health=Health(requests=100, errors=9, baseline_requests=100, baseline_errors=1),
    )
    assert d.action is Action.ROLL_BACK and "failing 9.0%" in d.reason, d.render()

    d = decide(
        [reading("extract", 40, 40)],
        canary,
        health=Health(100, 0, baseline_p95_ms=200, candidate_p95_ms=900),
    )
    assert d.action is Action.ROLL_BACK and "4.5× slower" in d.reason, d.render()

    # An unproven cluster is told how many more items would settle it, not just
    # that it is unproven. 30/32 is 94% with a lower bound of 80%.
    d = decide([reading("extract", 30, 32)], shadow)
    assert d.action is Action.HOLD and "more graded items would clear it" in d.reason, d.render()

    # A judge alone can veto — but only a judge somebody measured.
    from clickllm.prove.judge import Calibration

    bad_news = [reading("summarise", 2, 40, judge=True)]
    d = decide(bad_news, canary, health=Health(100, 0))
    assert d.action is Action.HOLD, d.render()
    assert "unmeasured instrument" in d.reason, d.render()
    d = decide(
        bad_news,
        canary,
        health=Health(100, 0),
        calibration=Calibration(agreed=19, total=20, model="claude-opus-5"),
    )
    assert d.action is Action.ROLL_BACK and d.automatic, d.render()

    # ...and a deterministic trigger in the same set is enough on its own: the
    # judge is corroborating there, not deciding.
    mixed = [reading("summarise", 2, 40, judge=True), reading("classify", 2, 40)]
    assert decide(mixed, canary, health=Health(100, 0)).action is Action.ROLL_BACK

    # The invariant that survives all of it: a judge-only cluster never advances,
    # however well calibrated the judge is.
    d = decide(
        [reading("summarise", 40, 40, judge=True)],
        shadow,
        calibration=Calibration(agreed=40, total=40, model="claude-opus-5"),
    )
    assert d.action is Action.HOLD and "judge alone" in d.reason, d.render()

    # A pinned cluster cannot regress what it does not serve.
    d = decide(
        [reading("extract", 40, 40, share=0.8), reading("classify", 2, 40, share=0.2)],
        canary,
        health=Health(100, 0),
        pinned=frozenset({"classify"}),
    )
    assert d.action is Action.ADVANCE, d.render()

    # An untrustworthy judge blocks the advance rather than being ignored.
    d = decide(
        [reading("extract", 40, 40)],
        shadow,
        agreement=Agreement(agreed=5, total=10, model="j"),
    )
    assert d.action is Action.HOLD and "not trustworthy" in d.reason, d.render()

    # Live traffic with no health signal is not "probably fine".
    d = decide([reading("extract", 40, 40)], canary)
    assert d.action is Action.HOLD and "no health signal" in d.reason, d.render()

    # The top of the ladder has nowhere to go.
    d = decide([reading("extract", 40, 40)], Stage("cut", 100), health=Health(100, 0))
    assert d.action is Action.HOLD and "top of the ladder" in d.reason, d.render()

    assert decide([], shadow).action is Action.HOLD

    print("gate: ok")


if __name__ == "__main__":
    demo()
