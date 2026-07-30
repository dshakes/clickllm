"""Confidence intervals for proportions, and the rest of the statistics.

Every equivalence score is a proportion — *k of n items matched* — and every one
is reported with an interval. A bare "96%" from 8 samples and a "96%" from 800
are different claims, and a report that renders them identically is misleading
in the direction that costs someone a production outage.

**Wilson score interval**, not the textbook normal approximation. The normal
approximation is wrong exactly where we live: small clusters, and scores near
100%. At k=n it produces a zero-width interval — "100%, ±0" — which is a
confident lie. Wilson stays sensible at the boundaries and at small n.

Five more estimators sit on top of it, each answering a question a per-arm pass
rate cannot, and each named in its own output so a reader can tell which number
came from which method:

| function | method | answers |
|---|---|---|
| [`wilson`] | Wilson (1927) score | how good is one arm, and how sure are we |
| [`samples_needed`] | Wilson, inverted for n | how much more evidence would settle it |
| [`difference`] | Newcombe (1998) method 10 | is the gap between two arms real |
| [`mcnemar`] | McNemar (1947), exact binomial | on *paired* items, who wins the disagreements |
| [`weighted_posterior`] | Jeffreys posterior draws | uncertainty on the traffic-weighted total |
| [`family_wise_z`] | Bonferroni | one of twelve clusters passing by luck |

Wilson does not apply to a weighted sum of proportions, which is exactly what the
headline verdict is, so that number is simulated and says so. Reporting a
weighted aggregate inside a Wilson interval would be borrowing a method's
authority for a number it never computed.

No SciPy and no NumPy: every formula here is a few lines of arithmetic, and a
sizing tool that drags in a numerical stack for them has made itself harder to
install for nothing. The one non-obvious import is
:class:`statistics.NormalDist`, stdlib since 3.8, for the inverse normal CDF.

References:
    Wilson EB (1927). "Probable inference, the law of succession, and statistical
        inference." *JASA* 22:209-212.
    Newcombe RG (1998). "Interval estimation for the difference between
        independent proportions: comparison of eleven methods."
        *Statistics in Medicine* 17:873-890.
    McNemar Q (1947). "Note on the sampling error of the difference between
        correlated proportions or percentages." *Psychometrika* 12:153-157.
    Brown LD, Cai TT, DasGupta A (2001). "Interval estimation for a binomial
        proportion." *Statistical Science* 16:101-133 — for the Jeffreys interval
        and why the plug-in bootstrap is the wrong tool at p̂ = 0 or 1.
    Efron B, Tibshirani R (1993). *An Introduction to the Bootstrap*, ch. 13.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist

# Two-sided z for a 95% interval. Named so the number is not a mystery constant.
Z_95 = 1.959_963_984_540_054

#: Below this many graded items, a cluster's score is too noisy to act on. It is
#: reported, but flagged — silently dropping it would read as "we covered
#: everything", and a small cluster may be the one that blocks the migration.
MIN_SAMPLES_FOR_CONFIDENCE = 8

#: Ceiling on [`samples_needed`]. Past this, "gather more evidence" has stopped
#: being advice: an observed rate this close to the bar is not going to clear it
#: on any eval set a team will actually label.
MAX_PROJECTED_SAMPLES = 100_000

#: Draws behind the weighted aggregate's interval. 2000 is the usual floor for a
#: percentile interval (Efron & Tibshirani ch. 13); the run costs milliseconds at
#: our sample sizes, so there is nothing to buy by going lower.
POSTERIOR_DRAWS = 2000

#: Fixed seed. The receipt must reproduce bit-for-bit on a re-run, and a simulation
#: seeded from the clock would break that for a third decimal place nobody reads.
POSTERIOR_SEED = 20260728


@dataclass(frozen=True, slots=True)
class Interval:
    """A proportion with its uncertainty.

    ``method`` names the estimator that produced ``low`` and ``high``. It is a
    field rather than a comment because two intervals in the same report can come
    from different methods, and a reader has to be able to tell which is which.
    """

    passed: int
    total: int
    point: float
    low: float
    high: float
    method: str = "Wilson score, 95%"

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def underpowered(self) -> bool:
        """True when there are too few samples to draw a conclusion from."""
        return self.total < MIN_SAMPLES_FOR_CONFIDENCE

    def render(self, places: int = 0) -> str:
        """Human form. Always shows the interval — a point estimate alone is the
        thing this module exists to prevent."""
        if self.total == 0:
            return "?"
        p = f"{self.point * 100:.{places}f}"
        lo = f"{self.low * 100:.{places}f}"
        hi = f"{self.high * 100:.{places}f}"
        flag = " ⚠" if self.underpowered else ""
        return f"{p}% [{lo}–{hi}]{flag}"

    def clearly_above(self, threshold: float) -> bool:
        """True only when the *whole* interval clears the bar.

        This is the gate a rollout should use. Comparing a point estimate to a
        threshold advances canaries on noise.
        """
        return self.low > threshold

    def clearly_below(self, threshold: float) -> bool:
        """True only when the whole interval sits under the bar."""
        return self.high < threshold


def wilson(passed: int, total: int, z: float = Z_95) -> Interval:
    """Wilson score interval for ``passed`` of ``total``.

    Raises:
        ValueError: if the counts are impossible. A negative or over-count is a
            bug upstream, and coercing it would hide the bug in a plausible number.
    """
    if total < 0 or passed < 0:
        raise ValueError(f"counts must be non-negative, got {passed}/{total}")
    if passed > total:
        raise ValueError(f"passed {passed} exceeds total {total}")
    named = "Wilson score, 95%" if z == Z_95 else f"Wilson score, z={z:.4g}"
    if total == 0:
        # No evidence is not zero quality. The interval spans everything, and
        # `render` shows "?" rather than a number nobody should read.
        return Interval(0, 0, 0.0, 0.0, 1.0, named)

    p = passed / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return Interval(
        passed=passed,
        total=total,
        point=p,
        low=max(0.0, center - margin),
        high=min(1.0, center + margin),
        method=named,
    )


def pooled(intervals: list[Interval]) -> Interval:
    """Combine intervals by pooling their raw counts.

    Counts, not averaged percentages: averaging a 100%-of-2 with a 50%-of-200
    would report 75% for something that is really 50.5%.
    """
    return wilson(sum(i.passed for i in intervals), sum(i.total for i in intervals))


def weighted_point(pairs: list[tuple[Interval, float]]) -> float | None:
    """Traffic-weighted point estimate across clusters.

    A cluster that is 38% of traffic must count five times one at 8%. Returns
    ``None`` when nothing was graded, rather than 0.0 — no evidence is not a
    score of zero.
    """
    usable = [(i, w) for i, w in pairs if i.total > 0 and w > 0]
    if not usable:
        return None
    total_w = sum(w for _, w in usable)
    if total_w <= 0:
        return None
    return sum(i.point * w for i, w in usable) / total_w


# --------------------------------------------------------------------------- #
# Power — turning "not proven" into an instruction
# --------------------------------------------------------------------------- #


def samples_needed(
    passed: int,
    total: int,
    bar: float,
    z: float = Z_95,
    cap: int = MAX_PROJECTED_SAMPLES,
) -> int | None:
    """How many graded items it would take for this rate to clear ``bar``.

    Wilson inverted for n, holding the *observed* rate fixed: the smallest ``n``
    whose lower bound would sit above the bar if the candidate keeps scoring
    exactly as it has so far. Not a promise — the rate can drop — which is why it
    is phrased as "at the observed rate" everywhere it is printed.

    "Not yet proven, gather more evidence" is a dead end; "not yet proven, 11 more
    items would settle it" is a task. That difference is the whole point of this
    function.

    The closed form at a perfect score is worth knowing, because it is the case
    that traps people: at ``k = n`` the Wilson lower bound is exactly
    ``n / (n + z²)``, so clearing a 90% bar needs ``n > 9z² = 34.57`` — **35
    flawless items**, and no fewer, however clean the run looks at 12.

    Args:
        passed: items that matched.
        total: items graded. Zero means no rate has been observed at all.
        bar: the equivalence threshold to clear.
        cap: refuse to project past this many items.

    Returns:
        The item count, or ``None`` when the observed rate is at or below the bar
        (no sample size rescues it — the candidate has to get better), or when it
        would take more than ``cap`` items.
    """
    if total <= 0 or not 0.0 < bar < 1.0:
        return None
    p = passed / total
    if p <= bar:
        # More of the same evidence converges on p, which is already under the
        # bar. Reporting a number here would send someone to label 400 items for
        # a result that is already decided.
        return None
    # ponytail: linear scan. The answer is ~35 in the common case and the loop
    # only gets long when p sits a hair above the bar, where `cap` stops it.
    # Bisection would be the upgrade, and is not worth it at these sizes.
    for n in range(max(total, 1), cap + 1):
        if wilson(round(p * n), n, z).low > bar:
            return n
    return None


# --------------------------------------------------------------------------- #
# Effect size — two arms, and the gap between them
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Difference:
    """The gap between two independent proportions, with an interval on the gap.

    Per-arm rates conflate two different claims: 98% against 100% is a real
    regression, 98% against 98% is a tie, and both render as "98%" if only one
    arm is shown. The interval here is on the *difference*, which is the quantity
    the decision actually turns on.
    """

    a: Interval
    b: Interval
    point: float
    low: float
    high: float
    method: str = "Newcombe (1998) method 10, score-interval MOVER"

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero — a difference we can stand behind."""
        return not (self.low <= 0.0 <= self.high)

    def render(self, places: int = 1) -> str:
        """`+2.0 pts [-1.5–+5.6] (Newcombe method 10)`, or `?` with no data."""
        if self.a.total == 0 or self.b.total == 0:
            return "?"
        p = f"{self.point * 100:+.{places}f}"
        lo = f"{self.low * 100:+.{places}f}"
        hi = f"{self.high * 100:+.{places}f}"
        verdict = "" if self.significant else ", includes 0"
        return f"{p} pts [{lo}–{hi}]{verdict}"


def difference(
    a_passed: int,
    a_total: int,
    b_passed: int,
    b_total: int,
    z: float = Z_95,
) -> Difference:
    """Interval for ``p_a - p_b``, two *independent* samples.

    Newcombe's method 10 (the score-interval / MOVER hybrid), which he recommends
    over the Wald interval for the same reason we use Wilson per arm: the Wald
    interval overshoots [-1, 1] and collapses to zero width at the boundaries,
    which is where model comparisons live.

    With ``l1,u1`` and ``l2,u2`` the Wilson roots of each arm and ``d = p1 - p2``::

        L = d - sqrt((p1 - l1)² + (u2 - p2)²)
        U = d + sqrt((u1 - p1)² + (p2 - l2)²)

    Worked example from the paper (56/70 against 48/80, z = 1.96) gives
    (0.0524, 0.3339); that is the value pinned in the tests.

    For *paired* data — the same eval items scored by both arms, which is the
    usual case here — this interval is conservative, because it throws away the
    pairing. Use :func:`mcnemar` for the paired view.
    """
    a, b = wilson(a_passed, a_total, z), wilson(b_passed, b_total, z)
    d = a.point - b.point
    if a_total == 0 or b_total == 0:
        # No overlap to compare. The full range, and `render` shows "?".
        return Difference(a, b, 0.0, -1.0, 1.0)
    low = d - math.sqrt((a.point - a.low) ** 2 + (b.high - b.point) ** 2)
    high = d + math.sqrt((a.high - a.point) ** 2 + (b.point - b.low) ** 2)
    return Difference(a, b, d, max(-1.0, low), min(1.0, high))


@dataclass(frozen=True, slots=True)
class McNemar:
    """The paired view: only the items where the two arms disagreed.

    Items both arms got right, and items both got wrong, carry no information
    about which is better — McNemar's insight, and the reason a paired comparison
    is far more powerful than two independent rates over the same eval set. What
    is left is the discordant pairs, and the question is whether they split evenly.
    """

    #: Items the candidate failed and the reference passed. The candidate's losses.
    worse: int
    #: Items the candidate passed and the reference failed. Its wins.
    better: int
    #: Normal approximation ``(better - worse) / sqrt(better + worse)``. ``None``
    #: when there are no discordant pairs at all.
    z: float | None
    #: Two-sided *exact* binomial probability of a split this lopsided under the
    #: null that the arms are equal. Exact rather than chi-square because our
    #: discordant counts are routinely in single digits, where the approximation
    #: is the thing that is wrong.
    p_value: float
    method: str = "McNemar (1947), exact two-sided binomial on discordant pairs"

    @property
    def discordant(self) -> int:
        return self.worse + self.better

    def render(self) -> str:
        """`8 wins / 2 losses of 10 discordant, p=0.109 (exact McNemar)`."""
        if self.discordant == 0:
            return "no discordant pairs — the two arms agreed on every item"
        return (
            f"{self.better} wins / {self.worse} losses of {self.discordant} "
            f"discordant, p={self.p_value:.3f}"
        )


def mcnemar(worse: int, better: int) -> McNemar:
    """McNemar's test on the discordant pairs of a paired comparison.

    Args:
        worse: items the candidate failed where the reference passed.
        better: items the candidate passed where the reference failed.

    The two-sided p-value is the exact binomial: ``2 * P(X <= min(b, c))`` for
    ``X ~ Binomial(b + c, 0.5)``, capped at 1. Agresti's presidential-approval
    example (86 and 150 discordant) gives ``z = -4.17``; a 1-against-9 split gives
    an exact ``p = 0.0215``. Both are pinned in the tests.

    Raises:
        ValueError: on negative counts, which are a bug upstream.
    """
    if worse < 0 or better < 0:
        raise ValueError(f"discordant counts must be non-negative, got {worse}/{better}")
    n = worse + better
    if n == 0:
        # Perfect agreement is not evidence of a difference in either direction.
        return McNemar(worse, better, None, 1.0)
    z = (better - worse) / math.sqrt(n)
    tail = sum(math.comb(n, i) for i in range(min(worse, better) + 1))
    # `2.0**n` is a FLOAT and raises OverflowError at n >= 1024. Discordant pairs
    # pass 1024 on any large capture, so that was a hard crash on exactly the
    # workloads this test exists for. `1 << n` is an arbitrary-precision int, and
    # int/int division underflows to 0.0 rather than raising — a p-value that
    # small IS zero to any precision that matters here.
    return McNemar(worse, better, z, min(1.0, (2 * tail) / (1 << n)))


# --------------------------------------------------------------------------- #
# The weighted aggregate, and the family of tests it sits in
# --------------------------------------------------------------------------- #


def weighted_posterior(
    pairs: list[tuple[Interval, float]],
    *,
    draws: int = POSTERIOR_DRAWS,
    seed: int = POSTERIOR_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Simulated interval for the traffic-weighted score.

    The headline verdict is ``Σ wₖ pₖ / Σ wₖ`` — a weighted sum of proportions
    from different-sized clusters. Wilson covers a single proportion and says
    nothing about that sum, so rendering it inside a Wilson interval would borrow
    a method's authority for a number it never computed. Each cluster's rate is
    drawn instead from its Jeffreys posterior ``Beta(k + ½, n - k + ½)``, the
    weighted score is recomputed per draw, and the 2.5th and 97.5th percentiles
    of the resulting distribution are the interval.

    **Why not the plug-in bootstrap.** Resampling items within a cluster —
    ``Binomial(n, p̂)`` — is the more familiar recipe and it collapses exactly
    where this tool lives: at ``p̂ = 1`` every resample returns n passes, so a
    45/45 cluster contributes zero width and the headline prints as "75%
    [75–75]". That is the same confident lie the normal approximation tells at
    the boundary, and it is the reason this module rejected Wald for Wilson in
    the first place. The Jeffreys posterior is the boundary-respecting analogue
    (Brown, Cai & DasGupta 2001), and it agrees with Wilson to about a point in
    the middle of the range.

    Deterministic by construction: the seed is a module constant, because a
    receipt that reproduces bit-for-bit is worth more than a fresh random draw.

    Returns:
        An :class:`Interval` whose ``method`` names the simulation, with pooled
        counts carried in ``passed``/``total`` for context. ``total == 0`` when
        nothing was graded, which renders as "?".
    """
    usable = [(i, w) for i, w in pairs if i.total > 0 and w > 0]
    named = f"Jeffreys posterior simulation, {draws} draws, seed {seed}"
    if not usable or draws < 1:
        return Interval(0, 0, 0.0, 0.0, 1.0, named)

    point = weighted_point(usable) or 0.0
    total_w = sum(w for _, w in usable)
    rng = random.Random(seed)
    sampled = sorted(
        sum(rng.betavariate(iv.passed + 0.5, iv.total - iv.passed + 0.5) * w for iv, w in usable)
        / total_w
        for _ in range(draws)
    )
    lo = sampled[max(0, math.floor(alpha / 2 * draws))]
    hi = sampled[min(draws - 1, math.ceil((1 - alpha / 2) * draws) - 1)]
    return Interval(
        passed=sum(i.passed for i, _ in usable),
        total=sum(i.total for i, _ in usable),
        point=point,
        low=lo,
        high=hi,
        method=named,
    )


def family_wise_z(comparisons: int, alpha: float = 0.05) -> float:
    """The z a *family* of tests needs, so one cluster does not pass by luck.

    Scoring twelve clusters against the same bar at 95% each gives roughly a 46%
    chance that at least one clears it on noise alone. Bonferroni is the blunt,
    honest correction: spend ``alpha / m`` on each test, so the family-wise error
    stays under ``alpha``.

    Bonferroni over Šidák on purpose — it is the more conservative of the two and
    the more widely recognised, and the difference between them at m ≤ 20 is in
    the third decimal place.

    ``family_wise_z(1)`` is 1.95996 (the ordinary 95% z) and ``family_wise_z(5)``
    is 2.57583 (the 99% z), both checkable against any published normal table.

    Raises:
        ValueError: if ``comparisons`` is below 1 or ``alpha`` is not a probability.
    """
    if comparisons < 1:
        raise ValueError(f"a family needs at least one comparison, got {comparisons}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be a probability, got {alpha}")
    return NormalDist().inv_cdf(1.0 - alpha / (2.0 * comparisons))


def demo() -> None:
    # The failure the normal approximation makes: a perfect score at small n
    # is not certainty.
    perfect = wilson(8, 8)
    assert perfect.point == 1.0
    assert perfect.low < 0.75, f"8/8 must not claim near-certainty, got {perfect.low:.3f}"
    assert perfect.high == 1.0

    # More evidence, same proportion, tighter interval.
    small, large = wilson(24, 25), wilson(240, 250)
    assert abs(small.point - large.point) < 1e-9
    assert large.width < small.width / 2, "more samples must narrow the interval"

    # Zero passes is not certainty of failure either.
    none_passed = wilson(0, 10)
    assert none_passed.point == 0.0 and none_passed.high > 0.0

    # No evidence renders as unknown, never as a number.
    empty = wilson(0, 0)
    assert empty.render() == "?"
    assert empty.low == 0.0 and empty.high == 1.0

    # Gates use the whole interval, not the point.
    assert not wilson(9, 10).clearly_above(0.95), (
        "90% point, wide interval — must not pass a 95% gate"
    )
    assert wilson(990, 1000).clearly_above(0.95), "99% of 1000 should clear it"
    assert wilson(1, 100).clearly_below(0.5)

    # Underpowered clusters are flagged, not hidden.
    assert wilson(3, 4).underpowered
    assert not wilson(50, 60).underpowered
    assert "⚠" in wilson(3, 4).render()

    # Pooling uses counts; averaging percentages would report 75% here.
    p = pooled([wilson(2, 2), wilson(100, 200)])
    assert abs(p.point - 102 / 202) < 1e-9, p.point

    # Weighting follows traffic share.
    big, small_c = wilson(50, 100), wilson(100, 100)
    assert abs(weighted_point([(big, 0.9), (small_c, 0.1)]) - 0.55) < 1e-9
    assert weighted_point([]) is None
    assert weighted_point([(wilson(0, 0), 1.0)]) is None

    for bad in [(-1, 5), (5, 3), (1, -1)]:
        try:
            wilson(*bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{bad} should have raised")

    # Power: "not proven" becomes a number of items. At k=n the Wilson lower
    # bound is n/(n+z²), so a 90% bar needs n > 9z² = 34.57 → 35, by hand.
    assert samples_needed(12, 12, 0.90) == 35, samples_needed(12, 12, 0.90)
    assert wilson(35, 35).low > 0.90 and wilson(34, 34).low <= 0.90
    # A rate at or below the bar is not a sample-size problem.
    assert samples_needed(85, 100, 0.90) is None
    assert samples_needed(0, 0, 0.90) is None
    # A rate above the bar but only just needs more evidence than we will project.
    assert samples_needed(9010, 10000, 0.90, cap=500) is None

    # Effect size: Newcombe (1998) method 10, his own worked example.
    d = difference(56, 70, 48, 80, z=1.96)
    assert abs(d.low - 0.0524) < 5e-5 and abs(d.high - 0.3339) < 5e-5, d.render()
    assert d.significant, "the published interval excludes zero"
    # Two arms at the same rate is a tie, and the interval says so.
    assert not difference(98, 100, 98, 100).significant
    assert difference(0, 0, 5, 5).render() == "?"

    # Paired: McNemar on the discordant pairs only.
    m = mcnemar(worse=1, better=9)  # exact two-sided = 2*(1+10)/2^10
    assert abs(m.p_value - 0.021484375) < 1e-12, m.p_value
    assert mcnemar(0, 0).p_value == 1.0 and mcnemar(0, 0).z is None
    assert "agreed on every item" in mcnemar(0, 0).render()

    # The weighted aggregate is simulated, and names which simulation rather than
    # borrowing Wilson's name for a number Wilson did not compute.
    agg = weighted_posterior([(wilson(90, 100), 0.6), (wilson(45, 50), 0.4)])
    assert "Jeffreys" in agg.method
    assert agg.low < agg.point < agg.high
    assert weighted_posterior([(wilson(9, 10), 1.0)]) == weighted_posterior(
        [(wilson(9, 10), 1.0)]
    ), "a receipt has to reproduce, so the seed is fixed"

    # A single cluster is a case Wilson *does* cover, so the two must agree —
    # if they do not, the simulation is wrong. ~2 points of Monte Carlo slack.
    one, ref = weighted_posterior([(wilson(90, 100), 1.0)]), wilson(90, 100)
    assert abs(one.low - ref.low) < 0.02 and abs(one.high - ref.high) < 0.02, one

    # The boundary that rules out the plug-in bootstrap: every resample of a
    # 45/45 cluster returns 45 passes, so a resampled interval reads [100–100].
    assert weighted_posterior([(wilson(45, 45), 1.0)]).low < 0.97

    # Multiplicity: the published z for 95% and, at five clusters, for 99%.
    assert abs(family_wise_z(1) - 1.959964) < 1e-6
    assert abs(family_wise_z(5) - 2.575829) < 1e-6

    print(f"8/8 = {perfect.render()}   240/250 = {large.render()}")
    print(f"12/12 clears a 90% bar at n={samples_needed(12, 12, 0.90)}")
    print(f"56/70 vs 48/80 = {d.render()}")
    print("ok")


if __name__ == "__main__":
    demo()
