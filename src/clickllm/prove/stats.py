"""Confidence intervals for proportions.

Every equivalence score is a proportion — *k of n items matched* — and every one
is reported with an interval. A bare "96%" from 8 samples and a "96%" from 800
are different claims, and a report that renders them identically is misleading
in the direction that costs someone a production outage.

**Wilson score interval**, not the textbook normal approximation. The normal
approximation is wrong exactly where we live: small clusters, and scores near
100%. At k=n it produces a zero-width interval — "100%, ±0" — which is a
confident lie. Wilson stays sensible at the boundaries and at small n.

No SciPy: the arithmetic is four lines, and a sizing tool that drags in a
numerical stack for one formula has made itself harder to install for nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Two-sided z for a 95% interval. Named so the number is not a mystery constant.
Z_95 = 1.959_963_984_540_054

#: Below this many graded items, a cluster's score is too noisy to act on. It is
#: reported, but flagged — silently dropping it would read as "we covered
#: everything", and a small cluster may be the one that blocks the migration.
MIN_SAMPLES_FOR_CONFIDENCE = 8


@dataclass(frozen=True, slots=True)
class Interval:
    """A proportion with its uncertainty."""

    passed: int
    total: int
    point: float
    low: float
    high: float

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
    if total == 0:
        # No evidence is not zero quality. The interval spans everything, and
        # `render` shows "?" rather than a number nobody should read.
        return Interval(0, 0, 0.0, 0.0, 1.0)

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

    print(f"8/8 = {perfect.render()}   240/250 = {large.render()}")
    print("ok")


if __name__ == "__main__":
    demo()
