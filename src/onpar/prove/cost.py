"""Money, computed in one place.

The saving is the number a reader most wants to believe and the one they are
least able to check. Two things follow, and they are the whole of this module.

**Invariant 6 applies to dollars exactly as it applies to scores.** A saving is
a range or it is nothing. The moved share is measured on a sample of captured
traffic, so it carries the sampling uncertainty of that sample — the same
Wilson interval this project already uses for quality — and the money inherits
it. A point estimate here would be a fabricated precision attached to the one
figure that gets pasted into a slide.

**Refuse rather than flatter.** Three things can be missing, and each is a
refusal that names what would fix it rather than a number that papers over it:
no cost rates, no captured traffic to extrapolate from, or a window too short
to speak about a month. Returning ``None`` rather than guessing was already the
rule in ``equivalence.Policy.blended_cost``; this is that rule with its reasons
made explicit and its arithmetic made shared.

The formula lives here because it was about to live in two modules. A fact
duplicated is a fact that gets fixed in one place — the most common defect in
this repo's history — and "what does the migration save" is exactly the fact
you cannot afford two answers to.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .stats import wilson

__all__ = ["MIN_WINDOW_DAYS", "Saving", "blended", "parse_window", "saving"]

#: Below this, a captured window does not support a *monthly* claim. Traffic has
#: a weekly shape — a Tuesday is not a Sunday, and month-end is not mid-month —
#: so extrapolating three days to thirty multiplies whatever part of the week
#: happened to be captured. Seven days is the shortest window that contains one
#: of every weekday, which is the least that can honestly be scaled.
#:
#: This is a calibration knob, not a truth: a workload with no weekly shape
#: could justify less, and a monthly billing cycle could justify more.
MIN_WINDOW_DAYS = 7.0

_WINDOW = re.compile(
    r"(?P<n>\d+(?:\.\d+)?)\s*(?P<unit>day|days|d|week|weeks|w|month|months|mo|hour|hours|h)\b",
    re.IGNORECASE,
)
_DAYS = {
    "h": 1 / 24,
    "hour": 1 / 24,
    "d": 1.0,
    "day": 1.0,
    "w": 7.0,
    "week": 7.0,
    "mo": 30.0,
    "month": 30.0,
}


def parse_window(text: str) -> float | None:
    """Days in a free-text traffic window, or ``None`` if it cannot be read.

    ``traffic_window`` is a human string — "14 days", "3 weeks", "over 36
    hours". Unreadable returns ``None`` rather than a default, because a
    default here would be the tool inventing the very fact the refusal exists
    to protect: how long you actually watched.
    """
    if not text:
        return None
    m = _WINDOW.search(text)
    if not m:
        return None
    unit = m.group("unit").lower().rstrip("s")
    scale = _DAYS.get(unit) or _DAYS.get(unit[:2]) or _DAYS.get(unit[:1])
    return float(m.group("n")) * scale if scale else None


def blended(incumbent: float | None, candidate: float | None, share: float) -> float | None:
    """Monthly cost of running the hybrid, or ``None`` if either rate is unknown.

    The one formula. ``share`` of the traffic costs the candidate's rate and the
    rest still costs the incumbent's — a migration that moves 60% of traffic
    does not stop paying for the other 40%, which is the arithmetic every
    optimistic estimate in this space quietly skips.
    """
    if incumbent is None or candidate is None:
        return None
    return candidate * share + incumbent * (1 - share)


@dataclass(frozen=True)
class Saving:
    """Monthly dollars saved as a range — or a refusal that says what is missing.

    Falsy when refused, so a caller that forgets to check prints nothing rather
    than printing a zero saving as though it were a measured one. That
    distinction is the entire point: "we cannot say" and "it saves nothing" are
    different claims, and only one of them is ever true here by default.
    """

    low: float = 0.0
    point: float = 0.0
    high: float = 0.0
    #: How it was derived, carried beside the number per invariant 6.
    basis: str = ""
    #: Non-empty when there is no saving to state. Names what would fix it.
    refusal: str = ""

    def __bool__(self) -> bool:
        return not self.refusal

    def render(self) -> str:
        """One line, always with its range or its reason."""
        if self.refusal:
            return f"Saving: unknown — {self.refusal}"
        return f"Saving: ${self.low:,.0f}–${self.high:,.0f}/mo ({self.basis})"


def saving(
    incumbent: float | None,
    candidate: float | None,
    moved_share: float,
    *,
    captures: int = 0,
    window: str = "",
) -> Saving:
    """What the migration saves per month, as a range, or a refusal.

    The range comes from the moved share, not from the rates: the rates are
    given, the share is *measured*, and measuring 60% on 40 requests is a
    different claim from measuring 60% on 4,000. A Wilson interval on the share
    at the observed sample size, carried through the same blended-cost formula,
    is what makes the dollar figure inherit that.

    Refusals, in the order they are checked — each names its fix:

    * no rate for either side (nothing to compute from);
    * no captured traffic (a share with no denominator is not measured);
    * a window under :data:`MIN_WINDOW_DAYS`, or one that cannot be read at all.

    ``captures`` is the sample the *share* was measured on. `suite` falls back
    to the eval-set size when the real capture count is not supplied, which is
    conservative here and only here: an eval set is drawn *from* captures, so
    the fallback is never larger than the truth, and a smaller denominator
    widens the interval. It can therefore understate confidence, never overstate
    it — the one direction this number is allowed to be wrong in.
    """
    if incumbent is None or candidate is None:
        return Saving(refusal="no cost rate configured; pass --incumbent-cost and --candidate-cost")
    for name, v in (("incumbent", incumbent), ("candidate", candidate)):
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{name} cost must be a non-negative number of dollars, got {v!r}")
    if captures <= 0:
        return Saving(refusal="no captured traffic to extrapolate from; run `onpar observe` first")

    days = parse_window(window)
    if days is None:
        return Saving(
            refusal=(
                f"the traffic window {window!r} cannot be read, so a monthly figure "
                "would be an extrapolation from an unknown period"
            )
            if window
            else "the traffic window is unrecorded; pass --traffic-window (captures carry "
            "no timestamps, so this cannot be derived)"
        )
    if days < MIN_WINDOW_DAYS:
        return Saving(
            refusal=(
                f"{days:.3g} days of traffic is too short to state a monthly figure; "
                f"capture at least {MIN_WINDOW_DAYS:.0f} days so a full week is represented"
            )
        )

    # The share as it was actually measured: a proportion of the captured
    # requests. Wilson widens on small samples, which is the property that makes
    # a thin run read as uncertain rather than as precise.
    iv = wilson(round(moved_share * captures), captures)

    def at(share: float) -> float:
        b = blended(incumbent, candidate, share)
        assert b is not None  # both rates checked above
        return incumbent - b

    # Ordered, not assumed. Saving rises with share only while the candidate is
    # cheaper; a candidate that costs *more* saves less the more you move, so
    # `at(iv.low)` is then the high end. Assuming the direction rendered the
    # bound backwards ("$-900–$-500") on exactly the input a reader most needs
    # to read correctly: the migration that is not worth doing.
    lo, hi = sorted((at(iv.low), at(iv.high)))
    return Saving(
        low=lo,
        point=at(iv.point),
        high=hi,
        basis=(
            f"{iv.render()} of traffic moved, measured on {captures} captured "
            f"requests over {window}; {iv.method}"
        ),
    )


def demo() -> None:
    """Self-check: a range, and three refusals that each name their fix."""
    assert parse_window("14 days") == 14.0
    assert parse_window("over 3 weeks") == 21.0
    assert parse_window("36 hours") == 1.5
    assert parse_window("recently") is None, "an unreadable window must not get a default"
    assert parse_window("") is None

    # A migration that moves 60% of traffic still pays for the other 40%.
    assert blended(1000.0, 0.0, 0.6) == 400.0
    assert blended(None, 0.0, 0.6) is None

    s = saving(2847.0, 317.0, 0.6, captures=400, window="14 days")
    assert s, s.refusal
    assert s.low < s.point < s.high, (s.low, s.point, s.high)
    assert "moved" in s.basis and "Wilson" in s.basis
    assert "$" in s.render() and "–" in s.render()

    # Invariant 6: the same share on a tenth of the evidence is a wider claim.
    thin = saving(2847.0, 317.0, 0.6, captures=40, window="14 days")
    assert thin.high - thin.low > s.high - s.low, "less evidence must read as less certain"

    # A candidate that costs more than the incumbent saves less the more you
    # move — the bound must still read low-to-high. This was backwards.
    worse = saving(300.0, 900.0, 0.6, captures=400, window="14 days")
    assert worse.low <= worse.point <= worse.high, (worse.low, worse.point, worse.high)
    assert worse.point < 0, "a dearer candidate must not report a saving"

    # Refuse rather than flatter. Each refusal is falsy and names its fix.
    for bad, wants in (
        (dict(incumbent=None, candidate=317.0), "--incumbent-cost"),
        (dict(captures=0), "observe"),
        (dict(window="2 days"), "at least 7 days"),
        (dict(window=""), "unrecorded"),
        (dict(window="lately"), "cannot be read"),
    ):
        kw = dict(incumbent=2847.0, candidate=317.0, captures=400, window="14 days")
        kw.update(bad)
        r = saving(kw.pop("incumbent"), kw.pop("candidate"), 0.6, **kw)
        assert not r, f"{bad} should have been refused"
        assert wants in r.refusal, r.refusal
        assert r.render().startswith("Saving: unknown"), r.render()

    for bad in (-1.0, float("nan")):
        try:
            saving(bad, 317.0, 0.6, captures=400, window="14 days")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{bad} should not be a cost")

    print("cost: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
