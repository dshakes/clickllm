"""The statistics, checked against numbers this repo did not produce.

`stats.demo()` walks the estimators. This file exists for one reason: every value
asserted below comes from a **published worked example or a closed form**, not
from running the implementation and writing down what it said. A test that pins
an implementation to its own output cannot fail, and a suite full of those is
worse than no suite, because it reads like verification.

Sources, per test:

- Newcombe RG (1998), *Statistics in Medicine* 17:873-890 — method 10 (the
  score-interval MOVER) and method 3 (Wilson), both with his worked data.
- Agresti A, *Categorical Data Analysis* — the presidential-approval paired table.
- The exact binomial / sign test, which is arithmetic anyone can redo by hand.
- The standard normal table, for the Bonferroni-adjusted z.
"""

from __future__ import annotations

import math

import pytest

from clickllm.prove.stats import (
    Z_95,
    difference,
    family_wise_z,
    mcnemar,
    samples_needed,
    weighted_posterior,
    wilson,
)

# --- Wilson, against Newcombe's published worked example ------------------------


def test_wilson_matches_the_published_interval_for_fifteen_of_one_hundred_and_fortyeight():
    """Newcombe (1998) method 3 on 15/148 is 0.0624 to 0.1605.

    Reproduced by hand from the formula at z = 1.96:
        denom  = 1 + 3.8416/148            = 1.025957
        centre = (0.101351 + 0.012978)/denom = 0.111436
        margin = (1.96/denom)·sqrt(0.101351·0.898649/148 + 3.8416/(4·148²))
               = 0.049051
    """
    i = wilson(15, 148, z=1.96)
    assert abs(i.low - 0.0624) < 5e-4, i.low
    assert abs(i.high - 0.1605) < 5e-4, i.high


def test_a_perfect_score_has_the_closed_form_lower_bound():
    """At k = n the Wilson lower bound collapses to n/(n+z²) — the algebra:

        centre - margin = [(1 + z²/2n) - z²/2n] / (1 + z²/n) = n/(n + z²)

    This is the number that decides whether a flawless cluster may move traffic,
    so it is pinned to the identity rather than to a printed value.
    """
    for n in (8, 12, 35, 100):
        assert abs(wilson(n, n).low - n / (n + Z_95**2)) < 1e-12


# --- power ---------------------------------------------------------------------


def test_a_ninety_percent_bar_costs_thirtyfive_flawless_items():
    """From the same closed form: n/(n+z²) > 0.9 ⟺ n > 9z² = 34.57 ⟺ n ≥ 35.

    Everyone's intuition here is wrong by a factor of three — twelve clean items
    feels conclusive and is not — which is exactly why the suite prints the number
    instead of "gather more evidence".
    """
    assert samples_needed(12, 12, 0.90) == 35
    assert samples_needed(1, 1, 0.90) == 35, "the projection is of the rate, not the count"
    # And the two sides of the boundary, from `wilson` itself.
    assert wilson(35, 35).clearly_above(0.90)
    assert not wilson(34, 34).clearly_above(0.90)


def test_a_ninetyfive_percent_bar_costs_more_than_a_ninety_percent_one():
    """Same closed form, different bar: n > (bar/(1-bar))·z² = 19·3.8415 = 72.99,
    so 73 flawless items at a 95% bar against 35 at a 90% one."""
    assert samples_needed(20, 20, 0.95) == 73, samples_needed(20, 20, 0.95)
    assert wilson(73, 73).clearly_above(0.95) and not wilson(72, 72).clearly_above(0.95)


def test_a_rate_at_or_below_the_bar_is_never_a_sample_size_problem():
    """More of the same evidence converges on p. Telling someone to label another
    400 items for a rate that is already under the bar wastes their week."""
    assert samples_needed(90, 100, 0.90) is None
    assert samples_needed(85, 100, 0.90) is None
    assert samples_needed(0, 40, 0.90) is None
    assert samples_needed(0, 0, 0.90) is None


def test_the_projection_refuses_to_run_away():
    """A rate a hair above the bar needs a corpus nobody will label. The ceiling
    is disclosed as `None` rather than by printing 90,000."""
    # 90.1% against a 90% bar: the margin has to shrink under a tenth of a point,
    # which takes ~340,000 items. The default ceiling stops well short of that.
    assert samples_needed(9010, 10000, 0.90) is None
    assert samples_needed(9010, 10000, 0.90, cap=500) is None
    # A rate with real headroom still gets an answer, and the answer satisfies the
    # definition: at n the bar is cleared, at n-1 it is not. Checked against
    # `wilson`, which is itself pinned to Newcombe's published interval above.
    n = samples_needed(95, 100, 0.90)
    assert n is not None
    assert wilson(round(0.95 * n), n).clearly_above(0.90)
    assert not wilson(round(0.95 * (n - 1)), n - 1).clearly_above(0.90)


# --- effect size, unpaired -----------------------------------------------------


def test_newcombes_own_worked_example_for_the_difference():
    """Newcombe (1998), 56/70 against 48/80, method 10: (0.0524, 0.3339).

    The paper compares eleven methods on this data; method 10 is the one
    implemented here, and this is the interval it publishes for it.
    """
    d = difference(56, 70, 48, 80, z=1.96)
    assert abs(d.point - 0.2) < 1e-12
    assert abs(d.low - 0.0524) < 5e-5, d.low
    assert abs(d.high - 0.3339) < 5e-5, d.high
    assert d.significant


def test_the_difference_never_leaves_the_possible_range():
    """The Wald interval's failure: at 100% against 100% it produces (0, 0), and
    at the extremes it wanders outside [-1, 1]. Method 10 does neither."""
    for a, b in ((40, 40), (0, 40), (1, 3)):
        d = difference(a, 40, b, 40)
        assert -1.0 <= d.low <= d.high <= 1.0, d
    same = difference(40, 40, 40, 40)
    assert same.point == 0.0 and not same.significant, same.render()


def test_two_arms_at_the_same_rate_are_not_a_finding():
    """98% against 98% and 98% against 100% render identically per-arm; the
    difference is the thing that tells them apart."""
    tie = difference(98, 100, 98, 100)
    gap = difference(98, 100, 100, 100)
    assert not tie.significant
    assert abs(gap.point + 0.02) < 1e-12
    assert "includes 0" in tie.render()


# --- effect size, paired -------------------------------------------------------


def test_mcnemar_matches_agrestis_presidential_approval_table():
    """Agresti's paired 2×2: 86 and 150 discordant, z = -4.17, p < 0.0001.

    Sign convention here is candidate-relative, so the candidate's 86 wins
    against 150 losses reproduces his statistic including its sign.
    """
    m = mcnemar(worse=150, better=86)
    assert m.z is not None and abs(m.z + 4.17) < 0.005, m.z
    assert m.p_value < 0.0001, m.p_value
    assert m.discordant == 236


def test_the_exact_two_sided_p_is_the_sign_test():
    """1 against 9 discordant: 2·(C(10,0) + C(10,1))/2¹⁰ = 22/1024 = 0.021484375.

    Exact rather than chi-square because our discordant counts are usually in
    single digits, where the approximation is the part that is wrong.
    """
    assert mcnemar(worse=1, better=9).p_value == pytest.approx(0.021484375, abs=1e-12)
    # 0 against 5: 2·(1/32) = 0.0625 — still not significant, which is the point
    # of reporting p rather than "the candidate won every disagreement".
    assert mcnemar(worse=0, better=5).p_value == pytest.approx(0.0625, abs=1e-12)


def test_agreement_on_every_item_is_not_evidence_of_a_difference():
    m = mcnemar(0, 0)
    assert m.p_value == 1.0 and m.z is None
    assert "agreed on every item" in m.render()


def test_negative_discordant_counts_are_a_bug_not_a_number():
    with pytest.raises(ValueError, match="non-negative"):
        mcnemar(-1, 4)


# --- the weighted aggregate ----------------------------------------------------


def test_the_weighted_interval_agrees_with_wilson_on_the_one_cluster_case():
    """A single cluster at weight 1 IS a single proportion, which Wilson covers.
    The simulation has to land on it, or it is measuring something else.

    Monte Carlo, so the tolerance is real: 2000 draws puts the percentile within
    about a point. This is the one number here with no published value behind it.
    """
    got, want = weighted_posterior([(wilson(90, 100), 1.0)]), wilson(90, 100)
    assert abs(got.low - want.low) < 0.02, (got.low, want.low)
    assert abs(got.high - want.high) < 0.02, (got.high, want.high)


def test_a_perfect_cluster_does_not_produce_a_zero_width_verdict():
    """The failure that ruled out the plug-in bootstrap: every resample of 45/45
    returns 45 passes, so the headline would print "100% [100–100]" off 45 items.
    """
    agg = weighted_posterior([(wilson(45, 45), 0.6), (wilson(44, 45), 0.4)])
    assert agg.width > 0.02, agg.render()
    assert agg.high <= 1.0 and agg.low > 0.8


def test_the_verdict_reproduces_exactly_because_a_receipt_has_to():
    pairs = [(wilson(90, 100), 0.6), (wilson(20, 25), 0.4)]
    assert weighted_posterior(pairs) == weighted_posterior(pairs)


def test_the_method_travels_with_the_number():
    """Two intervals in one report from two estimators; a reader has to be able
    to tell which is which without reading the source."""
    assert "Wilson" in wilson(9, 10).method
    assert "Jeffreys" in weighted_posterior([(wilson(9, 10), 1.0)]).method
    assert "z=2.394" in wilson(9, 10, z=family_wise_z(3)).method


def test_nothing_graded_is_not_a_score_of_zero():
    empty = weighted_posterior([(wilson(0, 0), 1.0)])
    assert empty.total == 0 and empty.render() == "?"


# --- multiplicity --------------------------------------------------------------


def test_the_family_wise_z_matches_the_normal_table():
    """m=1 is the ordinary 95% z; m=5 spends 0.01 in total, which is the 99% z.
    Both are table values, checkable in any statistics textbook's back cover."""
    assert family_wise_z(1) == pytest.approx(1.959964, abs=1e-6)
    assert family_wise_z(5) == pytest.approx(2.575829, abs=1e-6)
    assert family_wise_z(2) == pytest.approx(2.241403, abs=1e-6)  # z at 0.9875


def test_adjusting_for_a_family_only_ever_widens_an_interval():
    """The correction has to cost something, or it is decoration."""
    plain = wilson(96, 100)
    for m in (2, 5, 12):
        adjusted = wilson(96, 100, z=family_wise_z(m))
        assert adjusted.low < plain.low
        assert adjusted.width > plain.width


def test_a_family_needs_at_least_one_member():
    with pytest.raises(ValueError, match="at least one"):
        family_wise_z(0)
    with pytest.raises(ValueError, match="probability"):
        family_wise_z(3, alpha=1.5)


def test_mcnemar_survives_more_than_1024_discordant_pairs():
    """`2.0**n` is a float and raises OverflowError at n >= 1024.

    Discordant pairs pass 1024 on any large capture, so this crashed on exactly
    the workloads a paired test is for — and it crashed after the run, when the
    evidence had already been paid for. Found by the CI auditor, not by the
    suite, because every existing case used small hand-checked counts.
    """
    for worse, better in ((512, 512), (1000, 1000), (5, 2000), (2000, 5)):
        m = mcnemar(worse, better)
        assert 0.0 <= m.p_value <= 1.0, (worse, better, m.p_value)
        assert m.z is not None and math.isfinite(m.z)

    # A lopsided split at scale is overwhelming evidence; p must not round to 1.
    assert mcnemar(5, 2000).p_value < 1e-6
    # An even split at scale is no evidence at all.
    assert mcnemar(1000, 1000).p_value == pytest.approx(1.0, abs=0.05)
