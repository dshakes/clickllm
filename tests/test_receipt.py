"""The migration receipt.

`receipt.demo()` covers the round-trip and the partition. These are the
adversarial cases — the ones where someone wants the receipt to say something
it should not.
"""

from __future__ import annotations

import json

import pytest

from clickllm.prove.equivalence import CandidateReport, ClusterScore
from clickllm.prove.graders import EvalItem
from clickllm.prove.judge import Agreement
from clickllm.prove.receipt import (
    FORMAT,
    Claim,
    Receipt,
    eval_set_digest,
    issue,
    verify,
)
from clickllm.prove.stats import wilson


def cs(name: str, passed: int, total: int, share: float = 1.0) -> ClusterScore:
    return ClusterScore(name, name, share, wilson(passed, total), 0)


def receipt(*clusters: ClusterScore, **kw) -> Receipt:
    kw.setdefault("incumbent", "gpt-5")
    kw.setdefault("issued", "2026-07-27")
    kw.setdefault("eval_set", "a" * 64)
    return issue(CandidateReport("glm-5.2", tuple(clusters)), **kw)


# --- tamper resistance ---------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda b: b["proven"][0].__setitem__("passed", 120), id="a-number"),
        pytest.param(lambda b: b["regret"].clear(), id="drop-the-regret-set"),
        pytest.param(lambda b: b["unproven"].clear(), id="drop-the-unknowns"),
        pytest.param(lambda b: b.__setitem__("bar", 0.5), id="lower-the-bar"),
        pytest.param(lambda b: b.__setitem__("issued", "2027-01-01"), id="the-date"),
        pytest.param(lambda b: b.__setitem__("eval_set", "b" * 64), id="swap-the-questions"),
        pytest.param(lambda b: b.__setitem__("judge_trustworthy", True), id="launder-the-judge"),
    ],
)
def test_any_alteration_is_caught_by_the_digest(mutate):
    r = receipt(cs("extract", 118, 120, 0.6), cs("long", 41, 80, 0.3), cs("rare", 3, 3, 0.1))
    blob = json.loads(r.to_json())
    mutate(blob["receipt"])
    with pytest.raises(ValueError, match="altered"):
        Receipt.from_json(json.dumps(blob))


def test_a_receipt_cannot_disagree_with_its_own_digest():
    # The digest is computed, never stored, so there is no field to forge.
    r = receipt(cs("extract", 118, 120))
    assert "digest" not in json.loads(r.to_json())["receipt"]


def test_an_unknown_format_is_refused_rather_than_guessed_at():
    blob = json.loads(receipt(cs("e", 10, 10)).to_json())
    blob["receipt"]["format"] = "clickllm.receipt/v99"
    with pytest.raises(ValueError, match="unknown receipt format"):
        Receipt.from_json(json.dumps(blob))


def test_a_receipt_survives_a_round_trip_through_a_file(tmp_path):
    r = receipt(cs("extract", 118, 120, 0.7), cs("long", 41, 80, 0.3))
    p = tmp_path / "receipt.json"
    p.write_text(r.to_json())
    back = Receipt.from_json(p.read_text())
    assert back == r
    assert back.digest() == r.digest()
    assert back.render() == r.render()


# --- the partition is total ----------------------------------------------------


def test_every_cluster_lands_in_exactly_one_bucket():
    clusters = [
        cs("proven", 118, 120, 0.4),
        cs("regressed", 10, 80, 0.2),
        cs("thin", 3, 3, 0.2),
        cs("ungraded", 0, 0, 0.2),
    ]
    r = receipt(*clusters)
    landed = [c.cluster for c in (*r.proven, *r.regret, *r.unproven)]
    assert sorted(landed) == sorted(c.cluster for c in clusters)
    assert len(landed) == len(set(landed)), "a cluster may not appear twice"


def test_a_cluster_with_nothing_graded_is_unproven_not_dropped():
    r = receipt(cs("silent", 0, 0, 1.0))
    assert [c.cluster for c in r.unproven] == ["silent"]
    assert not r.complete
    assert r.unproven[0].render() == "?"


def test_movable_share_counts_only_proven_clusters():
    r = receipt(cs("good", 118, 120, 0.5), cs("bad", 10, 80, 0.3), cs("thin", 3, 3, 0.2))
    assert r.movable_share == pytest.approx(0.5)


# --- what the reader sees first ------------------------------------------------


def test_bad_news_is_rendered_above_good_news():
    text = receipt(
        cs("good", 118, 120, 0.5), cs("bad", 10, 80, 0.3), cs("thin", 3, 3, 0.2)
    ).render()
    assert text.index("MUST STAY") < text.index("Proven at or above")
    assert text.index("NOT PROVEN") < text.index("Proven at or above")


def test_incomplete_coverage_is_stated_not_left_to_inference():
    assert "incomplete" in receipt(cs("thin", 3, 3)).render()
    assert "incomplete" not in receipt(cs("good", 118, 120)).render()


def test_an_untrustworthy_judge_is_labelled_in_the_rendering():
    weak = receipt(cs("good", 118, 120), agreement=Agreement(20, 40, "j"))
    assert "NOT TRUSTWORTHY" in weak.render()
    strong = receipt(cs("good", 118, 120), agreement=Agreement(38, 40, "j"))
    assert "NOT TRUSTWORTHY" not in strong.render()


def test_no_judge_is_reported_as_a_strength_not_a_gap():
    text = receipt(cs("good", 118, 120)).render()
    assert "none used" in text and "deterministic" in text


def test_the_judge_model_is_not_printed_twice():
    text = receipt(cs("good", 118, 120), agreement=Agreement(38, 40, "claude-sonnet-5")).render()
    assert text.count("claude-sonnet-5") == 1, text


# --- verification --------------------------------------------------------------


def test_the_same_evidence_reproduces_the_same_receipt():
    args = dict(agreement=Agreement(38, 40, "j"), traffic_captures=100)
    a = receipt(cs("e", 118, 120, 0.6), cs("l", 41, 80, 0.4), **args)
    b = receipt(cs("e", 118, 120, 0.6), cs("l", 41, 80, 0.4), **args)
    ok, diffs = verify(a, b)
    assert ok and not diffs


def test_a_swapped_eval_set_is_named_as_the_first_discrepancy():
    # The commonest way a comparison quietly stops being a comparison.
    a = receipt(cs("e", 118, 120))
    b = receipt(cs("e", 118, 120), eval_set="b" * 64)
    ok, diffs = verify(a, b)
    assert not ok
    assert "different set of questions" in diffs[0].render()


def test_a_model_that_changed_behind_its_name_is_detected():
    a = receipt(cs("e", 118, 120), fingerprints={"glm-5.2": "e" * 64})
    b = receipt(cs("e", 70, 120), fingerprints={"glm-5.2": "9" * 64})
    ok, diffs = verify(a, b)
    assert not ok
    rendered = " | ".join(d.render() for d in diffs)
    assert "changed behind its name" in rendered
    assert "98%" in rendered and "58%" in rendered, rendered


def test_a_cluster_that_vanished_between_runs_is_reported():
    a = receipt(cs("e", 118, 120, 0.5), cs("gone", 100, 100, 0.5))
    b = receipt(cs("e", 118, 120, 0.5))
    ok, diffs = verify(a, b)
    assert not ok
    assert any("absent" in d.render() for d in diffs)


def test_differing_digests_never_report_zero_discrepancies():
    # "It does not verify, and I cannot tell you why" is a bug, not an answer.
    #
    # The trigger used to be `tool_version`, which is no longer evidentiary —
    # a rerun from a newer build reproducing the same numbers is a
    # reproduction. `traffic_captures` still is: drawing the eval set from a
    # different amount of traffic is different evidence. It is not itemised in
    # `verify`, so it still reaches the generic fallback this guards.
    a = receipt(cs("e", 118, 120), traffic_captures=12_400)
    b = receipt(cs("e", 118, 120), traffic_captures=900)
    ok, diffs = verify(a, b)
    assert not ok and diffs, "a failed verify must always explain itself"


def test_a_lowered_bar_is_a_discrepancy_not_a_silent_pass():
    a = receipt(cs("e", 95, 100), bar=0.90)
    b = receipt(cs("e", 95, 100), bar=0.50)
    ok, diffs = verify(a, b)
    assert not ok
    assert any("bar" in d.what for d in diffs)


# --- eval set identity ---------------------------------------------------------


def _items(n: int, prompt=lambda i: f"p{i}") -> list[EvalItem]:
    return [EvalItem(f"i{i}", "c", prompt(i), "base", "cand") for i in range(n)]


def test_the_eval_set_digest_ignores_order_but_not_content():
    items = _items(6)
    assert eval_set_digest(items) == eval_set_digest(list(reversed(items)))
    assert eval_set_digest(items) != eval_set_digest(items[:5])
    assert eval_set_digest(items) != eval_set_digest(_items(6, lambda i: f"q{i}"))


def test_the_eval_set_digest_does_not_depend_on_the_answers():
    # Otherwise nobody could run the set against a different model and compare,
    # which is the entire purpose of publishing the digest.
    a = [EvalItem("i", "c", "p", "baseline-a", "candidate-a")]
    b = [EvalItem("i", "c", "p", "baseline-b", "candidate-b")]
    assert eval_set_digest(a) == eval_set_digest(b)


def test_an_empty_eval_set_still_has_a_stable_digest():
    assert eval_set_digest([]) == eval_set_digest([])
    assert len(eval_set_digest([])) == 64


# --- format --------------------------------------------------------------------


def test_the_format_tag_is_present_and_versioned():
    assert receipt(cs("e", 10, 10)).format == FORMAT
    assert FORMAT.endswith("/v1")


def test_a_claim_with_no_items_renders_as_unknown_not_as_zero():
    # Reporting 0% for "we did not measure" is the single worst thing this file
    # could do — it converts an absence of evidence into evidence of failure.
    assert Claim("c", "c", 1.0, 0, 0, 0.0, 1.0).render() == "?"
    assert Claim("c", "c", 1.0, 0, 0, 0.0, 1.0).point is None


@pytest.mark.parametrize(
    "sabotage",
    [
        pytest.param(lambda b: b.pop("digest", None), id="digest-key-deleted"),
        pytest.param(lambda b: b.__setitem__("digest", ""), id="digest-empty-string"),
        pytest.param(lambda b: b.__setitem__("digest", None), id="digest-null"),
    ],
)
def test_a_receipt_without_a_digest_is_refused_not_trusted(sabotage):
    """Removing the digest used to remove the tamper check with it.

    The check was `if (stated := blob.get("digest")) and stated != r.digest()`.
    A falsy digest short-circuits the `and`, so the comparison never ran and the
    receipt parsed clean. Forging a result and deleting one line of JSON was
    enough: `guard` and `box --receipt` both act on the parsed object without
    re-running the eval, so the forged claim would authorise a cutover — the
    thing invariant 8 exists to prevent.

    Fails closed now: no digest means unverifiable, and unverifiable is refused.
    """
    import json

    from clickllm.prove.receipt import Claim, Receipt

    real = Receipt(
        incumbent="gpt-5",
        candidate="qwen3-30b-a3b",
        issued="2026-07-27",
        eval_set="abc123",
        bar=0.90,
        proven=(
            Claim(
                cluster="support",
                name="support",
                share=0.42,
                passed=95,
                total=100,
                low=0.89,
                high=0.98,
            ),
        ),
        regret=(),
        unproven=(),
    )
    blob = json.loads(real.to_json())
    blob["receipt"]["proven"][0]["passed"] = 100  # forge the result upward
    sabotage(blob)

    with pytest.raises(ValueError, match="no digest|does not match"):
        Receipt.from_json(json.dumps(blob))


def test_an_intact_receipt_still_round_trips():
    """The negative control for the above: the refusal must not eat good ones."""
    from clickllm.prove.receipt import Claim, Receipt

    real = Receipt(
        incumbent="gpt-5",
        candidate="qwen3-30b-a3b",
        issued="2026-07-27",
        eval_set="abc123",
        bar=0.90,
        proven=(
            Claim(
                cluster="support",
                name="support",
                share=0.42,
                passed=95,
                total=100,
                low=0.89,
                high=0.98,
            ),
        ),
        regret=(),
        unproven=(),
    )
    assert Receipt.from_json(real.to_json()).proven[0].passed == 95


# --- reproduction ---------------------------------------------------------------


def test_a_rerun_on_a_later_date_still_verifies():
    # The one case this feature exists for, and the one it failed. `digest()`
    # covers `issued` (correctly — editing the date is tampering), and `verify`
    # used to compare full digests, so reproducing identical evidence a month
    # later reported DOES NOT VERIFY with the opaque "content differs outside
    # the compared fields" — indistinguishable from a real regression.
    clusters = (cs("extract", 118, 120, 0.6), cs("long", 41, 80, 0.4))
    ok, diffs = verify(
        receipt(*clusters, issued="2026-07-27"),
        receipt(*clusters, issued="2026-08-24"),
    )
    assert ok, [d.render() for d in diffs]


def test_a_rerun_from_a_newer_build_still_verifies():
    clusters = (cs("extract", 118, 120, 1.0),)
    ok, _ = verify(
        receipt(*clusters, issued="2026-07-27", tool_version="0.9.0"),
        receipt(*clusters, issued="2026-08-24", tool_version="1.0.0"),
    )
    assert ok


def test_the_date_is_still_part_of_the_tamper_digest():
    # Excluding `issued` from the *reproduction* check must not excuse it from
    # the content address. Editing a receipt's date is still tampering.
    clusters = (cs("extract", 118, 120, 1.0),)
    a = receipt(*clusters, issued="2026-07-27")
    b = receipt(*clusters, issued="2026-08-24")
    assert a.digest() != b.digest()


def test_a_real_regression_still_fails_and_is_named():
    # The property the loosening must not cost: same date, different numbers.
    ok, diffs = verify(
        receipt(cs("extract", 118, 120, 1.0), issued="2026-07-27"),
        receipt(cs("extract", 85, 120, 1.0), issued="2026-08-24"),
    )
    assert not ok
    assert any("extract" in d.what for d in diffs), [d.render() for d in diffs]
    assert not any("outside the compared fields" in d.found for d in diffs), (
        "a named discrepancy must not fall through to the generic message"
    )


def test_a_swapped_eval_set_still_fails_even_on_the_same_date():
    ok, diffs = verify(
        receipt(cs("extract", 118, 120, 1.0), eval_set="a" * 64),
        receipt(cs("extract", 118, 120, 1.0), eval_set="b" * 64),
    )
    assert not ok
    assert any(d.what == "eval set" for d in diffs)


def test_a_claim_with_nothing_graded_still_discloses_what_was_merged():
    # `render()` returned a bare "?" before reaching the duplicates check, so a
    # cluster whose every item was duplicate-merged and then ungraded dropped
    # the one disclosure explaining why the denominator is nothing.
    assert Claim("c", "c", 0.1, 0, 0, 0.0, 0.0, duplicates=3).render() == (
        "? (3 duplicate prompt(s) merged)"
    )
    assert Claim("c", "c", 0.1, 0, 0, 0.0, 0.0).render() == "?"
