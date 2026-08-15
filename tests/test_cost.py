"""The money, and the four ways it is allowed to say nothing.

The saving is the number a reader most wants to believe and is least able to
check, so it gets the same treatment as a quality score: a range or a refusal,
never a bare figure. These tests are mostly about the refusals, because a
fabricated saving is the single most damaging thing this tool could print and
"we cannot say" is a claim it has to be able to make.
"""

from __future__ import annotations

import json

import pytest

from onpar.prove import EvalItem, cost, suite
from onpar.prove.receipt import FORMAT, SUPPORTED_FORMATS, Claim, Receipt
from onpar.prove.stats import wilson


def _items(n: int = 200, differing: int = 0) -> list[EvalItem]:
    return [
        EvalItem(
            item_id=str(i),
            cluster="c",
            prompt=f"p{i}",
            baseline="a",
            candidate="a" if i >= differing else f"x{i}",
        )
        for i in range(n)
    ]


def _saving_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("Saving"))


# --- the range -------------------------------------------------------------------


def test_a_saving_is_a_range_and_carries_how_it_was_derived():
    """Invariant 6 applies to dollars exactly as it applies to scores."""
    s = cost.saving(2847.0, 317.0, 0.6, captures=400, window="14 days")
    assert s, s.refusal
    assert s.low < s.point < s.high
    assert "Wilson" in s.basis and "400 captured" in s.basis
    assert "–" in s.render(), s.render()


def test_less_evidence_reads_as_less_certain():
    """The control for the range being real rather than decorative: the same
    share on a tenth of the evidence must be a visibly weaker claim."""
    wide = cost.saving(2847.0, 317.0, 0.6, captures=40, window="14 days")
    narrow = cost.saving(2847.0, 317.0, 0.6, captures=4000, window="14 days")
    assert wide.high - wide.low > (narrow.high - narrow.low) * 5


def test_a_dearer_candidate_reports_a_loss_the_right_way_round():
    """Saving rises with share only while the candidate is cheaper. Assuming
    the direction printed the bound backwards — "$-900–$-500" — on exactly the
    input a reader most needs to read correctly: the migration not worth doing.
    """
    s = cost.saving(300.0, 900.0, 0.6, captures=400, window="14 days")
    assert s.low <= s.point <= s.high
    assert s.point < 0


def test_the_blended_cost_still_pays_the_incumbent_for_what_did_not_move():
    """The arithmetic every optimistic estimate in this space skips."""
    assert cost.blended(1000.0, 0.0, 0.6) == pytest.approx(400.0)
    assert cost.blended(1000.0, 0.0, 0.0) == pytest.approx(1000.0)
    assert cost.blended(None, 0.0, 0.6) is None


def test_one_formula_not_two():
    """`Policy` and `Receipt` must not be able to disagree about the same
    migration. A fact duplicated is a fact fixed in one place — this repo's most
    common defect — and this is the fact you can least afford two answers to.

    The move has to be **partial**. The first version of this test used a
    fixture that moved 100% of traffic, where every plausible blended-cost
    formula gives the same answer — so a deliberately wrong second formula
    planted in `Policy` left it passing. A test whose control does not fire is
    not a weaker test, it is not a test.
    """
    items = [
        EvalItem(
            item_id=f"{cluster}-{i}",
            cluster=cluster,
            prompt=f"{cluster}-p{i}",
            baseline="a",
            # `good` matches everywhere; `bad` never does, so it stays behind.
            candidate="a" if cluster == "good" else f"x{i}",
        )
        for cluster in ("good", "bad")
        for i in range(100)
    ]
    r = suite(
        items,
        shares={"good": 0.6, "bad": 0.4},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        traffic_captures=400,
        traffic_window="14 days",
    )
    assert 0.0 < r.policy.moved_share < 1.0, (
        f"the fixture must move *some* traffic — at {r.policy.moved_share} "
        "every blended-cost formula agrees and this proves nothing"
    )
    policy_saving = r.policy.monthly_saving
    assert policy_saving is not None
    assert r.receipt.saving.point == pytest.approx(policy_saving)


# --- the refusals ----------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"incumbent": None}, "--incumbent-cost"),
        ({"candidate": None}, "--incumbent-cost"),
        ({"captures": 0}, "observe"),
        ({"window": ""}, "unrecorded"),
        ({"window": "whenever"}, "cannot be read"),
        ({"window": "2 days"}, "at least 7 days"),
        ({"window": "36 hours"}, "at least 7 days"),
    ],
)
def test_it_refuses_rather_than_flatters(kwargs, expect):
    """Each refusal names what would fix it. A refusal that just says no is a
    dead end; this is the difference between a gap and an answer."""
    args = {"incumbent": 2847.0, "candidate": 317.0, "captures": 400, "window": "14 days"}
    args.update(kwargs)
    s = cost.saving(args.pop("incumbent"), args.pop("candidate"), 0.6, **args)
    assert not s, "should have refused"
    assert expect in s.refusal, s.refusal
    assert s.render().startswith("Saving: unknown"), s.render()
    assert "$" not in s.render(), "a refusal must not print a figure"


def test_a_refusal_is_falsy_and_a_saving_is_not():
    """The property every caller relies on. `Saving(low=0.0)` and "we cannot
    say" are different claims, and only the second is ever true by default."""
    assert not cost.saving(None, None, 0.5)
    assert cost.saving(100.0, 10.0, 0.5, captures=400, window="14 days")


def test_a_window_shorter_than_a_week_cannot_be_scaled_to_a_month():
    """Traffic has a weekly shape, so three days scaled to thirty multiplies
    whichever part of the week happened to be captured."""
    assert cost.parse_window("6 days") == 6.0 < cost.MIN_WINDOW_DAYS
    assert not cost.saving(2847.0, 317.0, 0.6, captures=400, window="6 days")
    assert cost.saving(2847.0, 317.0, 0.6, captures=400, window="7 days")


@pytest.mark.parametrize(
    "text,days",
    [("14 days", 14.0), ("3 weeks", 21.0), ("36 hours", 1.5), ("1 month", 30.0), ("2d", 2.0)],
)
def test_window_parsing(text, days):
    assert cost.parse_window(text) == pytest.approx(days)


def test_an_unreadable_window_gets_no_default():
    """A default here would be the tool inventing the very fact the refusal
    exists to protect: how long you actually watched."""
    for text in ("", "recently", "a while", "lots"):
        assert cost.parse_window(text) is None


def test_a_negative_cost_is_refused_not_rendered():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="non-negative"):
            cost.saving(bad, 317.0, 0.6, captures=400, window="14 days")


# --- the receipt -----------------------------------------------------------------


def test_the_receipt_states_the_money_or_says_why_not():
    """It is the artifact a stakeholder receives, and it was the one surface
    that could not state what the migration saves."""
    with_money = suite(
        _items(),
        shares={"c": 1.0},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt
    line = _saving_line(with_money.render())
    assert "$" in line and "–" in line and "Wilson" in line

    without = suite(_items(), shares={"c": 1.0}, issued="2026-08-12").receipt
    assert "unknown" in _saving_line(without.render())
    assert "$" not in _saving_line(without.render())


def test_the_saving_is_derived_not_stored():
    """A stored saving is a number that can disagree with the claims above it
    on the same sealed document."""
    body = json.loads(
        suite(
            _items(),
            shares={"c": 1.0},
            issued="2026-08-12",
            incumbent_cost=2847.0,
            monthly_cost=317.0,
            traffic_captures=400,
            traffic_window="14 days",
        ).receipt.to_json()
    )["receipt"]
    assert "saving" not in body, "the saving must be recomputed, never sealed in"
    assert body["incumbent_cost"] == 2847.0


# --- the compatibility break this nearly shipped ---------------------------------


def _v1_receipt() -> Receipt:
    iv = wilson(100, 100)
    return Receipt(
        incumbent="gpt-4o",
        candidate="qwen3-32b",
        issued="2026-08-12",
        eval_set="a" * 64,
        bar=0.9,
        proven=(
            Claim(
                cluster="c", name="c", share=0.6, passed=100, total=100, low=iv.low, high=iv.high
            ),
        ),
        regret=(),
        unproven=(),
        traffic_captures=400,
        traffic_window="14 days",
        tool_version="1.0.0",
        format="onpar.receipt/v1",
    )


def _v1_receipt_json() -> str:
    """A receipt exactly as 1.0.0 wrote one: v1, and no cost keys in the body.

    Built from the real type and then stripped, rather than hand-written. A
    hand-written fixture is a guess about a file format, and the first version
    of this one was wrong in a way that looked exactly like the bug it was
    meant to catch — a digest mismatch. Deriving it from `Receipt` and removing
    only the keys that did not exist in 1.0.0 makes it faithful by
    construction: the stated digest is the one 1.0.0 would have computed,
    because that is what `digest()` computes for a v1 document.
    """
    from dataclasses import asdict

    r = _v1_receipt()
    body = asdict(r)
    for added_after_v1 in ("incumbent_cost", "candidate_cost"):
        body.pop(added_after_v1)
    return json.dumps({"receipt": body, "digest": r.digest()})


def test_a_receipt_issued_by_1_0_0_still_verifies():
    """The break this nearly shipped, and the reason the format is versioned.

    Adding *any* field to `Receipt` changes `asdict`, which changes the digest,
    which made every receipt 1.0.0 issued report as "altered since it was
    issued". That is worse than a parse error: it is a false accusation of
    forgery, on the artifact whose entire job is to be trusted, and a reader
    acts on it.
    """
    r = Receipt.from_json(_v1_receipt_json())
    assert r.format == "onpar.receipt/v1"
    assert r.incumbent_cost == 0.0
    # And it round-trips without acquiring a new identity.
    assert Receipt.from_json(r.to_json()).digest() == r.digest()


def test_a_v1_receipt_cannot_carry_a_cost_outside_its_own_digest():
    """The hole the version opened, closed. A v1 document is digested over the
    v1 fields, so a v1 receipt carrying a cost would carry the one number that
    matters most in a place nobody could detect being changed.
    """
    r = _v1_receipt()
    with pytest.raises(ValueError, match="cannot carry a cost"):
        Receipt(
            incumbent=r.incumbent,
            candidate=r.candidate,
            issued=r.issued,
            eval_set=r.eval_set,
            bar=r.bar,
            proven=r.proven,
            regret=(),
            unproven=(),
            incumbent_cost=5000.0,
            format="onpar.receipt/v1",
        )


def test_tamper_detection_still_works_on_both_formats():
    """The control. If the version check had been implemented by loosening the
    digest, every test above would still pass and the receipt would no longer
    detect a thing.
    """
    for text in (
        _v1_receipt_json(),
        suite(
            _items(),
            shares={"c": 1.0},
            issued="2026-08-12",
            incumbent_cost=2847.0,
            monthly_cost=317.0,
            traffic_captures=400,
            traffic_window="14 days",
        ).receipt.to_json(),
    ):
        blob = json.loads(text)
        blob["receipt"]["proven"][0]["share"] = 0.99
        with pytest.raises(ValueError, match="altered since it was issued"):
            Receipt.from_json(json.dumps(blob))


def test_the_new_cost_fields_are_inside_the_v2_digest():
    """Specifically: the money is sealed, not merely carried."""
    r = suite(
        _items(),
        shares={"c": 1.0},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt
    assert r.format == FORMAT
    blob = json.loads(r.to_json())
    blob["receipt"]["incumbent_cost"] = 99999.0
    with pytest.raises(ValueError, match="altered since it was issued"):
        Receipt.from_json(json.dumps(blob))


def test_an_unknown_format_is_still_rejected():
    blob = json.loads(_v1_receipt_json())
    blob["receipt"]["format"] = "onpar.receipt/v9"
    with pytest.raises(ValueError, match="unknown receipt format"):
        Receipt.from_json(json.dumps(blob))
    assert "onpar.receipt/v1" in SUPPORTED_FORMATS and FORMAT in SUPPORTED_FORMATS


def test_claim_is_importable_here():
    """Guards the fixture above from drifting out of the real type."""
    assert Claim.__name__ == "Claim"


@pytest.mark.parametrize(
    "name,mutate",
    [
        (
            "downgrade keeping the cost",
            lambda b: b["receipt"].update(format="onpar.receipt/v1"),
        ),
        (
            "downgrade and strip the cost",
            lambda b: b["receipt"].update(
                format="onpar.receipt/v1", incumbent_cost=0.0, candidate_cost=0.0
            ),
        ),
        (
            "drop the cost keys entirely",
            lambda b: [b["receipt"].pop(k) for k in ("incumbent_cost", "candidate_cost")],
        ),
    ],
)
def test_a_v2_receipt_cannot_be_downgraded_to_hide_its_money(name, mutate):
    """The attack the version split invites, and the reason it is checked here
    rather than assumed from the three reviewers' clean verdicts.

    A v1 document is digested over fewer fields. So the obvious move against a
    receipt whose saving you dislike is to relabel it v1 — which either drops
    the money out of the seal, or hides it entirely. All three shapes are
    refused: relabelling trips the construction guard, and stripping changes the
    payload the stated digest was computed over.
    """
    r = suite(
        _items(),
        shares={"c": 1.0},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt
    blob = json.loads(r.to_json())
    mutate(blob)
    with pytest.raises(ValueError):
        Receipt.from_json(json.dumps(blob))
