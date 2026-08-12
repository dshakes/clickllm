"""The briefing: a receipt as a page, for the person who signs off rather than
the person who ran it.

Two of these matter more than the rest. Cluster names are derived from a
customer's own captured prompts, so a briefing is the one place in this product
where untrusted text becomes a document someone opens in a browser — and the
reader is the person least equipped to notice if it does something. And a page
that reaches the network from a stakeholder's laptop would be a worse leak than
the one this product exists to prevent.
"""

from __future__ import annotations

import dataclasses
import html
import json
import re

import pytest

from clickllm.brief import FORMAT, render
from clickllm.prove import EvalItem, suite
from clickllm.prove.receipt import Receipt

HOSTILE = "<script>alert(1)</script>"


def _receipt(names=None, **kw):
    items = [
        EvalItem(
            item_id=f"{c}-{i}",
            cluster=c,
            prompt=f"{c} {i}",
            baseline="a",
            candidate="a" if c == "good" else f"x{i}",
        )
        for c in ("good", "bad")
        for i in range(100)
    ]
    args = {
        "shares": {"good": 0.6, "bad": 0.4},
        "names": names or {"good": "Refund requests", "bad": "Multi-step tool calls"},
        "issued": "2026-08-12",
        "incumbent_cost": 2847.0,
        "monthly_cost": 317.0,
        "traffic_captures": 400,
        "traffic_window": "14 days",
    }
    args.update(kw)
    return suite(items, **args).receipt


# --- captured traffic is data, never markup (invariant 7) ------------------------


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        '<img src=x onerror="alert(1)">',
        "</pre><script>x</script><pre>",
        '" onmouseover="alert(1)',
        "<style>body{display:none}</style>",
    ],
)
def test_a_hostile_cluster_name_is_shown_as_text_never_as_markup(hostile):
    """A cluster name comes from a customer's own prompts. Invariant 7 says
    captured traffic is data, never instructions — which in a renderer means
    never markup, and the reader here is the least equipped to notice."""
    page = render(_receipt(names={"good": hostile, "bad": "ordinary"}))
    assert hostile not in page, "the hostile name reached the page verbatim"
    assert html.escape(hostile, quote=True) in page, "it was dropped rather than shown"


def test_the_page_contains_no_script_at_all():
    """Not "no untrusted script": none. That is a property a test can check and
    a reader can verify with a text search, which a sanitiser is not."""
    page = render(_receipt(names={"good": HOSTILE, "bad": HOSTILE}))
    assert "<script" not in page.lower()
    assert "javascript:" not in page.lower()
    assert not re.search(r"\son\w+\s*=", page), "an inline event handler appeared"


def test_a_hostile_title_cannot_break_out_either():
    """The title reaches both a <title> element and an attribute-adjacent
    position, which is why escaping here uses quote=True."""
    page = render(_receipt(), title='"><script>alert(1)</script>')
    assert "<script" not in page.lower()


# --- zero egress (NFR-2) ---------------------------------------------------------


def test_the_page_cannot_reach_the_network():
    """A briefing that phones home from a stakeholder's laptop would be a worse
    leak than the one this product exists to prevent."""
    page = render(_receipt())
    for forbidden in ("http://", "https://", "//cdn", "<img", "<iframe", "<link", "url("):
        assert forbidden not in page.lower(), f"{forbidden} appeared in the page"


def test_it_is_one_file_with_nothing_to_fetch():
    page = render(_receipt())
    assert page.startswith("<!doctype html>")
    assert "<style>" in page, "styling must be inline, not a stylesheet request"
    assert FORMAT in page


# --- what it says, and in what order ---------------------------------------------


def test_the_caveats_come_before_the_headline():
    """A reader who stops after the first screen should stop having read the
    caveats. That is the opposite of what a document built to persuade does, and
    the reason this one can be trusted by the person it is shown to."""
    page = render(_receipt())
    assert page.index("Must stay on") < page.index("Safe to move")
    assert page.index("Safe to move") < page.index("What it saves")


def test_the_unproven_group_is_shown_as_unanswered_not_as_failure():
    """Reporting "we did not measure" as 0% is the single worst thing this
    renderer could do — it converts an absence of evidence into evidence of
    failure."""
    thin = [
        EvalItem(item_id=f"u{i}", cluster="unknown", prompt=f"p{i}", baseline="a", candidate="a")
        for i in range(3)
    ]
    items = [
        EvalItem(item_id=f"g{i}", cluster="good", prompt=f"g{i}", baseline="a", candidate="a")
        for i in range(100)
    ]
    r = suite(
        items + thin,
        shares={"good": 0.9, "unknown": 0.1},
        issued="2026-08-12",
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt
    assert r.unproven, "the fixture must contain an unresolved cluster"
    page = render(r)
    assert "Not proven either way" in page
    assert "not failures and not passes" in page


def test_the_move_section_makes_no_claim_when_nothing_is_proven():
    """The heading and body above the table both branch on `receipt.proven`,
    not just the table. An all-regret receipt still said "Safe to move" and
    "every kind of request cleared the bar" — a false positive on the one
    section a stakeholder is most likely to act on."""
    items = [
        EvalItem(item_id=f"{c}-{i}", cluster=c, prompt=f"{c} {i}", baseline="a", candidate=f"x{i}")
        for c in ("good", "bad")
        for i in range(100)
    ]
    r = suite(
        items,
        shares={"good": 0.6, "bad": 0.4},
        names={"good": "Refund requests", "bad": "Multi-step tool calls"},
        issued="2026-08-12",
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt
    assert not r.proven, "the fixture must contain no proven cluster"
    assert r.movable_share == 0
    page = render(r)
    assert "Nothing yet proven safe to move" in page
    # And it says so once, as a finding rather than as a failure. The heading,
    # the headline figure and the body each carried "nothing yet", which reads
    # to a stakeholder like the tool broke rather than like the answer being no.
    assert "That is a result, not an error" in page
    assert page.count("Nothing yet") == 1, "the same non-result is stated more than once"
    assert "Every kind of request below cleared" not in page
    assert "Safe to move to" not in page


def test_judge_trust_is_not_claimed_when_unmeasured():
    """`judge_trustworthy` is `None` until a human-agreement sample exists.
    Collapsing "unmeasured" into "measured and untrustworthy" mislabels a judge
    as NOT trustworthy when nobody has checked — `receipt.py`'s own `render()`
    guards on `judge_agreement` being present first, and this must match it."""
    r = dataclasses.replace(
        _receipt(), judge_model="judge-1", judge_agreement=None, judge_trustworthy=None
    )
    page = render(r)
    assert "NOT trustworthy" not in page
    assert "UNMEASURED" in page


def test_judge_trust_is_claimed_only_once_measured_and_false():
    r = dataclasses.replace(
        _receipt(),
        judge_model="judge-1",
        judge_agreement="Agreement with human review: 82%",
        judge_trustworthy=False,
    )
    page = render(r)
    assert "NOT trustworthy" in page


def test_the_money_is_a_range_with_its_basis():
    page = render(_receipt())
    assert "per month" in page and "–" in page
    assert "Wilson" in page, "the number must carry how it was derived"
    assert "A range, not a single figure" in page


def test_a_refused_saving_is_shown_as_prominently_as_a_figure():
    """A briefing that quietly omits the money when it cannot compute one reads
    as a migration nobody costed."""
    page = render(_receipt(incumbent_cost=None, monthly_cost=None))
    assert "Not stated" in page
    assert "--incumbent-cost" in page, "the refusal must say what would fix it"
    assert "$" not in page.split("<pre>")[0], "a refusal must not print a figure"


def test_it_says_that_nothing_here_authorises_a_cutover():
    """Invariant 8, on the document most likely to be waved at a decision."""
    assert "Shadow mode" in render(_receipt())


def test_the_judge_is_disclosed_either_way():
    """ "No judge was used" is a stronger result, not a missing one — and a
    briefing that simply omits the line leaves a reader unable to tell which."""
    page = render(_receipt())
    assert "No judge model was used" in page and "stronger result" in page


# --- it carries its own proof ----------------------------------------------------


def test_the_embedded_receipt_verifies_independently_of_the_page():
    """The point of embedding it: the reader need not trust this page."""
    r = _receipt()
    page = render(r)
    embedded = html.unescape(page.split("<pre>")[-1].split("</pre>")[0])
    parsed = Receipt.from_json(embedded)  # raises if it does not verify
    assert parsed.digest() == r.digest()
    assert json.loads(embedded)["digest"] == r.digest()
    assert r.digest()[:12] in page, "the digest a reader must compare is not shown"


def test_an_altered_page_is_detectable_from_what_it_embeds():
    """The control for the claim the page makes about itself. If someone edits
    the visible headline, the embedded receipt still says what was measured.
    """
    r = _receipt()
    page = render(r).replace("60% of captured traffic", "100% of captured traffic")
    embedded = html.unescape(page.split("<pre>")[-1].split("</pre>")[0])
    assert Receipt.from_json(embedded).movable_share == pytest.approx(0.6)


# --- accessibility (WCAG 2.2 AA basics) ------------------------------------------


def test_the_page_has_the_structure_a_screen_reader_needs():
    page = render(_receipt())
    assert '<html lang="en">' in page, "no language means the wrong voice reads it aloud"
    assert page.count("<h1") == 1, "exactly one page heading"
    assert "<title>" in page
    assert "<main>" in page


def test_headings_are_in_order_and_none_are_skipped():
    page = render(_receipt())
    levels = [int(m) for m in re.findall(r"<h([1-6])", page)]
    assert levels[0] == 1
    for prev, nxt in zip(levels, levels[1:], strict=False):
        assert nxt <= prev + 1, f"h{prev} jumped to h{nxt}"


def test_tables_are_tables_with_real_headers():
    """A screen reader announces "row: refunds, 98 percent" from a table with
    scoped headers, and reads a wall of numbers from styled divs."""
    page = render(_receipt())
    assert '<th scope="col">' in page and '<th scope="row">' in page
    assert "<caption>" in page


def test_every_state_is_named_in_words_not_only_in_colour():
    """For the reader who cannot distinguish them, and the one printing in
    greyscale."""
    page = render(_receipt())
    for phrase in ("Must stay on", "Safe to move", "did <strong>not</strong> meet the bar"):
        assert phrase in page
    # The classes exist, but never alone.
    assert 'class="bad"' in page or "class='bad'" in page


def test_an_unmeasured_judge_survives_the_disk_ingest_path_too():
    """The same defect, one construction path over.

    The first fix branched on `judge_agreement` being present, mirroring
    `receipt.render()`. Through `issue()` that is sound: both fields come from
    the same `Agreement`, so an agreement line implies a measured boolean. But
    `from_json` is the disk-ingest path, and a receipt on disk can carry an
    agreement string with `judge_trustworthy: null` — and truthiness inside the
    branch called that judge untrustworthy again.

    Fixing where the reviewer pointed rather than where the rule lives is how
    the same defect ships twice.
    """
    from clickllm.prove.receipt import Receipt

    r = Receipt(
        incumbent="a",
        candidate="b",
        issued="2026-08-12",
        eval_set="x" * 64,
        bar=0.9,
        proven=(),
        regret=(),
        unproven=(),
        judge_model="claude-sonnet-5",
        judge_agreement="judge claude-sonnet-5 agreed with humans on 18/20",
        judge_trustworthy=None,
    )
    page = render(r)
    assert "NOT trustworthy" not in page, "unmeasured reported as untrustworthy"
    assert "18/20" in page, "the agreement it does have was dropped"

    # A measured failure still says so, through the same path.
    from dataclasses import replace

    assert "NOT trustworthy" in render(replace(r, judge_trustworthy=False))
