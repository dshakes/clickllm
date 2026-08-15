"""The receipt, as a page a stakeholder can be sent.

`Receipt.render()` is a terminal summary, which is the right shape for the
person who ran it and the wrong shape for the person who signs off on it. This
renders the same receipt — the same claims, in the same order, with the same
refusals — as one self-contained HTML file that can be attached to an email and
opened offline in six months.

Three things decide the design.

**It renders captured traffic.** A cluster name comes from a customer's own
prompts. A prompt reading ``<img src=x onerror=…>`` becoming a cluster name and
then becoming markup would execute in the browser of the person least equipped
to notice — and invariant 7 says captured traffic is data, never instructions,
which in a renderer means *never markup*. So every value is escaped, and the
page contains **no script at all**. Not "no untrusted script": none. That is a
property a test can check and a reader can verify with a text search, which a
sanitiser is not.

**It must survive being forwarded.** No CDN, no fonts, no images, no network of
any kind (NFR-2) — a briefing that phones home from a stakeholder's laptop would
be a worse leak than the one this product exists to prevent. Everything is
inline, and the receipt's own JSON is embedded so the reader can re-verify the
digest with `onpar receipt` and no trust in this page.

**It leads with what is not proven.** Same order as `Receipt.render()`: the
regret set first, then what is unresolved, then the money. A reader who stops
after the first screen should stop having read the caveats, not the headline —
which is the opposite of what a document built to persuade would do, and the
reason this one can be trusted by the person it is shown to.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - types only
    from .prove.receipt import Claim, Receipt

__all__ = ["FORMAT", "render"]

FORMAT = "onpar.brief/v1"

#: Contrast checked against WCAG 2.2 AA on the background each is used on:
#: ink 16.1:1, muted 5.9:1, bad 5.4:1, good 4.6:1 — all above the 4.5:1 floor for
#: body text. `system-ui` rather than a webfont, because a font request is a
#: network request. Colour is never the only carrier of meaning: every state is
#: also named in words, for the reader who cannot distinguish them and for the
#: one printing in greyscale.
_CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#16161a;--muted:#55555f;--line:#e3e3e0;
--good:#1f6f43;--bad:#a4243b;--warn:#7a5c00}
@media(prefers-color-scheme:dark){:root{--bg:#16161a;--card:#1e1e24;--ink:#f2f2f0;
--muted:#b6b6bd;--line:#33333c;--good:#6cc48f;--bad:#f08c9c;--warn:#e0be5c}}
*{box-sizing:border-box}
body{margin:0;padding:2rem 1rem;background:var(--bg);color:var(--ink);
font:16px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:52rem;margin:0 auto}
h1{font-size:1.5rem;margin:0 0 .25rem}
h2{font-size:1.05rem;margin:2rem 0 .5rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 2rem;font-size:.9rem}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:1rem 1.25rem;margin:0 0 1rem}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
caption{text-align:left;color:var(--muted);font-size:.85rem;padding-bottom:.5rem}
th,td{text-align:left;padding:.45rem .5rem;border-bottom:1px solid var(--line)}
th{font-weight:600}
td.num,th.num{text-align:right}
.tag{font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.headline{font-size:1.75rem;font-weight:650;margin:.25rem 0}
.money{font-size:1.35rem;font-weight:650}
.refusal{color:var(--warn);font-weight:600}
.note{color:var(--muted);font-size:.85rem;margin:.35rem 0 0}
pre{white-space:pre-wrap;word-break:break-all;background:var(--bg);
border:1px solid var(--line);border-radius:6px;padding:.75rem;font-size:.72rem;margin:0}
code{font-size:.85em}
a{color:inherit}
:focus-visible{outline:3px solid var(--warn);outline-offset:2px}
@media print{body{background:#fff}section{break-inside:avoid}}
"""


def _e(value: object) -> str:
    """Escape anything on its way into the page.

    Every caller goes through this, including the ones whose value "cannot"
    contain markup. The cluster names here are derived from a customer's own
    prompts, and the one value someone decides is safe is the one that is not.
    `quote=True` so a value can never break out of an attribute either.
    """
    return html.escape(str(value), quote=True)


def _claims_table(caption: str, claims: tuple[Claim, ...], tone: str) -> str:
    """One group of clusters, as a table with real headers.

    A `<table>` with `<th scope>` rather than styled divs, because a screen
    reader announces "row: refunds, 98 percent, 24 percent of traffic" from the
    first and reads a wall of numbers from the second.
    """
    rows = "".join(
        "<tr>"
        f'<th scope="row">{_e(c.name)}</th>'
        f'<td class="num">{_e(c.render())}</td>'
        f'<td class="num">{c.share * 100:.0f}%</td>'
        "</tr>"
        for c in claims
    )
    return (
        f"<table><caption>{_e(caption)}</caption><thead><tr>"
        '<th scope="col">Kind of request</th>'
        f'<th scope="col" class="num">Match rate <span class="tag {tone}">'
        "with 95% interval</span></th>"
        '<th scope="col" class="num">Share of traffic</th>'
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def render(receipt: Receipt, *, title: str = "Migration briefing") -> str:
    """One self-contained HTML page for this receipt.

    Contains no script, no network reference, and no claim the receipt does not
    already make. The receipt's own JSON is embedded so the reader can verify
    the digest independently of anything this page says about it.
    """
    saving = receipt.saving
    if saving:
        money = (
            f'<p class="money">${saving.low:,.0f} – ${saving.high:,.0f} '
            "<span class='tag'>per month</span></p>"
            f'<p class="note">{_e(saving.basis)}</p>'
        )
    else:
        # A refusal, shown as prominently as a figure would be. A briefing that
        # quietly omits the money when it cannot compute one reads as a
        # migration nobody costed.
        money = (
            '<p class="refusal">Not stated — this cannot be computed from what was measured.</p>'
            f'<p class="note">{_e(saving.refusal)}</p>'
        )

    if receipt.judge_model:
        # NOT trustworthy is a claim about a *measurement* — it must never fire
        # from an unmeasured `judge_trustworthy is None`, only from a measured
        # `False`. `receipt.py`'s own `render()` has the same guard.
        if receipt.judge_agreement:
            # `is False`, not falsiness, even inside this branch. Reaching here
            # means an agreement line exists, and through `issue()` that implies
            # a measured boolean — but `from_json` is the disk-ingest path, and
            # a receipt on disk can carry an agreement string with
            # `judge_trustworthy: null`. Truthiness then calls an unmeasured
            # judge untrustworthy again, one construction path over.
            trust = (
                " This judge is NOT trustworthy on this set."
                if receipt.judge_trustworthy is False
                else ""
            )
            judge = (
                f"Judged by <code>{_e(receipt.judge_model)}</code>. "
                + _e(receipt.judge_agreement)
                + trust
            )
        else:
            judge = (
                f"Judged by <code>{_e(receipt.judge_model)}</code>. "
                "Agreement with human review: UNMEASURED."
            )
    else:
        judge = (
            "No judge model was used — every claim here comes from deterministic "
            "graders. That is a stronger result, not a missing one."
        )

    parts: list[str] = [
        f"<h1>{_e(title)}</h1>",
        f'<p class="sub">{_e(receipt.incumbent)} → {_e(receipt.candidate)} · '
        f"issued {_e(receipt.issued)} · "
        f"receipt <code>{_e(receipt.digest()[:12])}</code></p>",
    ]

    # What is *not* proven comes first, deliberately. See the module docstring.
    if receipt.regret:
        parts.append(
            "<section aria-labelledby='stay'><h2 id='stay' class='bad'>"
            f"Must stay on {_e(receipt.incumbent)}</h2>"
            "<p>These were measured and did <strong>not</strong> meet the bar. "
            "Moving them would be a known quality regression.</p>"
            + _claims_table("Proven below the bar", receipt.regret, "bad")
            + "</section>"
        )
    if receipt.unproven:
        parts.append(
            "<section aria-labelledby='unknown'><h2 id='unknown' class='warn'>"
            "Not proven either way</h2>"
            "<p>Not enough evidence to say. These are not failures and not passes — "
            "they are unanswered, and this is what unanswered looks like.</p>"
            + _claims_table("Insufficient evidence", receipt.unproven, "warn")
            + "</section>"
        )

    # The whole heading and body branch on `receipt.proven`, not just the table
    # below it — "Safe to move" and "every kind of request cleared the bar" are
    # claims about *something* having passed, and are false when nothing has.
    if receipt.proven:
        parts.append(
            "<section aria-labelledby='move'>"
            f"<h2 id='move' class='good'>Safe to move to {_e(receipt.candidate)}</h2>"
            f'<p class="headline">{receipt.movable_share:.0%} of captured traffic</p>'
            f"<p>Every kind of request below cleared the {receipt.bar:.0%} bar across its "
            "whole confidence interval — not just on average.</p>"
            + _claims_table(f"Proven at or above the {receipt.bar:.0%} bar", receipt.proven, "good")
            + "</section>"
        )
    else:
        parts.append(
            "<section aria-labelledby='move'>"
            "<h2 id='move' class='warn'>Nothing yet proven safe to move to "
            f"{_e(receipt.candidate)}</h2>"
            '<p class="headline">0% of captured traffic</p>'
            f"<p>No kind of request has cleared the {receipt.bar:.0%} bar across its whole "
            "confidence interval. <strong>That is a result, not an error</strong> — the "
            "measurement ran, and on this evidence there is nothing to migrate.</p>"
            "</section>"
        )

    parts.append(
        f"<section aria-labelledby='money'><h2 id='money'>What it saves</h2>{money}"
        "<p class='note'>A range, not a single figure: the share of traffic that moves was "
        "measured on a sample, and the money inherits that uncertainty.</p></section>"
    )

    drawn = (
        f"Drawn from {receipt.traffic_captures:,} captured requests"
        + (f" over {receipt.traffic_window}" if receipt.traffic_window else "")
        + "."
        if receipt.traffic_captures
        else "The number of captured requests behind this was not recorded."
    )
    redacted = (
        "Removed before anything was stored: "
        + ", ".join(f"{k} ×{v}" for k, v in sorted(receipt.redacted.items()))
        + "."
        if receipt.redacted
        else ""
    )
    parts.append(
        "<section aria-labelledby='how'><h2 id='how'>How this was measured</h2>"
        f"<p>{_e(drawn)}</p><p>{judge}</p>"
        + (f"<p>{_e(redacted)}</p>" if redacted else "")
        + (
            ""
            if receipt.complete
            else f"<p class='warn'><strong>Coverage is incomplete</strong> — "
            f"{len(receipt.unproven)} kind(s) of request remain unresolved.</p>"
        )
        + "<p class='note'>Nothing here authorises a cutover on its own. Shadow mode does.</p>"
        "</section>"
    )

    # The receipt itself, so the reader need not trust this page. `to_json`
    # rather than a summary: the point is that they can run it back through
    # `onpar receipt` and get the same digest, or find that they cannot.
    parts.append(
        "<section aria-labelledby='verify'><h2 id='verify'>Verify this yourself</h2>"
        "<p>This page makes no claim the receipt below does not. Save it as "
        "<code>receipt.json</code> and run <code>onpar receipt receipt.json</code>; "
        f"the digest must read <code>{_e(receipt.digest()[:12])}</code>. "
        "If it does not, this page has been altered.</p>"
        f"<pre>{_e(receipt.to_json())}</pre></section>"
    )

    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="generator" content="{FORMAT}">'
        f"<title>{_e(title)}: {_e(receipt.incumbent)} to {_e(receipt.candidate)}</title>"
        f"<style>{_CSS}</style></head><body><main>" + "".join(parts) + "</main></body></html>\n"
    )


def demo() -> None:
    """Self-check: no script, nothing escapes as markup, no network reference."""
    from .prove import EvalItem, suite

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
    r = suite(
        items,
        shares={"good": 0.6, "bad": 0.4},
        names={"good": "Refund requests", "bad": "<script>alert(1)</script>"},
        issued="2026-08-12",
        incumbent_cost=2847.0,
        monthly_cost=317.0,
        traffic_captures=400,
        traffic_window="14 days",
    ).receipt

    page = render(r)

    # Invariant 7, in a renderer: captured traffic is data, never markup.
    assert "<script" not in page.lower(), "a cluster name reached the page as markup"
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page, "the hostile name was not shown as text"

    # NFR-2: nothing on this page can reach the network.
    for scheme in ("http://", "https://", "//cdn", "src="):
        assert scheme not in page, f"{scheme} appeared — a briefing must not phone home"

    # The caveats come before the headline.
    assert page.index("Must stay on") < page.index("Safe to move"), "the headline came first"

    # Money as a range, with its basis.
    assert "per month" in page and "Wilson" in page

    # And it carries its own proof.
    assert r.digest()[:12] in page
    assert json.loads(_unescape_pre(page))["digest"] == r.digest()

    print("brief: ok")


def _unescape_pre(page: str) -> str:
    """The embedded receipt JSON, back out of the page. Test support."""
    body = page.split("<pre>")[-1].split("</pre>")[0]
    return html.unescape(body)


if __name__ == "__main__":  # pragma: no cover
    demo()
