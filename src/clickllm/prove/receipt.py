"""The migration receipt — a claim you can hand to someone who does not trust you.

Every eval tool in this market produces a **dashboard**: a number, in a web app,
behind a login, true at the moment you looked. That is fine for the engineer who
ran it and useless for everyone downstream — the VP signing off, the auditor
asking what changed, the customer's security review, the person who inherits this
system in a year and needs to know whether the decision was ever justified.

A receipt is the other thing. It is a **file**:

- **Self-contained.** Every number, its confidence interval, the bar it was
  measured against, the judge and how much that judge agreed with humans, the
  clusters that did *not* pass, and the traffic window it came from. Nothing to
  look up, nothing behind a login.
- **Content-addressed.** The eval set is identified by digest, so "which
  questions did you ask" has an exact answer rather than a description.
- **Reproducible.** Re-run the same eval set and you must get the same receipt
  digest. This is a stronger claim than a signature: a signature says *we said
  this*, reproduction says *and it is true*. [`verify`] is the check, and anyone
  with the eval set can run it — including someone who thinks we are lying.
- **Falsifiable later.** It records the exact model fingerprints it was issued
  against. When a provider silently updates a model behind the same name, the
  receipt stops matching, and that is detectable rather than invisible.

## Why the honest version is the more persuasive one

The instinct is to emit `PASSED`. A receipt that says *87% of your traffic is
proven equivalent, 13% is not, and here are the four clusters that are not* is
worth more, because the reader can tell it was not written to reassure them. A
report with no unknowns in it is a report whose unknowns were removed.

So the format has no field that can hold a bare verdict. `unproven` and `regret`
are required, not optional, and rendering shows them above the summary rather
than in a footnote.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from clickllm.prove.equivalence import (
    DEFAULT_EQUIVALENCE_BAR,
    CandidateReport,
    ClusterScore,
)
from clickllm.prove.graders import EvalItem
from clickllm.prove.judge import Agreement

__all__ = [
    "FORMAT",
    "Claim",
    "Discrepancy",
    "Receipt",
    "demo",
    "eval_set_digest",
    "issue",
    "verify",
]

#: Format identifier. A reader that does not recognise this must refuse the file
#: rather than interpret fields it may be guessing at.
FORMAT = "clickllm.receipt/v1"


def _canonical(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no incidental whitespace, UTF-8.

    Two receipts with the same content must produce the same bytes on any
    machine, or the digest means nothing.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()


def eval_set_digest(items: list[EvalItem]) -> str:
    """Content address for the questions that were asked.

    Only the *inputs* are hashed — item id, cluster, prompt. Not the responses:
    the point of an eval set is that someone else can run it against a different
    model and compare, which they cannot do if the digest is bound to our answers.

    Order-independent, because the sampler's ordering is not part of the identity
    of the question set.
    """
    parts = sorted(
        hashlib.sha256(_canonical([i.item_id, i.cluster, i.prompt])).hexdigest() for i in items
    )
    return hashlib.sha256("".join(parts).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class Claim:
    """What was proven about one cluster, with everything needed to doubt it."""

    cluster: str
    name: str
    #: Fraction of captured traffic this cluster represents.
    share: float
    passed: int
    total: int
    low: float
    high: float
    #: Items that could not be graded at all. Disclosed rather than dropped from
    #: the denominator, which is the usual way a pass rate gets flattered.
    ungraded: int = 0

    @property
    def point(self) -> float | None:
        """Equivalence rate, or None when nothing was graded."""
        return None if self.total == 0 else self.passed / self.total

    def render(self) -> str:
        """`94% [91–96] over 120 items`, or `?` when unknown."""
        if self.total == 0:
            return "?"
        return f"{self.point:.0%} [{self.low:.0%}–{self.high:.0%}] over {self.total} items"

    @classmethod
    def of(cls, s: ClusterScore) -> Claim:
        """Build from a scored cluster."""
        return cls(
            cluster=s.cluster,
            name=s.name,
            share=s.share,
            passed=s.interval.passed,
            total=s.interval.total,
            low=s.interval.low,
            high=s.interval.high,
            ungraded=s.ungraded,
        )


@dataclass(frozen=True, slots=True)
class Receipt:
    """A portable, reproducible statement of what was proven.

    `digest` is not stored — it is computed from the content, so a receipt cannot
    disagree with its own hash.
    """

    incumbent: str
    candidate: str
    #: ISO-8601 date. Passed in rather than read from the clock, so issuing is a
    #: pure function and two runs of the same evidence produce the same receipt.
    issued: str
    #: Content address of the questions asked.
    eval_set: str
    #: Equivalence threshold every claim below was measured against.
    bar: float
    #: Clusters at or above the bar.
    proven: tuple[Claim, ...]
    #: Clusters proven *below* it. These are the regret set — they must stay on
    #: the incumbent, and a receipt that omitted them would be marketing.
    regret: tuple[Claim, ...]
    #: Clusters with too little evidence to say either way. Required, because
    #: "we did not check" and "we checked and it passed" are different claims.
    unproven: tuple[Claim, ...]
    #: How many captures the eval set was drawn from, and over what period.
    traffic_captures: int = 0
    traffic_window: str = ""
    #: Judge model and its agreement with humans. `None` means no judge was used,
    #: which is a *stronger* result, not a missing one.
    judge_model: str | None = None
    judge_agreement: str | None = None
    judge_trustworthy: bool | None = None
    #: What redaction removed, by kind. Evidence the eval set is safe to share.
    redacted: dict[str, int] = field(default_factory=dict)
    #: Model fingerprints this was issued against, so a silent provider-side
    #: model change invalidates the receipt instead of quietly outliving it.
    fingerprints: dict[str, str] = field(default_factory=dict)
    tool_version: str = ""
    format: str = FORMAT

    # --- the claim ------------------------------------------------------------

    @property
    def movable_share(self) -> float:
        """Fraction of traffic the evidence supports moving."""
        return sum(c.share for c in self.proven)

    @property
    def complete(self) -> bool:
        """Whether every cluster reached a verdict.

        False is a normal and honest state. It is surfaced so a reader never has
        to infer coverage from the absence of a warning.
        """
        return not self.unproven

    def digest(self) -> str:
        """Content address of this receipt.

        Any alteration — a number, a cluster, the date — changes it.
        """
        return hashlib.sha256(_canonical(asdict(self))).hexdigest()

    # --- portability ----------------------------------------------------------

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise, with the digest alongside for a reader that wants it."""
        body = asdict(self)
        return json.dumps({"receipt": body, "digest": self.digest()}, indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Receipt:
        """Parse, checking the format tag and the digest.

        Raises:
            ValueError: unknown format, or content that does not match its digest.
        """
        blob = json.loads(text)
        body = blob.get("receipt", blob)
        if body.get("format") != FORMAT:
            raise ValueError(f"unknown receipt format {body.get('format')!r}")
        for key in ("proven", "regret", "unproven"):
            body[key] = tuple(Claim(**c) for c in body.get(key, ()))
        r = cls(**body)
        if (stated := blob.get("digest")) and stated != r.digest():
            raise ValueError(
                f"receipt digest {stated[:12]} does not match its content "
                f"{r.digest()[:12]} — it has been altered since it was issued"
            )
        return r

    # --- rendering ------------------------------------------------------------

    def render(self) -> str:
        """Terminal-shaped summary.

        What is *not* proven comes first. A reader who stops after three lines
        should stop having read the caveats, not the headline.
        """
        out = [
            f"Migration receipt · {self.incumbent} → {self.candidate}",
            f"issued {self.issued} · eval set {self.eval_set[:12]} · digest {self.digest()[:12]}",
            "",
        ]
        if self.regret:
            out.append(f"MUST STAY on {self.incumbent} — proven below the bar:")
            out += [f"  · {c.name:<28} {c.render()}" for c in self.regret]
            out.append("")
        if self.unproven:
            out.append("NOT PROVEN either way — not enough evidence:")
            out += [f"  · {c.name:<28} {c.render()}" for c in self.unproven]
            out.append("")
        if self.proven:
            out.append(f"Proven at or above the {self.bar:.0%} bar:")
            out += [f"  · {c.name:<28} {c.render()}" for c in self.proven]
            out.append("")

        out.append(f"Movable: {self.movable_share:.0%} of captured traffic")
        if not self.complete:
            out.append(f"Coverage: incomplete — {len(self.unproven)} cluster(s) unresolved")
        if self.judge_model:
            trust = "" if self.judge_trustworthy else "   NOT TRUSTWORTHY"
            # `Agreement.render()` already names the model; repeating it here
            # produced "Judge: X, agreement judge: X ...".
            out.append(f"{self.judge_agreement}{trust}")
        else:
            out.append("Judge: none used — every claim is from deterministic graders")
        if self.traffic_captures:
            out.append(
                f"Drawn from {self.traffic_captures} captured requests"
                + (f" over {self.traffic_window}" if self.traffic_window else "")
            )
        if self.redacted:
            kinds = ", ".join(f"{k}×{v}" for k, v in sorted(self.redacted.items()))
            out.append(f"Redacted before storage: {kinds}")
        return "\n".join(out)


def issue(
    report: CandidateReport,
    incumbent: str,
    issued: str,
    eval_set: str,
    *,
    bar: float = DEFAULT_EQUIVALENCE_BAR,
    agreement: Agreement | None = None,
    traffic_captures: int = 0,
    traffic_window: str = "",
    redacted: dict[str, int] | None = None,
    fingerprints: dict[str, str] | None = None,
    tool_version: str = "",
) -> Receipt:
    """Turn a scored candidate into a receipt.

    Every cluster lands in exactly one of proven / regret / unproven — the
    partition is total, so a cluster cannot be quietly dropped on its way into
    the document.
    """
    proven, regret, unproven = [], [], []
    for s in report.clusters:
        claim = Claim.of(s)
        band = s.band(bar)
        if band == "equivalent":
            proven.append(claim)
        elif band == "regressed":
            regret.append(claim)
        else:  # "unknown" and "unproven" are both "we cannot say"
            unproven.append(claim)

    return Receipt(
        incumbent=incumbent,
        candidate=report.model,
        issued=issued,
        eval_set=eval_set,
        bar=bar,
        proven=tuple(proven),
        regret=tuple(regret),
        unproven=tuple(unproven),
        traffic_captures=traffic_captures,
        traffic_window=traffic_window,
        judge_model=agreement.model if agreement else None,
        judge_agreement=agreement.render() if agreement else None,
        judge_trustworthy=agreement.trustworthy if agreement else None,
        redacted=dict(redacted or {}),
        fingerprints=dict(fingerprints or {}),
        tool_version=tool_version,
    )


@dataclass(frozen=True, slots=True)
class Discrepancy:
    """One way a re-run disagreed with a receipt."""

    what: str
    stated: str
    found: str

    def render(self) -> str:
        """`extract: stated 94% [91–96], found 71% [64–78]`."""
        return f"{self.what}: stated {self.stated}, found {self.found}"


def verify(receipt: Receipt, rerun: Receipt) -> tuple[bool, tuple[Discrepancy, ...]]:
    """Check a receipt against a fresh run of the same eval set.

    This is the property the whole artifact rests on, and it is checkable by
    someone who does not trust us — which is the point.

    Returns `(True, ())` when the digests match. Otherwise the specific
    disagreements, so "it does not verify" is never the whole answer.
    """
    if receipt.digest() == rerun.digest():
        return True, ()

    d: list[Discrepancy] = []
    if receipt.eval_set != rerun.eval_set:
        d.append(
            Discrepancy(
                "eval set",
                receipt.eval_set[:12],
                rerun.eval_set[:12] + " — a different set of questions was asked",
            )
        )
    if receipt.bar != rerun.bar:
        d.append(Discrepancy("bar", f"{receipt.bar:.0%}", f"{rerun.bar:.0%}"))
    for name, was, now in (
        ("incumbent", receipt.incumbent, rerun.incumbent),
        ("candidate", receipt.candidate, rerun.candidate),
    ):
        if was != now:
            d.append(Discrepancy(name, was, now))

    for key, old in receipt.fingerprints.items():
        new = rerun.fingerprints.get(key)
        if new is not None and new != old:
            d.append(
                Discrepancy(
                    f"fingerprint {key}",
                    old[:12],
                    new[:12] + " — the model changed behind its name",
                )
            )

    stated = {
        c.cluster: c for grp in (receipt.proven, receipt.regret, receipt.unproven) for c in grp
    }
    found = {c.cluster: c for grp in (rerun.proven, rerun.regret, rerun.unproven) for c in grp}
    for cluster in sorted(stated.keys() | found.keys()):
        a, b = stated.get(cluster), found.get(cluster)
        if a is None:
            d.append(Discrepancy(f"cluster {cluster}", "absent", b.render()))
        elif b is None:
            d.append(Discrepancy(f"cluster {a.name}", a.render(), "absent"))
        elif a.render() != b.render():
            d.append(Discrepancy(f"cluster {a.name}", a.render(), b.render()))

    if not d:
        # Digests differ but nothing above explains it — say so rather than
        # returning "verified" on a technicality.
        d.append(
            Discrepancy(
                "receipt digest",
                receipt.digest()[:12],
                rerun.digest()[:12] + " — content differs outside the compared fields",
            )
        )
    return False, tuple(d)


def demo() -> None:
    """Self-check. Run with `python -m clickllm.prove.receipt`."""
    from clickllm.prove.stats import wilson

    def cs(name, passed, total, share):
        return ClusterScore(name, name, share, wilson(passed, total), 0)

    report = CandidateReport(
        model="glm-5.2",
        clusters=(
            cs("extract", 118, 120, 0.52),
            cs("classify", 95, 100, 0.31),
            cs("long-ctx", 41, 80, 0.13),  # regressed
            cs("rare", 3, 3, 0.04),  # too thin to say
        ),
    )
    r = issue(
        report,
        incumbent="gpt-5",
        issued="2026-07-27",
        eval_set="a" * 64,
        agreement=Agreement(agreed=38, total=40, model="claude-sonnet-5"),
        traffic_captures=12_400,
        traffic_window="14 days",
        redacted={"email": 812, "card_number": 3},
        fingerprints={"gpt-5": "f" * 64, "glm-5.2": "e" * 64},
    )

    # The partition is total: nothing may be dropped on the way in.
    assert len(r.proven) + len(r.regret) + len(r.unproven) == len(report.clusters)
    assert [c.name for c in r.proven] == ["extract"]
    assert [c.name for c in r.regret] == ["long-ctx"]
    # `classify` is 95/100 — a fine point estimate whose lower bound (89%) has
    # not cleared the 90% bar. It is unproven, not proven, and the receipt says
    # so. This is exactly the case a dashboard would render green.
    assert [c.name for c in r.unproven] == ["classify", "rare"]
    assert abs(r.movable_share - 0.52) < 1e-9, r.movable_share
    assert not r.complete

    # Round-trips, and the digest survives it.
    back = Receipt.from_json(r.to_json())
    assert back == r and back.digest() == r.digest()

    # Tamper with a single number and the digest refuses it.
    blob = json.loads(r.to_json())
    blob["receipt"]["proven"][0]["passed"] = 120
    try:
        Receipt.from_json(json.dumps(blob))
        raise AssertionError("altered receipt must not parse")
    except ValueError as e:
        assert "altered" in str(e), e

    # Reproduction: same evidence, same receipt.
    ok, diffs = verify(
        r,
        issue(
            report,
            "gpt-5",
            "2026-07-27",
            "a" * 64,
            agreement=Agreement(38, 40, "claude-sonnet-5"),
            traffic_captures=12_400,
            traffic_window="14 days",
            redacted={"email": 812, "card_number": 3},
            fingerprints={"gpt-5": "f" * 64, "glm-5.2": "e" * 64},
        ),
    )
    assert ok and not diffs, diffs

    # A model that changed behind its name is caught and named.
    worse = CandidateReport(model="glm-5.2", clusters=(cs("extract", 70, 120, 0.52),))
    ok, diffs = verify(
        r,
        issue(
            worse,
            "gpt-5",
            "2026-07-27",
            "a" * 64,
            fingerprints={"gpt-5": "f" * 64, "glm-5.2": "9" * 64},
        ),
    )
    assert not ok
    rendered = " | ".join(d.render() for d in diffs)
    assert "changed behind its name" in rendered, rendered
    assert "extract" in rendered, rendered

    # An eval set swap is called out specifically — the commonest way a
    # comparison is quietly not a comparison.
    ok, diffs = verify(r, issue(report, "gpt-5", "2026-07-27", "b" * 64))
    assert not ok and "different set of questions" in diffs[0].render()

    # The unproven and regressed clusters are rendered before the good news.
    text = r.render()
    assert text.index("MUST STAY") < text.index("Proven at or above")
    assert text.index("NOT PROVEN") < text.index("Proven at or above")
    assert "?" in Claim("c", "c", 1.0, 0, 0, 0.0, 1.0).render()

    # Order of the eval items is not part of their identity.
    items = [EvalItem(f"i{i}", "c", f"p{i}", "b", "c") for i in range(5)]
    assert eval_set_digest(items) == eval_set_digest(list(reversed(items)))
    assert eval_set_digest(items) != eval_set_digest(items[:4])

    print("receipt: ok")


if __name__ == "__main__":
    demo()
