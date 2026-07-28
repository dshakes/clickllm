"""M10 — the guard: noticing when a proof stopped being true.

A migration receipt is a statement about a moment. Three things can happen after
it is issued, and the whole value of this module is that it refuses to treat them
as the same event:

**The model moved** — the thing you proved was replaced behind its name. The
receipt is *void*: not wrong, void. Urgency: now.

**Your traffic moved** — the eval set was drawn from work nobody does anymore.
The receipt answers questions that stopped being asked. Urgency: soon.

**The field moved** — something new was released. Your receipt is still true;
there may just be something better. Urgency: whenever.

Collapsing these into one "stale" flag is the mistake, because only the first two
mean *you do not currently know whether your production model is adequate*, and
the third is a shopping opportunity. A tool that pages you for a new Hugging Face
release with the same urgency as a silent provider-side model swap will be muted
within a week, and then it will not page you for the swap either.

## The one that nobody else can see

Model drift is detectable by anyone who stores a hash. **Traffic drift is not.**
It needs a stable, per-customer notion of "what tasks does this workload consist
of" — which is exactly what M7's clustering produces and what a generic eval
harness has no way to compute. If 40% of your volume is now a task shape your
eval set never sampled, your 94% equivalence figure is a true statement about a
workload you no longer run.

That is the failure mode this module exists for, and it is silent everywhere
else: nothing errors, no score moves, the dashboard stays green.

## What the guard never does

It never moves traffic. Every output is a [`Proposal`], and the strongest verb
available to it is "re-prove". The asymmetry from `prove.gate` holds here too —
and more so, because this runs unattended on a timer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from clickllm.prove.receipt import Receipt

__all__ = [
    "MAX_RECEIPT_AGE_DAYS",
    "TRAFFIC_DRIFT_LIMIT",
    "UNCOVERED_SHARE_LIMIT",
    "Drift",
    "Finding",
    "Proposal",
    "check",
    "demo",
    "traffic_distance",
]

# --- policy knobs --------------------------------------------------------------

#: Total-variation distance between the receipt's cluster mix and current traffic
#: above which the eval set no longer represents the workload. 0.2 means a fifth
#: of volume has moved between task shapes — enough that the weighted score is
#: answering a different question.
TRAFFIC_DRIFT_LIMIT = 0.20

#: Share of current traffic sitting in clusters the receipt never scored. Held
#: tighter than the drift limit: entirely *unmeasured* traffic is worse than
#: measured traffic in changed proportions.
UNCOVERED_SHARE_LIMIT = 0.10

#: Age at which a receipt is re-proved regardless of whether anything visibly
#: moved. Not a correctness threshold — a hedge against the drift we cannot see.
MAX_RECEIPT_AGE_DAYS = 90


class Drift(StrEnum):
    """What moved. Ordered by urgency, most urgent first."""

    #: A model changed behind its name. The receipt's claim is void.
    MODEL_CHANGED = "model_changed"
    #: Current traffic contains task shapes the receipt never scored.
    TRAFFIC_UNCOVERED = "traffic_uncovered"
    #: The mix of task shapes has moved materially.
    TRAFFIC_MOVED = "traffic_moved"
    #: The receipt is simply old.
    AGED = "aged"
    #: A new candidate model exists. The receipt is still true.
    NEW_CANDIDATE = "new_candidate"


#: Which drifts mean "you no longer know whether production is adequate".
#: A set rather than a method, so adding a kind forces a decision about it.
_INVALIDATING = frozenset({Drift.MODEL_CHANGED, Drift.TRAFFIC_UNCOVERED, Drift.TRAFFIC_MOVED})


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing that moved since the receipt was issued."""

    kind: Drift
    #: What moved — a model name, a cluster name, or the receipt itself.
    subject: str
    #: A sentence naming the number that triggered it.
    detail: str

    @property
    def invalidates(self) -> bool:
        """Whether this voids the receipt's claim about production.

        `AGED` and `NEW_CANDIDATE` do not: an old receipt about an unchanged
        model and unchanged traffic is still a true receipt, and saying otherwise
        would train people to ignore the flag that matters.
        """
        return self.kind in _INVALIDATING

    def render(self) -> str:
        """`[void] gpt-5: fingerprint changed …`"""
        mark = "void" if self.invalidates else "note"
        return f"[{mark}] {self.subject}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Proposal:
    """What the guard suggests. Never what it did.

    There is no `apply()`. Re-proving spends money and moving traffic is the
    thing this product exists to make deliberate; an unattended timer gets to
    raise its hand and nothing more.
    """

    findings: tuple[Finding, ...] = ()
    receipt_digest: str = ""

    @property
    def valid(self) -> bool:
        """Whether the receipt still describes production."""
        return not any(f.invalidates for f in self.findings)

    @property
    def urgent(self) -> tuple[Finding, ...]:
        """Findings that void the receipt, most urgent kind first."""
        order = list(Drift)
        return tuple(
            sorted(
                (f for f in self.findings if f.invalidates),
                key=lambda f: order.index(f.kind),
            )
        )

    @property
    def action(self) -> str:
        """The single thing a human should do next."""
        if not self.findings:
            return "nothing — the receipt still holds"
        if any(f.kind is Drift.MODEL_CHANGED for f in self.findings):
            return "re-prove now: the model you proved is not the model in production"
        if any(f.kind is Drift.TRAFFIC_UNCOVERED for f in self.findings):
            return "re-sample and re-prove: real traffic has moved outside the eval set"
        if any(f.kind is Drift.TRAFFIC_MOVED for f in self.findings):
            return "re-sample and re-prove: the eval set no longer matches the workload"
        if any(f.kind is Drift.AGED for f in self.findings):
            return "re-prove when convenient: nothing visibly moved, but the proof is old"
        return "evaluate when convenient: a new candidate is available"

    def render(self) -> str:
        """Terminal-shaped. Voiding findings first; the rest cannot bury them."""
        head = "RECEIPT VOID" if not self.valid else "receipt still holds"
        out = [f"{head} · {self.receipt_digest[:12]}", ""]
        for f in self.urgent:
            out.append(f"  {f.render()}")
        for f in self.findings:
            if not f.invalidates:
                out.append(f"  {f.render()}")
        out += ["", f"→ {self.action}"]
        return "\n".join(out)


def traffic_distance(before: dict[str, float], now: dict[str, float]) -> float:
    """Total variation distance between two cluster mixes, in `0.0..=1.0`.

    Reads directly as "the fraction of volume that moved between task shapes",
    which is the sentence an operator needs. Euclidean or KL would be defensible
    and neither is explainable at 3am.

    Shares are normalised first: two samples of different sizes are still
    comparable as *mixes*, and comparing raw shares that happen not to sum to 1
    would report drift that is only a sampling artefact.
    """
    keys = before.keys() | now.keys()
    a, b = sum(before.values()), sum(now.values())
    if a <= 0 or b <= 0:
        return 0.0
    return 0.5 * sum(abs(before.get(k, 0.0) / a - now.get(k, 0.0) / b) for k in keys)


def check(
    receipt: Receipt,
    *,
    today: date,
    fingerprints: dict[str, str] | None = None,
    traffic: dict[str, float] | None = None,
    available: frozenset[str] = frozenset(),
) -> Proposal:
    """Decide whether `receipt` still describes production.

    Args:
        receipt: the claim being checked.
        today: current date. Passed in, not read from the clock, so a check is
            reproducible and testable.
        fingerprints: current model fingerprints, keyed as in the receipt. A
            model absent here is *not* reported as changed — "we did not look"
            and "it changed" are different claims.
        traffic: current cluster shares, keyed by cluster. `None` skips the
            traffic checks entirely rather than assuming no drift.
        available: candidate models that exist now.

    Returns:
        A [`Proposal`]. Never an action.
    """
    found: list[Finding] = []

    # 1. Did the ground move? The most urgent, because nothing else in the
    #    receipt means anything if the model is not the model.
    for name, was in (receipt.fingerprints or {}).items():
        now = (fingerprints or {}).get(name)
        if now is not None and now != was:
            found.append(
                Finding(
                    Drift.MODEL_CHANGED,
                    name,
                    f"fingerprint {was[:12]} → {now[:12]}; every claim in this "
                    f"receipt was measured against a model that is no longer "
                    f"the one serving",
                )
            )

    if traffic:
        scored = {c.cluster: c.share for c in (*receipt.proven, *receipt.regret, *receipt.unproven)}

        # 2. Traffic the receipt says nothing about at all. Worse than changed
        #    proportions: it is volume running on an unproven basis.
        total = sum(traffic.values()) or 1.0
        uncovered = {k: v / total for k, v in traffic.items() if k not in scored}
        share = sum(uncovered.values())
        if share > UNCOVERED_SHARE_LIMIT:
            worst = ", ".join(k for k, _ in sorted(uncovered.items(), key=lambda kv: -kv[1])[:3])
            found.append(
                Finding(
                    Drift.TRAFFIC_UNCOVERED,
                    "workload",
                    f"{share:.0%} of current traffic is in task shapes this "
                    f"receipt never scored ({worst}) — above the "
                    f"{UNCOVERED_SHARE_LIMIT:.0%} limit",
                )
            )

        # 3. Same shapes, different mix. The weighted score is now answering a
        #    question about a workload that is not the one being run.
        moved = traffic_distance(scored, traffic)
        if moved > TRAFFIC_DRIFT_LIMIT:
            found.append(
                Finding(
                    Drift.TRAFFIC_MOVED,
                    "workload",
                    f"{moved:.0%} of volume has moved between task shapes since "
                    f"{receipt.issued}, past the {TRAFFIC_DRIFT_LIMIT:.0%} limit — "
                    f"the weighted score describes a workload you no longer run",
                )
            )

    # 4. Nothing visibly moved, but absence of evidence is not evidence of
    #    absence, and a proof has a shelf life.
    age = _age_days(receipt.issued, today)
    if age is not None and age > MAX_RECEIPT_AGE_DAYS:
        found.append(
            Finding(
                Drift.AGED,
                "receipt",
                f"issued {age} days ago, past the {MAX_RECEIPT_AGE_DAYS}-day "
                f"limit; nothing visibly moved, which is not the same as nothing "
                f"having moved",
            )
        )

    # 5. Opportunity, not a problem. Reported last and never voiding.
    known = {receipt.candidate, receipt.incumbent}
    for model in sorted(available - known):
        found.append(
            Finding(
                Drift.NEW_CANDIDATE,
                model,
                "released since this receipt; the receipt remains true, there "
                "may simply be something better",
            )
        )

    return Proposal(findings=tuple(found), receipt_digest=receipt.digest())


def _age_days(issued: str, today: date) -> int | None:
    """Days between `issued` and `today`, or None if the date is unparseable.

    An unreadable date is not silently treated as fresh *or* as ancient — it
    produces no finding, and the caller keeps whatever other evidence there is.
    """
    try:
        return (today - date.fromisoformat(issued)).days
    except ValueError:
        return None


def demo() -> None:
    """Self-check. Run with `python -m clickllm.guard`."""
    from clickllm.prove.equivalence import CandidateReport, ClusterScore
    from clickllm.prove.receipt import issue
    from clickllm.prove.stats import wilson

    def cs(name, passed, total, share):
        return ClusterScore(name, name, share, wilson(passed, total), 0)

    r = issue(
        CandidateReport(
            "glm-5.2",
            (cs("extract", 118, 120, 0.6), cs("classify", 96, 100, 0.4)),
        ),
        incumbent="gpt-5",
        issued="2026-07-01",
        eval_set="a" * 64,
        fingerprints={"gpt-5": "f" * 64, "glm-5.2": "e" * 64},
    )
    same_traffic = {"extract": 0.6, "classify": 0.4}
    today = date(2026, 7, 27)

    # Nothing moved: the receipt holds, and the guard says so plainly.
    p = check(r, today=today, fingerprints={"gpt-5": "f" * 64}, traffic=same_traffic)
    assert p.valid and not p.findings, p.render()
    assert p.action.startswith("nothing")

    # The model changed under us. Void, and the most urgent thing on the list.
    p = check(r, today=today, fingerprints={"gpt-5": "9" * 64}, traffic=same_traffic)
    assert not p.valid
    assert p.urgent[0].kind is Drift.MODEL_CHANGED
    assert "re-prove now" in p.action

    # A model we did not look at is not reported as changed.
    p = check(r, today=today, fingerprints={}, traffic=same_traffic)
    assert p.valid, p.render()

    # Traffic moved between known shapes.
    p = check(r, today=today, traffic={"extract": 0.2, "classify": 0.8})
    assert not p.valid and p.urgent[0].kind is Drift.TRAFFIC_MOVED
    assert "40% of volume" in p.urgent[0].detail, p.urgent[0].detail

    # Traffic appeared in a shape we never scored — the failure nobody else sees.
    p = check(r, today=today, traffic={"extract": 0.5, "classify": 0.2, "new-agent-loop": 0.3})
    kinds = {f.kind for f in p.findings}
    assert Drift.TRAFFIC_UNCOVERED in kinds
    assert "new-agent-loop" in p.render()
    assert "re-sample" in p.action

    # Uncovered outranks moved when both fire.
    assert p.urgent[0].kind is Drift.TRAFFIC_UNCOVERED

    # A new release is a note, not a void.
    p = check(r, today=today, traffic=same_traffic, available=frozenset({"qwen-4-next"}))
    assert p.valid, p.render()
    assert p.findings[0].kind is Drift.NEW_CANDIDATE
    assert "evaluate when convenient" in p.action
    # ...and the models already in the receipt are not announced as new.
    p2 = check(r, today=today, traffic=same_traffic, available=frozenset({"glm-5.2", "gpt-5"}))
    assert not p2.findings

    # Age alone does not void a receipt whose model and traffic held.
    old = check(r, today=date(2026, 12, 31), traffic=same_traffic)
    assert old.valid and old.findings[0].kind is Drift.AGED
    assert "re-prove when convenient" in old.action

    # An unparseable date produces no finding rather than a guess.
    weird = issue(
        CandidateReport("glm-5.2", (cs("extract", 118, 120, 1.0),)),
        incumbent="gpt-5",
        issued="whenever",
        eval_set="a" * 64,
    )
    assert not check(weird, today=today).findings

    # Distance is a proportion of volume, and normalises unequal totals.
    assert traffic_distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert traffic_distance({"a": 1.0}, {"b": 1.0}) == 1.0
    assert abs(traffic_distance({"a": 50, "b": 50}, {"a": 0.5, "b": 0.5})) < 1e-9
    assert traffic_distance({}, {"a": 1.0}) == 0.0

    print("guard: ok")


if __name__ == "__main__":
    demo()
