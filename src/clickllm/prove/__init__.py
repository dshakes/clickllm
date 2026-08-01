"""The eval suite — one call from captured traffic to a verdict you can act on.

Everything else under ``prove/`` is machinery: graders score an item, the judge
breaks ties, ``stats`` puts an interval round it, ``equivalence`` assembles the
matrix, ``receipt`` makes it portable. This module is the **kiosk over that
machinery** — the single entry point the CLI, the SDK, the MCP server and the
agent skill all route through, so an agent gets the same verdict a human does
because it is the same code path.

## Why the suite is shaped this way

**The eval set is derived, not written.** Every other tool starts from a test
set you author, which is why nobody has one: writing it is the work you were
trying to avoid. Here the items come from your own captured traffic via
:mod:`clickllm.distill`, so the thing being measured is the thing you actually
run.

**The judge is the last resort, not the first.** Deterministic graders run
first and an item they *disqualify* never reaches the judge — paying a model to
re-confirm that malformed JSON is malformed is spend with no information in it.
The judge is asked only where structure passed and semantics are still open.
That ordering is also why a judge outage degrades the run to "fewer graded
items" rather than "no result".

**Nothing here moves traffic.** ``run`` returns a :class:`~.equivalence.Matrix`
and ``suite`` adds a :class:`~.receipt.Receipt`. Neither has a side effect, and
no function in this module can advance a rollout — that lives behind
:mod:`clickllm.prove.gate` and terminates at a human. An agent is welcome to
run the whole suite; it structurally cannot be the thing that cuts over.

**The model callables are injected.** The suite never reaches out on your
behalf. You hand it items that already carry both answers, and a ``judge`` if
you want one. That is what keeps the package dependency-free and what keeps
captured prompts on the machine that captured them.

Getting those answers off a live endpoint is therefore a separate, explicit
step: :mod:`clickllm.prove.collect`, or ``clickllm prove --candidate-endpoint``.
It is not re-exported here, because ``run`` and ``suite`` making no network call
is a property worth being able to read off this module.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from .equivalence import (
    DEFAULT_EQUIVALENCE_BAR,
    CandidateReport,
    ClusterScore,
    HybridPolicy,
    Matrix,
    score_cluster,
)
from .gate import Action, Decision, Health, Reading, Stage, decide, read_cluster
from .graders import DEFAULT_GRADERS, EvalItem, Grader, ItemResult, Outcome, Score, Tier, grade
from .judge import (
    Agreement,
    Calibration,
    Comparison,
    JudgeFn,
    JudgeResult,
    Reply,
    Verdict,
    calibrate,
    judge_item,
)
from .receipt import Claim, Discrepancy, Receipt, eval_set_digest, issue, verify
from .stats import (
    Difference,
    Interval,
    McNemar,
    difference,
    family_wise_z,
    mcnemar,
    pooled,
    samples_needed,
    weighted_point,
    weighted_posterior,
    wilson,
)

__all__ = [
    "DEFAULT_EQUIVALENCE_BAR",
    "Action",
    "Agreement",
    "Calibration",
    "CandidateReport",
    "Claim",
    "ClusterScore",
    "Comparison",
    "Decision",
    "Difference",
    "Discrepancy",
    "EvalItem",
    "Grader",
    "Health",
    "HybridPolicy",
    "Interval",
    "ItemResult",
    "JudgeFn",
    "JudgeResult",
    "Matrix",
    "McNemar",
    "Outcome",
    "Reading",
    "Receipt",
    "Reply",
    "Score",
    "Stage",
    "SuiteResult",
    "Tier",
    "Verdict",
    "calibrate",
    "decide",
    "difference",
    "eval_set_digest",
    "family_wise_z",
    "grade",
    "issue",
    "judge_item",
    "mcnemar",
    "pooled",
    "read_cluster",
    "run",
    "samples_needed",
    "score_cluster",
    "shares_from_clusters",
    "suite",
    "verify",
    "weighted_point",
    "weighted_posterior",
    "wilson",
]


def shares_from_clusters(clusters: list) -> tuple[dict[str, float], dict[str, str]]:
    """Traffic shares and display names from :mod:`clickllm.distill` clusters.

    Shares come from observed cluster sizes rather than being asked for, because
    a share the user typed is a guess and a share measured from their own
    captures is not. Returns ``({key: share}, {key: name})``.

    Raises:
        ValueError: if the clusters hold no captures — a zero denominator would
            silently produce shares of zero and a report that looks measured.
    """
    total = sum(c.size for c in clusters)
    if total <= 0:
        raise ValueError("cannot derive traffic shares from clusters holding 0 captures")
    return (
        {c.key: c.size / total for c in clusters},
        {c.key: c.name for c in clusters},
    )


def _judged(
    result: ItemResult,
    item: EvalItem,
    judge: JudgeFn,
    judge_model: str,
) -> ItemResult:
    """Add a judge verdict, but only where it can change the answer.

    A structural failure is already disqualifying and conjunctive scoring means
    a judge PASS could not rescue it, so the call is skipped: same verdict, one
    fewer inference. Items with no applicable grader *do* go to the judge —
    that is precisely the blind spot it exists to cover.
    """
    if result.graded and not result.passed:
        return result
    verdict = judge_item(item, judge, judge_model)
    return replace(result, scores=result.scores + (verdict.to_score(),))


def _merge(a: ItemResult, b: ItemResult) -> ItemResult:
    """Fold a repeated prompt's second result into the first.

    Conjunctive: the surviving observation carries both copies' scores, so
    `passed` (which already requires every applicable grader to pass) is true
    only when both copies passed. Keeps `a`'s item_id — the report points at a
    real item someone can go read, and naming one of two arbitrary ids is more
    useful than inventing a synthetic one that matches nothing in the file.
    """
    return ItemResult(item_id=a.item_id, cluster=a.cluster, scores=a.scores + b.scores)


def run(
    items: list[EvalItem],
    *,
    shares: dict[str, float],
    names: dict[str, str] | None = None,
    candidate: str = "candidate",
    incumbent: str = "incumbent",
    graders: tuple[Grader, ...] = DEFAULT_GRADERS,
    judge: JudgeFn | None = None,
    judge_model: str = "",
    agreement: Agreement | None = None,
    bar: float = DEFAULT_EQUIVALENCE_BAR,
    monthly_cost: float | None = None,
    incumbent_cost: float | None = None,
) -> Matrix:
    """Score one candidate against the incumbent over an eval set.

    Args:
        items: the eval set, each carrying both answers. From
            :mod:`clickllm.distill` in the product flow.
        shares: cluster key -> fraction of traffic. Used to weight the verdict,
            so a cluster's influence matches how often it actually occurs.
        names: cluster key -> display name. Falls back to the key.
        judge: optional. Without one the run is deterministic-only, and the
            report says so rather than implying semantic checking happened.
        judge_model: the judge's identifier. Required when ``judge`` is given —
            an anonymous judge makes every score it touched unauditable.
        agreement: how often this judge matched a human. Absent means UNMEASURED,
            and the matrix presents judge-derived cells as unknown rather than
            as evidence.

    Returns:
        A :class:`~.equivalence.Matrix`. Clusters present in ``items`` but absent
        from ``shares`` are scored at share 0 — they appear in the report and
        cannot move traffic, which beats dropping them silently.

    Raises:
        ValueError: if ``items`` is empty, or ``judge`` is given without
            ``judge_model``.
    """
    if not items:
        raise ValueError("cannot prove anything from an empty eval set")
    if judge is not None and not judge_model.strip():
        raise ValueError("judge_model is required when a judge is supplied, for disclosure")

    by_cluster: dict[str, list[ItemResult]] = defaultdict(list)
    seen: dict[tuple[str, str], int] = {}
    dupes: dict[str, int] = defaultdict(int)
    for item in items:
        result = grade(item, graders)
        if judge is not None:
            result = _judged(result, item, judge, judge_model)
        # An identical prompt inside one cluster is the same question asked
        # twice, not two independent observations, and every interval below
        # assumes independence. Left in, the copies shrink the interval on
        # evidence nobody collected — and they move the point estimate too: a
        # real eval set here had 40 classification items over 28 distinct
        # prompts and read 95%, where the 28 read 90% and did not clear the bar.
        #
        # Collapsed conjunctively, matching how a single item is already scored
        # across graders: the observation passes only if every copy passed. A
        # prompt the model answers correctly only sometimes is not one it can be
        # trusted on, and this gate exists to move production traffic.
        key = (item.cluster, item.prompt)
        if key in seen:
            dupes[item.cluster] += 1
            prior = by_cluster[item.cluster][seen[key]]
            by_cluster[item.cluster][seen[key]] = _merge(prior, result)
            continue
        seen[key] = len(by_cluster[item.cluster])
        by_cluster[item.cluster].append(result)

    names = names or {}
    scores = [
        score_cluster(
            key, names.get(key, key), shares.get(key, 0.0), results, duplicates=dupes.get(key, 0)
        )
        for key, results in sorted(by_cluster.items())
    ]
    report = CandidateReport(model=candidate, clusters=tuple(scores), monthly_cost=monthly_cost)
    return Matrix(
        candidates=[report],
        agreement=agreement,
        incumbent=incumbent,
        incumbent_cost=incumbent_cost,
        bar=bar,
        # Free, and the only calibration most runs will ever have: the items the
        # judge and the deterministic graders both scored. Measured here rather
        # than asked for, because a number the user typed is a claim.
        calibration=(
            calibrate([r for rs in by_cluster.values() for r in rs], judge_model)
            if judge is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """A full run: the comparison, the policy it implies, and the proof."""

    matrix: Matrix
    receipt: Receipt
    policy: HybridPolicy

    def render(self) -> str:
        """The kiosk view — regret first, then the matrix, then the proposal.

        Deliberately one string: the report is the product, and a caller that
        has to assemble it from parts will assemble it differently each time.
        """
        return f"{self.matrix.render()}\n\n{self.policy.render()}"


def suite(
    items: list[EvalItem],
    *,
    shares: dict[str, float],
    issued: str,
    names: dict[str, str] | None = None,
    candidate: str = "candidate",
    incumbent: str = "incumbent",
    judge: JudgeFn | None = None,
    judge_model: str = "",
    agreement: Agreement | None = None,
    bar: float = DEFAULT_EQUIVALENCE_BAR,
    monthly_cost: float | None = None,
    incumbent_cost: float | None = None,
    traffic_captures: int = 0,
    traffic_window: str = "",
    redacted: dict[str, int] | None = None,
    fingerprints: dict[str, str] | None = None,
    tool_version: str = "",
) -> SuiteResult:
    """The whole kiosk: score, propose a split, and issue the receipt.

    ``issued`` is passed in rather than read from the clock so a run is
    reproducible — re-running the same eval set must reproduce the same digest,
    and a timestamp baked in here would break that for no benefit.

    Raises:
        ValueError: from :func:`run`, or if nothing could be scored at all.
    """
    matrix = run(
        items,
        shares=shares,
        names=names,
        candidate=candidate,
        incumbent=incumbent,
        judge=judge,
        judge_model=judge_model,
        agreement=agreement,
        bar=bar,
        monthly_cost=monthly_cost,
        incumbent_cost=incumbent_cost,
    )
    best = matrix.best()
    if best is None:
        raise ValueError(
            "no cluster could be scored — every item was ungraded. "
            "Supply a judge, or check that the candidate produced output at all."
        )
    return SuiteResult(
        matrix=matrix,
        receipt=issue(
            best,
            incumbent=incumbent,
            issued=issued,
            eval_set=eval_set_digest(items),
            bar=bar,
            agreement=agreement,
            calibration=matrix.calibration,
            traffic_captures=traffic_captures or len(items),
            traffic_window=traffic_window,
            redacted=redacted,
            fingerprints=fingerprints,
            tool_version=tool_version,
        ),
        policy=matrix.hybrid_for(best),
    )


def demo() -> None:
    def item(n: int, cluster: str, base: str, cand: str) -> EvalItem:
        return EvalItem(f"i{n}", cluster, f"prompt {n}", base, cand)

    # 45 clean JSON matches in "codegen", 15 shape breaks in "rare-json".
    # 45 is not arbitrary: a 0.90 bar needs n>=40 clean passes before the Wilson
    # lower bound clears it, so a 12-item cluster reads "unproven" no matter how
    # perfect the score. Sizing the demo below that would assert the engine is
    # broken when it is being careful.
    items = [item(i, "codegen", '{"a": 1}', '{"a": 1}') for i in range(45)]
    items += [item(100 + i, "rare-json", '{"a": 1}', '{"b": 1}') for i in range(15)]
    shares = {"codegen": 0.75, "rare-json": 0.25}

    m = run(items, shares=shares, candidate="glm-5.2", incumbent="gpt-5")
    good = next(c for c in m.candidates[0].clusters if c.cluster == "codegen")
    bad = next(c for c in m.candidates[0].clusters if c.cluster == "rare-json")
    assert good.band() == "equivalent", good.render_cell()
    assert bad.band() == "regressed", bad.render_cell()

    # A judge is never consulted for an item the graders already disqualified:
    # this one would raise if called, and the run still completes.
    def exploding_judge(c: Comparison) -> Reply:
        raise AssertionError("judge called on a structurally failed item")

    only_bad = [i for i in items if i.cluster == "rare-json"]
    m2 = run(
        only_bad,
        shares={"rare-json": 1.0},
        judge=exploding_judge,
        judge_model="claude-opus-5",
        agreement=Agreement(agreed=19, total=20, model="claude-opus-5"),
    )
    assert m2.candidates[0].clusters[0].band() == "regressed"

    # An anonymous judge is refused: an undisclosed judge is an unauditable score.
    try:
        run(items, shares=shares, judge=lambda c: Reply("A"), judge_model="  ")
        raise AssertionError("expected ValueError for a blank judge model")
    except ValueError as e:
        assert "disclosure" in str(e)

    # An empty eval set is refused rather than scored as 0%.
    try:
        run([], shares=shares)
        raise AssertionError("expected ValueError for an empty eval set")
    except ValueError as e:
        assert "empty eval set" in str(e)

    # The full kiosk: matrix + policy + a receipt whose digest is reproducible.
    r1 = suite(items, shares=shares, issued="2026-07-27", candidate="glm-5.2", incumbent="gpt-5")
    r2 = suite(items, shares=shares, issued="2026-07-27", candidate="glm-5.2", incumbent="gpt-5")
    assert r1.receipt.eval_set == r2.receipt.eval_set, "same eval set must digest identically"
    assert "codegen" in r1.render()
    # Regret is stated, never rounded away: rare-json is 25% of traffic.
    assert any(c.name == "rare-json" for c in r1.receipt.regret), r1.receipt.regret
    assert r1.policy.moved_share == 0.75, r1.policy.moved_share

    # Shares derived from observed cluster sizes, not asked for.
    try:
        shares_from_clusters([])
        raise AssertionError("expected ValueError for empty clusters")
    except ValueError as e:
        assert "0 captures" in str(e)

    print(f"suite: {len(items)} items, {len(m.candidates[0].clusters)} clusters")
    print(r1.render())


if __name__ == "__main__":
    demo()
