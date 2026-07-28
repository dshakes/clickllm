"""The equivalence matrix — the artifact the whole product exists to produce.

One table a CTO reads in ten seconds and an engineer drills into for an hour.
Four decisions shape it, and each one is a correction to how these comparisons
usually get presented:

**Traffic-weighted.** A cluster that is 38% of your load counts five times one at
8%. Unweighted averages flatter models that are good at rare tasks.

**Incumbent pinned at 100.** Nobody is asking whether the candidate is *good*.
They are asking whether switching is a *downgrade*. Everything is relative to
what you run today.

**Regret above the fold.** Where the candidate loses is printed first. The honest
failure is what makes the rest of the table credible — and it is what turns a
yes/no into a hybrid policy that actually ships.

**Never a number without its interval.** A cell with too little evidence renders
as unknown rather than as a confident score. See :mod:`.stats`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .graders import ItemResult
from .judge import Agreement
from .stats import Interval, wilson

#: A cluster whose whole interval sits below this is *regret*: keep the incumbent.
DEFAULT_EQUIVALENCE_BAR = 0.90


@dataclass(frozen=True, slots=True)
class ClusterScore:
    """One candidate's standing on one task cluster."""

    cluster: str
    name: str
    share: float
    interval: Interval
    ungraded: int

    @property
    def known(self) -> bool:
        """Whether anything was actually measured here."""
        return self.interval.total > 0

    def band(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> str:
        """A coarse label. Uses the interval, never the point estimate."""
        if not self.known:
            return "unknown"
        if self.interval.clearly_above(bar):
            return "equivalent"
        if self.interval.clearly_below(bar):
            return "regressed"
        return "unproven"  # straddles the bar — more evidence needed

    def render_cell(self) -> str:
        return "?" if not self.known else self.interval.render()


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Everything measured about one candidate model."""

    model: str
    clusters: tuple[ClusterScore, ...]
    monthly_cost: float | None = None

    def weighted_score(self) -> float | None:
        """Traffic-weighted equivalence, or ``None`` if nothing was graded."""
        usable = [c for c in self.clusters if c.known]
        if not usable:
            return None
        total_w = sum(c.share for c in usable)
        if total_w <= 0:
            return None
        return sum(c.interval.point * c.share for c in usable) / total_w

    def regret(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> tuple[ClusterScore, ...]:
        """Clusters where the candidate is *confidently* worse.

        Only ``regressed`` — not ``unproven``. Sending a cluster to the regret set
        because the evidence is thin would keep traffic on the incumbent forever;
        thin evidence means *gather more*, not *give up*.
        """
        return tuple(c for c in self.clusters if c.band(bar) == "regressed")

    def unproven(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> tuple[ClusterScore, ...]:
        return tuple(c for c in self.clusters if c.band(bar) == "unproven")

    def movable_share(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> float:
        """Fraction of traffic that can move today, on the evidence available."""
        return sum(c.share for c in self.clusters if c.band(bar) == "equivalent")


@dataclass(frozen=True, slots=True)
class HybridPolicy:
    """Move what is proven; keep the incumbent for the rest."""

    candidate: str
    moved_share: float
    regret_clusters: tuple[str, ...]
    unproven_clusters: tuple[str, ...]
    incumbent_cost: float | None
    candidate_cost: float | None

    @property
    def blended_cost(self) -> float | None:
        """Cost of running the hybrid, or ``None`` if either rate is unknown.

        Returning ``None`` rather than guessing: a fabricated saving is the
        single most damaging number this report could print.
        """
        if self.incumbent_cost is None or self.candidate_cost is None:
            return None
        return self.candidate_cost * self.moved_share + self.incumbent_cost * (1 - self.moved_share)

    @property
    def monthly_saving(self) -> float | None:
        b = self.blended_cost
        return None if b is None or self.incumbent_cost is None else self.incumbent_cost - b

    def render(self) -> str:
        lines = [f"Move {self.moved_share * 100:.0f}% of traffic to {self.candidate}."]
        if self.regret_clusters:
            lines.append(f"  Keep the incumbent for: {', '.join(self.regret_clusters)}")
        if self.unproven_clusters:
            lines.append(
                f"  Not yet proven (gather more evidence): {', '.join(self.unproven_clusters)}"
            )
        s = self.monthly_saving
        if s is None:
            lines.append("  Saving: unknown — no cost rate configured")
        else:
            pct = (s / self.incumbent_cost * 100) if self.incumbent_cost else 0
            lines.append(f"  Saving: ${s:,.0f}/mo ({pct:.0f}%) at zero measured quality loss")
        return "\n".join(lines)


@dataclass(slots=True)
class Matrix:
    """The full comparison across candidates."""

    candidates: list[CandidateReport] = field(default_factory=list)
    agreement: Agreement | None = None
    incumbent: str = "incumbent"
    incumbent_cost: float | None = None
    bar: float = DEFAULT_EQUIVALENCE_BAR

    @property
    def judge_trustworthy(self) -> bool:
        """Whether judge-derived cells should be presented as evidence."""
        return self.agreement is not None and self.agreement.trustworthy

    def best(self) -> CandidateReport | None:
        """Most traffic movable today, breaking ties on weighted score."""
        scored = [c for c in self.candidates if c.weighted_score() is not None]
        if not scored:
            return None
        return max(scored, key=lambda c: (c.movable_share(self.bar), c.weighted_score() or 0))

    def hybrid_for(self, candidate: CandidateReport) -> HybridPolicy:
        return HybridPolicy(
            candidate=candidate.model,
            moved_share=candidate.movable_share(self.bar),
            regret_clusters=tuple(c.name for c in candidate.regret(self.bar)),
            unproven_clusters=tuple(c.name for c in candidate.unproven(self.bar)),
            incumbent_cost=self.incumbent_cost,
            candidate_cost=candidate.monthly_cost,
        )

    def render(self) -> str:
        """Text form. Regret first, then the table, then provenance."""
        if not self.candidates:
            return "No candidates evaluated."

        out: list[str] = []
        best = self.best()

        # Regret above the fold.
        if best is not None:
            regret = best.regret(self.bar)
            if regret:
                out.append("REGRET — keep the incumbent for these:")
                for c in regret:
                    out.append(f"  {c.name}  ({c.share * 100:.0f}% of traffic)  {c.render_cell()}")
                out.append("")

        clusters = self.candidates[0].clusters
        head = f"{'':<22}" + "".join(f"{c.name[:16]:>18}" for c in clusters)
        out.append(head)
        out.append(f"{'':<22}" + "".join(f"{f'({c.share * 100:.0f}%)':>18}" for c in clusters))
        out.append("-" * len(head))
        for cand in self.candidates:
            row = f"{cand.model[:20]:<22}"
            for c in cand.clusters:
                row += f"{c.render_cell():>18}"
            ws = cand.weighted_score()
            row += f"   {ws * 100:.0f}% weighted" if ws is not None else "   ? weighted"
            out.append(row)
        out.append("-" * len(head))
        out.append(
            f"{self.incumbent + ' (incumbent)':<22}" + "".join(f"{'100%':>18}" for _ in clusters)
        )

        # Provenance last, always present.
        out.append("")
        if self.agreement is None:
            out.append("judge: not used — verdicts are from deterministic graders only")
        elif not self.judge_trustworthy:
            out.append(
                f"⚠ {self.agreement.render()} — below the trust bar; judge cells shown as unknown"
            )
        else:
            out.append(self.agreement.render())

        thin = [c for c in clusters if c.interval.underpowered]
        if thin:
            names = ", ".join(c.name for c in thin)
            out.append(f"⚠ underpowered clusters (too few samples to conclude): {names}")
        ung = sum(c.ungraded for c in clusters)
        if ung:
            out.append(
                f"⚠ {ung} items had no applicable grader and are excluded, not counted as passes"
            )
        return "\n".join(out)


def score_cluster(
    cluster: str,
    name: str,
    share: float,
    results: list[ItemResult],
) -> ClusterScore:
    """Turn per-item results into one cluster score.

    Items where no grader applied are counted separately and excluded from the
    denominator — including them as failures would punish a model for our
    instrument's blind spots, and as passes would be a lie.
    """
    graded = [r for r in results if r.graded]
    passed = sum(1 for r in graded if r.passed)
    return ClusterScore(
        cluster=cluster,
        name=name,
        share=share,
        interval=wilson(passed, len(graded)),
        ungraded=len(results) - len(graded),
    )


def demo() -> None:
    from .graders import EvalItem, grade

    def results(cluster: str, n_pass: int, n_fail: int, n_ungraded: int = 0) -> list[ItemResult]:
        out = []
        for i in range(n_pass):
            out.append(grade(EvalItem(f"{cluster}-p{i}", cluster, "p", '{"a":1}', '{"a":1}')))
        for i in range(n_fail):
            out.append(grade(EvalItem(f"{cluster}-f{i}", cluster, "p", '{"a":1}', "not json")))
        for i in range(n_ungraded):
            out.append(grade(EvalItem(f"{cluster}-u{i}", cluster, "p", "", "")))
        return out

    codegen = score_cluster("c1", "codegen", 0.60, results("c1", 96, 4))
    longctx = score_cluster("c2", "long-ctx refactor", 0.15, results("c2", 30, 70))
    thin = score_cluster("c3", "rare-json", 0.25, results("c3", 3, 0, n_ungraded=2))

    cand = CandidateReport("glm-5.2", (codegen, longctx, thin), monthly_cost=210.0)
    m = Matrix(
        [cand],
        agreement=Agreement(36, 40, "claude-opus-5"),
        incumbent="gpt-5",
        incumbent_cost=2847.0,
    )

    # Bands come from the interval, not the point.
    assert codegen.band() == "equivalent", codegen.render_cell()
    assert longctx.band() == "regressed"
    assert thin.band() == "unknown" or thin.interval.underpowered

    # Regret is only the confidently-worse cluster.
    regret = cand.regret()
    assert [c.name for c in regret] == ["long-ctx refactor"]

    # Ungraded items are excluded from the denominator, not silently passed.
    assert thin.ungraded == 2
    assert thin.interval.total == 3

    # Hybrid economics, and no fabricated saving without rates.
    policy = m.hybrid_for(cand)
    assert policy.monthly_saving is not None and policy.monthly_saving > 0
    no_rates = Matrix([CandidateReport("x", (codegen,))], incumbent_cost=None).hybrid_for(
        CandidateReport("x", (codegen,))
    )
    assert no_rates.monthly_saving is None, "must not invent a saving without a cost rate"

    # An untrustworthy judge is called out rather than quietly used.
    shaky = Matrix([cand], agreement=Agreement(5, 6, "m"), incumbent_cost=2847.0)
    assert not shaky.judge_trustworthy
    assert "below the trust bar" in shaky.render()

    text = m.render()
    assert text.index("REGRET") < text.index("codegen"), "regret must come first"
    assert "gpt-5 (incumbent)" in text and "100%" in text
    assert "underpowered" in text or "no applicable grader" in text

    print(text)
    print()
    print(m.hybrid_for(cand).render())
    print("ok")


if __name__ == "__main__":
    demo()
