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

from .cost import blended, saving
from .graders import ItemResult
from .judge import Agreement, Calibration
from .stats import (
    Difference,
    Interval,
    McNemar,
    difference,
    family_wise_z,
    is_real,
    mcnemar,
    samples_needed,
    weighted_posterior,
    wilson,
)

#: A cluster whose whole interval sits below this is *regret*: keep the incumbent.
DEFAULT_EQUIVALENCE_BAR = 0.90


def _real(value: object, what: str, where: str = "") -> None:
    """Refuse a non-number before anything tries to compare or measure it.

    Both checks below reached for the value's *range* first, which assumes it
    has one. `0.0 < "0.9"` raises `TypeError`, and so does `math.isfinite(None)`
    — and `cli.main()` catches `ValueError`, not `TypeError`. So a receipt file
    carrying `"bar": "0.9"` still failed closed, as a traceback rather than as
    the sentence-and-exit-2 the repo promises for an untrusted file.

    `bool` is rejected with the rest: `True` is an `int`, and a bar of `True` is
    a bar of 1.0 nobody typed.

    Finiteness lives here too, in `is_real`, which is also why this does not
    call `math.isfinite` itself: a JSON document can carry an `int` too large to
    convert to a float, and `isfinite` raises `OverflowError` on it — the same
    uncaught-exception-class problem one layer down.

    Two messages, not one: `-inf` *is* a number, so folding it into "must be a
    number" makes the sentence wrong about the value it is naming.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{what} must be a number{where}, got {value!r}")
    if not is_real(value):
        raise ValueError(f"{what} must be a finite number{where}, got {value!r}")


def check_bar(bar: float) -> None:
    """Refuse a threshold that is not an equivalence bar.

    `clearly_above(bar)` is `interval.low > bar`, so a bar at or below zero
    holds for essentially any interval: a cluster scoring 41/80 reads
    `equivalent`, `movable_share` says move everything, and the artifact a human
    uses to authorise a cutover says a clearly-regressed candidate holds
    (invariant 8). The MCP schema advertises bounds, but a JSON Schema is
    advisory and nothing enforced it.

    One home, because `bar` reaches two places that do not go through each
    other: `Matrix`, which every report is assembled in, and `issue()`, which
    takes it directly and never builds a Matrix — so guarding only the first
    left the *receipt* — the portable proof artifact — issuable at `bar=0.0`,
    rendering "Proven at or above the 0% bar" over a regression.

    Open interval, matching how `family_wise_z` treats `alpha`: 0 admits
    everything and 1 admits nothing, and neither is a threshold anyone meant to
    set.

    Raises:
        ValueError: naming the offending value.
    """
    _real(bar, "equivalence bar")
    if not 0.0 < bar < 1.0:
        raise ValueError(f"equivalence bar must be between 0 and 1, exclusive, got {bar}")


def check_share(share: float, *, cluster: str = "") -> None:
    """Refuse a value that is not a fraction of traffic.

    One home for the rule, because it has to be applied in two places that
    cannot substitute for each other: on `ClusterScore`, which every report is
    built from, and over the raw `shares` map in `run()`, because a share whose
    cluster has no eval items never becomes a `ClusterScore` at all.

    Raises:
        ValueError: naming the offending value, and the cluster when known.
    """
    where = f" for {cluster}" if cluster else ""
    # `_real` covers both the type and finiteness — NaN must be caught before
    # the range test, because it fails every ordering comparison and would read
    # as "not out of range". The separate `math.isfinite` that used to sit here
    # was a second guard on the same route, and it was the one that raised
    # `OverflowError` on a large `int`.
    _real(share, "traffic share", where)
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"traffic share must be a fraction of traffic{where}, got {share}")


@dataclass(frozen=True, slots=True)
class ClusterScore:
    """One candidate's standing on one task cluster."""

    cluster: str
    name: str
    share: float
    interval: Interval
    ungraded: int
    #: Per-item verdicts, `(item_id, passed)`. Carried so two candidates over the
    #: same eval set can be compared *paired* — which items each one won, rather
    #: than two independent rates that throw the pairing away. Empty for a score
    #: built by hand rather than from results.
    outcomes: tuple[tuple[str, bool], ...] = ()
    #: Items dropped because an identical prompt already appeared in this
    #: cluster. A repeated prompt is one observation asked twice, not two
    #: independent ones, and a Wilson interval assumes independence — counting
    #: the copies narrows the interval on evidence that was never collected.
    #: Zero for the normal case; the report only mentions it when it is not.
    duplicates: int = 0

    def __post_init__(self) -> None:
        """Refuse a share that is not a fraction of traffic.

        Guarding the estimators in `stats` was guarding the wrong layer: shares
        arrive through `run(shares=…)`, reach `ClusterScore.share`, and
        `movable_share()` sums them raw without ever calling a weighted
        function. `{"good": 10.0}` produced a policy reading **"Move 900% of
        traffic to candidate"**, and a NaN share was carried into the receipt
        as `nan`. So the check belongs here, where every construction path goes.

        It is not sufficient on its own — a share whose cluster has no eval
        items never becomes a `ClusterScore` at all. `run()` therefore calls
        `check_share` over the raw map as well; see the note there.
        """
        check_share(self.share, cluster=self.cluster)

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

    def needed(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> int | None:
        """Graded items this cluster would need to clear ``bar``, at its own rate.

        ``None`` means no sample size fixes it: either nothing has been measured,
        or the observed rate is at or below the bar, and the answer is a better
        candidate rather than a bigger eval set. See
        [`samples_needed`][onpar.prove.stats.samples_needed].
        """
        return samples_needed(self.interval.passed, self.interval.total, bar)

    def render_need(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> str:
        """One line telling the operator what would settle this cluster."""
        if not self.known:
            return f"{self.name}: nothing graded yet — no rate to project from"
        n = self.needed(bar)
        if n is None:
            return (
                f"{self.name}: {self.interval.render()} — the observed rate is at or "
                f"below the {bar:.0%} bar; more items will not clear it"
            )
        return (
            f"{self.name}: needs {n} graded items to clear the {bar:.0%} bar at its "
            f"observed {self.interval.point:.0%} (has {self.interval.total}, "
            f"{n - self.interval.total} to go)"
        )


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """Everything measured about one candidate model."""

    model: str
    clusters: tuple[ClusterScore, ...]
    monthly_cost: float | None = None

    def __post_init__(self) -> None:
        """Refuse a set of shares that add up to more traffic than exists.

        Per-share validation on `ClusterScore` is not enough, and missing that
        is the same mistake twice: `movable_share()` *sums* shares, so the
        value reaching the arithmetic is the total, not any one element. Two
        individually legal shares — `{"a": 0.9, "b": 0.9}` — rendered "Move
        180% of traffic to candidate" and a receipt claiming "Movable: 180% of
        captured traffic".

        Tolerance because these are floats and a caller who split traffic
        exactly can still land a few ulps above 1.0.
        """
        total = sum(c.share for c in self.clusters)
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"cluster shares must not exceed all of the traffic, got {total:.6g} "
                f"across {len(self.clusters)} clusters"
            )

    def weighted_score(self) -> float | None:
        """Traffic-weighted equivalence, or ``None`` if nothing was graded."""
        usable = [c for c in self.clusters if c.known]
        if not usable:
            return None
        total_w = sum(c.share for c in usable)
        if total_w <= 0:
            return None
        return sum(c.interval.point * c.share for c in usable) / total_w

    def weighted_interval(self) -> Interval:
        """The weighted score *with* its uncertainty, by simulation.

        [`weighted_score`][onpar.prove.equivalence.CandidateReport.weighted_score]
        is a point estimate and the headline of the whole report, which is exactly
        the kind of number this repo refuses to print bare. Wilson does not apply
        to a weighted sum of proportions, so this simulates the sum instead — and
        the method travels with the interval. See
        [`weighted_posterior`][onpar.prove.stats.weighted_posterior].
        """
        return weighted_posterior([(c.interval, c.share) for c in self.clusters])

    def difference_against(self, other: CandidateReport) -> dict[str, Difference]:
        """Per-cluster candidate-minus-other, with an interval on the difference.

        Unpaired (Newcombe method 10). Use alongside
        [`paired_against`][onpar.prove.equivalence.CandidateReport.paired_against],
        which is the sharper test when both were scored on the same items.
        """
        mine = {c.cluster: c for c in self.clusters}
        return {
            c.cluster: difference(
                mine[c.cluster].interval.passed,
                mine[c.cluster].interval.total,
                c.interval.passed,
                c.interval.total,
            )
            for c in other.clusters
            if c.cluster in mine
        }

    def paired_against(self, other: CandidateReport) -> dict[str, McNemar]:
        """Per-cluster McNemar over the items both candidates were graded on.

        Only clusters where per-item outcomes survived on both sides appear;
        a cluster scored from counts alone cannot be paired, and inventing a
        pairing would be worse than declining to report one.
        """
        theirs = {c.cluster: dict(c.outcomes) for c in other.clusters if c.outcomes}
        out: dict[str, McNemar] = {}
        for c in self.clusters:
            if not c.outcomes or c.cluster not in theirs:
                continue
            ref = theirs[c.cluster]
            shared = [(ok, ref[i]) for i, ok in c.outcomes if i in ref]
            if not shared:
                continue
            out[c.cluster] = mcnemar(
                worse=sum(1 for mine, ref_ok in shared if not mine and ref_ok),
                better=sum(1 for mine, ref_ok in shared if mine and not ref_ok),
            )
        return out

    def regret(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> tuple[ClusterScore, ...]:
        """Clusters where the candidate is *confidently* worse.

        Only ``regressed`` — not ``unproven``. Sending a cluster to the regret set
        because the evidence is thin would keep traffic on the incumbent forever;
        thin evidence means *gather more*, not *give up*.
        """
        return tuple(c for c in self.clusters if c.band(bar) == "regressed")

    def unproven(self, bar: float = DEFAULT_EQUIVALENCE_BAR) -> tuple[ClusterScore, ...]:
        """Everything that is neither proven nor regret — including unmeasured.

        `unknown` belongs here with `unproven`: traffic nobody measured cannot
        move, for the same reason traffic that straddles the bar cannot. Naming
        only the straddlers dropped an unmeasured cluster out of the hybrid
        policy entirely, so a share the report showed as `?` reached
        `SuiteResult.policy` and the MCP surface as if it did not exist. The
        receipt has always partitioned it this way — `Receipt.unproven` holds
        both — and the two spellings of one word disagreeing is what let it
        through.
        """
        return tuple(c for c in self.clusters if c.band(bar) in ("unproven", "unknown"))

    def measured_share(self) -> float:
        """Fraction of traffic any evidence was actually collected for."""
        return sum(c.share for c in self.clusters if c.known)

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
    #: The equivalence bar the moved clusters cleared, and the reason the saving
    #: line can describe itself honestly. Defaulted so nothing constructing a
    #: Policy positionally breaks.
    bar: float = DEFAULT_EQUIVALENCE_BAR
    #: What the moved share was measured on. Money without these is a point
    #: estimate, and invariant 6 applies to dollars exactly as to scores — so
    #: the policy line and the receipt say the same thing, from the same
    #: function, or they say nothing.
    traffic_captures: int = 0
    traffic_window: str = ""
    #: What each unproven cluster would need to settle, already worded — see
    #: [`ClusterScore.render_need`][onpar.prove.equivalence.ClusterScore.render_need].
    #: "Gather more evidence" without a number is an instruction nobody can follow.
    needs: tuple[str, ...] = ()

    @property
    def blended_cost(self) -> float | None:
        """Cost of running the hybrid, or ``None`` if either rate is unknown.

        Returning ``None`` rather than guessing: a fabricated saving is the
        single most damaging number this report could print.
        """
        # One formula, in `cost`. It was about to exist here and in `Receipt`,
        # and the fact you can least afford two answers to is what the
        # migration saves.
        return blended(self.incumbent_cost, self.candidate_cost, self.moved_share)

    @property
    def monthly_saving(self) -> float | None:
        b = self.blended_cost
        return None if b is None or self.incumbent_cost is None else self.incumbent_cost - b

    def render(self) -> str:
        lines = [f"Move {self.moved_share * 100:.0f}% of traffic to {self.candidate}."]
        if self.regret_clusters:
            lines.append(f"  Keep the incumbent for: {', '.join(self.regret_clusters)}")
        if self.unproven_clusters:
            lines.append(f"  Not yet proven: {', '.join(self.unproven_clusters)}")
            lines += [f"    → {n}" for n in self.needs]
        # The same `Saving` the receipt renders, from the same function. This
        # line and the receipt's are the same claim about the same migration;
        # computing them twice is how they come to disagree.
        #
        # It also said "at zero measured quality loss", which was false: a
        # cluster moves when its whole interval clears the bar, so at a 0.90
        # bar a cluster passing 98% [95-99] moves — four items in a 200-item run
        # whose answers measurably differed. That was the strongest possible
        # claim attached to the one number a reader most wants to believe. What
        # is true is that the moved traffic met the bar they chose, and the bar
        # is named so they can judge it.
        money = saving(
            self.incumbent_cost,
            self.candidate_cost,
            self.moved_share,
            captures=self.traffic_captures,
            window=self.traffic_window,
        )
        if not money:
            lines.append(f"  {money.render()}")
        else:
            pct = (money.point / self.incumbent_cost * 100) if self.incumbent_cost else 0
            lines.append(
                f"  Saving: ${money.low:,.0f}–${money.high:,.0f}/mo (~{pct:.0f}%) on traffic "
                f"proven at or above the {self.bar * 100:.0f}% bar"
            )
            lines.append(f"    {money.basis}")
        return "\n".join(lines)


@dataclass(slots=True)
class Matrix:
    """The full comparison across candidates."""

    candidates: list[CandidateReport] = field(default_factory=list)
    agreement: Agreement | None = None
    incumbent: str = "incumbent"
    incumbent_cost: float | None = None
    #: What the traffic mix was measured on, carried through to the saving so
    #: the money inherits the uncertainty of the share it is derived from.
    traffic_captures: int = 0
    traffic_window: str = ""
    bar: float = DEFAULT_EQUIVALENCE_BAR
    #: Judge-vs-deterministic-grader agreement measured during this run. Present
    #: whenever a judge was used, because a judge that was never checked against
    #: anything is a claim, not a measurement.
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        check_bar(self.bar)

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
        # `share > 0`, because a cluster carrying none of the traffic cannot
        # be moved, held back, or gathered more evidence for — naming it
        # under "Not yet proven" beside "Move 100% of traffic" reads as a
        # contradiction and is really just noise. `unproven()` keeps every
        # unknown, because as a *query* that is the honest answer; the
        # policy is a list of things to do about traffic. `needs` must use
        # the same filtered set, or a 0%-share cluster left out of the
        # header still gets an orphaned remediation bullet in render().
        actionable = tuple(c for c in candidate.unproven(self.bar) if c.share > 0)
        return HybridPolicy(
            candidate=candidate.model,
            moved_share=candidate.movable_share(self.bar),
            regret_clusters=tuple(c.name for c in candidate.regret(self.bar)),
            unproven_clusters=tuple(c.name for c in actionable),
            incumbent_cost=self.incumbent_cost,
            candidate_cost=candidate.monthly_cost,
            needs=tuple(c.render_need(self.bar) for c in actionable),
            bar=self.bar,
            traffic_captures=self.traffic_captures,
            traffic_window=self.traffic_window,
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

        # The header only needs cluster names and traffic shares, which every
        # candidate carries identically, so the first candidate is fine for it.
        header_clusters = self.candidates[0].clusters
        # The disclosures below are not. `ungraded` depends on the candidate's
        # own output — ToolArgs returns NOT_APPLICABLE when the candidate made
        # no tool calls — so ungraded, interval.total and therefore
        # `underpowered` genuinely differ per candidate. Reporting the first
        # candidate's excluded items next to a recommendation to migrate to a
        # different one attributes a real measurement to the wrong model.
        disclosed = (best or self.candidates[0]).clusters
        head = f"{'':<22}" + "".join(f"{c.name[:16]:>18}" for c in header_clusters)
        out.append(head)
        out.append(
            f"{'':<22}" + "".join(f"{f'({c.share * 100:.0f}%)':>18}" for c in header_clusters)
        )
        out.append("-" * len(head))
        for cand in self.candidates:
            row = f"{cand.model[:20]:<22}"
            for c in cand.clusters:
                row += f"{c.render_cell():>18}"
            # WITH its interval. This printed `weighted_score()` bare — the
            # headline of the whole report, on every render, for every
            # candidate. `weighted_interval` was already written and already
            # documented as "exactly the kind of number this repo refuses to
            # print bare"; it simply was not called. Invariant 6.
            ws = cand.weighted_score()
            if ws is None:
                row += "   ? weighted"
            else:
                iv = cand.weighted_interval()
                row += f"   {ws * 100:.0f}% [{iv.low * 100:.0f}–{iv.high * 100:.0f}] weighted"
                # Both the score and the interval renormalise over the traffic
                # that was actually measured, which is the only arithmetic
                # available — an unmeasured cluster has no rate to contribute.
                # Renormalising silently is what makes it a lie: 70% covered and
                # all passing printed a flat "100% weighted" while 30% of traffic
                # had never been looked at. The denominator travels with the
                # number, like the method does.
                measured = cand.measured_share()
                if measured < 0.999:
                    row += f" of {measured * 100:.0f}%"
            out.append(row)
        out.append("-" * len(head))
        out.append(
            f"{self.incumbent + ' (incumbent)':<22}"
            + "".join(f"{'100%':>18}" for _ in header_clusters)
        )

        out += self._render_statistics(best)

        # Provenance last, always present.
        out.append("")
        if self.agreement is None:
            out.append("judge: not used — verdicts are from deterministic graders only")
        elif not self.judge_trustworthy:
            # Says what actually happens. The scores are *in* the numbers above —
            # what an untrusted judge cannot do is move traffic, which is enforced
            # in `prove.gate`, not here. Claiming the cells were blanked when they
            # were not is the kind of reassuring inaccuracy this report exists to
            # avoid.
            out.append(
                f"⚠ {self.agreement.render()} — below the trust bar; its verdicts are "
                f"counted in the cells above but cannot advance any cluster"
            )
        else:
            out.append(self.agreement.render())
        if self.calibration is not None:
            out.append(self.calibration.render())

        thin = [c for c in disclosed if c.interval.underpowered]
        if thin:
            names = ", ".join(c.name for c in thin)
            out.append(f"⚠ underpowered clusters (too few samples to conclude): {names}")
        ung = sum(c.ungraded for c in disclosed)
        if ung:
            out.append(
                f"⚠ {ung} items had no applicable grader and are excluded, not counted as passes"
            )
        # Collapsing silently would be its own defect: the reader sees a
        # denominator smaller than the file they wrote and has no way to know
        # why, and the eval set stays wrong because nothing told them.
        dup = [c for c in disclosed if c.duplicates]
        if dup:
            total = sum(c.duplicates for c in dup)
            detail = ", ".join(f"{c.name} ({c.duplicates})" for c in dup)
            out.append(
                f"⚠ {total} duplicate prompt(s) collapsed — {detail}. A prompt repeated "
                f"within a cluster is one observation asked twice, and every interval "
                f"here assumes independent ones, so the copies are merged rather than "
                f"counted. An item passes only if every copy of it passed."
            )
        return "\n".join(out)

    def _render_statistics(self, best: CandidateReport | None) -> list[str]:
        """The methods block: what estimator produced which number, and what is
        still missing to settle the clusters that did not resolve.

        Every line here names its method. A reader who cannot tell a Wilson
        interval from a bootstrap one cannot audit the verdict, and a report they
        cannot audit is a dashboard with extra steps.
        """
        if best is None:
            return []
        out = [""]

        agg = best.weighted_interval()
        if agg.total:
            measured = best.measured_share()
            # "never measured" is true of an uncovered cluster and false of one
            # whose items all came back with no applicable grader: those were
            # collected and run, and excluded at grading. `known` means
            # *graded* (`interval.total > 0`), so both landed under one word.
            #
            # The distinction is free — a cluster carries its `ungraded` count —
            # so the sentence names whichever case it is rather than picking the
            # stronger claim and being wrong half the time.
            missing = [c for c in best.clusters if not c.known]
            collected = sum(c.share for c in missing if c.ungraded)
            coverage = (
                ""
                if measured >= 0.999
                else (
                    f" — over {measured:.0%} of traffic; the other {1 - measured:.0%} "
                    + (
                        "was collected but nothing in it could be graded"
                        if collected >= (1 - measured) - 1e-9
                        else "was never measured"
                        if collected <= 1e-9
                        else "was not graded — some of it never measured, some collected "
                        "with no applicable grader"
                    )
                    + " and is not in this number"
                )
            )
            out.append(f"weighted verdict {agg.render()}{coverage} — {agg.method}")
            out.append(
                "  per-cluster cells above are Wilson score intervals, 95%; the weighted "
                "figure is a sum of proportions, which Wilson does not cover"
            )

        # Multiplicity, stated rather than left implicit.
        scored = [c for c in best.clusters if c.known]
        if scored:
            z = family_wise_z(len(scored))
            still = sum(
                1
                for c in scored
                if wilson(c.interval.passed, c.interval.total, z).clearly_above(self.bar)
            )
            now = sum(1 for c in scored if c.band(self.bar) == "equivalent")
            survives = (
                "none of them clears it even unadjusted"
                if now == 0
                else f"{still} of the {now} that clear it would still clear it"
            )
            out.append(
                f"multiplicity: {len(scored)} clusters scored against the same "
                f"{self.bar:.0%} bar; the intervals above are UNADJUSTED. Under "
                f"Bonferroni (α=0.05/{len(scored)}, z={z:.2f}), {survives}."
            )

        # Power, as an instruction rather than a shrug.
        open_clusters = [c for c in best.clusters if c.band(self.bar) in ("unproven", "unknown")]
        if open_clusters:
            out.append("to settle the open clusters:")
            out += [f"  · {c.render_need(self.bar)}" for c in open_clusters]

        # Effect size only where a second arm actually exists. The incumbent
        # column is not one: every grader measures agreement *with* it, so its
        # 100% is a definition, not a measurement, and differencing against it
        # would dress the candidate's own interval up as a comparison.
        others = [c for c in self.candidates if c is not best]
        for other in others:
            diffs = best.difference_against(other)
            paired = best.paired_against(other)
            if not diffs:
                continue
            out.append(f"effect size, {best.model} − {other.model}:")
            for key, d in diffs.items():
                line = f"  · {key}: {d.render()}"
                if key in paired:
                    line += f" · paired: {paired[key].render()}"
                out.append(line)
            out.append(
                f"  differences are {next(iter(diffs.values())).method}; paired counts "
                f"are exact McNemar over the items both models were graded on"
            )
        return out


def score_cluster(
    cluster: str,
    name: str,
    share: float,
    results: list[ItemResult],
    duplicates: int = 0,
) -> ClusterScore:
    """Turn per-item results into one cluster score.

    Items where no grader applied are counted separately and excluded from the
    denominator — including them as failures would punish a model for our
    instrument's blind spots, and as passes would be a lie.

    ``duplicates`` is how many items were collapsed before this was called, for
    disclosure only; the arithmetic here already sees the collapsed set.
    """
    graded = [r for r in results if r.graded]
    passed = sum(1 for r in graded if r.passed)
    return ClusterScore(
        cluster=cluster,
        name=name,
        share=share,
        interval=wilson(passed, len(graded)),
        ungraded=len(results) - len(graded),
        outcomes=tuple((r.item_id, r.passed) for r in graded),
        duplicates=duplicates,
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

    # The headline number carries an interval, and names what produced it.
    assert "weighted verdict" in text and "Jeffreys" in text, text
    # Multiplicity is stated, never left for the reader to infer.
    assert "UNADJUSTED" in text and "Bonferroni" in text, text

    # An open cluster is told how much more evidence it needs. 3/3 is perfect and
    # useless; 35 flawless items is what a 90% bar costs.
    assert thin.needed() == 35, thin.needed()
    assert "needs 35 graded items" in text, text
    # ...and a cluster that is simply worse is not sent to gather more of it.
    assert longctx.needed() is None
    assert "will not clear it" in longctx.render_need()

    # Effect size, where a second arm actually exists.
    rival = CandidateReport(
        "qwen3-32b",
        (
            score_cluster("c1", "codegen", 0.60, results("c1", 90, 10)),
            score_cluster("c2", "long-ctx refactor", 0.15, results("c2", 30, 70)),
            score_cluster("c3", "rare-json", 0.25, results("c3", 3, 0, n_ungraded=2)),
        ),
    )
    d = cand.difference_against(rival)["c1"]
    assert abs(d.point - 0.06) < 1e-9, d.render()
    # 96% against 90% over 100 items each *looks* decisive and is not: the
    # interval on the difference still contains zero. This is the entire reason
    # the difference is reported instead of two pass rates side by side.
    assert not d.significant, d.render()
    assert cand.difference_against(rival)["c2"].point == 0.0, "a tie is a tie"
    two = Matrix([cand, rival], incumbent="gpt-5").render()
    assert "effect size, glm-5.2 − qwen3-32b" in two, two
    assert "Newcombe" in two and "McNemar" in two, two

    # The paired view sees what two independent rates cannot: these two agreed on
    # every item they were both graded on, so there is nothing to separate them.
    same = cand.paired_against(cand)
    assert same["c1"].discordant == 0 and same["c1"].p_value == 1.0

    print(text)
    print()
    print(m.hybrid_for(cand).render())
    print("ok")


if __name__ == "__main__":
    demo()
