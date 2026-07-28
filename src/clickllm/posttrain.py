"""Post-training — when proving a model cannot replace yours is the wrong answer.

Everything upstream answers *can this open model replace the closed one, as
shipped?* Sometimes the honest answer is no: the candidate is equivalent on
seven task clusters and loses badly on the eighth. The migration proceeds on the
seven, the eighth stays on the incumbent, and that is a real, defensible outcome.

But it is not the only one. **The corpus that proved the gap is also the corpus
that could close it.**

## The asset nobody else is holding

M6 captured real production traffic: the prompt your users actually sent, and
the answer your *incumbent* — the expensive, capable model — actually gave. That
is, precisely, a distillation dataset. Teacher outputs on in-distribution
prompts, which is the expensive part of building one, already collected as a
side effect of proving equivalence.

An eval harness cannot offer this. It has benchmark questions and no teacher. A
fine-tuning platform cannot offer it either: it has a trainer and no idea what
your traffic looks like. The loop only closes if the same tool captured the
traffic, clustered it, and knows *which cluster* failed.

So the recommendation this module makes is narrow and specific: **train on the
regret set, not on everything.** The clusters that already pass need no help,
and including them spends money to move a number that is already above the bar.

## What this module refuses to do

**Train anything.** No GPU is touched here and no dependency is added. It reads
the receipt, decides whether post-training is warranted, sizes the job, and
emits a recipe someone runs deliberately. Fine-tuning is expensive, it is easy
to make a model worse, and a tool that silently kicks off a training run because
a score dipped would be the single most dangerous thing in this repo.

**Promise an outcome.** Distillation on a few thousand in-distribution examples
routinely closes a modest gap and routinely fails to close a large one. Where the
gap is wide, this says so and recommends staying on the incumbent rather than
selling a training run that will not work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from clickllm.prove.receipt import Claim, Receipt

__all__ = [
    "MIN_EXAMPLES_PER_CLUSTER",
    "PLAUSIBLE_GAP",
    "WIDE_GAP",
    "Method",
    "Recipe",
    "Recommendation",
    "demo",
    "recommend",
]

# --- calibration ---------------------------------------------------------------

#: Below this many captured examples a cluster cannot support fine-tuning at all.
#: LoRA on a few dozen samples memorises them and generalises to nothing.
MIN_EXAMPLES_PER_CLUSTER = 200

#: An equivalence gap up to this size is the range where distillation on
#: in-distribution traffic routinely closes it.
PLAUSIBLE_GAP = 0.15

#: Beyond this, the candidate is not slightly behind — it is a different
#: capability level, and training will not bridge it. Saying so is worth more
#: than selling the attempt.
WIDE_GAP = 0.35


class Method(StrEnum):
    """How to close a gap. Ordered cheapest-first, which is also safest-first."""

    #: Better prompting on the failing cluster. Free, reversible, and skipped far
    #: too often in favour of training.
    PROMPT = "prompt"
    #: LoRA on captured incumbent outputs. The default when there is a real gap
    #: and real data — cheap, and the adapter can be served alongside the base.
    LORA = "lora"
    #: Preference tuning where the failure is about style or format rather than
    #: capability, and pairs can be built from graded outcomes.
    PREFERENCE = "preference"
    #: Do not train. The gap is capability, not calibration.
    STAY = "stay"


@dataclass(frozen=True, slots=True)
class Recipe:
    """A training job, sized and costed, that a human decides to run."""

    method: Method
    cluster: str
    examples: int
    #: Where the training pairs come from.
    source: str
    #: Concrete steps. Deliberately prose a person executes, not an API call this
    #: module makes.
    steps: tuple[str, ...]
    #: What could go wrong, stated before the money is spent.
    risks: tuple[str, ...] = ()

    def render(self) -> str:
        """The recipe, with its risks attached rather than in a footnote."""
        out = [f"{self.method.value} · {self.cluster} · {self.examples:,} examples"]
        out.append(f"  source: {self.source}")
        out += [f"  {i + 1}. {s}" for i, s in enumerate(self.steps)]
        if self.risks:
            out.append("  risks:")
            out += [f"    · {r}" for r in self.risks]
        return "\n".join(out)


@dataclass(frozen=True, slots=True)
class Recommendation:
    """What to do about the clusters that did not pass."""

    recipes: tuple[Recipe, ...] = ()
    #: Clusters left on the incumbent, with the reason.
    stay: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def worth_training(self) -> bool:
        """Whether any cluster is worth a training run."""
        return any(r.method is not Method.STAY for r in self.recipes)

    def render(self) -> str:
        """Terminal-shaped. What to leave alone comes first."""
        out: list[str] = []
        if self.stay:
            out.append("LEAVE ON THE INCUMBENT:")
            out += [f"  · {name} — {why}" for name, why in self.stay]
            out.append("")
        if self.recipes:
            out.append("WORTH TRAINING:")
            out += [r.render() for r in self.recipes]
        if not self.recipes and not self.stay:
            out.append("Nothing to do — every cluster is already at or above the bar.")
        if self.notes:
            out += [""] + [f"note: {n}" for n in self.notes]
        return "\n".join(out)


def _gap(claim: Claim, bar: float) -> float:
    """How far below the bar this cluster sits, at its point estimate."""
    return max(0.0, bar - (claim.point or 0.0))


def _lora(cluster: str, examples: int, gap: float, incumbent: str) -> Recipe:
    """A distillation recipe built from captured incumbent outputs."""
    return Recipe(
        method=Method.LORA,
        cluster=cluster,
        examples=examples,
        source=(
            f"captured traffic for this cluster — the prompt your users sent and "
            f"the answer {incumbent} gave. Already redacted, already clustered."
        ),
        steps=(
            f"export the {examples:,} captured pairs for this cluster only — "
            f"training on clusters that already pass spends money to move a "
            f"number that is already above the bar",
            "hold back 20% as a test split, sampled by the same clustering so the "
            "split is in-distribution rather than random",
            "LoRA at rank 16 on the candidate base; a full fine-tune is rarely "
            "warranted for a single task shape and is far harder to roll back",
            "serve the adapter alongside the base model — one set of weights, "
            "many adapters, so the other clusters are untouched",
            "re-prove against the SAME eval set. A new eval set would make the "
            "before and after incomparable, which is the commonest way a "
            "fine-tune appears to work",
        ),
        risks=(
            f"the gap is {gap:.0%}; distillation closes gaps of this size often, "
            f"not always. Budget for the possibility that it does not",
            "the adapter learns the incumbent's failures too — captured output is "
            "not ground truth, it is what the expensive model happened to say",
            "an adapter trained on today's traffic decays as traffic drifts; the "
            "guard's traffic-drift check applies to it as much as to the receipt",
        ),
    )


def recommend(
    receipt: Receipt,
    examples_per_cluster: dict[str, int] | None = None,
) -> Recommendation:
    """Decide what, if anything, is worth training.

    Args:
        receipt: the proof that identified the gaps. Its regret set is the input
            that matters — the clusters that were measured and lost.
        examples_per_cluster: captured examples available per cluster. A cluster
            absent here has no data, which is a reason to decline rather than to
            assume.

    Returns:
        A [`Recommendation`]. Nothing in it executes; every recipe is a job a
        human chooses to run.
    """
    have = examples_per_cluster or {}
    recipes: list[Recipe] = []
    stay: list[tuple[str, str]] = []
    notes: list[str] = []

    for claim in receipt.regret:
        gap = _gap(claim, receipt.bar)
        n = have.get(claim.cluster, 0)

        if gap > WIDE_GAP:
            stay.append(
                (
                    claim.name,
                    f"{gap:.0%} below the bar. That is a different capability "
                    f"level, not a calibration problem — training will not bridge "
                    f"it, and saying so is worth more than selling the attempt",
                )
            )
            continue

        if n < MIN_EXAMPLES_PER_CLUSTER:
            stay.append(
                (
                    claim.name,
                    f"only {n:,} captured examples against a {MIN_EXAMPLES_PER_CLUSTER:,} "
                    f"floor. LoRA on this little memorises the samples and "
                    f"generalises to nothing — capture more traffic first",
                )
            )
            continue

        # Cheapest first. A formatting failure is a prompt problem, and reaching
        # for a GPU before trying the free fix is how teams spend a week on a
        # missing instruction.
        if gap <= 0.05:
            recipes.append(
                Recipe(
                    method=Method.PROMPT,
                    cluster=claim.cluster,
                    examples=n,
                    source="the failing captures themselves — read twenty of them",
                    steps=(
                        "read the twenty worst-scoring captured pairs for this "
                        "cluster before writing any training config",
                        "a gap this small is usually a missing instruction or an "
                        "output-format mismatch, not a capability gap",
                        "re-prove after the prompt change; it is free and "
                        "reversible, which no training run is",
                    ),
                    risks=(
                        "none worth the word — this is the reversible option, and "
                        "trying it first costs a day rather than a GPU budget",
                    ),
                )
            )
        else:
            recipes.append(_lora(claim.cluster, n, gap, receipt.incumbent))

    if receipt.unproven:
        notes.append(
            f"{len(receipt.unproven)} cluster(s) were never proven either way. "
            f"They are not in this plan — training against an unmeasured cluster "
            f"is spending money to move a number nobody has."
        )
    if receipt.judge_trustworthy is False:
        notes.append(
            "the judge behind this receipt was not trustworthy, so the gaps it "
            "identified may not be real. Fix the judge before training on its "
            "opinion."
        )
    return Recommendation(tuple(recipes), tuple(stay), tuple(notes))


def demo() -> None:
    """Self-check. Run with `python -m clickllm.posttrain`."""
    from clickllm.prove.equivalence import CandidateReport, ClusterScore
    from clickllm.prove.receipt import issue
    from clickllm.prove.stats import wilson

    def cs(name, passed, total, share=0.2):
        return ClusterScore(name, name, share, wilson(passed, total), 0)

    r = issue(
        CandidateReport(
            "glm-5.2",
            (
                cs("extract", 118, 120),  # passes
                # 86% over 1000 items: a *proven* small gap. With fewer samples
                # the interval straddles the bar and the cluster is unproven
                # rather than regressed — which is why this fixture is large.
                cs("format", 860, 1000),
                cs("summarise", 90, 120),  # 75% — a real gap
                cs("reason", 40, 120),  # 33% — a different capability
                cs("rare", 100, 140),  # a real gap, almost no captures
            ),
        ),
        incumbent="gpt-5",
        issued="2026-07-27",
        eval_set="a" * 64,
    )
    counts = {"format": 4000, "summarise": 3000, "reason": 5000, "rare": 40}
    rec = recommend(r, counts)
    by_cluster = {x.cluster: x for x in rec.recipes}

    # A small gap gets the free fix, not a GPU.
    assert by_cluster["format"].method is Method.PROMPT
    assert "reversible" in by_cluster["format"].risks[0]

    # A real gap with real data gets distillation from captured traffic.
    lora = by_cluster["summarise"]
    assert lora.method is Method.LORA
    assert "gpt-5 gave" in lora.source, lora.source
    assert any("SAME eval set" in s for s in lora.steps)
    assert any("not ground truth" in s for s in lora.risks)

    # A wide gap is refused, and the refusal explains itself.
    stay = dict(rec.stay)
    assert "reason" in [n for n in stay] or any("reason" in k for k in stay)
    assert any("different capability level" in v for v in stay.values())

    # Not enough data is a refusal too, with the number.
    assert any("40 captured examples" in v for v in stay.values()), stay

    # Nothing in here trains anything.
    assert not hasattr(rec, "run") and not hasattr(Recipe, "execute")
    assert rec.worth_training

    # A clean receipt recommends nothing at all.
    clean = issue(
        CandidateReport("glm-5.2", (cs("extract", 118, 120, 1.0),)),
        incumbent="gpt-5",
        issued="2026-07-27",
        eval_set="a" * 64,
    )
    empty = recommend(clean, {"extract": 5000})
    assert not empty.recipes and not empty.stay
    assert "Nothing to do" in empty.render()

    # A cluster that was never proven either way is excluded from the plan, and
    # the exclusion is stated — training against an unmeasured gap is spending
    # money to move a number nobody has.
    murky = issue(
        CandidateReport(
            "glm-5.2",
            (cs("summarise", 90, 120, 0.5), cs("thin", 8, 10, 0.5)),
        ),
        incumbent="gpt-5",
        issued="2026-07-27",
        eval_set="a" * 64,
    )
    assert [c.cluster for c in murky.unproven] == ["thin"]
    murky_rec = recommend(murky, {"summarise": 3000, "thin": 5000})
    assert any("never proven either way" in n for n in murky_rec.notes), murky_rec.notes
    assert "thin" not in {r.cluster for r in murky_rec.recipes}

    # An untrustworthy judge invalidates the premise of training.
    from clickllm.prove.judge import Agreement

    shaky = issue(
        CandidateReport("glm-5.2", (cs("summarise", 90, 120, 1.0),)),
        incumbent="gpt-5",
        issued="2026-07-27",
        eval_set="a" * 64,
        agreement=Agreement(agreed=10, total=20, model="j"),
    )
    assert any("Fix the judge" in n for n in recommend(shaky, {"summarise": 3000}).notes)

    # What to leave alone is rendered before what to train.
    text = rec.render()
    assert text.index("LEAVE ON THE INCUMBENT") < text.index("WORTH TRAINING")

    print("posttrain: ok")


if __name__ == "__main__":
    demo()
