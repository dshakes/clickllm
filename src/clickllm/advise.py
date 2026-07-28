"""What you did not ask about — and what production is telling you.

[`plan`] answers the question asked: given these requirements, what engine and
what settings. It says what it set and what it *cannot* meet. What it never says
is what you are leaving on the table, because nothing in the requirements asked.

That gap is where most of the money is. Somebody deploys an agent fleet with a
2,000-token system prompt on every request, never sets a prefix-sharing figure
because no form demanded one, and pays to recompute the same prefix a million
times. The plan they got was correct. It was also 40% more expensive than it
needed to be, and nothing told them.

## Two directions

**Forward** — [`suggest`] reads the requirements and the plan and returns what a
careful reviewer would raise unprompted: the knob nobody set, the headroom nobody
spent, the budget that is about to be missed by a margin worth knowing.

**Backward** — [`reconcile`] takes what production actually did and compares it
to what the plan assumed. A plan built for concurrency 8 that is serving 40 is
not a plan any more, and the failure mode is that nobody notices until latency
complaints arrive. This is the self-healing seam: it proposes the correction,
with the evidence, and stops there.

## The rule

**Suggestions are proposals with evidence and an expected effect. Never actions.**

Every suggestion carries the observation that triggered it, so a wrong one is
visible and dismissible rather than mysterious. Nothing here mutates a plan,
writes a config, or restarts anything — the same boundary the rest of the
product holds. An agent may raise all of this; a human applies it.

Effects are labelled as estimates, because they are. A suggestion that promised
a measured 40% and delivered 4% would cost more trust than staying quiet.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .plan import (
    MAX_SPEC_DECODE_CONCURRENCY,
    PREFIX_CACHING_FLOOR,
    Plan,
    Requirements,
    Setting,
    Workload,
)

__all__ = [
    "Impact",
    "Observed",
    "Suggestion",
    "reconcile",
    "suggest",
]

#: Below this, a context allocation is so far above what the workload plausibly
#: sends that the KV cache is being sized for traffic nobody generates.
OVERSIZED_CONTEXT = 65_536

#: Headroom fraction above which a better quantisation is worth proposing.
#: Deliberately generous: proposing an upgrade that then does not fit is worse
#: than staying quiet, because the user pays for the failed attempt in time.
COMFORTABLE_HEADROOM = 0.35

#: Concurrency this far above plan means the plan is describing a different
#: deployment. 1.5x rather than 1.05x — normal traffic varies, and a suggestion
#: that fires on noise gets filtered out mentally within a week.
CONCURRENCY_DRIFT = 1.5


class Impact(StrEnum):
    """How much a suggestion is worth, for ordering.

    Three levels, not five. A scale finer than the evidence supports invites
    arguing about whether something is a 3 or a 4 instead of acting on it.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_RANK = {Impact.HIGH: 0, Impact.MEDIUM: 1, Impact.LOW: 2}


@dataclass(frozen=True, slots=True)
class Suggestion:
    """One thing worth changing, why, and what it should buy."""

    #: Short handle, stable enough to suppress by name.
    id: str
    impact: Impact
    #: What to do, imperatively.
    action: str
    #: The observation that triggered this. Without it a suggestion is an
    #: assertion, and an assertion cannot be argued with.
    because: str
    #: What it should buy, always hedged — these are estimates, not measurements.
    expect: str
    #: The knob this concerns, when there is one.
    setting: Setting | None = None

    def render(self) -> str:
        head = f"[{self.impact.value}] {self.action}"
        return f"{head}\n    because: {self.because}\n    expect:  {self.expect}"


@dataclass(frozen=True, slots=True)
class Observed:
    """What production actually did.

    Every field is optional because telemetry arrives in pieces, and a
    reconciliation that needs the full set before saying anything would say
    nothing on the day it mattered.
    """

    concurrency: int | None = None
    #: Measured fraction of prompt tokens shared across requests.
    prefix_sharing: float | None = None
    #: Observed p95 time-to-first-token, milliseconds.
    ttft_ms: int | None = None
    #: Observed p95 inter-token latency, milliseconds.
    itl_ms: int | None = None
    #: Longest prompt actually seen, in tokens.
    peak_context: int | None = None
    #: Fraction of KV cache in use at peak, `0.0..=1.0`.
    kv_utilisation: float | None = None


def suggest(req: Requirements, p: Plan) -> list[Suggestion]:
    """What a careful reviewer would raise about this plan, unprompted.

    Ordered by impact, then by the order found — stable, so the same inputs
    produce the same list and a diff between two runs means something changed.

    Returns an empty list when there is nothing worth saying. Padding it with
    filler would train people to skim past the one that mattered.
    """
    out: list[Suggestion] = []

    # The single most under-used number in LLM serving. Nobody sets it because
    # no form asks, and the default assumes every request is unique.
    if req.prefix_sharing < PREFIX_CACHING_FLOOR and req.workload is not Workload.BATCH:
        out.append(
            Suggestion(
                id="measure-prefix-sharing",
                impact=Impact.HIGH,
                action="Measure prefix sharing before accepting this plan.",
                because=(
                    f"prefix_sharing is {req.prefix_sharing:.2f}, so prefix reuse is off. "
                    f"An agent or chat deployment with a fixed system prompt is commonly "
                    f"0.8+, and that prefix is being recomputed on every request."
                ),
                expect=(
                    "if sharing is real, a large cut in prefill cost — the saving scales "
                    "with the shared fraction. Estimate, not measured: measure it."
                ),
                setting=Setting.PREFIX_REUSE,
            )
        )

    # Context is the quietest way to overpay: KV scales linearly with it, and
    # nobody revisits the number they typed on day one.
    if req.context > OVERSIZED_CONTEXT and req.workload is Workload.INTERACTIVE:
        out.append(
            Suggestion(
                id="right-size-context",
                impact=Impact.MEDIUM,
                action=f"Check whether {req.context:,} tokens of context is really needed.",
                because=(
                    "KV cache scales linearly with context, and interactive traffic rarely "
                    "approaches a long ceiling. Serving a limit nobody reaches costs memory "
                    "that could hold concurrency instead."
                ),
                expect=(
                    "halving context roughly halves KV memory, which buys either more "
                    "concurrent requests or a better quantisation. Roofline, not measured."
                ),
                setting=Setting.CONTEXT_LENGTH,
            )
        )

    # Speculative decoding: the trap is that the published numbers are single-stream.
    spec = p.get(Setting.SPECULATIVE)
    if spec is not None and spec.value != "off" and req.concurrency > 8:
        out.append(
            Suggestion(
                id="verify-speculative-at-your-batch",
                impact=Impact.MEDIUM,
                action="Benchmark speculative decoding at your real concurrency before keeping it.",
                because=(
                    f"it is on at concurrency {req.concurrency}. Published EAGLE figures are "
                    f"single-stream; the win narrows as batch grows and published numbers "
                    f"turn negative around batch 32, which is why this plan switches it off "
                    f"above {MAX_SPEC_DECODE_CONCURRENCY} rather than at the crossover."
                ),
                expect=(
                    "either a confirmed latency win or a throughput regression you would "
                    "otherwise have shipped. Measure both, at batch, not at 1."
                ),
                setting=Setting.SPECULATIVE,
            )
        )

    # Spare memory is not a virtue. It is a quantisation you did not take.
    fit = p.fit
    if fit is not None and fit.feasible and fit.total_bytes > 0:
        headroom = fit.headroom_bytes / (fit.total_bytes + fit.headroom_bytes)
        if headroom > COMFORTABLE_HEADROOM:
            out.append(
                Suggestion(
                    id="spend-the-headroom",
                    impact=Impact.MEDIUM,
                    action="Try a higher-precision quantisation, or more context.",
                    because=(
                        f"{headroom * 100:.0f}% of usable memory is unallocated at this "
                        f"configuration. Idle VRAM buys nothing."
                    ),
                    expect=(
                        "better output quality at the same throughput, or more concurrency. "
                        "Re-run fit to confirm the next step up actually fits."
                    ),
                    setting=Setting.QUANTIZATION,
                )
            )

    # A latency budget nobody stated is a budget nobody can miss — and a
    # deployment nobody can tell is failing.
    if req.workload is not Workload.BATCH and req.ttft_ms is None and req.itl_ms is None:
        out.append(
            Suggestion(
                id="state-a-latency-budget",
                impact=Impact.LOW,
                action="State a TTFT or ITL budget.",
                because=(
                    f"{req.workload.value} traffic with no stated budget cannot be "
                    "checked against reality — the planner has nothing to warn about and "
                    "the guard has nothing to detect a regression from."
                ),
                expect="warnings when the hardware cannot meet the budget, instead of silence.",
            )
        )

    # Structured output the engine cannot actually enforce is the worst kind of
    # gap: the config looks right and the constraint silently is not applied.
    if req.structured_output:
        _, gaps = p.command("placeholder/model")
        if any("structured" in g.lower() or "grammar" in g.lower() for g in gaps):
            out.append(
                Suggestion(
                    id="structured-output-unenforced",
                    impact=Impact.HIGH,
                    action="Validate outputs yourself, or pick an engine with a verified backend.",
                    because=(
                        "structured output was required but this engine has no verified flag "
                        "for it, so the emitted command does not enforce the schema."
                    ),
                    expect="schema violations caught at your boundary rather than in production.",
                    setting=Setting.STRUCTURED_OUTPUT,
                )
            )

    return sorted(out, key=lambda s: _RANK[s.impact])


def reconcile(req: Requirements, p: Plan, seen: Observed) -> list[Suggestion]:
    """Compare what production did against what the plan assumed.

    This is the self-healing seam. It proposes; it never applies. A plan built
    for one deployment and serving another is the common, silent failure — and
    the only signal is usually a complaint.

    Returns an empty list when reality matches the plan, which is the answer
    most of the time and should read as reassuring rather than broken.
    """
    out: list[Suggestion] = []

    if seen.concurrency is not None and seen.concurrency > req.concurrency * CONCURRENCY_DRIFT:
        out.append(
            Suggestion(
                id="replan-for-observed-concurrency",
                impact=Impact.HIGH,
                action=f"Re-plan for concurrency {seen.concurrency}.",
                because=(
                    f"the plan assumes {req.concurrency} in-flight requests; production is "
                    f"running {seen.concurrency}. KV cache, batch limits and the speculative "
                    f"decision were all derived from the smaller number."
                ),
                expect=(
                    "settings that match the traffic. At this concurrency the current plan "
                    "is describing a different deployment."
                ),
                setting=Setting.MAX_CONCURRENT,
            )
        )

    if (
        seen.prefix_sharing is not None
        and seen.prefix_sharing >= PREFIX_CACHING_FLOOR
        and req.prefix_sharing < PREFIX_CACHING_FLOOR
    ):
        out.append(
            Suggestion(
                id="enable-prefix-reuse",
                impact=Impact.HIGH,
                action="Turn on prefix reuse and re-plan.",
                because=(
                    f"measured prefix sharing is {seen.prefix_sharing:.2f}, but the plan was "
                    f"built with {req.prefix_sharing:.2f} and left reuse off. That prefix is "
                    f"being recomputed on every request."
                ),
                expect=(
                    f"prefill work cut roughly in proportion to the shared fraction "
                    f"({seen.prefix_sharing * 100:.0f}%). Measure it."
                ),
                setting=Setting.PREFIX_REUSE,
            )
        )

    if req.ttft_ms is not None and seen.ttft_ms is not None and seen.ttft_ms > req.ttft_ms:
        out.append(
            Suggestion(
                id="ttft-budget-missed",
                impact=Impact.HIGH,
                action="Reduce context, raise prefill chunking, or accept a slower budget.",
                because=(
                    f"p95 time-to-first-token is {seen.ttft_ms} ms against a stated budget of "
                    f"{req.ttft_ms} ms."
                ),
                expect="a budget that is met, or a stated one that is honest about the hardware.",
                setting=Setting.PREFILL_CHUNK,
            )
        )

    if req.itl_ms is not None and seen.itl_ms is not None and seen.itl_ms > req.itl_ms:
        out.append(
            Suggestion(
                id="itl-budget-missed",
                impact=Impact.HIGH,
                action="Lower concurrency, or move to a smaller model or faster quantisation.",
                because=(
                    f"p95 inter-token latency is {seen.itl_ms} ms against a stated budget of "
                    f"{req.itl_ms} ms. Decode is memory-bound, so this is bytes per token, "
                    f"not compute."
                ),
                expect="tokens arriving inside the budget, at some cost in throughput.",
                setting=Setting.MAX_CONCURRENT,
            )
        )

    # Context sized for traffic that never arrives — the reverse of the usual bug,
    # and only visible once you have seen real prompts.
    if seen.peak_context is not None and seen.peak_context * 2 < req.context:
        out.append(
            Suggestion(
                id="context-never-used",
                impact=Impact.MEDIUM,
                action=f"Consider serving {seen.peak_context * 2:,} tokens instead of "
                f"{req.context:,}.",
                because=(
                    f"the longest prompt actually seen is {seen.peak_context:,} tokens, less "
                    f"than half what is provisioned. The rest is KV memory held for traffic "
                    f"that has not arrived."
                ),
                expect=(
                    "memory back for concurrency or quantisation. Keep a margin — this is "
                    "what was seen, not a ceiling."
                ),
                setting=Setting.CONTEXT_LENGTH,
            )
        )

    if seen.kv_utilisation is not None and seen.kv_utilisation > 0.9:
        out.append(
            Suggestion(
                id="kv-pressure",
                impact=Impact.HIGH,
                action="Reduce context or concurrency before the cache starts evicting.",
                because=(
                    f"KV cache is at {seen.kv_utilisation * 100:.0f}% at peak. Past full, the "
                    f"scheduler preempts requests, and the symptom is latency variance rather "
                    f"than an error anyone can grep for."
                ),
                expect="stable tail latency instead of intermittent preemption.",
                setting=Setting.MAX_CONCURRENT,
            )
        )

    return sorted(out, key=lambda s: _RANK[s.impact])


def demo() -> None:
    from .hardware import Hardware
    from .plan import plan

    hw = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * 1024**3,
        usable_bytes=96 * 1024**3,
        bandwidth_gbps=546.0,
        cores=16,
    )

    # An agent fleet described the way people actually describe one: concurrency
    # and context, and nothing about the system prompt they send every time.
    req = Requirements(workload=Workload.INTERACTIVE, concurrency=16, context=131_072)
    p = plan(hw, req)
    s = suggest(req, p)
    ids = [x.id for x in s]

    assert "measure-prefix-sharing" in ids, ids
    assert "right-size-context" in ids, ids
    assert "state-a-latency-budget" in ids, ids
    # Highest impact first, so the list can be truncated without losing the point.
    assert s[0].impact is Impact.HIGH
    assert [_RANK[x.impact] for x in s] == sorted(_RANK[x.impact] for x in s)
    # Every suggestion carries its evidence and hedges its effect.
    for x in s:
        assert x.because and x.expect, x

    # A well-specified deployment gets a short list, not a padded one.
    tight = Requirements(
        workload=Workload.INTERACTIVE,
        concurrency=4,
        context=8_192,
        ttft_ms=400,
        itl_ms=40,
        prefix_sharing=0.8,
    )
    quiet = suggest(tight, plan(hw, tight))
    assert "measure-prefix-sharing" not in {x.id for x in quiet}
    assert "state-a-latency-budget" not in {x.id for x in quiet}

    # Reality matching the plan says nothing at all.
    assert reconcile(tight, plan(hw, tight), Observed(concurrency=4)) == []

    # Reality diverging says exactly what diverged.
    drifted = reconcile(
        tight,
        plan(hw, tight),
        Observed(concurrency=40, ttft_ms=900, kv_utilisation=0.95, peak_context=2_000),
    )
    got = {x.id for x in drifted}
    assert "replan-for-observed-concurrency" in got, got
    assert "ttft-budget-missed" in got, got
    assert "kv-pressure" in got, got
    assert "context-never-used" in got, got

    # Normal variance does not fire: 5 against a plan of 4 is traffic, not drift.
    assert reconcile(tight, plan(hw, tight), Observed(concurrency=5)) == []

    # Measured sharing the plan never knew about is the money case.
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=4, context=8_192)
    found = reconcile(loose, plan(hw, loose), Observed(prefix_sharing=0.85))
    assert [x.id for x in found] == ["enable-prefix-reuse"], found

    print(f"advise: {len(s)} suggestions forward, {len(drifted)} from observed reality")
    for x in s:
        print(f"  {x.render()}")


if __name__ == "__main__":
    demo()
