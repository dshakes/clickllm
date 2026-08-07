"""Fit solver: what actually runs on this machine, at this context, at this concurrency.

Every number here is auditable — `explain()` returns the arithmetic, because a
sizing recommendation you can't check won't be trusted with production.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import catalog
from .catalog import QUANT_BITS, ModelSpec
from .hardware import Hardware

GB = 1024**3
MB = 1024**2
KB = 1024

# Non-weight, non-KV memory: activations, workspace, allocator fragmentation.
# Scales with model size plus a fixed runtime floor.
OVERHEAD_FRACTION = 0.08
OVERHEAD_FLOOR = 1.5 * GB

# Fraction of theoretical bandwidth a real decode loop achieves.
# ponytail: single constant, split per-runtime if estimates drift from measurement.
BANDWIDTH_EFFICIENCY = 0.72


@dataclass(frozen=True, slots=True)
class Fit:
    model: ModelSpec
    quant: str
    context: int
    concurrency: int
    weight_bytes: int
    kv_bytes: int
    overhead_bytes: int
    usable_bytes: int
    tokens_per_sec: float | None
    #: Bytes/second the decode loop is assumed to actually achieve —
    #: `bandwidth x BANDWIDTH_EFFICIENCY`, carried rather than recovered.
    #:
    #: `aggregate_tokens_per_sec` used to rebuild this by inverting
    #: `tokens_per_sec`, which meant the read-per-token formula lived in two
    #: places, forwards in one and backwards in the other. Adding the KV term to
    #: `solve()` silently falsified the inverse, under-recovering bandwidth by up
    #: to 68% at 128k context and inflating `$/Mtok` roughly 3x — a bias against
    #: self-hosting, in the one number this product exists to get right. Both
    #: reviewers caught it; the fix is that the formula now has one home.
    effective_bw: float | None = None

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.overhead_bytes

    @property
    def feasible(self) -> bool:
        return self.total_bytes <= self.usable_bytes

    @property
    def headroom_bytes(self) -> int:
        return self.usable_bytes - self.total_bytes

    @property
    def aggregate_tokens_per_sec(self) -> float | None:
        """Tokens per second across ALL concurrent requests, not one of them.

        `tokens_per_sec` is single-stream and correctly labelled as such, but it
        was the only throughput figure in the codebase — so `$/Mtok` divided an
        hourly rate by one stream's output and read the same at concurrency 1 and
        concurrency 32. `clickllm host` then printed that beside hosted providers'
        per-token prices, which are batched. The comparison the whole product
        turns on was biased against self-hosting by roughly the batch factor.

        The model is the standard bandwidth account of a decode step. Weights are
        read once per forward pass however many sequences are in flight; each
        sequence additionally streams its own KV cache. So for batch B:

            step_seconds = (weight_read + B * kv_read) / effective_bandwidth
            aggregate    = B / step_seconds

        which scales almost linearly while the weight read dominates and
        saturates at `bandwidth / kv_read` once KV traffic does — the real shape,
        and it needs no invented ceiling constant to produce it.

        An estimate, like everything else here: it ignores prefill, scheduler
        overhead and compute-bound regimes at very large batch, all of which make
        the true number lower. Named `roofline` wherever it is printed.

        **For MoE, `weight_read` is the optimistic end of a range.** It is held
        at `active_b` for every batch size, which is exact only at B=1: each
        sequence routes to its own top-k experts, so the number of *distinct*
        experts touched in one forward pass grows with B and saturates toward
        the full set. The true read is somewhere in
        `[active_b, min(total_b, B * active_b)]`, and this takes the low end —
        which makes the aggregate figure high and `$/Mtok` low, the direction
        that flatters self-hosting. Both bounds are arithmetic; the curve
        between them depends on the router and the traffic, so it is disclosed
        rather than guessed at. `moe_batch_optimism` reports the spread, and
        `explain()` prints it. Measuring is the upgrade path, not a constant.
        """
        if self.effective_bw is None or self.concurrency < 1:
            return None
        weight_read = self.model.active_b * 1e9 * QUANT_BITS[self.quant] / 8
        if weight_read <= 0:
            return None
        kv_read = self.model.kv_bytes_per_token() * self.context
        return (self.concurrency * self.effective_bw) / (weight_read + self.concurrency * kv_read)

    @property
    def moe_batch_optimism(self) -> float | None:
        """How much the aggregate figure could be overstating an MoE at this batch.

        The ratio of the pessimistic weight read to the optimistic one that
        `aggregate_tokens_per_sec` actually uses. `1.0` means the two agree and
        the number carries no MoE routing risk — true for every dense model,
        and for MoE at concurrency 1. `None` when there is nothing to say.

        Not a correction factor. It is the width of the band the real answer
        sits in, so a reader can tell a figure that is solid from one that is
        the good end of a wide range.
        """
        if not self.model.is_moe or self.concurrency < 2:
            return None
        active, total = self.model.active_b, self.model.params_b
        if active <= 0 or total <= active:
            return None
        return min(total, self.concurrency * active) / active

    @property
    def slow(self) -> bool:
        """Fits, but you won't enjoy it."""
        return self.feasible and self.tokens_per_sec is not None and self.tokens_per_sec < 15

    def explain(self) -> str:
        m = self.model
        kv_per_tok = m.kv_bytes_per_token()
        lines = [
            f"{m.name} @ {self.quant}, {self.context:,} ctx, concurrency {self.concurrency}",
            "",
            f"  weights   {m.params_b:g}B params x {QUANT_BITS[self.quant]} bits / 8"
            f"           = {self.weight_bytes / GB:6.1f} GB"
            + ("   (MoE: ALL experts resident)" if m.is_moe else ""),
            f"  kv cache  {kv_per_tok:,} B/tok x {self.context:,} tok x {self.concurrency}"
            f"   = {self.kv_bytes / GB:6.1f} GB   ({m.kv_scheme.upper()})",
            f"  overhead  {OVERHEAD_FRACTION:.0%} of weights + {OVERHEAD_FLOOR / GB:.1f} GB floor"
            f"      = {self.overhead_bytes / GB:6.1f} GB",
            f"  {'-' * 62}",
            f"  {'required':<50}{self.total_bytes / GB:6.1f} GB",
            f"  {'usable':<50}{self.usable_bytes / GB:6.1f} GB",
            f"  {'headroom':<50}{self.headroom_bytes / GB:6.1f} GB",
        ]
        if self.tokens_per_sec:
            weights_read = m.active_b * 1e9 * QUANT_BITS[self.quant] / 8
            kv_read = m.kv_bytes_per_token() * self.context
            lines += [
                "",
                "  decode is bandwidth-bound, and reads both:",
                f"    weights  {weights_read / GB:5.1f} GB/token  ({m.active_b:g}B active)",
                f"    kv cache {kv_read / GB:5.1f} GB/token  (the whole cache, every token)",
                f"    total    {(weights_read + kv_read) / GB:5.1f} GB/token"
                f" at {BANDWIDTH_EFFICIENCY:.0%} of peak",
                f"  ~{self.tokens_per_sec:.0f} tok/s single-stream"
                "  (roofline estimate, not measured)",
            ]
            spread = self.moe_batch_optimism
            if spread is not None and spread > 1.0:
                lines += [
                    f"  ! MoE at batch {self.concurrency}: the weight read above is held at"
                    f" {m.active_b:g}B, which is exact only at batch 1.",
                    f"    Sequences route to different experts, so the real read is between"
                    f" {m.active_b:g}B and {min(m.params_b, self.concurrency * m.active_b):g}B"
                    f" — up to {spread:.1f}x.",
                    "    The aggregate throughput and $/Mtok figures take the low end, so"
                    " they flatter rather than warn.",
                ]
        if not m.verified:
            lines += ["", "  ! architecture unverified — KV figures are estimates"]
        return "\n".join(lines)


def _overhead(weight_bytes: int) -> int:
    return int(weight_bytes * OVERHEAD_FRACTION + OVERHEAD_FLOOR)


def solve(model: ModelSpec, quant: str, hw: Hardware, context: int, concurrency: int) -> Fit:
    """Size one model on one machine. Validates here, not at the surface.

    The same two bounds were enforced in `cli.py`'s argument parsing and in
    `sdk.fit()`, and NOT in `sdk.explain()` or the MCP tools — three doors into
    one calculation, guarded on two. Both unguarded doors could report a model
    that does not fit as FEASIBLE, with flattering headroom, because a
    non-positive context or concurrency makes the KV term vanish.

    A guard at the surface has to be re-earned by every new entry point. This
    one is at the solver, so there is one door. See ADR-0011.

    Raises:
        ValueError: context or concurrency below 1, naming the offending value.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if context < 1:
        raise ValueError(f"context must be >= 1, got {context}")
    w = model.weight_bytes(quant)
    kv = model.kv_bytes_per_token() * context * concurrency
    tps = None
    effective_bw = None
    if hw.bandwidth_gbps:
        # Decode reads the weights AND the KV cache, every token — attention
        # cannot run without the cache. This counted weights only, which is
        # short by kv_bytes_per_token x context and therefore *over*-predicted,
        # worst exactly where users push context hardest. On an M4 Max serving
        # Llama 3.1 8B q8 at 8k that omission is ~12% of the traffic.
        #
        # Full context rather than half: a decode loop at its configured ceiling
        # is the case worth quoting, and over-predicting throughput is how
        # someone buys hardware that cannot reach the number.
        #
        # Single-stream, matching the label this figure is printed under — the
        # batched case amortises weights across the batch and needs its own
        # derivation, not a factor bolted on here.
        read_per_token = model.active_b * 1e9 * QUANT_BITS[quant] / 8
        read_per_token += model.kv_bytes_per_token() * context
        effective_bw = hw.bandwidth_gbps * 1e9 * BANDWIDTH_EFFICIENCY
        tps = effective_bw / read_per_token
    return Fit(
        model, quant, context, concurrency, w, kv, _overhead(w), hw.usable_bytes, tps, effective_bw
    )


def max_context(model: ModelSpec, quant: str, hw: Hardware, concurrency: int = 1) -> int:
    """Largest context that fits, capped at the model's own limit."""
    w = model.weight_bytes(quant)
    spare = hw.usable_bytes - w - _overhead(w)
    if spare <= 0:
        return 0
    # `concurrency=0` divided by zero. The CLI clamps before it gets here, so
    # this was only reachable from the library and the SDK — which is exactly
    # who would hit it, since a caller computing a ceiling is likelier to pass a
    # loop variable than a validated flag.
    per_tok = model.kv_bytes_per_token() * max(1, concurrency)
    return min(model.max_context, int(spare / per_tok))


def max_concurrency(model: ModelSpec, quant: str, hw: Hardware, context: int) -> int:
    w = model.weight_bytes(quant)
    spare = hw.usable_bytes - w - _overhead(w)
    if spare <= 0:
        return 0
    # Same guard as `max_context`: `context=0` divided by zero.
    return int(spare / (model.kv_bytes_per_token() * max(1, context)))


def quant_preference(quants: tuple[str, ...]) -> list[str]:
    """Highest precision that fits, capped at 8-bit.

    Decode is bandwidth-bound, so fp16 halves tokens/sec versus 8-bit while
    buying quality that is negligible at inference time. Only reach above 8-bit
    when a model publishes nothing at or below it.
    """
    at_or_below_8 = [q for q in quants if QUANT_BITS[q] <= 8]
    pool = at_or_below_8 or list(quants)
    return sorted(pool, key=lambda q: -QUANT_BITS[q])


def best_quant(model: ModelSpec, hw: Hardware, context: int, concurrency: int) -> Fit | None:
    """Best quality-per-unit-bandwidth quant that fits."""
    for q in quant_preference(model.quants):
        fit = solve(model, q, hw, context, concurrency)
        if fit.feasible:
            return fit
    return None


def rank(
    hw: Hardware, context: int, concurrency: int
) -> tuple[list[Fit], list[tuple[ModelSpec, str]]]:
    """Returns (feasible fits, [(model, why-not) for the rest]).

    Feasible sorted by capability proxy (total params) descending — the biggest
    model you can run is usually the best one you can run.
    """
    feasible: list[Fit] = []
    rejected: list[tuple[ModelSpec, str]] = []

    for m in catalog.load():
        fit = best_quant(m, hw, context, concurrency)
        if fit:
            feasible.append(fit)
            continue
        smallest = min(m.quants, key=lambda q: QUANT_BITS[q])
        f = solve(m, smallest, hw, context, concurrency)
        if f.weight_bytes > hw.usable_bytes:
            why = f"weights alone need {f.weight_bytes / GB:,.0f} GB at {smallest}" + (
                f" — MoE sparsity ({m.active_b:g}B of {m.params_b:g}B active) "
                "cuts compute, not memory"
                if m.is_moe
                else ""
            )
        else:
            # The OVERAGE, not the whole KV figure. This printed `kv_bytes`
            # under the words "more than available", which reads as the
            # shortfall and is not: for qwen3-32b @ q4 on a 36 GB budget it
            # said "needs 32 GB more" when the real overage was 16 GB — 2x,
            # enough to make someone dismiss a machine that nearly fits. KV is
            # still named because it is usually the dominant term, but overhead
            # is in the total too and the number now says what it means.
            over = (f.total_bytes - f.usable_bytes) / GB
            why = (
                f"weights fit ({f.weight_bytes / GB:.0f} GB) but KV at {context:,} ctx "
                f"x{concurrency} ({f.kv_bytes / GB:.0f} GB) puts it "
                f"{over:.0f} GB over the {f.usable_bytes / GB:.0f} GB available"
            )
        rejected.append((m, why))

    feasible.sort(key=lambda f: -f.model.params_b)
    return feasible, rejected


@dataclass(frozen=True, slots=True)
class Placement:
    """Whether one hardware profile can serve one model, and how well."""

    profile_id: str
    profile_name: str
    fit: Fit | None
    reason: str
    hourly_usd: float | None

    @property
    def feasible(self) -> bool:
        return self.fit is not None

    @property
    def tokens_per_sec(self) -> float | None:
        return self.fit.tokens_per_sec if self.fit else None

    @property
    def moe_batch_optimism(self) -> float | None:
        """Carried through from the fit, because this is where the number is read.

        The caveat first landed only in `Fit.explain()`, which is a command
        someone runs deliberately. `cost_per_mtok_usd` below — and the aggregate
        throughput it divides — are what `clickllm where`, `clickllm host` and
        the workbench put in front of people who never ask for the arithmetic. A
        disclosure attached to the explanation and not to the figure is a
        disclosure the reader of the figure does not get.
        """
        return self.fit.moe_batch_optimism if self.fit else None

    @property
    def cost_per_mtok_usd(self) -> float | None:
        """Rough USD per million output tokens at the requested concurrency.

        A capacity-vs-speed reality check: a cheap card that decodes slowly can
        cost more per token than an expensive one that decodes fast.

        Uses AGGREGATE throughput, not single-stream. It used to divide by one
        stream's output, so the figure was identical at concurrency 1 and 32 and
        overstated self-hosting by roughly the batch factor — while `clickllm
        host` printed it beside hosted providers' prices, which are batched. The
        old docstring said "real cost is higher", which was true of idle time and
        wrong by more in the other direction.

        Still an estimate, and still assumes the machine is busy: a box at 10%
        utilisation costs ten times this per token.
        """
        if self.hourly_usd is None or self.fit is None:
            return None
        agg = self.fit.aggregate_tokens_per_sec
        if not agg:
            return None
        return (self.hourly_usd / (agg * 3600)) * 1_000_000


def _shortfall(n: int) -> str:
    """A shortfall, in a unit that never renders as zero.

    `f"{short_by / GB:,.0f} GB"` produced "short by 0 GB" for any deficit under
    half a gigabyte — a refusal claiming to be short by nothing, which reads as a
    bug in the solver rather than as the true answer (Qwen3-32B at q8 misses
    batch 19 by 0.49 GiB). The unit steps down so the number stays meaningful.
    """
    if n >= 10 * GB:
        return f"{n / GB:,.0f} GB"
    if n >= GB:
        return f"{n / GB:,.1f} GB"
    if n >= 10 * MB:
        return f"{n / MB:,.0f} MB"
    if n >= MB:
        return f"{n / MB:,.1f} MB"
    # The MB branch had the same bug one order down: anything under half a
    # mebibyte rendered "0 MB". That is reachable — a GQA model's KV is a few
    # hundred KB per token, so a shortfall at small context lands here — and a
    # refusal short by "0 MB" is the exact sentence this function was written
    # to delete. Bytes are the floor because a shortfall is a whole number of
    # them and cannot round to nothing.
    if n >= 10 * KB:
        return f"{n / KB:,.0f} KB"
    if n >= KB:
        return f"{n / KB:,.1f} KB"
    return f"{n:,} B"


def where(model: ModelSpec, context: int, concurrency: int = 1) -> list[Placement]:
    """Which hardware classes can serve `model`, cheapest capable first.

    The inverse of `rank`: instead of "what runs on my box", this answers "what
    box do I need". Infeasible profiles are returned too, with the reason — the
    gap between "needs 2 GB more" and "needs 10x more" is the difference between
    a config change and a different purchase.
    """
    from .hardware_catalog import PROFILES

    out: list[Placement] = []
    for p in PROFILES:
        hw = p.to_hardware()
        f = best_quant(model, hw, context, concurrency)
        if f:
            out.append(Placement(p.id, p.name, f, "", p.hourly_usd))
            continue
        smallest = min(model.quants, key=lambda q: QUANT_BITS[q])
        probe = solve(model, smallest, hw, context, concurrency)
        short_by = probe.total_bytes - hw.usable_bytes
        if probe.weight_bytes > hw.usable_bytes:
            why = (
                f"weights alone need {probe.weight_bytes / GB:,.0f} GB at {smallest}, "
                f"{hw.usable_gb:,.0f} GB usable"
            )
        else:
            why = (
                f"short by {_shortfall(short_by)} — weights fit, but KV at "
                f"{context:,} ctx x{concurrency} does not"
            )
        out.append(Placement(p.id, p.name, None, why, p.hourly_usd))

    # Feasible first, then by price where known, then by throughput.
    out.sort(
        key=lambda pl: (
            not pl.feasible,
            pl.hourly_usd if pl.hourly_usd is not None else float("inf"),
            -(pl.tokens_per_sec or 0.0),
        )
    )
    return out


def recommend_runtime(hw: Hardware, context: int, concurrency: int) -> tuple[str, str]:
    """(engine, why) — the engine `clickllm run` would actually start here.

    This used to carry its own table, and the two disagreed. On Apple silicon at
    concurrency 8, `clickllm fit` printed `runtime -> vllm-mlx` while `clickllm
    run` on the same box started `mlx` and explained that "the CUDA engines
    cannot run here at all" — refuting the first command's advice in the second
    command's output.

    Worse, the table named engines this tool cannot launch. `vllm-mlx` is not a
    real project name — the vLLM Apple-silicon plugin is `vllm-metal`, which
    runs vLLM on MLX and is worth an adapter — and `mlc-llm`, `llama.cpp (Metal)`
    and
    `llm-d + GAIE` have no adapter, so nothing in this codebase can configure or
    launch any of them. A first-touch command was recommending software a user
    cannot install, on the CLI, the MCP server and the SDK alike.

    So there is one selector now, and it is the one that has to be right because
    it produces the command: `plan._pick_engine`. A recommendation that cannot
    become a running server is not a recommendation.

    When the structural pick has no adapter, the alternative is found by asking
    `launch._launchable_alternative` — the exact function `clickllm run`'s own
    refusal uses to word its `NEXT` step — rather than a second search over
    concurrency. Substituting an engine that only appears at a *higher*
    concurrency must say so: at the concurrency actually asked for, nothing here
    can be launched at all, and hiding that behind "recommending what will
    actually start" was its own way of naming software `run` refuses.
    """
    # Deferred: `plan` and `launch` both import this module, so top-level
    # imports would cycle.
    from .engines import adapter_for
    from .launch import _launchable_alternative
    from .plan import Requirements, Workload, _pick_engine

    req = Requirements(Workload.INTERACTIVE, concurrency, context)
    engine, why = _pick_engine(hw, req)
    name = str(engine)
    if hw.kind == "cpu":
        why += " No accelerator on this machine — expect single-digit tok/s."
    if adapter_for(name) is not None:
        return (name, why)

    # `_pick_engine` answers "what is structurally best here", which is not
    # always "what this tool can launch". On Apple silicon at the DEFAULT
    # concurrency it picks llama.cpp — correct, and `clickllm run` then refuses
    # it for want of a verified flag dialect. Recommending it in silence made
    # the product's first two commands contradict each other on the most common
    # developer machine.
    alt = _launchable_alternative(hw, req)
    if alt is None:
        return (name, f"{why} NOTE: clickllm cannot launch {name} — no verified flag dialect.")
    alt_concurrency, alt_name = alt
    _, alt_why = _pick_engine(hw, Requirements(Workload.INTERACTIVE, alt_concurrency, context))
    return (
        alt_name,
        f"{alt_why} (Structurally {name} suits concurrency {concurrency} better, but "
        f"clickllm has no verified flag dialect for it, so `clickllm run` would refuse "
        f"here — nothing launches at concurrency {concurrency} on this hardware. "
        f"`clickllm run --concurrency {alt_concurrency}` is what starts {alt_name}; "
        f"that is a higher concurrency than asked for, not a free substitution.)",
    )


def demo() -> None:
    from .hardware import Hardware as HW

    m4 = HW(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=96 * GB,
        bandwidth_gbps=546,
        cores=16,
    )

    q3 = catalog.get("qwen3-32b")
    f = solve(q3, "q4", m4, 32768, 1)
    assert f.feasible
    # NOT `total == weights + kv + overhead`. That re-derives `total_bytes` from
    # the exact expression the property is defined with, so it cannot fail for
    # any input and would still pass with all three components wrong. It looked
    # like a sizing check and verified that Python adds the same way twice.
    #
    # Re-deriving each component from its own formula has the same defect one
    # level down, so these are pinned literals, observed once and written here.
    # They fail if the catalogue entry, the quantisation table or the arithmetic
    # moves — which is the whole point.
    #
    # Note 18.45 GB rather than 16.4: `QUANT_BITS["q4"]` is 4.5, not 4, because
    # a 4-bit checkpoint still carries scales. Writing this check is what
    # surfaced that; the tautological version could not have.
    assert f.weight_bytes == 18_449_999_999, f.weight_bytes
    assert f.kv_bytes == 8_589_934_592, f.kv_bytes  # 262,144 B/tok x 32,768
    assert f.overhead_bytes == 3_086_612_735, f.overhead_bytes
    assert f.total_bytes == 30_126_547_326, f.total_bytes
    assert f.headroom_bytes == f.usable_bytes - f.total_bytes

    # KV must scale linearly in both context and concurrency.
    a = solve(q3, "q4", m4, 8192, 1)
    b = solve(q3, "q4", m4, 16384, 1)
    c = solve(q3, "q4", m4, 8192, 2)
    assert b.kv_bytes == 2 * a.kv_bytes
    assert c.kv_bytes == 2 * a.kv_bytes

    # A 2.8T model cannot fit in 96 GB at any quant.
    assert best_quant(catalog.get("kimi-k3"), m4, 8192, 1) is None

    # Prefer precision when there's room, but cap at 8-bit: fp16 halves tok/s
    # for no meaningful inference-quality gain.
    small = best_quant(catalog.get("phi-4-14b"), m4, 8192, 1)
    assert small and small.quant == "q8", small.quant if small else None
    assert quant_preference(("fp16", "q8", "q4")) == ["q8", "q4"]
    assert quant_preference(("fp16",)) == ["fp16"], "16-bit only when it's all there is"

    # Sparse MoE decodes faster than a dense model 1/7th its total size.
    moe = solve(catalog.get("qwen3-30b-a3b"), "q4", m4, 8192, 1)
    assert moe.tokens_per_sec > f.tokens_per_sec, "3.3B active must beat 32.8B dense"

    # More concurrency lowers max context, and vice versa.
    assert max_context(q3, "q4", m4, 1) > max_context(q3, "q4", m4, 8)
    assert max_concurrency(q3, "q4", m4, 4096) > max_concurrency(q3, "q4", m4, 32768)
    assert max_context(q3, "q4", m4, 1) <= q3.max_context, "must respect model's own limit"

    # The engine named here must be the engine `run` starts, and must be one we
    # can actually configure. The old table asserted "vllm-mlx" and "mlc-llm",
    # neither of which exists — the assertions kept the defect in place.
    from .engines import adapter_for
    from .plan import Requirements, Workload, _pick_engine

    for conc, ctx in ((16, 8192), (1, 8192), (1, 131072)):
        named, why = recommend_runtime(m4, ctx, conc)
        # Never name an engine `run` cannot start — that was the defect.
        assert adapter_for(named) is not None, f"recommended {named}, which has no adapter"
        best, _ = _pick_engine(m4, Requirements(Workload.INTERACTIVE, conc, ctx))
        if adapter_for(str(best)) is not None:
            assert named == str(best), f"fit says {named}, run starts {best}"
        else:  # only when the structural pick has no dialect — rare, and shrinking
            assert str(best) in why, why

    # A substitution is not honest just because `_pick_engine` agrees with it —
    # the structural pick and its launchable substitute can need *different*
    # concurrencies, which `_pick_engine` alone cannot see. That regression was
    # real and is now nearly unreachable: llama.cpp has a dialect, so Apple at
    # concurrency 1 launches exactly what it recommends. The invariant is what
    # is checked, not the substitution that used to satisfy it — what `fit`
    # names must be what `run` starts, at the concurrency asked for.
    import tempfile
    from pathlib import Path

    from . import launch as _launch

    named, why = recommend_runtime(m4, 8192, 1)
    assert adapter_for(named) is not None, f"recommended {named}, which cannot launch"
    with tempfile.TemporaryDirectory() as tmp:
        started = _launch.plan(
            "llama-3.1-8b",
            hw=m4,
            context=8192,
            concurrency=1,
            exists=lambda r: True,
            cache=Path(tmp) / "weights.json",
        )
        if isinstance(started, _launch.LaunchPlan):
            assert started.engine == named, f"fit says {named}, run starts {started.engine}"
        else:
            # Still refusing: then the recommendation had to have said so.
            assert named in why or "would refuse" in why, why

    # A CPU-only box must still be told something it can launch.
    cpu = HW(
        kind="cpu",
        name="CPU only",
        total_bytes=64 * GB,
        usable_bytes=48 * GB,
        bandwidth_gbps=50.0,
        cores=16,
    )
    cpu_named, cpu_why = recommend_runtime(cpu, 8192, 1)
    assert adapter_for(cpu_named) is not None, f"cpu got {cpu_named}, which cannot launch"
    assert cpu_why

    feasible, rejected = rank(m4, 32768, 1)
    assert feasible and rejected
    assert all(x.feasible for x in feasible)

    # Inverse question: which machines run this model at all?
    placements = where(catalog.get("qwen3-32b"), 32768, 1)
    ok_pl = [p for p in placements if p.feasible]
    assert ok_pl, "a 32B model must fit something in the catalogue"
    assert not placements[-1].feasible, "infeasible profiles sort last"
    assert all(p.reason for p in placements if not p.feasible), "every rejection needs a reason"
    priced = [p for p in ok_pl if p.hourly_usd is not None]
    assert priced == sorted(priced, key=lambda p: p.hourly_usd)
    huge = where(catalog.get("kimi-k3"), 8192, 1)
    assert not any(p.feasible for p in huge)
    assert all("weights alone" in p.reason for p in huge)
    with_cost = [p for p in ok_pl if p.cost_per_mtok_usd is not None]
    assert with_cost and all(p.cost_per_mtok_usd > 0 for p in with_cost)

    print(f"96 GB / 32K ctx / c=1  →  {len(feasible)} feasible, {len(rejected)} rejected")
    print(f"qwen3-32b runs on {len(ok_pl)} of {len(placements)} hardware profiles")
    print(
        f"top: {feasible[0].model.name} @ {feasible[0].quant} "
        f"({feasible[0].total_bytes / GB:.0f} GB, ~{feasible[0].tokens_per_sec:.0f} tok/s)"
    )
    print("ok")


if __name__ == "__main__":
    demo()
