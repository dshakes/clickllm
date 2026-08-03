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
        """
        if self.tokens_per_sec is None or self.concurrency < 1:
            return None
        weight_read = self.model.active_b * 1e9 * QUANT_BITS[self.quant] / 8
        if weight_read <= 0:
            return None
        # `tokens_per_sec` already carries bandwidth x efficiency / weight_read,
        # so recover the effective bandwidth rather than re-deriving it from the
        # Hardware, which this dataclass does not keep.
        effective_bw = self.tokens_per_sec * weight_read
        kv_read = self.model.kv_bytes_per_token() * self.context
        return (self.concurrency * effective_bw) / (weight_read + self.concurrency * kv_read)

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
            read = m.active_b * 1e9 * QUANT_BITS[self.quant] / 8
            lines += [
                "",
                f"  decode is bandwidth-bound: {read / GB:.1f} GB read/token"
                f" ({m.active_b:g}B active) at {BANDWIDTH_EFFICIENCY:.0%} of peak",
                f"  ~{self.tokens_per_sec:.0f} tok/s single-stream"
                "  (roofline estimate, not measured)",
            ]
        if not m.verified:
            lines += ["", "  ! architecture unverified — KV figures are estimates"]
        return "\n".join(lines)


def _overhead(weight_bytes: int) -> int:
    return int(weight_bytes * OVERHEAD_FRACTION + OVERHEAD_FLOOR)


def solve(model: ModelSpec, quant: str, hw: Hardware, context: int, concurrency: int) -> Fit:
    w = model.weight_bytes(quant)
    kv = model.kv_bytes_per_token() * context * concurrency
    tps = None
    if hw.bandwidth_gbps:
        read_per_token = model.active_b * 1e9 * QUANT_BITS[quant] / 8
        tps = (hw.bandwidth_gbps * 1e9 * BANDWIDTH_EFFICIENCY) / read_per_token
    return Fit(model, quant, context, concurrency, w, kv, _overhead(w), hw.usable_bytes, tps)


def max_context(model: ModelSpec, quant: str, hw: Hardware, concurrency: int = 1) -> int:
    """Largest context that fits, capped at the model's own limit."""
    w = model.weight_bytes(quant)
    spare = hw.usable_bytes - w - _overhead(w)
    if spare <= 0:
        return 0
    per_tok = model.kv_bytes_per_token() * concurrency
    return min(model.max_context, int(spare / per_tok))


def max_concurrency(model: ModelSpec, quant: str, hw: Hardware, context: int) -> int:
    w = model.weight_bytes(quant)
    spare = hw.usable_bytes - w - _overhead(w)
    if spare <= 0:
        return 0
    return int(spare / (model.kv_bytes_per_token() * context))


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
            why = (
                f"weights fit ({f.weight_bytes / GB:.0f} GB) but KV at {context:,} ctx "
                f"x{concurrency} needs {f.kv_bytes / GB:.0f} GB more than available"
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
    return f"{n / (1024**2):,.0f} MB"


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

    Worse, the table named engines that do not exist: there is no `vllm-mlx`
    (vLLM has no Metal backend at all), and `mlc-llm`, `llama.cpp (Metal)` and
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
    assert f.total_bytes == f.weight_bytes + f.kv_bytes + f.overhead_bytes

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
        else:
            assert str(best) in why, f"substituted {named} for {best} without saying so"

    # A substitution is not honest just because `_pick_engine` agrees with it —
    # the structural pick and its launchable substitute can need *different*
    # concurrencies, which `_pick_engine` alone cannot see. Checked here against
    # the real launch path: at concurrency 1, `run` must still refuse, and the
    # recommendation must have said so rather than implying `mlx` starts now.
    import tempfile
    from pathlib import Path

    from . import launch as _launch

    named, why = recommend_runtime(m4, 8192, 1)
    assert named == "mlx" and "nothing launches at concurrency 1" in why, why
    with tempfile.TemporaryDirectory() as tmp:
        weights_cache = Path(tmp) / "weights.json"
        refused = _launch.plan(
            "qwen3-30b-a3b",
            hw=m4,
            context=8192,
            concurrency=1,
            exists=lambda r: True,
            cache=weights_cache,
        )
        assert isinstance(refused, _launch.Refusal), refused
        started = _launch.plan(
            "qwen3-30b-a3b",
            hw=m4,
            context=8192,
            concurrency=4,
            exists=lambda r: True,
            cache=weights_cache,
        )
        assert isinstance(started, _launch.LaunchPlan) and started.engine == named, started

    # CPU-only hardware lost its explicit warning when the old table was
    # replaced by `_pick_engine` — restored here rather than in the shared
    # selector, since the rest of `plan` has no CPU-specific reasoning to give.
    cpu = HW(
        kind="cpu",
        name="CPU box",
        total_bytes=64 * GB,
        usable_bytes=48 * GB,
        bandwidth_gbps=50.0,
        cores=8,
    )
    cpu_named, cpu_why = recommend_runtime(cpu, 8192, 1)
    assert adapter_for(cpu_named) is not None
    assert "no accelerator" in cpu_why.lower(), cpu_why

    feasible, rejected = rank(m4, 32768, 1)
    assert feasible and rejected
    assert all(x.feasible for x in feasible)

    # Inverse question: which machines run this model at all?
    placements = where(catalog.get("qwen3-32b"), 32768, 1)
    ok_pl = [p for p in placements if p.feasible]
    assert ok_pl, "a 32B model must fit something in the catalogue"
    assert not placements[-1].feasible, "infeasible profiles sort last"
    assert all(p.reason for p in placements if not p.feasible), "every rejection needs a reason"
    # Cheapest capable first among the feasible.
    priced = [p for p in ok_pl if p.hourly_usd is not None]
    assert priced == sorted(priced, key=lambda p: p.hourly_usd)
    # A 2.8T model fits nothing here, and says why rather than silently vanishing.
    huge = where(catalog.get("kimi-k3"), 8192, 1)
    assert not any(p.feasible for p in huge)
    assert all("weights alone" in p.reason for p in huge)
    # Cost per token rewards bandwidth, not just capacity.
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
