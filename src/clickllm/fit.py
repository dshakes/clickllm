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


def recommend_runtime(hw: Hardware, context: int, concurrency: int) -> tuple[str, str]:
    """(runtime, why). Encodes the tradeoffs that aren't obvious from docs."""
    if hw.kind == "apple":
        if concurrency >= 4:
            return (
                "vllm-mlx",
                f"continuous batching on Metal; at concurrency {concurrency} aggregate "
                "throughput dominates, despite ~15% lower single-stream than llama.cpp",
            )
        if context >= 65536:
            return ("mlc-llm", "paged KV holds throughput steadiest at 64K+ context")
        return (
            "llama.cpp (Metal)",
            "best single-stream decode and widest model support; "
            "mlx-lm is the alternative if you want speculative decoding",
        )
    if hw.kind == "nvidia":
        if hw.devices > 1:
            return (
                "llm-d + GAIE",
                f"{hw.devices} GPUs: disaggregated prefill/decode with KV-cache-aware "
                "routing (measured 3x output tok/s, 2x TTFT reduction)",
            )
        if concurrency >= 8:
            return (
                "sglang",
                "RadixAttention reuses shared prefixes — the win case for agent traffic "
                "with a fixed system prompt",
            )
        return (
            "vllm",
            "broadest hardware and feature support; enable EAGLE-3 spec-decode",
        )
    return ("llama.cpp (CPU)", "no accelerator — expect single-digit tok/s")


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

    # Runtime choice must flip on workload shape, not just hardware.
    assert recommend_runtime(m4, 8192, 16)[0] == "vllm-mlx"
    assert recommend_runtime(m4, 8192, 1)[0] == "llama.cpp (Metal)"
    assert recommend_runtime(m4, 131072, 1)[0] == "mlc-llm"

    feasible, rejected = rank(m4, 32768, 1)
    assert feasible and rejected
    assert all(x.feasible for x in feasible)

    print(f"96 GB / 32K ctx / c=1  →  {len(feasible)} feasible, {len(rejected)} rejected")
    print(
        f"top: {feasible[0].model.name} @ {feasible[0].quant} "
        f"({feasible[0].total_bytes / GB:.0f} GB, ~{feasible[0].tokens_per_sec:.0f} tok/s)"
    )
    print("ok")


if __name__ == "__main__":
    demo()
