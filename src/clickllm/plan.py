"""The planner — from what you need to the flags that deliver it.

`fit` answers *does this model fit*. This answers the harder question that
follows: **given what you are actually doing with it, which engine and which
settings?** Those are not the same question, and the second one is where most
open-model deployments quietly lose half their hardware.

The reason it is not one question: a serving stack tuned for throughput and a
stack tuned for tail latency are configured in *opposite* directions on almost
every knob, and the defaults ship somewhere in between. Batch scoring wants the
largest batch the memory allows; a voice agent wants the batch capped so a long
prefill cannot land in front of a token that is due in 40ms. Both are "running
vLLM". Only one of them is running it correctly.

## What the planner refuses to do

**Emit a config that cannot meet the requirement.** If the roofline says the
inter-token budget is unreachable on this hardware, the plan says so in
[`Plan.warnings`] rather than shipping flags that look authoritative. A tuner
that always produces an answer is a tuner you cannot trust when it matters.

**Hide its reasoning.** Every knob carries the sentence that produced it — see
[`Knob.why`]. A number you cannot argue with is a number you cannot fix, and the
defaults here are calibration, not physics.

## The three forks that decide almost everything

1. **Prefix sharing** picks the engine. When many requests share a long prefix —
   agent loops with a fixed system prompt, few-shot templates, RAG with a stable
   preamble — SGLang's RadixAttention reuses that prefix's KV across requests
   instead of recomputing it. That is a structural win no vLLM flag reproduces.
   Below that, vLLM's PagedAttention is the better-supported default.

2. **Concurrency** decides speculative decoding. Drafting is a bet that spare
   compute exists; at high batch there is none, the verify step competes with
   real tokens, and measured throughput goes *down*. It is a function of load,
   never a checkbox.

3. **Latency budget vs throughput** decides scheduling. Chunked prefill splits a
   long prompt so it cannot monopolise a step — it costs a little throughput and
   protects time-to-first-token. Batch work wants the opposite.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from clickllm.catalog import ModelSpec
from clickllm.engines import MTP_NATIVE, LoraFleet, Setting
from clickllm.fit import Fit, solve
from clickllm.hardware import Hardware

__all__ = [
    "MAX_SPEC_DECODE_CONCURRENCY",
    "PREFIX_SHARING_FOR_RADIX",
    "Engine",
    "Knob",
    "Plan",
    "Requirements",
    "Workload",
    "demo",
    "plan",
]

# --- calibration knobs ---------------------------------------------------------
# Judgement calls with a stated basis, not constants of nature. Each is the
# threshold at which the *sign* of a tradeoff flips.

#: Above this concurrency, speculative decoding costs more than it saves: the
#: draft-and-verify step competes for compute that a full batch is already using.
#: Published EAGLE-3 numbers turn negative somewhere around batch 32; 16 leaves
#: room for the fact that a real workload's concurrency is a distribution, not a
#: number.
MAX_SPEC_DECODE_CONCURRENCY = 16

#: Fraction of prompt tokens shared across requests at which RadixAttention's
#: prefix reuse outweighs vLLM's broader ecosystem support.
PREFIX_SHARING_FOR_RADIX = 0.55

#: Prefix sharing below which prefix caching is not worth its bookkeeping.
PREFIX_CACHING_FLOOR = 0.15

#: Memory left unallocated for activation spikes and fragmentation. The engine's
#: own default (0.92 in current vLLM) is a guess made without knowing the model;
#: this margin is applied to a figure the solver actually computed.
MEMORY_SAFETY_MARGIN = 0.08

#: Prefill chunk size, in tokens. Small enough that one long prompt cannot
#: monopolise a scheduler step, large enough not to shred prefill throughput.
PREFILL_CHUNK_TOKENS = 2048

#: KV cache fraction above which fp8 is worth its accuracy question.
#:
#: Below this, KV is not the binding constraint and halving it buys nothing that
#: matters — so the accuracy risk is taken for no reason. This is the threshold
#: at which the trade becomes worth *asking about*, not the point at which it is
#: automatically correct.
KV_PRESSURE_FOR_FP8 = 0.45

#: TPU generations vLLM lists as recommended rather than experimental.
#: Verified 2026-07-27 against docs.vllm.ai/projects/tpu — v3, v4 and v5p are
#: experimental, which is a materially different promise and worth surfacing.
TPU_RECOMMENDED = ("v5e", "v6e", "v7x")

#: Model traits vLLM's TPU backend lists as still maturing. Not "broken" — but a
#: DeepSeek-family model is MLA, and shipping someone onto a TPU without saying
#: so would be selling a maturity level that has not been claimed.
TPU_MATURING = ("MLA attention", "XL MoE", "vision encoders")


class Workload(StrEnum):
    """What the deployment is for.

    This is the input that matters most and the one every tuning guide assumes
    you already decided. The knobs below move in opposite directions across it.
    """

    #: Chat, agents, copilots. Time-to-first-token dominates the felt experience.
    INTERACTIVE = "interactive"
    #: Voice, autocomplete, anything with a hard deadline per token. Tail latency
    #: is the requirement; mean latency is not the thing being bought.
    REALTIME = "realtime"
    #: Offline scoring, backfills, evals. Nobody is waiting; tokens per second
    #: per dollar is the only number.
    BATCH = "batch"


class Engine(StrEnum):
    """Serving stack."""

    VLLM = "vllm"
    #: vLLM's TPU backend, which is a different engine wearing the same CLI.
    #: It lowers through JAX via the `tpu-inference` plugin rather than running
    #: the CUDA kernels, so feature support is its own matrix — see
    #: [`TPU_MATURING`].
    VLLM_TPU = "vllm-tpu"
    SGLANG = "sglang"
    LLMD = "llm-d"
    LLAMA_CPP = "llama.cpp"
    MLX = "mlx"
    OLLAMA = "ollama"


@dataclass(frozen=True, slots=True)
class Requirements:
    """What the deployment has to do.

    Latency budgets are optional. `None` means unconstrained — which is a real
    answer for batch work, and must not be confused with "fast please".
    """

    workload: Workload
    #: Expected simultaneous in-flight requests.
    concurrency: int = 1
    #: Context length to serve.
    context: int = 32_768
    #: Time-to-first-token budget in milliseconds.
    ttft_ms: int | None = None
    #: Inter-token latency budget in milliseconds — the gap between tokens once
    #: generation has started.
    itl_ms: int | None = None
    #: Fraction of prompt tokens shared across requests, `0.0..=1.0`. The single
    #: most under-used number in LLM serving: a fixed system prompt across an
    #: agent fleet is often 0.8+, and nothing in a default config exploits it.
    prefix_sharing: float = 0.0
    #: Whether outputs must be schema-valid. Constrained decoding is a first-class
    #: requirement, not a prompt trick.
    structured_output: bool = False
    #: Fine-tuned adapters to serve over one set of base weights. Empty means a
    #: plain base-model deployment. The cheapest personalisation available —
    #: one copy of the weights, N adapters — and the one most often left unused
    #: because the flags differ per engine.
    lora: LoraFleet | None = None


@dataclass(frozen=True, slots=True)
class Knob:
    """One setting and the reason for it."""

    #: What this asks to be true. Not a flag — engines spell these
    #: differently, and one of them inverts the polarity.
    name: Setting
    value: str | int | float | bool
    why: str

    def render(self) -> str:
        """`max_concurrent = 256`. The engine's spelling comes from the adapter."""
        return f"{self.name} = {self.value}"


@dataclass(frozen=True, slots=True)
class Plan:
    """An engine, its settings, and everything the planner could not promise."""

    engine: Engine
    engine_why: str
    knobs: tuple[Knob, ...]
    fit: Fit | None = None
    #: Requirements the hardware cannot meet. Non-empty is not a failure to plan
    #: — it is the plan, honestly stated.
    warnings: tuple[str, ...] = ()
    #: Things worth knowing that are not warnings.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def meets_requirements(self) -> bool:
        """Whether every stated budget is reachable on this hardware."""
        return not self.warnings

    def get(self, name: Setting) -> Knob | None:
        """One knob by the intent it expresses."""
        return next((k for k in self.knobs if k.name == name), None)

    def settings(self) -> dict[Setting, object]:
        """The plan as intent, ready for an engine adapter."""
        return {k.name: k.value for k in self.knobs}

    def command(self, model: str) -> tuple[list[str], tuple[str, ...]]:
        """The actual launch command for this plan's engine, plus any gaps.

        Gaps are intents the engine has no *verified* flag for. They are
        returned rather than dropped: a caller must be able to see that a
        requested optimisation did not make it into the command.
        """
        from clickllm.engines import adapter_for

        adapter = adapter_for(self.engine.value)
        if adapter is None:
            return [], (
                f"no verified flag dialect for {self.engine.value}; emitting "
                f"another engine's flags would produce a command that cannot run",
            )
        argv, gaps = adapter.command(model, self.settings())
        return argv, tuple(f"{g.setting}: {g.reason}" for g in gaps)

    def environment(self) -> tuple[tuple[str, str], ...]:
        """Environment this plan's engine needs beyond argv. Empty for most engines.

        Ollama has exactly one flag (`--help`); everything else — concurrency,
        context — is told through `OLLAMA_*` variables. `command()`'s argv alone
        is not a runnable Ollama deployment without this.
        """
        from clickllm.engines import adapter_for

        adapter = adapter_for(self.engine.value)
        if adapter is None:
            return ()
        return adapter.environment(self.settings())

    def explain(self) -> str:
        """Every choice with its reason, in the order it was decided."""
        head = [
            f"engine: {self.engine.value}",
            f"  {self.engine_why}",
            "",
        ]
        body = [f"  {k.name} = {k.value}\n      {k.why}" for k in self.knobs]
        tail = []
        if self.warnings:
            tail += ["", "CANNOT MEET:"] + [f"  · {w}" for w in self.warnings]
        if self.notes:
            tail += ["", "notes:"] + [f"  · {n}" for n in self.notes]
        return "\n".join(head + body + tail)


def _pick_engine(hw: Hardware, req: Requirements) -> tuple[Engine, str]:
    """Choose the serving stack.

    Hardware first — vLLM, SGLang and llm-d are all CUDA-only, so on Apple
    silicon the choice is made for us and pretending otherwise would emit a
    config that cannot start. (ADR-0002 is why this does not leak past the
    `Runtime` trait on the Rust side.)
    """
    if hw.kind == "tpu":
        # The only real option. SGLang and llm-d are CUDA-only, and llama.cpp
        # has no TPU path — offering any of them here produces a command that
        # cannot start, which is worse than no command.
        return (
            Engine.VLLM_TPU,
            "TPU: vLLM's TPU backend is the only engine here. It lowers through "
            "JAX via the tpu-inference plugin rather than running CUDA kernels, "
            "so it shares the CLI but not the feature matrix.",
        )

    if hw.kind == "apple":
        if req.structured_output or req.concurrency >= 4:
            return (
                Engine.MLX,
                "Apple silicon: the CUDA engines cannot run here at all. MLX has "
                "the better batching story of the two Metal options.",
            )
        return (
            Engine.LLAMA_CPP,
            "Apple silicon: the CUDA engines cannot run here. llama.cpp is the "
            "most predictable single-stream option on Metal.",
        )

    # The two branches below are CUDA-only builds, as this function's own
    # docstring says — but only "apple" and "tpu" were ever checked, so an AMD
    # host fell straight through to them. `clickllm box` would then bake
    # lmsysorg/sglang:latest into a linux-rocm artifact with /dev/kfd mounted
    # and an amd.com/gpu resource: a container that starts, finds no CUDA
    # device, and dies on the hardware its own header says it was tuned for.
    # Unreachable through `box` today only because prefix_sharing and workload
    # are not exposed there — incidental cover, not a guard.
    cuda_only_ok = hw.kind == "nvidia"

    if cuda_only_ok and req.prefix_sharing >= PREFIX_SHARING_FOR_RADIX:
        return (
            Engine.SGLANG,
            f"{req.prefix_sharing:.0%} of prompt tokens are shared across "
            f"requests. RadixAttention reuses that prefix's KV instead of "
            f"recomputing it per request — a structural win no vLLM flag "
            f"reproduces, and it grows with fleet size.",
        )

    if cuda_only_ok and req.workload is Workload.BATCH and req.concurrency >= 64:
        return (
            Engine.LLMD,
            "high-concurrency batch: prefill and decode have opposite hardware "
            "appetites, and disaggregating them lets each scale on the resource "
            "it is actually bound by.",
        )

    if not cuda_only_ok and (
        req.prefix_sharing >= PREFIX_SHARING_FOR_RADIX
        or (req.workload is Workload.BATCH and req.concurrency >= 64)
    ):
        return (
            Engine.VLLM,
            f"this workload has a structural case for SGLang or llm-d, but both "
            f"ship CUDA-only builds and this is {hw.kind} hardware. vLLM has a "
            f"ROCm build, so it is the engine that can actually start here — "
            f"the better stack is unavailable, not unconsidered.",
        )

    return (
        Engine.VLLM,
        "no structural reason to prefer another stack. PagedAttention is the "
        "best-supported default, and the widest set of models works on it.",
    )


def _memory_utilization(fit: Fit | None) -> Knob:
    """Derive the memory fraction from the solver rather than guessing.

    An engine's built-in default is chosen without knowing the model. Once
    weights and KV are computed, the right figure is arithmetic — and on a
    machine that is not dedicated to serving, a stock 0.92 is how an OOM
    happens two hours in.
    """
    if fit is None or fit.usable_bytes <= 0:
        return Knob(
            Setting.MEMORY_FRACTION,
            0.92,
            "no sizing available, so this is the engine's own default — a figure "
            "chosen without knowing the model. Re-plan with a model to derive it.",
        )
    need = fit.weight_bytes + fit.kv_bytes + fit.overhead_bytes
    frac = min(0.95, need / fit.usable_bytes + MEMORY_SAFETY_MARGIN)
    return Knob(
        Setting.MEMORY_FRACTION,
        round(frac, 2),
        f"weights + KV + overhead is {need / 2**30:.1f} GiB of "
        f"{fit.usable_bytes / 2**30:.1f} GiB usable, plus a "
        f"{MEMORY_SAFETY_MARGIN:.0%} margin for activation spikes and "
        f"fragmentation. Derived, not the engine's model-blind default.",
    )


def _max_num_seqs(req: Requirements) -> Knob:
    """Cap on simultaneous sequences — the throughput/tail-latency dial.

    Every additional sequence in a step adds to the step's duration, so this is
    the knob that converts concurrency into inter-token latency.
    """
    if req.workload is Workload.BATCH:
        return Knob(
            Setting.MAX_CONCURRENT,
            max(256, req.concurrency),
            "batch work: nobody is waiting, so run the largest batch memory "
            "allows. Every sequence added here is throughput bought at the cost "
            "of a latency nobody is measuring.",
        )
    if req.workload is Workload.REALTIME:
        cap = max(8, min(32, req.concurrency))
        return Knob(
            Setting.MAX_CONCURRENT,
            cap,
            f"real-time: each sequence in a step lengthens that step, and the "
            f"budget is per token. Capped at {cap} so tail latency stays bounded "
            f"even when a burst arrives.",
        )
    return Knob(
        Setting.MAX_CONCURRENT,
        max(32, min(128, req.concurrency * 2)),
        "interactive: headroom for bursts without letting a full batch stretch "
        "the gap between tokens past what reads as fluent.",
    )


def _speculative(req: Requirements, fit: Fit | None) -> Knob:
    """Speculative decoding — a bet on spare compute.

    The subtlety that costs people throughput: drafting is *free* only when the
    accelerator is underused. At high batch the verify pass competes with real
    tokens and measured throughput falls. So this is derived from concurrency,
    and turned off for batch regardless.
    """
    if req.workload is Workload.BATCH:
        return Knob(
            Setting.SPECULATIVE,
            "off",
            "batch: the accelerator is already saturated, so drafting competes "
            "with real tokens. Speculative decoding is a latency optimisation "
            "that reads as a throughput one.",
        )
    if req.concurrency > MAX_SPEC_DECODE_CONCURRENCY:
        return Knob(
            Setting.SPECULATIVE,
            "off",
            f"concurrency {req.concurrency} is above {MAX_SPEC_DECODE_CONCURRENCY}: "
            f"there is no spare compute for the draft pass, and published EAGLE "
            f"numbers turn negative around batch 32.",
        )
    slow = fit is not None and fit.tokens_per_sec is not None and fit.tokens_per_sec < 40
    # A model with its own MTP head needs no draft model — the weights are
    # already in the checkpoint. Asking for eagle3 here would request a draft
    # that does not exist for this family.
    model_id = fit.model.id if fit is not None else ""
    native_mtp = any(k in model_id for k in MTP_NATIVE)
    method = "mtp" if native_mtp else "eagle3"

    why = (
        f"concurrency {req.concurrency} leaves compute idle between tokens, which "
        f"is exactly what drafting spends"
    )
    if native_mtp:
        why += (
            ". This family ships its own multi-token-prediction head, so drafting "
            "reuses weights already resident rather than loading a separate draft "
            "model. One speculative token: running the same head repeatedly lowers "
            "the acceptance rate."
        )
    elif slow:
        why += (
            f". Decode is estimated at {fit.tokens_per_sec:.0f} tok/s, so the "
            f"latency win is worth the memory."
        )
    else:
        why += ". Expect roughly 1.3–1.8× on interactive traffic; measure it."

    return Knob(Setting.SPECULATIVE, method, why)


def _chunked_prefill(req: Requirements) -> Knob:
    """Whether a long prompt may monopolise a scheduler step."""
    if req.workload is Workload.BATCH:
        return Knob(
            Setting.PREFILL_CHUNK,
            8192,
            "batch: let prefill run in large chunks. There is no token waiting "
            "behind it, so the scheduling fairness chunking buys is worthless "
            "here and its overhead is not.",
        )
    return Knob(
        Setting.PREFILL_CHUNK,
        PREFILL_CHUNK_TOKENS,
        f"chunked prefill at {PREFILL_CHUNK_TOKENS} tokens: without it, one "
        f"{req.context:,}-token prompt occupies a full step and every in-flight "
        f"response stalls behind it. This is the single most effective knob for "
        f"time-to-first-token under mixed traffic.",
    )


def _prefix_caching(req: Requirements) -> Knob:
    """Reuse of KV for repeated prefixes."""
    on = req.prefix_sharing >= PREFIX_CACHING_FLOOR
    return Knob(
        Setting.PREFIX_REUSE,
        on,
        (
            f"{req.prefix_sharing:.0%} of prompt tokens repeat across requests; "
            f"caching them skips prefill entirely for that span."
            if on
            else f"only {req.prefix_sharing:.0%} of prompt tokens repeat — below "
            f"{PREFIX_CACHING_FLOOR:.0%} the block bookkeeping costs more than "
            f"the hits return."
        ),
    )


def _kv_cache_dtype(req: Requirements, fit: Fit | None, engine: Engine) -> Knob | None:
    """Quantise the KV cache — but only when KV is what is actually binding.

    fp8 roughly halves KV memory, which is the difference between fitting a
    workload and not. It is also the one knob here with a real accuracy question:
    vLLM's own documentation makes **no quantitative accuracy claim**, notes that
    some layers are more sensitive, and points at dataset calibration for the
    highest quality.

    So this is offered only when KV dominates the memory budget, and the reason
    says what is being traded rather than presenting it as free. Where KV is a
    small share, enabling it would take an accuracy risk to save memory nobody
    needed.

    Values verified against vLLM's quantized-KV-cache page (2026-07-27):
    `fp8_e4m3` is CUDA 11.8+ and ROCm, `fp8_e5m2` is CUDA only.
    """
    if fit is None or engine is Engine.VLLM_TPU:
        # No sizing means no basis; TPU's KV path is its own matrix and is not
        # claimed here. Absent is the honest answer for both.
        return None

    total = fit.weight_bytes + fit.kv_bytes + fit.overhead_bytes
    if total <= 0:
        return None
    kv_share = fit.kv_bytes / total
    if kv_share < KV_PRESSURE_FOR_FP8:
        return None

    saved = fit.kv_bytes / 2 / 2**30
    return Knob(
        Setting.KV_CACHE_DTYPE,
        "fp8_e4m3",
        f"the KV cache is {kv_share:.0%} of the memory budget, so it — not the "
        f"weights — is what limits this deployment. fp8 roughly halves it, "
        f"freeing about {saved:.0f} GiB for more context or more concurrency. "
        f"This is the one setting here with a real accuracy cost: vLLM publishes "
        f"no quantitative claim, some layers are more sensitive than others, and "
        f"the default scales are trivial unless you calibrate. Prove it before "
        f"you trust it. e4m3 rather than e5m2 because it also works on ROCm.",
    )


def _tensor_parallel(hw: Hardware, fit: Fit | None) -> Knob | None:
    """Shard across devices only when one will not do.

    Aggregate bandwidth scales sub-linearly — the interconnect and the
    all-reduce eat into it — so splitting a model that already fits makes it
    slower, not faster. A surprisingly common misconfiguration.
    """
    if hw.devices <= 1:
        return None
    if fit is not None and fit.feasible and fit.weight_bytes < fit.usable_bytes / hw.devices:
        return Knob(
            Setting.TENSOR_PARALLEL,
            1,
            f"the model fits on one of the {hw.devices} devices. Sharding it "
            f"anyway adds an all-reduce per layer and aggregates bandwidth "
            f"sub-linearly — measurably slower, for nothing.",
        )
    return Knob(
        Setting.TENSOR_PARALLEL,
        hw.devices,
        f"the model does not fit on one device, so all {hw.devices} are needed. "
        f"Expect well under {hw.devices}× the single-device throughput: "
        f"bandwidth aggregates sub-linearly.",
    )


def _lora(req: Requirements) -> Knob | None:
    """Multi-LoRA, when there are adapters to serve.

    The number that matters is how many *distinct* adapters may share one batch,
    not how many are loaded. Both engines default that cap to something small,
    so loading eight adapters and leaving it alone quietly serialises requests
    that could have run together — the deployment looks fine and throughput is a
    fraction of what the hardware would give.

    It is capped at the request concurrency because a batch cannot hold more
    distinct adapters than it holds requests, and every extra slot costs memory
    for nothing.
    """
    fleet = req.lora
    if fleet is None or not fleet.adapters:
        return None

    per_batch = min(len(fleet.adapters), max(req.concurrency, 1))
    return Knob(
        Setting.LORA_FLEET,
        replace(fleet, max_concurrent=per_batch),
        f"{len(fleet.adapters)} adapters over one set of base weights, up to "
        f"{per_batch} in a single batch. The per-batch cap defaults low in both "
        f"engines, and leaving it there serialises adapters that could have run "
        f"together. Capped at concurrency {req.concurrency}: a batch cannot hold "
        f"more distinct adapters than it holds requests.",
    )


def _structured(req: Requirements, engine: Engine) -> Knob | None:
    """Constrained decoding, when the output has to parse."""
    if not req.structured_output:
        return None
    return Knob(
        Setting.STRUCTURED_OUTPUT,
        "xgrammar",
        "outputs must be schema-valid. Constraining the sampler guarantees a "
        "parse; asking the model nicely and retrying does not, and the retries "
        "are billed. The flag that carries this differs per engine — and in "
        "vLLM's case was renamed — so it is emitted by the adapter, not here.",
    )


def _budget_warnings(req: Requirements, fit: Fit | None) -> tuple[str, ...]:
    """Requirements the hardware cannot meet, stated rather than configured around.

    Roofline estimates, explicitly labelled. The point is not precision — it is
    refusing to emit an authoritative-looking config for a budget that is out of
    reach by an order of magnitude.
    """
    out: list[str] = []
    if fit is None:
        return ()
    if fit.tokens_per_sec is None:
        # Returning () here made "could not check" indistinguishable from
        # "checked and met": `meets_requirements` is `not self.warnings`, so an
        # unverifiable budget read as a satisfied one.
        #
        # That is not a corner. `hardware.detect()` sets `bandwidth_gbps=None`
        # for every NVIDIA and AMD GPU it finds (hardware.py:159, :187), so
        # `tokens_per_sec` is None on essentially every real accelerator — and a
        # physically impossible `itl_ms=1` came back `meets_requirements=True`
        # with no warnings at all. The same request on a machine whose bandwidth
        # is known is correctly refused.
        #
        # Say so instead. An unchecked budget is a stated unknown, which is the
        # house rule: `?` rather than a fabricated score.
        stated = [
            n for n, v in (("inter-token", req.itl_ms), ("time-to-first-token", req.ttft_ms)) if v
        ]
        if stated:
            out.append(
                f"{' and '.join(stated)} budget could not be checked: this "
                f"hardware's memory bandwidth is unknown, so there is no "
                f"throughput estimate to compare against. Not a pass — unmeasured."
            )
        return tuple(out)

    if req.itl_ms is not None and fit.tokens_per_sec > 0:
        achievable = 1000.0 / fit.tokens_per_sec
        if achievable > req.itl_ms:
            out.append(
                f"inter-token budget is {req.itl_ms}ms, but decode is a roofline "
                f"estimate of {fit.tokens_per_sec:.0f} tok/s — about "
                f"{achievable:.0f}ms per token. Roofline estimate, not measured; "
                f"the gap is large enough that no flag closes it. Needs a smaller "
                f"model, a heavier quant, or more bandwidth."
            )

    if not fit.feasible:
        out.append(
            f"the model does not fit: needs {fit.total_bytes / 2**30:.1f} GiB "
            f"against {fit.usable_bytes / 2**30:.1f} GiB usable. Every knob below "
            f"is moot until that is resolved."
        )
    return tuple(out)


def _tpu_notes(hw: Hardware, model: ModelSpec | None) -> list[str]:
    """What a TPU deployment needs told, and what cannot be promised.

    Verified against vLLM's TPU project docs (2026-07-27). Every item is a
    published limitation, not an inference from first principles — the same rule
    the engine adapter follows, for the same reason.
    """
    notes = [
        "TPU serving runs through vLLM's tpu-inference plugin, which lowers via "
        "JAX. Flags are vLLM's; the feature matrix is its own and is published "
        "separately.",
        "multi-host serving is not supported on GKE, so memory cannot be "
        "aggregated across hosts — a model larger than one host is not a "
        "sharding problem here, it is out of reach.",
    ]
    if model is not None and getattr(model, "kv_scheme", None) == "mla":
        notes.append(
            "this model uses MLA attention, which vLLM lists as still maturing "
            "on TPU. It may run correctly and unoptimised, or not run — check "
            "the published support matrix before committing to this hardware."
        )
    if model is not None and getattr(model, "is_moe", False):
        notes.append(
            "large MoE models are listed as still maturing on TPU; validate "
            "throughput on your own traffic rather than trusting a projection."
        )
    if not any(g in hw.name.lower() for g in TPU_RECOMMENDED):
        notes.append(
            f"this generation is not on vLLM's recommended list "
            f"({', '.join(TPU_RECOMMENDED)}); the others are supported as "
            f"experimental, which is a different promise."
        )
    return notes


def plan(
    hw: Hardware,
    req: Requirements,
    model: ModelSpec | None = None,
    quant: str | None = None,
    force_engine: Engine | None = None,
) -> Plan:
    """Pick an engine and its settings for `req` on `hw`.

    Args:
        hw: the machine.
        req: what the deployment has to do.
        model: the model to serve. Optional — an engine and most scheduling
            knobs are decidable without it, and memory knobs say so when it is
            absent rather than inventing a figure.
        quant: quantisation, when a model is given.
        force_engine: bypass `_pick_engine` and use this one. Some engines have
            no hardware signature to be auto-selected by — Ollama runs equally
            well anywhere, and llm-d needs a workload a caller may not be
            asking for — so this is their only way in.

    Returns:
        A [`Plan`]. Check [`Plan.warnings`] before trusting the flags: a plan
        that cannot meet the requirement still emits its reasoning, and saying
        so is the point.
    """
    fit = (
        solve(model, quant or model.quants[0], hw, req.context, req.concurrency)
        if model is not None
        else None
    )
    if force_engine is not None:
        engine, why = force_engine, f"{force_engine.value} was explicitly requested"
    else:
        engine, why = _pick_engine(hw, req)

    knobs: list[Knob] = [
        Knob(
            Setting.CONTEXT_LENGTH,
            req.context,
            f"{req.context:,} tokens, as required. Larger costs KV memory for "
            f"context nobody sends; smaller truncates real requests.",
        ),
        _max_num_seqs(req),
        _chunked_prefill(req),
        _prefix_caching(req),
        _speculative(req, fit),
        _memory_utilization(fit),
    ]
    if (kv := _kv_cache_dtype(req, fit, engine)) is not None:
        knobs.append(kv)
    if (tp := _tensor_parallel(hw, fit)) is not None:
        knobs.append(tp)
    if (sd := _structured(req, engine)) is not None:
        knobs.append(sd)
    if (lora := _lora(req)) is not None:
        knobs.append(lora)
    if model is not None and quant:
        knobs.append(
            Knob(
                Setting.QUANTIZATION,
                quant,
                "decode is bandwidth-bound, so a smaller weight format is "
                "directly faster as well as smaller.",
            )
        )

    notes = []
    if engine is Engine.VLLM_TPU:
        notes.extend(_tpu_notes(hw, model))
    if engine is Engine.SGLANG:
        notes.append(
            "SGLang's flag names differ from vLLM's; the emitted config "
            "translates them. The reasoning above is engine-independent."
        )
    if req.workload is Workload.REALTIME and req.ttft_ms is None:
        notes.append(
            "no time-to-first-token budget was given, so scheduling was tuned "
            "for tail latency generally. Supply `ttft_ms` for a tighter plan."
        )

    return Plan(
        engine=engine,
        engine_why=why,
        knobs=tuple(knobs),
        fit=fit,
        warnings=_budget_warnings(req, fit),
        notes=tuple(notes),
    )


def demo() -> None:
    """Self-check. Run with `python -m clickllm.plan`."""
    from clickllm.hardware import Hardware

    h100 = Hardware(
        kind="nvidia",
        name="H100 80GB",
        total_bytes=80 * 2**30,
        usable_bytes=76 * 2**30,
        bandwidth_gbps=3350.0,
        cores=132,
    )
    from dataclasses import replace

    quad = replace(h100, devices=4)  # `Hardware` uses slots; no __dict__
    mac = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * 2**30,
        usable_bytes=96 * 2**30,
        bandwidth_gbps=546.0,
        cores=16,
    )

    # The same hardware, two workloads, opposite configurations. This is the
    # whole thesis of the module.
    batch = plan(h100, Requirements(Workload.BATCH, concurrency=128))
    live = plan(h100, Requirements(Workload.REALTIME, concurrency=8, itl_ms=50))
    assert batch.get(Setting.MAX_CONCURRENT).value > live.get(Setting.MAX_CONCURRENT).value
    assert batch.get(Setting.PREFILL_CHUNK).value > live.get(Setting.PREFILL_CHUNK).value
    assert batch.get(Setting.SPECULATIVE).value == "off"
    assert live.get(Setting.SPECULATIVE).value == "eagle3"

    # Speculative decoding is a function of load, not a checkbox.
    busy = plan(h100, Requirements(Workload.INTERACTIVE, concurrency=64))
    assert busy.get(Setting.SPECULATIVE).value == "off"
    assert "no spare compute" in busy.get(Setting.SPECULATIVE).why

    # Prefix sharing picks the engine.
    agent = plan(h100, Requirements(Workload.INTERACTIVE, concurrency=8, prefix_sharing=0.8))
    assert agent.engine is Engine.SGLANG, agent.engine
    assert "RadixAttention" in agent.engine_why
    assert plan(h100, Requirements(Workload.INTERACTIVE, prefix_sharing=0.1)).engine is Engine.VLLM

    # ...and separately decides prefix caching.
    assert agent.get(Setting.PREFIX_REUSE).value is True
    assert plan(h100, Requirements(Workload.INTERACTIVE)).get(Setting.PREFIX_REUSE).value is False

    # High-concurrency batch disaggregates.
    assert plan(h100, Requirements(Workload.BATCH, concurrency=256)).engine is Engine.LLMD

    # Apple silicon cannot run the CUDA engines, and the planner says so rather
    # than emitting a config that will not start.
    for req in (Workload.BATCH, Workload.INTERACTIVE):
        p = plan(mac, Requirements(req, concurrency=1))
        assert p.engine in (Engine.LLAMA_CPP, Engine.MLX), p.engine
        assert "cannot run" in p.engine_why
    assert plan(mac, Requirements(Workload.INTERACTIVE, concurrency=8)).engine is Engine.MLX

    # Structured output is a decoding constraint, not a prompt.
    s = plan(h100, Requirements(Workload.INTERACTIVE, structured_output=True))
    assert s.get(Setting.STRUCTURED_OUTPUT).value == "xgrammar"
    assert (
        plan(
            h100, Requirements(Workload.INTERACTIVE, structured_output=True, prefix_sharing=0.9)
        ).get(Setting.STRUCTURED_OUTPUT)
        is not None
    )

    # Tensor parallelism only appears on multi-device hardware.
    assert plan(h100, Requirements(Workload.BATCH)).get(Setting.TENSOR_PARALLEL) is None
    tp = plan(quad, Requirements(Workload.BATCH)).get(Setting.TENSOR_PARALLEL)
    assert tp is not None and tp.value == 4

    # Without a model, the memory knob admits it is a guess rather than deriving.
    mem = plan(h100, Requirements(Workload.BATCH)).get(Setting.MEMORY_FRACTION)
    assert mem.value == 0.92 and "without knowing the model" in mem.why, mem

    # Every knob carries its reasoning. A number you cannot argue with is a
    # number you cannot fix.
    for p in (batch, live, agent, s):
        assert all(len(k.why) > 30 for k in p.knobs), p.explain()
        assert p.explain()

    print("plan: ok")


if __name__ == "__main__":
    demo()
