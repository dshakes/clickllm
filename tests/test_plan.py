"""The workload planner.

`plan.demo()` covers each fork in isolation. These cover the claim the module is
actually making: that **the same model on the same hardware gets configured
differently depending on what it is for**, and that every difference is derived
from a stated reason rather than a preference.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from clickllm.engines import Setting
from clickllm.hardware import Hardware
from clickllm.plan import (
    MAX_SPEC_DECODE_CONCURRENCY,
    PREFIX_CACHING_FLOOR,
    PREFIX_SHARING_FOR_RADIX,
    Engine,
    Requirements,
    Workload,
    plan,
)

H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * 2**30,
    usable_bytes=76 * 2**30,
    bandwidth_gbps=3350.0,
    cores=132,
)
QUAD = replace(H100, devices=4)
MAC = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * 2**30,
    usable_bytes=96 * 2**30,
    bandwidth_gbps=546.0,
    cores=16,
)


def knob(p, name: Setting):
    k = p.get(name)
    assert k is not None, f"{name} missing from {[x.name for x in p.knobs]}"
    return k


# --- the claim: same hardware, opposite configs --------------------------------


def test_batch_and_realtime_are_configured_in_opposite_directions():
    """The whole thesis. If these converged, the module would be decoration."""
    batch = plan(H100, Requirements(Workload.BATCH, concurrency=128))
    live = plan(H100, Requirements(Workload.REALTIME, concurrency=8, itl_ms=50))

    assert knob(batch, Setting.MAX_CONCURRENT).value > knob(live, Setting.MAX_CONCURRENT).value
    assert knob(batch, Setting.PREFILL_CHUNK).value > knob(live, Setting.PREFILL_CHUNK).value
    assert knob(batch, Setting.SPECULATIVE).value == "off"
    assert knob(live, Setting.SPECULATIVE).value != "off"


def test_interactive_sits_between_the_two_extremes():
    def seqs(w, c):
        return knob(plan(H100, Requirements(w, concurrency=c)), Setting.MAX_CONCURRENT).value

    assert seqs(Workload.REALTIME, 16) <= seqs(Workload.INTERACTIVE, 16) <= seqs(Workload.BATCH, 16)


# --- speculative decoding is a function of load --------------------------------


@pytest.mark.parametrize("concurrency", [1, 4, 8, MAX_SPEC_DECODE_CONCURRENCY])
def test_speculative_decoding_is_on_below_the_threshold(concurrency):
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=concurrency))
    assert knob(p, Setting.SPECULATIVE).value == "eagle3"


@pytest.mark.parametrize("concurrency", [MAX_SPEC_DECODE_CONCURRENCY + 1, 64, 512])
def test_speculative_decoding_is_off_above_it_and_says_why(concurrency):
    # The failure this prevents: enabling drafting on a saturated accelerator,
    # where the verify pass competes with real tokens and throughput drops.
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=concurrency))
    k = knob(p, Setting.SPECULATIVE)
    assert k.value == "off"
    assert "spare compute" in k.why


def test_batch_disables_speculation_regardless_of_concurrency():
    p = plan(H100, Requirements(Workload.BATCH, concurrency=1))
    assert knob(p, Setting.SPECULATIVE).value == "off"
    assert "saturated" in knob(p, Setting.SPECULATIVE).why


# --- prefix sharing picks the engine -------------------------------------------


@pytest.mark.parametrize("sharing", [0.0, 0.2, PREFIX_SHARING_FOR_RADIX - 0.01])
def test_low_prefix_sharing_stays_on_vllm(sharing):
    assert (
        plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=sharing)).engine is Engine.VLLM
    )


@pytest.mark.parametrize("sharing", [PREFIX_SHARING_FOR_RADIX, 0.8, 1.0])
def test_high_prefix_sharing_picks_sglang_and_explains_radix(sharing):
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=sharing))
    assert p.engine is Engine.SGLANG
    assert "RadixAttention" in p.engine_why


def test_prefix_caching_is_decided_separately_from_the_engine():
    # A workload can benefit from caching without justifying an engine switch.
    mid = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=0.3))
    assert mid.engine is Engine.VLLM
    assert knob(mid, Setting.PREFIX_REUSE).value is True


@pytest.mark.parametrize("sharing", [0.0, PREFIX_CACHING_FLOOR - 0.01])
def test_prefix_caching_is_off_when_the_bookkeeping_would_not_pay(sharing):
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=sharing))
    assert knob(p, Setting.PREFIX_REUSE).value is False
    assert "bookkeeping" in knob(p, Setting.PREFIX_REUSE).why


# --- hardware constrains the engine --------------------------------------------


@pytest.mark.parametrize("workload", list(Workload))
def test_apple_silicon_never_gets_a_cuda_only_engine(workload):
    # vLLM, SGLang and llm-d are CUDA-only. Emitting one here produces a config
    # that cannot start, which is worse than no config.
    p = plan(MAC, Requirements(workload, concurrency=2))
    assert p.engine in (Engine.LLAMA_CPP, Engine.MLX)
    assert "cannot run" in p.engine_why


def test_high_concurrency_batch_disaggregates():
    assert plan(H100, Requirements(Workload.BATCH, concurrency=256)).engine is Engine.LLMD
    assert plan(H100, Requirements(Workload.BATCH, concurrency=8)).engine is Engine.VLLM


# --- tensor parallelism --------------------------------------------------------


def test_single_device_hardware_gets_no_tensor_parallel_flag():
    assert plan(H100, Requirements(Workload.BATCH)).get(Setting.TENSOR_PARALLEL) is None


def test_multi_device_hardware_gets_the_flag_and_a_caveat_about_scaling():
    k = knob(plan(QUAD, Requirements(Workload.BATCH)), Setting.TENSOR_PARALLEL)
    assert k.value == 4
    assert "sub-linear" in k.why


# --- honesty -------------------------------------------------------------------


def test_every_knob_carries_a_reason_worth_reading():
    for w in Workload:
        for sharing in (0.0, 0.9):
            p = plan(
                QUAD,
                Requirements(w, concurrency=8, prefix_sharing=sharing, structured_output=True),
            )
            for k in p.knobs:
                assert len(k.why) > 30, f"{k.name}: {k.why}"
                assert k.why[0].islower() or k.why[0].isdigit(), k.why


def test_without_a_model_the_memory_knob_admits_it_is_guessing():
    # Reporting a derived-looking figure with no model behind it would be the
    # exact dishonesty the repo's estimate-labelling convention exists to stop.
    k = knob(plan(H100, Requirements(Workload.BATCH)), Setting.MEMORY_FRACTION)
    assert k.value == 0.92
    assert "without knowing the model" in k.why


def test_a_plan_with_no_warnings_reports_that_it_meets_requirements():
    p = plan(H100, Requirements(Workload.BATCH, concurrency=32))
    assert p.meets_requirements and not p.warnings


def test_explain_renders_every_knob_and_its_reason():
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=4, structured_output=True))
    text = p.explain()
    for k in p.knobs:
        assert k.name in text
        assert k.why in text
    assert p.engine.value in text
    assert p.engine_why in text


# --- structured output ---------------------------------------------------------


def test_structured_output_is_an_intent_not_an_engine_flag():
    # The plan says "constrain the sampler". Which flag carries that — and
    # whether the engine even has a verified one — is the adapter's problem.
    for sharing, engine in ((0.0, Engine.VLLM), (0.9, Engine.SGLANG)):
        p = plan(
            H100,
            Requirements(Workload.INTERACTIVE, structured_output=True, prefix_sharing=sharing),
        )
        assert p.engine is engine
        assert knob(p, Setting.STRUCTURED_OUTPUT).value == "xgrammar"


def test_a_plan_emits_a_real_command_and_reports_what_it_could_not_express():
    vllm = plan(H100, Requirements(Workload.INTERACTIVE, structured_output=True))
    argv, gaps = vllm.command("Qwen/Qwen3-32B")
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-32B"]
    # The flag that no longer exists must never appear.
    assert "--guided-decoding-backend" not in argv
    assert "--structured-outputs-config" in argv
    assert not gaps

    sgl = plan(H100, Requirements(Workload.INTERACTIVE, structured_output=True, prefix_sharing=0.9))
    argv, gaps = sgl.command("Qwen/Qwen3-32B")
    assert "--context-length" in argv and "--max-model-len" not in argv
    # SGLang's grammar flag was not verifiable, so it is reported, not guessed.
    assert any("structured_output" in g for g in gaps), gaps


def test_an_engine_with_no_verified_dialect_refuses_rather_than_improvising():
    argv, gaps = plan(MAC, Requirements(Workload.INTERACTIVE)).command("m")
    assert argv == []
    assert gaps and "cannot run" in gaps[0]


def test_prefix_reuse_is_never_translated_into_its_opposite():
    # The inversion this layer exists for: SGLang's radix cache is on by
    # default, so "enable prefix reuse" must emit nothing there — certainly not
    # --disable-radix-cache.
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=0.9))
    assert knob(p, Setting.PREFIX_REUSE).value is True
    argv, _ = p.command("m")
    assert "--disable-radix-cache" not in argv


def test_structured_output_is_absent_when_not_required():
    p = plan(H100, Requirements(Workload.INTERACTIVE))
    assert p.get(Setting.STRUCTURED_OUTPUT) is None
    assert p.get(Setting.STRUCTURED_OUTPUT) is None


# --- context -------------------------------------------------------------------


def test_the_context_flag_follows_the_requirement_exactly():
    for ctx in (8192, 32_768, 131_072):
        p = plan(H100, Requirements(Workload.INTERACTIVE, context=ctx))
        assert knob(p, Setting.CONTEXT_LENGTH).value == ctx


def test_notes_flag_a_realtime_plan_with_no_stated_budget():
    vague = plan(H100, Requirements(Workload.REALTIME, concurrency=4))
    assert any("time-to-first-token budget" in n for n in vague.notes)
    precise = plan(H100, Requirements(Workload.REALTIME, concurrency=4, ttft_ms=200))
    assert not any("time-to-first-token budget" in n for n in precise.notes)


def test_an_sglang_plan_notes_that_flag_names_are_translated():
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=0.9))
    assert any("flag names differ" in n for n in p.notes)
