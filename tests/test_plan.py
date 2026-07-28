"""The workload planner.

`plan.demo()` covers each fork in isolation. These cover the claim the module is
actually making: that **the same model on the same hardware gets configured
differently depending on what it is for**, and that every difference is derived
from a stated reason rather than a preference.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

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


def knob(p, name):
    k = p.get(name)
    assert k is not None, f"{name} missing from {[x.name for x in p.knobs]}"
    return k


# --- the claim: same hardware, opposite configs --------------------------------


def test_batch_and_realtime_are_configured_in_opposite_directions():
    """The whole thesis. If these converged, the module would be decoration."""
    batch = plan(H100, Requirements(Workload.BATCH, concurrency=128))
    live = plan(H100, Requirements(Workload.REALTIME, concurrency=8, itl_ms=50))

    assert knob(batch, "--max-num-seqs").value > knob(live, "--max-num-seqs").value
    assert (
        knob(batch, "--max-num-batched-tokens").value > knob(live, "--max-num-batched-tokens").value
    )
    assert knob(batch, "--speculative-config").value == "off"
    assert knob(live, "--speculative-config").value != "off"


def test_interactive_sits_between_the_two_extremes():
    seqs = lambda w, c: knob(plan(H100, Requirements(w, concurrency=c)), "--max-num-seqs").value  # noqa: E731
    assert seqs(Workload.REALTIME, 16) <= seqs(Workload.INTERACTIVE, 16) <= seqs(Workload.BATCH, 16)


# --- speculative decoding is a function of load --------------------------------


@pytest.mark.parametrize("concurrency", [1, 4, 8, MAX_SPEC_DECODE_CONCURRENCY])
def test_speculative_decoding_is_on_below_the_threshold(concurrency):
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=concurrency))
    assert knob(p, "--speculative-config").value == "eagle3"


@pytest.mark.parametrize("concurrency", [MAX_SPEC_DECODE_CONCURRENCY + 1, 64, 512])
def test_speculative_decoding_is_off_above_it_and_says_why(concurrency):
    # The failure this prevents: enabling drafting on a saturated accelerator,
    # where the verify pass competes with real tokens and throughput drops.
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=concurrency))
    k = knob(p, "--speculative-config")
    assert k.value == "off"
    assert "spare compute" in k.why


def test_batch_disables_speculation_regardless_of_concurrency():
    p = plan(H100, Requirements(Workload.BATCH, concurrency=1))
    assert knob(p, "--speculative-config").value == "off"
    assert "saturated" in knob(p, "--speculative-config").why


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
    assert knob(mid, "--enable-prefix-caching").value is True


@pytest.mark.parametrize("sharing", [0.0, PREFIX_CACHING_FLOOR - 0.01])
def test_prefix_caching_is_off_when_the_bookkeeping_would_not_pay(sharing):
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=sharing))
    assert knob(p, "--enable-prefix-caching").value is False
    assert "bookkeeping" in knob(p, "--enable-prefix-caching").why


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
    assert plan(H100, Requirements(Workload.BATCH)).get("--tensor-parallel-size") is None


def test_multi_device_hardware_gets_the_flag_and_a_caveat_about_scaling():
    k = knob(plan(QUAD, Requirements(Workload.BATCH)), "--tensor-parallel-size")
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
    k = knob(plan(H100, Requirements(Workload.BATCH)), "--gpu-memory-utilization")
    assert k.value == 0.90
    assert "guess" in k.why


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


def test_structured_output_constrains_the_sampler_on_both_engines():
    vllm = plan(H100, Requirements(Workload.INTERACTIVE, structured_output=True))
    assert knob(vllm, "--guided-decoding-backend").value == "xgrammar"

    sgl = plan(H100, Requirements(Workload.INTERACTIVE, structured_output=True, prefix_sharing=0.9))
    assert sgl.engine is Engine.SGLANG
    assert knob(sgl, "--grammar-backend").value == "xgrammar"
    assert vllm.get("--grammar-backend") is None


def test_structured_output_is_absent_when_not_required():
    p = plan(H100, Requirements(Workload.INTERACTIVE))
    assert p.get("--guided-decoding-backend") is None
    assert p.get("--grammar-backend") is None


# --- context -------------------------------------------------------------------


def test_the_context_flag_follows_the_requirement_exactly():
    for ctx in (8192, 32_768, 131_072):
        p = plan(H100, Requirements(Workload.INTERACTIVE, context=ctx))
        assert knob(p, "--max-model-len").value == ctx


def test_notes_flag_a_realtime_plan_with_no_stated_budget():
    vague = plan(H100, Requirements(Workload.REALTIME, concurrency=4))
    assert any("time-to-first-token budget" in n for n in vague.notes)
    precise = plan(H100, Requirements(Workload.REALTIME, concurrency=4, ttft_ms=200))
    assert not any("time-to-first-token budget" in n for n in precise.notes)


def test_an_sglang_plan_notes_that_flag_names_are_translated():
    p = plan(H100, Requirements(Workload.INTERACTIVE, prefix_sharing=0.9))
    assert any("flag names differ" in n for n in p.notes)
