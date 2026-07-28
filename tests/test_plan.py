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


# --- TPU -----------------------------------------------------------------------
# Every fact asserted here is from Google Cloud's per-generation spec pages and
# vLLM's TPU project docs, checked 2026-07-27 — not inferred.


TPU_V6E = Hardware(
    kind="tpu",
    name="TPU v6e Trillium (8 chips)",
    total_bytes=256 * 2**30,
    usable_bytes=240 * 2**30,
    bandwidth_gbps=11384.0,
    cores=8,
    devices=8,
)


def test_a_tpu_never_gets_an_engine_that_cannot_run_there():
    # SGLang and llm-d are CUDA-only; llama.cpp has no TPU path. Offering any of
    # them produces a command that cannot start.
    from clickllm.plan import Engine

    for w in Workload:
        for sharing in (0.0, 0.9):  # 0.9 would pick SGLang on CUDA
            p = plan(TPU_V6E, Requirements(w, concurrency=64, prefix_sharing=sharing))
            assert p.engine is Engine.VLLM_TPU, f"{w}/{sharing}: {p.engine}"


def test_the_tpu_plan_says_it_is_a_different_engine_wearing_the_same_cli():
    p = plan(TPU_V6E, Requirements(Workload.INTERACTIVE, concurrency=8))
    joined = " ".join(p.notes)
    assert "tpu-inference" in joined and "JAX" in joined
    assert "feature matrix" in joined


def test_the_multi_host_ceiling_is_stated_because_it_is_not_a_sharding_problem():
    p = plan(TPU_V6E, Requirements(Workload.BATCH, concurrency=128))
    assert any("multi-host" in n and "aggregated" in n for n in p.notes), p.notes


def test_an_experimental_generation_is_flagged_as_a_different_promise():
    from dataclasses import replace

    v4 = replace(TPU_V6E, name="TPU v4 (8 chips)")
    assert any(
        "not on vLLM's recommended list" in n for n in plan(v4, Requirements(Workload.BATCH)).notes
    )
    # ...and a recommended one is not.
    assert not any(
        "recommended list" in n for n in plan(TPU_V6E, Requirements(Workload.BATCH)).notes
    )


def test_an_mla_model_on_tpu_is_flagged_rather_than_silently_planned():
    # DeepSeek-family models are MLA, which vLLM lists as still maturing on TPU.
    from clickllm.catalog import load

    mla = next((m for m in load() if getattr(m, "kv_scheme", "") == "mla"), None)
    assert mla is not None, "catalogue has no MLA model to test with"
    p = plan(
        TPU_V6E, Requirements(Workload.INTERACTIVE, concurrency=8), model=mla, quant=mla.quants[0]
    )
    assert any("MLA attention" in n and "maturing" in n for n in p.notes), p.notes


def test_a_cuda_plan_carries_no_tpu_notes():
    p = plan(H100, Requirements(Workload.INTERACTIVE, concurrency=8))
    assert not any("TPU" in n or "tpu-inference" in n for n in p.notes)


# --- KV cache dtype ------------------------------------------------------------
# The one setting with a real accuracy cost, so it is offered only when KV is
# what actually binds. Values verified against vLLM's quantized-KV-cache page.


def _with_model(concurrency: int, context: int, hw: Hardware = H100):
    from clickllm.catalog import load

    m = load()[0]
    return plan(
        hw,
        Requirements(Workload.INTERACTIVE, concurrency=concurrency, context=context),
        model=m,
        quant=m.quants[0],
    )


def test_fp8_kv_is_not_offered_when_weights_are_what_bind():
    # Taking an accuracy risk to save memory nobody needed is a bad trade, and
    # a tuner that always suggests it is one that cannot be trusted when it does.
    p = _with_model(concurrency=1, context=4096)
    assert p.get(Setting.KV_CACHE_DTYPE) is None


def test_fp8_kv_is_offered_when_the_cache_dominates_the_budget():
    p = _with_model(concurrency=32, context=32_768)
    k = knob(p, Setting.KV_CACHE_DTYPE)
    assert k.value == "fp8_e4m3", "e4m3 also works on ROCm; e5m2 is CUDA-only"
    assert "% of the memory budget" in k.why


def test_the_accuracy_cost_is_stated_not_buried():
    # vLLM publishes no quantitative accuracy claim for this, so neither do we.
    k = knob(_with_model(32, 32_768), Setting.KV_CACHE_DTYPE)
    assert "accuracy cost" in k.why
    assert "no quantitative claim" in k.why
    assert "Prove it before you trust it" in k.why


def test_fp8_kv_is_never_offered_without_a_model_to_size_against():
    # No sizing means no basis for claiming KV is the constraint.
    assert (
        plan(H100, Requirements(Workload.BATCH, concurrency=64)).get(Setting.KV_CACHE_DTYPE) is None
    )


def test_fp8_kv_is_not_claimed_on_tpu():
    # TPU's KV quantisation path is its own support matrix and is not verified
    # here, so it is absent rather than assumed to work.
    p = plan(TPU_V6E, Requirements(Workload.INTERACTIVE, concurrency=64, context=32_768))
    assert p.get(Setting.KV_CACHE_DTYPE) is None


def test_every_setting_the_adapter_supports_is_reachable_from_some_plan():
    """No dead capability: an adapter that can express a setting no planner ever
    emits is a feature that exists only in tests."""
    from dataclasses import replace

    from clickllm.catalog import load

    m = load()[0]
    reachable: set[Setting] = set()
    for hw in (H100, replace(H100, devices=4)):
        for w in Workload:
            for structured in (False, True):
                for conc, ctx in ((8, 8192), (32, 32_768)):
                    p = plan(
                        hw,
                        Requirements(
                            w, concurrency=conc, context=ctx, structured_output=structured
                        ),
                        model=m,
                        quant=m.quants[0],
                    )
                    reachable |= {k.name for k in p.knobs}
    assert reachable == set(Setting), (
        f"never emitted: {sorted(x.value for x in set(Setting) - reachable)}"
    )


# --- the kernel seam and the launcher ------------------------------------------


def test_every_vllm_plugin_group_documents_what_it_returns():
    # A wrong return is a plugin that loads and silently does nothing, so the
    # contract per group is the thing worth pinning.
    from clickllm.kernels import ENTRY_POINT_GROUPS, PluginKind

    assert set(ENTRY_POINT_GROUPS) == set(PluginKind)
    assert all(len(v) > 20 for v in ENTRY_POINT_GROUPS.values())
    # These are literal entry-point group names vLLM reads; a typo means a
    # plugin that is never discovered.
    assert PluginKind.GENERAL.value == "vllm.general_plugins"
    assert PluginKind.PLATFORM.value == "vllm.platform_plugins"


def test_a_scaffolded_plugin_declares_the_group_vllm_actually_reads():
    from clickllm.kernels import Plugin, PluginKind, scaffold

    p = Plugin("fused-rmsnorm", PluginKind.GENERAL, "fused_rmsnorm:register")
    files = scaffold(p)
    assert "vllm.general_plugins" in files["pyproject.toml"]
    assert 'fused-rmsnorm = "fused_rmsnorm:register"' in files["pyproject.toml"]


def test_the_scaffold_warns_about_re_entrancy():
    # register() runs once per worker under tensor parallelism; one that appends
    # to a list breaks the moment TP > 1, far from the cause.
    from clickllm.kernels import Plugin, PluginKind, scaffold

    src = scaffold(Plugin("k", PluginKind.GENERAL, "k:register"))["k/__init__.py"]
    assert "re-entrant" in src and "already registered" in src


def test_a_platform_plugin_returns_none_rather_than_raising_when_absent():
    from clickllm.kernels import Plugin, PluginKind, scaffold

    src = scaffold(Plugin("npu", PluginKind.PLATFORM, "npu:register"))["npu/__init__.py"]
    assert "return None" in src
    assert "not a failure" in src, "absence must be distinguished from breakage"


def test_a_bit_identical_claim_is_falsifiable_and_a_drifting_one_is_statistical():
    from clickllm.kernels import KernelClaim, verification_plan

    exact = " ".join(verification_plan(KernelClaim("e", 1.1, bit_identical=True)))
    fuzzy = " ".join(verification_plan(KernelClaim("f", 1.1, expected_drift="fp16 order")))
    assert "bit-identical output" in exact
    assert "lower bound" in fuzzy and "bit-identical output" not in fuzzy


def test_every_verification_plan_measures_at_real_concurrency():
    # Single-stream kernel wins routinely vanish under batching.
    from clickllm.kernels import KernelClaim, verification_plan

    for claim in (
        KernelClaim("a", 1.4, bit_identical=True),
        KernelClaim("b", 1.4, expected_drift="x"),
    ):
        plan_steps = " ".join(verification_plan(claim))
        assert "not at batch 1" in plan_steps
        assert "receipt" in plan_steps


def test_the_launcher_reuses_clickllm_ui_and_stays_on_loopback():
    from clickllm.desktop import launch_script

    s = launch_script("/usr/bin/python3", 7171)
    assert "clickllm.cli ui" in s, "must not reimplement serving"
    assert "127.0.0.1" in s and "0.0.0.0" not in s, "the workbench is loopback-only"
    # Double-clicking twice must open the running instance, not clash on the port.
    assert "exit 0" in s


def test_the_launcher_says_how_to_remove_itself(tmp_path):
    from clickllm.desktop import _macos

    lz = _macos(tmp_path, "/usr/bin/python3", 7171)
    assert lz.uninstall.startswith("rm -rf ")
    assert str(tmp_path) in lz.uninstall
    assert (lz.path / "Contents" / "Info.plist").exists()


def test_the_bundle_is_not_background_only():
    # A background-only app serving a web page is a process users cannot find
    # to quit.
    from clickllm.desktop import plist

    assert "LSBackgroundOnly" not in plist("clickllm")
