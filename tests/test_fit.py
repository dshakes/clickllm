"""The self-checks in each module are the real gate; this runs them under pytest
plus the cases that need a synthetic machine rather than the host's."""

import pytest

from clickllm import catalog, cli, fit, hardware
from clickllm.hardware import Hardware

GB = 1024**3


def _hw(usable_gb: float, *, kind="apple", bw=546.0, devices=1) -> Hardware:
    return Hardware(
        kind=kind,
        name="test",
        total_bytes=int(usable_gb / 0.75 * GB),
        usable_bytes=int(usable_gb * GB),
        bandwidth_gbps=bw,
        cores=16,
        devices=devices,
    )


def test_module_self_checks():
    hardware.demo()
    catalog.demo()
    fit.demo()


@pytest.mark.parametrize("model_id", [m.id for m in catalog.load()])
def test_every_catalog_entry_is_solvable(model_id):
    """No entry may crash the solver — a bad row must produce 'does not fit', not a traceback."""
    m = catalog.get(model_id)
    assert m.layers > 0 and m.kv_heads > 0 and m.head_dim > 0
    assert m.active_b <= m.params_b
    assert m.quants, "every model needs at least one quantization"
    if m.kv_scheme == "mla":
        assert m.kv_lora_rank, f"{m.id}: MLA requires kv_lora_rank or KV is off by ~50x"
    f = fit.solve(m, m.quants[0], _hw(96), 8192, 1)
    assert f.total_bytes > 0


def test_memory_scales_as_expected():
    m = catalog.get("qwen3-32b")
    hw = _hw(96)
    base = fit.solve(m, "q4", hw, 8192, 1)
    assert fit.solve(m, "q4", hw, 16384, 1).kv_bytes == 2 * base.kv_bytes
    assert fit.solve(m, "q4", hw, 8192, 4).kv_bytes == 4 * base.kv_bytes
    assert fit.solve(m, "q8", hw, 8192, 1).weight_bytes > base.weight_bytes


def test_tiny_machine_fits_nothing_but_does_not_crash():
    feasible, rejected = fit.rank(_hw(2), 8192, 1)
    assert feasible == []
    assert len(rejected) == len(catalog.load())
    assert all(why for _, why in rejected), "every rejection needs a reason"


def test_big_machine_fits_more_than_small_one():
    small, _ = fit.rank(_hw(24), 8192, 1)
    big, _ = fit.rank(_hw(192), 8192, 1)
    assert len(big) > len(small)


def test_moe_sizes_on_total_not_active_params():
    """The single most common sizing error in the wild."""
    moe = catalog.get("qwen3-235b-a22b")
    w = moe.weight_bytes("q4")
    assert w > 100 * GB, "must size on 235B total"
    assert w > moe.active_b * 1e9, "must not size on 22B active"


def test_mla_kv_is_far_smaller_than_gqa():
    mla = catalog.get("deepseek-v3")
    gqa = catalog.get("llama-3.3-70b")
    assert mla.kv_bytes_per_token() < gqa.kv_bytes_per_token() / 2


def test_quant_preference_caps_at_8bit():
    assert fit.quant_preference(("fp16", "q8", "q4")) == ["q8", "q4"]
    assert fit.quant_preference(("fp16", "bf16")) in (["fp16", "bf16"], ["bf16", "fp16"])
    assert fit.quant_preference(("q4",)) == ["q4"]


def test_runtime_recommendation_follows_workload_not_just_hardware():
    apple = _hw(96)
    assert fit.recommend_runtime(apple, 8192, 32)[0] == "vllm-mlx"
    assert fit.recommend_runtime(apple, 8192, 1)[0] == "llama.cpp (Metal)"
    assert fit.recommend_runtime(apple, 131072, 1)[0] == "mlc-llm"

    single = _hw(80, kind="nvidia", bw=None)
    assert fit.recommend_runtime(single, 8192, 1)[0] == "vllm"
    assert fit.recommend_runtime(single, 8192, 16)[0] == "sglang"
    assert (
        fit.recommend_runtime(_hw(320, kind="nvidia", bw=None, devices=4), 8192, 1)[0]
        == "llm-d + GAIE"
    )

    assert all(why for _, why in [fit.recommend_runtime(apple, 8192, n) for n in (1, 8)])


def test_max_context_and_concurrency_trade_off():
    m, hw = catalog.get("qwen3-32b"), _hw(96)
    assert fit.max_context(m, "q4", hw, 1) > fit.max_context(m, "q4", hw, 8)
    assert fit.max_context(m, "q4", hw, 1) <= m.max_context
    assert fit.max_concurrency(m, "q4", hw, 4096) > fit.max_concurrency(m, "q4", hw, 32768)
    # A model whose weights alone exceed memory has zero of both.
    assert fit.max_context(catalog.get("kimi-k3"), "q4", hw, 1) == 0


def test_parse_size():
    assert cli._parse_size("32k") == 32768
    assert cli._parse_size("128000") == 128000
    assert cli._parse_size("1m") == 1048576


def test_cli_runs(capsys):
    assert cli.main(["fit", "--context", "8k", "--quiet"]) == 0
    assert "runtime ->" in capsys.readouterr().out
    assert cli.main(["models"]) == 0
    capsys.readouterr()  # drop, so the JSON parse below sees only JSON

    assert cli.main(["fit", "--json"]) == 0

    import json

    payload = json.loads(capsys.readouterr().out)
    assert "hardware" in payload and "feasible" in payload and "runtime" in payload

    # Unknown model is a clean error, not a traceback.
    assert cli.main(["fit", "--explain", "no-such-model"]) == 2


def test_explain_shows_the_arithmetic():
    f = fit.solve(catalog.get("qwen3-32b"), "q4", _hw(96), 8192, 1)
    text = f.explain()
    for token in ("weights", "kv cache", "overhead", "required", "usable", "headroom"):
        assert token in text
    assert "GQA" in text


def test_unverified_models_are_flagged():
    unverified = [m for m in catalog.load() if not m.verified]
    assert unverified, "catalog should be honest about what isn't confirmed"
    f = fit.solve(unverified[0], unverified[0].quants[0], _hw(96), 8192, 1)
    assert "unverified" in f.explain()


# --------------------------------------------------------------------------- #
# Surfaces: SDK and MCP
# --------------------------------------------------------------------------- #


def test_sdk_and_mcp_self_checks():
    from clickllm import mcp, sdk

    sdk.demo()
    mcp.demo()


def test_sdk_labels_every_throughput_figure_as_an_estimate():
    from clickllm import sdk

    r = sdk.fit(context="8k", concurrency=2, hw=_hw(96))
    assert r.feasible, "96 GB should fit something"
    for c in r.feasible:
        assert c.estimate_basis == sdk.ESTIMATE_BASIS
        assert "not measured" in c.estimate_basis


def test_sdk_rejects_nonsense_inputs_rather_than_guessing():
    from clickllm import sdk

    for kwargs in ({"concurrency": 0}, {"concurrency": -3}, {"context": "0"}):
        with pytest.raises(ValueError):
            sdk.fit(**kwargs)


def test_sdk_commercially_clean_requires_licence_and_verified_architecture():
    from clickllm import sdk

    r = sdk.fit(context="8k", hw=_hw(192))
    for c in r.commercially_clean():
        assert c.license_clean_commercial and c.architecture_verified
    # An unverified-architecture model must never be called clean.
    assert all(c.architecture_verified for c in r.commercially_clean())


def test_sdk_best_prefers_a_fast_candidate_but_still_answers_when_all_are_slow():
    from clickllm import sdk

    r = sdk.fit(context="8k", hw=_hw(96))
    b = r.best()
    assert b is not None and b in r.feasible
    if any(not c.slow for c in r.feasible):
        assert not b.slow, "a non-slow candidate existed and was not chosen"


def test_sdk_report_serialises():
    import json

    from clickllm import sdk

    d = sdk.fit(context="8k", hw=_hw(96)).to_dict()
    json.dumps(d)  # must not raise
    assert {"hardware", "feasible", "rejected", "runtime"} <= d.keys()


def test_mcp_exposes_no_write_tools():
    """An agent may analyse and recommend; a human moves production traffic."""
    from clickllm import mcp

    listed = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    names = [t["name"] for t in listed["result"]["tools"]]
    assert names, "server must expose tools"
    for forbidden in ("cutover", "apply", "deploy", "promote", "rollback"):
        assert not any(forbidden in n for n in names), f"{forbidden} must not be a tool"


def test_mcp_unknown_tool_and_unknown_model_are_handled_distinctly():
    from clickllm import mcp

    # An unknown *tool* is a protocol error.
    bad_tool = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "nope", "arguments": {}},
        }
    )
    assert bad_tool["error"]["code"] == -32602

    # An unknown *model* is a normal tool outcome the agent can recover from.
    bad_model = mcp.handle(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "clickllm_explain", "arguments": {"model_id": "no-such"}},
        }
    )
    assert bad_model["result"]["isError"] is True
    assert "error" in bad_model["result"]["content"][0]["text"]


def test_mcp_tool_schemas_are_well_formed():
    from clickllm import mcp

    for name, (_, schema) in mcp.TOOLS.items():
        assert schema["description"].strip(), f"{name} needs a description"
        assert schema["inputSchema"]["type"] == "object", name
