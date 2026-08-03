"""The self-checks in each module are the real gate; this runs them under pytest
plus the cases that need a synthetic machine rather than the host's."""

import re

import pytest

from clickllm import catalog, cli, fit, hardware, sdk
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


def test_the_engine_fit_names_is_the_engine_run_starts():
    """These were two separate tables and they disagreed.

    On Apple silicon at concurrency 8, `clickllm fit` printed
    `runtime -> vllm-mlx` while `clickllm run` on the same box started `mlx` and
    explained that "the CUDA engines cannot run here at all" — the second command
    refuting the first in its own output. Three of four hardware/workload shapes
    disagreed.

    The old assertions are why it survived: they pinned `vllm-mlx` and `mlc-llm`,
    so the suite defended the wrong answer. Neither name exists — vLLM has no
    Metal backend, and no adapter in this repo can configure either, so `fit` was
    recommending software to install that cannot be installed, on the CLI, the
    MCP server and the SDK alike.
    """
    from clickllm.engines import adapter_for
    from clickllm.plan import Requirements, Workload, _pick_engine

    shapes = [
        (_hw(96), 8192, 32),
        (_hw(96), 8192, 1),
        (_hw(96), 131072, 1),
        (_hw(80, kind="nvidia", bw=None), 8192, 1),
        (_hw(80, kind="nvidia", bw=None), 8192, 16),
        (_hw(320, kind="nvidia", bw=None, devices=4), 8192, 1),
    ]
    for hw, ctx, conc in shapes:
        named, why = fit.recommend_runtime(hw, ctx, conc)
        assert why, "a recommendation with no reason is not one"

        # 1. Never name something that cannot become a running server. This is
        #    the whole point: `clickllm run` refuses an engine with no verified
        #    flag dialect, so naming one guarantees the next command fails.
        assert adapter_for(named) is not None, (
            f"recommended {named!r}, which has no adapter — `run` would refuse it"
        )

        # 2. Where the planner's choice IS launchable, the two must agree — no
        #    second opinion, which is what made them drift apart before.
        best, _ = _pick_engine(hw, Requirements(Workload.INTERACTIVE, conc, ctx))
        if adapter_for(str(best)) is not None:
            assert named == str(best), f"{hw.kind} c={conc}: fit {named}, run {best}"
        else:
            # 3. And where it is not, say so rather than quietly substituting.
            assert str(best) in why, f"substituted {named} for {best} without saying so"


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
    from clickllm import advise, mcp, sdk, session, watch

    sdk.demo()
    mcp.demo()
    advise.demo()
    watch.demo()
    session.demo()


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


# --------------------------------------------------------------------------- #
# M7 distill / M8 prove
# --------------------------------------------------------------------------- #


def test_distill_and_prove_self_checks():
    import clickllm.prove as suite
    from clickllm.distill import cluster as dcluster
    from clickllm.distill import shape as dshape
    from clickllm.prove import equivalence, graders, judge, stats

    dshape.demo()
    dcluster.demo()
    stats.demo()
    graders.demo()
    judge.demo()
    equivalence.demo()
    suite.demo()


def test_a_perfect_score_on_few_samples_is_not_certainty():
    from clickllm.prove.stats import wilson

    i = wilson(8, 8)
    assert i.point == 1.0
    assert i.low < 0.75, "8/8 must not read as near-certain"
    assert not i.clearly_above(0.95), "a 95% gate must not open on 8 samples"


def test_ungraded_items_never_count_as_passes():
    """The single most dangerous failure mode in the grader stack."""
    from clickllm.prove.graders import EvalItem, grade

    r = grade(EvalItem("i", "c", "prompt", baseline="", candidate=""))
    assert not r.graded
    assert not r.passed


def test_position_bias_is_flagged_not_averaged():
    from clickllm.prove.graders import EvalItem
    from clickllm.prove.judge import Reply, Verdict, judge_item

    item = EvalItem("i", "c", "p", baseline="A", candidate="B")
    biased = judge_item(item, lambda c: Reply("a"), model="m")
    assert biased.position_bias
    assert biased.verdict is Verdict.UNCERTAIN
    # An ambiguous instrument must not be scored against the candidate.
    assert not biased.to_score().applicable


def test_judge_requires_disclosure():
    from clickllm.prove.graders import EvalItem
    from clickllm.prove.judge import Reply, judge_item

    with pytest.raises(ValueError):
        judge_item(EvalItem("i", "c", "p", "a", "b"), lambda c: Reply("tie"), model="")


def test_unmeasured_judge_agreement_is_not_assumed_perfect():
    from clickllm.prove.judge import Agreement

    a = Agreement(0, 0, "m")
    assert a.rate is None and not a.trustworthy
    assert "UNMEASURED" in a.render()


def test_regret_excludes_merely_unproven_clusters():
    """Thin evidence means gather more, not give up — otherwise traffic never moves."""
    from clickllm.prove.equivalence import CandidateReport, ClusterScore
    from clickllm.prove.stats import wilson

    regressed = ClusterScore("a", "bad", 0.3, wilson(20, 100), 0)
    thin = ClusterScore("b", "thin", 0.3, wilson(3, 4), 0)
    good = ClusterScore("c", "good", 0.4, wilson(98, 100), 0)
    cand = CandidateReport("m", (regressed, thin, good))

    assert [c.name for c in cand.regret()] == ["bad"]
    assert "thin" in [c.name for c in cand.unproven()]
    assert cand.movable_share() == pytest.approx(0.4)


def test_no_cost_rate_means_no_fabricated_saving():
    from clickllm.prove.equivalence import CandidateReport, ClusterScore, Matrix
    from clickllm.prove.stats import wilson

    c = ClusterScore("a", "x", 1.0, wilson(99, 100), 0)
    cand = CandidateReport("m", (c,))  # no monthly_cost
    policy = Matrix([cand], incumbent_cost=1000.0).hybrid_for(cand)
    assert policy.monthly_saving is None


def test_matrix_puts_regret_before_the_table():
    from clickllm.prove.equivalence import CandidateReport, ClusterScore, Matrix
    from clickllm.prove.stats import wilson

    bad = ClusterScore("a", "long-ctx", 0.2, wilson(10, 100), 0)
    ok = ClusterScore("b", "codegen", 0.8, wilson(98, 100), 0)
    text = Matrix([CandidateReport("m", (bad, ok))]).render()
    assert text.index("REGRET") < text.index("codegen")


def test_clustering_is_deterministic_regardless_of_arrival_order():
    from clickllm.distill.cluster import cluster
    from clickllm.distill.shape import Capture

    caps = [
        Capture(
            f"r{i}",
            "gpt-5",
            ({"role": "system", "content": "s" + str(i % 3)},),
            response="x" * (i + 1),
            prompt_tokens=500,
        )
        for i in range(30)
    ]
    a = [c.key for c in cluster(caps)]
    b = [c.key for c in cluster(list(reversed(caps)))]
    assert a == b
    names = [c.name for c in cluster(caps)]
    assert len(set(names)) == len(names), "cluster names must be unique"


# --------------------------------------------------------------------------- #
# Model -> hardware ("where")
# --------------------------------------------------------------------------- #


def test_hardware_catalog_self_check():
    from clickllm import hardware_catalog

    hardware_catalog.demo()


def test_tensor_parallel_aggregates_bandwidth():
    """Treating a 4-GPU node as one device understates decode ~3.5x."""
    from clickllm.hardware_catalog import get

    one, four = get("h100"), get("h100-x4")
    assert four.effective_bandwidth_gbps > one.bandwidth_gbps * 3
    assert four.effective_bandwidth_gbps < one.bandwidth_gbps * 4, "all-reduce is not free"

    m = catalog.get("qwen3-32b")
    p1 = [p for p in fit.where(m, 8192) if p.profile_id == "h100"][0]
    p4 = [p for p in fit.where(m, 8192) if p.profile_id == "h100-x4"][0]
    assert p4.tokens_per_sec > p1.tokens_per_sec * 3


def test_where_sorts_feasible_first_then_by_price():
    m = catalog.get("qwen3-32b")
    places = fit.where(m, 32768)
    feasible = [p for p in places if p.feasible]
    assert feasible, "a 32B model must run somewhere"
    assert not places[-1].feasible, "infeasible must sort last"
    priced = [p.hourly_usd for p in feasible if p.hourly_usd is not None]
    assert priced == sorted(priced), "cheapest capable first"


def test_where_explains_every_rejection():
    places = fit.where(catalog.get("kimi-k3"), 8192)
    assert not any(p.feasible for p in places), "a 2.8T model fits nothing here"
    for p in places:
        assert p.reason, f"{p.profile_id} rejected with no reason"
        assert "weights alone" in p.reason


def test_where_distinguishes_weights_too_big_from_kv_too_big():
    """'Buy a different card' and 'lower your context' are different answers."""
    m = catalog.get("qwen3-32b")
    tight = fit.where(m, 131072, concurrency=16)
    kv_bound = [p for p in tight if not p.feasible and "short by" in p.reason]
    assert kv_bound, "a huge-context request should be KV-bound somewhere, not weight-bound"


def test_cost_per_mtok_rewards_bandwidth_not_just_capacity():
    m = catalog.get("qwen3-32b")
    by_id = {p.profile_id: p for p in fit.where(m, 8192)}
    l4, r5090 = by_id["l4"], by_id["rtx-5090"]
    if l4.feasible and r5090.feasible:
        # Same capacity class, 6x the bandwidth — cost per token must reflect it.
        assert r5090.cost_per_mtok_usd < l4.cost_per_mtok_usd


def test_where_cli(capsys):
    assert cli.main(["where", "qwen3-32b", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "$/Mtok" in out and "roofline estimates" in out

    assert cli.main(["where", "qwen3-32b", "--json"]) == 0
    import json

    payload = json.loads(capsys.readouterr().out)
    assert payload["model"] == "qwen3-32b"
    assert any(p["feasible"] for p in payload["placements"])
    assert all(p["reason"] for p in payload["placements"] if not p["feasible"])

    assert cli.main(["where", "no-such-model"]) == 2


# --------------------------------------------------------------------------- #
# Catalogue verification and discovery
# --------------------------------------------------------------------------- #


def test_catalog_update_self_check():
    from clickllm import catalog_update

    catalog_update.demo()


def test_config_parsing_never_guesses_a_missing_field():
    """A guessed geometry produces a confident, wrong memory figure."""
    from clickllm.catalog_update import ConfigError, parse_config

    for broken in (
        {},
        {"num_hidden_layers": 4},
        {"num_hidden_layers": 4, "num_attention_heads": 8},
    ):
        with pytest.raises(ConfigError):
            parse_config(broken)


def test_mla_is_detected_from_its_rank_not_inferred():
    from clickllm.catalog_update import parse_config

    base = {
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "head_dim": 128,
        "max_position_embeddings": 131072,
    }
    assert parse_config(base).kv_scheme == "mha"
    assert parse_config({**base, "kv_lora_rank": 512}).kv_scheme == "mla"


def test_significant_changes_are_the_ones_that_move_memory():
    from clickllm.catalog_update import FieldChange

    assert FieldChange("kv_heads", 8, 4).significant
    assert FieldChange("kv_scheme", "gqa", "mla").significant
    assert not FieldChange("max_context", 1, 2).significant


def test_module_import_opens_no_socket():
    """Air-gapped installs must be able to import this without reaching out."""
    import json

    from clickllm import catalog_update as cu

    called = []

    def spy(url):
        called.append(url)
        return json.dumps(
            {
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "head_dim": 64,
                "max_position_embeddings": 128,
            }
        )

    spec = catalog.get("qwen3-32b")
    cu.propose(spec, "a/b", spy)
    assert called, "the injected fetcher is the only path to the network"
    assert called[0].startswith("https://")


def test_a_fetch_failure_is_data_not_an_exception():
    from clickllm.catalog_update import propose

    def dead(_):
        raise OSError("offline")

    p = propose(catalog.get("qwen3-32b"), "a/b", dead)
    assert p.error and not p.has_changes


def test_discovery_excludes_known_repos_and_is_deterministic():
    import json

    from clickllm.catalog_update import discover

    index = json.dumps(
        [
            {"modelId": "org/known", "trendingScore": 9, "downloads": 1},
            {"modelId": "org/b", "trendingScore": 5, "downloads": 10},
            {"modelId": "org/a", "trendingScore": 5, "downloads": 99},
        ]
    )
    got = [d.repo for d in discover({"org/known"}, lambda _: index)]
    assert got == ["org/a", "org/b"]
    assert discover(set(), lambda _: index) == discover(set(), lambda _: index)


def test_catalog_repos_are_absent_rather_than_guessed():
    """Verifying against the wrong repo is worse than not verifying."""
    with_repo = [m for m in catalog.load() if m.repo]
    assert with_repo, "some entries should be verifiable"
    assert any(m.repo is None for m in catalog.load()), "unknown repos stay unknown"
    for m in with_repo:
        assert "/" in m.repo


def test_catalog_cli_requires_explicit_network_optin(capsys):
    assert cli.main(["catalog"]) == 0
    assert "opt-in" in capsys.readouterr().out
    assert cli.main(["discover"]) == 0
    assert "opt-in" in capsys.readouterr().out
    assert cli.main(["catalog", "--model", "no-such"]) == 2


# --------------------------------------------------------------------------- #
# Workbench
# --------------------------------------------------------------------------- #


def test_workbench_self_check():
    from clickllm import ui

    ui.demo()


def test_workbench_is_read_only():
    """It shows and explains. Downloading, deploying and moving traffic stay in
    the CLI, where a human runs them deliberately."""
    from clickllm import ui

    for verb in ("deploy", "apply", "promote", "cutover", "rollback", "pull", "delete"):
        assert not any(verb in route for route in ui.ROUTES)


def test_workbench_html_only_reads_fields_the_sdk_exposes():
    """The first render showed '—' for every throughput and 'unverified' for
    every model, because the HTML read field names the SDK does not have."""
    import re

    from clickllm import sdk, ui

    html = ui.ASSET.read_text()
    known = set(sdk.Candidate.__slots__) | {
        "model_id",
        "name",
        "id",
        "feasible",
        "quant",
        "total_gb",
        "reason",
        "tokens_per_sec",
        "hourly_usd",
        "usd_per_mtok",
        "placements",
        "params_b",
        "active_b",
        "license",
        "license_ok",
        "verified",
        "model",
        "arithmetic",
        "models",
        "is_moe",
        "max_context",
        "repo",
        "hardware",
        "context",
        "concurrency",
        "rejected",
        "runtime",
        "warnings",
        "usable_gb",
        "bandwidth_gbps",
        "note",
        "why",
        "headroom_gb",
        "length",
        "map",
        "join",
        "filter",
        "sort",
        "slice",
        "toLocaleString",
        "toFixed",
    }
    # Field accesses on a candidate row inside the fit table.
    used = set(re.findall(r"\bc\.([a-z_]+)\b", html))
    unknown = used - known
    assert not unknown, f"workbench reads fields the SDK does not expose: {sorted(unknown)}"


def test_workbench_routes_all_serialise():
    import json

    from clickllm import ui

    for path, fn in ui.ROUTES.items():
        q = {"model": ["qwen3-32b"]} if path.endswith(("where", "explain")) else {}
        json.dumps(fn(q), default=ui._json_default)


def test_workbench_reports_unknown_models_as_data_not_a_crash():
    from clickllm import ui

    with pytest.raises(KeyError):
        ui.ROUTES["/api/where"]({"model": ["no-such-model"]})


def test_a_refusal_never_claims_to_be_short_by_nothing():
    """`short by 0 GB` is not a reason — it reads as a solver bug.

    Qwen3-32B at q8 misses batch 19 by 0.49 GiB, which the old `,.0f` format
    rounded to a flat zero. Any real shortfall must render as a nonzero
    quantity in some unit.
    """
    m = catalog.get("qwen3-32b")
    ctx = sdk.parse_size("8k")
    seen = 0
    for conc in range(2, 60):
        for pl in fit.where(m, ctx, conc):
            if pl.feasible or not pl.reason:
                continue
            seen += 1
            assert not re.search(r"short by 0(\.0)? ", pl.reason), (
                f"shortfall rounded away at concurrency {conc}: {pl.reason!r}"
            )
    assert seen, "no refusals were produced; the check would be vacuous"


def test_shortfall_steps_down_a_unit_rather_than_rounding_to_zero():
    assert fit._shortfall(59 * 1024**2).endswith("MB")
    assert fit._shortfall(3 * 1024**3) == "3.0 GB"
    assert fit._shortfall(47 * 1024**3) == "47 GB"


def test_a_precision_label_is_never_emitted_as_a_quantisation_method():
    """`--quantization q4` is a flag the server rejects at startup.

    vLLM and SGLang take a METHOD (awq, gptq, fp8...), not a precision. awq,
    gptq, bitsandbytes and compressed-tensors are all "4-bit" and are not
    interchangeable, so q4 names none of them.

    `unknown_flags` cannot catch this — `--quantization` is a real flag and only
    its value is wrong — which is exactly why it needs a test of its own.
    """
    from clickllm.engines import Setting, SglangAdapter, Unsupported, VllmAdapter

    for adapter in (VllmAdapter(), SglangAdapter()):
        for label in ("q3", "q4", "q5", "q6", "q8", "fp16", "bf16"):
            out = adapter.translate(Setting.QUANTIZATION, label)
            assert isinstance(out, Unsupported), (
                f"{adapter.name} emitted {out} for {label!r}, which is a size not a method"
            )
        # fp8 really is a method name on both, so it is the one that passes.
        ok = adapter.translate(Setting.QUANTIZATION, "fp8")
        assert not isinstance(ok, Unsupported)
        assert ok.argv == ("--quantization", "fp8")


def test_the_cli_can_say_what_version_it_is():
    """A published CLI with no `--version` is not shippable — it is the first
    thing anyone runs after installing, and `clickllm version` answered with an
    argparse error over a list of twenty subcommands.

    Three spellings because people reach for all three, and the cost of guessing
    wrong is a bad first minute.
    """
    import clickllm

    assert clickllm.__version__ and clickllm.__version__ != "unknown"

    for argv in (["--version"], ["-V"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv)
        assert e.value.code == 0, argv

    assert cli.main(["version"]) == 0


def test_the_reported_version_is_the_packaged_one():
    """A version that drifts from pyproject is worse than none: it is what goes
    into a bug report."""
    import re
    from pathlib import Path

    import clickllm

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    assert m and clickllm.__version__ == m.group(1)


def test_upgrade_tells_you_how_rather_than_guessing():
    """clickllm may be installed by uv, pipx or pip, into a tool environment this
    process cannot see. A self-upgrade that guesses either fails confusingly or
    upgrades a different copy than the one running. Naming the commands is the
    honest answer and cannot corrupt anything."""
    assert cli.main(["upgrade"]) == 0
