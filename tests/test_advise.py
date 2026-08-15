"""Proactive suggestions and the self-healing seam.

`advise.demo()` walks the happy path. These pin the properties that make the
feature trustworthy rather than noisy: it stays quiet when there is nothing to
say, it never fires on normal variance, every item carries its evidence, and
nothing it returns can act on its own.
"""

from __future__ import annotations

from dataclasses import replace

from onpar import mcp
from onpar.advise import Impact, Observed, Suggestion, reconcile, suggest
from onpar.hardware import Hardware
from onpar.plan import Requirements, Workload, plan

HW = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * 1024**3,
    usable_bytes=96 * 1024**3,
    bandwidth_gbps=546.0,
    cores=16,
)

#: A deployment with nothing left unsaid — the control case.
TIGHT = Requirements(
    workload=Workload.INTERACTIVE,
    concurrency=4,
    context=8_192,
    ttft_ms=400,
    itl_ms=40,
    prefix_sharing=0.8,
)


def ids(items: list[Suggestion]) -> set[str]:
    return {s.id for s in items}


# --- staying quiet is a feature ------------------------------------------------


def test_a_well_specified_deployment_gets_a_short_list():
    """Padding the list trains people to skim past the one that mattered."""
    got = ids(suggest(TIGHT, plan(HW, TIGHT)))
    assert "measure-prefix-sharing" not in got
    assert "state-a-latency-budget" not in got
    assert "right-size-context" not in got


def test_reality_matching_the_plan_says_nothing():
    assert reconcile(TIGHT, plan(HW, TIGHT), Observed(concurrency=4)) == []


def test_normal_traffic_variance_does_not_fire():
    """5 against a plan of 4 is traffic. A suggestion that fires on noise gets
    mentally filtered out within a week, taking the real ones with it."""
    assert reconcile(TIGHT, plan(HW, TIGHT), Observed(concurrency=5)) == []
    assert reconcile(TIGHT, plan(HW, TIGHT), Observed(concurrency=6)) == []


def test_empty_telemetry_is_not_treated_as_a_zero_reading():
    """Absent is not zero. A missing metric must not read as a budget miss."""
    assert reconcile(TIGHT, plan(HW, TIGHT), Observed()) == []


def test_a_budget_that_is_met_is_not_reported_as_missed():
    seen = Observed(ttft_ms=200, itl_ms=20)
    assert reconcile(TIGHT, plan(HW, TIGHT), seen) == []


# --- firing when it should -----------------------------------------------------


def test_the_unset_prefix_sharing_knob_is_raised_unprompted():
    """The money case: nobody sets it because no form asks."""
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=8, context=8_192)
    assert "measure-prefix-sharing" in ids(suggest(loose, plan(HW, loose)))


def test_measured_sharing_the_plan_never_knew_about_is_surfaced():
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=4, context=8_192)
    found = reconcile(loose, plan(HW, loose), Observed(prefix_sharing=0.85))
    assert [s.id for s in found] == ["enable-prefix-reuse"]


def test_concurrency_far_above_plan_is_a_different_deployment():
    got = reconcile(TIGHT, plan(HW, TIGHT), Observed(concurrency=40))
    assert "replan-for-observed-concurrency" in ids(got)
    assert got[0].impact is Impact.HIGH


def test_kv_pressure_is_raised_before_it_becomes_latency_variance():
    got = reconcile(TIGHT, plan(HW, TIGHT), Observed(kv_utilisation=0.95))
    assert "kv-pressure" in ids(got)


def test_context_provisioned_for_traffic_that_never_arrived():
    got = reconcile(TIGHT, plan(HW, TIGHT), Observed(peak_context=2_000))
    assert "context-never-used" in ids(got)


def test_a_missed_ttft_budget_is_reported_against_the_stated_one():
    got = reconcile(TIGHT, plan(HW, TIGHT), Observed(ttft_ms=900))
    hit = next(s for s in got if s.id == "ttft-budget-missed")
    assert "900" in hit.because and "400" in hit.because


def test_batch_workloads_are_not_nagged_about_latency_budgets():
    """Unconstrained is a real answer for batch, not an omission."""
    batch = Requirements(workload=Workload.BATCH, concurrency=64, context=8_192)
    got = ids(suggest(batch, plan(HW, batch)))
    assert "state-a-latency-budget" not in got
    assert "measure-prefix-sharing" not in got


# --- the shape of a suggestion -------------------------------------------------


def test_every_suggestion_carries_its_evidence_and_hedges_its_effect():
    """A suggestion without its trigger is an assertion, and an assertion cannot
    be argued with — which is how people learn to ignore the whole feature."""
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=16, context=131_072)
    everything = suggest(loose, plan(HW, loose)) + reconcile(
        loose, plan(HW, loose), Observed(concurrency=80, ttft_ms=999, kv_utilisation=0.99)
    )
    assert everything
    for s in everything:
        assert s.because.strip(), s
        assert s.expect.strip(), s
        assert s.action.strip(), s
        assert s.id.strip(), s


def test_suggestions_are_ordered_by_impact():
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=16, context=131_072)
    got = suggest(loose, plan(HW, loose))
    rank = {Impact.HIGH: 0, Impact.MEDIUM: 1, Impact.LOW: 2}
    assert [rank[s.impact] for s in got] == sorted(rank[s.impact] for s in got)


def test_the_same_inputs_produce_the_same_list():
    """Stable, so a diff between two runs means something actually changed."""
    loose = Requirements(workload=Workload.INTERACTIVE, concurrency=16, context=131_072)
    a = [s.id for s in suggest(loose, plan(HW, loose))]
    b = [s.id for s in suggest(loose, plan(HW, loose))]
    assert a == b


def test_speculative_decoding_is_flagged_for_verification_at_real_batch():
    """Published EAGLE figures are single-stream — the trap this product exists
    to stop people walking into."""
    req = Requirements(
        workload=Workload.INTERACTIVE,
        concurrency=12,
        context=8_192,
        prefix_sharing=0.8,
        ttft_ms=400,
    )
    hit = next(
        (s for s in suggest(req, plan(HW, req)) if s.id == "verify-speculative-at-your-batch"),
        None,
    )
    assert hit is not None
    # The cutoff and the published crossover are different numbers; conflating
    # them would misreport why the plan switches it off.
    assert "32" in hit.because and "16" in hit.because


# --- the agent surface ---------------------------------------------------------


def test_the_agent_surface_is_read_only():
    """No MCP tool may be one that moves traffic.

    This used to check the *string literal* `"onpar_advise"` against the
    forbidden list, which is true by inspection and can never fail — adding a
    `onpar_deploy` tool tomorrow would have left it green. The boundary is
    only real if the check reads the live registry, so it does.

    `onpar run` and `onpar host` deliberately have no MCP tool. An agent
    may size, explain, advise and prove; starting a server or spending money
    stays a thing a human types.
    """
    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    assert mcp.TOOLS, "no tools registered; the check would be vacuous"
    offenders = {name: word for name in mcp.TOOLS for word in forbidden if word in name.lower()}
    assert not offenders, f"MCP tools that imply moving traffic: {offenders}"
    assert "onpar_advise" in mcp.TOOLS


def test_the_agent_gets_the_evidence_not_just_the_action():
    out = mcp._advise(context="128k", concurrency=16)
    assert out["suggestions"]
    for s in out["suggestions"]:
        assert s["because"] and s["expect"] and s["impact"] in {"high", "medium", "low"}
    assert "not actions" in out["advisory"]


def test_the_agent_gets_drift_only_when_it_supplies_telemetry():
    assert mcp._advise(context="8k", concurrency=4)["drift"] == []
    drifted = mcp._advise(context="8k", concurrency=4, observed={"concurrency": 40})
    assert [s["id"] for s in drifted["drift"]] == ["replan-for-observed-concurrency"]


# --- prefill/decode disaggregation: a topology, not a knob ---------------------

QUAD = Hardware(
    kind="nvidia",
    name="4xH100",
    total_bytes=320 * 1024**3,
    usable_bytes=288 * 1024**3,
    bandwidth_gbps=3350.0,
    cores=1,
    devices=4,
)

HEAVY = Requirements(
    workload=Workload.INTERACTIVE,
    concurrency=64,
    context=32_768,
    prefix_sharing=0.8,
    ttft_ms=500,
)


def test_pd_disaggregation_is_raised_when_prefill_and_decode_collide():
    hit = next(
        (s for s in suggest(HEAVY, plan(QUAD, HEAVY)) if s.id == "consider-pd-disaggregation"),
        None,
    )
    assert hit is not None
    # It must be honest that this is operational weight, not a flag flip.
    assert "experimental" in hit.expect
    assert "proxy" in hit.expect


def test_pd_disaggregation_does_not_fire_on_a_single_device():
    """Splitting prefill from decode needs somewhere to put each half."""
    got = {s.id for s in suggest(HEAVY, plan(HW, HEAVY))}
    assert "consider-pd-disaggregation" not in got


def test_pd_disaggregation_does_not_fire_at_low_concurrency_or_short_context():
    """Both conditions must hold: short prompts have little prefill to move, and
    low concurrency has nothing for it to collide with."""
    short = replace(HEAVY, context=2048)
    quiet = replace(HEAVY, concurrency=4)
    for req in (short, quiet):
        got = {s.id for s in suggest(req, plan(QUAD, req))}
        assert "consider-pd-disaggregation" not in got, req


def test_pd_disaggregation_is_never_suggested_for_batch():
    """Nobody is waiting on a batch job's first token, so the variance this fixes
    is not a cost worth operational weight."""
    batch = replace(HEAVY, workload=Workload.BATCH)
    got = {s.id for s in suggest(batch, plan(QUAD, batch))}
    assert "consider-pd-disaggregation" not in got
