"""Cluster-aware sizing, the CRD reconcile, and the controller loop.

Each module's `demo()` walks a worked example. These pin the things that are
easy to get quietly wrong: unit ambiguity in node labels, node selection
determinism, and the rule that the controller may lower a phase but never
raise it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from clickllm.k8s.controller import Kubectl, reconcile_once
from clickllm.k8s.nodes import (
    BYTES_THRESHOLD_MIB,
    GPU_MEMORY_LABEL,
    GPU_PRODUCT_LABEL,
    TPU_ACCELERATOR_LABEL,
    TPU_TOPOLOGY_LABEL,
    from_json,
)
from clickllm.k8s.reconcile import PHASE_ORDER, reconcile, select_node


def node_json(name, **kw):
    labels = kw.pop("labels", {})
    alloc = kw.pop("alloc", {})
    return {"metadata": {"name": name, "labels": labels}, "status": {"allocatable": alloc}}


def gpu(name="h100", mem="81559", count="8", **labels):
    return node_json(
        name,
        labels={GPU_MEMORY_LABEL: mem, GPU_PRODUCT_LABEL: "NVIDIA-H100-80GB-HBM3", **labels},
        alloc={"nvidia.com/gpu": count, "memory": "2Ti", "cpu": "192"},
    )


CLUSTER = from_json({"items": [gpu()]})


def workload(**over):
    return {
        "metadata": {"name": "triage", "namespace": "ml"},
        "spec": {"model": "Qwen/Qwen3-32B", "workload": "interactive", **over},
    }


# --- the unit ambiguity that would ruin every estimate -------------------------


def test_gpu_memory_in_mib_is_read_as_mib():
    n = from_json({"items": [gpu(mem="81559")]})[0]
    assert n.device_bytes == 81559 * 1024**2
    assert 79 < n.device_bytes / 1024**3 < 80
    assert not n.note


def test_gpu_memory_in_bytes_is_detected_not_multiplied_again():
    # The known GFD bug. Read as MiB this would be 42 petabytes and every model
    # would "fit" — the most dangerous possible failure for a sizing tool.
    n = from_json({"items": [gpu(mem="42505273344")]})[0]
    assert 39 < n.device_bytes / 1024**3 < 41
    assert "read as bytes" in n.note
    assert n.schedulable, "a reinterpreted unit is a caveat, not a refusal"


@pytest.mark.parametrize("raw", ["", "not-a-number", "0", "-5"])
def test_an_unusable_memory_label_refuses_rather_than_defaults(raw):
    n = from_json({"items": [gpu(mem=raw)]})[0]
    assert n.device_bytes is None
    assert not n.schedulable and n.to_hardware() is None
    assert n.unknown


def test_the_bytes_threshold_is_where_the_constant_says():
    below = from_json({"items": [gpu(mem=str(BYTES_THRESHOLD_MIB - 1))]})[0]
    above = from_json({"items": [gpu(mem=str(BYTES_THRESHOLD_MIB))]})[0]
    assert not below.note, "just under the threshold is still MiB"
    assert "read as bytes" in above.note


# --- TPU nodes -----------------------------------------------------------------


def test_tpu_chips_come_from_topology_and_memory_from_the_generation():
    n = from_json(
        {
            "items": [
                node_json(
                    "tpu",
                    labels={
                        TPU_ACCELERATOR_LABEL: "tpu-v6e-slice",
                        TPU_TOPOLOGY_LABEL: "2x4",
                    },
                    alloc={"google.com/tpu": "8", "memory": "1500Gi", "cpu": "180"},
                )
            ]
        }
    )[0]
    assert n.kind == "tpu" and n.devices == 8
    assert n.device_bytes == 32 * 1024**3
    assert n.bandwidth_gbps == 1638.0


def test_an_unrecognised_tpu_generation_refuses_rather_than_guessing():
    n = from_json(
        {
            "items": [
                node_json(
                    "tpu",
                    labels={TPU_ACCELERATOR_LABEL: "tpu-v99-slice"},
                    alloc={"google.com/tpu": "4", "memory": "1Ti"},
                )
            ]
        }
    )[0]
    assert n.device_bytes is None and not n.schedulable
    assert "unrecognised" in n.unknown


# --- node selection ------------------------------------------------------------


def test_node_selection_is_deterministic():
    # A controller that picked differently each pass would rewrite the
    # Deployment forever and never converge.
    nodes = from_json({"items": [gpu("a"), gpu("b"), gpu("c", mem="23034", count="1")]})
    first = select_node(nodes)[0].name
    assert all(select_node(list(p))[0].name == first for p in (nodes, reversed(nodes), nodes[::-1]))


def test_the_largest_usable_accelerator_wins():
    nodes = from_json({"items": [gpu("small", mem="23034", count="1"), gpu("big")]})
    assert select_node(nodes)[0].name == "big"


def test_a_selector_miss_names_what_it_missed():
    node, why = select_node(CLUSTER, {"pool": "absent"})
    assert node is None and "nodeSelector" in why


def test_cpu_only_and_unsized_clusters_give_different_reasons():
    cpu = from_json({"items": [node_json("plain", alloc={"memory": "64Gi", "cpu": "16"})]})
    assert "CPU" in select_node(cpu)[1]

    unsized = from_json({"items": [gpu(mem="")]})
    assert "could not be sized" in select_node(unsized)[1]

    assert "no nodes" in select_node([])[1]


# --- reconcile -----------------------------------------------------------------


def test_the_engine_is_derived_and_the_deployment_carries_its_reasoning():
    r = reconcile(workload(), CLUSTER)
    assert r.ready and r.status["engine"] == "vllm"
    ann = r.objects[0]["metadata"]["annotations"]
    assert ann["clickllm.dev/engine"] == "vllm"
    assert "standalone" in " ".join(ann)
    # Every knob explains itself — the reason the CRD is worth having.
    assert r.status["knobs"] and all(len(k["why"]) > 30 for k in r.status["knobs"])


def test_prefix_sharing_changes_the_engine_the_image_and_the_flags():
    r = reconcile(workload(prefixSharing=0.9), CLUSTER)
    assert r.status["engine"] == "sglang"
    c = r.objects[0]["spec"]["template"]["spec"]["containers"][0]
    assert c["image"].startswith("lmsysorg/")
    assert "--context-length" in c["args"] and "--max-model-len" not in c["args"]


def test_an_unverifiable_flag_is_reported_as_a_gap_not_emitted():
    r = reconcile(workload(prefixSharing=0.9, structuredOutput=True), CLUSTER)
    assert any("structured_output" in g for g in r.status["gaps"])
    args = " ".join(r.objects[0]["spec"]["template"]["spec"]["containers"][0]["args"])
    assert "grammar-backend" not in args


def test_gpu_count_follows_the_tensor_parallel_decision_not_the_node():
    # Claiming all 8 cards for a model that fits on one is how a cluster runs
    # out of GPUs it actually had.
    r = reconcile(workload(concurrency=4, context=4096), CLUSTER)
    limits = r.objects[0]["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]
    assert int(limits["nvidia.com/gpu"]) <= 8


def test_a_missing_model_is_a_condition_not_a_crash():
    r = reconcile({"metadata": {"name": "x"}, "spec": {"workload": "batch"}}, CLUSTER)
    assert not r.ready and r.status["conditions"][0]["reason"] == "NoModel"


def test_an_unsizable_cluster_still_produces_a_status_that_explains_itself():
    r = reconcile(workload(), from_json({"items": [gpu(mem="")]}))
    assert not r.ready
    msg = r.status["conditions"][0]["message"]
    assert "GPU Feature Discovery" in msg


def test_a_node_caveat_reaches_the_status_warnings():
    r = reconcile(workload(), from_json({"items": [gpu(mem="42505273344")]}))
    assert any("read as bytes" in w for w in r.status["warnings"])


# --- the asymmetry -------------------------------------------------------------


@pytest.mark.parametrize(
    ("phase", "expect"),
    [("cut", "canary"), ("canary", "shadow"), ("shadow", "off"), ("off", None)],
)
def test_a_regression_lowers_the_phase_exactly_one_step(phase, expect):
    r = reconcile(workload(phase=phase), CLUSTER, regressed=True, regression_reason="r")
    assert r.demote_to == expect


def test_the_controller_never_raises_a_phase():
    # Walk every phase, healthy and regressed. Nothing may ever propose a phase
    # further along the ladder than the one in spec.
    for phase in PHASE_ORDER:
        for regressed in (False, True):
            r = reconcile(workload(phase=phase), CLUSTER, regressed=regressed)
            if r.demote_to is None:
                continue
            assert PHASE_ORDER.index(r.demote_to) < PHASE_ORDER.index(phase), (
                f"{phase} → {r.demote_to} is an escalation"
            )


def test_a_rollback_says_that_raising_it_again_is_a_human_decision():
    r = reconcile(workload(phase="canary"), CLUSTER, regressed=True, regression_reason="x")
    assert any("human decision" in c["message"] for c in r.status["conditions"])


# --- the controller loop -------------------------------------------------------


class Recorder(Kubectl):
    """Records kubectl calls instead of making them."""

    CALLS: list = []
    ITEMS: list = []

    def run(self, args, stdin=None):  # type: ignore[override]
        Recorder.CALLS.append((list(args), stdin))
        if args[0] == "get":
            return json.dumps({"items": Recorder.ITEMS})
        return ""


def run_loop(items, nodes=CLUSTER):
    Recorder.CALLS, Recorder.ITEMS = [], items
    return dict(reconcile_once(Recorder(), nodes=nodes)), Recorder.CALLS


def test_one_broken_workload_does_not_stop_the_others():
    got, _ = run_loop([workload(), {"metadata": {"name": "bad", "namespace": "ml"}, "spec": {}}])
    assert got["ml/triage"].ready
    assert not got["ml/bad"].ready


def test_a_healthy_pass_never_writes_spec():
    _, calls = run_loop([workload()])
    spec_writes = [s for a, s in calls if a[0] == "patch" and s and "spec" in str(s)]
    assert not spec_writes
    assert any(a[0] == "apply" for a, _ in calls)
    assert any("--subresource=status" in a for a, _ in calls)


def test_a_failure_is_reported_on_the_resource_not_only_logged():
    class Failing(Recorder):
        def run(self, args, stdin=None):
            if args[0] == "apply":
                raise RuntimeError("admission webhook denied the request")
            return super().run(args, stdin)

    Recorder.CALLS, Recorder.ITEMS = [], [workload()]
    got = dict(reconcile_once(Failing(), nodes=CLUSTER))
    cond = got["ml/triage"].status["conditions"][0]
    assert cond["reason"] == "ReconcileFailed"
    assert "admission webhook" in cond["message"]


# --- the shipped CRD -----------------------------------------------------------


def test_the_crd_manifest_declares_what_the_reconciler_reads_and_writes():
    text = (pathlib.Path(__file__).resolve().parents[1] / "deploy" / "crd.yaml").read_text()
    # Spec fields the reconciler actually consumes.
    for field in (
        "model",
        "workload",
        "concurrency",
        "context",
        "prefixSharing",
        "structuredOutput",
        "phase",
        "nodeSelector",
    ):
        assert f"{field}:" in text, f"CRD does not declare spec.{field}"
    # Status fields it writes.
    for field in ("engine", "engineReason", "knobs", "gaps", "warnings", "conditions"):
        assert f"{field}:" in text, f"CRD does not declare status.{field}"
    assert "subresources" in text and "status: {}" in text
    # Every phase the reconciler can produce must be a legal enum value.
    for phase in PHASE_ORDER:
        assert phase in text
