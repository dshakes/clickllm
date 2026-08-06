"""Cluster-aware sizing, the CRD reconcile, and the controller loop.

Each module's `demo()` walks a worked example. These pin the things that are
easy to get quietly wrong: unit ambiguity in node labels, node selection
determinism, and the rule that the controller may lower a phase but never
raise it.
"""

from __future__ import annotations

import contextlib
import json
import pathlib
import subprocess
from unittest import mock

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
    assert "--context-length" in c["command"] and "--max-model-len" not in c["command"]


def test_an_unverifiable_flag_is_reported_as_a_gap_not_emitted():
    r = reconcile(workload(prefixSharing=0.9, structuredOutput=True), CLUSTER)
    assert any("structured_output" in g for g in r.status["gaps"])
    args = " ".join(r.objects[0]["spec"]["template"]["spec"]["containers"][0]["command"])
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


def _patch_bodies(calls) -> list[str]:
    """Bodies of every `kubectl patch`.

    They arrive in argv as `-p <json>`, not on stdin. An earlier version of
    these tests looked at stdin, where patch bodies never appear — so the
    "never writes spec" assertion below passed vacuously and would not have
    caught a controller that rewrote spec on every pass.
    """
    out = []
    for args, _stdin in calls:
        if args and args[0] == "patch" and "-p" in args:
            out.append(args[args.index("-p") + 1])
    return out


def test_a_healthy_pass_never_writes_spec():
    _, calls = run_loop([workload()])
    bodies = _patch_bodies(calls)
    assert bodies, "the pass should have written status at least"
    assert not [b for b in bodies if '"spec"' in b], bodies
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


# --- the four defects the PR reviewers caught ----------------------------------
# Each of these was shipped and blocked at review. They are pinned here because
# every one of them is invisible until production.


def test_the_rollback_path_is_actually_consulted_by_the_loop():
    """The worst of the four: documented at length, never wired.

    `reconcile_once` called `reconcile(wl, nodes)` with no regression input, so
    `regressed` was always False and the automatic rollback could not once have
    fired. Three paragraphs of documentation describing behaviour that did not
    exist.
    """
    wl = workload(phase="canary")
    got, calls = run_loop([wl])
    assert got["ml/triage"].demote_to is None, "no regression, no demotion"

    Recorder.CALLS, Recorder.ITEMS = [], [wl]
    got = dict(
        reconcile_once(
            Recorder(), nodes=CLUSTER, regression_check=lambda _: (True, "extract fell to 61%")
        )
    )
    r = got["ml/triage"]
    assert r.demote_to == "shadow", "a regression must lower the phase"
    # ...and it must reach the cluster, not just the return value.
    spec_patches = [b for b in _patch_bodies(Recorder.CALLS) if '"spec"' in b]
    assert spec_patches, "the demotion was computed but never applied"
    assert '"phase": "shadow"' in spec_patches[0]


def test_the_default_regression_check_reads_the_gates_verdict():
    from clickllm.k8s.controller import REGRESSION_ANNOTATION, gate_from_annotation

    assert gate_from_annotation({"metadata": {"annotations": {REGRESSION_ANNOTATION: "true"}}})[0]
    assert gate_from_annotation({})[0] is False


@pytest.mark.parametrize("value", ["", "false", "unknown", "TRUE-ish", "1", "yes", None])
def test_only_a_literal_true_triggers_a_rollback(value):
    # Rolling back on a value we do not understand would make a stray annotation
    # a production incident.
    from clickllm.k8s.controller import REGRESSION_ANNOTATION, gate_from_annotation

    ann = {} if value is None else {REGRESSION_ANNOTATION: value}
    assert gate_from_annotation({"metadata": {"annotations": ann}})[0] is False


def test_a_tpu_workload_produces_a_runnable_deployment():
    """Two bugs made this impossible: no adapter registered for vllm-tpu, and an
    argv slice that dropped the model name for anything not exactly VLLM."""
    tpu = from_json(
        {
            "items": [
                node_json(
                    "tpu-a",
                    labels={
                        TPU_ACCELERATOR_LABEL: "tpu-v6e-slice",
                        TPU_TOPOLOGY_LABEL: "2x4",
                    },
                    alloc={"google.com/tpu": "8", "memory": "1500Gi", "cpu": "180"},
                )
            ]
        }
    )
    r = reconcile(workload(), tpu)
    assert r.ready, r.status.get("conditions")
    assert r.status["engine"] == "vllm-tpu"
    c = r.objects[0]["spec"]["template"]["spec"]["containers"][0]
    # The model name must survive the argv slice — without it the container
    # boots with no model and crashes immediately.
    assert "Qwen/Qwen3-32B" in c["command"], c["command"]
    # The binary, not `serve`: vllm/vllm-openai's ENTRYPOINT is already
    # ["vllm", "serve"], so emitting the flags alone produced
    # `vllm serve serve MODEL` and the container never reached the engine.
    assert c["command"][:2] == ["vllm", "serve"], c["command"][:3]
    assert "args" not in c, "args would be appended to the image entrypoint, not replace it"
    assert "google.com/tpu" in c["resources"]["limits"]


@pytest.mark.parametrize(
    "bad",
    [
        {"concurrency": None},
        {"context": None},
        {"prefixSharing": None},
        {"context": "not-a-number"},
        {"workload": "nonsense"},
        {"workload": None},
        {"structuredOutput": None},
        {"prefixSharing": 7},
    ],
)
def test_a_malformed_field_cannot_crash_the_shared_controller(bad):
    """`concurrency: null` is valid YAML and reached `int(None)`, raising
    TypeError — uncaught, so anyone able to write an InferenceWorkload could
    crash-loop the cluster-wide daemon for everybody."""
    got, _ = run_loop([workload(**bad)])
    r = got["ml/triage"]
    assert r.status, "a malformed field must still produce a status"
    assert r.status["conditions"][0]["reason"] != "ReconcileFailed", r.status


def test_one_workload_raising_an_unexpected_error_does_not_stop_the_pass():
    class Exploding(Recorder):
        def run(self, args, stdin=None):
            if args[0] == "apply" and stdin and "triage" in stdin:
                raise TypeError("something nobody anticipated")
            return super().run(args, stdin)

    Recorder.CALLS, Recorder.ITEMS = [], [workload(), workload() | {}]
    Recorder.ITEMS = [
        workload(),
        {"metadata": {"name": "other", "namespace": "ml"}, "spec": dict(workload()["spec"])},
    ]
    got = dict(reconcile_once(Exploding(), nodes=CLUSTER))
    assert got["ml/triage"].status["conditions"][0]["reason"] == "ReconcileFailed"
    assert got["ml/other"].ready, "the other workload must still reconcile"


def test_the_module_entrypoint_runs_the_loop_not_the_self_check():
    # `deploy/README.md` tells operators to run `python -m clickllm.k8s.controller`.
    # It used to run demo() and exit.
    import inspect

    from clickllm.k8s import controller as mod

    assert hasattr(mod, "main")
    src = inspect.getsource(mod)
    tail = src[src.index('if __name__ == "__main__":') :]
    assert "main()" in tail and "demo()" not in tail
    # The self-check is still reachable, just not the default.
    assert "--self-check" in inspect.getsource(mod.main)


def test_every_engine_is_pointed_at_an_image_that_runs_its_own_argv():
    """The image has to be one where the argv we emit has a binary to be.

    llm-d is the case that got this wrong: it inherits vLLM's dialect, so its
    Deployment says `vllm serve …`, but the image was `ghcr.io/llm-d/llm-d`
    — a reference that issues no anonymous pull token, so the user got
    ImagePullBackOff. Tying the image to the dialect means the next person to
    change one has to look at the other.
    """
    from clickllm.engines import adapter_for
    from clickllm.k8s.reconcile import IMAGES
    from clickllm.plan import Engine

    assert IMAGES[Engine.LLMD] == IMAGES[Engine.VLLM], (
        "llm-d's data plane is vLLM, so it must run vLLM's image; "
        "if that stops being true, its dialect has to change too"
    )
    # And the dialect that justifies it is still vLLM's.
    assert adapter_for("llm-d").help_argv[0] == "vllm"


def test_every_catalogue_model_yields_a_name_a_service_will_accept():
    """A Deployment and its Service are named from the same string under two
    different rules, and only the Service refuses.

    Deployment names are DNS-1123 subdomains (dots legal); Service names are
    DNS-1035 labels (dots illegal). `Llama-3.1-8B-Instruct` therefore applied as
    a Deployment with a warning and was rejected as a Service — the pod ran and
    nothing could reach it. Four of the catalogue's models had a dot in their
    version. Checked here for all of them, offline, because the server-side
    dry-run that found it needs a cluster and this does not.
    """
    import re

    from clickllm import catalog
    from clickllm.k8s.reconcile import dns_label

    dns1035 = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
    for model in catalog.load():
        raw = (model.repo or model.id).split("/")[-1].lower()
        name = dns_label(raw)
        assert dns1035.match(name), f"{model.id}: {raw!r} -> {name!r} is not a DNS-1035 label"
        assert len(name) <= 63, f"{model.id}: {name!r} exceeds the 63-char limit"


def test_the_generated_service_and_deployment_agree_on_the_name():
    """The Service selects on the Deployment's labels and shares its name. If the
    sanitiser reached one object and not the other, the Service would resolve to
    no pods — which looks healthy and serves nothing."""
    from clickllm import catalog
    from clickllm.hardware import Hardware
    from clickllm.k8s.reconcile import deployment_for
    from clickllm.plan import Requirements, Workload, plan

    hw = Hardware(
        kind="nvidia",
        name="H100 80GB",
        total_bytes=80 * 2**30,
        usable_bytes=76 * 2**30,
        bandwidth_gbps=3350.0,
        cores=132,
    )
    model = catalog.get("llama-3.1-8b")
    p = plan(hw, Requirements(Workload.INTERACTIVE, 8), model, "fp16")
    dep, _ = deployment_for("llama-3.1-8b-instruct", "default", model.repo or model.id, p)
    assert dep["metadata"]["name"] == "llama-3-1-8b-instruct"
    assert dep["spec"]["selector"]["matchLabels"] == dep["spec"]["template"]["metadata"]["labels"]


@pytest.mark.parametrize("verb", ["apply", "patch", "delete", "replace", "annotate", "label"])
def test_dry_run_suppresses_every_mutating_verb_not_just_apply(verb):
    """`--dry-run` is documented as "applies nothing" and patched for real.

    The guard was `args[0] == "apply"`, so `patch` slipped through — and both
    `patch_status()` and `demote()` issue `patch`. An operator running
    `--dry-run --once` to preview a pass against a live cluster really wrote
    `status.conditions`, and really demoted `spec.phase` (canary -> shadow) when
    the gate saw a regression: a flag whose whole contract is "changes nothing"
    performing the one action in this system that moves production traffic.

    Parametrised over verbs the code does not currently issue, because the
    defect was an UNLISTED mutation. The guard now allow-lists reads, so a verb
    added later is suppressed until someone says otherwise.
    """
    from clickllm.k8s.controller import Kubectl

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    kc = Kubectl(dry_run=True)
    with mock.patch.object(subprocess, "run", fake_run), contextlib.suppress(Exception):
        kc.run([verb, "inferenceworkload/x", "-p", "{}"])
    assert "--dry-run=server" in seen.get("cmd", []), (
        f"{verb} ran without the dry-run flag: {seen.get('cmd')}"
    )


def test_dry_run_does_not_cripple_reads():
    """The control: suppressing writes must not suppress the reads a pass needs.

    A `get` that silently became a server-side dry run would make the loop see
    nothing and report a clean pass over a cluster it never looked at.
    """
    from clickllm.k8s.controller import Kubectl

    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    with mock.patch.object(subprocess, "run", fake_run):
        Kubectl(dry_run=True).run(["get", "inferenceworkloads", "-A", "-o", "json"])
    assert "--dry-run=server" not in seen.get("cmd", []), seen.get("cmd")


def test_the_operator_sizes_the_model_against_the_node_it_picked():
    """`reconcile` called `plan(hw, req)` with NO model, so `Plan.fit` was always
    None and no sizing happened at all.

    `select_node` read every node's accelerator memory, picked the largest, and
    that capacity was used for nothing — a Deployment for a 70B and one for an
    8B were planned identically on the same hardware, in the module whose
    docstring calls reconciling against real node capacity "the feature".
    See ADR-0013.
    """
    from clickllm.k8s.reconcile import reconcile

    def cond(wl):
        return (reconcile(wl, CLUSTER).status.get("conditions") or [{}])[0]

    # A request that cannot fit is now SAID to not fit, with the arithmetic.
    big = cond(workload(model="Qwen/Qwen3-32B", context=1000000))
    assert big["status"] == "False" and big["reason"] == "DoesNotFit", big
    assert "GiB" in big["message"], "the refusal must carry the arithmetic"

    # Control: an ordinary request still plans.
    ok = cond(workload(model="meta-llama/Llama-3.1-8B-Instruct"))
    assert ok["status"] == "True" and ok["reason"] == "Planned", ok


def test_a_repo_outside_the_catalogue_is_refused_with_the_remedy():
    """Sizing needs a `ModelSpec`; the CRD names a Hugging Face repo. A repo we
    do not carry is a stated unknown, never a pass (ADR-0012), because the
    alternative is applying a Deployment whose flags came from no hardware."""
    from clickllm.k8s.reconcile import reconcile

    r = reconcile(workload(model="acme/not-a-real-model"), CLUSTER)
    c = (r.status.get("conditions") or [{}])[0]
    assert not r.objects, "nothing may be applied for a model that was never sized"
    assert c["reason"] == "ModelNotInCatalogue"
    assert "catalog-add" in c["message"], "a refusal should name the way out"


def test_a_workload_declared_unfit_is_reported_but_not_applied():
    """The apply gate, live at last.

    Two earlier attempts at it were discarded as unreachable: nothing ever
    reported Ready=False while returning objects, because nothing was ever
    sized. Now that ADR-0013 sizes the model, this fires — and it is driven
    through `reconcile_once`, not through the predicate, because the first
    version of this test passed with the gate removed.
    """
    got, calls = run_loop([workload(model="Qwen/Qwen3-32B", context=1000000)])
    verbs = [c[0][0] for c in calls]
    assert "apply" not in verbs, f"applied a workload it declared unfit: {verbs}"
    assert "patch" in verbs, "the status explaining why must still be written"

    # Control: a workload that does fit is still applied.
    _, ok_calls = run_loop([workload(model="meta-llama/Llama-3.1-8B-Instruct")])
    assert "apply" in [c[0][0] for c in ok_calls]
