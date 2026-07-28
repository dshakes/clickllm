"""Reconcile an InferenceWorkload into a Deployment — as a pure function.

The reconcile loop is the part of an operator that everyone tests badly, because
it is usually welded to a client library and only runs against a real cluster. So
this half takes a spec and a list of nodes and returns the desired objects, with
no I/O anywhere in it. The loop that actually talks to the API server is thin
glue around it (see [`clickllm.k8s.controller`]).

## What makes this worth being a controller rather than a Deployment

Nothing, if it asked for an image and a command. The reason to run an operator is
that **the answer changes when the cluster changes**: a workload sized for an
H100 node that gets rescheduled onto an L4 needs different flags, and a
Deployment cannot know that. Reconciling against real node capacity is the
feature; the CRD is just how you ask for it.

## The asymmetry, again

`spec.phase` is owned by the human. This module will **lower** it — that is a
rollback, and it must work unattended — and will never raise it. The same rule
holds in `prove.gate` (advancing is a proposal) and in the gateway's control
surface (escalation needs `confirmed: true`). Three independent implementations
of one rule, because the cost of getting it wrong is production traffic on an
unproven model and no single layer should be the only thing standing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from clickllm.engines import Setting
from clickllm.k8s.nodes import Node
from clickllm.plan import Engine, Plan, Requirements, Workload, plan

__all__ = [
    "PHASE_ORDER",
    "Reconciled",
    "demo",
    "deployment_for",
    "reconcile",
    "select_node",
]

#: Rollout phases, least to most exposure. Ordering is what makes "lower" and
#: "raise" meaningful rather than a comparison of strings.
PHASE_ORDER = ("off", "shadow", "canary", "cut")

#: Container images per engine. Pinned to the engines' own published images —
#: clickllm never wraps an engine (NFR-4), so these are upstream, not ours.
IMAGES = {
    Engine.VLLM: "vllm/vllm-openai:latest",
    Engine.VLLM_TPU: "vllm/vllm-tpu:latest",
    Engine.SGLANG: "lmsysorg/sglang:latest",
    Engine.LLMD: "ghcr.io/llm-d/llm-d:latest",
}

#: Port each engine serves on by default.
PORTS = {Engine.SGLANG: 30000}
DEFAULT_PORT = 8000


@dataclass(frozen=True, slots=True)
class Reconciled:
    """What the controller should apply, and what it should report.

    `objects` may be empty while `status` is populated — a workload that cannot
    be sized still produces a status explaining why, which is more useful than
    an empty result and a log line nobody reads.
    """

    objects: tuple[dict[str, Any], ...] = ()
    status: dict[str, Any] = field(default_factory=dict)
    #: Phase the controller wants in `spec`, when it differs. Only ever lower.
    demote_to: str | None = None

    @property
    def ready(self) -> bool:
        """Whether this produced something applyable."""
        return bool(self.objects)


def select_node(
    nodes: list[Node], selector: dict[str, str] | None = None
) -> tuple[Node | None, str]:
    """Pick the node class to size against, and say why.

    Largest usable accelerator memory wins, because the binding constraint is
    almost always whether the model plus its KV cache fits at all. Ties break on
    device count, then name, so the choice is deterministic — a controller that
    picked a different node on each reconcile would rewrite the Deployment
    forever.

    Returns `(None, reason)` when nothing qualifies, and the reason names the
    specific obstacle rather than "no suitable node".
    """
    if not nodes:
        return None, "the cluster reported no nodes"

    matching = [n for n in nodes if all(n.labels.get(k) == v for k, v in (selector or {}).items())]
    if not matching:
        return None, f"no node matches nodeSelector {selector}"

    usable = [n for n in matching if n.schedulable and n.kind != "cpu"]
    if not usable:
        # Distinguish "no accelerators" from "accelerators we could not size" —
        # they need completely different fixes.
        unsized = [n for n in matching if n.unknown]
        if unsized:
            return None, (
                f"{len(unsized)} accelerator node(s) could not be sized: {unsized[0].unknown}"
            )
        cpu_only = [n for n in matching if n.kind == "cpu" and n.schedulable]
        if cpu_only:
            return None, (
                "only CPU nodes are available; serving a language model on CPU is "
                "an order of magnitude slower and is not planned for here"
            )
        return None, "no schedulable accelerator nodes"

    best = max(
        usable,
        key=lambda n: ((n.device_bytes or 0) * max(n.devices, 1), n.devices, n.name),
    )
    return best, f"largest usable accelerator memory among {len(usable)} candidate node(s)"


def _requirements(spec: dict[str, Any]) -> Requirements:
    """Map CRD spec fields onto the planner's own input type."""
    return Requirements(
        workload=Workload(spec.get("workload", "interactive")),
        concurrency=int(spec.get("concurrency", 8)),
        context=int(spec.get("context", 32_768)),
        ttft_ms=spec.get("ttftMs"),
        itl_ms=spec.get("itlMs"),
        prefix_sharing=float(spec.get("prefixSharing", 0.0)),
        structured_output=bool(spec.get("structuredOutput", False)),
    )


def deployment_for(
    name: str, namespace: str, model: str, p: Plan, replicas: int = 1
) -> tuple[dict[str, Any], list[str]]:
    """Build the Deployment for a plan, plus any intents it could not express.

    The container args come from the engine adapter, so they are the *engine's
    own* flags — a config that runs with clickllm uninstalled (NFR-4). Nothing
    here wraps an engine.
    """
    argv, gaps = p.command(model)
    if not argv:
        return {}, list(gaps)

    port = PORTS.get(p.engine, DEFAULT_PORT)
    image = IMAGES.get(p.engine, "")
    labels = {"app.kubernetes.io/name": name, "app.kubernetes.io/managed-by": "clickllm"}

    # Device count comes from the tensor-parallel decision, which is where the
    # planner already reasoned about whether sharding is warranted. Defaulting
    # to 1 rather than to the node's full complement: claiming eight cards for a
    # model that fits on one is how a cluster runs out of GPUs it had.
    tp = p.get(Setting.TENSOR_PARALLEL)
    count = str(tp.value if tp else 1)
    resource = "google.com/tpu" if p.engine is Engine.VLLM_TPU else "nvidia.com/gpu"
    resources: dict[str, Any] = {"limits": {resource: count}}

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "labels": labels,
            "annotations": {
                # Provenance, same contract as the generated files: this says
                # what was chosen and that it runs without clickllm installed.
                "clickllm.dev/engine": p.engine.value,
                "clickllm.dev/engine-reason": p.engine_why,
                "clickllm.dev/standalone": "these args are the engine's own; "
                "this Deployment runs with clickllm uninstalled",
            },
        },
        "spec": {
            "replicas": replicas,
            "selector": {"matchLabels": labels},
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "containers": [
                        {
                            "name": p.engine.value.replace("-", ""),
                            "image": image,
                            # Skip argv[0..] up to the model: the image's own
                            # entrypoint supplies the binary.
                            "args": argv[1:] if p.engine is Engine.VLLM else argv[3:],
                            "ports": [{"containerPort": port}],
                            "resources": resources,
                        }
                    ]
                },
            },
        },
    }, list(gaps)


def reconcile(
    obj: dict[str, Any],
    nodes: list[Node],
    *,
    regressed: bool = False,
    regression_reason: str = "",
) -> Reconciled:
    """Turn one InferenceWorkload into the objects and status it implies.

    Args:
        obj: the custom resource, as the API server returns it.
        nodes: cluster nodes, from [`clickllm.k8s.nodes`].
        regressed: whether the quality gate has found a regression. Supplied by
            the caller rather than computed here, because judging a regression
            needs eval data this function deliberately does not touch.
        regression_reason: what regressed, for the status and the demotion.

    Returns:
        A [`Reconciled`]. Check `demote_to` — a non-None value is a rollback the
        controller should apply to `spec.phase` without asking.
    """
    meta = obj.get("metadata", {})
    spec = obj.get("spec", {}) or {}
    name = meta.get("name", "workload")
    namespace = meta.get("namespace", "default")
    model = spec.get("model", "")
    phase = spec.get("phase", "shadow")

    def condition(t: str, ok: bool, reason: str, message: str) -> dict[str, str]:
        return {
            "type": t,
            "status": "True" if ok else "False",
            "reason": reason,
            "message": message,
        }

    if not model:
        return Reconciled(
            status={"conditions": [condition("Ready", False, "NoModel", "spec.model is required")]}
        )

    node, why = select_node(nodes, spec.get("nodeSelector"))
    if node is None:
        return Reconciled(
            status={
                "node": "",
                "conditions": [condition("Ready", False, "NoSuitableNode", why)],
            }
        )

    hw = node.to_hardware()
    if hw is None:  # pragma: no cover - select_node already excludes these
        return Reconciled(
            status={"conditions": [condition("Ready", False, "UnsizableNode", node.unknown)]}
        )

    p = plan(hw, _requirements(spec))
    obj_dep, gaps = deployment_for(name, namespace, model, p)

    status: dict[str, Any] = {
        "engine": p.engine.value,
        "engineReason": p.engine_why,
        "node": f"{node.name} ({node.product or node.kind})",
        # Every knob with the reasoning that produced it. This is the field that
        # makes the CRD worth having: `kubectl describe` explains the config.
        "knobs": [{"setting": k.name.value, "value": str(k.value), "why": k.why} for k in p.knobs],
        "gaps": gaps,
        "warnings": list(p.warnings) + ([node.note] if node.note else []),
        "observedPhase": phase,
    }

    # A rollback. Lowering only — see the module docstring.
    demote = None
    if regressed and phase != "off":
        idx = PHASE_ORDER.index(phase) if phase in PHASE_ORDER else 0
        demote = PHASE_ORDER[max(idx - 1, 0)]
        status["conditions"] = [
            condition("Ready", True, "RolledBack", regression_reason or "quality regression"),
            condition(
                "Progressing",
                False,
                "RolledBack",
                f"phase lowered {phase} → {demote} automatically; raising it again "
                f"is a human decision",
            ),
        ]
        return Reconciled(objects=(obj_dep,) if obj_dep else (), status=status, demote_to=demote)

    ok = bool(obj_dep) and not p.warnings
    status["conditions"] = [
        condition(
            "Ready",
            ok,
            "Planned" if ok else ("NoEngineDialect" if not obj_dep else "CannotMeetRequirements"),
            why if ok else (p.warnings[0] if p.warnings else gaps[0] if gaps else "unknown"),
        )
    ]
    return Reconciled(objects=(obj_dep,) if obj_dep else (), status=status)


def demo() -> None:
    """Self-check. Run with `python -m clickllm.k8s.reconcile`."""
    from clickllm.k8s.nodes import GPU_MEMORY_LABEL, GPU_PRODUCT_LABEL, from_json

    cluster = from_json(
        {
            "items": [
                {
                    "metadata": {
                        "name": "h100-a",
                        "labels": {
                            GPU_MEMORY_LABEL: "81559",
                            GPU_PRODUCT_LABEL: "NVIDIA-H100-80GB-HBM3",
                            "pool": "gpu",
                        },
                    },
                    "status": {
                        "allocatable": {"nvidia.com/gpu": "8", "memory": "2000Gi", "cpu": "192"}
                    },
                },
                {
                    "metadata": {"name": "l4-a", "labels": {GPU_MEMORY_LABEL: "23034"}},
                    "status": {
                        "allocatable": {"nvidia.com/gpu": "1", "memory": "200Gi", "cpu": "16"}
                    },
                },
                {
                    "metadata": {"name": "cpu-a", "labels": {}},
                    "status": {"allocatable": {"memory": "64Gi", "cpu": "16"}},
                },
            ]
        }
    )

    def workload(**over):
        return {
            "metadata": {"name": "triage", "namespace": "ml"},
            "spec": {"model": "Qwen/Qwen3-32B", "workload": "interactive", **over},
        }

    # The happy path: an engine is derived, not requested.
    r = reconcile(workload(concurrency=8), cluster)
    assert r.ready and r.status["engine"] == "vllm"
    assert "h100-a" in r.status["node"]
    dep = r.objects[0]
    assert dep["kind"] == "Deployment" and dep["metadata"]["namespace"] == "ml"
    assert dep["spec"]["template"]["spec"]["containers"][0]["image"].startswith("vllm/")
    # Every knob explains itself — this is the point of the status subresource.
    assert r.status["knobs"] and all(len(k["why"]) > 30 for k in r.status["knobs"])

    # Prefix sharing changes the engine, and therefore the image and the flags.
    s = reconcile(workload(prefixSharing=0.9, structuredOutput=True), cluster)
    assert s.status["engine"] == "sglang"
    args = s.objects[0]["spec"]["template"]["spec"]["containers"][0]["args"]
    assert "--context-length" in args and "--max-model-len" not in args
    # ...and SGLang's unverifiable grammar flag is reported, never guessed.
    assert any("structured_output" in g for g in s.status["gaps"]), s.status["gaps"]

    # nodeSelector actually restricts, and a miss says what it missed.
    miss = reconcile(workload(nodeSelector={"pool": "nope"}), cluster)
    assert not miss.ready and "nodeSelector" in miss.status["conditions"][0]["message"]

    # A cluster with only CPU nodes is refused with the real reason.
    cpu_only = [n for n in cluster if n.kind == "cpu"]
    c = reconcile(workload(), cpu_only)
    assert not c.ready and "CPU" in c.status["conditions"][0]["message"]

    # A GPU node the cluster never labelled is a different refusal.
    unlabelled = from_json(
        {
            "items": [
                {
                    "metadata": {"name": "mystery", "labels": {}},
                    "status": {"allocatable": {"nvidia.com/gpu": "2", "memory": "100Gi"}},
                }
            ]
        }
    )
    u = reconcile(workload(), unlabelled)
    assert not u.ready
    assert "GPU Feature Discovery" in u.status["conditions"][0]["message"]

    # Rollback lowers the phase, and only lowers it.
    back = reconcile(
        workload(phase="canary", canaryPercent=25),
        cluster,
        regressed=True,
        regression_reason="extract regressed to 61% [54–68]",
    )
    assert back.demote_to == "shadow", back.demote_to
    assert any("human decision" in c["message"] for c in back.status["conditions"])
    # Already off: nothing lower to go to, so no demotion is proposed.
    assert reconcile(workload(phase="off"), cluster, regressed=True).demote_to is None
    # And a healthy workload is never demoted.
    assert reconcile(workload(phase="cut"), cluster).demote_to is None

    # Missing model is a condition, not a crash.
    bad = reconcile({"metadata": {"name": "x"}, "spec": {"workload": "batch"}}, cluster)
    assert not bad.ready and bad.status["conditions"][0]["reason"] == "NoModel"

    # Node selection is deterministic — otherwise the controller rewrites the
    # Deployment on every pass forever.
    assert select_node(cluster)[0].name == select_node(list(reversed(cluster)))[0].name

    print("k8s.reconcile: ok")


if __name__ == "__main__":
    demo()
