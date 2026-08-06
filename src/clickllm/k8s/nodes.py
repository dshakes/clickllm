"""Turn real cluster nodes into hardware the solver understands.

`clickllm fit` sizes against the machine it is running on. In a cluster that is
the wrong machine — you are on a laptop, and the model has to fit on a node you
have never logged into. This reads what the cluster already knows.

## Where the numbers come from

Kubernetes itself reports **counts**, not capacities: `nvidia.com/gpu: 8` says
how many, never how big. GPU memory arrives as a *label*, published by NVIDIA's
GPU Feature Discovery, and TPU topology likewise from GKE's own labels.

That indirection is why this file exists rather than being three lines inline.

## The bug that would silently ruin every estimate

`nvidia.com/gpu.memory` is documented in MiB — an A100-40GB reports `40537`,
matching `nvidia-smi` exactly. **Some versions emit raw bytes instead**
(`42505273344` for the same card), because with MIG strategy `none` the value
comes straight from `nvmlDeviceGetMemoryInfo`.

A tool that assumed MiB would read that node as having 42 *petabytes* of GPU
memory and cheerfully declare that every model fits. So the unit is inferred
from magnitude and the inference is stated: no accelerator has a billion MiB of
memory, so anything above that threshold is bytes.

Sources, checked 2026-07-27:
- <https://github.com/NVIDIA/k8s-device-plugin/blob/main/docs/gpu-feature-discovery/README.md>
- <https://github.com/NVIDIA/gpu-feature-discovery/issues/26> (the bytes bug)
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, field

from clickllm.hardware import Hardware

__all__ = [
    "BYTES_THRESHOLD_MIB",
    "GPU_MEMORY_LABEL",
    "GPU_PRODUCT_LABEL",
    "Node",
    "demo",
    "from_json",
    "read_cluster",
]

#: Label carrying per-GPU memory. Units are MiB — usually.
GPU_MEMORY_LABEL = "nvidia.com/gpu.memory"
#: Label carrying the GPU product name, e.g. `A100-SXM4-40GB`.
GPU_PRODUCT_LABEL = "nvidia.com/gpu.product"
#: GKE's TPU accelerator label, e.g. `tpu-v6e-slice`.
TPU_ACCELERATOR_LABEL = "cloud.google.com/gke-tpu-accelerator"
#: GKE's TPU topology label, e.g. `2x4` — the chip count is its product.
TPU_TOPOLOGY_LABEL = "cloud.google.com/gke-tpu-topology"

#: Above this, the memory label is bytes rather than MiB.
#:
#: A billion MiB is a petabyte of GPU memory. Nothing has that, and nothing will
#: before this code is rewritten — so the threshold is safe by many orders of
#: magnitude rather than by a close call.
BYTES_THRESHOLD_MIB = 1_000_000_000

#: Per-chip HBM for TPU generations, GiB. Verified against Google Cloud's
#: per-generation pages (2026-07-27). Absent generations yield no memory figure
#: rather than a guess.
TPU_HBM_GIB = {"v5e": 16, "v6e": 32, "v5p": 95, "v4": 32, "v3": 16}

#: Per-chip HBM bandwidth, GB/s, same sources.
TPU_BANDWIDTH = {"v5e": 859.0, "v6e": 1638.0, "v5p": 2765.0}


@dataclass(frozen=True, slots=True)
class Node:
    """One schedulable node, reduced to what the solver needs."""

    name: str
    kind: str
    #: Accelerators on this node.
    devices: int = 0
    #: Memory per accelerator, bytes. `None` when the cluster does not publish
    #: it — which is common, and must not be filled in with a plausible number.
    device_bytes: int | None = None
    #: Bandwidth per accelerator, GB/s, when the generation is known.
    bandwidth_gbps: float | None = None
    product: str = ""
    #: Host RAM, bytes.
    host_bytes: int = 0
    cpus: int = 0
    #: Why this node cannot be sized, when it cannot.
    unknown: str = ""
    #: A caveat about a node that *was* sized — chiefly that its memory label
    #: had to be reinterpreted. Distinct from `unknown`: this node is usable,
    #: but somebody should know why the number looks like it does.
    note: str = ""
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def schedulable(self) -> bool:
        """Whether the solver can size against this node at all."""
        return not self.unknown and (self.device_bytes is not None or self.kind == "cpu")

    def to_hardware(self) -> Hardware | None:
        """As a `Hardware` the solver accepts, or None when unsizable.

        `None` rather than a best guess: a node whose GPU memory the cluster
        never published is a node we cannot size, and inventing 80 GB because
        it is called an H100 is exactly the kind of confident wrongness this
        codebase refuses elsewhere.
        """
        if not self.schedulable:
            return None
        if self.kind == "cpu":
            total = self.host_bytes
            usable = int(total * 0.75)
        else:
            assert self.device_bytes is not None  # schedulable implies this
            total = self.device_bytes * max(self.devices, 1)
            # Engines reserve some of the card. 0.92 mirrors vLLM's own default
            # and is a starting point the planner then refines from real sizing.
            usable = int(total * 0.92)
        return Hardware(
            kind=self.kind,
            name=f"{self.name} ({self.product or self.kind})",
            total_bytes=total,
            usable_bytes=usable,
            bandwidth_gbps=self.bandwidth_gbps,
            cores=self.cpus,
            devices=max(self.devices, 1),
            note=f"from cluster node {self.name}",
        )


#: The suffixes Kubernetes emits on a quantity. Binary ones never collide with
#: decimal ones — they all end in `i` — so iteration order does not matter.
_UNITS = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
}


def _finite(s: str) -> float | None:
    """`float(s)`, but only when the answer is a real number.

    "nan", "inf" and "1e309" all parse without complaint and then explode at
    `int()` — `ValueError` for NaN, `OverflowError` for infinity — from a line
    outside whichever `try` caught the parse. Three functions here had that
    hole, so the check belongs with the parse rather than after it, once.
    """
    try:
        n = float(s)
    except ValueError:
        return None
    return n if math.isfinite(n) else None


def _scaled(v: str | int | None) -> float | None:
    """A quantity's numeric value with its suffix applied, or None if unreadable.

    One parser; the three callers below differ only in what they do about a
    value they cannot use.
    """
    if v is None:
        return None
    s = str(v).strip()
    for suffix, mult in _UNITS.items():
        if s.endswith(suffix):
            n = _finite(s[: -len(suffix)])
            return None if n is None else n * mult
    return _finite(s)


def _quantity(v: str | int | None) -> int:
    """Parse a Kubernetes resource quantity into bytes.

    Handles the binary suffixes Kubernetes actually emits for memory
    (`Ki`/`Mi`/`Gi`) and a bare integer. Decimal suffixes are rare for memory
    but cheap to accept. Unreadable degrades to 0: a node with no usable
    memory figure sizes nothing, which is the safe direction.
    """
    n = _scaled(v)
    return 0 if n is None else int(n)


def _count(v: str | int | None) -> int:
    """A whole-unit count from a Kubernetes quantity, milli suffix included.

    `allocatable.cpu` is routinely `"15910m"` — the standard notation for
    fractional cores, and what a node reports once kube-reserved is subtracted
    from capacity, since that arithmetic rarely lands on a whole core.
    `int(float("15910m"))` raises, and nothing between here and
    `reconcile_once` catches it, so one such node took down the reconcile pass
    for every workload in the cluster — the opposite of that function's stated
    per-workload blast radius.

    Truncates rather than rounds: 15.91 cores are 15 whole ones, and
    over-reporting capacity is the direction that makes something appear to
    fit. Degrades to 0 like `_quantity` rather than raising — an unreadable
    count must read as "none", never as a crash halfway through a cluster.
    """
    if v is None:
        return 0
    s = str(v).strip()
    if s.endswith("m"):
        milli = _finite(s[:-1])
        return 0 if milli is None else int(milli / 1000)
    n = _scaled(s)
    return 0 if n is None else int(n)


def _devices(v: str | int | None) -> tuple[int, str]:
    """A whole accelerator count, or a refusal saying why not.

    Not `_count`. Extended resources like `nvidia.com/gpu` are integer-only in
    Kubernetes — `3000m` is a legal spelling of 3, `1500m` is rejected by the
    API server outright — so there is no fractional form here to truncate the
    way `cpu` has. Truncating one anyway invents a count; reading it as 0 is
    worse, because the node then falls through to the CPU branch and the
    planner sizes a model against host RAM on a box with eight H100s in it.
    Either way an unreadable count must make the node unsizable, which is what
    `unknown` is for.

    Suffixed forms go through `_scaled`, which already knows the Ki/Mi/Gi/K/M
    table — a count is a quantity like any other. Reading them via `_quantity`
    instead conflated "zero" with "unreadable", because that returns 0 for
    both: `"0Ki"` is a legal way to say no GPUs, and it refused the node.
    """
    if v is None:
        return 0, ""
    s = str(v).strip()
    if not s:
        return 0, ""
    if s.endswith("m"):
        milli = _finite(s[:-1])
        n = None if milli is None else milli / 1000
    else:
        n = _scaled(s)
    if n is None:
        return 0, f"unreadable accelerator count {s!r}"
    if n < 0:
        return 0, f"negative accelerator count {s!r}"
    if n != int(n):
        return 0, f"fractional accelerator count {s!r} — these are whole units"
    return int(n), ""


def _gpu_bytes(raw: str) -> tuple[int | None, str]:
    """Per-GPU memory in bytes, and a note when the units had to be inferred.

    See the module docstring: the label is MiB except when it is bytes.
    """
    if not raw:
        return None, f"the cluster does not publish {GPU_MEMORY_LABEL} for this node"
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        return None, f"{GPU_MEMORY_LABEL}={raw!r} is not a number"
    if value <= 0:
        return None, f"{GPU_MEMORY_LABEL}={raw!r} is not a usable capacity"
    if value >= BYTES_THRESHOLD_MIB:
        # The known GFD bug. Treating this as MiB would read a 40 GB card as
        # 42 petabytes and declare that everything fits.
        return value, (
            f"{GPU_MEMORY_LABEL}={value} is far too large for MiB, so it was read "
            f"as bytes — a known GPU Feature Discovery bug"
        )
    return value * 1024**2, ""


def _tpu(labels: dict[str, str], devices: int) -> tuple[int | None, float | None, str, str, int]:
    """TPU memory, bandwidth, product, note and chip count, from GKE's labels.

    One tuple shape on every path. The first draft returned a 4-tuple on the
    failure branch and a nested 2-tuple on the success branch, which typechecks
    as `Any` and blows up only when an unrecognised generation appears — caught
    by a test, not by the demo, because the demo only used a known generation.
    """
    accel = labels.get(TPU_ACCELERATOR_LABEL, "")
    gen = next((g for g in TPU_HBM_GIB if g in accel), "")
    if not gen:
        return None, None, accel, f"unrecognised TPU accelerator {accel!r}", devices
    # Topology like `2x4` is a chip layout; its product is the chip count, which
    # is more reliable than the resource count on multi-host slices.
    topo = labels.get(TPU_TOPOLOGY_LABEL, "")
    chips = devices
    if topo:
        try:
            n = 1
            for part in topo.split("x"):
                n *= int(part)
            chips = n
        except ValueError:
            pass
    return (
        TPU_HBM_GIB[gen] * 1024**3,
        TPU_BANDWIDTH.get(gen),
        accel or f"tpu-{gen}",
        "" if gen in TPU_BANDWIDTH else f"no published bandwidth for TPU {gen}",
        chips,
    )


def from_json(payload: dict | str) -> list[Node]:
    """Parse `kubectl get nodes -o json` output.

    Split from the `kubectl` call so the parsing is testable against real API
    output without a cluster — which is the only way this gets exercised in CI.
    """
    data = json.loads(payload) if isinstance(payload, str) else payload
    out: list[Node] = []

    for item in data.get("items", []):
        meta = item.get("metadata", {})
        name = meta.get("name", "<unnamed>")
        labels = meta.get("labels", {}) or {}
        alloc = item.get("status", {}).get("allocatable", {}) or {}
        host_bytes = _quantity(alloc.get("memory"))
        cpus = _count(alloc.get("cpu"))

        gpus, gpu_refusal = _devices(alloc.get("nvidia.com/gpu"))
        tpus, tpu_refusal = _devices(alloc.get("google.com/tpu"))

        # `or refusal` on the branch test, not just the count: a node whose
        # accelerator count we could not read is still an accelerator node, and
        # falling through to the CPU branch would hide it rather than refuse it.
        if gpus or gpu_refusal:
            device_bytes, note = _gpu_bytes(labels.get(GPU_MEMORY_LABEL, ""))
            unsized = device_bytes is None
            out.append(
                Node(
                    name=name,
                    kind="nvidia",
                    devices=gpus,
                    device_bytes=device_bytes,
                    product=labels.get(GPU_PRODUCT_LABEL, ""),
                    host_bytes=host_bytes,
                    cpus=cpus,
                    # A parse failure is a refusal; a reinterpreted unit is a
                    # caveat on a node we *can* size. Different fields, because
                    # collapsing them would either hide the caveat or reject a
                    # perfectly usable node.
                    unknown=(
                        gpu_refusal
                        or (
                            f"{note} — install GPU Feature Discovery or size this node manually"
                            if unsized
                            else ""
                        )
                    ),
                    note="" if unsized else note,
                    labels=labels,
                )
            )
        elif tpus or tpu_refusal:
            device_bytes, bw, product, note, chips = _tpu(labels, tpus)
            out.append(
                Node(
                    name=name,
                    kind="tpu",
                    devices=chips,
                    device_bytes=device_bytes,
                    bandwidth_gbps=bw,
                    product=product,
                    host_bytes=host_bytes,
                    cpus=cpus,
                    unknown=tpu_refusal or (note if device_bytes is None else ""),
                    # Same split as the GPU branch above, and it was missing:
                    # `_tpu` computes "no published bandwidth for TPU v4" for
                    # the generations absent from TPU_BANDWIDTH, and that note
                    # reached nothing. A v4 node sized fine and showed a blank
                    # bandwidth with no reason, in the field reconcile.py puts
                    # on the CRD status.
                    note="" if device_bytes is None else note,
                    labels=labels,
                )
            )
        else:
            out.append(
                Node(
                    name=name,
                    kind="cpu",
                    host_bytes=host_bytes,
                    cpus=cpus,
                    labels=labels,
                )
            )
    return out


def read_cluster(context: str | None = None, timeout: int = 20) -> list[Node]:
    """Read nodes from the current cluster via `kubectl`.

    `kubectl` rather than a Kubernetes client library, deliberately: it is
    already installed wherever this is useful, it already holds the user's
    kubeconfig, contexts and auth — including cloud exec plugins that a library
    would need re-implementing — and it keeps `clickllm fit` free of runtime
    dependencies.

    Raises:
        RuntimeError: with the command's own stderr, so a permissions or context
            problem is diagnosable rather than a generic failure.
    """
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if context:
        cmd += ["--context", context]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise RuntimeError("kubectl is not installed or not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"kubectl did not answer within {timeout}s") from e
    if r.returncode != 0:
        raise RuntimeError(f"kubectl failed: {r.stderr.strip() or r.returncode}")
    return from_json(r.stdout)


def demo() -> None:
    """Self-check. Run with `python -m clickllm.k8s.nodes`."""
    payload = {
        "items": [
            {
                "metadata": {
                    "name": "gpu-a",
                    "labels": {
                        GPU_MEMORY_LABEL: "81559",
                        GPU_PRODUCT_LABEL: "NVIDIA-H100-80GB-HBM3",
                    },
                },
                "status": {
                    "allocatable": {"nvidia.com/gpu": "8", "memory": "2000Gi", "cpu": "192"}
                },
            },
            {
                # The bytes bug: same card, 1024² times bigger a number.
                "metadata": {"name": "gpu-buggy", "labels": {GPU_MEMORY_LABEL: "42505273344"}},
                "status": {"allocatable": {"nvidia.com/gpu": "1", "memory": "200Gi", "cpu": "16"}},
            },
            {
                # GPU node with no GFD installed.
                "metadata": {"name": "gpu-unlabelled", "labels": {}},
                "status": {"allocatable": {"nvidia.com/gpu": "4", "memory": "500Gi", "cpu": "64"}},
            },
            {
                "metadata": {
                    "name": "tpu-a",
                    "labels": {
                        TPU_ACCELERATOR_LABEL: "tpu-v6e-slice",
                        TPU_TOPOLOGY_LABEL: "2x4",
                    },
                },
                "status": {
                    "allocatable": {"google.com/tpu": "8", "memory": "1500Gi", "cpu": "180"}
                },
            },
            {
                "metadata": {"name": "plain", "labels": {}},
                "status": {"allocatable": {"memory": "64Gi", "cpu": "16"}},
            },
        ]
    }
    nodes = {n.name: n for n in from_json(payload)}
    assert len(nodes) == 5

    # A normal GPU node sizes correctly.
    a = nodes["gpu-a"]
    assert a.kind == "nvidia" and a.devices == 8
    assert a.device_bytes == 81559 * 1024**2
    hw = a.to_hardware()
    assert hw is not None and hw.total_bytes == 8 * 81559 * 1024**2

    # The bytes bug is caught, not multiplied by another 1024².
    b = nodes["gpu-buggy"]
    assert b.device_bytes == 42_505_273_344, b.device_bytes
    assert 39 < b.device_bytes / 1024**3 < 41, "should read as a ~40 GB card"
    # The reinterpretation must be reported, not silently applied — a number
    # that quietly changed meaning is worse than one that failed.
    assert "read as bytes" in b.note, b.note
    assert b.schedulable, "a reinterpreted unit is a caveat, not a refusal"
    assert not nodes["gpu-a"].note, "an ordinary node carries no caveat"

    # A GPU node the cluster never labelled is unsizable, and says why.
    u = nodes["gpu-unlabelled"]
    assert not u.schedulable and u.to_hardware() is None
    assert "GPU Feature Discovery" in u.unknown

    # TPU: chips come from the topology, memory from the generation.
    t = nodes["tpu-a"]
    assert t.kind == "tpu" and t.devices == 8, t.devices
    assert t.device_bytes == 32 * 1024**3
    assert t.bandwidth_gbps == 1638.0
    assert t.to_hardware().total_bytes == 8 * 32 * 1024**3

    # A node with no accelerator is still usable, at 75% of host RAM.
    p = nodes["plain"]
    assert p.kind == "cpu" and p.schedulable
    assert p.to_hardware().usable_bytes == int(64 * 1024**3 * 0.75)

    # Quantities.
    assert _quantity("1Ki") == 1024
    assert _quantity("2Gi") == 2 * 1024**3
    assert _quantity("1000") == 1000
    assert _quantity(None) == 0 and _quantity("garbage") == 0

    print("k8s.nodes: ok")


if __name__ == "__main__":
    demo()
