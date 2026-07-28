"""`clickllm box` — the artifact ADR-0005 committed to, and nothing more.

A box is **a tuned starting point plus the evidence behind it**, never a frozen
command line. It is an OCI artifact you push to any registry:

```
manifest.yaml     model · quant · tuned knobs · provenance · eval scores
weights.lock      source + revision (weights fetched, not vendored)
bench.json        what was measured — and, here, what was not
README.md         what was chosen and why
targets/
  linux-cuda/     Dockerfile + compose + k8s manifests
  linux-rocm/     "
  darwin-metal/   native launch spec (mlx)
```

## Why the tree is not identical to the ADR's sketch

Two entries in that sketch are *absent by design*, and the manifest says so
rather than leaving a reader to notice:

- **`targets/cpu/`** — the fallback is llama.cpp, and [`engines`] has no verified
  flag dialect for it. Emitting another engine's flags to fill the directory
  produces a config that cannot start, which is the exact failure that module
  exists to prevent.
- **`bench.json` measurements** — nothing here runs a benchmark, so the file
  carries roofline *estimates*, each labelled as an estimate with the arithmetic
  that produced it. A measured-looking number nobody measured is the one thing
  this repo cannot ship (invariant 6).

## What this module refuses to do

**Push.** `oras push` and `docker push` are printed, exactly, for you to run.
Nothing here authenticates to a registry, reads a token, or touches your
environment — the same boundary [`host`][clickllm.host] keeps. A box is a file
tree until a human moves it.

**Vendor weights.** The lock records the repo *and the revision SHA the index
returned*, so a pull is reproducible. When the revision cannot be read, the lock
says `pinned: false` and why — pinning `main` and calling it a lock would be a
checksum that checks nothing.

**Bake one tuning for every host.** Each target is solved against its own host
class, and every file states which one. ADR-0005's re-solve-on-arrival is the
other half; a box that pretended an H100 config was portable to a 24 GB card is
the "wrong auto-tune" failure ADR-0004 warns about.
"""

from __future__ import annotations

import json
import shlex
import textwrap
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from . import catalog, fit
from .catalog import ModelSpec
from .engines import Setting, adapter_for
from .fit import Fit
from .hardware_catalog import get as profile_by_id
from .k8s.reconcile import DEFAULT_PORT, IMAGES, PORTS, VLLM_FAMILY, deployment_for
from .launch import Refusal, resolve_weights
from .plan import Knob, Plan, Requirements, Workload
from .plan import plan as configure
from .prove.receipt import Receipt

__all__ = [
    "TARGETS",
    "Box",
    "Skipped",
    "Target",
    "TargetPlan",
    "build",
    "demo",
    "hf_revision",
]

GB = 1024**3

#: Artifact type for the OCI manifest. A registry stores this verbatim, so a
#: puller can tell a box from an image before fetching a layer.
ARTIFACT_TYPE = "application/vnd.clickllm.box.v1+json"

FORMAT = "clickllm.box/v1"
LOCK_FORMAT = "clickllm.weights-lock/v1"
BENCH_FORMAT = "clickllm.bench/v1"


@dataclass(frozen=True, slots=True)
class Target:
    """One host class the box is tuned for, and how execution binds there."""

    id: str
    #: Hardware profile this target's knobs are solved against.
    profile_id: str
    #: `container` on everything a container can reach the GPU from; `native`
    #: on macOS, where no such mechanism exists (ADR-0005).
    binding: str
    why: str


#: The three targets ADR-0005 names. One profile each, deliberately: a box is a
#: starting point per *host class*, and the run-time re-solve is what handles the
#: machine it actually lands on.
TARGETS: tuple[Target, ...] = (
    Target(
        "linux-cuda",
        "h100",
        "container",
        "NVIDIA Container Toolkit passes the GPU into a container, so the "
        "artifact is the runtime image here.",
    ),
    Target(
        "linux-rocm",
        "mi300x",
        "container",
        "ROCm exposes /dev/kfd and /dev/dri to a container, so the artifact is "
        "the runtime image here too.",
    ),
    Target(
        "darwin-metal",
        "m4-max-128",
        "native",
        "Metal is a macOS-only API with no passthrough into a Linux VM, so the "
        "artifact is unpacked and a native engine is supervised against it.",
    ),
)

#: Where the target platform ships its own build of an engine. Keyed on
#: (target, engine) rather than folded into `k8s.reconcile.IMAGES`, which is
#: about clusters and has no notion of a ROCm host.
#: `rocm/vllm` is AMD's published vLLM image — hub.docker.com/r/rocm/vllm,
#: checked 2026-07-28.
IMAGE_OVERRIDE = {("linux-rocm", "vllm"): "rocm/vllm:latest"}

#: The accelerator resource a scheduler is asked for, per target. `amd.com/gpu`
#: is the AMD GPU device plugin's name; asking for `nvidia.com/gpu` on a ROCm
#: cluster schedules nowhere, forever, with no error.
GPU_RESOURCE = {"linux-rocm": "amd.com/gpu"}

#: Bind address for a containerised engine. Load-bearing: SGLang's own default
#: is 127.0.0.1, which inside a container is reachable by nothing.
CONTAINER_HOST = "0.0.0.0"  # noqa: S104 — inside a container namespace, published explicitly

#: Loopback for the native target. An inference endpoint has no authentication,
#: and this one runs on someone's laptop.
NATIVE_HOST = "127.0.0.1"

#: Parts of ADR-0005's sketch this build does not produce, and why. Carried into
#: the manifest so the absence is a statement rather than an oversight.
ABSENT = (
    {
        "path": "targets/cpu/",
        "reason": (
            "the CPU fallback is llama.cpp and clickllm has no verified flag "
            "dialect for it. Emitting another engine's flags would produce a "
            "config that cannot start, so nothing is emitted"
        ),
    },
    {
        "path": "bench.json measurements",
        "reason": (
            "nothing in this build ran a benchmark. bench.json carries roofline "
            "estimates, each labelled as an estimate; a measured-looking number "
            "nobody measured is not shipped"
        ),
    },
)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Skipped:
    """A target that produced no files, and the reason it produced none."""

    target: Target
    reason: str
    tried: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TargetPlan:
    """The tuned settings for one host class, and the files that express them."""

    target: Target
    host_class: str
    engine: str
    engine_why: str
    quant: str
    repo: str
    argv: tuple[str, ...]
    knobs: tuple[Knob, ...]
    gaps: tuple[str, ...]
    warnings: tuple[str, ...]
    fit: Fit
    #: `(path relative to the box root, content)`.
    files: tuple[tuple[str, str], ...]

    @property
    def speed(self) -> str:
        f = self.fit
        if f.tokens_per_sec is None:
            return "? — no bandwidth figure for this host class, so no estimate is offered"
        return f"~{f.tokens_per_sec:.0f} tok/s single-stream — roofline estimate, not measured"


@dataclass(frozen=True, slots=True)
class Weight:
    """One set of weights the box points at. Fetched at run time, never vendored."""

    repo: str
    revision: str | None
    targets: tuple[str, ...]
    #: Why no revision could be recorded. Empty when one was.
    unpinned_reason: str = ""

    @property
    def pinned(self) -> bool:
        return self.revision is not None


@dataclass(frozen=True, slots=True)
class Box:
    """A box on disk, plus the commands that move it. Never pushed for you."""

    model: ModelSpec
    generated: str
    targets: tuple[TargetPlan, ...]
    skipped: tuple[Skipped, ...]
    weights: tuple[Weight, ...]
    #: `(path relative to the box root, content)`, in the order worth reading.
    files: tuple[tuple[str, str], ...]
    #: Registry reference the push commands are written against.
    ref: str
    proven: bool

    def write(self, directory: str | Path) -> list[Path]:
        """Write the whole tree under `directory`, creating it if needed."""
        root = Path(directory)
        written = []
        for name, content in self.files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            written.append(path)
        return written

    def push_commands(self, directory: str | Path = ".") -> tuple[str, ...]:
        """The exact commands to publish this box. You run them, not clickllm.

        `oras` moves the artifact itself. `docker build`/`docker push` is the
        other half for the container targets, whose Dockerfile is in the tree.
        """
        root = str(directory)
        files = " ".join(
            f"{n}:{m}"
            for n, m in (
                ("manifest.yaml", "application/yaml"),
                ("weights.lock", "application/yaml"),
                ("bench.json", "application/json"),
                ("README.md", "text/markdown"),
            )
        )
        out = [
            f"cd {shlex.quote(root)}",
            f"oras push {self.ref} --artifact-type {ARTIFACT_TYPE} {files} targets",
            f"cosign sign {self.ref}      # optional, and the reason OCI was chosen",
        ]
        for t in self.targets:
            if t.target.binding == "container":
                image = f"{self.ref}-{t.target.id}"
                out.append(f"docker build -t {image} targets/{t.target.id} && docker push {image}")
        return tuple(out)

    def render(self) -> str:
        """Terminal summary. What is unproven comes before what is tuned."""
        proof = (
            "receipt attached"
            if self.proven
            else "none — no receipt was supplied, so this box carries no quality claim"
        )
        out = [
            f"  BOX       {self.model.id} · {self.model.name}",
            f"  PROOF     {proof}",
            "",
            "  TARGETS",
        ]
        for t in self.targets:
            out += [
                f"    {t.target.id:<14}{t.engine} @ {t.quant} on {t.host_class}",
                f"    {'':<14}{t.speed}",
            ]
        for s in self.skipped:
            out += [f"    {s.target.id:<14}not built — {s.reason}"]
        out += ["", "  WEIGHTS"]
        for w in self.weights:
            pin = w.revision[:12] if w.revision else f"unpinned — {w.unpinned_reason}"
            out.append(f"    {w.repo}  @ {pin}")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Weight revisions
# --------------------------------------------------------------------------- #


def hf_revision(repo: str, timeout: float = 10.0) -> str | None:
    """The commit SHA the model index reports for `repo`, or None.

    None is a real answer and is recorded as one: a lock that pinned `main`
    because it could not read a revision would be reproducible in name only.
    """
    import urllib.error
    import urllib.request

    from .catalog_update import USER_AGENT

    url = f"https://huggingface.co/api/models/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https only
            sha = json.loads(resp.read()).get("sha")
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return sha if isinstance(sha, str) and sha else None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def _field(label: str, text: str, width: int = 68) -> list[str]:
    """One provenance field, wrapped under a hanging indent."""
    lines = textwrap.wrap(text, width) or [""]
    pad = " " * 11
    return [f"{label:<11}{lines[0]}"] + [f"{pad}{rest}" for rest in lines[1:]]


def _commented(lines: list[str], marker: str = "#") -> str:
    return "\n".join(f"{marker} {line}".rstrip() for line in lines)


def _scalar(value: object) -> str:
    """One YAML scalar. Strings are double-quoted, which JSON already does.

    A double-quoted JSON string *is* a YAML 1.2 double-quoted scalar, so this
    needs no escaping table of its own — the flags carry JSON and colons, and
    a bare scalar would reparse as a mapping.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, (dict, list)):  # only reached when empty
        return "{}" if isinstance(value, dict) else "[]"
    # ensure_ascii=False: YAML is UTF-8, and a manifest full of — for every
    # em dash is a document nobody reads.
    return json.dumps(str(value), ensure_ascii=False)


def _yaml(value: Any, indent: int = 0) -> list[str]:
    """A strict YAML subset: mappings, sequences and quoted scalars.

    Written rather than imported because `clickllm fit` and everything it
    reaches has zero runtime dependencies, and a manifest format is not worth
    breaking that for. Nothing here emits an anchor, a tag or a folded block —
    if a value cannot be expressed as one of those three things, it is not put
    in the manifest.
    """
    pad = "  " * indent
    if isinstance(value, dict) and value:
        out: list[str] = []
        for k, v in value.items():
            if isinstance(v, (dict, list)) and v:
                out.append(f"{pad}{k}:")
                out += _yaml(v, indent + 1)
            else:
                out.append(f"{pad}{k}: {_scalar(v)}")
        return out
    if isinstance(value, list) and value:
        out = []
        for item in value:
            if isinstance(item, (dict, list)) and item:
                block = _yaml(item, indent + 1)
                out.append(f"{pad}- {block[0].lstrip()}")
                out += block[1:]
            else:
                out.append(f"{pad}- {_scalar(item)}")
        return out
    return [f"{pad}{_scalar(value)}"]


def _yaml_doc(header: list[str], body: dict[str, Any]) -> str:
    return "\n".join([_commented(header), ""] + _yaml(body) + [""])


def _header(model: ModelSpec, generated: str, extra: list[str]) -> list[str]:
    """Provenance. What was chosen, why, when, and that it stands alone."""
    return [
        f"clickllm box — generated {generated}",
        "",
        *_field("BOX", f"{model.id} · {model.name} · {model.params_b:g}B params"),
        *extra,
        *_field(
            "RETUNE",
            "every setting here was tuned for the host class it names. On a "
            "different one, re-solve before trusting it — a config tuned for "
            "80 GB is wrong on 24 GB and catastrophic on a Mac.",
        ),
        *_field(
            "STANDALONE",
            "this file has no clickllm dependency and runs with it uninstalled. "
            "Everything below is the target platform's own config.",
        ),
        *_field(
            "SECRETS",
            "clickllm holds none and asks for none. Anything secret below is "
            "passed through from your own environment, by you.",
        ),
    ]


def _target_header(model: ModelSpec, generated: str, t: TargetPlan, proof: str) -> list[str]:
    return _header(
        model,
        generated,
        [
            *_field("TARGET", f"{t.target.id} ({t.target.binding}) — {t.target.why}"),
            *_field("HOST CLASS", f"tuned on {t.host_class}"),
            *_field("MODEL", f"{model.name} @ {t.quant}  ({t.repo})"),
            *_field(
                "",
                f"{t.fit.total_bytes / GB:.1f} GB required of "
                f"{t.fit.usable_bytes / GB:.1f} GB usable, at "
                f"{t.fit.context:,} ctx x{t.fit.concurrency}",
            ),
            *_field("ENGINE", f"{t.engine} — {t.engine_why}"),
            *_field("SPEED", t.speed),
            *_field("PROOF", proof),
        ],
    )


# --------------------------------------------------------------------------- #
# Target files
# --------------------------------------------------------------------------- #


def _image_for(target: Target, engine: str) -> str:
    image = IMAGE_OVERRIDE.get((target.id, engine)) or IMAGES.get(engine)  # type: ignore[arg-type]
    if not image:
        raise ValueError(f"no container image recorded for engine {engine!r} on {target.id}")
    return image


def _bind(binding: str, argv: list[str]) -> list[str]:
    """The engine's argv with its bind address made explicit.

    Not left to the engine's default in either case. In a container SGLang's own
    default is 127.0.0.1, which inside the namespace is reachable by nothing; on
    the native target `mlx_lm.server` defaults to port 8080, so an endpoint URL
    written anywhere else in the box would be wrong. Loopback there, because an
    inference endpoint has no authentication and that one runs on a laptop.
    """
    if binding == "container":
        return argv + ["--host", CONTAINER_HOST]
    return argv + ["--host", NATIVE_HOST, "--port", str(DEFAULT_PORT)]


def _container_args(engine: str, argv: tuple[str, ...]) -> list[str]:
    """The container's args: the argv without the binary its entrypoint supplies.

    vLLM-family argv is `vllm serve MODEL …`, so drop one; SGLang's is
    `python3 -m sglang.launch_server …`, so drop three. Same rule as the
    Kubernetes emitter, from the same table — getting it wrong drops the model
    name and the container crashes on boot with no argument at all.
    """
    return list(argv[1:] if engine in VLLM_FAMILY else argv[3:])


def _dockerfile(head: list[str], image: str, args: list[str], port: int) -> str:
    return "\n".join(
        [
            _commented(head),
            "",
            f"FROM {image}",
            "",
            f"EXPOSE {port}",
            "# The base image's entrypoint supplies the binary; these are its own flags.",
            "CMD [" + ", ".join(json.dumps(a) for a in args) + "]",
            "",
        ]
    )


def _compose(
    target: Target, head: list[str], image: str, args: list[str], port: int, devices: int
) -> str:
    rocm = target.id == "linux-rocm"
    amd_devices = [
        "    # ROCm exposes the GPU as two device nodes rather than through a",
        "    # runtime shim, and the video group is what grants access to them.",
        "    devices:",
        "      - /dev/kfd",
        "      - /dev/dri",
        "    group_add:",
        "      - video",
        "    security_opt:",
        "      - seccomp:unconfined",
    ]
    nvidia = [
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        # The tensor-parallel decision, not the host's full complement: claiming
        # eight cards for a model that fits on one is how a machine runs out of
        # GPUs it had.
        f"              count: {devices}",
        "              capabilities: [gpu]",
    ]
    return "\n".join(
        [
            _commented(head),
            "",
            "services:",
            "  inference:",
            f"    image: {image}",
            # Every arg is a quoted YAML string. A bare `32768` parses as an
            # integer and compose refuses a non-string in `command`; one of these
            # flags also carries JSON, which unquoted would parse as a mapping.
            "    command:",
            *[f"      - {json.dumps(a)}" for a in args],
            "    ports:",
            f'      - "{port}:{port}"',
            "    # Gated repos need a token. It is taken from YOUR shell, never",
            "    # from a file clickllm wrote — this line names it and nothing more.",
            "    environment:",
            "      - HUGGING_FACE_HUB_TOKEN",
            "    # The weights are fetched on first start (see weights.lock).",
            "    # Mounting the cache means a restart does not fetch tens of GB again.",
            "    volumes:",
            "      - ./hf-cache:/root/.cache/huggingface",
            "    # vLLM and SGLang use shared memory for their worker processes;",
            "    # the default 64 MB is not enough and fails as a cryptic hang.",
            "    ipc: host",
            *(amd_devices if rocm else nvidia),
            "",
        ]
    )


def _kubernetes(head: list[str], t: TargetPlan, p: Plan, image: str, port: int) -> str:
    """Deployment + Service, from the controller's own reconcile rules.

    `deployment_for` is the operator's emitter: reusing it is what stops a box
    and a cluster that clickllm reconciles from disagreeing about the image, the
    args or the device count. What is patched afterwards is only what a *target*
    knows and a cluster does not — the ROCm image and its device plugin name.
    """
    dep, _gaps = deployment_for(t.repo.split("/")[-1].lower(), "default", t.repo, p)
    if not dep:  # pragma: no cover — a target with no dialect never gets here
        raise ValueError(f"no Deployment could be built for {t.engine} on {t.target.id}")

    container = dep["spec"]["template"]["spec"]["containers"][0]
    container["image"] = image
    container["ports"] = [{"containerPort": port}]
    # SGLang binds 127.0.0.1 by default, which in a pod is reachable by nothing.
    container["args"] = list(container["args"]) + ["--host", CONTAINER_HOST]
    if (resource := GPU_RESOURCE.get(t.target.id)) is not None:
        count = next(iter(container["resources"]["limits"].values()))
        container["resources"] = {"limits": {resource: count}}

    labels = dep["spec"]["selector"]["matchLabels"]
    service = {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": dep["metadata"]["name"],
            "namespace": dep["metadata"]["namespace"],
            "labels": labels,
        },
        "spec": {
            "selector": labels,
            "ports": [{"name": "http", "port": port, "targetPort": port}],
        },
    }
    return "\n".join([_commented(head), ""] + _yaml(dep) + ["---"] + _yaml(service) + [""])


def _native(head: list[str], t: TargetPlan, argv: list[str], install: str) -> list[tuple[str, str]]:
    """The macOS launch spec: what to run, and a script that runs it.

    No container, because none can reach Metal (ADR-0005). The spec is the
    portable half — a supervisor reads `launch.json`; `serve.sh` is the same
    thing you can run by hand.
    """
    spec = {
        "format": FORMAT,
        "target": t.target.id,
        "binding": "native",
        "why": t.target.why,
        "engine": t.engine,
        "engine_why": t.engine_why,
        "install": install,
        "repo": t.repo,
        "quant": t.quant,
        "argv": argv,
        "endpoint": f"http://{NATIVE_HOST}:{DEFAULT_PORT}/v1",
        "weights": "fetched by the engine on first start; see weights.lock for the pin",
        "not_expressed": list(t.gaps),
        "standalone": (
            f"the engine's own flags. This runs with clickllm uninstalled; "
            f"`{install}` is the only requirement."
        ),
    }
    script = "\n".join(
        [
            "#!/usr/bin/env sh",
            _commented(head),
            "",
            "set -eu",
            "",
            f"# {install}",
            "exec " + " ".join(shlex.quote(a) for a in argv),
            "",
        ]
    )
    return [
        ("launch.json", json.dumps(spec, indent=2) + "\n"),
        ("serve.sh", script),
    ]


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _install_hint(engine: str) -> str:
    from .launch import INSTALL_HINT

    return INSTALL_HINT.get(engine, f"install {engine}")


def _solve(
    model: ModelSpec, target: Target, quant: str | None, ctx: int, conc: int
) -> tuple[str, Fit, Plan] | Skipped:
    """(host class, Fit, Plan) for one target, or a Skipped saying why not."""
    profile = profile_by_id(target.profile_id)
    hw = profile.to_hardware()

    if quant is None:
        sized = fit.best_quant(model, hw, ctx, conc)
        if sized is None:
            smallest = min(model.quants, key=lambda q: catalog.QUANT_BITS[q])
            f = fit.solve(model, smallest, hw, ctx, conc)
            return Skipped(
                target,
                f"{model.name} does not fit {profile.name}: even at {smallest} it "
                f"needs {f.total_bytes / GB:.1f} GB against "
                f"{f.usable_bytes / GB:.1f} GB usable, at {ctx:,} ctx x{conc}",
            )
    else:
        sized = fit.solve(model, quant, hw, ctx, conc)
        if not sized.feasible:
            return Skipped(
                target,
                f"{model.name} at {quant} needs {sized.total_bytes / GB:.1f} GB "
                f"against {sized.usable_bytes / GB:.1f} GB usable on {profile.name}",
            )

    p = configure(
        hw, Requirements(Workload.INTERACTIVE, concurrency=conc, context=ctx), model, sized.quant
    )
    return profile.name, sized, p


def _evaluation(receipt: Receipt | None) -> dict[str, Any]:
    """Eval scores, or the explicit statement that none were measured.

    Invariant 6: a cell shows `?` rather than a fabricated score. Here the whole
    section says no proof was run, which is the same rule at document scale.
    """
    if receipt is None:
        return {
            "proven": False,
            "scores": None,
            "note": (
                "no proof was run for this box, so it carries no quality claim. "
                "`clickllm prove` measures a candidate against captured traffic "
                "and issues a receipt; any score written here without one would "
                "be fabricated"
            ),
        }

    def claims(items) -> list[dict[str, Any]]:
        return [
            {
                "cluster": c.cluster,
                "name": c.name,
                "share": round(c.share, 4),
                "rate": c.render(),
                "ungraded": c.ungraded,
            }
            for c in items
        ]

    return {
        "proven": True,
        "receipt": {
            "incumbent": receipt.incumbent,
            "candidate": receipt.candidate,
            "issued": receipt.issued,
            "digest": receipt.digest(),
            "eval_set": receipt.eval_set,
            "bar": receipt.bar,
            "movable_share": round(receipt.movable_share, 4),
            "complete": receipt.complete,
            "judge": receipt.judge_agreement
            or "none used — every claim is from deterministic graders",
            "judge_trustworthy": receipt.judge_trustworthy,
            "traffic_captures": receipt.traffic_captures,
        },
        "scores": {
            "proven": claims(receipt.proven),
            "regret": claims(receipt.regret),
            "unproven": claims(receipt.unproven),
        },
        "note": (
            "every rate carries its confidence interval and its denominator. "
            "Clusters under `regret` were proven BELOW the bar and must stay on "
            "the incumbent; `unproven` means not enough evidence either way"
        ),
    }


def _readme(box_model: ModelSpec, generated: str, targets, skipped, weights, ref, proof) -> str:
    lines = [
        f"# {box_model.name} — clickllm box",
        "",
        f"Generated {generated} by `clickllm box {box_model.id}`. "
        "One artifact, one registry, one command, one OpenAI-compatible endpoint.",
        "",
        f"**Proof:** {proof}",
        "",
        "## What is in here",
        "",
        "| Path | What it is |",
        "|---|---|",
        "| `manifest.yaml` | model, quant, tuned knobs, provenance, eval scores |",
        "| `weights.lock` | the repos and revisions to fetch — nothing is vendored |",
        "| `bench.json` | roofline estimates, explicitly not measurements |",
    ]
    for t in targets:
        lines.append(f"| `targets/{t.target.id}/` | {t.target.why} |")
    lines += [
        "",
        "## Targets",
        "",
    ]
    for t in targets:
        lines += [
            f"### `{t.target.id}` — {t.engine} @ {t.quant}",
            "",
            f"- tuned on **{t.host_class}**: {t.fit.total_bytes / GB:.1f} GB of "
            f"{t.fit.usable_bytes / GB:.1f} GB usable at {t.fit.context:,} ctx "
            f"x{t.fit.concurrency}",
            f"- engine chosen because {t.engine_why}",
            f"- speed: {t.speed}",
            f"- weights: `{t.repo}`",
        ]
        if t.gaps:
            lines.append("- not expressed by this engine: " + "; ".join(t.gaps))
        for w in t.warnings:
            lines.append(f"- warning: {w}")
        lines += [
            "",
            "```sh",
            (
                f"cd targets/{t.target.id} && docker compose up"
                if t.target.binding == "container"
                else f"sh targets/{t.target.id}/serve.sh"
            ),
            "```",
            "",
        ]
    if skipped:
        lines += ["## Not built", ""]
        lines += [f"- **{s.target.id}** — {s.reason}" for s in skipped]
        lines.append("")
    lines += ["## Not in this box", ""]
    lines += [f"- **{a['path']}** — {a['reason']}" for a in ABSENT]
    lines += [
        "",
        "## Weights",
        "",
        "Fetched at run time, never vendored. Pinned where the index gave a revision:",
        "",
    ]
    for w in weights:
        pin = f"`{w.revision}`" if w.revision else f"**unpinned** — {w.unpinned_reason}"
        lines.append(f"- `{w.repo}` @ {pin}  ({', '.join(w.targets)})")
    lines += [
        "",
        "## Publishing",
        "",
        "clickllm does not push. It holds no registry credential, reads no token, "
        "and touches nothing in your environment. Run these yourself:",
        "",
        "```sh",
        *(
            [
                f"oras push {ref} --artifact-type {ARTIFACT_TYPE} \\",
                "  manifest.yaml:application/yaml weights.lock:application/yaml \\",
                "  bench.json:application/json README.md:text/markdown targets",
            ]
        ),
        "```",
        "",
        "## Re-tune on arrival",
        "",
        "These knobs were solved for the host classes above. A box landing on a "
        "different one should re-detect, re-solve, re-benchmark, and revert any "
        "optimisation that does not help there — a box is a tuned starting point "
        "plus the evidence behind it, never a frozen command line (ADR-0005).",
        "",
    ]
    return "\n".join(lines)


def build(
    model_id: str,
    *,
    quant: str | None = None,
    context: int = 32_768,
    concurrency: int = 8,
    receipt: Receipt | None = None,
    ref: str | None = None,
    targets: tuple[Target, ...] = TARGETS,
    today: str | None = None,
    exists=None,
    revision=None,
    cache: Path | None = None,
) -> Box:
    """Assemble a box for `model_id`. Writes nothing; pushes nothing.

    Args:
        model_id: catalogue id, e.g. `qwen3-30b-a3b`.
        quant: force a quantisation on every target. Default is the best that
            fits each host class, which is usually not the same one.
        context: context length the knobs are tuned for.
        concurrency: simultaneous in-flight requests to tune for. 8 rather than
            1: a box is published for a fleet, not for a single-stream session.
        receipt: proof to attach. Absent means the manifest states no proof was
            run — never a score.
        ref: registry reference the printed push commands use.
        targets: host classes to build for.
        today: generation date. Injected so the output is not a function of the
            calendar.
        exists: repo existence check, forwarded to
            [`resolve_weights`][clickllm.launch.resolve_weights].
        revision: `repo -> sha | None`. Defaults to [`hf_revision`]. Injected so
            the test suite never opens a socket.
        cache: weight-resolution cache path, forwarded to `resolve_weights`.

    Returns:
        A [`Box`]. Call `write()` to put it on disk.

    Raises:
        KeyError: the model id is not in the catalogue.
        ValueError: `quant` is not one this model publishes, the receipt proves
            a different model, or no target could be built at all.
    """
    model = catalog.get(model_id)
    if quant is not None and quant not in model.quants:
        raise ValueError(f"{model.id} does not publish {quant!r}; it has {', '.join(model.quants)}")
    if receipt is not None:
        known = {model.id.lower(), model.name.lower(), (model.repo or "").lower()}
        if receipt.candidate.lower() not in known:
            raise ValueError(
                f"the receipt proves {receipt.candidate!r}, not {model.id!r} — "
                f"attaching it would claim evidence this box does not have"
            )

    generated = today or date.today().isoformat()
    reference = ref or f"ghcr.io/<your-org>/{model.id}:v1"
    revision_of = revision or hf_revision
    proof = (
        f"receipt {receipt.digest()[:12]} — {receipt.movable_share:.0%} of captured "
        f"traffic proven at the {receipt.bar:.0%} bar"
        if receipt is not None
        else "none. No proof was run, so this box makes no quality claim."
    )

    built: list[TargetPlan] = []
    skipped: list[Skipped] = []
    repos: dict[str, list[str]] = {}

    for target in targets:
        solved = _solve(model, target, quant, context, concurrency)
        if isinstance(solved, Skipped):
            skipped.append(solved)
            continue
        host_class, sized, p = solved

        engine = p.engine.value
        if adapter_for(engine) is None:
            skipped.append(
                Skipped(
                    target,
                    f"the planner chose {engine} for {host_class} and there is no "
                    f"verified flag dialect for it; emitting another engine's "
                    f"flags would produce a config that cannot start",
                )
            )
            continue

        resolved = resolve_weights(model, sized.quant, engine, exists=exists, cache=cache)
        if isinstance(resolved, Refusal):
            skipped.append(Skipped(target, resolved.reason, resolved.tried))
            continue

        argv, gaps = p.command(resolved.repo)
        argv = _bind(target.binding, argv)
        t = TargetPlan(
            target=target,
            host_class=host_class,
            engine=engine,
            engine_why=p.engine_why,
            quant=sized.quant,
            repo=resolved.repo,
            argv=tuple(argv),
            knobs=p.knobs,
            gaps=gaps,
            warnings=p.warnings,
            fit=sized,
            files=(),
        )
        built.append(replace(t, files=tuple(_files_for(model, generated, t, p, proof))))
        repos.setdefault(resolved.repo, []).append(target.id)

    if not built:
        detail = "; ".join(f"{s.target.id}: {s.reason}" for s in skipped)
        raise ValueError(f"no target could be built for {model.id!r} — {detail}")

    weights = []
    for repo, used_by in repos.items():
        sha = revision_of(repo)
        weights.append(
            Weight(
                repo=repo,
                revision=sha,
                targets=tuple(used_by),
                unpinned_reason=(
                    ""
                    if sha
                    else "the model index returned no revision for this repo, so "
                    "nothing is pinned here. Pulling `main` is not reproducible "
                    "and this lock will not pretend otherwise"
                ),
            )
        )

    files: list[tuple[str, str]] = [
        (
            "manifest.yaml",
            _manifest(model, generated, built, skipped, context, concurrency, receipt),
        ),
        ("weights.lock", _lock(model, generated, weights)),
        ("bench.json", _bench(generated, built)),
        (
            "README.md",
            _readme(model, generated, built, skipped, weights, reference, proof),
        ),
    ]
    for t in built:
        files += [(f"targets/{t.target.id}/{name}", body) for name, body in t.files]

    return Box(
        model=model,
        generated=generated,
        targets=tuple(built),
        skipped=tuple(skipped),
        weights=tuple(weights),
        files=tuple(files),
        ref=reference,
        proven=receipt is not None,
    )


def _files_for(
    model: ModelSpec, generated: str, t: TargetPlan, p: Plan, proof: str
) -> list[tuple[str, str]]:
    """The files for one target, relative to its own directory."""
    head = _target_header(model, generated, t, proof)
    if t.target.binding == "native":
        return _native(head, t, list(t.argv), _install_hint(t.engine))

    image = _image_for(t.target, t.engine)
    port = PORTS.get(p.engine, DEFAULT_PORT)
    args = _container_args(t.engine, t.argv)
    tp = p.get(Setting.TENSOR_PARALLEL)
    devices = int(tp.value) if tp else 1
    return [
        ("Dockerfile", _dockerfile(head, image, args, port)),
        ("docker-compose.yml", _compose(t.target, head, image, args, port, devices)),
        ("kubernetes.yaml", _kubernetes(head, t, p, image, port)),
    ]


def _manifest(
    model: ModelSpec,
    generated: str,
    built: list[TargetPlan],
    skipped: list[Skipped],
    context: int,
    concurrency: int,
    receipt: Receipt | None,
) -> str:
    head = _header(
        model,
        generated,
        [
            *_field(
                "MANIFEST",
                "the declaration is portable across hosts; only the execution "
                "bindings under targets/ differ.",
            )
        ],
    )
    body: dict[str, Any] = {
        "format": FORMAT,
        "generated": generated,
        "artifact_type": ARTIFACT_TYPE,
        "model": {
            "id": model.id,
            "name": model.name,
            "params_b": model.params_b,
            "active_b": model.active_b,
            "moe": model.is_moe,
            "kv_scheme": model.kv_scheme,
            "max_context": model.max_context,
            "license": model.license,
            "license_ok": model.license_ok,
            "catalogue_repo": model.repo,
            "architecture_verified": model.verified,
        },
        "quant": {t.target.id: t.quant for t in built},
        "workload": {
            "profile": Workload.INTERACTIVE.value,
            "context": context,
            "concurrency": concurrency,
        },
        "targets": [
            {
                "id": t.target.id,
                "binding": t.target.binding,
                "tuned_on": t.host_class,
                "engine": t.engine,
                "engine_why": t.engine_why,
                "quant": t.quant,
                "weights": t.repo,
                "argv": list(t.argv),
                "knobs": [
                    {"setting": k.name.value, "value": str(k.value), "why": k.why} for k in t.knobs
                ],
                "not_expressed": list(t.gaps),
                "warnings": list(t.warnings),
                "memory": {
                    "required_gb": round(t.fit.total_bytes / GB, 1),
                    "usable_gb": round(t.fit.usable_bytes / GB, 1),
                },
                "speed": t.speed,
            }
            for t in built
        ],
        "not_built": [{"id": s.target.id, "reason": s.reason} for s in skipped],
        "absent": [dict(a) for a in ABSENT],
        "evaluation": _evaluation(receipt),
        "provenance": {
            "tool": "clickllm box",
            "chosen": [
                f"{t.target.id}: {t.engine} @ {t.quant} on {t.host_class} — {t.engine_why}"
                for t in built
            ],
            "weights": (
                "fetched at run time from the repos in weights.lock; nothing is "
                "vendored in this artifact"
            ),
            "standalone": (
                "every file under targets/ is the platform's own config and runs "
                "with clickllm uninstalled"
            ),
            "retune": (
                "knobs were solved per host class above. On arrival, re-detect, "
                "re-solve, re-benchmark and revert what does not help here"
            ),
            "estimates": (
                "every tok/s figure in this box is a roofline estimate from memory "
                "bandwidth, not a measurement"
            ),
        },
    }
    return _yaml_doc(head, body)


def _lock(model: ModelSpec, generated: str, weights: list[Weight]) -> str:
    head = _header(
        model,
        generated,
        [
            *_field(
                "LOCK",
                "weights are fetched, not vendored. A repo with `pinned: false` "
                "has no revision to pin — pulling it is not reproducible, and "
                "this file will not pretend otherwise.",
            )
        ],
    )
    body = {
        "format": LOCK_FORMAT,
        "resolved": generated,
        "model": model.id,
        "weights": [
            {
                "repo": w.repo,
                "source": f"https://huggingface.co/{w.repo}",
                "revision": w.revision,
                "pinned": w.pinned,
                **({} if w.pinned else {"unpinned_reason": w.unpinned_reason}),
                "targets": list(w.targets),
                "fetch": f"huggingface-cli download {w.repo}"
                + (f" --revision {w.revision}" if w.revision else ""),
            }
            for w in weights
        ],
        "vendored": False,
        "note": (
            "the engine downloads these on first start. Nothing in this box contains model weights"
        ),
    }
    return _yaml_doc(head, body)


def _bench(generated: str, built: list[TargetPlan]) -> str:
    """Roofline estimates, and the fact that nothing was measured.

    ADR-0005 puts `bench.json` in the tree as "the measurements that ratified the
    tuning". This build ratifies nothing — so the file carries the estimate, the
    arithmetic behind it, and `measured: false` on every row.
    """
    body = {
        "format": BENCH_FORMAT,
        "generated": generated,
        "measured": False,
        "note": (
            "nothing in this box was benchmarked. Every figure below is a "
            "roofline estimate derived from memory bandwidth; a real number "
            "comes from running the box on the host and measuring it"
        ),
        "estimates": [
            {
                "target": t.target.id,
                "host_class": t.host_class,
                "engine": t.engine,
                "quant": t.quant,
                "tokens_per_sec": (
                    round(t.fit.tokens_per_sec, 1) if t.fit.tokens_per_sec else None
                ),
                "measured": False,
                "confidence": t.speed,
                "method": (
                    "memory bandwidth / bytes read per decoded token, derated for "
                    "achievable bandwidth. Single stream; batching changes it"
                ),
            }
            for t in built
        ],
    }
    return json.dumps(body, indent=2) + "\n"


def demo() -> None:
    """Self-check. Run with `python -m clickllm.box`. Touches no network."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "weights.json"
        box = build(
            "qwen3-30b-a3b",
            today="2026-07-28",
            exists=lambda r: True,
            revision=lambda r: "a" * 40 if r.startswith("Qwen/") else None,
            cache=cache,
        )

        names = {n for n, _ in box.files}
        assert {"manifest.yaml", "weights.lock", "bench.json", "README.md"} <= names, names
        assert any(n.startswith("targets/linux-cuda/") for n in names), names
        assert any(n.startswith("targets/darwin-metal/serve.sh") for n in names), names

        # The lock never claims a revision it does not have.
        unpinned = [w for w in box.weights if not w.pinned]
        assert unpinned and all(w.unpinned_reason for w in unpinned), box.weights
        lock = dict(box.files)["weights.lock"]
        assert "pinned: false" in lock and "unpinned_reason" in lock

        # No receipt, no score. Ever.
        manifest = dict(box.files)["manifest.yaml"]
        assert "proven: false" in manifest and "scores: null" in manifest
        assert "no proof was run" in manifest

        # Nothing generated needs clickllm installed.
        for name, content in box.files:
            assert "import clickllm" not in content, name
            assert "pip install clickllm" not in content, name

        # Publishing is a command you run, not something this did.
        pushes = box.push_commands(tmp)
        assert any(c.startswith("oras push") for c in pushes), pushes
        assert any("docker push" in c for c in pushes), pushes

        written = box.write(Path(tmp) / "out")
        assert len(written) == len(box.files) and all(p.read_text() for p in written)

    print(f"box: {box.model.id} -> {len(box.files)} files, {len(box.targets)} target(s)")
    print("ok")


if __name__ == "__main__":
    demo()
