"""The kernel seam — clickllm does not write kernels, it proves them.

[`CLAUDE.md`'s first invariant] says never build an inference engine. That
applies to kernels: out-engineering vLLM's kernel authors is not the gap, and a
custom attention kernel maintained against upstream churn is a permanent tax
whose failure mode is subtle numerical drift that presents as a *model quality*
problem.

But "we do not write them" is not the same as "you cannot". If you have a fused
kernel, a quantisation format, or a whole out-of-tree platform, the honest
question is not whether it compiles — it is **whether it actually helped**, and
that is the one question the rest of this codebase already answers.

So this module is two things and deliberately not a third:

1. **A description of the seam**, built on vLLM's own plugin entry points, so a
   kernel arrives as a normal Python package and not a fork.
2. **A proof harness**, which is the part nobody else gives you: run the same
   eval set against stock and against your kernel, and get a receipt saying
   whether the outputs are equivalent and whether the speedup is real.
3. **Not a kernel.** There is no CUDA in this repo and there should not be.

## Why the proof half matters more than the plumbing

A kernel that is 1.4× faster and changes one logit in ten thousand is not a
1.4× win, it is an unreviewed model change. The usual workflow — benchmark it,
eyeball a few outputs, ship — cannot tell those apart. `clickllm prove` can,
because it is the same machinery that decides whether a whole different model is
equivalent, and a kernel is a much smaller perturbation than that.

## The entry points

Verified against vLLM's plugin-system documentation (see [`SOURCE`]). Names and
semantics differ per group and getting them wrong means a plugin that silently
never loads:

| Group | What it registers | Returns |
|---|---|---|
| `vllm.general_plugins` | out-of-tree models, custom ops | nothing; side effects only |
| `vllm.platform_plugins` | an out-of-tree device | the platform class's FQN, or
  `None` if unusable here |
| `vllm.stat_logger_plugins` | a metrics logger | the entry point *is* the class |
| `vllm.io_processor_plugins` | prompt/output processing | the IOProcessor class's FQN |
| `vllm.endpoint_plugins` | extra HTTP routes | **not loaded by default** — opt in |

The constraint that catches people: **the callable must be re-entrant.** vLLM
loads plugins in every process it spawns, so a `register()` that appends to a
list or raises on a second call will fail once tensor parallelism starts more
than one worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ENTRY_POINT_GROUPS",
    "SOURCE",
    "KernelClaim",
    "Plugin",
    "PluginKind",
    "demo",
    "scaffold",
    "verification_plan",
]

#: Where the entry-point contract was verified.
SOURCE = "https://docs.vllm.ai/en/stable/design/plugin_system.html (checked 2026-07-27)"


class PluginKind(StrEnum):
    """A vLLM plugin group. The value is the literal entry-point group name."""

    GENERAL = "vllm.general_plugins"
    PLATFORM = "vllm.platform_plugins"
    STAT_LOGGER = "vllm.stat_logger_plugins"
    IO_PROCESSOR = "vllm.io_processor_plugins"
    ENDPOINT = "vllm.endpoint_plugins"


#: What each group expects back, because they genuinely differ and a wrong
#: return is a plugin that loads and does nothing.
ENTRY_POINT_GROUPS: dict[PluginKind, str] = {
    PluginKind.GENERAL: "no return value — register your op or model as a side effect",
    PluginKind.PLATFORM: "the platform class's fully-qualified name, or None if it "
    "cannot run in this environment",
    PluginKind.STAT_LOGGER: "the entry point is the class itself, subclassing StatLoggerBase",
    PluginKind.IO_PROCESSOR: "the IOProcessor class's fully-qualified name",
    PluginKind.ENDPOINT: "routes to add — note this group is NOT loaded by default",
}


#: What the entry point must resolve to, per group. `register` for the four
#: groups whose entry point is a callable, and the class itself for STAT_LOGGER,
#: whose contract is "the entry point is the class".
#:
#: One home, because `scaffold()` emits the module and `cli.py` builds the
#: target, and they disagreed: the STAT_LOGGER scaffold emitted `MyStatLogger`
#: while the CLI still wrote `pkg:register`, so the generated package installed
#: with an entry point resolving to a name that no longer existed in it. Two
#: places encoding one fact, fixed in one of them.
ENTRY_POINT_ATTR: dict[PluginKind, str] = {
    PluginKind.GENERAL: "register",
    PluginKind.PLATFORM: "register",
    PluginKind.STAT_LOGGER: "MyStatLogger",
    PluginKind.IO_PROCESSOR: "register",
    PluginKind.ENDPOINT: "register",
}


def entry_point_target(package: str, kind: PluginKind) -> str:
    """The `module:attribute` an entry point for this group must name."""
    return f"{package.replace('-', '_')}:{ENTRY_POINT_ATTR[kind]}"


@dataclass(frozen=True, slots=True)
class Plugin:
    """A kernel or platform plugin, as a package rather than a fork."""

    name: str
    kind: PluginKind
    #: Dotted path to the callable, e.g. `my_kernels:register`.
    target: str
    package: str = ""

    def entry_point(self) -> str:
        """The `name = module:callable` line for this plugin."""
        return f"{self.name} = {self.target}"

    def pyproject_fragment(self) -> str:
        """The `[project.entry-points]` block to paste into a pyproject.toml."""
        return f'[project.entry-points."{self.kind.value}"]\n{self.name} = "{self.target}"\n'


@dataclass(frozen=True, slots=True)
class KernelClaim:
    """What a kernel is asserted to do, so it can be checked rather than believed.

    Both fields are required on purpose. "It is faster" with no equivalence
    claim is the shape of every kernel regression that ever shipped: the speedup
    is measured, the output change is not, and the two are reported as one
    number.
    """

    #: Human name for the change, e.g. "fused RMSNorm + residual".
    name: str
    #: Expected speedup, as a multiple. Claimed, not measured — that is the point.
    claimed_speedup: float
    #: Whether the author expects bit-identical output. Almost always false for
    #: anything touching floating-point reduction order.
    bit_identical: bool = False
    #: What the author expects to change, if not bit-identical.
    expected_drift: str = ""

    def render(self) -> str:
        """One line, for a report."""
        exact = (
            "bit-identical"
            if self.bit_identical
            else f"drift expected: {self.expected_drift or 'unstated'}"
        )
        return f"{self.name}: claims {self.claimed_speedup:.2f}× · {exact}"


def verification_plan(claim: KernelClaim) -> list[str]:
    """The steps that turn a kernel claim into evidence.

    Ordered so the cheapest disqualifying check runs first. A kernel that
    changes outputs on the eval set has failed regardless of how fast it is, and
    finding that out before a benchmark sweep saves the sweep.
    """
    steps = [
        "capture or reuse an eval set from real traffic — a kernel that is "
        "correct on a benchmark and wrong on your prompts is the failure this "
        "whole loop exists to catch",
        "run the eval set against **stock** and record a receipt; that digest is "
        "the baseline the kernel is compared against, and it must not be "
        "regenerated afterwards",
    ]
    if claim.bit_identical:
        steps.append(
            "assert bit-identical output on every eval item. A single differing "
            "token falsifies the claim outright — no statistics needed, and no "
            "judgement call to argue about"
        )
    else:
        steps.append(
            f"the author expects drift ({claim.expected_drift or 'unspecified'}), "
            f"so equivalence is a statistical question: run `clickllm prove` and "
            f"require the interval's lower bound to clear the bar, not the point "
            f"estimate"
        )
    steps += [
        "measure throughput at YOUR concurrency, not at batch 1. Kernel speedups "
        "quoted single-stream routinely vanish under real batching, which is the "
        "same trap speculative decoding sets",
        f"compare against the claim: {claim.claimed_speedup:.2f}×. A kernel that "
        f"delivers materially less than claimed is not a win to be rounded up, it "
        f"is a result to be reported",
        "keep the receipt. When the kernel is rebuilt against a newer vLLM, the "
        "guard's fingerprint check tells you the proof no longer applies",
    ]
    return steps


def scaffold(plugin: Plugin, claim: KernelClaim | None = None) -> dict[str, str]:
    """Files for a plugin package that loads, and a note on what to prove.

    Returns `{path: contents}` rather than writing anything — the caller decides
    where these land, and a function that writes to disk cannot be tested
    without a filesystem.
    """
    pkg = plugin.package or plugin.name.replace("-", "_")
    files: dict[str, str] = {}

    files["pyproject.toml"] = (
        f"[project]\n"
        f'name = "{plugin.name}"\n'
        f'version = "0.1.0"\n'
        f'requires-python = ">=3.11"\n\n'
        f"# vLLM discovers plugins through standard entry points, so this is a\n"
        f"# normal package — not a fork, and not a patched vLLM image.\n"
        f"# Group semantics differ; see clickllm.kernels.ENTRY_POINT_GROUPS.\n"
        f"{plugin.pyproject_fragment()}\n"
        f"[build-system]\n"
        f'requires = ["hatchling"]\n'
        f'build-backend = "hatchling.build"\n'
    )

    # One template per group, because the five groups genuinely differ in what
    # the entry point must *be* — and this branched two ways, so STAT_LOGGER,
    # IO_PROCESSOR and ENDPOINT all received the GENERAL boilerplate: a
    # side-effecting `register()` that loads a torch op library and returns
    # nothing. For those three that is not merely unhelpful, it is the wrong
    # artifact. `ENTRY_POINT_GROUPS` above says so in this same file, which is
    # the part that makes it a defect rather than a gap: the module documented
    # the contract it then failed to emit.
    #
    # A plugin scaffolded wrong loads cleanly and does nothing, which is the
    # failure mode this module exists to prevent.
    if plugin.kind is PluginKind.STAT_LOGGER:
        # The entry point *is* the class. A `register()` here would resolve to a
        # function where vLLM expects a type, and the plugin would be skipped.
        module = (
            f'"""{plugin.name} — a {plugin.kind.value} plugin."""\n\n'
            "from __future__ import annotations\n\n"
            "from vllm.v1.metrics.loggers import StatLoggerBase\n\n\n"
            "class MyStatLogger(StatLoggerBase):\n"
            '    """The entry point is this class itself, not a function.\n\n'
            "    vLLM instantiates one per engine and calls `record` on every\n"
            "    iteration, so anything slow here is in the serving loop.\n"
            '    """\n\n'
            "    def __init__(self, vllm_config, engine_idx=0):\n"
            "        self.vllm_config = vllm_config\n"
            "        self.engine_idx = engine_idx\n\n"
            "    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, "
            "engine_idx=0):\n"
            "        raise NotImplementedError\n\n"
            "    def log(self):\n"
            "        raise NotImplementedError\n\n"
            "    def log_engine_initialized(self):\n"
            '        """Called once, after engine startup — before any `record` call."""\n'
            "        raise NotImplementedError\n"
        )
    elif plugin.kind is PluginKind.IO_PROCESSOR:
        module = (
            f'"""{plugin.name} — a {plugin.kind.value} plugin."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def register():\n"
            '    """Return the IOProcessor class\'s fully-qualified name.\n\n'
            "    A string, not a side effect: vLLM imports the name you return.\n"
            "    Returning None here means the processor is unavailable, which is\n"
            "    how you decline on a machine that cannot support it.\n"
            '    """\n'
            f'    return "{pkg}.processor.MyIOProcessor"\n'
        )
    elif plugin.kind is PluginKind.ENDPOINT:
        module = (
            f'"""{plugin.name} — a {plugin.kind.value} plugin."""\n\n'
            "from __future__ import annotations\n\n"
            "from fastapi import APIRouter, FastAPI\n\n\n"
            "class MyEndpointPlugin:\n"
            '    """Implements vLLM\'s `EndpointPlugin` protocol.\n\n'
            "    This group is NOT loaded by default — vLLM must be started with\n"
            "    the endpoint plugin explicitly enabled, so a scaffold that works\n"
            "    and appears to do nothing is usually this and not your code.\n"
            '    """\n\n'
            '    name = "my-endpoint"\n\n'
            "    #: Tasks the engine must support for this endpoint to attach.\n"
            "    #: The loader reads this before calling `attach_router`.\n"
            '    required_tasks = frozenset({"generate"})\n\n'
            "    def attach_router(self, app: FastAPI) -> None:\n"
            "        router = APIRouter()\n\n"
            '        @router.get("/my-endpoint")\n'
            "        async def my_endpoint():\n"
            '            return {"ok": True}\n\n'
            "        app.include_router(router)\n\n"
            "    def init_state(self, *args, **kwargs) -> None:\n"
            '        """Called once at startup with engine state. Nothing to keep here."""\n\n\n'
            "def register() -> MyEndpointPlugin:\n"
            '    """Zero-argument factory. The loader calls this, then reads\n'
            "    `required_tasks` and calls `attach_router` on what it returns —\n"
            "    an `APIRouter` here would resolve to a value the loader cannot\n"
            "    read `required_tasks` from at all.\n"
            '    """\n'
            "    return MyEndpointPlugin()\n"
        )
    elif plugin.kind is PluginKind.PLATFORM:
        module = (
            f'"""{plugin.name} — a {plugin.kind.value} plugin."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def register():\n"
            '    """Return the platform class, or None if it cannot run here.\n\n'
            "    Returning None is not a failure — it is how a plugin says the\n"
            "    hardware it targets is absent, which must not stop vLLM starting\n"
            "    on a machine that simply does not have it.\n"
            '    """\n'
            "    try:\n"
            "        import my_device_runtime  # noqa: F401\n"
            "    except ImportError:\n"
            "        return None\n"
            f'    return "{pkg}.platform.MyPlatform"\n'
        )
    else:
        module = (
            f'"""{plugin.name} — a {plugin.kind.value} plugin."""\n\n'
            "from __future__ import annotations\n\n\n"
            "def register():\n"
            '    """Register the op. Called once per process — must be re-entrant.\n\n'
            "    vLLM loads plugins in every process it spawns, so under tensor\n"
            "    parallelism this runs once per worker. A register() that appends\n"
            "    to a list or raises on a second call breaks the moment TP > 1,\n"
            "    and the error surfaces far from the cause.\n"
            '    """\n'
            "    import torch\n\n"
            '    if hasattr(torch.ops, "my_kernels"):\n'
            "        return  # already registered in this process\n"
            "    torch.ops.load_library(_library_path())\n"
        )

    files[f"{pkg}/__init__.py"] = module

    if plugin.kind is PluginKind.IO_PROCESSOR:
        # `register()` above returns this FQN as a string; vLLM imports it by
        # name, so the class it names must actually exist in the package.
        files[f"{pkg}/processor.py"] = (
            f'"""{plugin.name} — the class `register()`\'s FQN points at."""\n\n'
            "from __future__ import annotations\n\n"
            "from vllm.plugins.io_processors.interface import IOProcessor\n\n\n"
            "class MyIOProcessor(IOProcessor):\n"
            '    """The IOProcessor `register()` names by fully-qualified name."""\n\n'
            "    def pre_process(self, prompt, request_id=None, **kwargs):\n"
            "        raise NotImplementedError\n\n"
            "    def post_process(self, model_output, request_id=None, **kwargs):\n"
            "        raise NotImplementedError\n"
        )

    steps = verification_plan(claim) if claim else []
    files["PROVING.md"] = (
        f"# Proving {plugin.name}\n\n"
        + (f"**Claim:** {claim.render()}\n\n" if claim else "")
        + "Loading is the easy half. These steps are the half that decides whether\n"
        "the kernel should ship.\n\n"
        + "".join(f"{i + 1}. {s}\n" for i, s in enumerate(steps))
        + "\n> A kernel that is 1.4× faster and changes one logit in ten thousand is\n"
        "> not a 1.4× win. It is an unreviewed model change wearing a benchmark.\n"
    )
    return files


def demo() -> None:
    """Self-check. Run with `python -m clickllm.kernels`."""
    p = Plugin(name="fused-rmsnorm", kind=PluginKind.GENERAL, target="fused_rmsnorm:register")
    assert p.entry_point() == "fused-rmsnorm = fused_rmsnorm:register"
    assert "vllm.general_plugins" in p.pyproject_fragment()

    files = scaffold(p, KernelClaim("fused RMSNorm + residual", 1.18, expected_drift="last-bit"))
    assert set(files) == {"pyproject.toml", "fused_rmsnorm/__init__.py", "PROVING.md"}
    # The generated package must declare the group vLLM actually reads.
    assert PluginKind.GENERAL.value in files["pyproject.toml"]
    # Re-entrancy is the constraint that bites under tensor parallelism.
    assert "re-entrant" in files["fused_rmsnorm/__init__.py"]
    assert "already registered" in files["fused_rmsnorm/__init__.py"]

    # A platform plugin returns a name, and None means "not here" rather than
    # "broken" — a distinction that decides whether vLLM starts at all.
    plat = Plugin("my-npu", PluginKind.PLATFORM, "my_npu:register")
    src = scaffold(plat)["my_npu/__init__.py"]
    assert "return None" in src and "not a failure" in src
    assert "my_npu.platform.MyPlatform" in src

    # A bit-identical claim is falsifiable outright; a drifting one is not.
    exact = verification_plan(KernelClaim("exact", 1.1, bit_identical=True))
    fuzzy = verification_plan(KernelClaim("fuzzy", 1.1, expected_drift="fp16 reduction order"))
    assert any("bit-identical" in s for s in exact)
    assert any("lower bound" in s for s in fuzzy)
    assert not any("bit-identical output" in s for s in fuzzy)

    # Every plan measures at real concurrency, because single-stream kernel wins
    # routinely vanish under batching.
    for plan in (exact, fuzzy):
        assert any("not at batch 1" in s for s in plan)
        assert any("keep the receipt" in s for s in plan)

    # Every group has its return contract documented — a wrong return is a
    # plugin that loads and does nothing.
    assert set(ENTRY_POINT_GROUPS) == set(PluginKind)
    assert "NOT loaded by default" in ENTRY_POINT_GROUPS[PluginKind.ENDPOINT]

    print("kernels: ok")


if __name__ == "__main__":
    demo()
