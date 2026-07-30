"""The flags we emit, checked against the engine that has to accept them.

`engines.py` cites a docs URL for every flag. Docs have been wrong twice already
in ways that passed every test in this repo and could not start a server:
`--speculative-config eagle3` (needs JSON) and `--quantization q4` (needs a
method name, not a precision). MLX escaped both, because MLX is the one engine
the author can install and `installed_flags` reads its `--help` off the binary.

This does the same for vLLM and SGLang. It cannot run on a Mac — neither engine
builds there — so it runs on Linux in CI (`.github/workflows/engine-flags.yml`)
and for free on any developer machine where the engine happens to be installed.

## The skip is the dangerous part

`installed_flags` returns `None` for "could not ask", which is deliberately not
the same as "accepts nothing". Locally that is a skip. In CI a skip would be a
green tick over an unasked question — the exact defect this repo removed from
its own CI once already — so `CLICKLLM_REQUIRE_ENGINES=1` turns "could not ask"
into a failure. That variable is what makes the check able to fail, and a check
that cannot fail is worse than none.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from clickllm import catalog
from clickllm.engines import LoraFleet, adapter_for
from clickllm.hardware import Hardware
from clickllm.plan import Requirements, Workload, plan

#: Synthetic accelerators. The CI runner has no GPU and never will — this check
#: is about what the engine's argument parser accepts, not about running a model.
H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * 2**30,
    usable_bytes=76 * 2**30,
    bandwidth_gbps=3350.0,
    cores=132,
)
QUAD = replace(H100, devices=4)

#: Set in CI. Makes "the engine could not be interrogated" a failure rather than
#: a skip, so an install that silently broke cannot read as a pass.
REQUIRE = "CLICKLLM_REQUIRE_ENGINES"

#: (label, model id, quantisation, hardware, requirements). Chosen to cover every
#: flag the planner can emit, not to be representative of demand: tensor
#: parallelism, structured output, both LoRA dialects, fp8 quantisation, the MTP
#: speculative config, and both workloads' opposite prefill settings.
_CASES = (
    ("interactive 8B", "llama-3.1-8b", "fp16", H100, Requirements(Workload.INTERACTIVE, 8)),
    ("batch 32B", "qwen3-32b", "q4", H100, Requirements(Workload.BATCH, 32)),
    (
        "realtime 70B, 4-way, structured",
        "llama-3.3-70b",
        "q4",
        QUAD,
        Requirements(Workload.REALTIME, 8, itl_ms=50, structured_output=True),
    ),
    (
        # The only case that emits --speculative-config: this family carries its
        # own MTP head, so the planner has a method it can name without a draft
        # checkpoint. Also the only one that emits --quantization.
        "deepseek-v3 fp8, mtp speculative",
        "deepseek-v3",
        "fp8",
        QUAD,
        Requirements(Workload.INTERACTIVE, 4),
    ),
    (
        "vllm multi-LoRA",
        "llama-3.1-8b",
        "fp16",
        H100,
        Requirements(
            Workload.INTERACTIVE,
            2,
            lora=LoraFleet((("support", "org/support"), ("sql", "org/sql")), 48, 2),
        ),
    ),
    (
        # prefix_sharing above PREFIX_SHARING_FOR_RADIX is what selects SGLang;
        # below it the planner picks vLLM and this file would never test SGLang.
        "sglang multi-LoRA, high prefix sharing",
        "qwen3-32b",
        "q4",
        H100,
        Requirements(
            Workload.INTERACTIVE,
            8,
            prefix_sharing=0.8,
            lora=LoraFleet((("support", "org/support"), ("sql", "org/sql")), 48, 2),
        ),
    ),
    (
        "sglang batch, high prefix sharing",
        "mistral-small-24b",
        "fp16",
        H100,
        Requirements(Workload.BATCH, 32, prefix_sharing=0.9),
    ),
)


def cases() -> list[tuple[str, str, list[str]]]:
    """(label, engine, argv) for every case, generated exactly as the CLI does.

    Through `plan()` and `Plan.command()` rather than by hand: a hand-written
    argv would test this file's idea of the flags, which is the thing already
    known to drift. `--host`/`--port` are appended because `launch` appends them,
    and they are flags the engine has to accept too.
    """
    out = []
    for label, model_id, quant, hw, req in _CASES:
        model = catalog.get(model_id)
        p = plan(hw, req, model, quant)
        argv, _gaps = p.command(model.repo or model.id)
        if not argv:
            # No verified dialect for the chosen engine (llm-d). Nothing to check.
            continue
        out.append((label, p.engine.value, argv + ["--host", "127.0.0.1", "--port", "8000"]))
    return out


def _for(engine: str) -> list[tuple[str, list[str]]]:
    return [(label, argv) for label, eng, argv in cases() if eng == engine]


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_every_generated_flag_is_one_the_installed_engine_accepts(engine: str):
    """Every `--flag` we emit appears in the installed engine's own `--help`.

    Not a docs check. `installed_flags` runs the real binary, so this fails the
    day an engine renames a flag underneath us — which is how the last two got
    found, months after they shipped.
    """
    from clickllm.engines import installed_flags, unknown_flags

    adapter = adapter_for(engine)
    assert adapter is not None, engine

    if installed_flags(adapter) is None:
        why = f"{engine} could not be interrogated: `{' '.join(adapter.help_argv)}` did not run"
        if os.environ.get(REQUIRE):
            pytest.fail(
                f"{why}. {REQUIRE} is set, so this is a failure rather than a skip: "
                f"an engine that cannot be asked has not been checked, and a green "
                f"tick over an unasked question is what this job exists to prevent"
            )
        pytest.skip(f"{why} (set {REQUIRE}=1 to make this a failure)")

    checked = _for(engine)
    assert checked, f"no case in this file plans onto {engine}; it is going unchecked"

    bad = {label: rejected for label, argv in checked if (rejected := unknown_flags(adapter, argv))}
    assert not bad, (
        f"the installed {engine} rejects flags this repo generates:\n"
        + "\n".join(f"  {label}: {', '.join(flags)}" for label, flags in bad.items())
        + f"\n{engine}'s own --help is the authority; the dialect in engines.py has drifted"
    )


def test_the_cases_cover_the_flags_worth_covering():
    """A case list that stopped exercising a flag would pass while proving less.

    Runs everywhere, including on the Mac where neither engine installs, so a
    change that quietly narrows the coverage is caught before CI is reached.
    """
    emitted = {a for _, _, argv in cases() for a in argv if a.startswith("--")}
    for flag in (
        "--max-model-len",  # vllm
        "--context-length",  # sglang: same intent, different name
        "--enable-chunked-prefill",
        "--chunked-prefill-size",
        "--tensor-parallel-size",
        "--structured-outputs-config",  # the flag that replaced --guided-decoding-backend
        "--speculative-config",  # the JSON one that shipped broken
        "--quantization",
        "--lora-modules",  # vllm dialect
        "--lora-paths",  # sglang dialect
        "--max-loras",
        "--max-loras-per-batch",
        "--kv-cache-dtype",
        "--host",
    ):
        assert flag in emitted, f"no case emits {flag}; it is validated by nothing"

    engines = {eng for _, eng, _ in cases()}
    assert engines == {"vllm", "sglang"}, engines


if __name__ == "__main__":
    # `python3 tests/test_engine_flags.py` prints what CI is about to validate.
    # The list belongs in the job's log: a check whose subject is invisible is
    # one nobody can tell has narrowed.
    for label, engine, argv in cases():
        print(f"[{engine}] {label}\n  {' '.join(argv)}\n")
