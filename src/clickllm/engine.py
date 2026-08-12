"""The one place a product question is answered. Surfaces render; they do not decide.

ADR-0016. Before this module, four surfaces each answered "what runs on this
machine" by calling `fit.rank()` and assembling the result themselves — the CLI,
the MCP server, the SDK and the browser workbench. They agreed on the answer and
produced four different shapes of it, agreeing on only four field names out of
thirteen.

That is not a tidiness problem. `clickllm fit --json` was the one that dropped
`estimate_basis`, so a program reading the documented machine-readable surface
got `"tokens_per_sec": 15` with no way to know it was a roofline projection —
while every other surface said so. **While each surface assembles its own
result, a disclosure is per-surface, and one of them will eventually omit it.**

So the domain model is computed once, here, and returned typed. What a surface
may still choose is its *wire format*: the CLI's JSON has been stable since
1.0.0 and MCP's has its own spelling, and converging those is a versioned change
with a deprecation note rather than something to slip into a refactor. The
difference this module makes is that those are now two renderings of one
computation instead of two computations that happen to agree.

What belongs here: deciding, validating, computing.
What belongs in a surface: formatting, transport, authentication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time cost for a type only
    from .hardware import Hardware
    from .sdk import FitReport

__all__ = ["CONTRACT", "fit"]

#: The engine's result contract, versioned like the artifacts already are
#: (`clickllm.receipt/v2`, `clickllm.box/v1`). A surface may render these
#: results however it likes; what this names is the *computation* behind them,
#: so that a change to what the engine answers is visible rather than silent.
CONTRACT = "clickllm.engine/v1"


def fit(
    context: str | int = "32k",
    concurrency: int = 1,
    hw: Hardware | None = None,
) -> FitReport:
    """What runs on this machine, at this context and concurrency.

    The single computation behind `clickllm fit`, `clickllm_fit`, `sdk.fit` and
    the workbench's `/api/fit`. Returns the typed report; rendering it is the
    caller's business.

    `hw` is injectable so a caller can ask about a machine it is not running on
    — and so a test can pin one. Detection happens here rather than in each
    surface, which is what let the goldens encode a developer's laptop.

    Validation is deliberately *not* here: `sdk.fit` already refuses a
    concurrency below 1 and a context below 1, and the solver refuses again
    beneath it. ADR-0011 — the constraint belongs to the thing it protects, not
    to whichever surface happened to reach it.
    """
    from .sdk import fit as _fit

    return _fit(context=context, concurrency=concurrency, hw=hw)


def demo() -> None:
    """Self-check: one computation, and the surfaces that render it agree."""
    from .hardware import Hardware

    gb = 1024**3
    machine = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * gb,
        usable_bytes=96 * gb,
        bandwidth_gbps=546.0,
        cores=16,
    )

    report = fit(context="32k", concurrency=8, hw=machine)
    assert report.feasible, "nothing fits on a 96 GB machine"
    assert report.context == 32768, report.context
    assert report.concurrency == 8
    assert report.runtime, "no runtime recommended"

    # The injected machine is the machine reported on — the property that lets a
    # test pin hardware instead of encoding the host it happened to run on.
    assert report.hardware.name == "M4 Max"

    # Two calls, same answer. Not a tautology: it fails if the engine reads any
    # ambient state between calls, which is how a "pure" function acquires a
    # dependency on the machine's mood.
    again = fit(context="32k", concurrency=8, hw=machine)
    assert [c.model_id for c in again.feasible] == [c.model_id for c in report.feasible]

    # Validation lives beneath, per ADR-0011, and reaches this entry point.
    for bad in ({"concurrency": 0}, {"context": "0"}):
        try:
            fit(hw=machine, **bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{bad} should have been refused by the solver")

    assert CONTRACT.startswith("clickllm.engine/"), CONTRACT
    print("engine: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
