"""Python SDK — the programmatic entry point.

The CLI and the MCP server both go through this, so there is one implementation
of every behaviour rather than three that drift.

    >>> from clickllm import sdk
    >>> report = sdk.fit(context="32k", concurrency=8)
    >>> [m.model_id for m in report.feasible][:2]              # doctest: +SKIP
    ['qwen3-32b', 'qwen3-30b-a3b']
    >>> report.best().tokens_per_sec_estimate                  # doctest: +SKIP
    119.4

Every throughput figure is a **memory-bandwidth roofline estimate**, never a
measurement, and the type says so — see [`Candidate.estimate_basis`]. Reporting
an estimate as a measurement is the failure mode of every sizing tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from . import catalog, hardware
from . import fit as _fit
from .catalog import QUANT_BITS, ModelSpec
from .hardware import Hardware
from .prove import EvalItem, SuiteResult
from .prove import suite as _suite

GB = 1024**3

ESTIMATE_BASIS = "memory-bandwidth roofline, not measured"


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model that fits, with the numbers behind the verdict."""

    model_id: str
    name: str
    quant: str
    weights_gb: float
    kv_gb: float
    total_gb: float
    headroom_gb: float
    tokens_per_sec_estimate: float | None
    estimate_basis: str
    license: str
    license_clean_commercial: bool
    architecture_verified: bool
    slow: bool

    @property
    def usable_commercially_without_review(self) -> bool:
        """True only for permissive licences with confirmed architecture.

        Deliberately conservative: an unverified architecture means the KV figures
        carry error bars, and a non-permissive licence needs a human to read it.
        """
        return self.license_clean_commercial and self.architecture_verified


@dataclass(frozen=True, slots=True)
class Rejection:
    """A model that cannot run here, and why."""

    model_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class FitReport:
    """The result of a fit solve."""

    hardware: Hardware
    context: int
    concurrency: int
    feasible: list[Candidate]
    rejected: list[Rejection]
    runtime: str
    runtime_reason: str
    warnings: list[str] = field(default_factory=list)

    def best(self) -> Candidate | None:
        """Highest-capability candidate that is not flagged slow.

        Falls back to the top candidate when everything is slow — "slow but
        running" beats "nothing", provided the caller can see the flag.
        """
        quick = [c for c in self.feasible if not c.slow]
        pool = quick or self.feasible
        return pool[0] if pool else None

    def commercially_clean(self) -> list[Candidate]:
        """Candidates needing no licence review before production use."""
        return [c for c in self.feasible if c.usable_commercially_without_review]

    def to_dict(self) -> dict:
        """JSON-serialisable form, matching the CLI's ``--json`` output."""
        return {
            "hardware": self.hardware.to_dict(),
            "context": self.context,
            "concurrency": self.concurrency,
            "feasible": [{k: getattr(c, k) for k in Candidate.__slots__} for c in self.feasible],
            "rejected": [{"id": r.model_id, "reason": r.reason} for r in self.rejected],
            "runtime": {"name": self.runtime, "why": self.runtime_reason},
            "warnings": self.warnings,
        }


def parse_size(value: str | int) -> int:
    """``'32k'`` → ``32768``. Accepts an int unchanged."""
    if isinstance(value, int):
        return value
    s = value.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def detect_hardware() -> Hardware:
    """Profile this machine."""
    return hardware.detect()


def fit(
    context: str | int = "32k",
    concurrency: int = 1,
    hw: Hardware | None = None,
) -> FitReport:
    """Solve which catalogue models run here.

    Args:
        context: Context length to plan for. Use the p95 you actually observe,
            not a model's advertised maximum — sizing for unused context wastes
            KV memory.
        concurrency: Simultaneous requests.
        hw: Override detection, e.g. to plan for a machine you do not have.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    ctx = parse_size(context)
    if ctx < 1:
        raise ValueError(f"context must be >= 1, got {ctx}")

    target = hw or detect_hardware()
    feasible, rejected = _fit.rank(target, ctx, concurrency)
    name, why = _fit.recommend_runtime(target, ctx, concurrency)

    warnings: list[str] = []
    if any(not f.model.verified for f in feasible):
        warnings.append(
            "some candidates have unverified architecture; their KV figures are estimates"
        )
    if target.bandwidth_gbps is None:
        warnings.append("no bandwidth figure for this hardware; throughput is not estimated")

    return FitReport(
        hardware=target,
        context=ctx,
        concurrency=concurrency,
        feasible=[
            Candidate(
                model_id=f.model.id,
                name=f.model.name,
                quant=f.quant,
                weights_gb=round(f.weight_bytes / GB, 2),
                kv_gb=round(f.kv_bytes / GB, 2),
                total_gb=round(f.total_bytes / GB, 2),
                headroom_gb=round(f.headroom_bytes / GB, 2),
                tokens_per_sec_estimate=(round(f.tokens_per_sec, 1) if f.tokens_per_sec else None),
                estimate_basis=ESTIMATE_BASIS,
                license=f.model.license,
                license_clean_commercial=f.model.license_ok,
                architecture_verified=f.model.verified,
                slow=f.slow,
            )
            for f in feasible
        ],
        rejected=[Rejection(model_id=m.id, reason=r) for m, r in rejected],
        runtime=name,
        runtime_reason=why,
        warnings=warnings,
    )


def explain(
    model_id: str,
    context: str | int = "32k",
    concurrency: int = 1,
    hw: Hardware | None = None,
) -> str:
    """The arithmetic behind one model's verdict.

    Raises:
        KeyError: if ``model_id`` is not in the catalogue.
    """
    target = hw or detect_hardware()
    ctx = parse_size(context)
    m = catalog.get(model_id)
    f = _fit.best_quant(m, target, ctx, concurrency) or _fit.solve(
        m, min(m.quants, key=lambda q: QUANT_BITS[q]), target, ctx, concurrency
    )
    return f.explain()


def models(
    license_filter: Literal["all", "permissive"] = "all",
) -> list[ModelSpec]:
    """The catalogue, optionally restricted to cleanly-commercial licences."""
    all_models = list(catalog.load())
    if license_filter == "permissive":
        return [m for m in all_models if m.license_ok]
    return all_models


def prove(
    items: list[EvalItem],
    *,
    shares: dict[str, float],
    issued: str,
    **kw: object,
) -> SuiteResult:
    """Run the eval suite: score a candidate, propose a split, issue a receipt.

    A thin pass-through to [`clickllm.prove.suite`] so the SDK, the CLI, the MCP
    server and the agent skill all reach the same implementation. See that
    function for the full argument list.

    >>> from clickllm import sdk
    >>> result = sdk.prove(items, shares=shares, issued="2026-07-27")  # doctest: +SKIP
    >>> print(result.render())                                        # doctest: +SKIP

    It returns a verdict and never applies one — advancing traffic lives behind
    [`clickllm.prove.gate`] and ends at a human.
    """
    return _suite(items, shares=shares, issued=issued, **kw)  # type: ignore[arg-type]


def demo() -> None:
    r = fit(context="8k", concurrency=4)
    assert r.context == 8192
    assert r.concurrency == 4
    assert r.runtime and r.runtime_reason

    for c in r.feasible:
        assert c.estimate_basis == ESTIMATE_BASIS, "throughput must be labelled an estimate"
        assert c.total_gb > 0

    b = r.best()
    if b is not None:
        assert b in r.feasible

    assert set(r.commercially_clean()) <= set(r.feasible)
    assert all(m.license_ok for m in models("permissive"))
    assert len(models()) >= len(models("permissive"))

    assert "weights" in explain("qwen3-32b")

    for bad in [dict(concurrency=0), dict(context="0")]:
        try:
            fit(**bad)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{bad} should have been rejected")

    print(f"sdk: {len(r.feasible)} feasible, runtime={r.runtime}")
    print("ok")


if __name__ == "__main__":
    demo()
