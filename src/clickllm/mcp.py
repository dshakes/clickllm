"""MCP server — the loop as tools, so Claude Code and Cursor can drive it.

Deliberately **read/plan-heavy, write-light**. `cutover_advance` and
`deploy_apply` are *not* exposed. An agent should be able to analyse a workload
and recommend a migration; a human presses the button that moves production
traffic. That is a trust boundary, not friction.

Speaks MCP over stdio using JSON-RPC 2.0 framed by Content-Length headers, with
no third-party dependency — the CLI must stay installable with nothing but the
standard library.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from collections.abc import Callable
from typing import Any, BinaryIO

from . import __version__, catalog, fit, hardware

PROTOCOL_VERSION = "2025-06-18"
# Derived, never typed. This was the literal "0.1.0" through four releases, so
# every receipt written by 0.1.1 through 0.1.4 claims it was produced by 0.1.0 —
# on the one artifact whose whole purpose is provenance, and which `clickllm
# receipt --against` exists to audit. `__version__` already resolves from
# installed metadata with a pyproject fallback; there was never a second source
# of truth, only a copy that stopped tracking it.
SERVER_INFO = {"name": "clickllm", "version": __version__}

GB = 1024**3


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _fit(context: str = "32k", concurrency: int = 1) -> dict[str, Any]:
    """What runs on this machine, at this context and concurrency."""
    from . import engine

    # One computation, rendered. This used to call `fit.rank()` and assemble a
    # dict itself, which is how four surfaces came to spell the same answer four
    # ways — and how `clickllm fit --json` came to omit the roofline disclosure
    # that this one carried. ADR-0016.
    #
    # The *wire spelling* below is unchanged on purpose: `id` and
    # `recommended_runtime` have been what agents receive since 1.0.0, and
    # renaming them is a versioned change with a deprecation note, not a
    # side effect of a refactor. What changed is that this is now a rendering
    # of the engine's result rather than a second computation of it.
    report = engine.fit(context=context, concurrency=concurrency)
    return {
        "hardware": report.hardware.to_dict(),
        "context": report.context,
        "concurrency": report.concurrency,
        "feasible": [
            {
                "id": c.model_id,
                "quant": c.quant,
                "total_gb": round(c.total_gb, 1),
                "headroom_gb": round(c.headroom_gb, 1),
                # Rounded here, not in the model: this is what agents have
                # received since 1.0.0, and migrating onto a pre-rounded
                # `Candidate` silently turned 15 into 14.5.
                "tokens_per_sec_estimate": (
                    round(c.tokens_per_sec_estimate) if c.tokens_per_sec_estimate else None
                ),
                "estimate_basis": c.estimate_basis,
                "license": c.license,
                "license_clean_commercial": c.license_clean_commercial,
                "architecture_verified": c.architecture_verified,
            }
            for c in report.feasible
        ],
        "rejected": [{"id": r.model_id, "reason": r.reason} for r in report.rejected],
        "recommended_runtime": {"name": report.runtime, "why": report.runtime_reason},
    }


def _explain(model_id: str, context: str = "32k", concurrency: int = 1) -> dict[str, Any]:
    """Show the arithmetic behind one model's verdict."""
    from .cli import _parse_size

    hw = hardware.detect()
    ctx = _parse_size(context)
    m = catalog.get(model_id)
    f = fit.best_quant(m, hw, ctx, concurrency) or fit.solve(
        m, min(m.quants, key=lambda q: catalog.QUANT_BITS[q]), hw, ctx, concurrency
    )
    return {"model": model_id, "fits": f.feasible, "arithmetic": f.explain()}


def _build(
    description: str = "",
    machine: str = "",
    state: dict[str, Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Drive the whole flow one turn at a time, carrying state between calls.

    This is the multi-turn surface: pass `state` back from the previous call and
    the conversation continues. Read-only — it produces a command and a proof
    plan, and runs neither.
    """
    import json as _json

    from .session import Session

    s = Session.from_json(_json.dumps(state)) if state else Session()
    # Mutate state directly rather than calling tell()/on()/set() (which each
    # call step() internally) — a worth-asking question gets committed to
    # `s.asked` inside step(), and if that commit fires on an intermediate
    # call whose Turn is then discarded, the single most valuable question is
    # silently lost before it ever reaches whoever is having this conversation.
    # One step() call at the end means exactly one commit, in priority order.
    if description:
        s._apply_text(description)
    if machine or s.hw is None:
        s._apply_hardware(machine or None)
    known = {k: v for k, v in overrides.items() if v is not None}
    if known:
        s._apply_fields(**known)

    turn = s.step()
    return {
        "stage": turn.stage.value,
        "said": turn.said,
        # One question, never a list. Ask it, then call back with `state`.
        "question": turn.question,
        "evidence": list(turn.evidence),
        "assuming": list(turn.assuming),
        "answer": s.answer(),
        "suggestions": [
            {"impact": x.impact.value, "action": x.action, "because": x.because}
            for x in s.optimizations()
        ],
        # Pass this back verbatim on the next call to continue the conversation.
        "state": _json.loads(s.to_json()),
        "advisory": (
            "A plan and a command, not an action. Deployment is a human step and "
            "no eval result moves production traffic without one."
        ),
    }


def _advise(
    context: str = "32k",
    concurrency: int = 1,
    workload: str = "interactive",
    ttft_ms: int | None = None,
    itl_ms: int | None = None,
    prefix_sharing: float = 0.0,
    structured_output: bool = False,
    observed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Proactive suggestions for a deployment, and drift against observed reality.

    Read-only: it proposes, with the evidence for each proposal, and applies
    nothing.
    """
    from .advise import Observed, reconcile, suggest
    from .cli import _parse_size
    from .plan import Requirements, Workload, plan

    hw = hardware.detect()
    req = Requirements(
        workload=Workload(workload),
        concurrency=concurrency,
        context=_parse_size(context),
        ttft_ms=ttft_ms,
        itl_ms=itl_ms,
        prefix_sharing=prefix_sharing,
        structured_output=structured_output,
    )
    p = plan(hw, req)

    def _out(s: Any) -> dict[str, Any]:
        return {
            "id": s.id,
            "impact": s.impact.value,
            "action": s.action,
            "because": s.because,
            "expect": s.expect,
            "setting": s.setting.value if s.setting else None,
        }

    drift = reconcile(req, p, Observed(**observed)) if observed else []
    return {
        "engine": p.engine.value,
        "engine_why": p.engine_why,
        "suggestions": [_out(s) for s in suggest(req, p)],
        "drift": [_out(s) for s in drift],
        "cannot_meet": list(p.warnings),
        "advisory": (
            "Proposals with evidence, not actions. Effects are estimates unless "
            "labelled otherwise; nothing here has been applied."
        ),
    }


def eval_root() -> pathlib.Path:
    """The directory `clickllm_prove` may read an eval set from.

    The working directory unless the operator says otherwise via
    `CLICKLLM_EVAL_ROOT`. An environment variable rather than a tool argument or
    a CLI flag on purpose: a flag is set by whoever composes the command, which
    for an MCP server started by an agent harness may be the agent. The variable
    is set when the server is launched, by the party the boundary protects.

    See ADR-0014.
    """
    return pathlib.Path(os.environ.get("CLICKLLM_EVAL_ROOT", ".")).resolve()


def _within_eval_root(candidate: str) -> pathlib.Path:
    """Resolve `candidate` and refuse it if it leaves the eval root.

    This is the one MCP tool where an agent names a filesystem path and the
    contents come back into its context — a file-read primitive addressable by
    whatever is steering that agent, which invariant 7 says may itself have come
    out of a customer's request log.

    Resolved *before* the comparison, so a symlink cannot walk out of the root by
    pointing at something outside it.

    Raises:
        ValueError: naming both the path and the root, because a refusal nobody
            can act on is only marginally better than the read.
    """
    root = eval_root()
    path = pathlib.Path(candidate).expanduser()
    resolved = (root / path if not path.is_absolute() else path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(
            f"{candidate} is outside the eval root ({root}). Set "
            f"CLICKLLM_EVAL_ROOT to the directory holding your eval sets, or "
            f"pass a path inside it. The CLI is unrestricted — this applies to "
            f"paths an agent names, not paths you type."
        )
    return resolved


def _prove(
    eval_set: str,
    candidate: str = "candidate",
    incumbent: str = "incumbent",
    issued: str = "",
    bar: float = 0.90,
) -> dict[str, Any]:
    """Score a candidate over an eval set and return the verdict and receipt.

    Read-only by construction: it returns a *proposal* and touches nothing. The
    tool that would act on it does not exist — see the boundary assertion in
    :func:`demo`.
    """
    import json as _json
    from datetime import date

    from .prove import EvalItem, suite

    raw = _json.loads(_within_eval_root(eval_set).read_text())
    rows = raw.get("items", []) if isinstance(raw, dict) else raw
    shares = raw.get("shares", {}) if isinstance(raw, dict) else {}
    names = raw.get("names", {}) if isinstance(raw, dict) else {}
    if not rows:
        raise ValueError(f"{eval_set} contains no eval items")

    items = [
        EvalItem(
            item_id=str(r.get("item_id", i)),
            cluster=str(r.get("cluster", "all")),
            prompt=str(r.get("prompt", "")),
            baseline=str(r.get("baseline", "")),
            candidate=str(r.get("candidate", "")),
            baseline_tool_calls=tuple(r.get("baseline_tool_calls", ()) or ()),
            candidate_tool_calls=tuple(r.get("candidate_tool_calls", ()) or ()),
            response_format=r.get("response_format"),
        )
        for i, r in enumerate(rows)
    ]
    equal_weighted = not shares
    if equal_weighted:
        keys = sorted({i.cluster for i in items})
        shares = {k: 1 / len(keys) for k in keys}

    result = suite(
        items,
        shares=shares,
        names=names,
        issued=issued or date.today().isoformat(),
        candidate=candidate,
        incumbent=incumbent,
        bar=bar,
        tool_version=SERVER_INFO["version"],
    )
    # to_json() nests the document under {digest, receipt}. Flattened here so an
    # agent reads `receipt.regret` rather than `receipt.receipt.regret` — a shape
    # that invites exactly one silent KeyError per integration.
    doc = _json.loads(result.receipt.to_json())
    return {
        "report": result.render(),
        "movable_share": result.policy.moved_share,
        "regret_clusters": list(result.policy.regret_clusters),
        "unproven_clusters": list(result.policy.unproven_clusters),
        # Stated so an agent cannot present an equal-weighted verdict as a
        # traffic-weighted one — a different and much stronger claim.
        "traffic_weighted": not equal_weighted,
        "judge_used": result.receipt.judge_model is not None,
        "receipt": doc["receipt"],
        "receipt_digest": doc["digest"],
        "advisory": (
            "This is a proposal, not an action. Moving production traffic is a "
            "human decision and no tool here performs it."
        ),
    }


def _where(model: str, context: str = "32k", concurrency: int = 1) -> dict[str, Any]:
    """The inverse of `fit`: what would it take to run this model?

    Read-only and hardware-independent — it answers about machines the caller
    does not have, which is the question an agent asks the moment `fit` says
    "nothing here runs it". Without this the agent's only move is to guess a
    box, and a guessed box is how people buy the wrong GPU.
    """
    from .cli import _parse_size

    spec = catalog.get(model)
    ctx = _parse_size(context)
    placements = fit.where(spec, ctx, concurrency)
    return {
        "model": spec.id,
        "context": ctx,
        "concurrency": concurrency,
        # Invariant 6. Every throughput number below is arithmetic, not a
        # measurement, and an agent quoting it to a human must be able to say so.
        "estimates_are_roofline_not_measured": True,
        "placements": [
            {
                "profile": pl.profile_id,
                "name": pl.profile_name,
                "fits": pl.fit is not None,
                "quant": pl.fit.quant if pl.fit else None,
                "total_gb": round(pl.fit.total_bytes / GB, 1) if pl.fit else None,
                "tokens_per_sec_estimate": (
                    round(pl.fit.tokens_per_sec) if pl.fit and pl.fit.tokens_per_sec else None
                ),
                "hourly_usd": pl.hourly_usd,
                "reason": pl.reason,
            }
            for pl in placements
        ],
    }


def _receipt(path: str) -> dict[str, Any]:
    """Read a receipt and return what it claims, refusing one that does not hold together.

    Confined to the eval root for the same reason `clickllm_prove` is: the path
    comes from the caller, the contents land in the agent's context, and the
    caller may itself have been steered by captured traffic (invariant 7).
    See ADR-0014.
    """
    from .prove import Receipt

    r = Receipt.from_json(_within_eval_root(path).read_text())
    return {
        "incumbent": r.incumbent,
        "candidate": r.candidate,
        "issued": r.issued,
        "bar": r.bar,
        "eval_set_digest": r.eval_set,
        # Regret first, then unproven, then proven — the same order the rendered
        # receipt uses, and for the same reason: an agent summarising this to a
        # human should hit the bad news before it has a chance to lead with a
        # headline number.
        "keep_on_incumbent": [c.cluster for c in r.regret],
        "not_proven_either_way": [c.cluster for c in r.unproven],
        "proven_above_bar": [c.cluster for c in r.proven],
        "judge_model": r.judge_model,
        "judge_human_agreement": r.judge_agreement,
        "rendered": r.render(),
    }


def _guard(path: str, today: str = "", available: list[str] | None = None) -> dict[str, Any]:
    """Whether a receipt still holds, and if not, which of three ways it stopped.

    The distinction every other tool collapses into one "stale" flag: the model
    changed behind its name (the proof is void), the traffic moved (the eval set
    answers questions nobody asks now), or something new was released (the proof
    is still true). Only the first two mean you no longer know whether
    production is adequate.
    """
    from datetime import date

    from . import guard as guard_mod
    from .prove import Receipt

    r = Receipt.from_json(_within_eval_root(path).read_text())
    when = date.fromisoformat(today) if today else date.today()
    proposal = guard_mod.check(r, today=when, available=frozenset(available or ()))
    return {
        "receipt_digest": proposal.receipt_digest,
        "still_holds": proposal.valid,
        "action": proposal.action,
        "findings": [
            {
                "kind": f.kind.value,
                "subject": f.subject,
                "detail": f.detail,
                "voids_the_receipt": f.invalidates,
            }
            for f in proposal.findings
        ],
        "rendered": proposal.render(),
    }


def _catalog() -> dict[str, Any]:
    """The model catalogue with licences and architecture-verification flags."""
    return {
        "models": [
            {
                "id": m.id,
                "name": m.name,
                "params_b": m.params_b,
                "active_b": m.active_b if m.is_moe else None,
                "is_moe": m.is_moe,
                "max_context": m.max_context,
                "license": m.license,
                "license_clean_commercial": m.license_ok,
                "architecture_verified": m.verified,
            }
            for m in catalog.load()
        ]
    }


TOOLS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "clickllm_fit": (
        _fit,
        {
            "description": (
                "Which open models can run on this machine, at a given context length "
                "and concurrency. Throughput figures are memory-bandwidth roofline "
                "ESTIMATES, not measurements — say so when reporting them."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "Context length, e.g. '8k', '32k', '128000'.",
                    },
                    "concurrency": {
                        "type": "integer",
                        "description": "Simultaneous requests to plan for.",
                        "minimum": 1,
                    },
                },
            },
        },
    ),
    "clickllm_explain": (
        _explain,
        {
            "description": (
                "The full arithmetic behind one model's fit verdict: weights, KV cache, "
                "overhead, and the throughput roofline. Use this whenever a user asks "
                "why a model does or does not fit."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "context": {"type": "string"},
                    "concurrency": {"type": "integer", "minimum": 1},
                },
                "required": ["model_id"],
            },
        },
    ),
    "clickllm_build": (
        _build,
        {
            "description": (
                "The whole flow in one multi-turn call: read what the user is building, "
                "size their hardware, choose a model, configure the engine, critique the "
                "plan, and hand back a command plus a proof plan. Pass 'state' back from "
                "the previous response to continue the conversation. If 'question' is "
                "non-null it is the ONE thing that would change the answer — ask the user "
                "that and call again. There is always a usable 'answer', so never make the "
                "user answer a question before showing them something. Report 'assuming' "
                "alongside it: those are defaults, not findings. Produces a command; runs "
                "nothing."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Plain language, e.g. 'coding assistant for 20 engineers'.",
                    },
                    "machine": {
                        "type": "string",
                        "description": "Hardware profile id; omit to detect the local machine.",
                    },
                    "state": {
                        "type": "object",
                        "description": "The 'state' object from the previous call.",
                    },
                    "concurrency": {"type": "integer", "minimum": 1},
                    "context": {"type": "integer"},
                    "ttft_ms": {"type": "integer"},
                    "itl_ms": {"type": "integer"},
                    # Accepted by `_apply_fields` and, until this PR, broken on
                    # arrival. Advertised now so a schema-driven agent can find
                    # it rather than learning it exists from an error message
                    # listing the known fields — which is how it was being
                    # discovered, and how the broken call was being made.
                    "workload": {"enum": ["interactive", "realtime", "batch"]},
                    "prefix_sharing": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    ),
    "clickllm_advise": (
        _advise,
        {
            "description": (
                "What a careful reviewer would raise about a deployment unprompted: the "
                "knob nobody set, the headroom nobody spent, the budget nobody stated. "
                "Pass 'observed' with real telemetry to also get drift — where production "
                "diverged from what the plan assumed. Every item carries the observation "
                "that triggered it; report that, not just the action. Proposals only: "
                "nothing here is applied."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context": {"type": "string", "description": "e.g. '8k', '32k'."},
                    "concurrency": {"type": "integer", "minimum": 1},
                    "workload": {"enum": ["interactive", "realtime", "batch"]},
                    "ttft_ms": {"type": "integer", "description": "Time-to-first-token budget."},
                    "itl_ms": {"type": "integer", "description": "Inter-token latency budget."},
                    "prefix_sharing": {"type": "number", "minimum": 0, "maximum": 1},
                    "structured_output": {"type": "boolean"},
                    "observed": {
                        "type": "object",
                        "description": (
                            "Measured reality: concurrency, prefix_sharing, ttft_ms, "
                            "itl_ms, peak_context, kv_utilisation. Any subset."
                        ),
                    },
                },
            },
        },
    ),
    "clickllm_prove": (
        _prove,
        {
            "description": (
                "Run the eval suite over an eval set and return the equivalence verdict, "
                "the traffic split it supports, and a reproducible receipt. Report the "
                "regret clusters and the confidence intervals, never the point estimate "
                "alone. If 'traffic_weighted' is false the clusters were weighted equally "
                "and the verdict is weaker than a traffic-weighted one. This returns a "
                "proposal only — moving production traffic is a human decision."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "eval_set": {
                        "type": "string",
                        "description": "Path to an eval-set JSON file.",
                    },
                    "candidate": {"type": "string", "description": "Open model under test."},
                    "incumbent": {"type": "string", "description": "Model being replaced."},
                    "issued": {"type": "string", "description": "ISO date for the receipt."},
                    "bar": {
                        "type": "number",
                        "description": (
                            "Equivalence bar. A cluster moves only when its whole "
                            "confidence interval clears this. Strictly between 0 "
                            "and 1: a bar of 0 admits every candidate and a bar of "
                            "1 admits none."
                        ),
                        # Exclusive, matching the runtime check. Inclusive bounds
                        # here would advertise 0 and 1 as valid and then refuse
                        # them — a schema-valid call that fails is worse than one
                        # the client could have rejected itself.
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 1,
                    },
                },
                "required": ["eval_set"],
            },
        },
    ),
    "clickllm_where": (
        _where,
        {
            "description": (
                "The inverse of fit: which hardware would run this model, at what "
                "quantisation, and roughly how fast. Answers about machines the "
                "caller does not have — ask this when clickllm_fit found nothing "
                "local. Throughput figures are roofline arithmetic, not measured, "
                "and the reply says so."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": {"type": "string", "description": "catalogue model id"},
                    "context": {"type": "string", "description": "e.g. 32k, 128000"},
                    "concurrency": {"type": "integer", "minimum": 1},
                },
                "required": ["model"],
            },
        },
    ),
    "clickllm_receipt": (
        _receipt,
        {
            "description": (
                "Read a migration receipt: what is proven above the bar, what must "
                "stay on the incumbent, and what is not proven either way. Refuses a "
                "receipt whose fields contradict its own counts. Paths are confined "
                "to the eval root (ADR-0014)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "receipt JSON, inside the eval root"}
                },
                "required": ["path"],
            },
        },
    ),
    "clickllm_guard": (
        _guard,
        {
            "description": (
                "Does a receipt still hold? Separates three things: the model changed "
                "behind its name (proof void), traffic moved (eval set is stale), or "
                "something new was released (proof still true). Only the first two "
                "mean you no longer know whether production is adequate."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "receipt JSON, inside the eval root"},
                    "today": {"type": "string", "description": "ISO date to evaluate against"},
                    "available": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "candidate models that exist now",
                    },
                },
                "required": ["path"],
            },
        },
    ),
    "clickllm_catalog": (
        _catalog,
        {
            "description": (
                "The model catalogue: parameters, MoE split, context, and licence. "
                "'license_clean_commercial' false means the licence carries caps or "
                "restrictions that need reading before production use."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
    ),
}


# --------------------------------------------------------------------------- #
# JSON-RPC / stdio transport
# --------------------------------------------------------------------------- #


def _read_message(stream: BinaryIO) -> dict[str, Any] | None:
    """Read one Content-Length framed JSON-RPC message. None at clean EOF."""
    length = None
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break  # end of headers
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1])
            except ValueError:
                return None
    if length is None:
        return None
    body = stream.read(length)
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _write_message(stream: BinaryIO, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode()
    stream.write(b"Content-Length: %d\r\n\r\n" % len(body))
    stream.write(body)
    stream.flush()


def handle(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle one JSON-RPC request. None for notifications, which get no reply."""
    method = request.get("method")
    req_id = request.get("id")

    if req_id is None:  # notification
        return None

    def ok(result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    def err(code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            }
        )

    if method == "tools/list":
        return ok(
            {"tools": [{"name": name, **schema} for name, (_, schema) in sorted(TOOLS.items())]}
        )

    if method == "tools/call":
        params = request.get("params") or {}
        name = params.get("name")
        entry = TOOLS.get(name)
        if entry is None:
            return err(-32602, f"unknown tool: {name}")
        func, _ = entry
        try:
            result = func(**(params.get("arguments") or {}))
        except KeyError as e:
            # An unknown model is a normal outcome, not a transport failure — report
            # it as tool content so the agent can correct itself.
            return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        except TypeError as e:
            return err(-32602, f"bad arguments for {name}: {e}")
        except Exception as e:  # noqa: BLE001 — see below; this is the boundary
            # A bad ARGUMENT must not kill the SESSION. Only KeyError and
            # TypeError were caught, and `serve()` has no handler at all, so any
            # other exception unwound the `while True` loop and took the process
            # with it — every queued and future call in that session included.
            #
            # Reachable without contrivance, because this server does no schema
            # validation: `clickllm_advise(workload="chat")` raises ValueError
            # from `Workload("chat")`, and `clickllm_fit(context="a few
            # thousand")` raises from `_parse_size`. An agent exploring the
            # tool surface — which is what agents do — takes the server down.
            #
            # Reported as tool content rather than a transport error: the agent
            # can read it, correct itself and continue, which is the whole point
            # of a surface described as safe to hand an agent.
            return ok(
                {
                    "content": [{"type": "text", "text": f"error: {type(e).__name__}: {e}"}],
                    "isError": True,
                }
            )
        return ok(
            {
                "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                "structuredContent": result,
            }
        )

    return err(-32601, f"method not found: {method}")


def serve(stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    """Run the stdio server until EOF."""
    rx = stdin if stdin is not None else sys.stdin.buffer
    tx = stdout if stdout is not None else sys.stdout.buffer
    while True:
        request = _read_message(rx)
        if request is None:
            return 0
        # Belt and braces: `handle` now contains tool failures, but a defect in
        # `handle` itself — or in serialising a response — must not end the
        # session either. Two layers, because this loop is the only thing
        # standing between one bad call and every later one.
        try:
            response = handle(request)
        except Exception as e:  # noqa: BLE001 — the loop must outlive one request
            response = {
                "jsonrpc": "2.0",
                "id": (request or {}).get("id"),
                "error": {"code": -32603, "message": f"internal error: {type(e).__name__}: {e}"},
            }
        if response is not None:
            _write_message(tx, response)


def demo() -> None:
    assert (
        handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})["result"]["protocolVersion"]
        == PROTOCOL_VERSION
    )

    listed = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert {t["name"] for t in listed} == set(TOOLS)
    # Write-side tools must never appear: a human presses the button that moves
    # traffic. The vocabulary is broad on purpose — the failure this guards is
    # someone adding a helpful-looking `clickllm_promote` and nothing objecting.
    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    leaked = [t["name"] for t in listed if any(w in t["name"] for w in forbidden)]
    assert not leaked, f"write-side tools exposed to agents: {leaked}"

    called = handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "clickllm_fit", "arguments": {"context": "8k"}},
        }
    )["result"]
    assert "hardware" in called["structuredContent"]

    # The eval suite over a real file, through the real transport — an agent's
    # path to a verdict, exercised end to end rather than asserted about.
    import json as _json
    import tempfile

    items = [
        {"item_id": f"c{i}", "cluster": "codegen", "prompt": f"p{i}"}
        | {"baseline": '{"a": 1}', "candidate": '{"a": 1}'}
        for i in range(45)
    ] + [
        {"item_id": f"r{i}", "cluster": "rare-json", "prompt": f"q{i}"}
        | {"baseline": '{"a": 1}', "candidate": '{"b": 1}'}
        for i in range(15)
    ]
    with tempfile.TemporaryDirectory() as d:
        # Declare the root, as an operator whose eval sets live on a mounted
        # volume would. ADR-0014 confined this tool to one; a demo that pointed
        # it somewhere arbitrary and passed would mean the guard was not on the
        # path the agent takes.
        os.environ["CLICKLLM_EVAL_ROOT"] = d
        p = f"{d}/evalset.json"
        with open(p, "w") as fh:
            _json.dump({"items": items, "shares": {"codegen": 0.75, "rare-json": 0.25}}, fh)
        proved = handle(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "clickllm_prove", "arguments": {"eval_set": p}},
            }
        )["result"]["structuredContent"]
        os.environ.pop("CLICKLLM_EVAL_ROOT", None)

    assert proved["movable_share"] == 0.75, proved["movable_share"]
    assert proved["regret_clusters"] == ["rare-json"], proved["regret_clusters"]
    assert proved["traffic_weighted"] is True
    assert not proved["judge_used"], "no judge was supplied; it must not claim one"
    # The regret cluster must survive into the receipt, not just the summary.
    assert any(c["name"] == "rare-json" for c in proved["receipt"]["regret"])

    assert handle({"jsonrpc": "2.0", "id": 4, "method": "nope"})["error"]["code"] == -32601
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    print(f"mcp: {len(TOOLS)} tools, all read-only")
    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(serve())
