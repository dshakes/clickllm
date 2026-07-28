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
import sys
from collections.abc import Callable
from typing import Any, BinaryIO

from . import catalog, fit, hardware

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "clickllm", "version": "0.1.0"}

GB = 1024**3


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def _fit(context: str = "32k", concurrency: int = 1) -> dict[str, Any]:
    """What runs on this machine, at this context and concurrency."""
    from .cli import _parse_size

    hw = hardware.detect()
    ctx = _parse_size(context)
    feasible, rejected = fit.rank(hw, ctx, concurrency)
    name, why = fit.recommend_runtime(hw, ctx, concurrency)
    return {
        "hardware": hw.to_dict(),
        "context": ctx,
        "concurrency": concurrency,
        "feasible": [
            {
                "id": f.model.id,
                "quant": f.quant,
                "total_gb": round(f.total_bytes / GB, 1),
                "headroom_gb": round(f.headroom_bytes / GB, 1),
                "tokens_per_sec_estimate": (round(f.tokens_per_sec) if f.tokens_per_sec else None),
                "estimate_basis": "memory-bandwidth roofline, not measured",
                "license": f.model.license,
                "license_clean_commercial": f.model.license_ok,
                "architecture_verified": f.model.verified,
            }
            for f in feasible
        ],
        "rejected": [{"id": m.id, "reason": why} for m, why in rejected],
        "recommended_runtime": {"name": name, "why": why},
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
    import pathlib
    from datetime import date

    from .prove import EvalItem, suite

    raw = _json.loads(pathlib.Path(eval_set).read_text())
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
                            "confidence interval clears this."
                        ),
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": ["eval_set"],
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
        response = handle(request)
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
