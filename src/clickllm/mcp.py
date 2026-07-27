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
    # Write-side tools must never appear: a human presses the button that moves traffic.
    assert not any("cutover" in t["name"] or "apply" in t["name"] for t in listed)

    called = handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "clickllm_fit", "arguments": {"context": "8k"}},
        }
    )["result"]
    assert "hardware" in called["structuredContent"]

    assert handle({"jsonrpc": "2.0", "id": 4, "method": "nope"})["error"]["code"] == -32601
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    print(f"mcp: {len(TOOLS)} tools, all read-only")
    print("ok")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        sys.exit(serve())
