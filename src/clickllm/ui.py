"""The local workbench — `clickllm ui`.

A browser workbench, not a desktop app. That choice is deliberate: the people who
most need this are deploying models on a remote GPU box over SSH, and a Mac app
that can only see localhost is useless to them. This ships inside the binary they
already run, works through a port-forward, and needs no install, no updates and
no code signing.

**Every panel is backed by real computation.** Nothing here renders a number the
CLI could not also print. Panels whose backend does not exist yet are absent
rather than mocked — a dashboard of plausible fictions is worse than no
dashboard, because it looks like evidence.

Stdlib only: `clickllm` must stay installable with nothing but Python.
"""

from __future__ import annotations

import contextlib
import json
import webbrowser
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import catalog, fit, hardware, sdk

GB = 1024**3
ASSET = Path(__file__).parent / "workbench.html"


def _hardware() -> dict:
    hw = hardware.detect()
    return {
        **hw.to_dict(),
        "bandwidth_gbps": hw.bandwidth_gbps,
        "roofline_note": "throughput figures are memory-bandwidth estimates, not measurements",
    }


def _fit(ctx: str, conc: int) -> dict:
    from . import engine

    # Through the engine, like every other surface — ADR-0016. This was
    # already one computation (it called the SDK) rather than a second
    # one; entering by the same door is what keeps it that way.
    report = engine.fit(context=ctx, concurrency=conc)
    return report.to_dict()


def _where(model_id: str, ctx: str, conc: int) -> dict:
    m = catalog.get(model_id)
    places = fit.where(m, sdk.parse_size(ctx), conc)
    return {
        "model": m.id,
        "name": m.name,
        "params_b": m.params_b,
        "active_b": m.active_b if m.is_moe else None,
        "license": m.license,
        "license_ok": m.license_ok,
        "verified": m.verified,
        "placements": [
            {
                "id": p.profile_id,
                "name": p.profile_name,
                "feasible": p.feasible,
                "quant": p.fit.quant if p.fit else None,
                "total_gb": round(p.fit.total_bytes / GB, 1) if p.fit else None,
                "tokens_per_sec": round(p.tokens_per_sec) if p.tokens_per_sec else None,
                # $/Mtok divides by this, not by the single-stream figure above.
                # A cost claim whose denominator is invisible is a verdict
                # without its basis.
                "aggregate_tokens_per_sec": (
                    round(p.fit.aggregate_tokens_per_sec)
                    if p.fit and p.fit.aggregate_tokens_per_sec
                    else None
                ),
                "hourly_usd": p.hourly_usd,
                "usd_per_mtok": round(p.cost_per_mtok_usd, 2) if p.cost_per_mtok_usd else None,
                # And how optimistic both are for an MoE at this batch: the
                # weight read is held at the active-parameter count, which is
                # exact only at batch 1. Null for dense models and at batch 1.
                "moe_batch_optimism": (
                    round(p.moe_batch_optimism, 1) if p.moe_batch_optimism else None
                ),
                "reason": p.reason or None,
            }
            for p in places
        ],
    }


def _explain(model_id: str, ctx: str, conc: int) -> dict:
    return {"model": model_id, "arithmetic": sdk.explain(model_id, context=ctx, concurrency=conc)}


def _catalog() -> dict:
    return {
        "models": [
            {
                **{k: getattr(m, k) for k in ("id", "name", "params_b", "max_context", "license")},
                "active_b": m.active_b if m.is_moe else None,
                "is_moe": m.is_moe,
                "license_ok": m.license_ok,
                "verified": m.verified,
                "repo": m.repo,
            }
            for m in catalog.load()
        ]
    }


def _build(say: str, state: str = "", machine: str = "") -> dict:
    """One turn of the build conversation, without the server remembering anything.

    `session.py` is a complete multi-turn engine — one question at a time by
    construction, and a computable relevance filter that suppresses any question
    whose answer would not change the plan. The workbench reimplemented
    "understand the user" with six client-side regexes and never imported it, so
    the browser was the one surface that could not ask a follow-up.

    **The session lives in the browser, not here.** `state` arrives as the JSON
    the last turn handed back and leaves the same way. That keeps this route a
    GET with no side effect, which is what lets the workbench stay read-only
    (`demo()` asserts there are no write routes) — and it means two tabs, a
    reload, or ten users are ten independent conversations with no server-side
    store to grow, expire or leak between them.
    """
    from .session import Session

    session = Session.from_json(state) if state else Session()
    # The machine is passed, not parsed out of the sentence. `Session.turn`
    # takes it as its own argument for a reason — "on an M4 Max" in free text
    # would need a hardware-name parser that does not exist, and inventing one
    # here would put a second, worse copy of the catalogue in the surface layer.
    # `detect` is honest on this route specifically: the workbench is bound to
    # localhost, so the browser and the hardware are the same machine.
    turn = session.turn(
        say,
        machine=None if machine in ("", "detect") else machine,
        detect_hardware=machine == "detect",
    )
    return {
        "said": turn.said,
        "question": turn.question,
        "evidence": list(turn.evidence),
        "assuming": list(turn.assuming),
        "stage": str(turn.stage),
        "done": turn.done,
        # Handed straight back to the next call. Opaque to the page on purpose:
        # the browser carries it, it does not interpret it.
        "state": session.to_json(),
    }


#: Read-only, all of it. The workbench shows and explains; it does not deploy,
#: promote or move traffic. Same boundary the MCP server enforces.
ROUTES = {
    "/api/hardware": lambda q: _hardware(),
    "/api/catalog": lambda q: _catalog(),
    "/api/fit": lambda q: _fit(q.get("context", ["32k"])[0], int(q.get("concurrency", ["1"])[0])),
    "/api/where": lambda q: _where(
        q["model"][0], q.get("context", ["32k"])[0], int(q.get("concurrency", ["1"])[0])
    ),
    # A conversation turn is still a GET: it computes an answer from what the
    # caller sent and stores nothing. See `_build`.
    "/api/build": lambda q: _build(
        q.get("say", [""])[0], q.get("state", [""])[0], q.get("machine", [""])[0]
    ),
    "/api/explain": lambda q: _explain(
        q["model"][0], q.get("context", ["32k"])[0], int(q.get("concurrency", ["1"])[0])
    ),
}


class Handler(BaseHTTPRequestHandler):
    """Serves the workbench and its read-only API."""

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002
        return  # the workbench is not a web server; its access log is noise

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # Local-only tool: no caching, and nothing to share cross-origin.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, ASSET.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/healthz":
            self._send(200, b"ok", "text/plain")
            return

        handler = ROUTES.get(path)
        if handler is None:
            self._send(404, b'{"error":"not found"}', "application/json")
            return

        try:
            payload = handler(parse_qs(parsed.query))
        except KeyError as e:
            # An unknown model is a normal outcome; report it as data.
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            return
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            return
        except Exception as e:  # noqa: BLE001
            # Anything else is the server's problem, not the request's — a
            # missing $CLICKLLM_CATALOG file raises FileNotFoundError on every
            # catalogue route for the life of the process. Uncaught, the socket
            # closes with no response at all and the panel shows a bare network
            # error. Report it as JSON like everything else.
            self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")
            return
        self._send(200, json.dumps(payload, default=_json_default).encode(), "application/json")


def _json_default(o: object):
    try:
        return asdict(o)  # type: ignore[arg-type]
    except TypeError:
        return str(o)


def serve(host: str = "127.0.0.1", port: int = 7171, open_browser: bool = True) -> int:
    """Run the workbench until interrupted.

    Binds loopback by default. This surfaces hardware detail and model
    configuration; it is not something to expose on an interface by accident.
    """
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"\n  clickllm workbench → {url}")
    print("  read-only · local · no telemetry · ^C to stop\n")
    if open_browser:
        # Headless is a normal state, not an error worth reporting.
        with contextlib.suppress(Exception):
            webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped\n")
    finally:
        server.server_close()
    return 0


def demo() -> None:
    assert ASSET.exists(), "workbench.html must ship with the package"

    hw = _hardware()
    assert "usable_gb" in hw and "not measurements" in hw["roofline_note"]

    f = _fit("8k", 2)
    assert {"hardware", "feasible", "rejected", "runtime"} <= f.keys()

    w = _where("qwen3-32b", "8k", 1)
    assert w["placements"] and any(p["feasible"] for p in w["placements"])
    assert all(p["reason"] for p in w["placements"] if not p["feasible"])

    # The cost figure's denominator must travel with it.
    cheap = [p for p in w["placements"] if p["usd_per_mtok"]]
    assert cheap and all(p["aggregate_tokens_per_sec"] for p in cheap)

    assert "weights" in _explain("qwen3-32b", "8k", 1)["arithmetic"]

    # Not `len(...) == len(...)`: that comprehension emits one row per model by
    # construction, so it held however badly the fields were mapped.
    src = {m.id: m for m in catalog.load()}
    rows = {r["id"]: r for r in _catalog()["models"]}
    assert rows.keys() == src.keys(), "catalogue rows must not drop or invent a model"
    for mid, r in rows.items():
        m = src[mid]
        assert (r["params_b"], r["repo"], r["license"]) == (m.params_b, m.repo, m.license), mid
        # active_b is meaningful only for MoE; a dense model must not report one.
        assert r["active_b"] == (m.active_b if m.is_moe else None), mid

    # Every route must be JSON-serialisable — a workbench that 500s on render
    # is worse than one that is absent. Routes that take no model ignore the
    # argument, so pass it to all of them: the old spelling guessed which routes
    # needed it from their path, and half that guess ("model" in path) was never
    # true of any route.
    for fn in ROUTES.values():
        json.dumps(fn({"model": ["qwen3-32b"]}), default=_json_default)

    # No route hands the serialiser a raw object today, so its fallback is only
    # reachable from here.
    assert _json_default(hardware.detect())["usable_bytes"] == hardware.detect().usable_bytes
    assert _json_default(ASSET) == str(ASSET)

    # No route writes anything.
    for name in ("deploy", "apply", "promote", "cutover", "rollback", "pull"):
        assert not any(name in p for p in ROUTES), f"{name} must not be a workbench route"

    print(f"workbench: {len(ROUTES)} read-only routes")
    print("ok")


if __name__ == "__main__":
    demo()
