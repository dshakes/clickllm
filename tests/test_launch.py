"""The launcher.

`launch.demo()` walks a whole resolution. These pin the four refusals, which are
the design: this is the only module that starts a process, so everything it
declines to do is a failure someone does not have to debug at 2am with 30 GB
already downloaded.

Every test here is offline. `exists` is injected rather than mocked, and the
resolution cache is written under `tmp_path` — a test suite that reaches the
network fails on a train, which is where this repo's users are.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from clickllm import catalog, launch
from clickllm.hardware import Hardware

GB = 1024**3

M4 = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * GB,
    usable_bytes=96 * GB,
    bandwidth_gbps=546.0,
    cores=16,
)
AIR = Hardware(
    kind="apple",
    name="MacBook Air",
    total_bytes=16 * GB,
    usable_bytes=12 * GB,
    bandwidth_gbps=120.0,
    cores=8,
)

#: Verified against the live index while writing this: the naive construction is
#: a 404 and the real repo carries a `Meta-` prefix the base repo never had.
NAIVE = "mlx-community/Llama-3.1-8B-Instruct-8bit"
REAL = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"


def only(*repos: str):
    """An existence check that says yes to exactly these."""
    return lambda repo: repo in repos


# --- resolution ----------------------------------------------------------------


def test_the_first_candidate_that_really_exists_is_the_one_returned(tmp_path):
    seen = []

    def exists(repo: str) -> bool:
        seen.append(repo)
        return repo == REAL

    got = launch.resolve_weights(
        catalog.get("llama-3.1-8b"),
        "q8",
        "mlx",
        exists=exists,
        cache=tmp_path / "w.json",
    )
    assert isinstance(got, launch.Resolved)
    assert got.repo == REAL
    # The naive spelling was checked and rejected rather than assumed either way.
    assert seen[0] == NAIVE
    assert seen[-1] == REAL


def test_a_repo_name_is_never_constructed_without_being_confirmed(tmp_path):
    """The whole reason this is a search: `base name + suffix` is a 404 here."""
    got = launch.resolve_weights(
        catalog.get("llama-3.1-8b"),
        "q8",
        "mlx",
        exists=only(NAIVE),  # if it only checked the obvious one it would "work"
        cache=tmp_path / "w.json",
    )
    assert isinstance(got, launch.Resolved) and got.repo == NAIVE
    # ...and with nothing existing, nothing is returned.
    gone = launch.resolve_weights(
        catalog.get("llama-3.1-8b"),
        "q8",
        "mlx",
        exists=only(),
        cache=tmp_path / "empty.json",
    )
    assert isinstance(gone, launch.Refusal)


def test_a_failed_resolution_reports_every_candidate_it_tried(tmp_path):
    """'Not found' is unactionable. 'These six do not exist' is a starting point."""
    gone = launch.resolve_weights(
        catalog.get("qwen3-30b-a3b"), "q4", "mlx", exists=only(), cache=tmp_path / "w.json"
    )
    assert isinstance(gone, launch.Refusal)
    assert "mlx-community/Qwen3-30B-A3B-4bit" in gone.tried
    assert len(gone.tried) >= 2
    assert all(t in gone.render() for t in gone.tried)


def test_vllm_starts_from_the_base_repo_rather_than_searching():
    """vLLM quantises on load, so the catalogue's repo is already the answer."""
    assert launch.candidates(catalog.get("qwen3-30b-a3b"), "q4", "vllm") == ("Qwen/Qwen3-30B-A3B",)


def test_a_model_with_no_known_repo_refuses_instead_of_inventing_one(tmp_path):
    """models.json leaves `repo` absent when we do not know it — on purpose."""
    glm = catalog.get("glm-5.2")
    assert glm.repo is None
    out = launch.resolve_weights(glm, "q4", "vllm", exists=only(), cache=tmp_path / "w.json")
    assert isinstance(out, launch.Refusal)
    assert "no Hugging Face repo is known" in out.reason
    assert "catalog-add" in out.next_step


def test_a_confirmed_resolution_is_remembered_so_the_second_run_is_offline(tmp_path):
    cache = tmp_path / "w.json"
    llama = catalog.get("llama-3.1-8b")
    launch.resolve_weights(llama, "q8", "mlx", exists=only(REAL), cache=cache)

    def never(repo: str) -> bool:
        raise AssertionError(f"should have come from cache; asked about {repo}")

    again = launch.resolve_weights(llama, "q8", "mlx", exists=never, cache=cache)
    assert isinstance(again, launch.Resolved)
    assert again.repo == REAL and again.from_cache
    assert json.loads(cache.read_text())["mlx:llama-3.1-8b:q8"] == REAL


def test_the_cache_lives_outside_the_catalogue(tmp_path, monkeypatch):
    """A resolution is a fact about the Hub, not about the model. Writing it into
    models.json would let one machine's launch edit every other machine's sizing."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    path = launch.state_dir() / "weights.json"
    launch.resolve_weights(catalog.get("llama-3.1-8b"), "q8", "mlx", exists=only(REAL))
    assert path.exists()
    for src in catalog.sources():
        assert path != src and path.parent != src.parent


def test_a_corrupt_cache_does_not_stop_a_launch(tmp_path):
    cache = tmp_path / "w.json"
    cache.write_text("{not json")
    got = launch.resolve_weights(
        catalog.get("llama-3.1-8b"), "q8", "mlx", exists=only(REAL), cache=cache
    )
    assert isinstance(got, launch.Resolved) and got.repo == REAL


def test_being_offline_is_reported_as_being_offline(tmp_path):
    """Not as 'none of these repos exist' — that sends someone hunting a naming
    bug that is not there."""

    def dead(repo: str) -> bool:
        raise OSError("no route to host")

    out = launch.resolve_weights(
        catalog.get("llama-3.1-8b"), "q8", "mlx", exists=dead, cache=tmp_path / "w.json"
    )
    assert isinstance(out, launch.Refusal)
    assert "could not reach" in out.reason
    assert out.tried


# --- planning ------------------------------------------------------------------


def test_a_model_that_does_not_fit_refuses_and_names_the_hosting_next_step(tmp_path):
    out = launch.plan(
        "qwen3-235b-a22b", hw=AIR, exists=only(), cache=tmp_path / "w.json", context=8192
    )
    assert isinstance(out, launch.Refusal)
    assert "short by" in out.reason and "GB" in out.reason
    assert "clickllm host" in out.next_step
    assert "clickllm where" in out.next_step


def test_a_flag_the_installed_engine_rejects_is_a_refusal(tmp_path, monkeypatch):
    """The defect this repo has shipped before: a command every test asserted was
    present and no test ever tried to run."""
    monkeypatch.setattr(launch, "unknown_flags", lambda adapter, argv: ["--prefill-step-size"])
    out = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal)
    assert "--prefill-step-size" in out.reason
    assert "does not accept" in out.reason


def test_a_plan_that_holds_carries_the_command_and_the_endpoint(tmp_path):
    p = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert isinstance(p, launch.LaunchPlan)
    assert p.engine == "mlx"
    assert p.repo == "mlx-community/Qwen3-30B-A3B-8bit"
    assert p.argv[:2] == ("mlx_lm.server", "--model")
    assert p.base == "http://127.0.0.1:8000"
    assert p.command in p.render()
    # What could not be expressed is shown, not dropped.
    assert p.gaps and "NOT EXPRESSED" in p.render()


def test_the_default_bind_address_is_loopback(tmp_path):
    p = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert p.argv[-4:] == ("--host", "127.0.0.1", "--port", "8000")
    wide = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        host="0.0.0.0",  # noqa: S104 — the point of the test
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert any("unauthenticated" in n for n in wide.notes)


def test_an_engine_with_no_verified_dialect_refuses_with_a_way_forward(tmp_path):
    """On Apple at concurrency 1 the planner picks llama.cpp, which has no
    dialect here. The alternative is computed by re-planning, not tabulated."""
    out = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=1,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal)
    assert "llama.cpp" in out.reason
    assert "--concurrency" in out.next_step and "mlx" in out.next_step


def test_an_unpublished_quant_is_refused_with_the_ones_that_exist(tmp_path):
    with pytest.raises(ValueError, match="q3"):
        launch.plan("qwen3-30b-a3b", quant="q3", hw=M4, cache=tmp_path / "w.json")


def test_an_unknown_model_names_itself(tmp_path):
    with pytest.raises(KeyError, match="nope-9b"):
        launch.plan("nope-9b", hw=M4, cache=tmp_path / "w.json")


def test_this_module_never_downloads_anything(tmp_path):
    """The engine fetches its own weights — mlx_lm, vLLM and SGLang all do. A
    second downloader here is 30 GB pulled twice and a cache nobody knows about."""
    p = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    assert any("downloads the weights itself" in n for n in p.notes)
    src = Path(launch.__file__).read_text()
    assert "urlretrieve" not in src


# --- actually starting something -------------------------------------------------
#
# The only code in the repo that spawns a process, so it gets a real one. A
# three-line HTTP server stands in for the engine: what is under test is the
# health poll and the promise not to leave an orphan, neither of which needs
# 30 GB of weights to exercise.

#: The stub reproduces the behaviour that broke the real thing: it answers
#: `/v1/models` with 200 the instant it binds, and refuses to complete a token
#: for the first `LOADING_SECONDS`. That is measured, not invented — against a
#: real `mlx_lm.server` still fetching weights, `/v1/models` returned 200 in 4ms
#: while a genuine completion returned nothing in 30s.
#:
#: So a health poll that watches `/v1/models` passes this stub immediately and
#: is WRONG. Weakening the probe back to a GET makes
#: `test_serve_waits_until_a_real_token_completes` fail, which is the point.
STUB_ENGINE = """
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

T0 = time.monotonic()
LOADING_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
BOUND_AT = time.monotonic()


class H(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # Served from static config, exactly like the real engine: available
        # long before the weights are.
        self._json(200, {"data": [{"id": "stub"}]})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if time.monotonic() - BOUND_AT < LOADING_SECONDS:
            # Still loading. The real server simply does not answer; a 503 is
            # the same signal without holding the socket open for the test.
            self._json(503, {"error": "model is still loading"})
            return
        self._json(200, {"choices": [{"message": {"content": "1"}}]})

    def log_message(self, *a):
        pass


class Reusable(HTTPServer):
    # The port came from a socket the test bound and closed, so it can still be
    # in TIME_WAIT when this process starts.
    allow_reuse_address = True


def bind(port, deadline=20.0):
    end = time.monotonic() + deadline
    last = None
    while True:
        try:
            return Reusable(("127.0.0.1", port), H)
        except OSError as e:
            last = e
            if time.monotonic() > end:
                # Say why on the way out. A stub that dies silently is read by
                # the test as "the engine never answered", which sent one
                # investigation down entirely the wrong path already.
                sys.stderr.write("STUB_BIND_FAILED %s: %r\\n" % (port, last))
                sys.stderr.flush()
                raise
            time.sleep(0.1)


try:
    srv = bind(int(sys.argv[1]))
    # Timestamped so a failure distinguishes "slow to start" from "never
    # started". Interpreter spawn under uv on a macOS CI runner has taken long
    # enough to blow a 30s budget, which read as "never bound" and sent one
    # investigation down entirely the wrong path.
    sys.stderr.write("STUB_BOUND %s after %.1fs\\n" % (sys.argv[1], time.monotonic() - T0))
    sys.stderr.flush()
    srv.serve_forever()
except BaseException as e:
    sys.stderr.write("STUB_DIED %r\\n" % (e,))
    sys.stderr.flush()
    raise
"""


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stub_plan(tmp_path, argv: tuple[str, ...], port: int) -> launch.LaunchPlan:
    p = launch.plan(
        "qwen3-30b-a3b",
        hw=M4,
        context=8192,
        concurrency=8,
        port=port,
        exists=only("mlx-community/Qwen3-30B-A3B-8bit"),
        cache=tmp_path / "w.json",
    )
    return replace(p, argv=argv)


def test_serve_waits_until_a_real_token_completes(tmp_path, capsys):
    """Readiness is a completed token, not a 200 on the model list.

    The stub answers `/v1/models` immediately and refuses completions for two
    seconds, which is what a real engine does while its weights load. A probe
    that watched `/v1/models` would return in milliseconds and hand back an
    endpoint that cannot serve — the defect this reproduces.
    """
    port = _free_port()
    argv = (sys.executable, "-c", STUB_ENGINE, str(port), "2.0")
    plan = _stub_plan(tmp_path, argv, port)

    started = time.monotonic()
    ep = launch.serve(plan, poll=0.05, timeout=180.0)
    waited = time.monotonic() - started
    try:
        assert ep.base == f"http://127.0.0.1:{port}"
        assert ep.pid > 0
        assert waited >= 2.0, (
            f"serve returned after {waited:.2f}s — it cannot have waited for a "
            "real completion, so it is watching the wrong signal"
        )
        assert launch._healthy(ep.base), "serve returned before the endpoint answered"
        # The command is printed before it runs — you can always copy it out.
        assert " ".join(plan.argv[:1]) in capsys.readouterr().out
    finally:
        ep.stop()
    assert ep.process.poll() is not None, "stop() must leave nothing behind"


def test_a_200_on_the_model_list_is_not_treated_as_ready_threaded():
    """The negative control, in-process.

    This does not need a subprocess — only a server that answers `/v1/models`
    and refuses completions. Running it in a thread removes interpreter spawn
    latency, which on a macOS CI runner exceeded the 30s budget and read as
    "never bound", and it binds port 0 itself so there is no window between
    choosing a port and claiming it.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Loading(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib's spelling
            body = b'{"data":[{"id":"stub"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = b'{"error":"model is still loading"}'
            self.send_response(503)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Loading)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        import urllib.request

        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:  # noqa: S310
            assert r.status == 200, "the premise: the model list answers while loading"
        assert not launch._healthy(base, timeout=5.0), (
            "the model list answers 200 while loading; _healthy must not treat "
            "that as ready, or `clickllm run` prints an endpoint that hangs"
        )
    finally:
        srv.shutdown()
        srv.server_close()


def test_an_engine_that_never_answers_is_stopped_not_left_running(tmp_path):
    port = _free_port()
    # Alive, listening to nothing. The failure mode this guards is a launcher
    # that gives up on the wait and leaves the process holding the GPU.
    plan = _stub_plan(tmp_path, (sys.executable, "-c", "import time; time.sleep(300)"), port)
    with pytest.raises(TimeoutError, match="did not complete a single token"):
        launch.serve(plan, poll=0.05, timeout=1.0)


def test_a_missing_engine_says_how_to_install_it(tmp_path):
    port = _free_port()
    plan = _stub_plan(tmp_path, ("clickllm-no-such-engine",), port)
    with pytest.raises(FileNotFoundError, match="pip install mlx-lm"):
        launch.serve(plan, poll=0.05, timeout=5.0)


# --- the module's own self-check ------------------------------------------------


def test_demo_self_check():
    launch.demo()


def test_a_refusal_lists_only_the_repos_it_actually_reached():
    """From the deep review of launch.py (#44).

    When the network fails mid-loop, `Refusal.tried` carried the full candidate
    list — reporting "none of these exist" for repos never asked about. The whole
    value of listing candidates is sending someone to the right place, and five
    unreached names dressed as five confirmed absences sends them to the wrong one.
    """
    calls = []

    def dies_after_one(repo: str) -> bool:
        calls.append(repo)
        if len(calls) > 1:
            raise OSError("no route to host")
        return False

    out = launch.resolve_weights(
        catalog.get("llama-3.1-8b"),
        "q8",
        "mlx",
        exists=dies_after_one,
        cache=Path("/nonexistent/never-written.json"),
    )
    assert isinstance(out, launch.Refusal)
    assert len(out.tried) == len(calls), "reported more candidates than it asked about"
    assert len(out.tried) < len(launch.candidates(catalog.get("llama-3.1-8b"), "q8", "mlx"))
    assert "not absent" in out.reason, "must distinguish unreached from confirmed-missing"


def test_the_readiness_probe_cannot_outlast_the_deadline_it_serves(monkeypatch, tmp_path):
    """From the deep review of launch.py (#44).

    `_healthy`'s timeout defaulted to 10s regardless of `serve`'s budget, so
    `serve(timeout=2.0)` could block ten seconds inside the first probe — the
    caller's deadline overrun 5x by the very thing measuring it. Asserted on the
    timeout the probe is actually HANDED, not on the source text: a source check
    passes for code that computes the right number and then ignores it.
    """
    handed: list[float] = []

    def record(base, model="clickllm-probe", timeout=10.0):
        handed.append(timeout)
        return False  # never ready, so serve() runs its budget out

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    lp = launch.plan(
        "llama-3.1-8b", hw=M4, context=8192, exists=lambda r: True, cache=tmp_path / "w.json"
    )
    assert isinstance(lp, launch.LaunchPlan), lp

    # Patched only now: planning itself shells out to the engine for its flag
    # vocabulary, and a Popen stub installed earlier would intercept that too.
    monkeypatch.setattr(launch, "_healthy", record)
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **k: FakeProc())

    with pytest.raises(TimeoutError):
        launch.serve(lp, timeout=2.0, poll=0.05)

    assert handed, "the probe was never called"
    assert max(handed) <= 2.0, f"probe allowed {max(handed)}s inside a 2.0s budget"
    # And it shrinks as the budget is spent, rather than being pinned at the cap.
    assert handed[-1] < handed[0]
    # Strict at the small end too: a `max(0.5, ...)` floor would re-open the same
    # hole for `serve(timeout=0.1)`, handing the probe 5x the whole budget. Every
    # timeout is strictly positive, so urlopen never gets a nonsensical one.
    assert min(handed) > 0, f"probe handed a non-positive timeout: {min(handed)}"

    handed.clear()
    with pytest.raises(TimeoutError):
        launch.serve(lp, timeout=0.15, poll=0.01)
    assert all(0 < h <= 0.15 for h in handed), f"overran a 0.15s budget: {handed}"
