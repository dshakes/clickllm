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

import contextlib
import json
import os
import signal
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pytest

from onpar import catalog, launch
from onpar.hardware import Hardware

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
    # The key carries the resolver's version, so match on the suffix rather than
    # pinning a literal that a resolver bump would break for no real reason.
    stored = json.loads(cache.read_text())
    key = next(k for k in stored if k.endswith("mlx:llama-3.1-8b:q8"))
    assert stored[key] == REAL


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
    assert "onpar host" in out.next_step
    assert "onpar where" in out.next_step


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


def test_an_engine_with_no_verified_dialect_refuses_with_a_way_forward():
    """Every engine in `Engine` now has a dialect, so `launch.plan` can no longer
    reach its refusal branch through the planner.

    The branch stays: the next engine added to the enum will hit it before its
    adapter exists, and refusing beats emitting another engine's flags. What
    changes is that this test asserts the invariant that made the branch
    unreachable, rather than pretending to exercise a path it cannot get to.
    """
    from onpar.engines import adapter_for
    from onpar.plan import Engine

    undialected = [str(e) for e in Engine if adapter_for(str(e)) is None]
    assert not undialected, f"planner can pick these and launch cannot build them: {undialected}"
    assert adapter_for("not-an-engine") is None, "adapter_for must still refuse the unknown"


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
            "that as ready, or `onpar run` prints an endpoint that hangs"
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
    plan = _stub_plan(tmp_path, ("onpar-no-such-engine",), port)
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
    # The candidate list is still handed over — offline, "go look for these" is
    # the actionable thing. What must not happen is claiming they were reached.
    assert out.tried, "dropped the candidate list the user needs offline"
    assert f"{len(calls) - 1} of {len(out.tried)} candidates answered" in out.reason
    assert "TRIED" not in out.render(), "labelled unreached candidates as tried"
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

    def record(base, model="onpar-probe", timeout=10.0):
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
    # This test is about the probe's budget, not teardown. The Popen stub above
    # is global to the module, so the `ps` call inside `_descendants` would get
    # a FakeProc too — stubbing it out keeps the stub's blast radius equal to
    # what the test actually means to replace.
    monkeypatch.setattr(launch, "_descendants", lambda pid: ())

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


def test_a_confirmed_repo_does_not_claim_candidates_it_never_looked_at(tmp_path):
    """The same overclaim as the refusal path, on the success path — and this one
    reaches the screen. `LaunchPlan.render()` prints "confirmed; N candidate(s)
    checked" straight off `Resolved.tried`, so matching on the first candidate
    printed the whole list's length as though every name had been looked up.
    """
    asked: list[str] = []

    def first_one_wins(repo: str) -> bool:
        asked.append(repo)
        return True

    model = catalog.get("llama-3.1-8b")
    assert len(launch.candidates(model, "q8", "mlx")) > 1, "fixture: >1 candidate"

    out = launch.resolve_weights(
        model, "q8", "mlx", exists=first_one_wins, cache=tmp_path / "a.json"
    )
    assert isinstance(out, launch.Resolved)
    assert len(asked) == 1, "stopped at the first match, as expected"
    assert len(out.tried) == 1, f"claimed {len(out.tried)} checked, asked about {len(asked)}"

    # And the number that reaches the screen is the one that was true.
    asked.clear()
    lp = launch.plan(
        "llama-3.1-8b", hw=M4, context=8192, exists=first_one_wins, cache=tmp_path / "b.json"
    )
    assert isinstance(lp, launch.LaunchPlan), lp
    assert f"confirmed; {len(asked)} candidate(s) checked" in lp.render()


# --- teardown: the engine AND what it spawned ----------------------------------


def test_stopping_an_engine_kills_the_workers_it_spawned():
    """From the deep review of launch.py (#44), the finding left open.

    `stop()` signalled only the immediate child. vLLM runs an API server that
    forks engine workers, each holding GPU memory — terminate the parent alone
    and the workers are reparented to init and keep the device, while
    `serve()`'s timeout path reports the engine "has been stopped rather than
    left running".

    Real processes, not a stub: the whole defect is about what the OS does with
    orphans, and a stub would model exactly the assumption under test.
    """
    import subprocess as sp
    import sys
    import time as t

    # A parent that spawns two children and then sleeps. Children outlive it
    # unless something goes looking for them.
    script = (
        "import subprocess,sys,time;"
        "ks=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'])"
        " for _ in range(2)];"
        "print(' '.join(str(k.pid) for k in ks),flush=True);"
        "time.sleep(120)"
    )
    kids: list[int] = []
    proc = sp.Popen([sys.executable, "-c", script], stdout=sp.PIPE, text=True)
    try:
        kids = [int(p) for p in proc.stdout.readline().split()]
        assert len(kids) == 2
        t.sleep(0.3)
        assert all(launch._running(k) for k in kids), "fixture: children did not start"
        assert set(kids) <= set(launch._descendants(proc.pid)), "did not find the children"

        ep = launch.Endpoint("http://127.0.0.1:1", "m", proc.pid, 0.0, proc)
        survivors = ep.stop(grace=3.0)

        assert survivors == (), f"left running: {survivors}"
        for k in kids:
            assert not launch._running(k), f"worker {k} outlived the engine"
    finally:
        for p in [proc.pid, *kids]:
            with contextlib.suppress(OSError):
                os.kill(p, signal.SIGKILL)


def test_descendants_degrades_to_empty_rather_than_raising(monkeypatch):
    """Best effort. A machine without `ps` must not take the shutdown path with
    it — reporting what it could not do beats raising during teardown."""

    def boom(*a, **k):
        raise FileNotFoundError("ps")

    monkeypatch.setattr(launch.subprocess, "run", boom)
    assert launch._descendants(os.getpid()) == ()


def test_two_concurrent_resolutions_do_not_lose_each_others_cache_entry(tmp_path):
    """From the same review. `_cache_write` did read-modify-write with no lock
    and a FIXED temp filename, so two `onpar run` invocations racing lost one
    entry and could publish a half-written file."""
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "weights.json"
    keys = [f"model-{i}" for i in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda k: launch._cache_write(path, k, f"repo/{k}"), keys))

    got = launch._cache_read(path)
    missing = sorted(set(keys) - set(got))
    assert not missing, f"{len(missing)} entries lost to the race: {missing[:5]}"
    assert all(got[k] == f"repo/{k}" for k in keys)


def test_no_module_writes_json_through_a_shared_scratch_file():
    """The scratch name was `weights.json.tmp` — identical in every process, so
    two writers wrote the same path and one renamed what the other was still
    writing. Fixed in launch.py first, and the same code sat in cache.py and
    catalog_update.py for another release, because fixing one instance of a
    duplicated defect is not fixing the defect.

    Asserted across the tree rather than inside one function: the failure mode
    is a fourth copy, and a test that greps a single module cannot see one.
    """
    import re as _re

    root = Path(__file__).resolve().parents[1] / "src" / "onpar"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "atomicio.py":
            continue  # the one implementation, which is tested on its own terms
        text = path.read_text()
        for m in _re.finditer(r'with_suffix\(\s*["\']\.[\w.]*tmp["\']\s*\)', text):
            offenders.append(f"{path.name}: {m.group(0)}")
    assert not offenders, (
        "these build a scratch filename that another process will pick too — "
        "use onpar.atomicio instead:\n  " + "\n  ".join(offenders)
    )


def test_a_repo_cached_by_older_resolver_logic_is_not_served_by_the_new_one():
    """Proved the hard way while building the llama.cpp adapter.

    Its first cut proposed the transformers repo for llama.cpp. That answer was
    cached, and after the GGUF fix landed the resolver kept serving
    `meta-llama/Llama-3.1-8B-Instruct` to `llama-server`, which cannot load it —
    a command that fails on start-up, produced by corrected code.

    A cached repo is a fact about (model, quant, engine) AND about the logic
    that proposed it. A wrong answer that outlives the bug which made it is
    worse than no cache: upgrading does not clear it and nothing says why.
    """
    import json

    from onpar import catalog

    with tempfile.TemporaryDirectory() as d:
        cache = Path(d) / "w.json"
        # A resolution written by an older resolver, under the old key shape.
        cache.write_text(json.dumps({"llama.cpp:llama-3.1-8b:q4": "meta-llama/Llama-3.1-8B"}))

        asked: list[str] = []

        def only_gguf(repo: str) -> bool:
            asked.append(repo)
            return repo.endswith("-GGUF")

        out = launch.resolve_weights(
            catalog.get("llama-3.1-8b"), "q4", "llama.cpp", exists=only_gguf, cache=cache
        )
        assert isinstance(out, launch.Resolved), out
        # The quant is tagged onto the repo (`--hf-repo` selects the file with
        # it), so the bare repo id is a prefix, not the whole answer.
        assert out.repo.split(":")[0].endswith("-GGUF"), (
            f"served a stale non-GGUF answer: {out.repo}"
        )
        assert out.repo.endswith(":Q4_K_M"), f"lost the quant tag: {out.repo}"
        assert asked, "trusted the stale entry instead of re-resolving"


def test_llamacpp_is_offered_gguf_repos_and_ollama_is_offered_none():
    """llama.cpp does not load a transformers repo; Ollama has its own registry
    and pulls on first use, so there is no Hub repo to confirm at all."""
    from onpar import catalog

    m = catalog.get("llama-3.1-8b")
    gguf = launch.candidates(m, "q4", "llama.cpp")
    assert gguf and all(c.endswith("-GGUF") for c in gguf), gguf
    assert launch.candidates(m, "q4", "ollama") == ()


# --- hardware compatibility for a forced engine ---------------------------------

H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * GB,
    usable_bytes=76 * GB,
    bandwidth_gbps=3350.0,
    cores=132,
)

TPU_V6E = Hardware(
    kind="tpu",
    name="TPU v6e Trillium (8 chips)",
    total_bytes=256 * GB,
    usable_bytes=240 * GB,
    bandwidth_gbps=11384.0,
    cores=8,
    devices=8,
)


def test_forcing_a_cuda_only_engine_onto_apple_hardware_refuses(tmp_path):
    """--engine bypasses `_pick_engine`, which is the only place hardware ever
    gated an engine choice — so `plan()` has to check it itself, or a caller can
    force sglang onto a Mac and get a command that cannot start."""
    out = launch.plan(
        "llama-3.1-8b",
        engine="sglang",
        hw=M4,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal)
    assert "sglang" in out.reason
    assert "apple" in out.reason
    assert "--engine" in out.next_step


def test_forcing_ollama_onto_tpu_hardware_refuses(tmp_path):
    """`ENGINE_HARDWARE` once listed Ollama as TPU-compatible — a copy-paste
    leftover contradicting `plan._pick_engine`'s own rule that vLLM's TPU
    backend is the only engine that runs there (it lowers through JAX; Ollama
    has no such path). `--engine ollama` on a TPU host must refuse, not pass
    the gate and hand back a command that cannot start."""
    out = launch.plan(
        "llama-3.1-8b",
        engine="ollama",
        hw=TPU_V6E,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal)
    assert "ollama" in out.reason
    assert "tpu" in out.reason
    assert "--engine" in out.next_step


def test_forcing_an_engine_the_hardware_actually_runs_does_not_refuse(tmp_path):
    """The gate must not reject a legitimate forced choice — only a mismatch."""
    out = launch.plan(
        "llama-3.1-8b",
        quant="q4",
        engine="vllm",
        hw=H100,
        context=8192,
        exists=only("meta-llama/Llama-3.1-8B-Instruct"),
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.LaunchPlan), out
    assert out.engine == "vllm"


# --- Ollama: real tags in, real concurrency checked -----------------------------


def test_ollama_resolution_refuses_rather_than_fabricating_a_tag(tmp_path):
    """The catalogue id is not an Ollama tag. Returning it as `Resolved` would
    have `_healthy` poll a model name Ollama was never asked to serve, and the
    printed command would start a bare `ollama serve` with nothing telling
    anyone what to request from it."""
    # Dimensions that actually fit: at the defaults this model does not, and the
    # sizing refusal fires first — so the test passed through a branch that says
    # nothing about tags.
    out = launch.plan(
        "llama-3.1-8b",
        engine="ollama",
        hw=M4,
        context=8192,
        concurrency=1,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal), out
    assert "not an Ollama tag" in out.reason, out.reason
    assert "--tag" in out.next_step


def test_ollama_resolution_accepts_an_explicit_tag(tmp_path):
    out = launch.plan(
        "llama-3.1-8b",
        engine="ollama",
        tag="llama3.1:8b-instruct-q4_K_M",
        hw=M4,
        context=8192,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.LaunchPlan), out
    assert out.repo == "llama3.1:8b-instruct-q4_K_M"


def test_ollama_is_fit_checked_at_the_planners_padded_slot_count_not_requested_concurrency(
    tmp_path,
):
    """Interactive concurrency=1 becomes MAX_CONCURRENT=32 in the planner, and
    OLLAMA_NUM_PARALLEL carries that padded count — not the concurrency the fit
    was originally checked at. Proves the same re-solve guard llama.cpp has
    also covers Ollama."""
    out = launch.plan(
        "llama-3.1-8b",
        engine="ollama",
        tag="llama3.1:8b-instruct-q4_K_M",
        quant="q4",
        concurrency=1,
        context=8192,
        hw=M4,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.LaunchPlan), out
    assert out.fit.concurrency == 32
    assert ("OLLAMA_NUM_PARALLEL", "32") in out.env


def test_ollama_refuses_rather_than_launch_more_parallelism_than_was_checked(tmp_path):
    """Same guard, its refusal side: a machine that fits one stream but not the
    32 the planner pads interactive workloads to must not silently approve."""
    out = launch.plan(
        "llama-3.1-8b",
        engine="ollama",
        tag="llama3.1:8b-instruct-q4_K_M",
        quant="q4",
        concurrency=1,
        context=8192,
        hw=AIR,
        cache=tmp_path / "w.json",
    )
    assert isinstance(out, launch.Refusal), out
    assert "short by" in out.reason and "GB" in out.reason
