"""Measuring throughput, and refusing to.

Issue #80 filed the constraint before the feature existed, because getting this
wrong makes the product *less* trustworthy than shipping only estimates: a
measured number carries more authority and it is sticky — written into a receipt
or a box's `bench.json` and believed next quarter, long after the browser tab
that caused it was closed.

So most of these are about the refusals. The one that measures a real endpoint
is here to prove the arithmetic tracks reality, because a refusal machine that
cannot measure anything would pass every test above it.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clickllm import measure as M

ROOT = Path(__file__).resolve().parents[1]


def _samples(*rates: float, tokens: int = 100) -> list[M.Sample]:
    """Samples that decode at exactly the given rates."""
    return [M.Sample(tokens=tokens, decode_seconds=tokens / r, ttft_seconds=0.1) for r in rates]


def _take(samples: list[M.Sample]):
    it = iter(samples)
    return lambda: next(it)


# --- what makes a measurement ----------------------------------------------------


def test_steady_samples_are_a_measurement():
    m = M.measure("http://x/v1", "m", cores=16, samples=3, sampler=_take(_samples(50, 50.5, 49.5)))
    assert m.usable, m.refused
    assert m.median is not None and 49 < m.median < 51


def test_the_number_80_actually_saw_is_refused():
    """46.93 and 33.69 tok/s on identical inputs, on an idle-looking laptop.

    Tested with the figures that motivated the constraint rather than invented
    ones, so a future change to the limit has to argue with the real case.
    """
    m = M.measure("http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(46.93, 33.69)))
    assert not m.usable
    assert any("disagree" in r for r in m.refused), m.refused
    assert "NOT A MEASUREMENT" in m.render()


def test_a_wide_spread_is_reported_as_the_finding_not_hidden_behind_a_median():
    """Rule 3: the spread *is* the finding. A median alone would look fine."""
    m = M.measure("http://x/v1", "m", cores=16, samples=3, sampler=_take(_samples(60, 45, 30)))
    assert m.median == 45, "the median is unremarkable; that is the point"
    assert not m.usable
    assert m.spread is not None and m.spread > M.SPREAD_LIMIT


def test_a_busy_machine_is_refused_even_when_the_samples_agree():
    """Rule 2. Steady numbers under load can mean everything was equally slow,
    which is a stable measurement of the wrong thing."""
    m = M.measure("http://x/v1", "m", cores=4, samples=2, sampler=_take(_samples(20, 20)))
    contended = M.Measurement(
        model=m.model,
        endpoint=m.endpoint,
        samples=m.samples,
        load_before=M.Load(one_minute=17.06, cores=4),
        load_after=m.load_after,
    )
    assert contended.load_before.contended
    assert contended.load_before.per_core is not None
    assert contended.load_before.per_core > M.LOAD_PER_CORE_LIMIT


def test_load_is_recorded_with_every_measurement_even_a_good_one():
    """Rule 1. The conditions travel with the number, not just with refusals."""
    m = M.measure("http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(50, 50)))
    text = m.render()
    assert "before:" in text and "after:" in text
    assert "load" in text
    assert json.loads(m.to_json())["load_before"]["cores"] == 16


def test_an_unknown_load_is_reported_as_unknown_rather_than_zero():
    """Windows has no load average. Zero would read as "the machine was idle",
    which is the most flattering possible reading of no information."""
    unknown = M.Load(one_minute=None, cores=8)
    assert unknown.per_core is None
    assert not unknown.contended, "unknown must not be treated as contended either"
    assert "unknown" in unknown.render()


def test_one_sample_is_refused_because_it_has_no_spread():
    with pytest.raises(ValueError, match="spread"):
        M.measure("http://x/v1", "m", cores=8, samples=1, sampler=_take(_samples(50)))


# --- rule 4: measured must not beat estimated on authority alone -----------------


def test_both_numbers_are_shown_with_the_ratio_between_them():
    m = M.measure(
        "http://x/v1", "m", cores=16, samples=2, roofline=100.0, sampler=_take(_samples(50, 50))
    )
    text = m.render()
    assert "roofline" in text and "estimate" in text
    assert "50% of the estimate" in text, text
    assert m.ratio_to_roofline == pytest.approx(0.5)


def test_a_refused_measurement_says_the_estimate_stands():
    m = M.measure(
        "http://x/v1", "m", cores=16, samples=2, roofline=100.0, sampler=_take(_samples(60, 30))
    )
    text = m.render()
    assert "roofline estimate stands" in text
    assert json.loads(m.to_json())["measured"] is False


def test_the_json_says_whether_it_counts():
    """Whatever consumes this downstream must be able to tell without parsing
    prose — a `bench.json` that omitted the distinction is exactly how a
    contended sample ends up trusted."""
    good = M.measure("http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(50, 50)))
    bad = M.measure("http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(60, 30)))
    assert json.loads(good.to_json())["measured"] is True
    assert json.loads(bad.to_json())["measured"] is False
    assert json.loads(bad.to_json())["refused"], "a refusal with no reason is not one"


# --- the arithmetic, against a server that decodes at a known rate ---------------


SERVER = """
import json, os, time, socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

RATE = float(os.environ["RATE_TOK_S"])
JITTER = float(os.environ.get("JITTER", "0"))
N = 0

class Server(HTTPServer):
    # See tests/test_observe.py: `HTTPServer.server_bind` calls `getfqdn`
    # between bind() and listen(), which blocks on a runner with no PTR record.
    allow_reuse_address = True
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        global N
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        want = int(body.get("max_tokens") or 32)
        rate = RATE * (1 + JITTER if N % 2 else 1 - JITTER)
        N += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        def frame(o):
            self.wfile.write(("data: " + json.dumps(o) + "\\n\\n").encode()); self.wfile.flush()
        frame({"choices":[{"delta":{"role":"assistant"}}]})
        time.sleep(0.05)
        for i in range(want):
            frame({"choices":[{"delta":{"content":"t%d " % i}}]})
            time.sleep(1.0 / rate)
        self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()

HTTPServer.allow_reuse_address = True
Server(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
"""


def _free_ports(n: int) -> list[int]:
    """Held open together, then released — asking one at a time can return the
    port the previous call just freed."""
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind(("127.0.0.1", 0))
        return [int(s.getsockname()[1]) for s in socks]
    finally:
        for s in socks:
            s.close()


def _serve(tmp_path: Path, port: int, rate: float, jitter: float = 0.0):
    src = tmp_path / f"server_{port}.py"
    src.write_text(SERVER)
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env={**os.environ, "RATE_TOK_S": str(rate), "JITTER": str(jitter), "PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return proc
        except OSError:
            if proc.poll() is not None:
                _, err = proc.communicate()
                pytest.fail(f"server exited: {err.decode(errors='replace')[-500:]}")
            time.sleep(0.1)
    proc.kill()
    pytest.fail(f"server never listened on {port}")


@pytest.mark.parametrize("rate", [25.0, 100.0])
def test_the_measurement_tracks_a_server_that_decodes_at_a_known_rate(tmp_path, rate):
    """The control for every refusal above.

    A module that only ever declines would pass all of them, so this checks the
    stopwatch: a server pacing tokens at a known rate must be measured at
    something near it, and a faster one must measure faster. The absolute figure
    lands under nominal because the server's own write and flush cost is real
    time a client waits — which is the point of measuring end to end rather than
    timing a kernel.
    """
    port = _free_ports(1)[0]
    proc = _serve(tmp_path, port, rate)
    try:
        m = M.measure(
            f"http://127.0.0.1:{port}/v1",
            "fake",
            cores=os.cpu_count() or 4,
            samples=3,
            max_tokens=40,
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)

    assert m.median is not None
    # Wide bounds on purpose: this asserts the measurement is *of* the server,
    # not that CI can hit a millisecond. A stopwatch returning a constant, or
    # counting frames instead of timing them, fails this at any tolerance.
    assert 0.4 * rate < m.median < 1.15 * rate, f"{m.median:.1f} tok/s from a {rate} tok/s server"
    assert all(s.ttft_seconds > 0 for s in m.samples), "prefill was never observed"


def test_a_faster_server_measures_faster(tmp_path):
    """Ordering, which no constant can satisfy."""
    slow_port, fast_port = _free_ports(2)
    slow = _serve(tmp_path, slow_port, 25.0)
    fast = _serve(tmp_path, fast_port, 100.0)
    try:
        s = M.measure(f"http://127.0.0.1:{slow_port}/v1", "f", cores=4, samples=3, max_tokens=40)
        f = M.measure(f"http://127.0.0.1:{fast_port}/v1", "f", cores=4, samples=3, max_tokens=40)
    finally:
        for p in (slow, fast):
            p.kill()
            p.wait(timeout=5)
    assert s.median is not None and f.median is not None
    assert f.median > s.median * 1.5, f"slow {s.median:.1f}, fast {f.median:.1f}"


def test_a_jittery_server_is_refused_end_to_end(tmp_path):
    """The whole point, against a real socket rather than injected samples."""
    port = _free_ports(1)[0]
    proc = _serve(tmp_path, port, 50.0, jitter=0.30)
    try:
        m = M.measure(f"http://127.0.0.1:{port}/v1", "f", cores=4, samples=4, max_tokens=40)
    finally:
        proc.kill()
        proc.wait(timeout=5)
    assert not m.usable, f"a ±30% server was accepted: spread {m.spread}"
    assert any("disagree" in r for r in m.refused)


def test_an_endpoint_that_streams_nothing_is_an_error_not_a_zero(tmp_path):
    """Zero tokens per second would be a number, and a wrong one. A server that
    answered without streaming is a setup problem, and it says so."""
    port = _free_ports(1)[0]
    src = tmp_path / "empty.py"
    src.write_text(SERVER.replace("for i in range(want):", "for i in range(0):"))
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env={**os.environ, "RATE_TOK_S": "50", "PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    try:
        with pytest.raises(ValueError, match="no streamed tokens"):
            M.measure(f"http://127.0.0.1:{port}/v1", "f", cores=4, samples=2, max_tokens=8)
    finally:
        proc.kill()
        proc.wait(timeout=5)


# --- the CLI --------------------------------------------------------------------


def test_the_cli_defaults_match_the_modules():
    """`cli.py` mirrors these so building the parser does not import `measure`.
    A mirror that drifts hands a user a different default from `--help` than the
    module applies, which is the quietest kind of wrong."""
    src = (ROOT / "src" / "clickllm" / "cli.py").read_text()
    for name, value in (
        ("M_DEFAULT_SAMPLES", M.DEFAULT_SAMPLES),
        ("M_DEFAULT_MAX_TOKENS", M.DEFAULT_MAX_TOKENS),
    ):
        m = re.search(rf"^{name} = (\d+)$", src, re.M)
        assert m, f"{name} is gone from cli.py"
        assert int(m.group(1)) == value, f"{name} is {m.group(1)}, module says {value}"


def test_measure_exits_nonzero_when_it_refuses(tmp_path):
    """So a script that wanted a number knows it did not get one. Not an error:
    "the machine was too busy" is a legitimate outcome and the report says so."""
    port = _free_ports(1)[0]
    proc = _serve(tmp_path, port, 50.0, jitter=0.30)
    try:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "clickllm.cli",
                "measure",
                "--endpoint",
                f"http://127.0.0.1:{port}/v1",
                "--served-model",
                "fake",
                "--samples",
                "4",
                "--max-tokens",
                "40",
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)
    assert out.returncode == 1, out.stdout + out.stderr
    assert "NOT A MEASUREMENT" in out.stdout
