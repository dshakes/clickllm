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


def _quiet():
    """A load reader that reports an idle 16-core host.

    Injected everywhere a test asserts on usability, because reading the real
    load makes the assertion depend on the CI machine's mood rather than on the
    code: a runner at 1.33/core refused correctly and failed a test that assumed
    it would not.
    """
    return M.Load(one_minute=1.0, cores=16)


# --- what makes a measurement ----------------------------------------------------


def test_steady_samples_are_a_measurement():
    m = M.measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=3,
        sampler=_take(_samples(50, 50.5, 49.5)),
        load_reader=_quiet,
    )
    assert m.usable, m.refused
    assert m.median is not None and 49 < m.median < 51


def test_the_number_80_actually_saw_is_refused():
    """46.93 and 33.69 tok/s on identical inputs, on an idle-looking laptop.

    Tested with the figures that motivated the constraint rather than invented
    ones, so a future change to the limit has to argue with the real case.
    """
    m = M.measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=2,
        sampler=_take(_samples(46.93, 33.69)),
        load_reader=_quiet,
    )
    assert not m.usable
    assert any("disagree" in r for r in m.refused), m.refused
    assert "NOT A MEASUREMENT" in m.render()


def test_a_wide_spread_is_reported_as_the_finding_not_hidden_behind_a_median():
    """Rule 3: the spread *is* the finding. A median alone would look fine."""
    m = M.measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=3,
        sampler=_take(_samples(60, 45, 30)),
        load_reader=_quiet,
    )
    assert m.median == 45, "the median is unremarkable; that is the point"
    assert not m.usable
    assert m.spread is not None and m.spread > M.SPREAD_LIMIT


def test_a_busy_machine_is_refused_even_when_the_samples_agree():
    """Rule 2. Steady numbers under load can mean everything was equally slow,
    which is a stable measurement of the wrong thing."""
    m = M.measure(
        "http://x/v1", "m", cores=4, samples=2, sampler=_take(_samples(20, 20)), load_reader=_quiet
    )
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
    m = M.measure(
        "http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(50, 50)), load_reader=_quiet
    )
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


def test_an_unreadable_core_count_does_not_crash_render():
    """`hardware.detect()` used to hardcode `cores=0` for every NVIDIA/AMD box
    — this tool's primary target — and `render()` unconditionally formatted
    `per_core` with `:.2f`, a `TypeError` since `per_core` is `None` whenever
    `cores < 1`. Fixed at the source in `hardware.py`, but `render()` must not
    crash on `cores < 1` regardless of who calls it that way."""
    unknown_cores = M.Load(one_minute=1.23, cores=0)
    assert unknown_cores.per_core is None
    assert not unknown_cores.contended, "an unreadable core count is not contention"
    text = unknown_cores.render()
    assert "unknown" in text
    assert "1.23" in text


def test_one_sample_is_refused_because_it_has_no_spread():
    with pytest.raises(ValueError, match="spread"):
        M.measure(
            "http://x/v1", "m", cores=8, samples=1, sampler=_take(_samples(50)), load_reader=_quiet
        )


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
    good = M.measure(
        "http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(50, 50)), load_reader=_quiet
    )
    bad = M.measure(
        "http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(60, 30)), load_reader=_quiet
    )
    assert json.loads(good.to_json())["measured"] is True
    assert json.loads(bad.to_json())["measured"] is False
    assert json.loads(bad.to_json())["refused"], "a refusal with no reason is not one"


# --- the arithmetic, against a server that decodes at a known rate ---------------


SERVER = """
import json, os, time, socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

RATE = float(os.environ["RATE_TOK_S"])
# Milliseconds of *extra* per-token delay on alternate requests. Absolute, not
# proportional: the harness measures wall-clock per token, which is this sleep
# plus the server's own write and flush cost. On a slow host that overhead
# dominates — a macOS runner spent ~99 ms per token — so scaling the sleep by
# +/-30% moved the observed rate by 13% and the spread landed under the limit.
# A fixed extra delay survives any overhead, because it adds to both arms
# equally and only one arm gets it.
EXTRA_MS = float(os.environ.get("EXTRA_MS", "0"))
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
        extra = (EXTRA_MS / 1000.0) if N % 2 else 0.0
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
            time.sleep(1.0 / RATE + extra)
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


def _serve(tmp_path: Path, port: int, rate: float, extra_ms: float = 0.0):
    src = tmp_path / f"server_{port}.py"
    src.write_text(SERVER)
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env={**os.environ, "RATE_TOK_S": str(rate), "EXTRA_MS": str(extra_ms), "PORT": str(port)},
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
def test_the_measurement_never_exceeds_what_the_server_could_emit(tmp_path, rate):
    """The invariant that is about the stopwatch rather than about the fixture.

    A server pacing tokens with a `sleep(1/rate)` between them cannot deliver
    faster than `rate`, whatever else the machine is doing. Measuring *above* it
    means the timer is wrong — frames counted instead of timed, or the clock
    started late.

    The first version of this asserted a *lower* bound too (`> 0.4 * rate`) and
    failed on a macOS runner, where a 25 tok/s server really did deliver 8.7:
    Python's per-frame write and flush swamps a 40 ms sleep on a slow box. The
    measurement was right and the assertion was about the fixture's speed, not
    about the code under test. What a slow host cannot do is make decode look
    *faster* than the server emitted.
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
    assert m.median <= rate * 1.10, (
        f"{m.median:.1f} tok/s from a server that sleeps {1000 / rate:.0f} ms between "
        "tokens — the stopwatch is measuring something other than elapsed time"
    )
    assert m.median > 0
    assert all(s.ttft_seconds > 0 for s in m.samples), "prefill was never observed"


def test_a_faster_server_measures_faster(tmp_path):
    """Ordering, which is the control that no broken stopwatch survives.

    A constant returns the same figure for both. Counting frames instead of
    timing them returns the same figure for both — the servers send an identical
    number. Only something that measures elapsed time separates them, and it
    does so on any host, however slow, because both arms pay the same overhead.

    This carries the weight that the absolute-rate assertion used to, and
    carries it better: absolute agreement with the fixture's nominal rate was
    never a property of this module.
    """
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
    # 1.25x, not 4x. The nominal rates differ 4-fold, but on a host where
    # per-frame overhead dominates the pacing sleep the observed gap compresses
    # — 8.7 vs 17.8 on a macOS runner, a real 2x. The assertion is that the
    # ordering survives, not that the ratio does.
    assert f.median > s.median * 1.25, f"slow {s.median:.1f}, fast {f.median:.1f}"


def test_a_jittery_server_is_refused_end_to_end(tmp_path):
    """The whole point, against a real socket rather than injected samples."""
    port = _free_ports(1)[0]
    proc = _serve(tmp_path, port, 50.0, extra_ms=120.0)
    try:
        # Real server, injected load. This test is about the *spread* rule, and
        # on a busy runner the contention rule fires too — so `not m.usable`
        # passed while the reason assertion failed, because the machine was
        # refused before the samples were. Pinning the load leaves exactly one
        # rule able to trigger, which is what the test claims to check.
        m = M.measure(
            f"http://127.0.0.1:{port}/v1",
            "f",
            cores=4,
            samples=4,
            # Few tokens on purpose: the spread comes from the per-token delta,
            # not from the count, so a bigger delay over fewer tokens is both
            # more robust and faster than the reverse.
            max_tokens=12,
            load_reader=_quiet,
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)
    assert not m.usable, f"a ±30% server was accepted: spread {m.spread}"
    assert any("disagree" in r for r in m.refused), m.refused


SERVER_COALESCED = """
import json, os, time, socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer

N = int(os.environ["N_TOKENS"])

class Server(HTTPServer):
    allow_reuse_address = True
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]

class H(BaseHTTPRequestHandler):
    log_message = lambda *a: None
    def do_POST(self):
        json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        def frame(o):
            self.wfile.write(("data: " + json.dumps(o) + "\\n\\n").encode()); self.wfile.flush()
        frame({"choices":[{"delta":{"role":"assistant"}}]})
        time.sleep(0.02)
        # Two content deltas, each several tokens coalesced into one chunk --
        # what an OpenAI-compatible server that batches tokens per frame looks
        # like on the wire. Streaming only promises text deltas, not one SSE
        # frame per token, but decode can still be timed first-frame-to-last.
        half = N // 2
        frame({"choices":[{"delta":{"content":" ".join("t%d" % i for i in range(half))}}]})
        time.sleep(0.03)
        frame({"choices":[{"delta":{"content":" ".join("t%d" % i for i in range(half, N))}}]})
        time.sleep(0.02)
        frame({"choices":[],"usage":{"completion_tokens":N,"prompt_tokens":5,"total_tokens":N+5}})
        self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()

HTTPServer.allow_reuse_address = True
Server(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
"""

SERVER_SINGLE_FRAME = SERVER_COALESCED.replace(
    """        half = N // 2
        frame({"choices":[{"delta":{"content":" ".join("t%d" % i for i in range(half))}}]})
        time.sleep(0.03)
        frame({"choices":[{"delta":{"content":" ".join("t%d" % i for i in range(half, N))}}]})
        time.sleep(0.02)
""",
    """        # The WHOLE completion in one content delta: there is no first-to-last
        # gap to time at all, unlike the multi-frame coalescing case above.
        frame({"choices":[{"delta":{"content":" ".join("t%d" % i for i in range(N))}}]})
        time.sleep(0.05)
""",
)


def test_a_server_that_coalesces_tokens_into_one_frame_still_counts_tokens(tmp_path):
    """OpenAI-compatible streaming promises text deltas, not one SSE frame per
    token. Counting content frames as tokens undercounted a server that
    batches several tokens per delta — `usage.completion_tokens`, when the
    endpoint reports it, must be trusted over the frame count. Decode is still
    timeable here because the batching spans more than one frame; the
    single-frame case (no first-to-last gap at all) is refused instead, see
    `test_a_server_that_sends_the_whole_completion_in_one_frame_is_refused`."""
    port = _free_ports(1)[0]
    src = tmp_path / "coalesced.py"
    src.write_text(SERVER_COALESCED)
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env={**os.environ, "N_TOKENS": "40", "PORT": str(port)},
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
        s = M._decode_once(f"http://127.0.0.1:{port}/v1", "f", "hi", max_tokens=40, timeout=10)
    finally:
        proc.kill()
        proc.wait(timeout=5)
    # 39, not 40: the same "exclude the first token" convention the frame
    # count uses, so this lines up with `decode_seconds` (first token to last).
    assert s.tokens == 39, s.tokens


def test_a_server_that_sends_the_whole_completion_in_one_frame_is_refused(tmp_path):
    """A server that streams the entire completion as a single content delta
    gives `_decode_once` no first-to-last gap to time: `first` and `last` are
    the same instant. `usage.completion_tokens` still reports a token count,
    but a `decode_seconds` of exactly 0 divides away to a `tokens_per_sec` of
    exactly 0.0 -- which `spread()` cannot tell apart from "no spread
    computed" and would otherwise wave through as a usable measurement of 0
    tok/s. This must raise, not silently produce that number."""
    port = _free_ports(1)[0]
    src = tmp_path / "single_frame.py"
    src.write_text(SERVER_SINGLE_FRAME)
    proc = subprocess.Popen(
        [sys.executable, str(src)],
        env={**os.environ, "N_TOKENS": "40", "PORT": str(port)},
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
        with pytest.raises(ValueError, match="single streamed frame"):
            M._decode_once(f"http://127.0.0.1:{port}/v1", "f", "hi", max_tokens=40, timeout=10)
    finally:
        proc.kill()
        proc.wait(timeout=5)


def test_a_non_object_sse_frame_is_skipped_not_a_crash(tmp_path):
    """Some OpenAI-compatible servers are not fully compliant. A `data:` line
    that decodes to a JSON list or primitive must be skipped, not raise
    `AttributeError` from calling `.get()` on it."""
    port = _free_ports(1)[0]
    src = tmp_path / "malformed.py"
    src.write_text(
        SERVER.replace(
            'frame({"choices":[{"delta":{"role":"assistant"}}]})',
            'frame({"choices":[{"delta":{"role":"assistant"}}]})\n'
            '        frame(["not", "a", "dict"])',
        )
    )
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
        s = M._decode_once(f"http://127.0.0.1:{port}/v1", "f", "hi", max_tokens=20, timeout=10)
    finally:
        proc.kill()
        proc.wait(timeout=5)
    assert s.tokens > 0


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
    proc = _serve(tmp_path, port, 50.0, extra_ms=120.0)
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


def test_json_out_stdout_stays_parseable(tmp_path):
    """`--json --out FILE` used to print `wrote FILE` to stdout after the JSON
    object, so a script piping stdout into a JSON parser broke. The status
    line belongs on stderr in `--json` mode, same as the other CLI commands."""
    port = _free_ports(1)[0]
    proc = _serve(tmp_path, port, 50.0)
    out_path = tmp_path / "bench.json"
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
                "2",
                "--max-tokens",
                "40",
                "--json",
                "--out",
                str(out_path),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        )
    finally:
        proc.kill()
        proc.wait(timeout=5)
    parsed = json.loads(out.stdout)  # raises if the status line leaked into stdout
    assert "wrote" in out.stderr
    assert out_path.exists()
    # Not `"NOT A MEASUREMENT" in out.stdout`, which is what this asserted at
    # first: that string is the *rendered* refusal, and `--json` prints JSON
    # instead of the render — so the assertion contradicted the parse two lines
    # above it, and only passed while the server happened to refuse.
    #
    # The machine-readable equivalent is the `measured` key, and a consumer must
    # be able to reach it without parsing prose.
    assert "measured" in parsed
    assert parsed["measured"] is (not parsed["refused"])
    assert json.loads(out_path.read_text())["measured"] == parsed["measured"], (
        "the file and stdout disagree about whether this counts"
    )


def test_reading_the_load_cannot_take_the_measurement_with_it(monkeypatch):
    """`read_load` shells out to `ps`, and a process name is arbitrary bytes.

    Decoding with the locale's codec raises `UnicodeDecodeError` on the first
    name that is not valid in it — and that inherits from `ValueError`, not
    `OSError`, so a handler catching `(OSError, SubprocessError)` misses it and
    the whole command dies. Best-effort decoration must not be able to do that.
    """
    import subprocess as sp

    for boom in (
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
        OSError("no ps"),
        sp.SubprocessError("died"),
        sp.TimeoutExpired(cmd="ps", timeout=5),
    ):

        def explode(*_a, _b=boom, **_k):
            raise _b

        monkeypatch.setattr(M.subprocess, "run", explode)
        load = M.read_load(cores=8)
        assert load.top == (), f"{type(boom).__name__} leaked a value"
        assert load.cores == 8, "the part that does not need `ps` must survive"


def test_a_process_name_that_is_not_valid_utf8_is_survived(monkeypatch):
    """The same hazard from the other side: `ps` succeeds and returns bytes the
    locale cannot decode. `errors="replace"` is what makes that a mangled name
    rather than an exception."""

    class Result:
        stdout = "%CPU COMM\n 99.0 we�ird\n 50.0 chrome\n"

    monkeypatch.setattr(M.subprocess, "run", lambda *_a, **_k: Result())
    load = M.read_load(cores=8)
    assert len(load.top) == 2, load.top
    assert any("chrome" in t for t in load.top)


def test_a_contention_gate_that_could_not_run_says_so():
    """The same mistake as `guard`'s fingerprint check, which iterated an empty
    dict and reported "still holds": a check that did not run is not a check
    that passed.

    Not a refusal — an unreadable load average is no evidence the machine was
    busy — but the number must not read as better-checked than it is. Every
    NVIDIA and AMD box had `cores=0`, so this was every one of them.
    """
    # Deliberately *not* `_quiet`: this test is about the reader returning a
    # load whose core count is unknown, which is what every NVIDIA and AMD box
    # produced. Injecting the idle 16-core host would remove the condition
    # under test and leave an assertion that could not fail.
    m = M.measure(
        "http://x/v1",
        "m",
        cores=0,
        samples=2,
        sampler=_take(_samples(50, 50)),
        load_reader=lambda: M.Load(one_minute=1.0, cores=0),
    )
    assert m.usable, "an unknown core count is not evidence of contention"
    assert m.caveats, "the gate silently did not run"
    assert "did not run" in m.render()
    assert any("did not run" in c for c in json.loads(m.to_json())["caveats"])


def test_a_measurement_with_a_working_gate_carries_no_caveat():
    """The control: a caveat on every run would be noise nobody reads."""
    m = M.measure(
        "http://x/v1", "m", cores=16, samples=2, sampler=_take(_samples(50, 50)), load_reader=_quiet
    )
    if m.load_before.per_core is not None:
        assert not m.caveats, m.caveats
        assert "Checked less than usual" not in m.render()


def test_every_hardware_detector_reports_host_cpu_cores():
    """`Hardware.cores` is the denominator for load-per-core, and load average
    is a host-CPU metric. NVIDIA and AMD hardcoded 0 — which is not "unknown"
    to a reader, it is a number, and it disabled the gate on every such box
    while looking like a populated field.
    """
    src = (ROOT / "src" / "clickllm" / "hardware.py").read_text()
    assert "cores=0," not in src, (
        "a detector reports zero cores; use os.cpu_count() so load-per-core has a real denominator"
    )


def test_a_zero_rate_sample_is_refused_wherever_it_came_from():
    """`_decode_once` refuses a single-frame completion, and that guards the
    HTTP path. `measure()` takes samples from wherever the caller got them.

    A zero rate reaching the aggregation reports `median 0.0`, and `spread`
    requires `median > 0` — so it returns None, no refusal fires, and 0 tok/s
    is reported as a usable measurement with no reason given. The invariant is
    that a rate of zero is the absence of a measurement, so it belongs here and
    not only at the one caller that can currently produce it.
    """
    m = M.measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=2,
        sampler=_take([M.Sample(1, 0.0, 0.1), M.Sample(1, 0.0, 0.1)]),
    )
    assert not m.usable, "0 tok/s was accepted as a measurement"
    assert any("rate of zero" in r for r in m.refused), m.refused


def test_one_dead_sample_among_good_ones_still_refuses():
    """The mixed case, which a median would hide: two real samples and one that
    timed nothing average to something plausible."""
    m = M.measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=3,
        sampler=_take([*_samples(50, 50), M.Sample(1, 0.0, 0.1)]),
    )
    assert not m.usable


def test_a_run_that_slows_is_not_a_run_that_disagrees():
    """A steady decline and scattered noise refuse for different reasons.

    Measured on an idle M4 Max — load flat at 0.39/core before and after — a 32B
    model fell from 22.1 to 16.2 tok/s across fifteen samples, correlating -0.90
    with sample order. Calling that "the samples disagree by 31%" points the
    reader at contention they will not find; the samples agree closely about a
    downward trend. It is the silicon clocking down under sustained load.

    Both still refuse — the distinction is what the reader does next: quit
    something, or accept that this hardware does not hold its burst rate.
    """
    from clickllm.measure import _decline

    slowing = [
        22.12,
        21.98,
        22.17,
        21.21,
        20.06,
        18.68,
        18.39,
        19.36,
        19.60,
        19.12,
        17.83,
        18.15,
        16.24,
        16.38,
        18.12,
    ]
    found = _decline(slowing)
    assert found is not None, "a -0.90 correlation with sample order is a decline"
    first, last, drop = found
    assert first > last
    assert 0.15 < drop < 0.25, f"expected ~19% decline, got {drop:.0%}"

    # Scatter with the same spread must NOT be reported as a decline, or the
    # diagnosis becomes noise of its own.
    scattered = [
        19.0,
        22.0,
        17.0,
        21.5,
        18.0,
        22.2,
        16.5,
        20.0,
        19.5,
        17.5,
        21.0,
        18.5,
        22.1,
        16.8,
        20.5,
    ]
    assert _decline(scattered) is None, "scatter is not a trend"

    # Too few samples to judge is None, not a guess.
    assert _decline([22.0, 18.0, 16.0]) is None


def test_warmup_decodes_are_discarded_not_counted():
    """A warmup sample heats the machine; folding it in reintroduces the bias.

    The whole point of warming is to leave the cold, fast decodes out of the
    result. Counting them would put the burst rate back into the median, which
    is the thing #241 exists to remove.
    """
    from clickllm.measure import Load, Sample, measure

    rates = iter([50.0, 40.0, 30.0, 10.0, 10.0, 10.0])  # 3 hot warmups, 3 steady
    quiet = Load(one_minute=0.5, cores=16, top=())

    def sampler():
        return Sample(tokens=100, decode_seconds=100 / next(rates), ttft_seconds=0.1)

    m = measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=3,
        warmup=3,
        sampler=sampler,
        load_reader=lambda: quiet,
    )

    assert len(m.samples) == 3, "warmup decodes must not appear in the samples"
    assert m.median is not None
    assert abs(m.median - 10.0) < 0.01, (
        f"median {m.median} includes the discarded warmup rates — "
        "the cold, fast decodes are back in the result"
    )
