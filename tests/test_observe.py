"""The chain, joined: observe → distill → prove.

Three finished components with nothing between them was the actual state of
this repo — a gateway with its own test suite that no command could start, a
clustering module that nothing fed, and a prover that read a file no one wrote.
Each half passed its own tests the whole time.

So these tests are mostly about the joins, which is where all four of the
defects found while building this lived: a flag the binary never had, an import
of a function `atomicio` does not export, a tool-call shape the grader could not
read, and a field the recorder never wrote.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clickllm import observe

ROOT = Path(__file__).resolve().parents[1]
GATEWAY_SRC = ROOT / "clickllm-gateway" / "src" / "main.rs"


# --- the joins, checkable without building anything ------------------------------


def test_every_flag_we_pass_is_a_flag_the_gateway_declares():
    """The first version passed `--listen`, which the gateway has never had.

    The argv looked entirely reasonable, every assertion about it passed, and
    the process refused to start. A list of strings is not an interface; the
    binary's own parser is.
    """
    argv = observe.gateway_argv(
        Path("/usr/bin/true"),
        "http://up",
        port=9,
        candidate="http://cand",
        capture=Path("/tmp/c"),
        key=Path("/tmp/k"),
    )
    usage = GATEWAY_SRC.read_text()
    for flag in {a for a in argv if a.startswith("--")}:
        assert f'"{flag}"' in usage, f"the gateway has no {flag}"


def test_nothing_in_the_launch_path_can_move_traffic():
    """Invariant 8, at the surface that would be easiest to breach: a flag.

    The gateway refuses a startup `--percent` on its own, and this asserts the
    Python side never tries — two independent refusals, because the one that
    matters is whichever one is still there next year.
    """
    for kwargs in ({}, {"candidate": "http://c"}, {"no_capture": True}):
        argv = observe.gateway_argv(Path("/usr/bin/true"), "http://u", **kwargs)
        assert not any(a in argv for a in ("--percent", "--cut", "--promote")), argv


def test_a_named_binary_that_does_not_exist_is_an_error_not_a_fallback(monkeypatch, tmp_path):
    """Falling through to `PATH` would start a *different* process than the one
    the user named, and put it in their request path."""
    monkeypatch.setenv("CLICKLLM_GATEWAY_BIN", str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="not a file"):
        observe.find_gateway()


def test_the_home_directory_is_overridable():
    """A log of production prompts should be placeable on a volume the user
    chose, not only in their home directory."""
    import os

    old = os.environ.get("CLICKLLM_HOME")
    try:
        os.environ["CLICKLLM_HOME"] = "/tmp/somewhere-else"
        assert observe.state_dir() == Path("/tmp/somewhere-else")
    finally:
        os.environ.pop("CLICKLLM_HOME", None)
        if old is not None:
            os.environ["CLICKLLM_HOME"] = old


# --- distill's output is what prove reads ----------------------------------------


def _rows(n_text: int = 12, n_tool: int = 3) -> list[dict]:
    rows = [
        {
            "request_id": f"r{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"summarise document {i}"}],
            "response": "a summary",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [],
            "tool_calls": [],
            "response_format": None,
        }
        for i in range(n_text)
    ]
    rows += [
        {
            "request_id": f"t{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"refund order {i}"}],
            "response": "",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [{"function": {"name": "refund"}}],
            "tool_calls": ["refund"],
            "response_format": "json_object",
        }
        for i in range(n_tool)
    ]
    return rows


def test_the_eval_set_prove_reads_is_the_eval_set_distill_writes():
    """The seam that crashed: distill emitted bare tool names and the grader
    read `.get` on them, four frames down, killing the whole run."""
    from clickllm.prove import EvalItem
    from clickllm.prove.graders import ToolChoice

    doc, _ = observe.distill(_rows(), budget=50, min_per_cluster=2)
    tool_rows = [i for i in doc["items"] if i["baseline_tool_calls"]]
    assert tool_rows, "the fixture has tool calls; the eval set must carry them"

    item = EvalItem(
        item_id="x",
        cluster="c",
        prompt=tool_rows[0]["prompt"],
        baseline=tool_rows[0]["baseline"],
        candidate="",
        baseline_tool_calls=tuple(tool_rows[0]["baseline_tool_calls"]),
        candidate_tool_calls=tuple(tool_rows[0]["baseline_tool_calls"]),
    )
    assert ToolChoice().grade(item).outcome.name == "PASS"


def test_distill_tells_you_the_endpoint_is_required_not_just_the_label(
    monkeypatch, tmp_path, capsys
):
    """`--candidate <model>` alone is a label with nothing behind it — collection
    only happens with `--candidate-endpoint`. Printing the label-only form as
    "next" leads straight to scoring blank candidate answers as a real proof."""
    from clickllm import cli, core

    log, key = tmp_path / "captures.log", tmp_path / "capture.key"
    log.write_bytes(b"not read, only checked for existence")
    key.write_bytes(b"not read, only checked for existence")
    monkeypatch.setattr(core, "available", lambda: True)
    monkeypatch.setattr(core, "read_captures", lambda *a, **k: _rows())

    out = tmp_path / "evalset.json"
    code = cli.main(
        [
            "distill",
            "--capture",
            str(log),
            "--key",
            str(key),
            "--out",
            str(out),
            "--budget",
            "50",
            "--min-per-cluster",
            "2",
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "--candidate-endpoint" in printed, (
        "the next-step hint must include --candidate-endpoint, or following it "
        "silently proves nothing"
    )


def test_a_hand_written_tool_call_does_not_kill_the_run():
    """An eval set is a file — from distill, a hand edit, or another tool. A
    bare string is an obvious way to write one by hand, and invariant 7 applies
    to eval sets as much as to the corpus."""
    from clickllm.prove.graders import _call_names

    assert _call_names(("refund",)) == ["refund"]
    assert _call_names(({"name": "refund"},)) == ["refund"]
    assert _call_names(({"function": {"name": "refund"}},)) == ["refund"]
    assert _call_names((None, 7, {}, {"function": "not a dict"})) == []


def test_a_hand_written_tool_call_survives_the_full_grader_stack():
    """`_call_names` accepting bare strings is not enough — ToolArgs pulls each
    call apart looking for `arguments` too, four frames past `_call_names`, and
    used to raise `AttributeError: 'str' object has no attribute 'get'` on
    exactly the input the sibling ToolChoice fix claims to handle safely."""
    from clickllm.prove import EvalItem
    from clickllm.prove.graders import ToolArgs, ToolChoice

    item = EvalItem(
        item_id="x",
        cluster="c",
        prompt="p",
        baseline="ok",
        candidate="ok",
        baseline_tool_calls=("refund",),
        candidate_tool_calls=("refund",),
    )
    assert ToolChoice().grade(item).outcome.name == "PASS"
    assert ToolArgs().grade(item).outcome.name == "PASS"


def test_a_cluster_claims_its_share_of_traffic_not_of_the_sample():
    """Twelve summaries and three tool calls, sampled two and two, must still
    report 80/20 — a small cluster sampled up to the floor must not thereby
    claim a larger slice of the verdict than it holds of the workload."""
    doc, _ = observe.distill(_rows(), budget=4, min_per_cluster=2)
    tool_key = next(k for k, v in doc["names"].items() if "tool" in v)
    assert abs(doc["shares"][tool_key] - 0.2) < 1e-9, doc["shares"]


def test_clusters_that_got_nothing_are_named_rather_than_omitted():
    """A cluster sampled to zero still has a key in `sampled` and reads as
    covered. Silence there is a proof that looks complete and is not."""
    doc, report = observe.distill(_rows(), budget=1, min_per_cluster=0)
    assert report.uncovered or report.items <= 1
    if report.uncovered:
        assert all(u in report.render() for u in report.uncovered)
        assert "nothing about that traffic" in report.render()


def test_the_file_says_its_baselines_are_not_ground_truth():
    """The claim a reader will otherwise make for us. The baseline is the
    incumbent's reply — what a candidate must match, which is weaker."""
    doc, _ = observe.distill(_rows(), budget=50)
    assert "not ground truth" in doc["provenance"]["note"]


# --- end to end, when this machine can run it ------------------------------------


def _extension_available() -> bool:
    from clickllm import core

    return core.available()


def _gateway_binary() -> Path | None:
    for profile in ("release", "debug"):
        p = ROOT / "target" / profile / "clickllm-gateway"
        if p.is_file():
            return p
    found = shutil.which("clickllm-gateway")
    return Path(found) if found else None


UPSTREAM = """
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = " ".join(m.get("content") or "" for m in body.get("messages", []))
        if "refund" in text:
            msg = {"role":"assistant","content":None,"tool_calls":[
                {"id":"c1","type":"function","function":{"name":"refund","arguments":"{}"}}]}
        else:
            msg = {"role":"assistant","content":"Summary. Contact ada@example.com."}
        out = json.dumps({"choices":[{"message":msg}],
                          "usage":{"prompt_tokens":9,"completion_tokens":12}}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(out))); self.end_headers(); self.wfile.write(out)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
"""


@pytest.mark.skipif(not _extension_available(), reason="needs the compiled extension")
@pytest.mark.skipif(_gateway_binary() is None, reason="needs a built gateway binary")
def test_the_whole_chain_runs_on_one_machine(tmp_path):
    """observe → real traffic → distill → an eval set `prove` can read.

    The acceptance test for the whole thing, and the first check that runs the
    parts *together*: every join between them was broken at some point while
    each end's own tests stayed green.
    """
    import socket

    from clickllm import core

    def two_free_ports() -> tuple[int, int]:
        # Both sockets held open at once, then released together. Asking twice
        # in sequence can hand back the same port the first call just freed —
        # the OS reuses ephemeral ports eagerly — and the two processes then
        # race for one port, with the loser silently dead. That is what this
        # test did on macOS CI: "upstream never accepted a connection".
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0))
            b.bind(("127.0.0.1", 0))
            return int(a.getsockname()[1]), int(b.getsockname()[1])

    up_port, gw_port = two_free_ports()
    assert up_port != gw_port
    up_src = tmp_path / "upstream.py"
    up_src.write_text(UPSTREAM.replace("PORT", str(up_port)))
    # Captured, not discarded. A child that fails to bind writes a traceback
    # and exits; with the pipes thrown away the only symptom was a timeout
    # thirty seconds later that said nothing about why.
    upstream = subprocess.Popen(
        [sys.executable, str(up_src)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    log = tmp_path / "captures.log"
    key = tmp_path / "capture.key"
    gateway = subprocess.Popen(
        observe.gateway_argv(
            _gateway_binary(), f"http://127.0.0.1:{up_port}", port=gw_port, capture=log, key=key
        ),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    def wait_for(port: int, what: str) -> None:
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                time.sleep(0.1)
            for proc, label in ((upstream, "upstream"), (gateway, "gateway")):
                if label == what and proc.poll() is not None:
                    out, err = proc.communicate()
                    pytest.fail(
                        f"{what} exited {proc.returncode} before listening on "
                        f"{port}:\n{err.decode(errors='replace')[-800:]}"
                    )
        pytest.fail(f"{what} never accepted a connection on {port}")

    try:
        # Both, not just the gateway. Waiting only on the gateway passed on a
        # laptop where the upstream had been running for minutes and 502'd in
        # CI, where it is spawned fresh: the gateway binds immediately and
        # forwards to a socket nobody is listening on yet.
        wait_for(up_port, "upstream")
        wait_for(gw_port, "gateway")

        import urllib.request

        def ask(content: str, tools: bool) -> None:
            body: dict = {"model": "gpt-5", "messages": [{"role": "user", "content": content}]}
            if tools:
                body["tools"] = [{"type": "function", "function": {"name": "refund"}}]
                body["response_format"] = {"type": "json_object"}
            req = urllib.request.Request(
                f"http://127.0.0.1:{gw_port}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"content-type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10).read()

        for i in range(6):
            ask(f"summarise document {i} for ada@example.com", tools=False)
        for i in range(2):
            ask(f"refund order {i}", tools=True)

        for _ in range(100):
            if log.exists() and len(core.read_captures(str(log), key.read_bytes())) >= 8:
                break
            time.sleep(0.1)

        rows = core.read_captures(str(log), key.read_bytes())
        assert len(rows) >= 8, f"only {len(rows)} captures landed"

        # NFR-3, checked at the far end of the pipeline rather than at the
        # write path that guarantees it. Both matter: the guarantee is
        # structural, and this is what a user would actually look at.
        raw = log.read_bytes()
        assert b"ada@example.com" not in raw
        assert b"summarise document" not in raw

        doc, report = observe.distill(rows, budget=50, min_per_cluster=2)
        assert report.clusters == 2, f"tool and text traffic must not merge: {report.labels}"
        assert "ada@example.com" not in json.dumps(doc), "redaction must survive to the eval set"
        assert any(i["baseline_tool_calls"] for i in doc["items"])

        out = observe.write_eval_set(doc, tmp_path / "evalset.json")
        reread = json.loads(out.read_text())
        assert reread["items"] == doc["items"]
    finally:
        for p in (gateway, upstream):
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def test_the_cli_exposes_both_halves_and_neither_can_promote():
    """The read-only boundary the MCP surface already enforces, applied to the
    two commands that touch the datapath."""
    src = (ROOT / "src" / "clickllm" / "cli.py").read_text()
    assert 'sub.add_parser("observe"' in src
    assert 'sub.add_parser("distill"' in src
    observe_block = re.search(r'ob = sub\.add_parser\("observe".*?ob\.set_defaults', src, re.S)
    assert observe_block
    for verb in ("percent", "cutover", "promote", "advance"):
        assert verb not in observe_block.group(0), f"observe must not expose --{verb}"
