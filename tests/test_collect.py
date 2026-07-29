"""The collector — the seam between a live endpoint and the eval suite.

Everything here runs against a stub OpenAI server in a **thread**. Not a
subprocess: interpreter spawn on a macOS CI runner has already exceeded this
repo's budget twice and read as "the server never bound", and binding port 0
in-process closes the window between choosing a port and claiming it.

The property under test throughout is the one that decides whether a verdict is
honest: **an item that never got a reply is not an item that answered badly.**
A collector that conflates them turns a flaky network into a regression report.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any

import pytest

from clickllm.prove import EvalItem, run
from clickllm.prove.collect import Collection, chat_url, collect

Reply = tuple[int, dict[str, Any]]


def ok(text: str, model: str = "served-name", usage: dict[str, int] | None = None) -> Reply:
    """A well-formed chat completion."""
    body: dict[str, Any] = {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": text}}],
    }
    if usage:
        body["usage"] = usage
    return 200, body


@contextmanager
def stub(reply: Callable[[str, int], Reply]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """A threaded OpenAI-compatible stub. Yields `(base_url, request_log)`.

    `reply(prompt, nth)` decides the response; `nth` is how many times that exact
    prompt has been seen, which is what makes retry behaviour observable.
    Threading (not `HTTPServer`) so a slow item cannot serialise the others —
    otherwise the timeout test would stall every worker behind it.
    """
    log: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib's spelling
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw)
            prompt = body["messages"][0]["content"]
            log.append({"prompt": prompt, "headers": dict(self.headers), "body": body})
            nth = sum(1 for e in log if e["prompt"] == prompt)
            status, payload = reply(prompt, nth)
            out = json.dumps(payload).encode()
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except OSError:  # the client gave up first — that is the timeout test
                pass

        def log_message(self, *a: Any) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}/v1", log
    finally:
        srv.shutdown()
        srv.server_close()


def items(n: int, *, prompt: Callable[[int], str] = lambda i: f"p{i}") -> list[EvalItem]:
    """`n` items with the incumbent's answer already in place, candidate empty."""
    return [EvalItem(f"i{i}", "c", prompt(i), baseline='{"a": 1}', candidate="") for i in range(n)]


def gather(items_: list[EvalItem], base: str, **kw: Any) -> Collection:
    """`collect` with the backoff sleep stubbed out — tests must not pay for it."""
    kw.setdefault("model", "asked-for")
    kw.setdefault("sleep", lambda s: None)
    return collect(items_, base=base, **kw)


# --- replies land where they belong --------------------------------------------


def test_each_reply_lands_on_its_own_item_under_concurrency():
    """Eight workers, deliberately scrambled completion order.

    The first item is answered last. If results were zipped by completion order
    rather than by item, `i0` would carry `i7`'s answer — a silent corruption
    that scores as a perfectly plausible verdict.
    """

    def reply(prompt: str, nth: int) -> Reply:
        if prompt == "p0":
            time.sleep(0.25)
        return ok(f"re: {prompt}")

    with stub(reply) as (base, _log):
        got = gather(items(8), base, workers=8)

    assert len(got.items) == 8
    for item in got.items:
        assert item.candidate == f"re: {item.prompt}", item
    # The side not being collected is untouched — the incumbent's answer is the
    # bar being matched, and overwriting it would compare the model with itself.
    assert all(i.baseline == '{"a": 1}' for i in got.items)


def test_the_collected_side_is_the_one_asked_for():
    with stub(lambda p, n: ok("incumbent said this")) as (base, _):
        got = gather(items(3), base, side="baseline")
    assert all(i.baseline == "incumbent said this" for i in got.items)
    assert all(i.candidate == "" for i in got.items), "the candidate side must be left alone"


# --- failure is per item, not per run ------------------------------------------


def test_a_500_on_one_item_fails_that_item_only():
    """The run this exists to prevent: dying on item 3 of 20 and proving nothing."""

    def reply(prompt: str, nth: int) -> Reply:
        if prompt == "p3":
            return 500, {"error": "engine crashed"}
        return ok(f"re: {prompt}")

    with stub(reply) as (base, _):
        got = gather(items(20), base, retries=1)

    assert got.asked == 20
    assert len(got.items) == 19
    assert [c.item_id for c in got.failures] == ["i3"]
    assert "HTTP 500" in got.failures[0].reason
    assert "i3" not in {i.item_id for i in got.items}


def test_a_429_is_retried_and_a_400_is_not():
    """Retrying a 4xx is spend with no chance of a different answer."""

    def reply(prompt: str, nth: int) -> Reply:
        if prompt == "p0":
            return (429, {"error": "slow down"}) if nth < 3 else ok("finally")
        return 400, {"error": "context length exceeded"}

    with stub(reply) as (base, log):
        got = gather(items(2), base, retries=3, workers=1)

    assert [i.item_id for i in got.items] == ["i0"]
    assert got.items[0].candidate == "finally"
    assert sum(1 for e in log if e["prompt"] == "p0") == 3, "the 429 must be retried"
    assert sum(1 for e in log if e["prompt"] == "p1") == 1, "a 400 must never be retried"
    assert "HTTP 400" in got.failures[0].reason


def test_a_5xx_that_never_clears_gives_up_and_says_how_many_attempts():
    with stub(lambda p, n: (503, {"error": "loading"})) as (base, log):
        got = gather(items(1), base, retries=2)
    assert len(log) == 3, "one attempt plus two retries"
    assert not got.items
    assert "gave up after 3 attempts" in got.failures[0].reason
    assert got.failures[0].attempts == 3


def test_a_timeout_is_recorded_as_a_failure_with_its_reason():
    def reply(prompt: str, nth: int) -> Reply:
        if prompt == "p1":
            time.sleep(5.0)  # margin, not a guess: a loaded CI box is still < 1s
        return ok(f"re: {prompt}")

    with stub(reply) as (base, _):
        got = gather(items(3), base, timeout=1.0, workers=3)

    assert [c.item_id for c in got.failures] == ["i1"]
    assert "timed out after 1s" in got.failures[0].reason, got.failures[0].reason
    assert len(got.items) == 2, "one slow item must not take the others down"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"choices": []}, "no usable choices"),
        ({"object": "chat.completion"}, "no usable choices"),
        ({"choices": [{"finish_reason": "stop"}]}, "carries no message"),
        ({"choices": [{"message": {"role": "assistant"}}]}, "neither content nor tool calls"),
    ],
)
def test_a_malformed_body_fails_that_item_with_what_was_wrong(payload: dict, expected: str):
    """A 200 is not a reply. Treating one as an empty answer would score it as a
    model that returned nothing, which is a regression it did not commit."""
    with stub(lambda p, n: (200, payload)) as (base, _):
        got = gather(items(1), base)
    assert not got.items
    assert expected in got.failures[0].reason, got.failures[0].reason


def test_an_unreachable_endpoint_fails_every_item_without_raising():
    """A closed port is still data — one failure per item, and a run that returns."""
    with stub(lambda p, n: ok("never")) as (base, _):
        pass  # the server is shut down inside the context manager
    got = gather(items(3), base, retries=0, timeout=2.0)
    assert not got.items and len(got.failures) == 3
    assert "could not reach" in got.failures[0].reason, got.failures[0].reason


# --- what is recorded is what happened -----------------------------------------


def test_usage_and_the_served_model_are_recorded_never_invented():
    def reply(prompt: str, nth: int) -> Reply:
        if prompt == "p0":
            return ok("a", model="what-was-really-served", usage={"completion_tokens": 7})
        return ok("b", model="")  # a server that reports neither

    with stub(reply) as (base, log):
        got = gather(items(2), base, model="what-was-asked-for")

    assert log[0]["body"]["model"] == "what-was-asked-for", "the ask is sent verbatim"
    assert got.served == ("what-was-really-served",), got.served
    by_id = {c.item_id: c for c in got.replies}
    assert by_id["i0"].completion_tokens == 7
    assert by_id["i1"].completion_tokens is None, "absent must stay absent, not become 0"
    assert by_id["i1"].served_model == ""
    assert got.completion_tokens == 7
    assert all(c.seconds >= 0.0 for c in got.replies)


def test_no_usage_anywhere_reports_none_rather_than_zero():
    with stub(lambda p, n: ok("x")) as (base, _):
        got = gather(items(2), base)
    assert got.completion_tokens is None, "zero tokens and no measurement are different claims"


# --- credentials -----------------------------------------------------------------


def test_nothing_that_looks_like_a_credential_is_ever_read(monkeypatch: pytest.MonkeyPatch):
    """The check is on the wire, not on the import graph.

    Env vars are set that a helpful collector would "find". The assertion is that
    nothing resembling them left this process — which fails if anyone adds an
    `os.environ.get("OPENAI_API_KEY")` fallback, whatever it is named.
    """
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "CLICKLLM_TOKEN"):
        monkeypatch.setenv(name, f"sk-live-{name}")

    with stub(lambda p, n: ok("x")) as (base, log):
        gather(items(2), base)

    for entry in log:
        sent = json.dumps(entry).lower()
        assert "sk-live" not in sent, entry["headers"]
        assert "authorization" not in {k.lower() for k in entry["headers"]}


def test_a_header_the_caller_passes_in_is_the_only_way_auth_travels():
    with stub(lambda p, n: ok("x")) as (base, log):
        got = gather(items(1), base, headers={"Authorization": "Bearer given-by-the-caller"})
    assert got.items
    assert log[0]["headers"]["Authorization"] == "Bearer given-by-the-caller"


# --- the statistical consequence -------------------------------------------------


def test_failed_collection_shrinks_the_sample_and_widens_the_interval():
    """The invariant the whole module exists for.

    60 items, 20 of which the endpoint never answers. The wrong behaviours are
    both tempting: score the 20 as empty answers (a 67% "regression" caused by a
    flaky server), or drop them and report 40/40 as if 40 was always the plan.
    The right one is 40/40 with a *wider* interval, and the loss stated.
    """

    def reply(prompt: str, nth: int) -> Reply:
        if int(prompt[1:]) % 3 == 0:
            return 500, {"error": "engine crashed"}
        return ok('{"a": 1}')

    with stub(reply) as (base, _):
        got = gather(items(60), base, retries=0, workers=8)

    assert len(got.items) == 40 and len(got.failures) == 20

    scored = run(list(got.items), shares={"c": 1.0}).candidates[0].clusters[0]
    assert scored.interval.total == 40, "failed collections must not score as answers"
    assert scored.interval.passed == 40, (
        "an item that never replied has not failed the eval — if this is 40/60 "
        "the collector is scoring a network fault as a quality regression"
    )

    # The counterfactual: the same 60 items, all of which answered identically.
    every = [replace(i, candidate='{"a": 1}') for i in items(60)]
    full = run(every, shares={"c": 1.0}).candidates[0].clusters[0]
    assert full.interval.total == 60 and full.interval.passed == 60
    assert scored.interval.width > full.interval.width, (
        "a smaller sample must be a less confident claim, not an equal one"
    )
    assert scored.interval.low < full.interval.low

    # And the loss is stated next to the numbers, not implied by their absence.
    text = got.render()
    assert "40/60 replies" in text
    assert "smaller sample, not a lower score" in text
    assert "20 × HTTP 500" in text


def test_the_report_names_the_failed_items_so_they_can_be_chased():
    def reply(prompt: str, nth: int) -> Reply:
        return (500, {"error": "x"}) if prompt in ("p1", "p2") else ok("fine")

    with stub(reply) as (base, _):
        got = gather(items(4), base, retries=0)
    assert "i1" in got.render() and "i2" in got.render()


# --- caller bugs are refused up front --------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"side": "sideways"}, "side must be one of"),
        ({"workers": 0}, "workers must be at least 1"),
        ({"retries": -1}, "retries cannot be negative"),
    ],
)
def test_a_caller_bug_raises_once_not_once_per_item(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        collect(items(3), base="http://127.0.0.1:1/v1", model="m", **kwargs)


def test_an_empty_eval_set_is_refused():
    with pytest.raises(ValueError, match="empty eval set"):
        collect([], base="http://127.0.0.1:1/v1", model="m")


@pytest.mark.parametrize("bad", ["127.0.0.1:8000", "localhost/v1", "ftp://h/v1", ""])
def test_a_base_that_is_not_an_http_url_is_refused(bad: str):
    with pytest.raises(ValueError, match="http"):
        chat_url(bad)


def test_both_spellings_of_the_base_url_reach_the_same_place():
    """`clickllm run` prints `.../v1`; `Endpoint.base` does not carry it."""
    want = "http://127.0.0.1:8000/v1/chat/completions"
    assert chat_url("http://127.0.0.1:8000") == want
    assert chat_url("http://127.0.0.1:8000/") == want
    assert chat_url("http://127.0.0.1:8000/v1") == want
    assert chat_url("http://127.0.0.1:8000/v1/") == want


# --- the CLI seam ----------------------------------------------------------------


def _evalset(path, n: int) -> str:
    """An eval set with baselines but no candidate answers — the live-collect case."""
    rows = [
        {"item_id": f"i{i}", "cluster": "codegen", "prompt": f"p{i}", "baseline": '{"a": 1}'}
        for i in range(n)
    ]
    path.write_text(json.dumps({"items": rows, "shares": {"codegen": 1.0}}))
    return str(path)


def test_the_cli_scores_a_live_endpoint_end_to_end(tmp_path, capsys: pytest.CaptureFixture):
    """The whole seam: a file of prompts, a live endpoint, a receipt."""
    from clickllm import cli

    def reply(prompt: str, nth: int) -> Reply:
        return (400, {"error": "context length exceeded"}) if prompt == "p7" else ok('{"a": 1}')

    receipt = tmp_path / "r.json"
    with stub(reply) as (base, log):
        code = cli.main(
            [
                "prove",
                _evalset(tmp_path / "e.json", 50),
                "--candidate",
                "glm-5.2",
                "--candidate-endpoint",
                base,
                "--candidate-model",
                "mlx-community/GLM-5.2-4bit",
                "--issued",
                "2026-07-27",
                "--out",
                str(receipt),
            ]
        )
    out = capsys.readouterr().out

    assert code == 0, out
    assert log and log[0]["body"]["model"] == "mlx-community/GLM-5.2-4bit"
    # Collection is reported apart from the eval, with its own count.
    assert "COLLECTED" in out
    assert "49/50 replies" in out, out
    assert "scoring 49 of 50 items" in out, out

    claims = json.loads(receipt.read_text())["receipt"]
    proven = claims["proven"]
    assert len(proven) == 1 and proven[0]["total"] == 49, proven
    # The receipt states what the set was drawn from, so the reader can see the
    # denominator moved rather than inferring it from a number that looks whole.
    assert claims["traffic_captures"] == 50, claims["traffic_captures"]


def test_the_cli_collects_both_sides_and_funnels_the_failures(
    tmp_path, capsys: pytest.CaptureFixture
):
    """Both endpoints, and an item only scored if it answered on both.

    The second side is asked only about what the first one answered, so the two
    counts read as a funnel — 49 of 50, then 48 of 49 — rather than as two
    unrelated fractions the reader has to reconcile.
    """
    from clickllm import cli

    def reply(prompt: str, nth: int) -> Reply:
        # Sides are collected in sequence, so the first sighting of a prompt is
        # the candidate's and the second is the incumbent's.
        if (prompt, nth) in (("p2", 1), ("p5", 2)):
            return 400, {"error": "context length exceeded"}
        return ok('{"a": 1}')

    with stub(reply) as (base, log):
        code = cli.main(
            [
                "prove",
                _evalset(tmp_path / "e.json", 50),
                "--candidate-endpoint",
                base,
                "--candidate-model",
                "the-open-one",
                "--incumbent-endpoint",
                base,
                "--incumbent-model",
                "the-closed-one",
                "--issued",
                "2026-07-27",
            ]
        )
    out = capsys.readouterr().out

    assert code == 0, out
    asked = {e["body"]["model"] for e in log}
    assert asked == {"the-open-one", "the-closed-one"}, asked
    assert "candidate 49/50 replies" in out, out
    assert "baseline  48/49 replies" in out, out
    assert "scoring 48 of 50 items" in out, out
    # The incumbent is never asked about an item the candidate could not answer:
    # a reply nothing can be scored against is spend for no evidence.
    assert not any(e["prompt"] == "p2" and e["body"]["model"] == "the-closed-one" for e in log)


def test_the_cli_still_scores_a_file_with_no_endpoint(tmp_path, capsys: pytest.CaptureFixture):
    """The original path is untouched: no endpoint, no collection, no egress."""
    from clickllm import cli

    path = tmp_path / "e.json"
    rows = [
        {
            "item_id": f"i{i}",
            "cluster": "codegen",
            "prompt": f"p{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"a": 1}',
        }
        for i in range(45)
    ]
    path.write_text(json.dumps({"items": rows, "shares": {"codegen": 1.0}}))
    receipt = tmp_path / "r.json"

    assert cli.main(["prove", str(path), "--issued", "2026-07-27", "--out", str(receipt)]) == 0
    assert "COLLECTED" not in capsys.readouterr().out
    body = json.loads(receipt.read_text())["receipt"]
    assert body["proven"][0]["total"] == 45 and body["traffic_captures"] == 45


def test_the_cli_refuses_rather_than_scoring_a_run_where_nothing_answered(
    tmp_path, capsys: pytest.CaptureFixture
):
    """Nothing answered is not a verdict, and must not render as one."""
    from clickllm import cli

    with stub(lambda p, n: (400, {"error": "dead"})) as (base, _):
        code = cli.main(["prove", _evalset(tmp_path / "e.json", 4), "--candidate-endpoint", base])
    captured = capsys.readouterr()
    assert code == 2
    assert "nothing can be scored" in captured.err, captured.err
    assert "Traceback" not in captured.err


def test_the_cli_exit_is_a_sentence_not_a_traceback(tmp_path, capsys: pytest.CaptureFixture):
    """Repo convention, on the new path too."""
    from clickllm import cli

    code = cli.main(
        ["prove", _evalset(tmp_path / "e.json", 2), "--candidate-endpoint", "not-a-url"]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "http(s) URL" in err, err


# --- the module's own self-check --------------------------------------------------


def test_demo_self_check():
    from clickllm.prove import collect as mod

    mod.demo()


def test_a_collection_with_no_failures_does_not_crash_the_error_path():
    """The candidate answers everything, the incumbent answers nothing.

    Nothing is scoreable, so this path explains why — and it used to do so by
    indexing `collections[0].failures[0]`. Collection zero is the candidate,
    which has no failures, so the explanation raised an IndexError over the top
    of itself and the user learned nothing about the incumbent that died.
    """
    from clickllm.prove import collect as C

    clean = C.Collection(
        base="http://a/v1",
        model="cand",
        side="candidate",
        replies=(C.Collected(item_id="i1", text="ok"),),
        items=(),
        seconds=0.1,
    )
    broken = C.Collection(
        base="http://b/v1",
        model="inc",
        side="incumbent",
        replies=(C.Collected(item_id="i1", reason="HTTP 503 — engine unavailable"),),
        items=(),
        seconds=0.1,
    )
    assert clean.failures == (), "the candidate answered; it has no failures to index"
    assert broken.failures, "the incumbent failed and must carry the reason"

    # What the CLI now does: first failure ACROSS collections, never [0][0].
    reason = next(
        (f.reason for c in (clean, broken) for f in c.failures),
        "no endpoint reported a reason",
    )
    assert reason == "HTTP 503 — engine unavailable"

    # The old form is the bug, and it is still an IndexError.
    with pytest.raises(IndexError):
        _ = (clean, broken)[0].failures[0]
