"""The workbench's error contract, driven over a real socket.

`demo()` calls the route functions directly, so it cannot see what `do_GET`
does with what they raise — and what it did with `FileNotFoundError` was drop
the connection with no response at all. These tests go through HTTP for that
reason: asserting on the handler's helpers would have passed either way.
"""

from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from clickllm import ui

CATALOGUE_ROUTES = ("/api/catalog", "/api/fit", "/api/where?model=qwen3-32b")


@pytest.fixture
def workbench():
    """A live workbench on an ephemeral port; yields a GET returning (status, body)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def get(path: str) -> tuple[int, bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    try:
        yield get
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("path", CATALOGUE_ROUTES)
def test_a_missing_catalogue_file_answers_rather_than_hanging_up(workbench, monkeypatch, path):
    # A one-character typo in $CLICKLLM_CATALOG made every catalogue-backed
    # route close the socket with no HTTP response, for the life of the process.
    monkeypatch.setenv("CLICKLLM_CATALOG", "/tmp/clickllm-does-not-exist.json")
    status, body = workbench(path)
    assert status == 500
    assert b"catalogue file not found" in body


def test_an_unknown_model_is_still_a_400_not_a_500(workbench):
    # The catch-all must not swallow the errors that already had a contract.
    status, body = workbench("/api/where?model=no-such-model")
    assert status == 400
    assert b"no-such-model" in body


def test_the_cost_figure_travels_with_the_throughput_it_divides_by(workbench):
    import json

    status, body = workbench("/api/where?model=qwen3-32b&concurrency=8")
    assert status == 200
    priced = [p for p in json.loads(body)["placements"] if p["usd_per_mtok"]]
    assert priced, "no placement carried a cost figure"
    for p in priced:
        assert p["aggregate_tokens_per_sec"], p["name"]
        # Single-stream and aggregate are different numbers at concurrency 8;
        # showing only the first beside a cost derived from the second is the
        # verdict-without-its-basis this field exists to prevent.
        assert p["aggregate_tokens_per_sec"] > p["tokens_per_sec"], p["name"]
