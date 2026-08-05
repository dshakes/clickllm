def test_run_tears_the_engine_down_on_sigterm_not_just_ctrl_c(monkeypatch):
    """Measured against a real run of published 0.1.8: SIGTERM killed clickllm
    and left `mlx_lm` serving on its port, holding the device.

    `Endpoint.wait()` catches KeyboardInterrupt, which SIGINT raises — but
    SIGTERM does not raise anything. Python's default SIGTERM handler
    terminates the process outright, so the teardown never ran. That is the
    signal systemd, `docker stop` and a Kubernetes eviction all send.

    Interactive Ctrl-C hid it: a tty signals the whole foreground process
    group, so the engine got its own SIGINT from the terminal rather than from
    us. Every non-tty stop reaches clickllm alone.
    """
    import signal as _signal

    from clickllm import cli

    stopped = []

    class FakeEndpoint:
        base, model, pid, ready_seconds = "http://127.0.0.1:1", "m", 1, 0.0

        def stop(self, grace: float = 10.0):
            stopped.append(True)
            return ()

        def wait(self):
            # Whatever handlers are installed while blocked here are what a
            # signal would actually find.
            installed = {s: _signal.getsignal(s) for s in (_signal.SIGINT, _signal.SIGTERM)}
            self.seen = installed
            return 0

    ep = FakeEndpoint()
    before = {s: _signal.getsignal(s) for s in (_signal.SIGINT, _signal.SIGTERM)}
    assert cli._serve_until_signalled(ep) == 0

    for sig in (_signal.SIGINT, _signal.SIGTERM):
        handler = ep.seen[sig]
        assert callable(handler), f"{sig.name} had no handler while serving"
        assert handler not in (_signal.SIG_DFL, _signal.SIG_IGN), f"{sig.name} left at default"

    # Handlers are restored afterwards — a CLI that leaks them changes how the
    # rest of the process responds to signals.
    assert {s: _signal.getsignal(s) for s in (_signal.SIGINT, _signal.SIGTERM)} == before

    # And the handler actually stops the engine.
    ep.seen[_signal.SIGTERM](int(_signal.SIGTERM), None)
    assert stopped, "the SIGTERM handler did not stop the engine"

    # `cmd_run` must go through it. Testing the helper alone would pass for a
    # cmd_run that calls `ep.wait()` directly — which is exactly the code this
    # replaced, so the test has to assert the wiring, not just the part.
    import inspect

    body = inspect.getsource(cli.cmd_run)
    assert "_serve_until_signalled(ep)" in body, "cmd_run bypasses the teardown"
    assert "return ep.wait()" not in body


def test_a_runtime_error_is_a_sentence_and_exit_two_not_a_traceback(monkeypatch, capsys):
    """`main()` caught KeyError/ValueError/OSError — not RuntimeError.

    `desktop.install()` documents and raises `RuntimeError` on any unsupported
    platform, so `clickllm desktop` on Linux printed a traceback. The convention
    is stated two lines below the handler it escaped: a bad path or a refusal is
    a nonzero exit with a sentence, never a traceback.
    """
    from clickllm import cli, desktop

    def boom(**_):
        raise RuntimeError("this platform has no desktop entry point")

    monkeypatch.setattr(desktop, "install", boom)
    assert cli.main(["desktop"]) == 2
    assert "this platform has no desktop entry point" in capsys.readouterr().err


def test_explain_honours_json_instead_of_silently_ignoring_it(capsys):
    """`--explain` returned before the `--json` branch, so the flag did nothing.

    A flag that silently does nothing is worse than one that errors: a script
    consuming `fit --explain X --json` parses prose as JSON.
    """
    import json as _json

    from clickllm import cli

    assert cli.main(["fit", "--explain", "llama-3.1-8b", "--json"]) in (0, 1)
    payload = _json.loads(capsys.readouterr().out)
    assert payload["model"] == "llama-3.1-8b"
    assert payload["total_bytes"] > 0
    assert "explain" in payload, "the human arithmetic must still be reachable"
