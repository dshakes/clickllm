"""Structured events for the half of this product that decides things.

The Rust datapath runs every fallible operation inside a `tracing` span carrying
model, runtime and path. The Python control plane — the half that solves, sizes,
collects, judges and issues receipts — emitted **nothing**. A `grep` for
logging, tracing or otel across `src/onpar/` returned no hits at all. So a
prove run that took forty minutes and refused at the end left no record of which
part was slow or which endpoint was the one retrying.

Three constraints shaped this, and each ruled out the obvious answer.

**Zero runtime dependencies.** `onpar fit` must work under `uvx` with nothing
installed (NFR, and a promise worth more than any library). That rules out
`opentelemetry`, `structlog`, and everything else on PyPI. Standard `logging`
does what is needed.

**Zero egress by default (NFR-2).** Captured traffic is the most sensitive data
a customer has, and an event carrying a cluster name or a model id is derived
from it. There is no exporter here, no collector endpoint, and no configuration
that adds one — events go to stderr or to a local file, and that is the entire
set of destinations. This is the one place where "industry best practice" and
this product's promises genuinely conflict, and the promise wins.

**Silent unless asked.** A tool that starts printing spans at people is a tool
they turn off. Nothing is emitted until `ONPAR_LOG` is set.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["configure", "event", "span"]

LOGGER = "onpar"

#: Where events go, and the only thing that turns them on. `ONPAR_LOG=debug`
#: for stderr, or `ONPAR_LOG=/path/to/file.log` for a file. There is
#: deliberately no value that means "send them somewhere" — see the module
#: docstring.
ENV = "ONPAR_LOG"

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING}


def _fields(pairs: dict[str, Any]) -> str:
    """`key=value` in a stable order, skipping what was not supplied.

    Sorted rather than insertion-ordered so two runs of the same operation
    produce diffable lines. `None` is dropped rather than rendered: "the judge
    was not used" and "the judge was `None`" read differently to a human
    scanning a log, and only the first is a fact.
    """
    return " ".join(f"{k}={v}" for k, v in sorted(pairs.items()) if v is not None)


def configure(spec: str | None = None) -> bool:
    """Turn events on from `ONPAR_LOG`, and report whether they are on.

    `spec` is a level name (`debug`, `info`, `warn`) or a path to append to.
    Anything else is ignored rather than raising: a typo in an environment
    variable must not take down a sizing run, and this is diagnostics.

    Idempotent — calling it twice does not double every line, which is the
    classic way a CLI ends up printing everything in duplicate.
    """
    spec = (spec if spec is not None else os.environ.get(ENV, "")).strip()
    log = logging.getLogger(LOGGER)
    if not spec:
        return False
    if log.handlers:
        return True

    level = _LEVELS.get(spec.lower())
    if level is not None:
        handler: logging.Handler = logging.StreamHandler()
    else:
        # A path. Failing to open it must not fail the run — if the directory is
        # unwritable the answer is "no diagnostics", never "no answer".
        try:
            handler = logging.FileHandler(spec, encoding="utf-8")
        except OSError:
            return False
        level = logging.DEBUG

    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)
    # Not to the root logger: a library that reconfigures logging for the whole
    # process is a library people vendor around.
    log.propagate = False
    return True


def event(name: str, **fields: Any) -> None:
    """One thing happened. Free when events are off."""
    log = logging.getLogger(LOGGER)
    if not log.handlers:
        return
    log.info("%s %s", name, _fields(fields))


@contextmanager
def span(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time an operation and record how it ended.

    Yields a dict the body can add fields to as it learns them — how many items
    were collected, which model was chosen — so the closing line carries the
    result rather than only the question.

    An exception is recorded and **re-raised**. A span that swallowed would turn
    a diagnostic into a behaviour change, which is the one thing observability
    must never do.
    """
    log = logging.getLogger(LOGGER)
    if not log.handlers:
        # Still yields a dict, so the body's `extra["items"] = n` works
        # identically whether or not anyone is listening. A span that returned
        # None when disabled would make every call site need a branch, and one
        # of them would get it wrong.
        yield {}
        return

    extra: dict[str, Any] = {}
    started = time.monotonic()
    log.debug("%s.start %s", name, _fields(fields))
    try:
        yield extra
    except BaseException as e:
        log.warning(
            "%s.failed %s",
            name,
            _fields(
                {
                    **fields,
                    **extra,
                    "ms": round((time.monotonic() - started) * 1000),
                    "error": type(e).__name__,
                }
            ),
        )
        raise
    else:
        log.info(
            "%s.ok %s",
            name,
            _fields({**fields, **extra, "ms": round((time.monotonic() - started) * 1000)}),
        )


def demo() -> None:
    """Self-check: silent by default, structured when asked, never swallowing."""
    import io

    log = logging.getLogger(LOGGER)
    saved = list(log.handlers)
    log.handlers.clear()

    try:
        # Off by default. Not a no-op *by accident* — the span still yields a
        # usable dict, so call sites do not branch.
        assert not configure("")
        with span("fit", model="x") as extra:
            extra["chosen"] = "qwen3-32b"
        event("nothing", a=1)

        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
        log.propagate = False

        with span("fit", model="x") as extra:
            extra["chosen"] = "qwen3-32b"
        out = buf.getvalue()
        assert "fit.start" in out and "fit.ok" in out, out
        assert "model=x" in out and "chosen=qwen3-32b" in out, out
        assert "ms=" in out, out

        # A field that is None is absent, not rendered as the word None.
        buf.truncate(0), buf.seek(0)
        event("prove", judge=None, items=12)
        assert "judge" not in buf.getvalue(), buf.getvalue()
        assert "items=12" in buf.getvalue()

        # Failure is recorded and re-raised. Observability must not change what
        # the program does.
        buf.truncate(0), buf.seek(0)
        try:
            with span("prove", set_id="abc"):
                raise ValueError("nope")
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError("the span swallowed the exception")
        assert "prove.failed" in buf.getvalue() and "error=ValueError" in buf.getvalue()

        # Fields render in a stable order, so two runs diff cleanly.
        assert _fields({"b": 2, "a": 1}) == "a=1 b=2"
    finally:
        log.handlers.clear()
        log.handlers.extend(saved)

    print("events: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
