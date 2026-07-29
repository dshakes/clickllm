"""The collector — point an eval set at a live endpoint and bring back the replies.

The rest of ``prove/`` scores items that already carry both answers. Nothing in
it fetches an answer, on purpose: :func:`~clickllm.prove.run` is pure, and a
suite that could reach out on your behalf is a suite that can send captured
prompts somewhere you did not name. That left one gap, though — ``clickllm run``
hands you a live OpenAI-compatible endpoint and ``clickllm prove`` could only
read a file of replies somebody else had already gathered. This module is the
one, explicit step between them.

**Egress is the whole risk here, so it is the whole design.** One destination,
named by the caller. No credential is ever read from the environment — if the
endpoint needs auth, the caller passes the header value in, because a tool that
goes looking for anything shaped like a token will eventually find one and send
it to a host the user did not intend.

## Failure is data

A collection over a real eval set is hundreds of round trips against a server
that is often a laptop under load. Some will fail. The temptation is to raise on
the first one, and the cost of that is a run that dies on item 200 of 400 having
proven nothing. So every per-item failure — a timeout, a 5xx, a body with no
choices in it — is recorded on that item as a reason and the run continues.

The consequence matters more than the mechanism: **a failed collection is not a
failed answer.** An item that never got a reply must not be scored as a model
that answered badly. It is dropped from the sample, which makes the sample
smaller, which makes the Wilson interval wider — and the count of what was lost
is reported next to the verdict so nobody reads the narrower denominator as a
cleaner result.

## What is recorded, and what is never invented

The model id the server echoed, token usage if the server sent it, wall time,
and attempts. Absent means absent: a server that reports no usage produces
``None``, never a plausible-looking estimate, because an invented token count is
indistinguishable from a measured one once it is in a report.

## The limit worth knowing

Tool-using items cannot be reproduced faithfully. :class:`~.graders.EvalItem`
records the tool calls the incumbent *made*, not the tool definitions it was
*offered*, so there is nothing to send. Any reply's ``tool_calls`` are captured
if the server volunteers them, but a cluster whose baseline called tools will
grade against an empty call list and read as a regression it may not be. Collect
those from your own harness, where the tool definitions live.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from .graders import EvalItem

__all__ = ["Collected", "Collection", "collect", "demo"]

#: Which answer on an :class:`~.graders.EvalItem` a collection fills in.
#: ``baseline`` is the incumbent's side — the bar being matched, not ground truth.
SIDES = ("candidate", "baseline")

USER_AGENT = "clickllm-prove/0.1"

DEFAULT_WORKERS = 8
#: Generous: the target is frequently a local engine that has just loaded weights
#: and is answering its first real requests.
DEFAULT_TIMEOUT = 120.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF = 0.5
#: Ceiling on one backoff sleep, including a server-supplied `Retry-After`. A
#: server asking us to wait an hour gets a failed item instead of a hung run.
MAX_BACKOFF = 30.0
#: How much of an error body is kept. Enough to identify the fault, not enough to
#: paste a stack trace into a report.
DETAIL_CHARS = 200


class _Failed(Exception):  # noqa: N818 - it is a failure, not an error condition
    """Terminal for this item: retrying would ask the same question again.

    Never escapes the module — it is converted into a :class:`Collected` with a
    reason on it, because a collector that raises has thrown away the other 399
    items it already had.
    """


class _Retry(Exception):  # noqa: N818
    """Transient: 429 or 5xx. Worth another attempt, up to the cap."""

    def __init__(self, reason: str, after: float | None = None) -> None:
        super().__init__(reason)
        #: Seconds the server asked us to wait, when it said.
        self.after = after


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Collected:
    """One item's round trip — or the reason it did not complete one."""

    item_id: str
    text: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    #: Empty when the reply arrived. Non-empty is the whole failure record.
    reason: str = ""
    #: The model id the *server* echoed, which is not always the one asked for.
    #: Empty means the server did not say, never a guess at what it served.
    served_model: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return not self.reason


@dataclass(frozen=True, slots=True)
class Collection:
    """Everything one endpoint returned, including what it did not.

    ``items`` holds only the items that answered, with the collected side filled
    in — the input to :func:`~clickllm.prove.run`. The ones that failed are in
    ``failures``, deliberately not in ``items``: an item with no reply scored as
    a model failure would turn a network fault into a quality verdict.
    """

    base: str
    model: str
    side: str
    replies: tuple[Collected, ...]
    items: tuple[EvalItem, ...]
    seconds: float

    @property
    def asked(self) -> int:
        return len(self.replies)

    @property
    def failures(self) -> tuple[Collected, ...]:
        return tuple(c for c in self.replies if not c.ok)

    @property
    def served(self) -> tuple[str, ...]:
        """Distinct model ids the server echoed. More than one is worth seeing."""
        return tuple(sorted({c.served_model for c in self.replies if c.served_model}))

    @property
    def completion_tokens(self) -> int | None:
        """Total tokens generated, or ``None`` if the server never reported any."""
        counted = [c.completion_tokens for c in self.replies if c.completion_tokens is not None]
        return sum(counted) if counted else None

    def render(self) -> str:
        """One block per collection, failures itemised under it."""
        served = ", ".join(self.served) or "not stated by the server"
        head = (
            f"  {self.side:<9} {len(self.items)}/{self.asked} replies · {self.base} · "
            f"served: {served} · {self.seconds:.2f}s"
        )
        if (tok := self.completion_tokens) is not None:
            head += f" · {tok:,} completion tokens"
        out = [head]
        if not self.failures:
            return out[0]

        out.append(
            f"    {len(self.failures)} item(s) never got a reply and are NOT scored — "
            f"this is a smaller sample, not a lower score"
        )
        grouped: dict[str, list[str]] = {}
        for c in self.failures:
            grouped.setdefault(c.reason, []).append(c.item_id)
        for reason, ids in sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:5]:
            shown = ", ".join(ids[:3]) + (f", +{len(ids) - 3} more" if len(ids) > 3 else "")
            out.append(f"      · {len(ids)} × {reason[:90]}   [{shown}]")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# One request
# --------------------------------------------------------------------------- #


def chat_url(base: str) -> str:
    """The completions URL for an OpenAI-compatible base.

    Both spellings are in circulation and both arrive here: `clickllm run` prints
    `http://host:port/v1` while :attr:`~clickllm.launch.Endpoint.base` carries
    `http://host:port`. Guessing wrong produces a 404 that reads like the model
    is missing, so this accepts either.

    Raises:
        ValueError: if `base` is not an http(s) URL. A bare hostname would be
            posted at some other scheme's default and fail unrecognisably.
    """
    b = base.strip().rstrip("/")
    if not b.startswith(("http://", "https://")):
        raise ValueError(f"endpoint must be an http(s) URL, got {base!r}")
    return f"{b}/chat/completions" if b.endswith("/v1") else f"{b}/v1/chat/completions"


def _retry_after(headers: Any) -> float | None:
    """`Retry-After` in seconds, when the server sent a number we can use."""
    try:
        return float(headers.get("Retry-After"))
    except (AttributeError, TypeError, ValueError):
        return None


def _once(url: str, payload: bytes, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    """One POST. Raises :class:`_Retry` for 429/5xx, :class:`_Failed` otherwise."""
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = (e.read() or b"")[:DETAIL_CHARS].decode("utf-8", "replace").strip()
        text = f"HTTP {e.code}" + (f" — {detail}" if detail else "")
        # 4xx other than 429 is our request being wrong; sending it again is spend
        # with no chance of a different answer.
        if e.code == 429 or e.code >= 500:
            raise _Retry(text, _retry_after(e.headers)) from e
        raise _Failed(text) from e
    except TimeoutError as e:
        raise _Failed(f"timed out after {timeout:g}s") from e
    except urllib.error.URLError as e:
        if isinstance(e.reason, TimeoutError):
            raise _Failed(f"timed out after {timeout:g}s") from e
        raise _Failed(f"could not reach {url} — {e.reason}") from e
    except OSError as e:
        raise _Failed(f"could not reach {url} — {e}") from e

    try:
        body = json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError as e:
        raise _Failed(f"reply is not JSON: {raw[:DETAIL_CHARS]!r}") from e
    if not isinstance(body, dict):
        raise _Failed(f"reply is not a JSON object: {type(body).__name__}")
    return body


def _reply(body: dict[str, Any]) -> tuple[str, tuple[dict[str, Any], ...]]:
    """The assistant's text and tool calls, or why the body was unusable."""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise _Failed(f"no usable choices in the reply: {json.dumps(body)[:DETAIL_CHARS]}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise _Failed(f"choice carries no message: {json.dumps(choices[0])[:DETAIL_CHARS]}")
    text = message.get("content")
    calls = tuple(c for c in (message.get("tool_calls") or ()) if isinstance(c, dict))
    if text is None and not calls:
        raise _Failed("the reply carried neither content nor tool calls")
    return str(text or ""), calls


def _int_or_none(usage: Any, key: str) -> int | None:
    """A token count the server actually reported. Absent stays absent."""
    if isinstance(usage, dict) and isinstance(v := usage.get(key), int):
        return v
    return None


def _delay(attempt: int, backoff: float, after: float | None) -> float:
    """Exponential backoff, or the server's own answer, capped either way."""
    return min(after if after is not None else backoff * 2**attempt, MAX_BACKOFF)


def _ask(
    item: EvalItem,
    *,
    url: str,
    payload_of: Callable[[str], bytes],
    headers: dict[str, str],
    timeout: float,
    retries: int,
    backoff: float,
    sleep: Callable[[float], None],
) -> Collected:
    """One item, retried where retrying can help. Never raises."""
    payload = payload_of(item.prompt)
    started = time.monotonic()
    last = ""
    for attempt in range(retries + 1):
        try:
            body = _once(url, payload, headers, timeout)
            text, calls = _reply(body)
        except _Failed as e:
            return Collected(
                item.item_id,
                reason=str(e),
                seconds=time.monotonic() - started,
                attempts=attempt + 1,
            )
        except _Retry as e:
            last = str(e)
            if attempt < retries:
                sleep(_delay(attempt, backoff, e.after))
            continue
        usage = body.get("usage")
        return Collected(
            item.item_id,
            text=text,
            tool_calls=calls,
            served_model=str(body.get("model") or ""),
            prompt_tokens=_int_or_none(usage, "prompt_tokens"),
            completion_tokens=_int_or_none(usage, "completion_tokens"),
            seconds=time.monotonic() - started,
            attempts=attempt + 1,
        )
    return Collected(
        item.item_id,
        reason=f"{last} — gave up after {retries + 1} attempts",
        seconds=time.monotonic() - started,
        attempts=retries + 1,
    )


# --------------------------------------------------------------------------- #
# The collection
# --------------------------------------------------------------------------- #


def _filled(item: EvalItem, got: Collected, side: str) -> EvalItem:
    """The item with one side replaced by what was actually returned."""
    if side == "candidate":
        return replace(item, candidate=got.text, candidate_tool_calls=got.tool_calls)
    return replace(item, baseline=got.text, baseline_tool_calls=got.tool_calls)


def collect(
    items: list[EvalItem],
    *,
    base: str,
    model: str,
    side: str = "candidate",
    headers: dict[str, str] | None = None,
    workers: int = DEFAULT_WORKERS,
    timeout: float = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
    sleep: Callable[[float], None] = time.sleep,
) -> Collection:
    """Send every item's prompt to one OpenAI-compatible endpoint and collect the replies.

    Args:
        items: the eval set. Prompts are sent as-is; captured traffic is data,
            not instruction, and nothing here interprets it.
        base: the endpoint, with or without a trailing `/v1`. The only host this
            function will ever talk to.
        model: the model id to ask for. Sent verbatim; whatever the server echoes
            back is recorded separately, because they are not always the same.
        side: which answer to fill in — `"candidate"` or `"baseline"`.
        headers: extra request headers, e.g. `{"Authorization": "Bearer ..."}`.
            Passed in, never read from the environment.
        workers: parallel requests. Serial collection over a real eval set takes
            hours; unbounded parallelism queues a laptop engine into timeouts.
        temperature: 0 by default — an eval you cannot re-run is not evidence.
        max_tokens: omitted from the request when `None`, so the server's own
            default applies rather than a cap invented here.
        retries: extra attempts on 429/5xx only. A 4xx is never retried.
        sleep: injected so tests do not spend the backoff.

    Returns:
        A :class:`Collection`. Items that failed are in `failures`, not in
        `items` — they shrink the sample rather than scoring as bad answers.

    Raises:
        ValueError: empty `items`, an unknown `side`, a non-positive `workers`,
            or a `base` that is not an http(s) URL. These are caller bugs that
            would otherwise fail once per item, several hundred times.
    """
    if not items:
        raise ValueError("cannot collect replies for an empty eval set")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if retries < 0:
        raise ValueError(f"retries cannot be negative, got {retries}")
    url = chat_url(base)

    def payload_of(prompt: str) -> bytes:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return json.dumps(body).encode()

    request_headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    } | dict(headers or {})

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        replies = tuple(
            pool.map(
                lambda item: _ask(
                    item,
                    url=url,
                    payload_of=payload_of,
                    headers=request_headers,
                    timeout=timeout,
                    retries=retries,
                    backoff=backoff,
                    sleep=sleep,
                ),
                items,
            )
        )

    return Collection(
        base=base,
        model=model,
        side=side,
        replies=replies,
        items=tuple(_filled(i, c, side) for i, c in zip(items, replies, strict=True) if c.ok),
        seconds=time.monotonic() - started,
    )


def demo() -> None:
    """Self-check. Run with `python -m clickllm.prove.collect`. Binds loopback only."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    state = {"429s": 0}

    class Stub(BaseHTTPRequestHandler):
        """Answers most prompts, fails a couple in the ways that really happen."""

        def do_POST(self) -> None:  # noqa: N802 - stdlib's spelling
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            prompt = body["messages"][0]["content"]
            if prompt == "boom":
                return self._send(500, {"error": "engine crashed"})
            if prompt == "busy" and state["429s"] < 2:
                state["429s"] += 1
                return self._send(429, {"error": "slow down"})
            if prompt == "junk":
                return self._send(200, {"object": "chat.completion"})  # no choices
            self._send(
                200,
                {
                    "model": "served-name",
                    "choices": [{"message": {"role": "assistant", "content": f"re: {prompt}"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5},
                },
            )

        def _send(self, code: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a: Any) -> None:
            pass

    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/v1"

    def item(n: int, prompt: str) -> EvalItem:
        return EvalItem(f"i{n}", "c", prompt, baseline="was", candidate="")

    items = [item(0, "hello"), item(1, "boom"), item(2, "busy"), item(3, "junk")]
    try:
        got = collect(items, base=base, model="asked-for", retries=3, sleep=lambda s: None)
    finally:
        srv.shutdown()
        srv.server_close()

    # Replies land on their own items, and only on the side being collected.
    assert got.asked == 4 and len(got.items) == 2, got.render()
    ok = {i.item_id: i for i in got.items}
    assert ok["i0"].candidate == "re: hello", ok["i0"]
    assert ok["i0"].baseline == "was", "the other side must be left alone"

    # A 500 fails that item and nothing else; a 429 is retried until it works.
    fails = {c.item_id: c.reason for c in got.failures}
    assert "HTTP 500" in fails["i1"], fails
    assert "no usable choices" in fails["i3"], fails
    assert "i2" in ok and ok["i2"].candidate == "re: busy", "429 must be retried"
    assert next(c for c in got.replies if c.item_id == "i2").attempts == 3

    # What the server said is recorded; what it did not say is not invented.
    assert got.served == ("served-name",), got.served
    assert got.completion_tokens == 10, got.completion_tokens
    assert next(c for c in got.replies if c.item_id == "i1").served_model == ""

    # The failures are visible in the report, never folded into the denominator.
    text = got.render()
    assert "2/4 replies" in text and "smaller sample, not a lower score" in text

    # A 4xx that is not 429 is never retried — it would be the same request.
    assert _delay(0, 0.5, None) == 0.5 and _delay(9, 0.5, None) == MAX_BACKOFF
    assert chat_url("http://h:1/v1") == chat_url("http://h:1/") == "http://h:1/v1/chat/completions"
    for bad in (
        lambda: collect([], base=base, model="m"),
        lambda: collect(items, base=base, model="m", side="sideways"),
        lambda: collect(items, base="h:1", model="m"),
    ):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass

    print(f"collect: {len(got.items)}/{got.asked} replies, {len(got.failures)} failed")
    print(got.render())
    print("ok")


if __name__ == "__main__":
    demo()
