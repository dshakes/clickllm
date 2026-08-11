"""Measure decode throughput, and refuse to call a contended sample a measurement.

Every throughput figure in this project is a roofline estimate, labelled as one
in all 35 places it surfaces. This is the module that can replace one with an
observation — and the reason it is careful is [#80], filed before any of this
existed:

    46.93 tok/s   (first run, quiet-ish machine)
    33.69         (best of 4 later passes)

A 40% spread on identical inputs, on an idle-looking laptop, tracking the load
average rather than temperature. Best-of-N did not converge, because the machine
got busier as the benchmark ran.

**A measured number that is 40% low is worse than the estimate it replaces**,
because it carries more authority and it is sticky: written into a receipt or a
box's `bench.json` and trusted long after the browser tab that caused it was
closed. So this module's job is as much refusing as measuring.

The four rules, from that issue:

1. Report load conditions with every measurement.
2. Refuse rather than silently substitute, above a contention threshold.
3. Repeat and disclose the spread. A wide spread *is* the finding.
4. Never let `measured` beat `estimated` on authority alone.

## What is measured

Decode rate: first token to last, divided by the tokens between them. Not total
request time, which includes prefill and would understate decode by an amount
that depends on prompt length — and the roofline it is compared against models a
decode step, so anything else is comparing two different quantities and calling
the difference an error.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "LOAD_PER_CORE_LIMIT",
    "Load",
    "Measurement",
    "SPREAD_LIMIT",
    "Sample",
    "measure",
    "read_load",
]

#: Relative spread — `(max - min) / median` — above which a set of samples is
#: not one measurement of one thing. **A calibration knob, not a truth.**
#:
#: The denominator is stated because it moves the number: #80's samples of
#: 33.69 and 46.93 tok/s are a 39% spread against the low one (which is what
#: that issue called "40%"), 33% against the median, and 28% against the high
#: one. Same observation, three headline figures. This uses the median, so the
#: figure does not change depending on which end you anchor to, and 0.20 sits
#: comfortably under that case while tolerating ordinary jitter.
#:
#: It is the *primary* gate because it is self-validating: computed from the
#: samples themselves rather than from a guess about what a busy machine looks
#: like.
SPREAD_LIMIT = 0.20

#: Load average per core above which the machine is too busy to measure on.
#: **A weaker signal than the spread, and a rougher knob**: #80 saw usable
#: numbers around 0.55/core and unusable ones at 1.07/core, which is one
#: anecdote on one machine. Set between them, and it only ever *adds* a refusal
#: — a quiet machine that still produces a wide spread is refused by the rule
#: above. Raise it if it proves noisy on a busy CI box; the spread gate is what
#: is really doing the work.
LOAD_PER_CORE_LIMIT = 0.80

#: Samples per measurement. Enough for a median and a spread to mean something.
DEFAULT_SAMPLES = 5

#: Tokens to ask for per sample. Long enough that decode dominates the timed
#: window and one slow token cannot define the rate.
DEFAULT_MAX_TOKENS = 128


@dataclass(frozen=True, slots=True)
class Load:
    """What else the machine was doing. Recorded, never inferred."""

    one_minute: float | None
    cores: int
    #: Best-effort, and empty when it cannot be read. Named processes make a
    #: refusal actionable — "load 17" tells you to try later, "load 17, mostly
    #: `cargo` and `Chrome`" tells you what to close.
    top: tuple[str, ...] = ()

    @property
    def per_core(self) -> float | None:
        if self.one_minute is None or self.cores < 1:
            return None
        return self.one_minute / self.cores

    @property
    def contended(self) -> bool:
        pc = self.per_core
        return pc is not None and pc > LOAD_PER_CORE_LIMIT

    def render(self) -> str:
        if self.one_minute is None:
            return "load unknown on this platform"
        if self.per_core is None:
            # `cores < 1`: the host's CPU count could not be read. Still show
            # the raw load rather than crash formatting a per-core figure that
            # does not exist — `per_core` already refuses to divide by it.
            out = f"load {self.one_minute:.2f} (core count unknown)"
        else:
            out = f"load {self.one_minute:.2f} over {self.cores} cores ({self.per_core:.2f}/core)"
        if self.top:
            out += " · busiest: " + ", ".join(self.top)
        return out


def read_load(cores: int) -> Load:
    """Current load, with the top CPU consumers when they can be read."""
    try:
        one = os.getloadavg()[0]
    except (OSError, AttributeError):
        # Windows has no load average. Unknown is reported as unknown rather
        # than as zero, which would read as "the machine was idle".
        one = None

    top: list[str] = []
    ps = shutil.which("ps")
    if ps:
        try:
            out = subprocess.run(  # noqa: S603 - fixed argv, no shell
                [ps, "-Ao", "%cpu,comm"],
                capture_output=True,
                text=True,
                # A process name is arbitrary bytes. Without this, decoding uses
                # the locale's codec and raises `UnicodeDecodeError` on the first
                # one that is not valid in it — and that inherits from
                # `ValueError`, not `OSError`, so the handler below would not
                # catch it. Reading the load is best-effort; crashing the
                # measurement because someone has an emoji in a binary name is
                # not best-effort.
                errors="replace",
                timeout=5,
                check=False,
            ).stdout
            rows = []
            for line in out.splitlines()[1:]:
                pct, _, comm = line.strip().partition(" ")
                try:
                    rows.append((float(pct), comm.strip().rsplit("/", 1)[-1]))
                except ValueError:
                    continue
            rows.sort(reverse=True)
            top = [f"{c} {p:.0f}%" for p, c in rows[:3] if p >= 5.0]
        except (OSError, subprocess.SubprocessError, ValueError):
            # `ValueError` too, and belt-and-braces with `errors="replace"`
            # above: this whole block is decoration on a number, and no failure
            # inside it may take the measurement with it.
            top = []
    return Load(one_minute=one, cores=cores, top=tuple(top))


@dataclass(frozen=True, slots=True)
class Sample:
    """One timed decode."""

    tokens: int
    #: Seconds between the first token and the last. Prefill is excluded on
    #: purpose — see the module docstring.
    decode_seconds: float
    ttft_seconds: float

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.decode_seconds if self.decode_seconds > 0 else 0.0


@dataclass(frozen=True, slots=True)
class Measurement:
    """Samples, the conditions they were taken in, and whether they count."""

    model: str
    endpoint: str
    samples: tuple[Sample, ...]
    load_before: Load
    load_after: Load
    roofline_tokens_per_sec: float | None = None
    #: Populated when this is not usable as a measurement. Non-empty means the
    #: estimate stands and this must not overwrite it.
    refused: tuple[str, ...] = field(default_factory=tuple)
    #: Things a reader must know that are not disqualifying — chiefly a check
    #: that could not run. A caveat does not make the number unusable; leaving
    #: it out would make the number look better-checked than it is.
    caveats: tuple[str, ...] = field(default_factory=tuple)

    @property
    def rates(self) -> list[float]:
        return [s.tokens_per_sec for s in self.samples]

    @property
    def median(self) -> float | None:
        return statistics.median(self.rates) if self.samples else None

    @property
    def spread(self) -> float | None:
        """`(max - min) / median`. The honest one-number summary of stability."""
        if len(self.samples) < 2:
            return None
        r = self.rates
        med = statistics.median(r)
        return (max(r) - min(r)) / med if med > 0 else None

    @property
    def usable(self) -> bool:
        """Whether this may be reported as a measurement at all."""
        return not self.refused

    @property
    def ratio_to_roofline(self) -> float | None:
        if self.roofline_tokens_per_sec and self.median:
            return self.median / self.roofline_tokens_per_sec
        return None

    def to_json(self) -> str:
        d = asdict(self)
        d["median_tokens_per_sec"] = self.median
        d["spread"] = self.spread
        d["measured"] = self.usable
        d["ratio_to_roofline"] = self.ratio_to_roofline
        return json.dumps(d, indent=2, sort_keys=True, default=str) + "\n"

    def render(self) -> str:
        lines = ["", f"  {self.model} via {self.endpoint}", ""]
        for i, s in enumerate(self.samples, 1):
            lines.append(
                f"    sample {i}   {s.tokens_per_sec:7.2f} tok/s decode"
                f"   ttft {s.ttft_seconds * 1000:6.0f} ms   {s.tokens} tokens"
            )
        if self.median is not None:
            spread = f"{self.spread * 100:.1f}%" if self.spread is not None else "n/a"
            lines += ["", f"    median   {self.median:7.2f} tok/s   spread {spread}"]
        if self.roofline_tokens_per_sec:
            lines.append(f"    roofline {self.roofline_tokens_per_sec:7.2f} tok/s   estimate")
            if self.ratio_to_roofline is not None:
                lines.append(f"    measured is {self.ratio_to_roofline * 100:.0f}% of the estimate")
        lines += [
            "",
            f"    before: {self.load_before.render()}",
            f"    after:  {self.load_after.render()}",
        ]

        if self.caveats:
            lines += ["", "  Checked less than usual:"]
            lines += [f"    · {c}" for c in self.caveats]

        if self.refused:
            lines += ["", "  NOT A MEASUREMENT:"]
            lines += [f"    · {r}" for r in self.refused]
            lines += [
                "",
                "  The roofline estimate stands. A number taken under these",
                "  conditions would carry more authority than the estimate and",
                "  less truth, and it would outlive the conditions that produced",
                "  it — written into a receipt and trusted next quarter.",
            ]
        else:
            lines += [
                "",
                "  Usable. Still one machine, one moment, one prompt shape —",
                "  report it as measured here, not as this model's speed.",
            ]
        lines.append("")
        return "\n".join(lines)


def _decode_once(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
    api_key: str = "",
) -> Sample:
    """One streamed completion, timed from first token to last."""
    from .prove.collect import chat_url

    payload = json.dumps(
        {
            "model": model,
            "stream": True,
            # Ask for a trailing usage frame. Streaming only promises text
            # *deltas*, not one SSE frame per token — an endpoint that
            # coalesces several tokens into one content delta would otherwise
            # be undercounted as chunks-per-second dressed up as tok/s.
            # completion_tokens, when the endpoint sends it, overrides the
            # frame count below.
            "stream_options": {"include_usage": True},
            "max_tokens": max_tokens,
            # Greedy: sampling adds variance that is not the machine's, and this
            # is measuring the machine.
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()
    headers = {"content-type": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(  # noqa: S310 - caller-supplied endpoint, by design
        chat_url(endpoint), data=payload, headers=headers, method="POST"
    )
    started = time.perf_counter()
    first: float | None = None
    last = started
    tokens = 0
    content_frames = 0
    usage_tokens: int | None = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                # A compliant server sends an object; do not assume it.
                continue
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                usage_tokens = usage["completion_tokens"]
            choices = event.get("choices") or [{}]
            delta = (choices[0] or {}).get("delta") or {}
            if not delta.get("content"):
                # Role-only preamble and usage frames are not tokens. Counting
                # them would inflate the rate by a couple of free "tokens".
                continue
            now = time.perf_counter()
            content_frames += 1
            if first is None:
                first = now
            else:
                tokens += 1
            last = now

    if usage_tokens is not None:
        # Authoritative when the endpoint reports it: the frame count above is
        # a lower bound, not the true token count, whenever frames and tokens
        # are not 1:1. Same "exclude the first token" convention as the frame
        # count, so it lines up with `decode_seconds` (first token to last).
        tokens = max(usage_tokens - 1, 0)

    if first is None or tokens < 1:
        raise ValueError(
            f"{model} at {endpoint} produced no streamed tokens. Is it serving, "
            "and does it support `stream: true`?"
        )
    if content_frames < 2:
        # Everything arrived in one SSE frame: `first` and `last` are the same
        # instant, so `decode_seconds` would be 0 — not a fast decode, an
        # unmeasurable one. `usage.completion_tokens` can still say how many
        # tokens came back, but it says nothing about *when*, and a 0-second
        # decode_seconds divides away to a `tokens_per_sec` of exactly 0.0,
        # which `spread()` then can't distinguish from "no spread computed" and
        # would wave through as a measurement. Refuse instead of measuring a
        # rate this can't time.
        raise ValueError(
            f"{model} at {endpoint} sent the whole completion in a single "
            "streamed frame. Decode cannot be timed between a first and last "
            "token that are the same frame — is the server coalescing the "
            "full response instead of streaming it incrementally?"
        )
    return Sample(tokens=tokens, decode_seconds=last - first, ttft_seconds=first - started)


def measure(
    endpoint: str,
    model: str,
    *,
    cores: int,
    samples: int = DEFAULT_SAMPLES,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt: str = "Count slowly from one to two hundred, one number per line.",
    timeout: float = 300.0,
    roofline: float | None = None,
    api_key: str = "",
    sampler: Any = None,
    load_reader: Any = None,
) -> Measurement:
    """Take `samples` timed decodes and decide whether they are a measurement.

    `sampler` exists so the decision logic can be tested without an inference
    server, and `load_reader` for the same reason one level out: a test of the
    contention rules that reads the *host's* load passes or fails with the CI
    machine's mood rather than with the code. That is not a hypothetical — the
    self-check below asserted a clean measurement and failed on a runner sitting
    at 1.33/core, where refusing was the correct behaviour.

    The product path leaves both None and uses the real ones.
    """
    if samples < 2:
        raise ValueError(
            f"samples must be >= 2, got {samples}: a single observation has no "
            "spread, and the spread is what decides whether this is a measurement"
        )

    take = sampler or (
        lambda: _decode_once(
            endpoint, model, prompt, max_tokens=max_tokens, timeout=timeout, api_key=api_key
        )
    )
    look = load_reader or (lambda: read_load(cores))
    before = look()
    taken = [take() for _ in range(samples)]
    after = look()

    m = Measurement(
        model=model,
        endpoint=endpoint,
        samples=tuple(taken),
        load_before=before,
        load_after=after,
        roofline_tokens_per_sec=roofline,
    )

    refused: list[str] = []
    caveats: list[str] = []

    # The second door to the same failure. `_decode_once` refuses a completion
    # that arrived in one frame, because first and last are then the same
    # instant — but that guards the HTTP path, and this function takes samples
    # from wherever the caller got them. A zero rate reaching here reports
    # `median 0.0`, and `spread` needs `median > 0` so it returns None and no
    # refusal fires: 0 tok/s, usable, no reason given.
    #
    # A rate of zero is not a slow measurement, it is the absence of one, so the
    # invariant belongs here rather than only at the one caller that can
    # currently produce it.
    if any(sample.tokens_per_sec <= 0 for sample in taken):
        refused.append(
            "at least one sample decoded no tokens in no time — a rate of zero "
            "is the absence of a measurement, not a slow one"
        )

    spread = m.spread
    if spread is not None and spread > SPREAD_LIMIT:
        refused.append(
            f"the samples disagree by {spread * 100:.0f}% "
            f"({min(m.rates):.1f}–{max(m.rates):.1f} tok/s), over the {SPREAD_LIMIT * 100:.0f}% "
            "limit — that is not one measurement of one thing"
        )
    # The load gate not running is not the same as it passing, and the same
    # mistake in `guard` (a fingerprint check that iterated nothing and reported
    # "still holds") is the reason this is stated rather than assumed. Not a
    # refusal — an unreadable load average is no evidence the machine was busy —
    # but it must not read as a clean bill of health either.
    if before.per_core is None or after.per_core is None:
        caveats.append(
            "the contention gate did not run: "
            + (
                "this platform has no load average"
                if before.one_minute is None
                else "the host CPU count could not be read"
            )
            + ", so only the spread was checked"
        )

    for when, load in (("before", before), ("after", after)):
        if load.contended:
            refused.append(
                f"the machine was busy {when}: {load.render()}, over {LOAD_PER_CORE_LIMIT:.2f}/core"
            )
    return Measurement(
        model=m.model,
        endpoint=m.endpoint,
        samples=m.samples,
        load_before=before,
        load_after=after,
        roofline_tokens_per_sec=roofline,
        refused=tuple(refused),
        caveats=tuple(caveats),
    )


def demo() -> None:
    """Self-check: the refusals, without an inference server *or* a quiet host.

    Every case here injects both the samples and the load, so what is being
    checked is the decision and not the machine it runs on.
    """
    quiet = lambda: Load(one_minute=1.0, cores=16)  # noqa: E731 - one expression
    steady = iter([Sample(100, 2.00, 0.1), Sample(100, 2.02, 0.1), Sample(100, 1.98, 0.1)])
    m = measure(
        "http://x/v1", "m", cores=16, samples=3, sampler=lambda: next(steady), load_reader=quiet
    )
    assert m.usable, m.refused
    assert m.median is not None and 49 < m.median < 51, m.median
    assert m.spread is not None and m.spread < 0.05, m.spread
    assert "Usable" in m.render()

    # #80's actual numbers: 46.93 and 33.69 tok/s on identical inputs. This is
    # the case the whole module exists for, so it is checked with the figures
    # that motivated it rather than with invented ones.
    wobbly = iter([Sample(100, 100 / 46.93, 0.1), Sample(100, 100 / 33.69, 0.1)])
    w = measure(
        "http://x/v1", "m", cores=16, samples=2, sampler=lambda: next(wobbly), load_reader=quiet
    )
    assert not w.usable, "a 39% spread was accepted as a measurement"
    # 33%, not the 39% #80 quotes: that issue anchored on the low sample and
    # this anchors on the median. The same observation, a different denominator
    # — which is exactly why the constant above names its own.
    assert "disagree by 33%" in w.render(), w.render()
    assert "NOT A MEASUREMENT" in w.render()

    # A steady sample on a machine that is too busy is refused too: stability
    # under load can mean everything was equally slow.
    busy = iter([Sample(100, 4.0, 0.1), Sample(100, 4.0, 0.1)])
    b = measure(
        "http://x/v1",
        "m",
        cores=4,
        samples=2,
        sampler=lambda: next(busy),
        load_reader=quiet,
    )
    b = Measurement(
        model=b.model,
        endpoint=b.endpoint,
        samples=b.samples,
        load_before=Load(one_minute=17.06, cores=4),
        load_after=b.load_after,
        refused=("the machine was busy before: load 17.06 over 4 cores (4.27/core)",),
    )
    assert not b.usable
    assert "4.27/core" in b.render()

    # The comparison never claims the measurement wins on authority: both
    # numbers are shown, with the ratio between them.
    steady2 = iter([Sample(100, 2.0, 0.1), Sample(100, 2.0, 0.1)])
    r = measure(
        "http://x/v1",
        "m",
        cores=16,
        samples=2,
        roofline=100.0,
        sampler=lambda: next(steady2),
        load_reader=quiet,
    )
    text = r.render()
    assert "roofline" in text and "estimate" in text
    assert "50% of the estimate" in text, text

    assert json.loads(r.to_json())["measured"] is True
    try:
        measure(
            "http://x/v1",
            "m",
            cores=1,
            samples=1,
            sampler=lambda: Sample(1, 1.0, 0.1),
            load_reader=quiet,
        )
    except ValueError as e:
        assert "spread" in str(e)
    else:  # pragma: no cover
        raise AssertionError("one sample has no spread and must be refused")
    print("measure: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
