"""Captured exchanges, and the task shape extracted from them.

The load-bearing idea of stage ②: **task identity is structural, not semantic.**

Two requests asking to refactor different files, with the same system prompt and
the same tool schema, are the *same task* even though their text shares almost no
words. An embedding model would score them apart; what actually distinguishes a
migration-relevant task cluster is the shape of the exchange — system prompt,
tools offered, output format demanded, context size, whether tools were called.

Structural clustering is also **deterministic**, which matters more here than
sophistication: the same traffic must produce the same eval set every time, or an
equivalence verdict cannot be reproduced or audited.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

# Context buckets, in tokens. A 900-token and a 1,100-token prompt are the same
# workload; a 900-token and a 90,000-token prompt are not.
CONTEXT_BUCKETS: tuple[int, ...] = (1_024, 4_096, 16_384, 65_536, 262_144)


@dataclass(frozen=True, slots=True)
class Capture:
    """One request/response exchange recorded by the gateway.

    ``response`` is the **incumbent's** answer. It is not ground truth — it is the
    baseline a candidate must match, which is a different and weaker claim.
    """

    request_id: str
    model: str
    messages: tuple[dict[str, Any], ...]
    response: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tools: tuple[dict[str, Any], ...] = ()
    tool_calls: tuple[dict[str, Any], ...] = ()
    response_format: str | None = None
    latency_ms: int | None = None

    @property
    def system_prompt(self) -> str:
        for m in self.messages:
            if m.get("role") == "system":
                return str(m.get("content") or "")
        return ""

    @property
    def user_text(self) -> str:
        """Concatenated user turns — the part that varies between requests."""
        return "\n".join(
            str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
        )

    @property
    def approx_context_tokens(self) -> int:
        """Prompt size, preferring the reported count.

        Falls back to a chars/4 approximation, which is crude but only ever used
        to pick a bucket — an estimate is fine when the output is an order of
        magnitude, and the buckets are an order of magnitude apart.
        """
        if self.prompt_tokens is not None:
            return self.prompt_tokens
        chars = sum(len(str(m.get("content") or "")) for m in self.messages)
        return chars // 4


def _sha(*parts: str) -> str:
    """A stable key over an ordered list of parts.

    64 bits, not 48. These are grouping keys held in memory for one run, never
    persisted, so the width costs nothing — and a truncation collision here
    merges two genuinely different task shapes into one cluster silently,
    which is the one failure this module has no way to detect. Cheap insurance
    against the only outcome that would be invisible.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\x00")  # separator, so ("ab","c") != ("a","bc")
    return h.hexdigest()[:16]


def context_bucket(tokens: int) -> str:
    """Label the order of magnitude of a context length."""
    for b in CONTEXT_BUCKETS:
        if tokens <= b:
            return f"<={b // 1024}k" if b >= 1024 else f"<={b}"
    return f">{CONTEXT_BUCKETS[-1] // 1024}k"


def detect_output_format(text: str) -> str:
    """What shape of answer this is.

    Cheap and structural. Misclassifying prose as prose costs nothing; the point
    is to separate "must emit valid JSON" from "writes English", because those
    fail in completely different ways and need different graders.
    """
    s = text.strip()
    if not s:
        return "empty"
    if s.startswith(("{", "[")):
        try:
            json.loads(s)
            return "json"
        except json.JSONDecodeError:
            return "json-ish"
    if re.search(r"^(diff --git|--- |\+\+\+ |@@ )", s, re.MULTILINE):
        return "diff"
    if s.startswith("```") or re.search(r"^```", s, re.MULTILINE):
        return "code-block"
    # Deliberately NO length-based "short answer" class. A response-length
    # threshold splits one task into two clusters depending on how terse the
    # model happened to be, which breaks the promise that the same task always
    # lands in the same cluster. Classification workloads are already separated
    # by their system prompt and by response_format; length is not a task shape.
    return "prose"


@dataclass(frozen=True, slots=True)
class TaskShape:
    """The structural signature that defines a task cluster."""

    system_prompt_hash: str
    tool_names: tuple[str, ...]
    used_tools: bool
    output_format: str
    context_bucket: str
    response_format: str | None = None

    @property
    def signature(self) -> str:
        """Stable cluster key. Identical shapes share it exactly."""
        return _sha(
            self.system_prompt_hash,
            # Each name its own part, so _sha's NUL separator applies to them
            # too. Joined on a comma they did not: one tool named "get,weather"
            # and two named "get" and "weather" produced the same string, and a
            # one-tool bot and a two-tool bot merged into one cluster with
            # nothing downstream able to tell. The separator two functions up
            # exists for exactly this and was being bypassed here.
            *self.tool_names,
            str(self.used_tools),
            self.output_format,
            self.context_bucket,
            self.response_format or "",
        )

    def describe(self) -> str:
        """Human-readable name. A cluster nobody can name is a cluster nobody
        will act on, and the regret set has to be nameable to be useful."""
        bits = []
        if self.used_tools:
            n = len(self.tool_names)
            bits.append(f"tool-calling ({n} tool{'' if n == 1 else 's'})")
        elif self.tool_names:
            bits.append("tools offered, unused")
        bits.append(
            {
                "json": "structured JSON",
                "json-ish": "JSON-shaped (malformed)",
                "diff": "code diff",
                "code-block": "code generation",
                "prose": "free text",
                "empty": "empty response",
            }.get(self.output_format, self.output_format)
        )
        bits.append(f"{self.context_bucket} context")
        return " · ".join(bits)


def extract_shape(cap: Capture) -> TaskShape:
    """Derive the task shape of one capture."""
    return TaskShape(
        # The system prompt is hashed, not stored: it is often the most sensitive
        # text in the corpus, and clustering only needs identity, not content.
        system_prompt_hash=_sha(cap.system_prompt.strip()),
        tool_names=tuple(sorted(_tool_names(cap.tools))),
        used_tools=bool(cap.tool_calls),
        output_format=detect_output_format(cap.response),
        context_bucket=context_bucket(cap.approx_context_tokens),
        response_format=cap.response_format,
    )


def _tool_names(tools: tuple[dict[str, Any], ...]) -> list[str]:
    """Unique tool names. A gateway that accumulates schemas across turns
    declares the same tool twice, and `describe()` rendered that as
    "tool-calling (2 tools)" for a bot with one — the number a human uses to
    judge whether a cluster deserves its own eval."""
    names: list[str] = []
    for t in tools:
        # Both the flat and the OpenAI-nested spellings appear in the wild.
        name = t.get("name") or (t.get("function") or {}).get("name")
        if name and str(name) not in names:
            names.append(str(name))
    return names


@dataclass
class Cluster:
    """A group of captures sharing one task shape."""

    shape: TaskShape
    captures: list[Capture] = field(default_factory=list)
    #: Display name, made unique across a cluster set by
    #: :func:`clickllm.distill.cluster._disambiguate`. Empty until then.
    label: str = ""

    @property
    def key(self) -> str:
        return self.shape.signature

    @property
    def name(self) -> str:
        """Unambiguous display name, falling back to the raw description."""
        return self.label or self.shape.describe()

    @property
    def size(self) -> int:
        return len(self.captures)

    def share_of(self, total: int) -> float:
        """Fraction of total traffic. This is the weight everything downstream
        uses — a cluster that is 38% of traffic matters 5x one at 8%."""
        return self.size / total if total else 0.0


def demo() -> None:
    def cap(system: str, resp: str, tokens: int = 500, **kw) -> Capture:
        return Capture(
            request_id="r",
            model="gpt-5",
            messages=({"role": "system", "content": system}, {"role": "user", "content": "hi"}),
            response=resp,
            prompt_tokens=tokens,
            **kw,
        )

    # Same structure, different text -> same cluster. This is the whole point.
    a = extract_shape(cap("You are a code reviewer.", "```py\nx=1\n```"))
    b = extract_shape(cap("You are a code reviewer.", "```py\nyyy=2222\n```"))
    assert a.signature == b.signature, "same shape must cluster together"

    # Different system prompt -> different task.
    c = extract_shape(cap("You are a translator.", "```py\nx=1\n```"))
    assert c.signature != a.signature

    # Different output format -> different task, because it fails differently.
    d = extract_shape(cap("You are a code reviewer.", '{"ok": true}'))
    assert d.output_format == "json" and d.signature != a.signature

    # Context magnitude separates workloads.
    e = extract_shape(cap("You are a code reviewer.", "```py\nx=1\n```", tokens=200_000))
    assert e.signature != a.signature

    # Response length must NOT create a cluster: same task, same cluster.
    short = extract_shape(cap("You are a translator.", "hola"))
    long_ = extract_shape(cap("You are a translator.", "hola " * 200))
    assert short.signature == long_.signature, (
        "terse and verbose answers to the same task must not split into two clusters"
    )

    assert detect_output_format("{bad json") == "json-ish"
    assert detect_output_format("--- a\n+++ b\n") == "diff"
    assert detect_output_format("") == "empty"
    assert context_bucket(500) == "<=1k"
    assert context_bucket(10**9).startswith(">")
    assert "code generation" in a.describe()

    print(f"shape: {a.describe()}")
    print("ok")


if __name__ == "__main__":
    demo()
