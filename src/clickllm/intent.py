"""From a sentence to a deployment — and the questions it could not answer alone.

Everything else in this package takes structured input: a `Requirements`, a
`ModelSpec`, a context length. That is the correct shape for a library and the
wrong shape for a product. Someone who needs to run an open model does not
arrive holding a `Workload` enum. They arrive holding a sentence:

    "coding assistant for about 20 engineers, needs to feel snappy"
    "score 4 million support tickets overnight"
    "voice agent, has to reply in under a second"

This module turns that into a plan. It is the layer that decides whether the
tool is for people who already know what `--max-num-batched-tokens` does.

## Why this is not a language model call

The obvious implementation is to ask an LLM. It is rejected for three reasons,
in order of how much they matter:

1. **It would guess silently.** An LLM handed an ambiguous sentence returns a
   confident answer with no seam where the ambiguity was. This returns the
   inference *and the words that produced it*, so a wrong reading is visible and
   correctable rather than buried in a config.
2. **It would break the offline promise.** `clickllm fit` has zero runtime
   dependencies and runs under `uvx` on a plane. Making the entry point require
   an API key would put a login in front of the first thing a new user does.
3. **It is not actually hard.** The signal is in a few dozen words. Deterministic
   extraction gets it right, is testable, and costs nothing.

## The rule that makes this trustworthy

**Infer what the words support. State the inference and its evidence. Ask about
the rest — never assume it.**

A field nobody mentioned does not get a confident default pretending to be a
finding. It becomes a [`Question`] with a stated default, so "20 engineers" does
not silently become a claim about latency budgets nobody made. An agent that
asks one good question is worth more than one that guesses ten times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from clickllm.plan import Requirements, Workload

__all__ = [
    "Inference",
    "Intent",
    "Question",
    "demo",
    "read",
]


@dataclass(frozen=True, slots=True)
class Inference:
    """Something read out of the sentence, and the words that justified it."""

    field: str
    value: object
    #: The literal words that produced this. Quoted back so a misreading is
    #: obvious at a glance rather than discovered in production.
    evidence: str

    def render(self) -> str:
        """`workload = realtime   (from "voice agent")`"""
        return f'{self.field} = {self.value}   (from "{self.evidence}")'


@dataclass(frozen=True, slots=True)
class Question:
    """Something the sentence did not say, with the default if it goes unanswered.

    Carrying the default here rather than applying it silently is the whole
    point: the user can ignore every question and still get a deployment, but
    nothing that was assumed is invisible.
    """

    field: str
    ask: str
    default: object
    why_it_matters: str

    def render(self) -> str:
        """`concurrency? … (assuming 4)`"""
        return f"{self.ask}\n      assuming {self.default} — {self.why_it_matters}"


@dataclass(frozen=True, slots=True)
class Intent:
    """What the sentence asked for, what was assumed, and what is still unknown."""

    text: str
    requirements: Requirements
    inferred: tuple[Inference, ...]
    questions: tuple[Question, ...]

    @property
    def confident(self) -> bool:
        """Whether every field was supported by the words rather than defaulted."""
        return not self.questions

    def render(self) -> str:
        """What was understood, then what is being assumed."""
        out = [f'"{self.text}"', ""]
        if self.inferred:
            out.append("understood:")
            out += [f"  · {i.render()}" for i in self.inferred]
        else:
            out.append("understood: nothing specific — every field below is a default")
        if self.questions:
            out += ["", "assuming, unless you say otherwise:"]
            out += [f"  · {q.render()}" for q in self.questions]
        return "\n".join(out)


# Ordered most-specific first: "real-time batch scoring" is batch work with a
# careless adjective, and matching REALTIME on the adjective would be wrong.
_WORKLOAD_SIGNALS: tuple[tuple[Workload, tuple[str, ...]], ...] = (
    (
        Workload.BATCH,
        (
            "batch",
            "overnight",
            "offline",
            "backfill",
            "bulk",
            "score all",
            "nightly",
            "one-off",
            "dataset",
            "corpus",
            "reprocess",
            # No bare "million" — see _BULK_VOLUME. A magnitude alone is not a
            # mode, but a magnitude with a bulk verb is.
        ),
    ),
    (
        Workload.REALTIME,
        (
            "voice",
            "real-time",
            "realtime",
            "speech",
            "autocomplete",
            "sub-second",
            "interrupt",
            "live transcription",
            # "phone" is in _EXACT, not narrowed to "phone call". The bare
            # needle's problem was matching inside "phonebook" and
            # "microphone", which an end anchor fixes; narrowing the phrase
            # instead dropped "phone agent, has to reply under 800ms" out of
            # real-time altogether, taking its ITL budget with it.
            "phone",
        ),
    ),
    (
        Workload.INTERACTIVE,
        (
            "chat",
            "assistant",
            "copilot",
            "agent",
            "coding",
            "support bot",
            "conversation",
            "interactive",
            "customer",
        ),
    ),
)

#: Phrases implying many requests share a long prefix. The most valuable and
#: least-stated fact about a workload — an agent fleet on one system prompt is
#: routinely 80%+ shared, and nothing in a default config exploits it.
_PREFIX_SIGNALS: tuple[tuple[str, float], ...] = (
    ("same system prompt", 0.85),
    ("shared prompt", 0.85),
    # A FLEET of agents shares a system prompt. One agent shares it with
    # nobody, and bare "agent" claimed 0.7 from "build a support agent for one
    # customer". The workload table still matches the singular — one agent is
    # interactive work; it just is not evidence of a shared prefix.
    ("agent fleet", 0.75),
    ("agents", 0.7),
    ("few-shot", 0.7),
    ("rag", 0.6),
    ("retrieval", 0.6),
    ("template", 0.6),
    ("same instructions", 0.8),
)

_STRUCTURED_SIGNALS = (
    "json",
    "schema",
    "structured",
    "function call",
    "tool call",
    "extract",
    "parse",
    "classification",
    "classify",
)


#: A bulk verb with a volume close behind it. Bare "million" used to be a BATCH
#: signal on its own, which made "chat assistant for 2 million daily active
#: users, needs to feel snappy" an offline batch job — BATCH is checked first,
#: so it beat both "chat" and "assistant". Dropping the volume signal entirely
#: went too far the other way: "score 4 million support tickets" with no
#: "overnight" in it is batch work by any reading. The verb is what separates
#: them. Nothing scores four million tickets while someone waits.
#:
#: The verbs come in two spellings because English does. An open stem
#: (`translate\w*`) never matches "translating" — the silent e is dropped, so
#: "translate" is not a prefix of it, the same trap that made me claim "parse"
#: matched "parsing". Those verbs are written stem-without-e plus an explicit
#: ending list, which is also why they are not `grad\w*`: that would match
#: "gradual" and "gradient".
_BULK_VERB = (
    r"(?:scor|grad|translat|transcrib|annotat|rewrit|summaris|summariz|categoris"
    r"|categoriz)(?:e|es|ed|ing)"
    r"|classif\w*"
    r"|(?:label|tag|rank|process|extract|embed|index)\w*"
)

#: `\d{4,}` alone missed every number a person actually types: "4,000" has no
#: run of four digits in it. Grouped thousands count from 1,000 up, so matching
#: the grouping IS the magnitude test — no second threshold needed. The closing
#: `\b` keeps "million" out of "millionaire".
_VOLUME = r"(?:\d{1,3}(?:,\d{3})+|\d{4,}|million|billion)\b"

_NEAR = r"(?:\s+\S+){0,3}\s+"

#: Both orders, because English has both: "score 4 million tickets" and
#: "4 million tickets to score". The second requires the "to", which is what
#: makes it an instruction rather than a description — without it "2 million
#: users, ranked by activity" would read as a batch job, and it is a sentence
#: about an interactive product.
_BULK_VOLUME = re.compile(
    rf"\b(?:{_BULK_VERB})\b{_NEAR}{_VOLUME}"
    rf"|\b{_VOLUME}{_NEAR}to\s+(?:{_BULK_VERB})\b"
)

#: Needles with no inflection worth catching, so they anchor at both ends. RAG
#: is an acronym — there is no "rags" to match — and leaving its end open made
#: "ragged prompts" and "ragtime" claim retrieval, the same false positive one
#: word further along than the one this anchoring fixed.
_EXACT = frozenset({"rag", "phone"})


def _find(text: str, needles: tuple[str, ...]) -> str | None:
    """First needle present at a word start, so it can be quoted back as evidence.

    Plain containment found "rag" inside "average" and "storage", "parse"
    inside "sparse", and "phone" inside "microphone" — then quoted the needle
    back as the user's own words, which is the one thing `Inference.evidence`
    promises it is. Anchoring the start fixes every one of those.

    The END is deliberately unanchored: these are stems, and "parsing",
    "agents" and "batching" must still match. That leaves word-START
    collisions ("rag" in "ragtime") reachable in principle; they need a
    sentence that opens the colliding word, which is a far narrower door than
    matching anywhere inside any word.
    """
    return next(
        (
            n
            for n in needles
            if re.search(rf"\b{re.escape(n)}\b" if n in _EXACT else rf"\b{re.escape(n)}", text)
        ),
        None,
    )


def _people(text: str) -> tuple[int, str] | None:
    """Concurrency implied by a headcount.

    A team of N does not produce N simultaneous requests — people think, read,
    and go to lunch. Roughly a fifth is the working assumption, floored at 1, and
    it is stated rather than buried so it can be argued with.
    """
    m = re.search(
        r"(\d[\d,]*)\s*(?:\+\s*)?(?:people|engineers?|users?|devs?|developers?|"
        r"employees|staff|agents?|analysts?|seats?)",
        text,
    )
    if not m:
        return None
    n = int(m.group(1).replace(",", ""))
    return max(1, round(n / 5)), m.group(0)


def _explicit_concurrency(text: str) -> tuple[int, str] | None:
    """Concurrency stated outright."""
    m = re.search(
        r"(\d[\d,]*)\s*(?:concurrent|simultaneous|parallel|in flight|in-flight|"
        r"at once|qps|rps|requests? per second)",
        text,
    )
    if not m:
        return None
    return max(1, int(m.group(1).replace(",", ""))), m.group(0)


def _latency(text: str) -> tuple[int, str] | None:
    """A stated response-time budget, normalised to milliseconds."""
    m = re.search(
        r"(?:under|within|below|less than|<)\s*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)",
        text,
    )
    if not m:
        return None
    value = float(m.group(1))
    ms = int(value if m.group(2).startswith("m") else value * 1000)
    return ms, m.group(0)


def _context(text: str) -> tuple[int, str] | None:
    """A stated context length."""
    m = re.search(r"(\d+)\s*k\s*(?:token|context|ctx|window)", text)
    if m:
        return int(m.group(1)) * 1024, m.group(0)
    if (
        m := re.search(r"(?:long|large|big)\s*(?:context|documents?|files?|pdfs?)", text)
    ) is not None:
        return 131_072, m.group(0)
    return None


def read(text: str) -> Intent:
    """Read a plain-language description into requirements.

    Args:
        text: what the user actually typed.

    Returns:
        An [`Intent`] carrying the requirements, the evidence for everything
        inferred, and a question for everything assumed. The requirements are
        always usable — the questions refine, they do not block.
    """
    low = text.lower()
    inferred: list[Inference] = []
    questions: list[Question] = []

    # --- workload -------------------------------------------------------------
    workload = Workload.INTERACTIVE
    # Checked ahead of the signal table for the same reason BATCH sits first in
    # it: a bulk verb over a large volume is the least ambiguous thing a
    # sentence can say about its workload.
    if (bulk := _BULK_VOLUME.search(low)) is not None:
        workload = Workload.BATCH
        inferred.append(Inference("workload", Workload.BATCH.value, bulk.group(0)))
        signal_hit = True
    else:
        signal_hit = False
        for candidate, signals in _WORKLOAD_SIGNALS:
            if (hit := _find(low, signals)) is not None:
                workload = candidate
                signal_hit = True
                inferred.append(Inference("workload", candidate.value, hit))
                break
    if not signal_hit:
        questions.append(
            Question(
                "workload",
                "Is this serving people waiting on a response, or processing a backlog?",
                Workload.INTERACTIVE.value,
                "it decides the engine's scheduling in opposite directions; "
                "getting it wrong wastes roughly half the hardware",
            )
        )

    # --- concurrency ----------------------------------------------------------
    if (explicit := _explicit_concurrency(low)) is not None:
        concurrency, evidence = explicit
        inferred.append(Inference("concurrency", concurrency, evidence))
    elif (headcount := _people(low)) is not None:
        concurrency, evidence = headcount
        inferred.append(
            Inference(
                "concurrency",
                concurrency,
                f"{evidence} — about a fifth in flight at once, since people "
                f"read and think between requests",
            )
        )
    else:
        concurrency = 64 if workload is Workload.BATCH else 4
        questions.append(
            Question(
                "concurrency",
                "How many requests will be in flight at the same time?",
                concurrency,
                "it decides batch caps and whether speculative decoding helps "
                "or actively costs throughput",
            )
        )

    # --- latency budget -------------------------------------------------------
    ttft_ms = itl_ms = None
    if (budget := _latency(low)) is not None:
        ms, evidence = budget
        ttft_ms = ms
        # A per-response budget is not a per-token one. Splitting it needs a
        # length assumption, which is exactly the kind of guess to surface.
        if workload is Workload.REALTIME:
            itl_ms = max(10, ms // 20)
            inferred.append(
                Inference(
                    "itl_ms",
                    itl_ms,
                    f"{evidence} — real-time work is judged per token, so this "
                    f"is the budget spread over roughly 20 tokens",
                )
            )
        inferred.append(Inference("ttft_ms", ms, evidence))
    elif workload is not Workload.BATCH:
        questions.append(
            Question(
                "ttft_ms",
                "How long may the first token take?",
                "unconstrained",
                "without it, scheduling is tuned for tail latency generally "
                "rather than for your actual deadline",
            )
        )

    # --- context --------------------------------------------------------------
    if (ctx := _context(low)) is not None:
        context, evidence = ctx
        inferred.append(Inference("context", context, evidence))
    else:
        context = 32_768
        questions.append(
            Question(
                "context",
                "How long are the prompts?",
                "32k tokens",
                "KV memory is linear in this; over-provisioning it is the "
                "commonest way a model stops fitting",
            )
        )

    # --- prefix sharing -------------------------------------------------------
    prefix_sharing = 0.0
    for phrase, value in _PREFIX_SIGNALS:
        # Through _find, not `phrase in low` — this loop had its own copy of the
        # matching rule, so fixing the one in _find would have left it behind.
        if _find(low, (phrase,)) is not None:
            prefix_sharing = value
            inferred.append(
                Inference(
                    "prefix_sharing",
                    value,
                    f"{phrase} — implies requests share a long prefix, which "
                    f"changes which engine is correct",
                )
            )
            break

    # --- structured output ----------------------------------------------------
    structured = (hit := _find(low, _STRUCTURED_SIGNALS)) is not None
    if structured:
        inferred.append(Inference("structured_output", True, hit))

    return Intent(
        text=text,
        requirements=Requirements(
            workload=workload,
            concurrency=concurrency,
            context=context,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            prefix_sharing=prefix_sharing,
            structured_output=structured,
        ),
        inferred=tuple(inferred),
        questions=tuple(questions),
    )


def demo() -> None:
    """Self-check. Run with `python -m clickllm.intent`."""
    # A coding assistant: interactive, concurrency from headcount, prefix sharing
    # from the word "agent" would be wrong here — check it is not claimed.
    i = read("coding assistant for about 20 engineers, needs to feel snappy")
    assert i.requirements.workload is Workload.INTERACTIVE
    assert i.requirements.concurrency == 4, i.requirements.concurrency
    assert any("20 engineers" in x.evidence for x in i.inferred)

    # Batch beats a careless adjective: "real-time" here modifies the data, and
    # the work is plainly a backlog.
    b = read("score 4 million support tickets overnight, real-time dashboards after")
    assert b.requirements.workload is Workload.BATCH, b.requirements.workload
    assert b.requirements.concurrency == 64, "batch defaults to a real batch"

    # Voice with a stated budget becomes a per-token budget, and says why.
    v = read("voice agent, has to reply in under 800ms")
    assert v.requirements.workload is Workload.REALTIME
    assert v.requirements.ttft_ms == 800
    assert v.requirements.itl_ms == 40, v.requirements.itl_ms
    assert any("spread over roughly 20 tokens" in x.evidence for x in v.inferred)

    # Seconds normalise to milliseconds.
    assert read("chat, respond within 2 seconds").requirements.ttft_ms == 2000

    # Explicit concurrency beats a headcount in the same sentence.
    e = read("50 engineers, expect 30 concurrent requests")
    assert e.requirements.concurrency == 30, e.requirements.concurrency

    # Prefix sharing is the fact nobody states and everybody has.
    a = read("agent fleet, all using the same system prompt")
    assert a.requirements.prefix_sharing >= 0.8, a.requirements.prefix_sharing

    # Structured output is detected from the task, not from a flag.
    assert read("extract JSON from invoices").requirements.structured_output
    assert not read("summarise articles").requirements.structured_output

    # Context, stated and implied.
    assert read("chat over 128k token documents").requirements.context == 128 * 1024
    assert read("summarise long documents").requirements.context == 131_072

    # The honesty rule: what was not said becomes a question with a stated
    # default, never a silent assumption.
    bare = read("run a model")
    assert not bare.confident
    fields = {q.field for q in bare.questions}
    assert {"workload", "concurrency", "context"} <= fields, fields
    assert all(q.default is not None and q.why_it_matters for q in bare.questions)
    # ...and it is still usable without answering anything.
    assert bare.requirements.workload is Workload.INTERACTIVE

    # A fully-specified sentence asks nothing.
    full = read(
        "chat assistant, 30 concurrent requests, under 500ms, 8k token context, "
        "same system prompt every time"
    )
    assert full.confident, [q.field for q in full.questions]

    # Every inference quotes the words that produced it, so a misreading is
    # visible rather than buried in a config.
    for text in ("voice agent under 1s", "batch scoring 2 million rows", "coding copilot"):
        for inf in read(text).inferred:
            # Evidence is `"<the literal words>" — <why that reading follows>`.
            # Only the quote has to appear verbatim; the reasoning is ours.
            quote = inf.evidence.split(" — ")[0]
            assert quote and quote in text.lower(), (text, inf)

    assert "understood:" in read("voice agent").render()
    assert "assuming" in bare.render()

    print("intent: ok")


if __name__ == "__main__":
    demo()
