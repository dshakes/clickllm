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

import math
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

#: Nouns that make a number something other than a backlog: an audience, or a
#: rate. "5000 monthly users" is who it serves; "10000 requests per second" is
#: how fast; "4 million tickets" is the pile of work.
#:
#: **Plural, for the audience half, and that is what does the work.** An
#: English noun modifying another noun is singular — "customer tickets", "user
#: records" — so a plural audience noun is the head of its phrase while a
#: singular one describes the items.
#:
#: "accounts" is deliberately absent — reconciling 20,000 accounts is real
#: batch work, unlike 20,000 subscribers.
_AUDIENCE = (
    r"(?:users|people|humans|customers|clients|employees|engineers|staff|seats"
    r"|subscribers|members|players|students|patients|developers|devs)"
)

#: A rate needs its denominator. "requests" and "queries" name the units of a
#: rate AND the units of a backlog — and a corpus of captured requests is the
#: thing this product exists to process, so "classify 1 million requests from
#: captured traffic" is as central a sentence as clickllm has. Excluding the
#: bare noun took that out. Only "per <something>" makes it a rate; qps/rps/tps
#: carry their denominator in the word.
#: The unit noun, and then a denominator in any of the spellings people use:
#: "per second", "/sec", "/s". Without the slash forms "score 4000
#: requests/sec under 200ms" read as a backlog of four thousand.
_RATE_UNIT = r"(?:requests?|queries|calls|messages|events|req|msgs?|evts?)"
#: A denominator, and only a TIME one. `per \w+` accepted "per customer" and
#: "per user", which are ratios rather than rates: "classify 10000 requests per
#: customer from captured traffic" is a backlog described per head. The slash
#: form was already restricted; the spelled-out form was not, because I wrote
#: it first and widened the wrong one.
#: The denominators, and the seconds in each. The pattern is GENERATED from
#: this map rather than written beside it: they were two lists, and "hrs"
#: existed in the pattern and not in the map, so "3600 requests per hrs" fell
#: back to per-second — a 3600x over-provision from one missing key.
_SECONDS_IN = {
    "s": 1,
    "sec": 1,
    "second": 1,
    "min": 60,
    "minute": 60,
    "hr": 3600,
    "hour": 3600,
    "day": 86400,
}
_TIME = "(?:" + "|".join(sorted((k + "s?" for k in _SECONDS_IN), key=len, reverse=True)) + ")"
_PER = rf"(?:\s+per\s+{_TIME}\b|\s*/\s*{_TIME}\b)"

_RATE = rf"(?:qps|rps|tps|{_RATE_UNIT}{_PER})"

_NOT_A_BACKLOG = rf"(?:{_AUDIENCE}|{_RATE})"

#: What may sit between the number and that noun: adjectives, and nothing that
#: opens a new phrase. This is the part four rounds of review kept moving.
#: Allowing any two words made "classify 2 million documents for clients" an
#: audience — the counted items are documents and the clients are a phrase
#: later. Allowing none made every adjective ("5000 monthly users") a backlog.
#: A preposition is the boundary between the two, so the gap admits words that
#: are not prepositions. Same rule rescues "score 4 million tickets from our
#: users", which the previous spelling conceded as a known miss.
_ADJECTIVES = (
    r"(?:\s+(?!for\b|of\b|in\b|on\b|per\b|from\b|to\b|with\b|by\b|across\b|via\b)\w+){0,2}"
)

#: `\d{4,}` alone missed every number a person actually types: "4,000" has no
#: run of four digits in it. Grouped thousands count from 1,000 up, so matching
#: the grouping IS the magnitude test — no second threshold needed. The closing
#: `\b` keeps "million" out of "millionaire", and the lookahead keeps a user
#: count from reading as a backlog: "scoring app for 5000 users" is a product,
#: not a batch job.
#: Two volumes, differing in whether people can be the work items.
#:
#: A plain count of people is an audience — "scoring app for 5000 users" is a
#: product. Millions of people are not an audience for anything interactive;
#: at that magnitude the people ARE the records, which is why "rank 4 million
#: customers by priority" and "score 4 million users for churn risk" are batch
#: jobs and "chat tool for ranking 5000 users" is not. So the audience
#: exclusion applies to digit counts and not to million/billion.
#:
#: A RATE is excluded at any magnitude — it has a denominator, and nothing
#: with a denominator is a pile.
_VOLUME = (
    rf"(?:\d[\d,.]*\s+(?:million|billion)\b(?!{_ADJECTIVES}\s+{_RATE}\b)"
    rf"|(?:\d{{1,3}}(?:,\d{{3}})+|\d{{4,}})\b(?!{_ADJECTIVES}\s+{_NOT_A_BACKLOG}\b))"
)

#: "…for 2 million users", "…across our 2 million customers", "…for about 2
#: million daily active paying users" — the sentence says who is being SERVED,
#: and that beats any bulk verb elsewhere in it, at any magnitude. A plain
#: count of people is already excluded by _VOLUME; this covers the millions,
#: where the people could equally have been records.
#:
#: A separate pattern rather than a lookbehind on _VOLUME: the preposition is
#: several words from the numeral ("for about 2 million"), lookbehinds must be
#: fixed-width, and enumerating "for about "/"across our "/… as fixed widths is
#: how the previous spelling missed all three phrasings the reviewer sent.
_SERVED_AUDIENCE = re.compile(
    rf"\b(?:for|serving|across|to)\b(?:\s+\w+){{0,3}}\s+\d[\d,.]*\s+(?:million|billion)"
    rf"(?:\s+\w+){{0,3}}\s+{_AUDIENCE}\b"
)

#: Words that cannot be the noun a singular people-noun modifies. Seeing one
#: means the headcount was the head of its phrase after all: "1 developer WHO
#: wants to work offline" is one developer, while "20000 user RECORDS" is
#: records. A bare `(?!\s+\w)` could not tell those apart and rejected every
#: singular headcount that had anything at all after it — invisible because
#: every test case I wrote put the noun at the end of the string.
_PHRASE_END = (
    r"(?:who|that|which|whom|and|or|but|with|to|for|of|in|on|at|by|from|per"
    r"|using|needs?|wants?|will|would|can|only|each|about|max|maximum|plus)"
)

#: Nouns that are work items rather than people. A million of them is a
#: backlog whatever the verb: "triage 2 million support tickets" and
#: "summarization of 2 million support tickets" are batch jobs, and neither
#: verb is in _BULK_VERB — which is the trouble with a verb whitelist, since
#: English has more verbs than I will think of. Bare "million" used to carry
#: these, and removing it dropped them.
#:
#: Plural, for the reason _AUDIENCE is: a singular noun here is a modifier.
_BACKLOG_NOUN = (
    r"(?:tickets|documents|docs|records|rows|messages|emails|transcripts"
    r"|conversations|items|files|images|photos|articles|reviews|logs|events"
    r"|pages|posts|comments|chunks|entries|samples|examples|labels|pairs"
    r"|utterances|snippets|abstracts|papers|listings|products|sessions)"
)

_NEAR = r"(?:\s+\S+){0,3}\s+"

#: Both orders, because English has both: "score 4 million tickets" and
#: "4 million tickets to score". The second requires the "to", which is what
#: makes it an instruction rather than a description — without it "2 million
#: users, ranked by activity" would read as a batch job, and it is a sentence
#: about an interactive product.
_BULK_VOLUME = re.compile(
    rf"\b(?:{_BULK_VERB})\b{_NEAR}{_VOLUME}"
    rf"|\b{_VOLUME}{_NEAR}to\s+(?:{_BULK_VERB})\b"
    # No verb needed when the noun says it: a million tickets is a pile.
    # ...unless that noun carries a denominator. "2 million events/sec" is a
    # rate whose unit happens to be a work item, and the noun branch skipped
    # the rate test the other two branches get from _VOLUME.
    rf"|\b\d[\d,.]*\s+(?:million|billion)\b{_ADJECTIVES}\s+{_BACKLOG_NOUN}\b(?!{_PER})"
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

    A singular people-noun followed by another word is a MODIFIER, not a
    headcount — the same rule `_NOT_A_BACKLOG` runs on: "20000 user records"
    counts records. Without the lookahead it read as four thousand concurrent
    people and skipped the concurrency question entirely, which is a worse
    outcome than the batch default it replaced.
    """
    m = re.search(
        # Every branch ends at a word boundary: without one, "20 userspace
        # processes" matched "20 users" and "300 staffordshire branches"
        # matched "300 staff" — the same prefix-matching defect this PR fixed
        # in _find, in the function next door.
        r"(\d[\d,]*)\s*(?:\+\s*)?(?:(?:people|employees|staff)\b"
        r"|(?:engineer|user|dev|developer|agent|seat|analyst)s\b"
        rf"|(?:engineer|user|dev|developer|agent|seat|analyst)\b"
        rf"(?!\s+(?!{_PHRASE_END}\b)\w))",
        text,
    )
    if not m:
        return None
    n = int(m.group(1).replace(",", ""))
    return max(1, round(n / 5)), m.group(0)


def _explicit_concurrency(text: str) -> tuple[int, str] | None:
    """Concurrency stated outright — in flight at the same time, said as such."""
    m = re.search(
        r"(\d[\d,]*)\s*(?:concurrent|simultaneous|parallel|in flight|in-flight|at once)",
        text,
    )
    if not m:
        return None
    return max(1, int(m.group(1).replace(",", ""))), m.group(0)


def _rate_per_second(text: str) -> tuple[float, str] | None:
    """A throughput rate, normalised to per-second. NOT a concurrency.

    Concurrency in this codebase is requests in flight at the same time, and a
    rate does not give you that on its own — "120 requests per minute, each
    taking 30 seconds" is two per second and about SIXTY in flight. Dividing
    the rate by its denominator and calling the answer concurrency was wrong
    by the service time, which is a factor this module has no way to know
    unless the sentence states one. See `_little`.
    """
    m = re.search(
        r"(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>million|billion)?\s*(?:"
        rf"(?:qps|rps|tps)|{_RATE_UNIT}(?:\s+per\s+|\s*/\s*)(?P<unit>{_TIME})\b)",
        text,
    )
    if not m:
        return None
    try:
        n = float(m.group("n").replace(",", ""))
    except ValueError:  # a malformed span like "1..5" — refuse rather than raise
        return None
    n *= {"million": 1e6, "billion": 1e9}.get(m.group("scale") or "", 1)
    per = _SECONDS_IN.get((m.group("unit") or "s").rstrip("s") or "s")
    if per is None:  # a denominator we cannot convert is a question, not a guess
        return None
    return n / per, m.group(0)


def _little(rate_per_second: float, service_ms: int) -> int:
    """Little's Law: in flight = arrival rate x time spent in the system.

    Ceiling, not round(): banker's rounding sent 16.5 to 16, and more
    concurrency reserves more KV, which is the conservative direction for a
    sizing tool.
    """
    return max(1, math.ceil(rate_per_second * service_ms / 1000))


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
    # A bulk verb over a volume of WORK ITEMS. The volume itself carries the
    # discrimination (see _AUDIENCE), which is why there is no rule here about
    # stated modes outranking inferred ones: that rule read "score 4 million
    # chat transcripts" as interactive on the word "chat", and it existed only
    # to rescue sentences whose volume counted users. Fixing the volume made it
    # dead weight.
    # Scoped to the span, not the sentence: "classify 1 million requests from
    # captured traffic for 2 million customers" states a backlog AND who it is
    # for, and only the second volume is the audience. Suppressing on any
    # served phrase anywhere lost the first one.
    served = [m.span() for m in _SERVED_AUDIENCE.finditer(low)]
    bulk = next(
        (
            m
            for m in _BULK_VOLUME.finditer(low)
            if not any(m.start() < e and s < m.end() for s, e in served)
        ),
        None,
    )
    if bulk is not None:
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
    # The latency budget is read first because a rate needs it: a rate is a
    # concurrency statement only once you know how long each request stays in
    # the system, and without that this used to answer the question anyway —
    # and remove it, so nobody could correct the answer.
    budget = _latency(low)
    rate = _rate_per_second(low)
    if rate is not None and budget is not None:
        concurrency = _little(rate[0], budget[0])
        inferred.append(
            Inference(
                "concurrency",
                concurrency,
                f"{rate[1]} at {budget[1]} — Little's Law: arrivals x time in "
                f"the system. A rate alone does not give this",
            )
        )
    elif (explicit := _explicit_concurrency(low)) is not None:
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
    if budget is not None:
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
