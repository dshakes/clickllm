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
    #: True when the VALUE, not just the field, was computed rather than read
    #: — "20 agents" implies a headcount, but the concurrency number is a
    #: divide-by-5 guess, and a guess is exactly what is worth still asking
    #: about. False (the default) means the words directly determined the
    #: value, the way a phrase in `_PREFIX_SIGNALS` directly determines
    #: `prefix_sharing` through a lookup rather than arithmetic.
    guess: bool = False

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
    r"|categoriz|analys|analyz|triag)(?:e|es|ed|ing)"
    r"|verif(?:y|ies|ied|ying)"
    r"|classif\w*"
    r"|(?:label|tag|rank|process|extract|embed|index|eval|review|audit|inspect)\w*"
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
    r"(?:users|people|humans|customers|clients|employees|engineers|seats"
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

#: A count the sentence itself calls in-flight is not a pile: "process 1000
#: concurrent requests" states its concurrency, and reading the same number as
#: a backlog picked batch scheduling for a live service — while the
#: concurrency parser, further down, read it correctly.
_INFLIGHT = r"(?:concurrent|simultaneous|parallel|in[- ]flight|at once)"

_NOT_A_BACKLOG = rf"(?:{_AUDIENCE}|{_RATE}|{_INFLIGHT})"

#: What may sit between the number and that noun: up to three adjectives, and
#: nothing that opens a new phrase. Three because _SERVED_AUDIENCE allows
#: three, and two windows for one idea is how "5000 monthly active paying
#: users" ended up a batch job while "for 2 million monthly active paying
#: users" did not. This is the part four rounds of review kept moving.
#: Allowing any two words made "classify 2 million documents for clients" an
#: audience — the counted items are documents and the clients are a phrase
#: later. Allowing none made every adjective ("5000 monthly users") a backlog.
#: A preposition is the boundary between the two, so the gap admits words that
#: are not prepositions. Same rule rescues "score 4 million tickets from our
#: users", which the previous spelling conceded as a known miss.
#: A preposition ends a noun phrase. Shared, so the two windows that must stop
#: at one cannot drift apart.
_PREPOSITION = r"(?:for|of|in|on|per|from|to|with|by|across|via)\b"

_ADJECTIVES = rf"(?:\s+(?!{_PREPOSITION})\w+){{0,3}}"

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
#: A magnitude, however it is written. English spells small counts as words —
#: "score a million support tickets" and "one million tickets to score" are the
#: same instruction as "4 million", and bare "million" used to carry all three.
#: Digit-led only, they fell through to the workload question.
#:
#: One definition, three uses (_VOLUME, _SERVED_AUDIENCE, the backlog-noun
#: branch). Written out three times it would be the shape that put "hrs" in one
#: list and not the other.
_MAGNITUDE = (
    # Any token, or none, in front of the magnitude — rather than a list of
    # number words, which stopped at ten and lost "twenty million". Bare
    # "million" is what main matched; requiring a bulk verb or a work-item
    # noun near it, as _BULK_VOLUME does, is strictly narrower than that.
    r"(?:\S+\s+)?(?:million|billion)s?"
)

_VOLUME = (
    rf"(?:{_MAGNITUDE}\b(?!{_ADJECTIVES}\s+{_RATE}\b)"
    # `(?![,\d])` so the match cannot stop at the first thousands group: for
    # "1,000,000 users" the audience lookahead failed on the whole number, the
    # engine backtracked to "1,000", and a served audience became a batch job
    # with "rank content for 1,000" quoted as the evidence.
    rf"|(?:\d{{1,3}}(?:,\d{{3}})+|\d{{4,}})\b(?![,\d])(?!{_ADJECTIVES}\s+{_NOT_A_BACKLOG}\b))"
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
#: Needles with no inflection worth catching, so they anchor at both ends. RAG
#: is an acronym — there is no "rags" to match — and leaving its end open made
#: "ragged prompts" and "ragtime" claim retrieval, the same false positive one
#: word further along than the one this anchoring fixed.
_EXACT = frozenset({"rag", "phone"})

#: How far a realtime word reaches forward to govern a verb: "voice scoring",
#: "real-time support bot scoring". Counted in WORDS, which is what this
#: comment always said — implemented as 12 characters it stopped inside
#: "real-time support bot scoring 5000 calls", a voice product, and made it a
#: batch job. The window is part of the pattern now, so the number and the
#: unit cannot disagree again.
_GOVERNS_WORDS = 3

#: Mode words that describe the SERVICE and nothing else, so they may govern a
#: bulk verb near them. Every realtime signal qualifies. From the interactive
#: table only "interactive" and "copilot" do: "chat", "agent", "customer" and
#: "conversation" double as descriptions of the ITEMS ("chat transcripts",
#: "customer tickets"), and letting those govern made real backlogs
#: interactive — that is round six of this PR, and it is why the list is
#: explicit rather than "the whole table".
_GOVERNING_MODES = (*_WORKLOAD_SIGNALS[1][1], "interactive", "copilot")

_REALTIME_PHRASE = re.compile(
    "(?:"
    + "|".join(
        rf"\b{re.escape(w)}\b" if w in _EXACT else rf"\b{re.escape(w)}" for w in _GOVERNING_MODES
    )
    # The reach stops at a preposition, for the reason _ADJECTIVES does: in
    # "real-time dashboard FOR scoring 4 million tickets" the realtime word
    # describes the dashboard, and the LLM work is still a batch job.
    + rf")(?:\s+(?!{_PREPOSITION})\w+){{0,{_GOVERNS_WORDS}}}"
)

#: A mode ADVERB in the final position governs the sentence: "score 5000
#: accounts interactively" is an interactive product described back to front.
#:
#: An adverb, not any trailing mode word — that spelling erased real backlogs
#: in "score 4 million support tickets with copilot" and "...tickets realtime",
#: where the word is a preposition's object or an appended noun and modifies
#: nothing. In practice this is a rule about "interactively", and saying so is
#: better than a general-looking pattern that generalises the wrong way.
#:
#: Final position only — reaching backwards a few words instead would swallow
#: "score 4 million tickets, realtime dashboard after", a different clause
#: about a different thing.
_TRAILING_MODE = re.compile(
    "(?:" + "|".join(rf"\b{re.escape(w)}" for w in _GOVERNING_MODES) + r")\w*ly\b\s*[.!?]?\s*$"
)

#: Behind a preposition, these are an audience too — "classification API FOR
#: 5000 accounts" serves account holders. They are absent from _AUDIENCE
#: itself, and deliberately: as a direct object they are records, and
#: "reconcile 20000 accounts" is real batch work. The position is what
#: separates the two readings, which is exactly what this pattern encodes.
_SERVED_ONLY = r"(?:accounts|tenants|orgs|organisations|organizations|teams|workspaces)"

_SERVED_AUDIENCE = re.compile(
    # Any count, not just a magnitude: it is the SERVED POSITION that decides,
    # and "for 5000 accounts" is as much an audience as "for 2 million".
    # One word between the preposition and the count, not three: "to score for
    # 5000 users" reached across an infinitive and swallowed the backlog in
    # "4 million tickets to score for 5000 users". A preposition introduces
    # the number it governs; it does not reach over a verb to find one.
    rf"\b(?:for|serving|across|to)\b(?:\s+\w+){{0,1}}\s+(?:{_MAGNITUDE}|\d[\d,]*)"
    rf"(?:\s+\w+){{0,3}}\s+(?:{_AUDIENCE}|{_SERVED_ONLY})\b"
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
    r"(?:tickets|documents|docs|records|rows|messages|emails|transcripts|prompts"
    r"|invoices|orders|claims|receipts|transactions|cases|reports|forms"
    # requests/queries are backlog nouns as well as rate units — a corpus of
    # captured requests is what this tool processes. The branch that uses this
    # carries a (?!_PER) guard, so "2 million requests per second" is still a
    # rate and only the bare noun reads as a pile.
    r"|requests|queries|prompts|completions|responses"
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
    rf"|\b{_MAGNITUDE}\b{_ADJECTIVES}\s+{_BACKLOG_NOUN}\b(?!{_PER})"
)


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
        r"(\d[\d,]*)\s*(?:\+\s*)?(?:(?:people|employees)\b"
        r"|(?:engineer|user|dev|developer|agent|seat|analyst)s\b"
        # "staff" is a collective plural, so it modifies like a singular:
        # "20000 staff records" is records. Same guard, same reason.
        rf"|staff\b(?!\s+(?!{_PHRASE_END}\b)\w)"
        rf"|(?:engineer|user|dev|developer|agent|seat|analyst)\b"
        rf"(?!\s+(?!{_PHRASE_END}\b)\w))",
        text,
    )
    if not m:
        return None
    n = _digits(m.group(1))
    if n is None:
        return None
    whole = _whole(n / 5)
    return (whole, m.group(0)) if whole else None


#: Beyond this, a number is not an answer. A billion requests in flight, or a
#: latency budget of eleven days, is a typo or an adversarial input — and both
#: are better as the question than as a claim. It also keeps arbitrary-precision
#: integers away from float arithmetic: `10**300 / 5` raises OverflowError.
_SANE_MAX = 10**9


def _digits(raw: str) -> int | None:
    """A captured digit string as an int, or None when it is not usable.

    Python 3.11 caps int() at 4300 digits (the CVE-2020-10735 mitigation), so a
    long enough run of digits in a prompt raises ValueError before any bound
    can look at the value — `int()` is not the safe conversion I had been
    treating it as. A string longer than _SANE_MAX is out of range by
    definition, so one length check answers both questions without converting.
    """
    clean = raw.replace(",", "")
    return None if len(clean) > len(str(_SANE_MAX)) else int(clean)


def _whole(n: float | int) -> int | None:
    """A usable whole number, or None — the ONE place this module converts.

    Guarding the inputs was not enough three times running: the overflow is
    created by the arithmetic between the check and the conversion, so
    `math.isfinite` on the operands says nothing about the product. Every
    `round`/`ceil` in this file now happens here, and a test asserts that.

    Ceiling, not round(): banker's rounding sent 16.5 to 16, and rounding a
    concurrency up reserves more KV, which is the conservative direction for a
    sizing tool.
    """
    # NaN first, because the bound cannot see it: every comparison with NaN is
    # False, so `n > _SANE_MAX` waves it through to ceil(), which raises.
    # Infinity is caught by either, and is caught here.
    if isinstance(n, float) and not math.isfinite(n):
        return None
    if n > _SANE_MAX:
        return None
    return max(1, math.ceil(n))


def _explicit_concurrency(text: str) -> tuple[int, str] | None:
    """Concurrency stated outright — in flight at the same time, said as such."""
    m = re.search(
        r"(\d[\d,]*)\s*(?:concurrent|simultaneous|parallel|in flight|in-flight|at once)",
        text,
    )
    if not m:
        return None
    n = _digits(m.group(1))
    whole = _whole(n) if n is not None else None
    return (whole, m.group(0)) if whole else None


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
        r"(?<![\d.])(?P<n>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>million|billion)?\s*(?:"
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


_DURATION_UNIT = r"(?:ms|milliseconds?|s|secs?|seconds?|min|mins|minutes?)"


def _service_time(text: str) -> tuple[int, str] | None:
    """How long a request stays in the system, in ms, when the sentence says.

    NOT the latency budget. `_latency` reads a time-to-first-token target, and
    a request lives longer than its first token — using it here produced a
    concurrency that was a lower bound, which under-sizes KV and is the one
    direction that makes a deployment look feasible when it is not. So this
    matches a stated duration and nothing else, and without one the question
    gets asked.
    """
    # The subject is REQUIRED. Optional, it matched "first token takes 2
    # seconds" and "startup takes 2 seconds" — the first being exactly the
    # figure the docstring above says must not be used as time-in-system.
    # The unit nouns match the ones _RATE_UNIT accepts: "100 events/sec, each
    # event takes 1 second" derived nothing while the same sentence about
    # "calls" worked, because two lists of the same units had drifted apart.
    unit = r"(?:request|call|job|generation|response|message|event|item|one)"
    subject = rf"(?:(?:each|every)(?:\s+{unit})?|(?:a|the|one)\s+{unit})"
    m = re.search(
        rf"{subject}\s+(?:can\s+|may\s+|will\s+|usually\s+|typically\s+)?"
        rf"(?:takes?|taking|lasts?|runs? for)\s*(?:about|around|roughly|up to)?\s*"
        rf"(\d+(?:\.\d+)?)\s*({_DURATION_UNIT})\b"
        rf"|(\d+(?:\.\d+)?)\s*({_DURATION_UNIT})\s+per\s+(?:request|call|job)\b",
        text,
    )
    if not m:
        return None
    n = float(m.group(1) or m.group(3))
    unit = (m.group(2) or m.group(4)).rstrip("s")
    per_ms = {"m": 60_000, "min": 60_000, "minute": 60_000, "s": 1000, "sec": 1000, "second": 1000}
    ms = n if unit.startswith("m") and unit not in ("min", "minute") else n * per_ms.get(unit, 1000)
    whole = _whole(ms)
    return (whole, m.group(0)) if whole else None


def _little(rate_per_second: float, service_ms: int) -> int | None:
    """Little's Law: in flight = arrival rate x time spent in the system.

    None when the product is not a real number. Guarding the two inputs was
    not enough and I shipped that twice: 1e300 and 1000 are both finite, and
    their product is not. The overflow is in the multiplication, so the check
    is on the multiplication.

    Ceiling, not round(): banker's rounding sent 16.5 to 16, and more
    concurrency reserves more KV, which is the conservative direction for a
    sizing tool.
    """
    return _whole(rate_per_second * service_ms / 1000)


def _latency(text: str) -> tuple[int, str] | None:
    """A stated response-time budget, normalised to milliseconds."""
    m = re.search(
        r"(?:under|within|below|less than|<)\s*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)",
        text,
    )
    if not m:
        return None
    value = float(m.group(1))
    ms = _whole(value if m.group(2).startswith("m") else value * 1000)
    return (ms, m.group(0)) if ms else None


#: A bare token count next to any of these describes what comes OUT of the
#: model, not what goes in. "500 tokens" alone is ambiguous and still reads
#: as prompt length, same as before this existed — this only excludes the
#: clauses that name which side of the request they mean.
_OUTPUT_LENGTH_WORDING = re.compile(
    r"\b(?:response|responses|output|outputs|completion|completions|reply|replies|answer|answers)\b"
)


def _context(text: str) -> tuple[int, str] | None:
    """A stated context length. The LAST one stated wins.

    Position matters because `Session` accumulates every turn into one string,
    so a correction arrives appended to what it corrects. Returning the first
    match by branch order made "a few thousand tokens" then "actually make that
    16000 tokens" keep 4096 — the correction parsed and was then discarded.
    Candidates carry their offset and the rightmost wins, so a later sentence
    overrides an earlier one whichever branch matched.
    """
    found: list[tuple[int, int, str]] = []  # (offset, context, evidence)

    # Through _digits like the other two captures: a 5000-digit "k tokens"
    # raised ValueError out of read(). Through _whole as well: _digits bounds
    # the CAPTURE and the x1024 happens after it, so a large-but-legal "k"
    # produced a context of 10^12 tokens that fed straight into sizing as a
    # confident requirement. The guard has to be on the value that leaves.
    for m in re.finditer(r"(\d+)\s*k\s*(?:token|context|ctx|window)", text):
        k = _digits(m.group(1))
        ctx = _whole(k * 1024) if k else None
        if ctx:
            found.append((m.start(), ctx, m.group(0)))

    # The worded forms the question itself offers. "a few thousand tokens" is
    # the tool's own suggested wording and parsed to nothing at all.
    for pat, size in (
        (r"tens of thousands\s*(?:of\s*)?(?:token|word)", 32768),
        (r"(?:a\s*)?few\s*thousand\s*(?:token|word)", 4096),
        (r"(?:a\s*)?couple\s*(?:of\s*)?thousand\s*(?:token|word)", 2048),
    ):
        for m in re.finditer(pat, text):
            ctx = _whole(size)
            if ctx:
                found.append((m.start(), ctx, m.group(0)))

    # Scoped per clause, not per sentence: "about 2000 tokens in, 300 out"
    # must read the prompt length off the first clause and never reach the
    # second. A bare count qualified by response/output/completion wording
    # is a completion length, not a prompt length — "responses are about 500
    # tokens" would otherwise infer a context window from the answer, not the
    # question, and suppress the context question on a field it never read.
    # The comma only splits clauses apart when it is not a thousands
    # separator: "in, 300 out" is two clauses, but "2,000 tokens" is one
    # number, told apart by whether a digit follows immediately.
    #
    # `(?!{_PER})` because a throughput rate is not a context length: without
    # it "2000 tokens/sec" set a 2048 context, a performance requirement
    # re-read as a prompt size.
    offset = 0
    for clause in re.split(r"[.;!?\n]|,(?!\d)", text):
        # `tokens?`/`toks?` spelled out, not `token\b`: the word boundary after
        # "tok" falls before the plural "s", and `_PER`'s own `ss?` alternative
        # then matched that "s" — so every bare "2000 tokens" looked like a
        # rate and was rejected.
        # `\b` AFTER the alternation, not inside it. Without the boundary the
        # engine backtracks: blocked at "tokens", it retries the shorter
        # "token", the lookahead then sees "s/sec" rather than "/sec", `_PER`
        # does not match, and "2000 tokens/sec" was read as a 2048 context —
        # a throughput rate re-read as a prompt size, which is the defect this
        # guard exists to stop.
        m = re.search(rf"(\d[\d,]*)\s*(?:tokens|token|toks|tok)\b(?!{_PER})", clause)
        if m is not None and not _OUTPUT_LENGTH_WORDING.search(clause):
            n = _digits(m.group(1))
            # The same 3-to-7-digit floor, applied to the parsed value rather
            # than the raw digit run — that run breaks on the comma in
            # "2,000 tokens" and matched "000" out of it instead.
            if n is not None and 100 <= n <= 9_999_999:
                window = 1024
                # The cap must clear the accepted range above, not a round
                # number that happens to look like one: 1_048_576 sat below
                # 9_999_999, so "5000000 tokens" rounded DOWN to roughly a
                # fifth of what was stated — a stated prompt length silently
                # became smaller than itself. 16_777_216 is the next
                # power-of-two at or above the top of the accepted range.
                while window < n and window < 16_777_216:
                    window *= 2
                ctx = _whole(window)
                if ctx:
                    found.append((offset + m.start(), ctx, m.group(0)))
        offset += len(clause) + 1

    if found:
        _, ctx, evidence = max(found, key=lambda c: c[0])
        return ctx, evidence

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
    # A stated REALTIME mode outranks the inference. Only realtime: those words
    # ("voice", "real-time", "speech", "phone") describe the service and
    # nothing else, so "voice scoring 5000 calls under 200ms" is a voice
    # product. The INTERACTIVE words do not get this, because they double as
    # descriptions of the ITEMS — "chat transcripts", "customer tickets",
    # "agent logs" — and giving it to them made those backlogs interactive.
    #
    # This only disables _BULK_VOLUME. The signal table still checks BATCH
    # first, so "score 4 million tickets overnight, real-time dashboards after"
    # is still batch on "overnight".
    served = [m.span() for m in _SERVED_AUDIENCE.finditer(low)]
    # A realtime word suppresses the bulk match only when it GOVERNS it —
    # immediately before the verb, as in "voice scoring 5000 calls". Applied to
    # the whole sentence it also caught "score 4 million tickets, realtime
    # dashboard after", where the realtime phrase is a different clause about a
    # different thing. Same scoping mistake as the sentence-wide served-audience
    # guard two commits ago, so the same fix: a span, not a flag.
    served += [rt.span() for rt in _REALTIME_PHRASE.finditer(low)]
    if _TRAILING_MODE.search(low):
        served.append((0, len(low)))
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
    service = _service_time(low)
    if (explicit := _explicit_concurrency(low)) is not None:
        # A stated in-flight count beats a derived one, the same way a stated
        # workload beats an inferred one: "100 concurrent requests at 10 qps
        # under 200ms" says 100, and the derivation said 2.
        concurrency, evidence = explicit
        inferred.append(Inference("concurrency", concurrency, evidence))
    elif rate is not None and service is not None and (little := _little(rate[0], service[0])):
        concurrency = little
        inferred.append(
            Inference(
                "concurrency",
                concurrency,
                f"{rate[1]} x {service[1]} — Little's Law: arrivals x time "
                f"in the system. A rate alone gives no number at all, and the "
                f"latency budget is not this number: it is time to first "
                f"token, and a request outlives its first token",
                guess=True,
            )
        )
    elif (headcount := _people(low)) is not None:
        concurrency, evidence = headcount
        inferred.append(
            Inference(
                "concurrency",
                concurrency,
                f"{evidence} — about a fifth in flight at once, since people "
                f"read and think between requests",
                guess=True,
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
                    guess=True,
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

    # Past _SANE_MAX a number is a question, not a claim. The literal is
    # deliberate: written as `_SANE_MAX + 1` this check would move with the
    # constant and notice nothing, which is the shape the demo ratchet exists
    # to catch.
    over = read("2000000000 concurrent")
    assert over.requirements.concurrency == 4, over.requirements.concurrency
    assert "concurrency" in {q.field for q in over.questions}
    assert read("900000000 concurrent").requirements.concurrency == 900_000_000

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
