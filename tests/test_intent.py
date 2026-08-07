"""What the sentence actually said.

`intent.demo()` walks sentences that classify correctly. These pin the ones
that classified confidently and wrongly — a signal matched inside an unrelated
word, then quoted back as the user's own words. `Inference.evidence` promises
"the literal words that produced this", so a match the user never wrote is not
a near miss; it is the field lying about its own contents.
"""

from __future__ import annotations

import pathlib

import pytest

from clickllm.intent import _SANE_MAX, _digits, _whole, read
from clickllm.plan import Workload


def infer(text: str, field: str):
    """The inference for one field, or None if it was never claimed."""
    return next((i for i in read(text).inferred if i.field == field), None)


@pytest.mark.parametrize(
    ("text", "word"),
    [
        ("chat assistant, average request is short, needs low latency", "average"),
        ("needs fast local storage for model weights", "storage"),
        ("write one paragraph per record", "paragraph"),
    ],
)
def test_rag_is_not_found_inside_an_ordinary_word(text, word):
    # "rag" lives inside av-e-RAG-e and sto-RAG-e. Matching it there set
    # prefix_sharing to 0.6 — which feeds engine selection — and rendered
    # "rag" as the evidence for a sentence that never mentions retrieval.
    assert word in text
    assert read(text).requirements.prefix_sharing == 0.0
    assert infer(text, "prefix_sharing") is None


def test_parse_is_not_found_inside_sparse():
    text = "summarise articles using a sparse mixture-of-experts model"
    assert not read(text).requirements.structured_output
    assert infer(text, "structured_output") is None


def test_phone_is_not_found_inside_microphone():
    assert (
        read("transcribe microphone input to a file").requirements.workload is not Workload.REALTIME
    )


def test_the_stems_these_signals_are_written_as_still_match():
    # The anchor is on the START of the needle only, so the inflections have to
    # survive it — otherwise the fix trades one wrong answer for another. Each
    # of these has exactly one signal in it, so a pass means that signal matched
    # its inflected form and not some other word in the sentence.
    assert read("batching jobs").requirements.workload is Workload.BATCH
    assert read("run it over our datasets").requirements.workload is Workload.BATCH
    assert read("a fleet of agents").requirements.prefix_sharing == 0.7
    assert read("we reuse templates").requirements.prefix_sharing == 0.6


@pytest.mark.parametrize("text", ["ragged prompts", "ragtime music assistant"])
def test_rag_anchors_at_both_ends_because_there_is_no_plural_of_an_acronym(text):
    assert read(text).requirements.prefix_sharing == 0.0
    # Not a blanket end-anchor, though: the acronym itself still reads.
    assert read("a rag pipeline over our docs").requirements.prefix_sharing == 0.6


def test_phonebook_is_not_a_real_time_workload():
    assert read("phonebook search assistant").requirements.workload is Workload.INTERACTIVE
    assert read("transcribe microphone input").requirements.workload is not Workload.REALTIME
    # Fixed by anchoring both ends of "phone", not by narrowing it to "phone
    # call" — that spelling dropped the commonest real-time phrasing there is,
    # and its ITL budget with it.
    assert read("phone agent, has to reply under 800ms").requirements.workload is Workload.REALTIME
    assert read("voice agent on a phone call").requirements.workload is Workload.REALTIME


def test_one_agent_is_not_a_fleet():
    # The comment justifying 0.7 says "an agent fleet on one system prompt".
    # One agent shares a prefix with nobody.
    single = read("build a support agent for one customer")
    assert single.requirements.prefix_sharing == 0.0
    # Still interactive work — it is the prefix claim that was unfounded.
    assert single.requirements.workload is Workload.INTERACTIVE
    assert read("an agent fleet on one prompt").requirements.prefix_sharing == 0.75


def test_evidence_is_a_word_the_user_wrote_not_a_fragment_inside_one():
    # demo()'s own check — evidence is a substring of the input — is satisfied
    # by the bug that produced it. This one requires the match to begin a word.
    for text in (
        "chat assistant, average request is short",
        "needs fast local storage",
        "summarise with a sparse model",
        "transcribe microphone input",
    ):
        for i in read(text).inferred:
            for word in str(i.evidence).split():
                assert any(w.startswith(word) for w in text.lower().split()), (text, i)


# --- the workload table -------------------------------------------------------
#
# One table rather than a test per fix. This file took eight review rounds, and
# every round was a rule that fixed its own example and broke someone else's —
# so the cases live together where a new rule has to face all of them at once.
# Adding a row is the cost of a new claim about this classifier.

WORKLOADS = [
    # A bulk verb over a volume of work items, in either order, however the
    # verb and the number are spelled.
    (Workload.BATCH, "score 4 million support tickets"),
    (Workload.BATCH, "4 million support tickets to score"),
    (Workload.BATCH, "2 million documents to classify"),
    (Workload.BATCH, "30000 rows to embed"),
    (Workload.BATCH, "translating 5 million tickets"),  # silent-e gerund
    (Workload.BATCH, "scoring 4 million tickets"),
    (Workload.BATCH, "transcribing 900000 calls"),
    (Workload.BATCH, "summarising 30000 reports"),
    (Workload.BATCH, "score 4,000 tickets"),  # grouped digits
    (Workload.BATCH, "rank 1,500 answers"),
    (Workload.BATCH, "score 4 million tickets overnight"),
    # A million work items is a backlog with no verb from the whitelist at
    # all — which is the trouble with a verb whitelist. Bare "million" used to
    # carry these and removing it dropped them.
    (Workload.BATCH, "triage 2 million support tickets"),
    (Workload.BATCH, "2 million support tickets to triage"),
    (Workload.BATCH, "summarization of 2 million support tickets"),
    # clickllm's own vocabulary: evaluating a captured corpus is the thing
    # this tool exists to do, and none of it was in the verb or noun lists.
    (Workload.BATCH, "evaluate 1 million captured prompts"),
    (Workload.BATCH, "eval 1 million captured requests"),
    (Workload.BATCH, "2 million requests to evaluate"),
    # English spells small counts as words, and bare "million" used to carry
    # all of these. Digit-led only, they fell through to the question.
    (Workload.BATCH, "score a million support tickets"),
    (Workload.BATCH, "score one million support tickets"),
    (Workload.BATCH, "a million tickets to score"),
    (Workload.BATCH, "two million tickets to triage"),
    # Any spelling of the count, not a list of number words that stopped at ten.
    (Workload.BATCH, "score twenty million support tickets"),
    (Workload.BATCH, "triage fifty million tickets"),
    (Workload.BATCH, "score millions of tickets"),
    (Workload.INTERACTIVE, "rank content for a million users under 200ms"),
    # ...unless the noun carries a denominator. A rate whose unit happens to
    # be a work item is still a rate.
    (Workload.REALTIME, "realtime classify 2 million events/sec under 200ms"),
    # A stated REALTIME mode outranks the inference. Those words describe the
    # service and nothing else; the INTERACTIVE ones double as descriptions of
    # the items ("chat transcripts"), which is why they do not get this.
    (Workload.REALTIME, "voice scoring 5000 calls under 200ms"),
    (Workload.REALTIME, "real-time scoring for 5000 accounts under 200ms"),
    (Workload.REALTIME, "realtime classification API for 5000 accounts under 200ms"),
    # ...and the table still checks BATCH first, so this stays batch.
    (Workload.BATCH, "score 4 million support tickets overnight, real-time dashboards after"),
    # ...and without "overnight" to fall back on. The realtime phrase is a
    # different clause about a different thing; it governs a verb only when it
    # is right in front of one.
    (Workload.BATCH, "score 4 million support tickets, realtime dashboard after"),
    (Workload.REALTIME, "voice bot scoring 5000 calls under 200ms"),
    # Three words of reach, counted in words. Twelve characters stopped inside
    # this one and made a voice product a batch job.
    (Workload.REALTIME, "real-time support bot scoring 5000 calls under 200ms"),
    # "interactive" and "copilot" govern too — they name the service and
    # nothing else. "chat"/"agent"/"customer" still do not, because those
    # double as descriptions of the items.
    (Workload.INTERACTIVE, "interactive scoring for 5000 accounts"),
    (Workload.INTERACTIVE, "score 5000 accounts interactively"),  # trailing adverb
    # Served position decides, at any scale: "for 5000 accounts" is an
    # audience, while "process 20000 accounts overnight" is a backlog.
    (Workload.INTERACTIVE, "classification API for 5000 accounts under 200ms"),
    (Workload.BATCH, "process 20000 accounts overnight"),
    # A backlog that also names an audience, with an infinitive in between:
    # the preposition introduces the count it governs, it does not reach over
    # a verb to find one.
    (Workload.BATCH, "4 million tickets to score for 5000 users"),
    # And a realtime word that describes a different noun. "real-time" is
    # about the dashboard; the LLM work is still a batch job.
    (Workload.BATCH, "real-time dashboard for scoring 4 million tickets"),
    # A count the sentence itself calls in-flight is not a pile — and the
    # concurrency parser further down was already reading it correctly.
    (Workload.INTERACTIVE, "process 1000 concurrent requests under 200ms"),
    # A trailing mode ADVERB governs; a trailing mode NOUN does not, because
    # it is a preposition's object or an appended label and modifies nothing.
    (Workload.INTERACTIVE, "score 5000 accounts interactively"),
    (Workload.BATCH, "score 4 million support tickets with copilot"),
    (Workload.BATCH, "score 4 million support tickets realtime"),
    # Ordinary backlog wording the verb and noun lists had missed.
    (Workload.BATCH, "review 2 million invoices"),
    (Workload.BATCH, "analyze 2 million invoices"),
    (Workload.BATCH, "millions of invoices to review"),
    (Workload.INTERACTIVE, "handle 2 million messages per second"),
    # Items named by an interactive-sounding word are still items.
    (Workload.BATCH, "score 4 million chat transcripts"),
    (Workload.BATCH, "classify 2 million support bot conversations"),
    # A singular people-noun modifies the item; it does not replace it.
    (Workload.BATCH, "score 4 million customer tickets"),
    (Workload.BATCH, "process 20000 user records"),
    (Workload.BATCH, "index 50000 customer documents"),
    # An audience behind a preposition belongs to a later phrase.
    (Workload.BATCH, "classify 2 million documents for clients"),
    (Workload.BATCH, "score 4 million tickets from our users"),
    # A volume of PEOPLE is an audience, through any number of adjectives.
    (Workload.INTERACTIVE, "scoring app for 5000 users"),
    (Workload.INTERACTIVE, "scoring app for 5000 active users"),
    # Three modifiers, matching what _SERVED_AUDIENCE already allowed for
    # million-scale. The two windows were different numbers for one idea.
    (Workload.INTERACTIVE, "scoring app for 5000 monthly active paying users"),
    # A grouped number must be consumed whole. Matching only "1,000" of
    # "1,000,000" let the audience lookahead miss the noun entirely.
    (Workload.INTERACTIVE, "rank content for 1,000,000 users under 200ms"),
    (Workload.INTERACTIVE, "classify messages for 1,000,000 customers under 200ms"),
    (Workload.BATCH, "classify 2,000,000 documents"),
    (Workload.REALTIME, "real-time scoring for 5000 monthly active paying users under 200ms"),
    (Workload.INTERACTIVE, "customer scoring app for 5000 users"),
    (Workload.INTERACTIVE, "interactive scoring app for 5000 users"),
    (Workload.INTERACTIVE, "chat tool for ranking 5000 users"),
    (Workload.INTERACTIVE, "classification API for 5000 enterprise customers"),
    (Workload.INTERACTIVE, "chat assistant for 2 million daily active users"),
    (Workload.REALTIME, "real-time scoring for 5000 users"),
    (Workload.REALTIME, "real-time scoring for 5000 monthly users"),
    # A volume of REQUESTS is a rate only when it has a denominator. A corpus
    # of captured requests is the thing this product exists to process, so the
    # bare noun must stay a backlog.
    (Workload.REALTIME, "realtime classification API for 10000 requests per second under 200ms"),
    (Workload.REALTIME, "voice bot at 200 requests per second"),
    # A denominator spelled with a slash is still a denominator.
    (Workload.INTERACTIVE, "score 4000 requests/sec under 200ms"),
    (Workload.INTERACTIVE, "classify 5000 events/sec"),
    (Workload.INTERACTIVE, "rank 8000 req/s"),
    # "per <not a time>" is a ratio, not a rate — still a backlog.
    (Workload.BATCH, "classify 10000 requests per customer from captured traffic"),
    (Workload.BATCH, "score 20000 events per user"),
    # An audience is an audience at any magnitude when the sentence says it is
    # being SERVED. "millions of people are records" was right about scoring
    # them and wrong about serving them.
    (Workload.INTERACTIVE, "scoring app for 2 million users"),
    (Workload.INTERACTIVE, "rank content for 2 million users under 200ms"),
    (Workload.INTERACTIVE, "classify messages for 2 million customers under 200ms"),
    # ...through a quantifier, a possessive, or three adjectives. A fixed-width
    # lookbehind could reach none of these.
    (Workload.INTERACTIVE, "rank content for about 2 million users, needs to feel snappy"),
    (Workload.INTERACTIVE, "classify messages across our 2 million customers under 200ms"),
    (Workload.INTERACTIVE, "rank content for 2 million daily active paying users under 200ms"),
    # A sentence can state a backlog AND who it is for. Only the second volume
    # is the audience, so the suppression is scoped to the span rather than the
    # sentence.
    (Workload.BATCH, "classify 1 million requests from captured traffic for 2 million customers"),
    (Workload.BATCH, "score 4 million support tickets for 2 million users"),
    (Workload.BATCH, "classify 1 million requests from captured traffic"),
    (Workload.BATCH, "embed 1 million queries from logs"),
    (Workload.BATCH, "score 4 million captured requests"),
    # A corpus of captured requests with no verb at all — the phrasing this
    # tool exists for. requests/queries are backlog nouns as well as rate
    # units, and the branch that reads them keeps its denominator guard.
    (Workload.BATCH, "1 million captured requests"),
    (Workload.BATCH, "2 million support requests"),
    (Workload.BATCH, "1 million queries from logs"),
    (Workload.INTERACTIVE, "2 million requests per second"),
    # Silent-e verbs again: "triage" is not a prefix of "triaging", nor
    # "verify" of "verifying".
    # "widgets" is in no noun list, so only the verb can classify these — the
    # first spelling I wrote used backlog nouns and passed with the verbs
    # removed, which proved nothing.
    (Workload.BATCH, "triaging 2 million widgets"),
    (Workload.BATCH, "verifying 2 million widgets"),
    (Workload.BATCH, "triage 2 million widgets"),
    # People CAN be the work items, at a magnitude where they are not an
    # audience for anything interactive. Four million concurrent users is not
    # a product; four million customer records is a scoring job.
    (Workload.BATCH, "rank 4 million customers by priority"),
    (Workload.BATCH, "score 4 million users for churn risk"),
    (Workload.BATCH, "classify 4 million patients by triage category"),
    # Words that merely contain a signal.
    (Workload.INTERACTIVE, "label 5 millionaire profiles"),  # not "million"
    (Workload.INTERACTIVE, "gradual rollout to 2 million users"),  # not "grade"
    (Workload.INTERACTIVE, "a scoreboard for 5000 players"),  # not "score"
    (Workload.INTERACTIVE, "phonebook search assistant"),  # not "phone"
    # Volume then verb with no "to" describes; it does not instruct.
    (Workload.INTERACTIVE, "2 million users, ranked by activity"),
    # The signals still classify on their own.
    (Workload.INTERACTIVE, "customer support chat"),
    (Workload.REALTIME, "phone agent, has to reply under 800ms"),
    (Workload.REALTIME, "voice agent on a phone call"),
    (Workload.BATCH, "batching jobs"),
    (Workload.BATCH, "run it over our datasets"),
]


@pytest.mark.parametrize(("want", "text"), WORKLOADS)
def test_the_workload_table(want, text):
    assert read(text).requirements.workload is want


def test_a_latency_sensitive_product_is_still_asked_for_its_budget():
    # Batch skips the TTFT question outright, so a misclassification here is
    # not just a label — it drops the question that would have caught it.
    i = read("chat assistant for 2 million daily active users, needs to feel snappy")
    assert "ttft_ms" in {q.field for q in i.questions}


def test_the_batch_evidence_is_the_user_s_own_span():
    i = read("score 4 million support tickets")
    hit = next(x for x in i.inferred if x.field == "workload")
    assert hit.evidence in i.text


# --- what the number was counting ---------------------------------------------

CONCURRENCY = [
    # A headcount, divided by five because people think between requests.
    ("coding assistant for about 20 engineers", 4, True),
    ("chat for 500 people", 100, True),
    ("assistant for 40 devs", 8, True),
    ("a tool for 1 user", 1, True),
    # A singular headcount with anything after it. Every case I wrote put the
    # noun at the end of the string, so `(?!\s+\w)` looked correct and
    # rejected all of these.
    ("coding assistant for 1 developer who wants to work offline", 1, True),
    ("translation service for 1 user with low volume", 1, True),
    ("support bot for 1 agent to reply to emails", 1, True),
    # NOT a headcount: a singular people-noun modifying an item noun. Adding
    # "process 20000 user records" to the workload table without checking this
    # is how it shipped inferring four thousand concurrent people.
    ("process 20000 user records", 64, False),
    ("index 50000 customer documents", 64, False),
    ("score 4 million user tickets", 64, False),
    # "analyst" kept its optional plural in the UNGUARDED branch, so it skipped
    # the modifier check the other five nouns got.
    ("process 20000 analyst records", 64, False),
    # A plural stem needs a trailing boundary too: "userspace" is not users.
    ("20 userspace processes", 4, False),
    ("20 devservers", 4, False),
    ("tool for 300 staffordshire branches", 4, False),
    ("200 analysts", 40, True),
    ("process 1000 concurrent requests under 200ms", 1000, True),
    # A rate IS the concurrency, however it is spelled — and reading it as a
    # backlog cost both the workload and this number.
    ("rank 8000 req/s, each takes 250ms", 2000, True),
    ("score 4000 requests per second", 4, False),
    # Rates have decimals. `(\d[\d,]*)` could not match "1.5", so the engine
    # scanned past it and matched the "5".
    ("voice bot at 1.5 qps", 4, False),
    ("0.5 rps", 4, False),
    ("4,000 requests per second, each takes 1 second", 4000, True),
    # A rate is NOT a concurrency statement on its own. Concurrency here is
    # requests in flight, and that is arrivals x time in the system: "120 per
    # minute, each taking 30 seconds" is two per second and sixty in flight.
    # Without a stated service time the honest answer is the question.
    ("chat API at 120 requests per minute", 4, False),
    ("200 qps", 4, False),
    ("2 million events per minute", 4, False),
    ("3600 requests per hrs", 4, False),
    ("1..5 rps", 4, False),  # malformed: refuses to raise
    # A stated in-flight count beats a derived one.
    ("100 concurrent requests at 10 qps under 200ms", 100, True),
    # "staff" is a collective plural, so it modifies like a singular.
    ("process 20000 staff records", 64, False),
    ("chat tool for 5000 staff", 1000, True),
    # With a stated SERVICE TIME it is Little's Law, and the arithmetic is
    # quoted. Not with a latency budget: that is time to first token, a
    # request outlives its first token, and the resulting number was a lower
    # bound — which under-sizes KV and makes a deployment look feasible.
    ("chat API at 120 requests per minute, each can take 30 seconds", 60, True),
    ("10 qps, each request takes 2 seconds", 20, True),
    ("100 rps with 500ms per request", 50, True),
    # The service-time units match the rate units; two lists of the same
    # things had drifted, so "events" and "messages" derived nothing.
    ("100 events/sec, each event takes 1 second", 100, True),
    ("100 messages per second, each message takes 1 second", 100, True),
    # The subject is required. Optional, it matched "first token takes 2
    # seconds" — the exact figure this must not use as time-in-system.
    ("100 rps, first token takes 2 seconds", 4, False),
    ("100 rps, startup takes 2 seconds", 4, False),
    ("voice bot at 200 requests per second under 200ms", 4, False),
    # Malformed: the number is anchored, so "1..5" is refused rather than
    # read as the "5" the engine would find by scanning past it.
    ("1..5 rps, each takes 1 second", 4, False),
    # The denominator's spellings come from the same map as its value, so
    # "hrs" cannot exist in one and not the other — it did, and read as
    # per-second: a 3600x error.
    ("3600 requests per hrs, each takes 1 second", 1, True),
    ("3600 requests per second, each takes 1 second", 3600, True),
    ("classify 10000 requests per customer from captured traffic", 64, False),
]


@pytest.mark.parametrize(("text", "want", "inferred"), CONCURRENCY)
def test_a_headcount_is_counted_and_a_modifier_is_not(text, want, inferred):
    i = read(text)
    assert i.requirements.concurrency == want
    claimed = [x for x in i.inferred if x.field == "concurrency"]
    assert bool(claimed) is inferred, claimed
    # And when it is not inferred, it is asked rather than assumed silently.
    assert inferred or "concurrency" in {q.field for q in i.questions}


@pytest.mark.parametrize(
    "text",
    [
        "9" * 400 + " qps, each request takes 1 second",
        # Finite on its own, infinite once multiplied — so the check belongs
        # after the multiplier, not before it.
        "9" * 300 + " billion qps, each request takes 1 second",
        "9" * 300 + " million events per second, each takes 1 second",
        "10 qps, each request takes " + "9" * 400 + " seconds",
        "1..5 rps, each takes 1 second",
        "score " + "9" * 400 + " tickets",
    ],
)
def test_a_number_no_float_can_hold_returns_an_intent_rather_than_raising(text):
    # 400 digits parses to inf and math.ceil(inf) raises OverflowError. read()
    # must always return an Intent — its contract is that every sentence
    # produces a plan plus questions, never a traceback. Same defect shape as
    # the one in k8s/nodes: float() accepts it and int() explodes.
    i = read(text)
    assert i.requirements.concurrency >= 1
    assert "concurrency" in {q.field for q in i.questions}


@pytest.mark.parametrize(
    "text",
    [
        "10 qps, each request takes " + "9" * 307 + " seconds",  # ms overflows
        "chat, respond within " + "9" * 307 + " seconds",
        "9" * 300 + " users",  # int/5 raises before it can be rounded
        "9" * 300 + " concurrent",
        "9" * 400 + " qps, each request takes 1 second",
        "9" * 300 + " billion qps, each request takes 1 second",
        # Python 3.11 caps int() at 4300 digits, so int() is not the safe
        # conversion I had been treating it as — these raise ValueError before
        # any bound can look at the value.
        "9" * 5000 + " users",
        "9" * 5000 + " concurrent",
        "9" * 5000 + " requests per second",
        "chat for " + "9" * 5000 + " people",
        "9" * 5000 + "k tokens",  # _context, missed when the other two were fixed
        "9" * 5000 + "k context window",
    ],
)
def test_no_number_can_make_read_raise(text):
    # Guarding the inputs was not enough three times running: the overflow is
    # created by the arithmetic BETWEEN the check and the conversion. read()
    # must always return an Intent — a traceback is the one thing it may not
    # do — so an unusable number becomes the question instead.
    i = read(text)
    assert 1 <= i.requirements.concurrency <= 10**9
    assert i.requirements.ttft_ms is None or i.requirements.ttft_ms <= 10**9


def test_every_rounding_in_this_module_goes_through_the_one_guarded_helper():
    # The structural version of the three fixes above, and the same check
    # k8s/nodes.py carries for float(): a conversion added outside _whole is
    # the fourth overflow waiting to be found by an input rather than a review.
    import ast

    src = pathlib.Path(__file__).resolve().parents[1] / "src/clickllm/intent.py"
    tree = ast.parse(src.read_text())

    def rounds(node):
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Call):
                name = getattr(n.func, "id", "") or getattr(n.func, "attr", "")
                if name in {"round", "ceil"}:
                    out.add(n.lineno)
        return out

    inside = {
        line
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef) and f.name == "_whole"
        for line in rounds(f)
    }
    stray = sorted(rounds(tree) - inside)
    assert not stray, f"round()/ceil() outside _whole at intent.py:{stray}"


def test_no_captured_digit_string_is_converted_outside_the_one_helper():
    # The structural half of the 4300-digit fix. int() on a regex capture is
    # not safe — Python caps it — and I fixed two of the three sites, then a
    # reviewer found the third. A grep cannot tell int(m.group(1)) from
    # int(0.92 * n), so this parses.
    import ast

    src = pathlib.Path(__file__).resolve().parents[1] / "src/clickllm/intent.py"
    tree = ast.parse(src.read_text())

    def group_ints(node):
        out = set()
        for n in ast.walk(node):
            if not (isinstance(n, ast.Call) and getattr(n.func, "id", "") == "int"):
                continue
            # int(...) whose argument mentions a regex group is a captured
            # string; int(some_float) is not what this is about.
            if any(isinstance(c, ast.Attribute) and c.attr == "group" for c in ast.walk(n)):
                out.add(n.lineno)
        return out

    inside = {
        line
        for f in ast.walk(tree)
        if isinstance(f, ast.FunctionDef) and f.name in {"_digits", "_latency", "_service_time"}
        for line in group_ints(f)
    }
    stray = sorted(group_ints(tree) - inside)
    assert not stray, f"int() on a capture outside _digits at intent.py:{stray}"


@pytest.mark.parametrize(
    ("value", "want"),
    [
        (float("nan"), None),  # the bound cannot see this: NaN > x is False
        (float("inf"), None),
        (float("-inf"), None),
        (10**300, None),  # past the bound, so a question rather than a claim
        (_SANE_MAX + 1, None),
        (_SANE_MAX, _SANE_MAX),
        (16.5, 17),  # ceiling: banker's rounding sent this to 16
        (0.5, 1),
        (-3, 1),
        (0, 1),
    ],
)
def test_the_one_conversion_helper(value, want):
    assert _whole(value) == want


@pytest.mark.parametrize(
    ("raw", "want"),
    [
        ("9" * 5000, None),  # past Python 3.11's 4300-digit int() limit
        ("9" * 11, None),  # past _SANE_MAX, so out of range anyway
        ("4,000", 4000),
        ("900000000", 900000000),
        ("0", 0),
    ],
)
def test_the_digit_string_conversion(raw, want):
    assert _digits(raw) == want


@pytest.mark.parametrize(
    ("text", "want", "asked"),
    [
        ("128k context", 131_072, False),
        ("32k tokens", 32_768, False),
        ("long documents", 131_072, False),
        # _digits bounds the CAPTURE and the x1024 happens after it, so this
        # produced a context of 10**12 tokens as a confident requirement.
        ("999999999k tokens", 32_768, True),
        ("9" * 5000 + "k tokens", 32_768, True),
    ],
)
def test_a_context_length_is_bounded_after_its_multiplier_not_before(text, want, asked):
    i = read(text)
    assert i.requirements.context == want
    assert ("context" in {q.field for q in i.questions}) is asked
