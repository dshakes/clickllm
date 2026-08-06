"""What the sentence actually said.

`intent.demo()` walks sentences that classify correctly. These pin the ones
that classified confidently and wrongly — a signal matched inside an unrelated
word, then quoted back as the user's own words. `Inference.evidence` promises
"the literal words that produced this", so a match the user never wrote is not
a near miss; it is the field lying about its own contents.
"""

from __future__ import annotations

import pytest

from clickllm.intent import read
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
    # ...unless the noun carries a denominator. A rate whose unit happens to
    # be a work item is still a rate.
    (Workload.REALTIME, "realtime classify 2 million events/sec under 200ms"),
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
    # A rate IS the concurrency, however it is spelled — and reading it as a
    # backlog cost both the workload and this number.
    ("score 4000 requests/sec under 200ms", 4000, True),
    ("rank 8000 req/s", 8000, True),
    ("score 4000 requests per second", 4000, True),
    # Rates have decimals. `(\d[\d,]*)` could not match "1.5", so the engine
    # scanned past it and matched the "5".
    ("voice bot at 1.5 qps", 2, True),
    ("0.5 rps", 1, True),
    ("4,000 requests per second", 4000, True),
    # A per-MINUTE rate is not a concurrency statement: 200 per minute is
    # about three in flight. Reading it as 200 sized a GPU cluster for a
    # workload a laptop serves — and the previous version of this row asserted
    # exactly that, which is the "test enshrines the bug" shape again.
    ("voice bot at 200 requests per minute", 4, False),
    ("10000 requests per day", 4, False),
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
