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


def test_a_large_user_count_does_not_make_an_interactive_product_a_batch_job():
    # BATCH is checked before INTERACTIVE by design, so a bare "million"
    # anywhere in the sentence beat "chat" and "assistant". The workload
    # decides scheduling in opposite directions.
    i = read("chat assistant for 2 million daily active users, needs to feel snappy")
    assert i.requirements.workload is Workload.INTERACTIVE
    # It is latency-sensitive, so the TTFT question must still be asked — a
    # batch classification skips that question entirely.
    assert "ttft_ms" in {q.field for q in i.questions}


@pytest.mark.parametrize(
    "text",
    [
        "score 4 million support tickets",  # no "overnight" to fall back on
        "classify 2 million documents",
        "rank 500000 candidate answers",
        # The gerunds, which an open stem misses on every silent-e verb:
        # "translate" is not a prefix of "translating".
        "translating 5 million tickets",
        "scoring 4 million tickets",
        "grading 2 million essays",
        "transcribing 900000 calls",
        "annotating 2 million images",
        "rewriting 40000 product blurbs",
        "summarising 30000 reports",
        # Written the way a person writes them. "4,000" contains no run of
        # four digits, so a bare \d{4,} missed every grouped number.
        "score 4,000 tickets",
        "classify 2,000,000 documents",
        "rank 1,500 answers",
        # Noun first, verb after — the other order English uses for the same
        # instruction. The "to" is what makes it one.
        "4 million support tickets to score",
        "2 million documents to classify",
        "30000 rows to embed",
    ],
)
def test_a_bulk_verb_over_a_large_volume_is_still_batch_without_a_batch_word(text):
    # Dropping bare "million" fixed the interactive case and broke this one:
    # these are batch work by any reading and carry no "overnight"/"offline".
    # The verb is what separates the two — nothing scores four million tickets
    # while someone waits.
    i = read(text)
    assert i.requirements.workload is Workload.BATCH
    # And the evidence is the user's own span, not a bare magnitude.
    assert next(x for x in i.inferred if x.field == "workload").evidence in text


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


@pytest.mark.parametrize(
    "text",
    [
        "gradual rollout to 2 million users",  # not "grade"
        "gradient work on 900000 samples",  # not "grade"
        "a scoreboard for 5000 players",  # not "score"
        "label 5 millionaire profiles",  # not "million"
        # Volume then verb with no "to" is a description, not an instruction.
        "2 million users, ranked by activity",
    ],
)
def test_the_bulk_verbs_are_not_open_stems(text):
    # The reason those verbs are spelled stem-plus-endings rather than
    # `grad\\w*`: the open stem catches the gerund and half the dictionary
    # with it, which trades a missed batch job for a fabricated one.
    assert read(text).requirements.workload is not Workload.BATCH


@pytest.mark.parametrize(
    ("text", "want"),
    [
        # The volume decides, because the volume is what differs. Each of these
        # has a bulk verb in it; only the second group counts work items.
        ("scoring app for 5000 users", Workload.INTERACTIVE),
        ("customer scoring app for 5000 users", Workload.INTERACTIVE),
        ("interactive scoring app for 5000 users", Workload.INTERACTIVE),
        ("chat tool for ranking 5000 users", Workload.INTERACTIVE),
        ("real-time scoring for 5000 users", Workload.REALTIME),
        # Work items, even when an interactive-sounding word names them. A
        # rule that let "chat" outrank the volume made this one interactive.
        ("score 4 million chat transcripts", Workload.BATCH),
        ("classify 2 million support bot conversations", Workload.BATCH),
        ("score 4 million customer tickets", Workload.BATCH),
        # And the mode words still classify on their own.
        ("customer support chat", Workload.INTERACTIVE),
    ],
)
def test_a_volume_of_users_is_an_audience_and_a_volume_of_items_is_a_backlog(text, want):
    assert read(text).requirements.workload is want
