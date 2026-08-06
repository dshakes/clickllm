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
    # The anchor is on the START of the needle only, so the plural and the
    # participle must survive it — otherwise the fix trades one wrong answer
    # for another.
    assert read("parsing the output into json").requirements.structured_output
    assert read("batching overnight jobs").requirements.workload is Workload.BATCH
    assert read("a fleet of agents").requirements.prefix_sharing == 0.7


def test_a_large_user_count_does_not_make_an_interactive_product_a_batch_job():
    # BATCH is checked before INTERACTIVE by design, so a bare "million"
    # anywhere in the sentence beat "chat" and "assistant". The workload
    # decides scheduling in opposite directions.
    i = read("chat assistant for 2 million daily active users, needs to feel snappy")
    assert i.requirements.workload is Workload.INTERACTIVE
    # And a batch job still reads as one, on the words that actually mean it.
    assert read("score 4 million support tickets overnight").requirements.workload is Workload.BATCH


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
