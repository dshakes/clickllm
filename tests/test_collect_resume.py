"""A run that dies at 380 of 400 must not re-buy the 380.

Collection is the only part of this tool that spends real money, and it was the
part with no memory: a `Collection` lived entirely in one process, so a killed
run, a dropped SSH session or a laptop lid closing threw away every reply
already paid for.

The tests here are mostly about the ways a cache can be *worse* than none —
serving an answer to a question it does not answer, making a transient failure
permanent, or losing the whole ledger because its last line is torn.
"""

from __future__ import annotations

import json

import pytest

from clickllm.prove import collect as C
from clickllm.prove.collect import Collected, collect
from clickllm.prove.graders import EvalItem


def _items(n: int = 6) -> list[EvalItem]:
    return [
        EvalItem(
            item_id=f"i{i}", cluster="c", prompt=f"question {i}", baseline=f"b{i}", candidate=""
        )
        for i in range(n)
    ]


@pytest.fixture
def spy(monkeypatch):
    """Record which item_ids actually went over the wire."""
    asked: list[str] = []

    def fake_ask(item_id, prompt, **kw):
        asked.append(item_id)
        return Collected(item_id=item_id, text=f"answer to {prompt}", served_model="m")

    monkeypatch.setattr(C, "_ask", fake_ask)
    return asked


def _collect(items, resume=None, **kw):
    return collect(items, base="http://x", model="m", resume=resume, workers=2, **kw)


def test_a_resumed_run_only_pays_for_what_is_missing(tmp_path, spy):
    """The whole point. Half the eval set is already in hand; only the rest is
    asked for, and the result still covers everything."""
    ledger = tmp_path / "run.jsonl"
    items = _items(6)

    first = _collect(items[:3], resume=ledger)
    assert sorted(spy) == ["i0", "i1", "i2"]
    assert len(first.items) == 3

    spy.clear()
    second = _collect(items, resume=ledger)
    assert sorted(spy) == ["i3", "i4", "i5"], "it re-bought answers it already had"
    assert len(second.items) == 6, "the cached replies must still reach the result"
    assert [i.item_id for i in second.items] == [f"i{n}" for n in range(6)], (
        "cached replies must land in the order they were asked, not at the end"
    )


def test_without_a_ledger_nothing_is_written(tmp_path, spy):
    """Zero egress, zero surprise files (NFR-2). The cache is opt-in."""
    _collect(_items(3))
    assert list(tmp_path.iterdir()) == []


def test_a_cached_reply_is_bound_to_the_question_it_answered(tmp_path, spy):
    """The failure a cache must make impossible, because it is invisible in the
    result: serving one model's replies as another's.

    `item_id` alone is not a key. The same eval set against a different model,
    endpoint, side, or with an edited prompt is a different question, and a
    reused answer would silently score as if it had been asked.
    """
    ledger = tmp_path / "run.jsonl"
    items = _items(3)
    _collect(items, resume=ledger)
    assert len(spy) == 3

    for label, kw, changed in (
        ("model", {"model": "other"}, items),
        ("endpoint", {"base": "http://elsewhere"}, items),
        ("side", {"side": "baseline"}, items),
        (
            "prompt",
            {},
            [
                EvalItem(
                    item_id=i.item_id,
                    cluster="c",
                    prompt=i.prompt + "!",
                    baseline="b",
                    candidate="",
                )
                for i in items
            ],
        ),
    ):
        spy.clear()
        args = {"base": "http://x", "model": "m", "resume": ledger, "workers": 2}
        args.update(kw)
        collect(changed, **args)
        assert len(spy) == 3, f"a changed {label} reused a cached answer for a different question"


def test_a_failure_is_never_cached(tmp_path, monkeypatch):
    """An item that failed is an item to retry. Caching a 503 makes a transient
    outage permanent — a bad five minutes becomes a permanently smaller eval set.

    Two doors hold this, deliberately: the writer never appends a failure, and
    the reader never loads one. Breaking either alone leaves this test passing —
    which was checked, not assumed — so it is a test of the conjunction. The
    redundancy is the point on a money path, and this note is here because the
    natural reading of a green suite is that each guard is independently
    covered. It is not; do not delete either door on that basis.
    """
    ledger = tmp_path / "run.jsonl"
    calls = {"n": 0}

    def flaky(item_id, prompt, **kw):
        calls["n"] += 1
        if item_id == "i1" and calls["n"] <= 3:
            return Collected(item_id=item_id, reason="HTTP 503 — overloaded")
        return Collected(item_id=item_id, text="ok", served_model="m")

    monkeypatch.setattr(C, "_ask", flaky)
    first = _collect(_items(3), resume=ledger)
    assert len(first.failures) == 1

    got = _collect(_items(3), resume=ledger)
    assert not got.failures, "the retry never happened — the failure was cached"
    assert len(got.items) == 3


def test_a_torn_final_line_costs_one_item_not_the_run(tmp_path, spy):
    """The run this file exists for is one that was *killed*, so a
    half-written last line is the expected state, not an exceptional one.
    Refusing the whole ledger would throw away 380 answers to protect the 381st.
    """
    ledger = tmp_path / "run.jsonl"
    _collect(_items(4), resume=ledger)
    text = ledger.read_text()
    ledger.write_text(text[: -len(text.splitlines()[-1]) // 2])  # tear the tail

    spy.clear()
    got = _collect(_items(4), resume=ledger)
    assert 1 <= len(spy) <= 2, f"expected to re-ask only the torn item, re-asked {spy}"
    assert len(got.items) == 4


def test_a_foreign_or_corrupt_ledger_degrades_to_collecting_again(tmp_path, spy):
    """A cache is an optimisation. A bad one must cost money, never the run."""
    ledger = tmp_path / "run.jsonl"
    ledger.write_text('not json\n{"key": "x"}\n[]\n{"key": "y", "bogus_field": 1}\n\n')
    got = _collect(_items(3), resume=ledger)
    assert len(spy) == 3
    assert len(got.items) == 3


def test_a_fully_cached_run_asks_for_nothing_and_still_returns_everything(tmp_path, spy):
    """The empty-work path. `min(workers, len(todo))` is zero here, which
    `ThreadPoolExecutor` rejects — so this is a crash, not an optimisation,
    if the branch is missing."""
    ledger = tmp_path / "run.jsonl"
    _collect(_items(3), resume=ledger)
    spy.clear()
    got = _collect(_items(3), resume=ledger)
    assert spy == [], "it re-asked a fully cached run"
    assert len(got.items) == 3
    assert got.asked == 3


def test_progress_counts_cached_work_as_done(tmp_path, spy):
    """Otherwise a resumed run reports 1/400 while holding 380, and the ETA
    inherits the lie."""
    ledger = tmp_path / "run.jsonl"
    _collect(_items(6)[:4], resume=ledger)

    seen = []
    _collect(_items(6), resume=ledger, on_progress=seen.append)
    assert seen, "no progress was reported"
    assert seen[0].done == 4, f"first report should already count the 4 in hand, got {seen[0].done}"
    assert seen[-1].done == 6
    assert all(p.total == 6 for p in seen)


def test_the_ledger_is_written_as_replies_land(tmp_path, spy):
    """Flushed per reply, not at the end — a ledger written on completion is
    empty in exactly the case it exists for."""
    ledger = tmp_path / "run.jsonl"
    written = []

    def watch(_p):
        written.append(len(ledger.read_text().splitlines()) if ledger.exists() else 0)

    _collect(_items(4), resume=ledger, on_progress=watch)
    assert written and written[-1] == 4, written
    assert max(written) > 0, "nothing was on disk until the run finished"


def test_every_ledger_line_is_one_json_object_with_its_key(tmp_path, spy):
    """The format, pinned. It is read back by a later process, which makes it
    an interface even though it is not published."""
    ledger = tmp_path / "run.jsonl"
    _collect(_items(3), resume=ledger)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert len(row["key"]) == 64, "the key must be the full content hash"
        assert row["item_id"] and row["text"]
        assert row["reason"] == "", "only successful replies belong in the ledger"
