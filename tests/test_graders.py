"""The deterministic graders, on inputs that are valid JSON and look like failure.

`graders.demo()` walks the ordinary cases. This covers the one where the JSON
spec and Python's `None` collide: `null` is valid JSON, and a parser that
returns `None` to mean "not JSON" cannot tell the two apart.
"""

from __future__ import annotations

import pytest

from clickllm.prove.graders import _NOT_JSON, EvalItem, Outcome, _parse_json, grade


def score(baseline: str, candidate: str, grader: str) -> tuple[Outcome, str]:
    scores = grade(EvalItem("i", "c", "p", baseline, candidate)).scores
    hit = next(s for s in scores if s.grader == grader)
    return hit.outcome, hit.detail


def test_the_json_literal_null_parses_rather_than_reading_as_invalid():
    assert _parse_json("null") is None, "null is valid JSON and parses to None"
    assert _parse_json("{not json") is _NOT_JSON
    assert _parse_json("") is _NOT_JSON


def test_a_candidate_that_correctly_answered_null_is_not_graded_unparseable():
    # The defect: both "invalid JSON" and "the valid literal null" came back as
    # None, so a candidate reproducing the baseline exactly was scored FAIL on
    # json-valid — a wrong verdict against the candidate, on correct output.
    outcome, detail = score("null", "null", "json-valid")
    assert outcome is Outcome.PASS, detail


def test_a_null_baseline_still_holds_the_candidate_to_valid_json():
    # The mirror: the baseline check must not be excused as "not JSON" on a
    # baseline that is JSON, or the grader stops applying at all.
    outcome, _ = score("null", "not json at all", "json-valid")
    assert outcome is Outcome.FAIL


@pytest.mark.parametrize("literal", ["null", "true", "false", "0", '""', "[]", "{}"])
def test_every_bare_json_literal_is_treated_as_json(literal):
    # `null` is the one that broke, but the same conflation would hit any value
    # a truthiness test mishandles. These are all valid JSON documents.
    assert _parse_json(literal) is not _NOT_JSON
    outcome, detail = score(literal, literal, "json-valid")
    assert outcome is Outcome.PASS, detail


def test_genuinely_invalid_json_still_fails():
    outcome, _ = score('{"a": 1}', "{not json", "json-valid")
    assert outcome is Outcome.FAIL


def test_a_non_json_baseline_is_still_not_applicable():
    outcome, detail = score("just prose", "also prose", "json-valid")
    assert outcome is Outcome.NOT_APPLICABLE
    assert "not JSON" in detail
