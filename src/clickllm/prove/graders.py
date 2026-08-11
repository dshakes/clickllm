"""The grader stack — tiers 1 and 2: deterministic, cheap, and unarguable.

Three tiers, escalating only when the cheaper one cannot decide:

    1. assert   structure — valid JSON, same keys, same tool called, non-empty
    2. task     task semantics — exact match, numbers within tolerance, same files
    3. judge    pairwise LLM comparison, last resort (see :mod:`.judge`)

**A grader that does not apply returns NOT_APPLICABLE, never PASS.** That is the
single most important rule here. If "no assertion fired" silently counted as
success, every model would score near 100% on any cluster our graders don't
understand, and the resulting matrix would be confidently, uselessly green.

Tiers 1 and 2 are worth more than they look. Most real regressions are
structural — the candidate emits prose where JSON was required, calls the wrong
tool, or drops a file from a diff — and those are caught here, deterministically,
for free, without asking a language model to have an opinion.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class Tier(StrEnum):
    """Which stage of the stack a grader belongs to."""

    ASSERT = "assert"
    TASK = "task"
    JUDGE = "judge"


class Outcome(StrEnum):
    """What a grader concluded."""

    PASS = "pass"
    FAIL = "fail"
    #: The grader had nothing to say. Never counted as a pass.
    NOT_APPLICABLE = "n/a"


@dataclass(frozen=True, slots=True)
class Score:
    """One grader's verdict on one item."""

    grader: str
    tier: Tier
    outcome: Outcome
    detail: str = ""

    @property
    def applicable(self) -> bool:
        return self.outcome is not Outcome.NOT_APPLICABLE

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS


@dataclass(frozen=True, slots=True)
class EvalItem:
    """One prompt, the incumbent's answer, and the candidate's answer.

    ``baseline`` is what the model being replaced actually produced. It is not
    ground truth — it is the bar to match, which is a weaker and more honest claim.
    """

    item_id: str
    cluster: str
    prompt: str
    baseline: str
    candidate: str
    baseline_tool_calls: tuple[dict[str, Any], ...] = ()
    candidate_tool_calls: tuple[dict[str, Any], ...] = ()
    #: Either a bare name ("enum", "json_schema") or the OpenAI request shape
    #: `{"type": "json_object"}` — the CLI reads this straight out of an eval-set
    #: file, so whatever a user wrote arrives here unvalidated.
    response_format: str | dict | None = None


@runtime_checkable
class Grader(Protocol):
    """Anything that can judge one item."""

    name: str
    tier: Tier

    def grade(self, item: EvalItem) -> Score: ...


#: Returned by `_parse_json` for text that is not JSON at all. A sentinel rather
#: than `None`, because `null` is perfectly valid JSON that parses *to* `None` —
#: conflating the two graded a candidate that correctly answered `null` as
#: unparseable, and excused the baseline check as "not JSON" on a baseline that
#: was.
_NOT_JSON = object()


def _parse_json(text: str) -> Any:
    """Parsed JSON, or `_NOT_JSON`. Never `None` to mean failure."""
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return _NOT_JSON


# --------------------------------------------------------------------------- #
# Tier 1 — structure
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class NonEmpty:
    """The candidate must answer when the incumbent did."""

    name: str = "non-empty"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline.strip():
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline was empty")
        if item.candidate.strip():
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(self.name, self.tier, Outcome.FAIL, "candidate returned nothing")


@dataclass(frozen=True, slots=True)
class JsonValid:
    """If the incumbent produced JSON, the candidate must produce valid JSON."""

    name: str = "json-valid"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if _parse_json(item.baseline) is _NOT_JSON:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not JSON")
        if _parse_json(item.candidate) is _NOT_JSON:
            return Score(self.name, self.tier, Outcome.FAIL, "candidate is not parseable JSON")
        return Score(self.name, self.tier, Outcome.PASS)


@dataclass(frozen=True, slots=True)
class JsonShape:
    """Same top-level keys. Values may differ; the contract is the shape.

    Downstream code indexes by key. A response with the right values under the
    wrong keys breaks the caller just as hard as garbage.
    """

    name: str = "json-shape"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        b, c = _parse_json(item.baseline), _parse_json(item.candidate)
        if not isinstance(b, dict):
            return Score(
                self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a JSON object"
            )
        if not isinstance(c, dict):
            return Score(self.name, self.tier, Outcome.FAIL, "candidate is not a JSON object")
        missing, extra = set(b) - set(c), set(c) - set(b)
        if missing or extra:
            bits = []
            if missing:
                bits.append(f"missing {sorted(missing)}")
            if extra:
                bits.append(f"unexpected {sorted(extra)}")
            return Score(self.name, self.tier, Outcome.FAIL, "; ".join(bits))
        return Score(self.name, self.tier, Outcome.PASS)


def _call_names(calls: tuple[object, ...]) -> list[str]:
    """Tool names, from any of the three shapes a call arrives in.

    An eval set is a file. It comes from `clickllm distill`, from a hand edit,
    or from another tool entirely, and invariant 7 applies to it as much as to
    the corpus: it is data, not a contract. A bare `"refund"` is an obvious way
    to write a tool call by hand, and raising `AttributeError` from four frames
    inside a grader is the wrong answer to it — the whole run dies over one
    malformed row.
    """
    out = []
    for c in calls:
        if isinstance(c, str):
            n: object = c
        elif isinstance(c, dict):
            fn = c.get("function")
            n = c.get("name") or (fn.get("name") if isinstance(fn, dict) else None)
        else:
            continue
        if n:
            out.append(str(n))
    return out


@dataclass(frozen=True, slots=True)
class ToolChoice:
    """The candidate must call the same tools, in the same order.

    Order matters: a plan that reads before writing is not the same plan as one
    that writes before reading, even with identical tool names.
    """

    name: str = "tool-choice"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline called no tools")
        want, got = _call_names(item.baseline_tool_calls), _call_names(item.candidate_tool_calls)
        if want == got:
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(self.name, self.tier, Outcome.FAIL, f"expected {want}, got {got}")


@dataclass(frozen=True, slots=True)
class ToolArgs:
    """Required argument *keys* must match on every shared tool call.

    Values are not compared — a different file path can be a correct answer to a
    different prompt. A missing argument cannot.
    """

    name: str = "tool-args"
    tier: Tier = Tier.ASSERT

    def grade(self, item: EvalItem) -> Score:
        if not item.baseline_tool_calls or not item.candidate_tool_calls:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "no comparable tool calls")
        problems = []
        # Length mismatches are ToolChoice's job; here we only compare the pairs.
        for i, (b, c) in enumerate(
            zip(item.baseline_tool_calls, item.candidate_tool_calls, strict=False)
        ):
            bk, ck = set(_args(b)), set(_args(c))
            if bk - ck:
                problems.append(f"call {i} missing {sorted(bk - ck)}")
        if problems:
            return Score(self.name, self.tier, Outcome.FAIL, "; ".join(problems))
        return Score(self.name, self.tier, Outcome.PASS)


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = call.get("arguments") or (call.get("function") or {}).get("arguments") or {}
    if isinstance(raw, str):
        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else {}
    return raw if isinstance(raw, dict) else {}


# --------------------------------------------------------------------------- #
# Tier 2 — task semantics
# --------------------------------------------------------------------------- #

_CLASSIFICATION_MAX_WORDS = 3


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """For classification-shaped answers, the label must match.

    Applies only when the incumbent's answer is a short label — for prose, exact
    match is the wrong question and would fail every correct paraphrase.
    """

    name: str = "exact-match"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        b = item.baseline.strip()
        if not _looks_like_a_label(b, item.response_format):
            return Score(
                self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a short label"
            )
        if _label_key(b) == _label_key(item.candidate):
            return Score(self.name, self.tier, Outcome.PASS)
        return Score(
            self.name,
            self.tier,
            Outcome.FAIL,
            f"expected {b!r}, got {item.candidate.strip()[:60]!r}",
        )


def _label_key(text: str) -> str:
    """A label reduced to what actually distinguishes it from another label.

    Measured against a real model: asked "capital of France? Reply with only the
    city name", Llama 3.1 answers `Paris.` — correct, and scored FAIL against a
    baseline of `Paris` for the full stop. That ran across all ten extraction
    items and the suite reported REGRET, keep the incumbent, for a model that
    got every question right.

    Only wrapping punctuation is removed: surrounding quotes and backticks, and
    trailing sentence marks. Internal punctuation is untouched, because it
    carries meaning — `3.14` and `314` are different answers, and so are
    `no, refund` and `no refund`. If stripping would empty the string the raw
    text is kept, so a label that IS punctuation still compares as itself.
    """
    t = text.strip()
    stripped = t.strip("\"'`\u201c\u201d\u2018\u2019").rstrip(".!?").strip()
    return (stripped or t).casefold()


def _format_name(response_format: str | dict | None) -> str | None:
    """The format's name, from either spelling people actually write.

    `{"type": "json_object"}` is the OpenAI request shape and the one the
    `prove` help text points at, so it is what ends up in an eval-set file. It
    reached `x in {"enum", "json_schema"}` as a dict and raised TypeError —
    unhashable — which killed the entire suite AFTER every reply had been
    collected and paid for. A malformed field must cost one grader's opinion,
    never the run.
    """
    if isinstance(response_format, str):
        return response_format
    if isinstance(response_format, dict):
        kind = response_format.get("type")
        return kind if isinstance(kind, str) else None
    return None


def _looks_like_a_label(text: str, response_format: str | dict | None) -> bool:
    """Whether an answer is a classification label rather than a short sentence.

    Word count alone is not enough: "Revenue was 42.5m." is three words and a
    sentence, and exact-matching it would fail every correct paraphrase. Sentence
    punctuation is the discriminator. When the request declared an enum or
    json_schema format, that is authoritative and the heuristic is skipped.
    """
    # `json_object` is deliberately NOT here: it constrains the answer to JSON,
    # which is the opposite of a short label — exact-matching a serialised object
    # would fail on key order and whitespace. Only formats that pin the answer to
    # a fixed set are authoritative.
    if _format_name(response_format) in {"enum", "json_schema"}:
        return True
    if not text or "\n" in text:
        return False
    if len(text.split()) > _CLASSIFICATION_MAX_WORDS:
        return False
    # A label does not end a sentence and does not contain one.
    return not text.endswith((".", "!", "?", ":", ";")) and "," not in text


#: Numbers, including thousands-grouped ones. The grouped alternative comes
#: FIRST because Python's `|` is first-match-wins: with the plain form first,
#: `1,234,567` matched as `1` and the rest was rescanned.
#:
#: That is not cosmetic. `-?\d+(?:\.\d+)?` split `$1,234,567` into `1`, `234`,
#: `567`, and `grade` only checks each value appears *somewhere* in the
#: candidate — so "1 prior conviction … served 234 days … 567 dollars in fines"
#: scored PASS against a baseline awarding $1,234,567. A deterministic grader,
#: the tier trusted to advance traffic, passing a completely wrong money figure
#: — and grouping is exactly what money figures have.
#:
#: `\d{1,3}(?:,\d{3})+` requires full three-digit groups, so a genuine list like
#: "1, 234" (with the space) still reads as two numbers. "1,234" with no space
#: is read as one, which is the right call when the alternative is the above.
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _as_float(token: str) -> float:
    """A matched token as a number, with any thousands separators removed."""
    return float(token.replace(",", ""))


@dataclass(frozen=True, slots=True)
class NumericAgreement:
    """Numbers in the answer must agree within a relative tolerance.

    Catches the failure mode where a candidate writes fluent, plausible prose
    around a wrong figure — which an LLM judge frequently waves through.
    """

    tolerance: float = 0.01
    name: str = "numeric-agreement"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        want = [_as_float(x) for x in _NUM.findall(item.baseline)]
        if not want:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline has no numbers")
        got = {_as_float(x) for x in _NUM.findall(item.candidate)}
        missing = [w for w in want if not any(_close(w, g, self.tolerance) for g in got)]
        if missing:
            return Score(self.name, self.tier, Outcome.FAIL, f"missing or altered: {missing[:5]}")
        return Score(self.name, self.tier, Outcome.PASS)


def _close(a: float, b: float, tol: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= tol


_DIFF_FILE = re.compile(r"^\+\+\+ [ab]/(.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class DiffFiles:
    """A code diff must touch the same files.

    The long-context refactor failure in practice: the candidate produces a
    perfectly valid diff that silently omits one of the files that needed changing.
    """

    name: str = "diff-files"
    tier: Tier = Tier.TASK

    def grade(self, item: EvalItem) -> Score:
        want = set(_DIFF_FILE.findall(item.baseline))
        if not want:
            return Score(self.name, self.tier, Outcome.NOT_APPLICABLE, "baseline is not a diff")
        got = set(_DIFF_FILE.findall(item.candidate))
        if missing := want - got:
            return Score(
                self.name, self.tier, Outcome.FAIL, f"files not modified: {sorted(missing)}"
            )
        return Score(self.name, self.tier, Outcome.PASS)


#: The deterministic stack, cheapest first. Run in order; each is independent.
DEFAULT_GRADERS: tuple[Grader, ...] = (
    NonEmpty(),
    JsonValid(),
    JsonShape(),
    ToolChoice(),
    ToolArgs(),
    ExactMatch(),
    NumericAgreement(),
    DiffFiles(),
)


@dataclass(frozen=True, slots=True)
class ItemResult:
    """Every grader's verdict on one item."""

    item_id: str
    cluster: str
    scores: tuple[Score, ...]

    @property
    def applicable(self) -> tuple[Score, ...]:
        return tuple(s for s in self.scores if s.applicable)

    @property
    def graded(self) -> bool:
        """Whether any grader had an opinion at all."""
        return bool(self.applicable)

    @property
    def passed(self) -> bool:
        """Every applicable grader passed.

        Conjunctive on purpose: a response that is valid JSON with the wrong keys
        has not "mostly passed".
        """
        return self.graded and all(s.passed for s in self.applicable)

    @property
    def failures(self) -> tuple[Score, ...]:
        return tuple(s for s in self.applicable if not s.passed)


def grade(item: EvalItem, graders: tuple[Grader, ...] = DEFAULT_GRADERS) -> ItemResult:
    """Run the deterministic stack over one item."""
    return ItemResult(
        item_id=item.item_id,
        cluster=item.cluster,
        scores=tuple(g.grade(item) for g in graders),
    )


def demo() -> None:
    def item(base: str, cand: str, **kw) -> EvalItem:
        return EvalItem("i1", "c1", "prompt", base, cand, **kw)

    # An identical answer passes everything that applies.
    same = grade(item('{"a": 1, "b": 2}', '{"a": 1, "b": 2}'))
    assert same.passed and same.graded

    # Right values, wrong keys — callers index by key, so this is a failure.
    shape = grade(item('{"a": 1}', '{"b": 1}'))
    assert not shape.passed
    assert any(s.grader == "json-shape" for s in shape.failures)

    # Prose where JSON was required.
    broke = grade(item('{"a": 1}', "Sure! Here's the answer: a is 1."))
    assert not broke.passed
    assert any(s.grader == "json-valid" for s in broke.failures)

    # THE critical invariant: nothing applicable must not read as success.
    nothing = grade(item("", ""))
    assert not nothing.graded, "empty baseline should leave nothing to grade"
    assert not nothing.passed, "ungraded must never count as passed"

    # Fluent prose around a wrong number is caught deterministically.
    num = grade(item("Revenue was 42.5 million.", "Revenue came to about 51.9 million."))
    assert not num.passed
    assert any(s.grader == "numeric-agreement" for s in num.failures)
    assert grade(item("Revenue was 42.5m.", "It was 42.5m, roughly.")).passed

    # Tool choice and order.
    def calls(*ns):
        return tuple({"name": n, "arguments": {"path": "x"}} for n in ns)

    assert grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=calls("read", "write"),
            candidate_tool_calls=calls("read", "write"),
        )
    ).passed
    wrong_order = grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=calls("read", "write"),
            candidate_tool_calls=calls("write", "read"),
        )
    )
    assert not wrong_order.passed, "tool order is part of the plan"

    # A dropped argument fails; a different value does not.
    dropped = grade(
        item(
            "ok",
            "ok",
            baseline_tool_calls=({"name": "w", "arguments": {"path": "a", "mode": "x"}},),
            candidate_tool_calls=({"name": "w", "arguments": {"path": "b"}},),
        )
    )
    assert any(s.grader == "tool-args" for s in dropped.failures)

    # A diff that silently drops a file.
    d1 = "--- a/x.py\n+++ b/x.py\n@@\n-1\n+2\n--- a/y.py\n+++ b/y.py\n@@\n-3\n+4\n"
    d2 = "--- a/x.py\n+++ b/x.py\n@@\n-1\n+2\n"
    dropped_file = grade(item(d1, d2))
    assert not dropped_file.passed
    assert any("y.py" in s.detail for s in dropped_file.failures)

    # Classification uses exact match; prose does not.
    assert grade(item("spam", "spam")).passed
    assert not grade(item("spam", "ham")).passed
    prose = grade(item("The meeting is on Tuesday at three.", "It's Tuesday at 3."))
    assert all(s.grader != "exact-match" for s in prose.applicable), (
        "exact match must not fire on prose — it would fail every valid paraphrase"
    )

    print(f"graders: {len(DEFAULT_GRADERS)} deterministic, tiers 1-2")
    print("ok")


if __name__ == "__main__":
    demo()
