"""The three doors an agent comes through: MCP, the SDK, and the session funnel.

ADR-0011's lesson was that a constraint enforced at one surface is a constraint
the next surface has to re-earn. These cover the cases where a surface was wider
than the thing behind it — a value it accepted and could not use, a filter that
failed open, and a threshold that made the verdict trivially true.

The fourth case from the same review — `clickllm_prove` reading any path an
agent names — is deliberately not here. Confining it is a new trust boundary,
and `tests/test_suite.py` drives that tool with a path outside the working
directory on purpose, to show the agent and the CLI reach the same verdict. That
is a contract to change with an ADR, not inside a batch of defect fixes.
"""

from __future__ import annotations

import pytest

from clickllm import sdk
from clickllm.mcp import handle
from clickllm.plan import Workload
from clickllm.prove.equivalence import CandidateReport, ClusterScore, Matrix
from clickllm.prove.stats import wilson
from clickllm.session import Session


def call(tool: str, **arguments):
    r = handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }
    )
    return r.get("result", {}), r.get("error")


# --- the field the error message advertised and the code could not take ---------


@pytest.mark.parametrize("value", ["batch", "interactive", "realtime"])
def test_a_workload_string_is_coerced_rather_than_kept_as_a_string(value):
    # `_apply_fields` rejects an unknown key with "Known: ... workload ...", so
    # the error names the field — and sending the documented value produced
    # `AttributeError: 'str' object has no attribute 'value'` further down. The
    # message invited the call that then broke.
    s = Session()
    s._apply_fields(workload=value)
    assert isinstance(s.requirements.workload, Workload)
    assert s.requirements.workload.value == value


def test_an_already_typed_workload_passes_through_untouched():
    s = Session()
    s._apply_fields(workload=Workload.REALTIME)
    assert s.requirements.workload is Workload.REALTIME


def test_a_workload_that_is_not_one_refuses_by_name():
    with pytest.raises(ValueError, match="workload must be Workload"):
        Session()._apply_fields(workload="nonsense")


def test_the_mcp_build_tool_takes_the_workload_it_documents():
    res, _ = call("clickllm_build", description="a nightly job", workload="batch")
    assert not res.get("isError"), res.get("content")


# --- a licence filter that failed open -------------------------------------------


def test_the_permissive_filter_actually_filters():
    permissive = sdk.models("permissive")
    assert permissive
    assert all(m.license_ok for m in permissive)
    assert len(permissive) < len(sdk.models("all")), "fixture must have both kinds"


@pytest.mark.parametrize("bad", ["PERMISSIVE", "Permissive", "permisive", "open", ""])
def test_a_filter_that_is_not_one_refuses_rather_than_returning_everything(bad):
    # A `Literal` is a type-checker's promise, not a runtime one. Anything that
    # was not exactly "permissive" fell through to the unfiltered list, so a
    # caller asking for cleanly-commercial models got every restrictively
    # licensed one back — failing open, in the direction that matters.
    with pytest.raises(ValueError, match="license_filter"):
        sdk.models(bad)


def test_the_default_is_unchanged():
    assert len(sdk.models()) == len(sdk.models("all"))


# --- a bar that made the verdict trivially true ----------------------------------


def _regressed_matrix(bar: float) -> Matrix:
    # 41/80 — a candidate nobody should migrate to.
    return Matrix(
        [CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))], bar=bar
    )


@pytest.mark.parametrize("bar", [0.0, -1.0, -100.0, 1.0, 2.0])
def test_a_bar_outside_the_open_unit_interval_is_refused(bar):
    # `clearly_above(bar)` is `interval.low > bar`, so a bar at or below zero is
    # true for essentially any interval: a 51% cluster reads `equivalent` and
    # `movable_share` says move everything. That report is what a human uses to
    # authorise a cutover (invariant 8), and the MCP schema's `minimum`/`maximum`
    # are advisory only.
    with pytest.raises(ValueError, match="equivalence bar"):
        _regressed_matrix(bar)


def test_a_real_bar_still_calls_a_regression_a_regression():
    m = _regressed_matrix(0.90)
    assert m.candidates[0].clusters[0].band(m.bar) == "regressed"


def test_a_lower_but_legitimate_bar_is_still_allowed():
    # The guard must refuse the degenerate values without deciding policy: 0.5
    # is a choice somebody can defend, and this has no business blocking it.
    assert _regressed_matrix(0.5).bar == 0.5


def test_the_receipt_cannot_be_issued_with_a_degenerate_bar():
    # `issue()` takes `bar` directly and never builds a Matrix, so guarding
    # only the Matrix left the *receipt* — the portable proof artifact, the one
    # thing designed to be handed to someone who does not trust us — issuable
    # at bar=0.0, rendering "Proven at or above the 0% bar" over a 41/80
    # regression with movable_share 1.0.
    from clickllm.prove import issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    for bar in (0.0, -1.0, 1.0, 2.0):
        with pytest.raises(ValueError, match="equivalence bar"):
            issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=bar)


def test_a_real_bar_still_issues_a_receipt_that_names_the_regression():
    from clickllm.prove import issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    r = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    assert [c.cluster for c in r.regret] == ["x"]
    assert r.movable_share == 0


def test_the_mcp_schema_advertises_the_bounds_the_code_enforces():
    # A schema-valid call that the runtime refuses is worse than one the client
    # could have rejected itself. The schema said minimum/maximum 0 and 1 while
    # the guard rejects both endpoints.
    from clickllm.mcp import TOOLS

    _, schema = TOOLS["clickllm_prove"]
    bar = schema["inputSchema"]["properties"]["bar"]
    assert bar.get("exclusiveMinimum") == 0
    assert bar.get("exclusiveMaximum") == 1
    assert "minimum" not in bar and "maximum" not in bar


def test_a_receipt_read_from_disk_with_a_degenerate_bar_is_refused():
    # `issue()` covers the receipts this tool writes. `from_json` is what it
    # *reads* — the ingest path behind `clickllm receipt`, `clickllm guard` and
    # the box — and a file with `bar: 0.0` and a perfectly valid digest parsed
    # and rendered "Proven at or above the 0% bar" with movable_share 1.0.
    #
    # The digest is no help: it is computed over that content, so such a
    # receipt is internally consistent. Tamper detection answers "was this
    # altered", not "was this ever true".
    import json as _json

    from clickllm.prove import Receipt, issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    blob = _json.loads(good.to_json())
    blob["receipt"]["bar"] = 0.0
    with pytest.raises(ValueError, match="equivalence bar"):
        Receipt.from_json(_json.dumps(blob))


def test_an_honest_receipt_still_round_trips():
    from clickllm.prove import Receipt, issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    assert Receipt.from_json(good.to_json()) == good


def test_the_build_schema_advertises_the_workload_it_now_accepts():
    # It was accepted by the code and absent from the schema, so a schema-driven
    # agent could only discover it from the error message that lists the known
    # fields — which is exactly how the broken call was being made.
    from clickllm.mcp import TOOLS

    _, schema = TOOLS["clickllm_build"]
    workload = schema["inputSchema"]["properties"].get("workload")
    assert workload, "clickllm_build must advertise workload"
    assert set(workload["enum"]) == {"interactive", "realtime", "batch"}


# --- the receipt envelope, read off disk from a stranger -------------------------


def _good_receipt():
    from clickllm.prove import issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(41, 80), 0),))
    return issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)


@pytest.mark.parametrize("digest", [True, 7, 3.5, ["abc"], {"a": 1}])
def test_a_digest_that_is_not_a_string_is_a_sentence_not_a_traceback(digest):
    """The digest check runs on the *failure* path — the one that executes when
    a receipt does not verify — and it sliced `stated[:12]` to name it.

    `true`, `7` and `3.5` are not subscriptable and `{"a": 1}` raises `KeyError`
    on the slice, so the code that exists to report a forged receipt crashed
    instead. It still failed closed; it failed closed as a traceback, and
    `cli.main()` catches `ValueError`.
    """
    import json as _json

    from clickllm.prove import Receipt

    blob = _json.loads(_good_receipt().to_json())
    blob["digest"] = digest
    with pytest.raises(ValueError, match="digest"):
        Receipt.from_json(_json.dumps(blob))


@pytest.mark.parametrize("body", [True, 7, "text", [], None])
def test_a_receipt_envelope_that_is_not_an_object_is_refused_the_same_way(body):
    """Not in the finding that prompted this, and worse: `blob.get` needs the
    document to be a mapping and `body.get` needs the inner one to be, so
    `{"receipt": 7}` and a bare `[]` document both raised `AttributeError` —
    before the digest was ever reached."""
    import json as _json

    from clickllm.prove import Receipt

    blob = _json.loads(_good_receipt().to_json())
    blob["receipt"] = body
    with pytest.raises(ValueError, match="JSON object"):
        Receipt.from_json(_json.dumps(blob))


@pytest.mark.parametrize("doc", ["null", "[]", '"text"', "7", "true"])
def test_a_document_that_is_not_an_object_at_all_is_refused(doc):
    from clickllm.prove import Receipt

    with pytest.raises(ValueError, match="JSON object"):
        Receipt.from_json(doc)


def test_an_honest_receipt_is_unaffected_and_tampering_is_still_named():
    """The negative control for the three sweeps above: a guard that refused
    every document would pass all of them and break the only path that matters.
    """
    import json as _json

    from clickllm.prove import Receipt

    good = _good_receipt()
    assert Receipt.from_json(good.to_json()) == good

    blob = _json.loads(good.to_json())
    blob["receipt"]["incumbent"] = "someone else"
    with pytest.raises(ValueError, match="has been altered"):
        Receipt.from_json(_json.dumps(blob))


@pytest.mark.parametrize(
    "proven",
    [[7], ["a"], [{"bogus": 1}], 7, "text", [[]], [None], {"a": 1}],
)
def test_a_claim_group_that_is_not_claims_is_refused_as_a_value_error(proven):
    """Shape has been guarded one level at a time across four reviews — the
    document, the envelope, the digest, and now the claim groups inside it.
    Each fix was right and the next reviewer found the next level.

    So this is not a fifth enumeration: `from_json` now converts any `TypeError`
    or `KeyError` from `**`-construction into `ValueError`. The contract is that
    the file is a stranger's, and every way it fails to be a receipt is the one
    exception `cli.main()` catches.
    """
    import json as _json

    from clickllm.prove import Receipt

    blob = _json.loads(_good_receipt().to_json())
    blob["receipt"]["proven"] = proven
    with pytest.raises(ValueError):
        Receipt.from_json(_json.dumps(blob))


@pytest.mark.parametrize("key", ["movable_share", "extra_key", "bar_typo"])
def test_a_key_the_receipt_does_not_have_is_refused_the_same_way(key):
    """`cls(**body)` raises `TypeError` for an unexpected keyword, which is the
    same family and was equally uncaught."""
    import json as _json

    from clickllm.prove import Receipt

    blob = _json.loads(_good_receipt().to_json())
    blob["receipt"][key] = "x"
    with pytest.raises(ValueError):
        Receipt.from_json(_json.dumps(blob))


def test_a_fields_own_validator_keeps_its_better_sentence():
    """The blanket conversion must not swallow the specific messages.

    `match="equivalence bar"` alone does not test this: adding `ValueError` to
    the caught tuple wraps the message rather than replacing it, so the original
    text is still a substring and the assertion holds either way. That was the
    first version, and its control passed. The claim is that the sentence is not
    buried under a generic prefix.
    """
    import json as _json

    from clickllm.prove import Receipt

    blob = _json.loads(_good_receipt().to_json())
    blob["receipt"]["bar"] = 0.0
    with pytest.raises(ValueError) as caught:
        Receipt.from_json(_json.dumps(blob))
    msg = str(caught.value)
    assert msg.startswith("equivalence bar"), msg
    assert "not readable as a receipt" not in msg


def test_no_field_of_a_receipt_survives_being_the_wrong_type(tmp_path):
    """A forger edits the content and recomputes the digest over it, so the file
    is internally consistent and tamper detection has nothing to say.

    Thirty combinations of field and wrong type parsed and then crashed
    `render()` or `movable_share` with `TypeError`, inside `cmd_receipt`, which
    catches `ValueError`. Seven reviews found seven levels of this document
    unguarded, each fix a hand-written list of the shapes known at the time —
    so the check is derived from the annotations the classes already declare,
    and a field added later is covered by having been declared.

    This sweeps every field of `Claim` in every group against five wrong types
    and asserts that nothing reaches a caller as a non-`ValueError`.
    """
    import json as _json
    from dataclasses import fields

    from clickllm.prove import Receipt, issue
    from clickllm.prove.receipt import Claim

    report = CandidateReport(
        "m",
        (
            ClusterScore("good", "good", 0.5, wilson(80, 80), 0),
            ClusterScore("bad", "bad", 0.5, wilson(41, 80), 0),
        ),
    )
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)

    def forged(group: str, field: str, value):
        blob = _json.loads(good.to_json())
        if not blob["receipt"][group]:
            return None
        blob["receipt"][group][0][field] = value
        body = dict(blob["receipt"])
        for k in ("proven", "regret", "unproven"):
            body[k] = tuple(Claim(**c) for c in body.get(k, ()))
        r = Receipt(**body)  # recompute the digest over the tampered content
        return _json.dumps({"receipt": blob["receipt"], "digest": r.digest()})

    escaped, checked = [], 0
    for group in ("proven", "regret", "unproven"):
        for f in fields(Claim):
            for bad in ("1.0", None, [1], True, {"a": 1}):
                try:
                    doc = forged(group, f.name, bad)
                except ValueError:
                    checked += 1  # refused at construction, which is the point
                    continue
                if doc is None:
                    continue
                checked += 1
                try:
                    r = Receipt.from_json(doc)
                except ValueError:
                    continue
                for label, call in (
                    ("render", r.render),
                    ("movable_share", lambda r=r: r.movable_share),
                    ("to_json", r.to_json),
                ):
                    try:
                        call()
                    except ValueError:
                        pass
                    except Exception as e:  # noqa: BLE001 — the type is the point
                        escaped.append(f"{group}.{f.name}={bad!r} -> {label}: {type(e).__name__}")

    assert checked > 50, "the sweep did not actually exercise anything"
    assert not escaped, "these reach a caller as a non-ValueError:\n  " + "\n  ".join(escaped)


def test_an_honest_receipt_is_unchanged_by_the_type_sweep():
    """The negative control, and it is load-bearing: `float` must admit `int`,
    because JSON writes `1` for `1.0` and a receipt this tool issued has to
    survive its own round trip."""
    from clickllm.prove import Receipt, issue

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(80, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    assert Receipt.from_json(good.to_json()) == good
    assert good.render()
    assert good.movable_share == 1.0


def test_a_whole_number_share_written_as_an_integer_still_parses():
    """The specific round-trip hazard the control above generalises: a hand-
    written or re-serialised receipt carrying `"share": 1` rather than `1.0`."""
    import json as _json

    from clickllm.prove import Receipt, issue
    from clickllm.prove.receipt import Claim

    report = CandidateReport("m", (ClusterScore("x", "x", 1.0, wilson(80, 80), 0),))
    good = issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)
    blob = _json.loads(good.to_json())
    blob["receipt"]["proven"][0]["share"] = 1
    body = dict(blob["receipt"])
    for k in ("proven", "regret", "unproven"):
        body[k] = tuple(Claim(**c) for c in body.get(k, ()))
    doc = _json.dumps({"receipt": blob["receipt"], "digest": Receipt(**body).digest()})
    assert Receipt.from_json(doc).movable_share == 1


# --- the right shape, and an impossible measurement ------------------------------


def _refuses_forgery(good, mutate) -> str:
    """Tamper with a receipt, reseal it with a valid digest, and read it back.

    Returns the refusal message. Resealing itself goes through the same
    constructors, so a guard can bite either while the forgery is being built or
    while it is being parsed — both are the guard working, and which one fires
    is an accident of how a forger happens to assemble the file. Anything other
    than `ValueError` fails here, because that is the whole contract:
    `cli.main()` catches `ValueError` and nothing else.
    """
    import json as _json

    from clickllm.prove import Receipt
    from clickllm.prove.receipt import Claim

    blob = _json.loads(good.to_json())
    mutate(blob["receipt"])
    try:
        body = dict(blob["receipt"])
        for k in ("proven", "regret", "unproven"):
            body[k] = tuple(Claim(**c) for c in body.get(k, ()))
        doc = _json.dumps({"receipt": blob["receipt"], "digest": Receipt(**body).digest()})
        Receipt.from_json(doc)
    except ValueError as e:
        return str(e)
    raise AssertionError("the forged receipt was accepted")


def _two_cluster_receipt():
    from clickllm.prove import issue

    report = CandidateReport(
        "m",
        (
            ClusterScore("good", "good", 0.5, wilson(80, 80), 0),
            ClusterScore("bad", "bad", 0.5, wilson(41, 80), 0),
        ),
    )
    return issue(report, incumbent="i", issued="2026-08-07", eval_set="a" * 64, bar=0.90)


@pytest.mark.parametrize(
    ("label", "field", "value"),
    [
        ("a share larger than all traffic", "share", 10.0),
        ("a negative share", "share", -1.0),
        ("a negative pass count", "passed", -5),
        ("a negative denominator", "total", -1),
        ("a negative exclusion count", "ungraded", -3),
        ("a negative merge count", "duplicates", -2),
        ("an interval bound outside 0..1", "low", -2.0),
        ("a negative sample requirement", "needed", -1),
    ],
)
def test_a_claim_that_is_the_right_shape_and_an_impossible_measurement(label, field, value):
    """The type sweep answers "is this an `int`", which is a different question
    from "is this a count".

    A forged receipt recomputes its own digest, so every value arrives
    self-consistent and, until now, unexamined. `share: 10.0` rendered
    `Movable: 1000% of captured traffic` off a file that verified.
    """
    _refuses_forgery(_two_cluster_receipt(), lambda b: b["proven"][0].__setitem__(field, value))


def test_a_pass_count_cannot_exceed_its_denominator():
    msg = _refuses_forgery(
        _two_cluster_receipt(), lambda b: b["proven"][0].update({"passed": 500, "total": 80})
    )
    assert "cannot exceed total" in msg, msg


def test_an_inverted_interval_is_refused():
    msg = _refuses_forgery(
        _two_cluster_receipt(), lambda b: b["proven"][0].update({"low": 0.9, "high": 0.1})
    )
    assert "inverted" in msg, msg


def test_claims_cannot_together_account_for_more_traffic_than_exists():
    """Per-claim validation is not enough, and missing that was the same mistake
    made on `CandidateReport` earlier in this branch: the value that reaches the
    arithmetic is the *sum*. Two individually legal 0.9 shares claimed 180% of
    the traffic and reported `movable_share` 90% off it."""
    msg = _refuses_forgery(
        _two_cluster_receipt(),
        lambda b: (
            b["proven"][0].__setitem__("share", 0.9),
            b["regret"][0].__setitem__("share", 0.9),
        ),
    )
    assert "exceed all of the traffic" in msg, msg


@pytest.mark.parametrize(
    ("key", "value"),
    [("redacted", 123), ("redacted", None), ("fingerprints", "x"), ("fingerprints", 7)],
)
def test_a_container_field_that_is_not_a_container_is_refused(key, value):
    """`_permitted` returned `None` for any subscripted hint, so `redacted:
    dict[str, int]` was unchecked entirely: `123` parsed and then raised
    `AttributeError: 'int' object has no attribute 'items'` inside `render()`."""
    _refuses_forgery(_two_cluster_receipt(), lambda b: b.__setitem__(key, value))


def test_the_honest_receipt_survives_all_of_it():
    """The negative control for this whole block, including the one case these
    guards must not break: an exact traffic split summing to 1.0."""
    from clickllm.prove import Receipt, issue

    good = _two_cluster_receipt()
    assert Receipt.from_json(good.to_json()) == good
    assert good.render()
    assert good.movable_share == 0.5

    split = issue(
        CandidateReport(
            "m",
            tuple(ClusterScore(k, k, s, wilson(80, 80), 0) for k, s in (("a", 0.6), ("b", 0.4))),
        ),
        incumbent="i",
        issued="2026-08-07",
        eval_set="a" * 64,
        bar=0.90,
    )
    assert Receipt.from_json(split.to_json()).movable_share == 1.0
