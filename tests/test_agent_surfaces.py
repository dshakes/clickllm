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
