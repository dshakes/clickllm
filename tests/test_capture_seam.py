"""The seam between the recorder and the reader.

One shape is defined three times — a Rust struct in `capture/store.rs`, a dict
built by the PyO3 bridge, and a dataclass in `distill/shape.py` — in two
languages, with nothing forcing them to agree. They did not agree:

* the recorder called the measurement `duration_ms`, the reader `latency_ms`;
* the recorder never wrote `tools`, `tool_calls` or `response_format`, so every
  tool-using workload clustered as toolless and nothing said so.

The second is the dangerous one. A name mismatch raises on the first row. A
field that is simply never recorded produces a clustering that is well-formed,
plausible, and blind along one of the six dimensions it claims to use — which
is the shape of every defect this repo keeps finding.

These tests read the Rust source rather than running the bridge, because the
bridge needs `maturin develop` and this must fail in CI on a bare checkout. The
bridge itself is exercised by the eight tests that skip without it.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path

import pytest

from clickllm.distill.shape import Capture, extract_shape, from_capture_row

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "clickllm-py" / "src" / "lib.rs"
STORE = ROOT / "clickllm-gateway" / "src" / "capture" / "store.rs"
PROXY = ROOT / "clickllm-gateway" / "src" / "proxy.rs"

#: Keys the bridge emits for provenance that `Capture` has no field for. Listed
#: rather than tolerated silently: a new one appearing should be a decision.
PROVENANCE = {"backend", "redacted"}


def _bridge_keys() -> set[str]:
    """Every key `capture_to_dict` sets."""
    body = re.search(r"fn capture_to_dict.*?\n}\n", BRIDGE.read_text(), re.S)
    assert body, "capture_to_dict not found; did the bridge move?"
    return set(re.findall(r'd\.set_item\("([^"]+)"', body.group(0)))


def test_every_field_the_reader_needs_is_a_key_the_bridge_emits():
    """The direction that fails loudly, asserted anyway: `Capture(**row)` would
    raise on a missing one, but only once someone ran the bridge."""
    missing = {f.name for f in fields(Capture)} - _bridge_keys()
    assert not missing, (
        f"the bridge never emits {sorted(missing)}, so every capture reads as "
        "the field's default and the distiller clusters on a value it never saw"
    )


def test_the_bridge_emits_nothing_the_reader_silently_drops():
    """The other direction, which fails quietly: a key with no field is either
    provenance (fine, and listed) or a measurement being thrown away."""
    extra = _bridge_keys() - {f.name for f in fields(Capture)} - PROVENANCE
    assert not extra, (
        f"the bridge emits {sorted(extra)} and nothing reads it — add a field, "
        "or add it to PROVENANCE and say why"
    )


def test_the_recorder_records_every_field_the_bridge_forwards():
    """The end that was actually broken. The bridge can only forward what the
    struct holds, and the struct held six of the ten."""
    struct = re.search(r"pub struct Capture \{(.*?)\n\}", STORE.read_text(), re.S)
    assert struct, "Capture struct not found"
    recorded = set(re.findall(r"^\s*pub (\w+):", struct.group(1), re.M))
    # `duration_ms` is `latency_ms` on the other side; the bridge renames it and
    # that rename is asserted below.
    needed = {f.name for f in fields(Capture)} - {"latency_ms"} | {"duration_ms"}
    assert not needed - recorded, f"the gateway never records {sorted(needed - recorded)}"


def test_the_one_renamed_field_is_renamed_in_exactly_one_place():
    """`duration_ms` → `latency_ms` happens at the bridge. If the struct ever
    grows a `latency_ms` too, one of them is dead and the other is a coin
    flip."""
    assert 'd.set_item("latency_ms", c.duration_ms)' in BRIDGE.read_text()
    struct = re.search(r"pub struct Capture \{(.*?)\n\}", STORE.read_text(), re.S)
    assert struct and "latency_ms" not in struct.group(1)


def test_the_proxy_fills_the_fields_and_does_not_leave_them_default():
    """A field on the struct that nothing assigns is the same blindness with an
    extra step — the struct would satisfy the test above and still record
    nothing. Both completion paths must set the response-side one."""
    src = PROXY.read_text()
    for expr in (
        'parsed\n            .get("tools")',
        'parsed\n            .get("response_format")',
    ):
        assert expr in src, f"the proxy never reads {expr!r} off the request"
    assert "store::body_tool_calls(&bytes)" in src, "non-streamed responses record no tool calls"
    assert "store::delta_tool_call(ev)" in src, "streamed responses record no tool calls"


# --- the converter itself --------------------------------------------------------


def _row(**over: object) -> dict:
    row = {
        "request_id": "r-1",
        "model": "gpt-5",
        "backend": "incumbent",
        "messages": [{"role": "user", "content": "refund order 12"}],
        "response": "done",
        "prompt_tokens": 11,
        "completion_tokens": 4,
        "latency_ms": 91,
        "tools": [{"function": {"name": "refund"}}],
        "tool_calls": ["refund"],
        "response_format": "json_object",
        "redacted": {"email": 1},
    }
    row.update(over)
    return row


def test_a_bridge_row_converts_without_the_provenance_keys_raising():
    c = from_capture_row(_row())
    assert c.request_id == "r-1"
    assert c.latency_ms == 91
    assert c.tools and c.tool_calls
    assert c.response_format == "json_object"


def test_a_tool_using_workload_no_longer_clusters_as_toolless():
    """The defect, stated as behaviour rather than as a field list."""
    called = extract_shape(from_capture_row(_row()))
    offered_only = extract_shape(from_capture_row(_row(tool_calls=[])))
    toolless = extract_shape(from_capture_row(_row(tools=[], tool_calls=[])))

    assert called.used_tools and called.tool_names == ("refund",)
    assert not offered_only.used_tools and offered_only.tool_names == ("refund",)
    assert not toolless.used_tools and toolless.tool_names == ()
    assert len({called.signature, offered_only.signature, toolless.signature}) == 3, (
        "three different workloads must not share one cluster key"
    )


def test_the_response_format_separates_workloads_that_fail_differently():
    a = extract_shape(from_capture_row(_row(response_format="json_object")))
    b = extract_shape(from_capture_row(_row(response_format=None)))
    assert a.signature != b.signature


@pytest.mark.parametrize("bad", [None, "tools", 7, {"a": 1}])
def test_a_malformed_tools_block_is_a_shape_not_an_exception(bad):
    """Captured traffic is data, not a contract (invariant 7). A client that
    sent something strange still produced a workload with a shape."""
    assert from_capture_row(_row(tools=bad)).tools == ()


def test_a_row_that_is_not_a_capture_says_so():
    with pytest.raises(ValueError, match="missing"):
        from_capture_row({"model": "gpt-5"})
    with pytest.raises(TypeError, match="must be a dict"):
        from_capture_row(["not", "a", "row"])
