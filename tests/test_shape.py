"""Task-shape identity.

`shape.demo()` walks shapes that differ in the obvious places — prompt, output
format, context bucket. These pin the identity itself: two different shapes
must not share a signature, because a collision merges them into one cluster
and nothing downstream can tell that it happened.
"""

from __future__ import annotations

from clickllm.distill.shape import Capture, extract_shape


def cap(tools=(), tool_calls=({"name": "x"},), **kw):
    return Capture(
        request_id=kw.pop("request_id", "r"),
        model="m",
        messages=({"role": "system", "content": "s"}, {"role": "user", "content": "hi"}),
        response="ok",
        tools=tools,
        tool_calls=tool_calls,
        **kw,
    )


def test_a_comma_in_a_tool_name_does_not_merge_two_different_shapes():
    # One tool named "get,weather" and two named "get" and "weather" joined to
    # the same string. _sha separates its parts with NUL for exactly this
    # reason, and folding the names together bypassed that protection.
    one = extract_shape(cap(tools=({"name": "get,weather"},)))
    two = extract_shape(cap(tools=({"name": "get"}, {"name": "weather"})))
    assert one.signature != two.signature


def test_a_separator_inside_a_tool_name_does_not_merge_two_shapes_either():
    # The comma fix moved the collision one layer down rather than removing it:
    # a NUL separator is only unambiguous while no part contains a NUL, and
    # tool names come out of captured schemas, which this repo treats as
    # untrusted data. Hence length prefixes, which a payload cannot forge.
    for sep in ("\x00", ",", ":", "1:"):
        one = extract_shape(cap(tools=({"name": f"get{sep}weather"},)))
        two = extract_shape(cap(tools=({"name": "get"}, {"name": "weather"})))
        assert one.signature != two.signature, sep


def test_the_same_tools_in_any_order_are_the_same_shape():
    # The other half of the same property: separating the names must not make
    # the signature order-sensitive, since tool_names is sorted for that reason.
    a = extract_shape(cap(tools=({"name": "get"}, {"name": "weather"})))
    b = extract_shape(cap(tools=({"name": "weather"}, {"name": "get"})))
    assert a.signature == b.signature


def test_a_tool_declared_twice_is_one_tool():
    # A gateway that accumulates schemas across turns declares the same tool
    # twice. "tool-calling (2 tools)" for a bot with one overstates the tool
    # surface a human is judging the cluster on.
    s = extract_shape(cap(tools=({"name": "search"}, {"name": "search"})))
    assert s.tool_names == ("search",)
    assert "(1 tool)" in s.describe()
    # And it is the same shape as the capture that declared it once.
    assert s.signature == extract_shape(cap(tools=({"name": "search"},))).signature


def test_the_nested_and_flat_spellings_of_one_tool_are_one_tool():
    s = extract_shape(cap(tools=({"name": "search"}, {"function": {"name": "search"}})))
    assert s.tool_names == ("search",)


def test_two_tools_still_read_as_two():
    s = extract_shape(cap(tools=({"name": "search"}, {"name": "fetch"})))
    assert "(2 tools)" in s.describe()
