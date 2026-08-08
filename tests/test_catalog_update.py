"""M?? — the catalogue proposal path.

`catalog_update.demo()` covers the happy diff. These are the two ways the
proposal could mislead or crash: a change the report calls cosmetic while the
solver computes with it, and a response that is valid JSON but not a config.
"""

from __future__ import annotations

import pytest

from clickllm import catalog_update as cu
from clickllm.catalog import load

_BASE = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 131072,
}


def _spec():
    return load()[0]


def _fetcher(body: str):
    def fetch(url: str) -> str:
        return body

    return fetch


# --- the field in one list and not the other -------------------------------------


def test_every_field_the_proposal_compares_is_marked_significant():
    """`propose` compared six fields; `significant` named five.

    The missing one was `max_context`, and `fit.max_context()` caps every
    context figure the solver prints at `min(model.max_context, ...)`. A vendor
    advertising 128k against a config publishing 32k is an ordinary
    discrepancy — and it rendered without the `!` marker, stayed out of
    `UpdateReport.significant`, and so never triggered the "re-run fit before
    deploying" line in `cmd_catalog`.

    Both now read `SOLVER_FIELDS`. This asserts they cannot drift apart again:
    a seventh field added to the comparison is significant by construction, and
    one that genuinely is not forces `significant` to stop being a membership
    test rather than being quietly dropped.
    """
    assert cu.SOLVER_FIELDS, "the comparison would be empty"
    for name in cu.SOLVER_FIELDS:
        assert cu.FieldChange(name, 1, 2).significant, f"{name} is compared but not significant"


def test_nothing_both_sides_declare_is_left_out_of_the_comparison():
    """The stronger half, because the sweep above cannot catch a *removal*.

    Now that `significant` and `propose` read one list, dropping a field from it
    drops it from both and the sweep still passes. What cannot be quietly
    dropped is this: every field that `Architecture` (the parsed truth) and
    `ModelSpec` (the catalogue's claim) both declare is a field where the two
    can disagree, and one that is not compared is one where the catalogue drifts
    from the published config with nothing ever reporting it.

    A new geometry field added to both — MoE expert counts are the obvious next
    one, and CLAUDE.md's first sizing rule is that MoE needs *total* params
    resident — fails here until it is compared.
    """
    from dataclasses import fields

    from clickllm.catalog import ModelSpec

    shared = {f.name for f in fields(cu.Architecture)} & {f.name for f in fields(ModelSpec)}
    assert shared - set(cu.SOLVER_FIELDS) == set(), (
        "these are parsed from the config and stored in the catalogue, but never "
        f"compared: {sorted(shared - set(cu.SOLVER_FIELDS))}"
    )


def test_max_context_specifically_carries_the_marker():
    change = cu.FieldChange("max_context", 131072, 32768)
    assert change.significant
    assert change.render().lstrip().startswith("!")


def test_a_field_outside_the_comparison_is_not_significant():
    """The negative control: `significant` returning True unconditionally would
    satisfy the sweep above and make the `!` marker meaningless."""
    for name in ("license", "name", "params_b", "quants"):
        assert not cu.FieldChange(name, 1, 2).significant


def test_the_comparison_still_finds_a_real_geometry_change():
    """`propose` was rewritten to loop over `SOLVER_FIELDS`; this is the control
    that it still reads the same values off the parsed architecture."""
    spec = _spec()
    cfg = {
        "num_hidden_layers": spec.layers + 7,
        "num_attention_heads": 32,
        "num_key_value_heads": spec.kv_heads,
        "head_dim": spec.head_dim,
        "max_position_embeddings": spec.max_context,
    }
    import json

    p = cu.propose(spec, spec.repo or "o/r", _fetcher(json.dumps(cfg)))
    assert p.error is None, p.error
    assert [c.field for c in p.changes] == ["layers"]
    assert p.changes[0].proposed == spec.layers + 7
    assert p.significant


# --- valid JSON that is not a config ---------------------------------------------


@pytest.mark.parametrize("body", ["null", "[]", "3", '"a string"', "true"])
def test_a_json_body_that_is_not_an_object_is_reported_not_raised(body):
    """`json.loads` succeeds on all of these, so the fetch guard never sees
    them, and every field read starts with `cfg.get(...)`.

    The first one raised `AttributeError`, which is not a `ConfigError` —
    `propose` catches only that — so it escaped the list comprehension in
    `cmd_catalog`, took every model after it down with it, and reached the user
    as a traceback. `cli.main()`'s handler does not list `AttributeError`
    either. The convention is a sentence and exit 2.
    """
    spec = _spec()
    p = cu.propose(spec, spec.repo or "o/r", _fetcher(body))
    assert p.error is not None
    assert "not a JSON object" in p.error


@pytest.mark.parametrize("body", ["null", "[]", "3"])
def test_the_guard_lives_where_all_three_callers_reach_it(body):
    """`cli.py` and `watch.py` call `parse_config` on a freshly-parsed body too.
    Guarding inside `propose` would have left both raising `AttributeError`
    through their `except ConfigError` and into their broad fallback, which
    reports "could not read config: 'NoneType' object has no attribute 'get'"
    instead of saying what is wrong."""
    import json

    with pytest.raises(cu.ConfigError, match="not a JSON object"):
        cu.parse_config(json.loads(body))


def test_an_object_that_is_simply_incomplete_still_says_which_field():
    # The negative control: a guard that rejected every dict would pass the
    # tests above and break the real path.
    with pytest.raises(cu.ConfigError, match="no layer count"):
        cu.parse_config({"a": 1})


def test_a_real_config_still_parses():
    arch = cu.parse_config(
        {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "max_position_embeddings": 131072,
        }
    )
    assert (arch.layers, arch.kv_heads, arch.kv_scheme) == (32, 8, "gqa")


def test_the_module_self_check_still_passes():
    cu.demo()


# --- the same assumption, five more times --------------------------------------

# The non-dict guard above covers `cfg` itself. These are the *fields* inside a
# well-formed object, and inside rows of the discovery index — every one of them
# coerced without checking what it was. Found by sweeping the file for reads of
# untrusted JSON after a reviewer named two of them; the sweep found six.


@pytest.mark.parametrize("value", [7, "Llama", {"a": 1}, 3.5, True])
def test_architectures_that_is_not_a_list_does_not_abort_the_parse(value):
    """`or []` is not a type check — `7 or []` is `7`, and iterating it raises
    `TypeError`, which `propose` does not catch.

    The same field was read twice in this function, and only the second read
    tested `isinstance(..., list)`.
    """
    # `== "" or isinstance(str)` was the first version of this line and is
    # tautologically true for a `str`-typed field — it would have passed against
    # the unfixed code had the unfixed code not raised. The claim is that the
    # value is *ignored*, which is the empty string here because `_BASE` carries
    # no `model_type` to fall back to.
    assert cu.parse_config({**_BASE, "architectures": value}).architecture == ""


def test_the_model_type_fallback_still_applies_when_architectures_is_unusable():
    # ...and the control that "" above means "ignored", not "cleared".
    got = cu.parse_config({**_BASE, "architectures": 7, "model_type": "qwen3"})
    assert got.architecture == "qwen3"


def test_a_real_architectures_list_is_still_read():
    assert cu.parse_config({**_BASE, "architectures": ["Qwen3ForCausalLM"]}).architecture == (
        "Qwen3ForCausalLM"
    )


def test_the_mla_family_is_still_detected_through_the_architecture_name():
    """The negative control for the guard above: dropping the read entirely
    would satisfy every test in this block and disable the MLA refusal, which
    is the one that stops a ~50x KV overestimate."""
    with pytest.raises(cu.ConfigError, match="MLA"):
        cu.parse_config({**_BASE, "architectures": ["DeepseekV3ForCausalLM"]})


def _index(row: dict):
    def fetch(url: str) -> str:
        import json

        return json.dumps([row])

    return fetch


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("downloads", "many"),
        ("downloads", [1]),
        ("downloads", {"a": 1}),
        ("likes", {"a": 1}),
        ("likes", "lots"),
        ("trendingScore", "hot"),
        ("trendingScore", []),
        ("cardData", "mit"),
        ("cardData", ["mit"]),
        ("cardData", 7),
    ],
)
def test_one_odd_field_in_the_index_does_not_abort_discovery(field, value):
    """`int("many")` raises `ValueError`, `int([1])` `TypeError`, and
    `(cardData or {}).get(...)` `AttributeError` for a string. None is
    `Unreachable`, so a single malformed row took the whole discovery run down
    with a traceback — the failure this module's own comment says it exists to
    notice, one level below where it was checked.
    """
    got = cu.discover(set(), _index({"modelId": "o/r", field: value}))
    assert [d.repo for d in got] == ["o/r"], "the row must survive its own bad field"


def test_a_sane_row_still_carries_every_field():
    """The negative control: coercing everything to the default would satisfy
    the sweep above and silently zero the ranking the discovery order uses."""
    got = cu.discover(
        set(),
        _index(
            {
                "modelId": "o/r",
                "downloads": 5,
                "likes": 2,
                "trendingScore": 1.5,
                "cardData": {"license": "mit"},
            }
        ),
    )
    d = got[0]
    assert (d.downloads, d.likes, d.trending, d.license) == (5, 2, 1.5, "mit")


@pytest.mark.parametrize("field", ["downloads", "likes", "trendingScore"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_in_the_index_does_not_abort_discovery(field, value):
    """The first version of `_number` checked only the *type*, and
    `isinstance(float("nan"), float)` is `True`.

    `json.loads` accepts the literals `NaN`, `Infinity` and `-Infinity`, so both
    walked through: `int(nan)` raises `ValueError`, `int(inf)` `OverflowError`,
    and neither is `Unreachable`. Swept for the wrong thing — the values, not
    just the shapes.
    """
    got = cu.discover(set(), _index({"modelId": "o/r", field: value}))
    assert [d.repo for d in got] == ["o/r"]


def test_a_nan_score_cannot_poison_the_ordering():
    """The half that does not crash, and is worse for it. `trending` reaches
    `out.sort(key=lambda d: (-d.trending, ...))`, where NaN compares false
    against everything and silently makes the order the comment two lines below
    calls "Deterministic" depend on the order the rows arrived in."""
    import json
    import math

    rows = [
        {"modelId": "o/high", "trendingScore": 9.0},
        {"modelId": "o/nan", "trendingScore": float("nan")},
        {"modelId": "o/mid", "trendingScore": 5.0},
    ]

    def fetch(url: str) -> str:
        return json.dumps(rows, allow_nan=True)

    got = cu.discover(set(), fetch)
    assert all(math.isfinite(d.trending) for d in got)
    assert [d.repo for d in got] == ["o/high", "o/mid", "o/nan"]


def test_a_boolean_is_not_a_count():
    """`True` is an `int` in Python, so `int(r.get("downloads") or 0)` turned a
    JSON `true` into one download. A fabricated number is worse than a missing
    one — this module's whole point is that estimates are labelled."""
    got = cu.discover(set(), _index({"modelId": "o/r", "downloads": True, "likes": True}))
    assert (got[0].downloads, got[0].likes) == (0, 0)
