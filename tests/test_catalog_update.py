"""M?? — the catalogue proposal path.

`catalog_update.demo()` covers the happy diff. These are the two ways the
proposal could mislead or crash: a change the report calls cosmetic while the
solver computes with it, and a response that is valid JSON but not a config.
"""

from __future__ import annotations

import pytest

from clickllm import catalog_update as cu
from clickllm.catalog import load


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
