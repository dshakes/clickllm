"""The proactive watcher.

`watch.demo()` walks one pass. These pin the guardrails, which are the whole
design: a job that runs unattended and writes near the catalogue can quietly
corrupt the thing every sizing answer depends on.
"""

from __future__ import annotations

import pytest

import json

from clickllm import catalog, watch

INDEX = json.dumps(
    [
        {"id": "acme/new-9b", "downloads": 90_000, "likes": 400, "pipeline_tag": "text-generation"},
        {"id": "acme/broken", "downloads": 80_000, "likes": 300, "pipeline_tag": "text-generation"},
    ]
)
GOOD = json.dumps(
    {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "max_position_embeddings": 32768,
    }
)


def fetch(url: str) -> str:
    if "new-9b" in url:
        return GOOD
    if "broken" in url:
        return json.dumps({"hidden_size": 4096})  # no layer count
    return INDEX


# --- the guardrails ------------------------------------------------------------


def test_a_discovery_never_reaches_the_solver_until_a_human_promotes_it(tmp_path, monkeypatch):
    """The bug this exists to prevent: the first cut wrote into models.d/, and
    because a discovery has no parameter count, every later `fit` failed on the
    file the robot had just written."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    watch.run(fetch)

    assert watch.staging_dir().exists(), "something should have been staged"
    # The staging directory must not be on the catalogue's read path.
    assert watch.staging_dir() not in catalog.sources()
    for src in catalog.sources():
        assert watch.staging_dir() not in src.parents
    # And the catalogue still loads.
    assert len(catalog.load()) > 5


def test_a_robot_never_marks_an_entry_verified(tmp_path):
    """`verified` means a human checked it. A robot writing true would make the
    catalogue's `?` marker meaningless."""
    watch.run(fetch, dest=tmp_path)
    for spec in watch.pending(tmp_path):
        assert spec["verified"] is False


def test_a_robot_never_marks_a_licence_clean(tmp_path):
    """license_ok gates commercial use — a reading task, not a parsing task."""
    watch.run(fetch, dest=tmp_path)
    for spec in watch.pending(tmp_path):
        assert spec["license_ok"] is False


def test_a_parameter_count_is_never_invented(tmp_path):
    """config.json does not carry one. An invented count looks right and is not."""
    watch.run(fetch, dest=tmp_path)
    for spec in watch.pending(tmp_path):
        assert "params_b" not in spec


def test_an_unparseable_config_is_skipped_with_its_reason(tmp_path):
    """Filling in the missing field is the one thing this must never do."""
    r = watch.run(fetch, dest=tmp_path)
    bad = next(o for o in r.outcomes if o.repo == "acme/broken")
    assert not bad.added
    assert "layer count" in bad.reason


def test_geometry_is_taken_from_the_config_not_guessed(tmp_path):
    watch.run(fetch, dest=tmp_path)
    spec = next(s for s in watch.pending(tmp_path) if s["id"] == "new-9b")
    assert spec["layers"] == 32
    assert spec["kv_heads"] == 8
    assert spec["head_dim"] == 4096 // 32


def test_the_run_is_capped(tmp_path):
    many = json.dumps(
        [
            {
                "id": f"acme/m{i}",
                "downloads": 1000 - i,
                "likes": 9,
                "pipeline_tag": "text-generation",
            }
            for i in range(20)
        ]
    )

    def f(url: str) -> str:
        return GOOD if "/m" in url and "config" in url else many

    r = watch.run(f, dest=tmp_path, max_add=3)
    assert len(r.added) == 3
    assert any("cap" in o.reason for o in r.outcomes if not o.added)


def test_a_second_pass_adds_nothing(tmp_path):
    """Idempotent, so a daily timer does not accumulate duplicates."""
    watch.run(fetch, dest=tmp_path)
    assert not watch.run(fetch, dest=tmp_path).added


def test_offline_is_a_normal_state_not_an_error(tmp_path):
    """A laptop that was asleep must not produce an alarm."""

    def dead(url: str) -> str:
        raise OSError("no network")

    r = watch.run(dead, dest=tmp_path)
    assert r.offline
    assert "unreachable" in r.render()


def test_the_run_leaves_a_machine_readable_receipt(tmp_path):
    """'Where did this model come from' must always have an answer."""
    doc = json.loads(watch.run(fetch, dest=tmp_path).to_json())
    assert doc["added"] == ["acme/new-9b"]
    assert doc["skipped"][0]["repo"] == "acme/broken"
    assert doc["skipped"][0]["why"]


def test_a_corrupt_staging_file_does_not_break_the_listing(tmp_path):
    watch.run(fetch, dest=tmp_path)
    (tmp_path / "corrupt.json").write_text("{not json")
    assert watch.pending(tmp_path), "the good entries must still list"


def test_nothing_is_scheduled_without_being_asked():
    """Installing a recurring networked job is the user's decision."""
    fragment, where = watch.install_schedule(interval_hours=12)
    assert "clickllm.cli" in fragment and "watch" in fragment
    assert where
    # 12 hours, expressed in whatever this platform's units are.
    assert "43200" in fragment or "12h" in fragment


def test_offline_means_offline_and_not_three_other_things():
    """`discover()` returned [] for three unrelated states and `run()` called
    them all `offline=True`.

    The three: the fetch raised (genuinely offline), the index was malformed,
    and the index was fine but listed nothing we do not already carry. Reporting
    a corrupt index as "nothing new upstream" is the exact drift this module
    exists to notice — a watcher that reports no change when it could not check.
    """
    import json as _json

    from clickllm import catalog_update as cu

    rows = [{"modelId": "acme/new", "downloads": 5, "likes": 1, "trendingScore": 9.0}]

    with pytest.raises(cu.Unreachable):
        cu.discover(set(), lambda _u: (_ for _ in ()).throw(OSError("no network")))

    with pytest.raises(cu.Unreachable):
        cu.discover(set(), lambda _u: _json.dumps({"error": "nope"}))

    # Reached the index; it simply had nothing new. Empty, but NOT a failure.
    assert cu.discover({"acme/new"}, lambda _u: _json.dumps(rows)) == []


def test_the_licence_the_index_reported_is_the_licence_that_is_staged():
    """`watch` read `getattr(d, "license", "")` off a `Discovery` that had no
    such field, so every staged entry was written `"unknown"` — against a
    docstring promising "the licence string as the index reported it".

    A human reviewing a staged discovery to decide whether to promote it saw
    "unknown" for every model, forcing the lookup the tool claims to have done.
    """
    import json as _json

    from clickllm import catalog_update as cu

    rows = [
        {
            "modelId": "acme/a",
            "downloads": 5,
            "likes": 1,
            "trendingScore": 9.0,
            "license": "apache-2.0",
        },
        {
            "modelId": "acme/b",
            "downloads": 4,
            "likes": 1,
            "trendingScore": 8.0,
            "cardData": {"license": "mit"},
        },
        {"modelId": "acme/c", "downloads": 3, "likes": 1, "trendingScore": 7.0},
    ]
    got = {d.repo: d.license for d in cu.discover(set(), lambda _u: _json.dumps(rows))}
    assert got["acme/a"] == "apache-2.0"
    assert got["acme/b"] == "mit", "the index also reports it under cardData"
    # Absent stays absent — never guessed. That distinction is the invariant.
    assert got["acme/c"] == ""
