"""The weight cache the engines actually fill.

`cache.demo()` walks one sweep. These pin the properties that make deletion
safe, because every one of them is a way to lose 40 GB you needed: a size read
off symlinks, an LRU order that is not least-recently-used, a pin that a tight
budget overrides, and a plan that deletes before it is confirmed.

Every test builds its own cache under `tmp_path`. Nothing here reads the
developer's real one.
"""

from __future__ import annotations

import os
import time

import pytest

from clickllm import cache
from clickllm.cli import main

GB = 1024**3

#: `refs/main` is a real file in the layout and really costs bytes, so the
#: expected sizes below carry it rather than pretending a repo is only blobs.
SHA = "0123456789abcdef"
REF = len(SHA)
INCUMBENT, CANDIDATE, EVALSET = 40_000 + REF, 20_000 + REF, 5_000 + REF
TOTAL = INCUMBENT + CANDIDATE + EVALSET


def build(root, repos):
    """A synthetic hub cache: `repos` is (dirname, blob_bytes, age_days).

    Laid out the way the Hub really does it — a `blobs/` directory holding the
    bytes and a `snapshots/<sha>/` directory of symlinks into it — because that
    layout is exactly what a naive size calculation gets wrong.
    """
    now = time.time()
    for name, size, age_days in repos:
        blob = root / name / "blobs" / f"sha-{name}"
        blob.parent.mkdir(parents=True)
        blob.write_bytes(b"\0" * size)
        snap = root / name / "snapshots" / SHA
        snap.mkdir(parents=True)
        (snap / "model.safetensors").symlink_to(blob)
        (root / name / "refs").mkdir()
        (root / name / "refs" / "main").write_text(SHA)
        when = now - age_days * 86400
        for path in (blob, root / name / "refs" / "main"):
            os.utime(path, (when, when))
    return root


@pytest.fixture
def hub(tmp_path):
    """Three repos. The incumbent is both the oldest and the largest — the case
    where an unpinned LRU sweep would take exactly the wrong thing."""
    root = tmp_path / "hub"
    root.mkdir()
    return build(
        root,
        [
            ("models--acme--incumbent-70b", 40_000, 30),
            ("models--acme--candidate-8b", 20_000, 1),
            ("datasets--acme--evalset", 5_000, 10),
        ],
    )


# --- what is on disk -----------------------------------------------------------


def test_size_is_what_is_really_on_disk(hub):
    """Snapshots are symlinks into blobs. Following them reports a 4 GB model as
    8 GB, which is the first bug anyone writing this hits."""
    by_repo = {e.repo: e for e in cache.entries(hub)}
    assert by_repo["acme/incumbent-70b"].bytes == INCUMBENT
    assert by_repo["acme/candidate-8b"].bytes == CANDIDATE
    assert by_repo["acme/evalset"].bytes == EVALSET

    on_disk = sum(
        os.lstat(os.path.join(p, f)).st_size
        for p, _d, files in os.walk(hub)
        for f in files
        if not os.path.islink(os.path.join(p, f))
    )
    assert cache.usage(hub).total_bytes == on_disk == TOTAL


def test_the_repo_kind_comes_from_the_directory_prefix(hub):
    kinds = {e.repo: e.kind for e in cache.entries(hub)}
    assert kinds["acme/evalset"] == "dataset"
    assert kinds["acme/incumbent-70b"] == "model"


def test_non_repo_entries_are_ignored(hub):
    """The hub root also holds CACHEDIR.TAG, version.txt and .locks/."""
    (hub / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172")
    (hub / "version.txt").write_text("1")
    (hub / ".locks").mkdir()
    assert len(cache.entries(hub)) == 3


def test_a_missing_cache_is_empty_not_an_error(tmp_path):
    assert cache.entries(tmp_path / "nope") == []
    assert cache.usage(tmp_path / "nope").total_bytes == 0


def test_the_hub_overrides_are_honoured(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assert cache.hub_dir() == tmp_path / "hf" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "elsewhere"))
    assert cache.hub_dir() == tmp_path / "elsewhere"


# --- the eviction plan ---------------------------------------------------------


def test_eviction_order_is_genuinely_least_recently_used(hub):
    found = cache.entries(hub)
    plan = cache.plan_eviction(found, CANDIDATE)
    # 30d then 10d; the 1d repo is what the budget leaves room for.
    assert [e.repo for e in plan.evict] == ["acme/incumbent-70b", "acme/evalset"]
    assert plan.after_bytes == CANDIDATE
    assert plan.freed_bytes == INCUMBENT + EVALSET


def test_it_stops_as_soon_as_the_budget_is_met(hub):
    """One repo over budget is one repo deleted, not the whole tail."""
    plan = cache.plan_eviction(cache.entries(hub), CANDIDATE + EVALSET)
    assert [e.repo for e in plan.evict] == ["acme/incumbent-70b"]
    assert plan.after_bytes == CANDIDATE + EVALSET


def test_a_budget_already_met_evicts_nothing(hub):
    assert cache.plan_eviction(cache.entries(hub), 10 * GB).evict == ()


def test_a_pinned_repo_is_never_in_a_plan_even_when_it_is_the_oldest(hub):
    """The point of the feature: an eval sweep must not evict the incumbent it
    is comparing against, even when the budget cannot otherwise be met."""
    found = cache.entries(hub)
    plan = cache.plan_eviction(found, 1_000, pinned=("acme/incumbent-70b",))

    assert "acme/incumbent-70b" not in [e.repo for e in plan.evict]
    # Everything else went, and it is still over. The plan says so.
    assert sorted(e.repo for e in plan.evict) == ["acme/candidate-8b", "acme/evalset"]
    assert plan.after_bytes == INCUMBENT
    assert plan.short_by == INCUMBENT - 1_000
    assert "pinned" in plan.render()


def test_a_pin_matches_regardless_of_case(hub):
    plan = cache.plan_eviction(cache.entries(hub), 0, pinned=("ACME/Incumbent-70B",))
    assert "acme/incumbent-70b" not in [e.repo for e in plan.evict]


def test_a_negative_budget_is_refused_with_its_value(hub):
    with pytest.raises(ValueError, match="-5"):
        cache.plan_eviction(cache.entries(hub), -5)


# --- deletion ------------------------------------------------------------------


def test_nothing_is_deleted_without_confirmation(hub):
    plan = cache.plan_eviction(cache.entries(hub), 0)
    result = cache.apply_eviction(plan, root=hub)

    assert result.removed == ()
    assert not result.confirmed
    for e in plan.evict:
        assert e.path.exists()


def test_evict_without_yes_deletes_nothing(hub, tmp_path, monkeypatch, capsys):
    """The CLI path, which is the one a person actually runs."""
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["cache", "evict", "--budget", "1K"]) == 0

    out = capsys.readouterr().out
    assert "would delete" in out and "re-run with --yes" in out
    for name in ("models--acme--incumbent-70b", "models--acme--candidate-8b"):
        assert (hub / name).exists(), "evict without --yes must not touch the disk"
        assert (hub / name / "blobs").exists()


def test_yes_deletes_exactly_the_plan(hub):
    plan = cache.plan_eviction(cache.entries(hub), CANDIDATE)
    result = cache.apply_eviction(plan, confirm=True, root=hub)

    assert result.removed == ("acme/incumbent-70b", "acme/evalset")
    assert result.freed_bytes == INCUMBENT + EVALSET
    assert not (hub / "models--acme--incumbent-70b").exists()
    assert (hub / "models--acme--candidate-8b" / "blobs").exists()


def test_nothing_outside_the_cache_root_is_ever_a_target(hub, tmp_path):
    """A path that is not a repo directory directly under the root raises rather
    than being skipped — reaching that point is a defect, and stopping is the
    only response that cannot lose data."""
    precious = tmp_path / "models--acme--elsewhere"
    precious.mkdir()
    (precious / "keep").write_text("do not delete me")

    outside = cache.Entry("acme/elsewhere", "model", precious, 10, time.time())
    plan = cache.Plan(evict=(outside,), kept_pins=(), budget_bytes=0, before_bytes=10)
    with pytest.raises(ValueError, match="outside the hub cache"):
        cache.apply_eviction(plan, confirm=True, root=hub)
    assert (precious / "keep").exists()


def test_the_cache_root_itself_is_never_a_target(hub):
    entry = cache.Entry("acme/x", "model", hub, 1, time.time())
    plan = cache.Plan(evict=(entry,), kept_pins=(), budget_bytes=0, before_bytes=1)
    with pytest.raises(ValueError, match="outside the hub cache"):
        cache.apply_eviction(plan, confirm=True, root=hub)
    assert hub.exists()


def test_a_non_repo_directory_inside_the_root_is_refused(hub):
    locks = hub / ".locks"
    locks.mkdir()
    entry = cache.Entry("locks", "model", locks, 1, time.time())
    plan = cache.Plan(evict=(entry,), kept_pins=(), budget_bytes=0, before_bytes=1)
    with pytest.raises(ValueError, match="non-repo directory"):
        cache.apply_eviction(plan, confirm=True, root=hub)
    assert locks.exists()


# --- pins and budget persist outside HF's cache --------------------------------


def test_pins_live_in_clickllm_state_not_in_the_hub_cache(hub, tmp_path):
    state = tmp_path / "cache.json"
    cache.pin("acme/incumbent-70b", state)

    assert cache.load_state(state).pinned == ("acme/incumbent-70b",)
    assert state.exists()
    # Nothing was written into the directory huggingface_hub owns.
    assert sorted(p.name for p in hub.iterdir()) == [
        "datasets--acme--evalset",
        "models--acme--candidate-8b",
        "models--acme--incumbent-70b",
    ]


def test_pinning_twice_is_not_two_pins(tmp_path):
    state = tmp_path / "cache.json"
    cache.pin("acme/x", state)
    assert cache.pin("acme/x", state).pinned == ("acme/x",)


def test_unpinning_something_that_was_not_pinned_says_so(tmp_path):
    state = tmp_path / "cache.json"
    cache.pin("acme/x", state)
    with pytest.raises(ValueError, match="acme/typo"):
        cache.unpin("acme/typo", state)
    assert cache.load_state(state).pinned == ("acme/x",)


def test_a_corrupt_state_file_is_empty_state_not_a_crash(tmp_path):
    state = tmp_path / "cache.json"
    state.write_text("{not json")
    assert cache.load_state(state) == cache.State()


def test_a_pin_survives_into_the_cli_plan(hub, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["cache", "pin", "acme/incumbent-70b"]) == 0
    capsys.readouterr()
    assert main(["cache", "evict", "--budget", "1K", "--yes"]) == 0

    out = capsys.readouterr().out
    assert "acme/incumbent-70b" not in out.split("kept, pinned:")[0]
    assert (hub / "models--acme--incumbent-70b").exists(), "a pin must survive --yes"
    assert not (hub / "models--acme--candidate-8b").exists()


def test_a_remembered_budget_is_announced_and_reused(hub, tmp_path, monkeypatch, capsys):
    """Persisting state as a side effect of a report is only acceptable if it
    is asked for and said out loud."""
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["cache", "--budget", "1K"]) == 0
    assert "budget remembered" in capsys.readouterr().out

    # evict now needs no --budget, and nothing is deleted regardless.
    assert main(["cache", "evict"]) == 0
    assert "would delete" in capsys.readouterr().out
    assert (hub / "models--acme--incumbent-70b").exists()


def test_a_report_without_a_budget_writes_nothing(hub, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))

    assert main(["cache"]) == 0
    capsys.readouterr()
    assert not (tmp_path / "state").exists(), "a read-only report must not create state"


def test_evict_with_no_budget_anywhere_says_so(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert main(["cache", "evict"]) == 2  # a sentence and a nonzero exit, never a traceback


def test_pin_without_a_repo_is_a_nonzero_exit(hub, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert main(["cache", "pin"]) == 2


def test_the_report_reads_the_cache_and_changes_nothing(hub, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    before = sorted(p.name for p in hub.iterdir())

    assert main(["cache"]) == 0

    out = capsys.readouterr().out
    assert "acme/incumbent-70b" in out and "free on disk" in out
    assert sorted(p.name for p in hub.iterdir()) == before


def test_a_budget_over_the_limit_names_the_overage(hub):
    u = cache.usage(hub, budget_bytes=10_000)
    assert u.over_by == TOTAL - 10_000
    assert "over budget" in u.render()


# --- sizes ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("200G", 200 * GB), ("200gb", 200 * GB), ("1.5T", int(1.5 * 1024**4)), ("4096", 4096)],
)
def test_budgets_parse(text, expected):
    assert cache.parse_bytes(text) == expected


@pytest.mark.parametrize("text", ["", "lots", "200X", "-5G"])
def test_a_bad_budget_carries_its_value(text):
    """A misread budget deletes the wrong amount, so it must never be guessed."""
    with pytest.raises(ValueError, match="not a size|negative"):
        cache.parse_bytes(text)


def test_demo_runs():
    cache.demo()
