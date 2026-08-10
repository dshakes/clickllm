"""Discovery staging, advice that has to be actionable, and an installer that
must stay inside the directory it was given.

Three modules, one shape between them: each was correct on the path its `demo()`
walks and wrong on a path nothing exercised.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clickllm import desktop, watch
from clickllm.advise import Observed, Plan, Requirements, Workload, reconcile
from clickllm.plan import Engine

_ARCH = {
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 131072,
}


def _index(rows: list[dict]):
    """A fetcher: the model index for the listing URL, a config for the rest."""

    def fetch(url: str) -> str:
        if "models" in url and "resolve" not in url:
            return json.dumps(rows)
        return json.dumps(_ARCH)

    return fetch


# --- two orgs, one short name ----------------------------------------------------


def test_two_repos_with_the_same_short_name_are_both_staged(tmp_path):
    """The staged filename was `discovered-{id}.json`, and `id` is the last path
    segment — so `org/alpha` and `other/alpha` produced one filename.

    The second was reported "already discovered", which was false: it had never
    been seen. It was never staged, never listed, and never mentioned again.
    """
    rows = [{"modelId": "org/alpha", "downloads": 9}, {"modelId": "other/alpha", "downloads": 8}]
    report = watch.run(_index(rows), dest=tmp_path)

    added = sorted(o.repo for o in report.outcomes if o.added)
    assert added == ["org/alpha", "other/alpha"], [
        (o.repo, o.added, o.reason) for o in report.outcomes
    ]
    assert len(sorted(tmp_path.glob("*.json"))) == 2


def test_the_staged_name_keeps_the_org(tmp_path):
    assert watch.staged_name("org/alpha") != watch.staged_name("other/alpha")
    assert watch.staged_name("org/alpha").startswith("discovered-org-alpha-")
    assert watch.staged_name("org/alpha").endswith(".json")


def test_the_staged_name_does_not_collide_on_where_the_hyphen_falls(tmp_path):
    """`/` and `-` both sanitised to `-`, so `meta-llama/3` and `meta/llama-3`
    produced the identical filename — the second one, once staged, then read
    as "already discovered" for a repo that had never been seen."""
    assert watch.staged_name("meta-llama/3") != watch.staged_name("meta/llama-3")


def test_a_repo_really_seen_before_is_still_skipped(tmp_path):
    """The negative control: a filename unique enough to never collide would
    also never dedupe, and every run would restage everything."""
    rows = [{"modelId": "org/alpha", "downloads": 9}]
    first = watch.run(_index(rows), dest=tmp_path)
    second = watch.run(_index(rows), dest=tmp_path)

    assert [o.added for o in first.outcomes] == [True]
    assert [o.added for o in second.outcomes] == [False]
    assert second.outcomes[0].reason == "already discovered"


# --- a staged file that cannot be read -------------------------------------------


def test_an_unreadable_staged_file_does_not_block_rediscovery_forever(tmp_path):
    """Existence was the test, and readability is the question.

    A truncated file — a crash mid-write, a bad editor save — reported "already
    discovered" on every subsequent run while `pending()` skipped it silently.
    The model was invisible in both directions, permanently.
    """
    (tmp_path / watch.staged_name("org/alpha")).write_text('{"models": [{"id": "alpha"')

    report = watch.run(_index([{"modelId": "org/alpha", "downloads": 9}]), dest=tmp_path)
    # Exactly one outcome for the one repo checked — a rewrite is a single
    # event, not a skip followed by an add for the same model.
    assert len(report.outcomes) == 1, report.outcomes
    outcome = report.outcomes[0]
    assert outcome.added, "the rewrite must count as added, not skipped"
    assert "unreadable" in outcome.reason, outcome.reason
    assert "already discovered" not in outcome.reason

    # ...and it was rewritten, so the next run is a normal one.
    staged = json.loads((tmp_path / watch.staged_name("org/alpha")).read_text())
    assert staged["models"][0]["repo"] == "org/alpha"


def test_an_unreadable_staged_file_is_visible_in_the_listing(tmp_path):
    """`pending()` is the one place a human looks. It caught the parse error and
    `continue`d, which kept the listing working and made the problem invisible."""
    (tmp_path / "discovered-broken.json").write_text("{not json")
    listed = watch.pending(dest=tmp_path)
    assert any("unreadable" in str(entry.get("id", "")) for entry in listed), listed


def test_a_healthy_listing_is_not_polluted(tmp_path):
    """The negative control: reporting unreadable files must not put a marker in
    a listing that has nothing wrong with it."""
    watch.run(_index([{"modelId": "org/alpha", "downloads": 9}]), dest=tmp_path)
    listed = watch.pending(dest=tmp_path)
    assert [e["id"] for e in listed] == ["alpha"]


# --- advice that cannot be taken -------------------------------------------------


def _advice(peak: int) -> str | None:
    plan = Plan(engine=Engine.VLLM, engine_why="x", knobs=())
    req = Requirements(workload=Workload.INTERACTIVE, context=32768)
    hit = [
        s for s in reconcile(req, plan, Observed(peak_context=peak)) if s.id == "context-never-used"
    ]
    return hit[0].action if hit else None


@pytest.mark.parametrize("peak", [0, 1, 100, 500, 1023])
def test_a_context_nobody_can_serve_is_not_recommended(peak):
    """`peak_context * 2` with no floor produced "Consider serving 0 tokens
    instead of 32,768", and at a peak of 1, "serve 2 tokens".

    Zero is the worse half: it means nothing was observed, which is an absence
    of evidence rather than evidence of over-provisioning.
    """
    assert _advice(peak) is None, _advice(peak)


@pytest.mark.parametrize("peak", [1024, 4000, 8000])
def test_a_real_over_provisioning_is_still_reported(peak):
    """The negative control: a floor high enough to silence the absurd cases
    must not silence the case the suggestion exists for."""
    action = _advice(peak)
    assert action and f"{peak * 2:,}" in action, action


def test_a_context_that_is_used_is_left_alone():
    assert _advice(20000) is None


# --- an installer that stays where it was put ------------------------------------


def test_a_linux_install_writes_nothing_outside_the_directories_it_was_given(tmp_path):
    """`bin_dir` was hard-coded to the real home, so a caller passing `target=`
    had the entry land where they asked and the *script* land in
    `~/.local/bin` regardless.

    freedesktop genuinely does separate the two, which is why this is a second
    argument rather than something derived from `target` — but both must be the
    caller's to choose, or a test writes to the user's home.
    """
    launcher = desktop._linux(tmp_path / "apps", "python3", 8009, bin_dir=tmp_path / "bin")

    written = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file())
    assert written == ["apps/clickllm.desktop", "bin/clickllm-ui"], written
    assert str(tmp_path) in launcher.uninstall
    assert "clickllm-ui" in launcher.uninstall, "uninstall must remove both halves"


def test_the_entry_points_at_the_script_that_was_actually_written(tmp_path):
    """The two halves have to agree — an entry pointing at a path nothing wrote
    is a launcher that installs cleanly and does nothing."""
    desktop._linux(tmp_path / "apps", "python3", 8009, bin_dir=tmp_path / "bin")
    entry = (tmp_path / "apps" / "clickllm.desktop").read_text()
    exec_line = next(x for x in entry.splitlines() if x.startswith("Exec="))
    assert Path(exec_line.removeprefix("Exec=")).is_file()


def test_the_uninstall_command_survives_a_space_in_the_path(tmp_path):
    """The uninstall string is meant to be copy-pasted into a shell. Unquoted,
    a `bin_dir` like `/home/user/my setup/bin` turned one path into two shell
    words and `rm -f` deleted the wrong thing."""
    import shlex as _shlex

    spaced = tmp_path / "my setup"
    launcher = desktop._linux(spaced / "apps", "python3", 8009, bin_dir=spaced / "bin")

    words = _shlex.split(launcher.uninstall)
    assert str(spaced / "apps" / "clickllm.desktop") in words
    assert str(spaced / "bin" / "clickllm-ui") in words


def test_the_default_is_still_the_freedesktop_location(monkeypatch, tmp_path):
    """The default must not move: `~/.local/bin` is on PATH and the temporary
    directory a test passes is not."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    desktop._linux(tmp_path / "apps", "python3", 8009)
    assert (tmp_path / ".local" / "bin" / "clickllm-ui").is_file()


def test_the_module_self_checks_still_pass():
    watch.demo()
    with tempfile.TemporaryDirectory():
        pass
