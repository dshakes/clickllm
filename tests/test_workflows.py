"""The CI workflows are code, and they have had the same defect as the code.

The shape this repo keeps removing is a green — or red — signal standing over
something that did not happen: a skipped audit reading as clean, a release
workflow no-oping on a version tag, a reviewer filing a placeholder body. The
workflows themselves were never asserted on, so each of those was found by a
human noticing, one at a time, after it had already misled someone.

These tests are the cheap half of that. They cannot prove a workflow runs
correctly on GitHub, and they do not try; they assert the wiring whose absence
already caused a specific failure, so re-introducing it costs a red test rather
than a wasted run.

`pyyaml` is a test-time dependency only — `clickllm fit` still has none.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    import yaml
except ImportError:  # pragma: no cover — depends on how the suite was invoked
    yaml = None

# A silent skip is the exact defect these tests exist to catch: the file looks
# present, the run looks green, and nothing was checked. So it is a skip on a
# laptop and a hard stop in CI, the same way CLICKLLM_REQUIRE_ENGINES makes
# "could not ask the engine" fail rather than pass quietly. Raised at import so
# it reads as one clear cause, not as every test below failing on a NoneType.
if yaml is None and os.environ.get("CI"):
    raise RuntimeError(
        "pyyaml is missing in CI, so every workflow test in this file would have "
        "skipped without checking anything. Restore `--with pyyaml` in ci.yml."
    )

pytestmark = pytest.mark.skipif(
    yaml is None,
    reason="pyyaml not installed — run with `uv run --with pytest --with pyyaml`",
)

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def _steps(name: str, job: str) -> list[dict]:
    doc = yaml.safe_load((WORKFLOWS / name).read_text())
    return doc["jobs"][job]["steps"]


def _by_name(steps: list[dict], prefix: str) -> dict:
    hits = [s for s in steps if str(s.get("name", "")).startswith(prefix)]
    assert len(hits) == 1, f"expected exactly one step starting {prefix!r}, got {len(hits)}"
    return hits[0]


def test_every_workflow_parses():
    """A workflow with a YAML error does not fail loudly — GitHub just never runs
    it, which looks identical to a workflow that ran and found nothing."""
    files = sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml"))
    assert files, "no workflows found — wrong path?"
    for f in files:
        doc = yaml.safe_load(f.read_text())
        assert isinstance(doc, dict) and "jobs" in doc, f"{f.name}: no jobs"


def test_a_builder_that_runs_out_of_turns_does_not_lose_the_work_it_did():
    """Observed on #50: the auto-fixer hit its 25-turn cap, the action exited 1,
    and the push step below it was skipped by GitHub's default `success()` — so
    any commit it had already made was discarded, and the PR carried a red
    `fix` check that read like a broken build.

    The builder must tolerate its own failure, and the push must run anyway.
    """
    steps = _steps("sdlc-fix.yml", "fix")
    builder = _by_name(steps, "Claude fix")
    push = _by_name(steps, "Commit & push")

    assert builder.get("continue-on-error") is True, (
        "the builder's failure will skip the push step and discard its commits"
    )
    assert builder.get("id") == "builder", "the outcome step reads steps.builder.outcome"
    assert "always()" in str(push.get("if", "")), (
        "the push must run even when the builder failed, or partial work is lost"
    )
    assert push.get("id") == "push", "the outcome step reads steps.push.outputs.pushed"


def test_the_fix_job_says_which_of_its_three_outcomes_happened():
    """ "Gave up" and "found nothing to change" need opposite responses from a
    human, and both produced the same silence. The reporting step must
    distinguish them, and must run even when the builder failed."""
    steps = _steps("sdlc-fix.yml", "fix")
    say = _by_name(steps, "Say which of the three")

    assert "always()" in str(say.get("if", "")), "must report on the failure path too"
    run = say["run"]
    # The two causes are named separately, off the builder's own outcome.
    assert "$OUTCOME" in run and "failure" in run
    assert "turn cap" in run, "must name the cause a human would act on"
    assert "produced no commit" in run
    # And it stays quiet on the happy path rather than commenting every round.
    assert 'if [ "$PUSHED" = "true" ]' in run and "exit 0" in run


def test_a_queued_build_compiled_leg_cannot_block_later_releases():
    """On 2026-08-13 the macos-13 leg of `build-compiled` sat `queued` for four
    hours waiting for runner capacity. The workflow-level `concurrency: release`
    lock at the time covered the whole run, so that queued (not even started)
    job held the lock and every later release queued behind it forever.
    `timeout-minutes` cannot fix this — it only bounds time after a runner
    picks the job up, never time spent queued.

    The fix is scope, not a bigger timeout: no workflow-level `concurrency:`,
    and `build-compiled` must not opt into the `release` group that the jobs
    touching shared state (the tag, PyPI, npm, the Homebrew PR) use.
    """
    doc = yaml.safe_load((WORKFLOWS / "release.yml").read_text())
    assert "concurrency" not in doc, (
        "a workflow-level concurrency group locks the whole run, including "
        "build-compiled — a queued matrix leg would hold it again"
    )

    jobs = doc["jobs"]
    assert "concurrency" not in jobs["build-compiled"], (
        "build-compiled must stay outside the release lock, or a runner "
        "shortage on one matrix leg blocks every other release"
    )

    locked_jobs = ["build", "publish-pypi", "publish-compiled", "publish-npm", "homebrew"]
    for name in locked_jobs:
        c = jobs[name].get("concurrency")
        assert c, f"{name} must serialize against other release runs via job-level concurrency"
        assert c.get("group") == "release", f"{name} concurrency group: {c.get('group')!r}"
        assert c.get("cancel-in-progress") is False, (
            f"{name}: an in-progress release must not be cancelled by a newer one"
        )


def test_the_release_workflow_still_fires_on_a_version_tag():
    """It was `workflow_dispatch`-only once. Pushing v0.1.0 did nothing, said
    nothing, and left a tag on origin with no release behind it."""
    doc = yaml.safe_load((WORKFLOWS / "release.yml").read_text())
    # `on` is parsed as the boolean True by YAML 1.1 — the classic gotcha.
    trigger = doc.get("on", doc.get(True))
    assert trigger, "no trigger block"
    tags = trigger.get("push", {}).get("tags", [])
    assert any("[0-9]" in t for t in tags), f"a version tag will not release: {tags}"
