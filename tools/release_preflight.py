#!/usr/bin/env python3
"""Check the repo settings this project's automation assumes, before a release.

Cutting v1.0.0 failed three times, and not one of the failures was in the code:

1. Three crates pinned intra-workspace deps at `0.1.0` — a caret requirement
   that matched every 0.1.x and cannot match 1.0.0.
2. The Homebrew step could not open a PR, because "Allow GitHub Actions to
   create and approve pull requests" was off — and it was the last step of
   `build`, so `publish-pypi` never ran and 1.0.0 existed as a GitHub Release
   while PyPI still served 0.1.9.
3. The pinned PyPI publish action bundled a twine too old for the
   `Metadata-Version: 2.5` that `uv build` now emits.

Each was discoverable in seconds and discovered in a half-hour round trip
through CI. This is that half-hour, run beforehand.

    python3 tools/release_preflight.py            # check
    python3 tools/release_preflight.py --strict   # exit 1 on warnings too

**This is a preflight, not a gate.** It cannot enforce anything — the settings
it reads belong to whoever owns the repo, and reading some of them needs admin
rights this may not have. A check it could not perform is reported as
`UNKNOWN`, never as a pass, because "I could not look" and "it is fine" are
different claims and this whole project is about not confusing them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "FAIL", "unknown"


def _gh(*args: str) -> tuple[bool, str]:
    """Run `gh` and return (succeeded, output). Never raises."""
    gh = shutil.which("gh")
    if gh is None:
        return False, "gh is not on PATH"
    try:
        p = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [gh, *args], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    return p.returncode == 0, (p.stdout or p.stderr).strip()


def _repo() -> str | None:
    ok, out = _gh("repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner")
    return out if ok else None


def check_actions_can_open_prs(repo: str) -> tuple[str, str]:
    """The setting that stranded the v1.0.0 formula bump."""
    ok, out = _gh("api", f"repos/{repo}/actions/permissions/workflow")
    if not ok:
        return UNKNOWN, "could not read workflow permissions (needs admin)"
    try:
        can = json.loads(out).get("can_approve_pull_request_reviews")
    except json.JSONDecodeError:
        return UNKNOWN, "unreadable response"
    if can:
        return OK, "Actions may open PRs"
    return (
        FAIL,
        "Actions cannot open PRs — the Homebrew formula bump will push a branch "
        "and fail. Settings → Actions → General → Workflow permissions",
    )


def check_auto_merge(repo: str) -> tuple[str, str]:
    """`release.yml` calls `gh pr merge --auto`, which needs this on."""
    ok, out = _gh("api", f"repos/{repo}", "--jq", ".allow_auto_merge")
    if not ok:
        return UNKNOWN, "could not read repo settings"
    return (
        (OK, "auto-merge enabled")
        if out == "true"
        else (
            WARN,
            "auto-merge is off, so the formula PR waits for a manual merge",
        )
    )


def check_branch_protection(repo: str) -> tuple[str, str]:
    """Several comments in this repo assert main is protected. It was not.

    A warning rather than a failure: an unprotected main is a real and
    defensible choice for a solo repo, and this tool does not get to overrule
    it. What it does get to do is stop the workflows claiming otherwise.
    """
    # Two APIs, and only checking the old one is why this reported the exact
    # opposite of the truth. Classic branch protection lives at
    # `/branches/main/protection`; a **ruleset** does not appear there at all —
    # that endpoint 404s — and `main` is governed by a ruleset requiring
    # `review` and `qa`. So this printed "no branch protection" on 2026-08-13
    # while a direct `git push` to main was being rejected with GH013 by the
    # very rules it could not see. A control that cannot observe the thing it
    # guards is worse than no control: it reports safety as absent and, had the
    # rules been removed, would have said nothing different.
    ok, out = _gh("api", f"repos/{repo}/rules/branches/main")
    if ok:
        try:
            kinds = {r.get("type") for r in json.loads(out)}
        except (json.JSONDecodeError, TypeError, AttributeError):
            kinds = set()
        if "required_status_checks" in kinds:
            return OK, "main is governed by a ruleset requiring status checks"
        if kinds:
            return (
                WARN,
                f"main has a ruleset ({', '.join(sorted(kinds))}) but it does not "
                "require status checks, so a red PR can still merge",
            )
    if _gh("api", f"repos/{repo}/branches/main/protection")[0]:
        return OK, "main is protected (classic branch protection)"
    return (
        WARN,
        "main has no branch protection or ruleset — every 'a human merges this' "
        "guarantee in the workflows is convention, not enforcement",
    )


def check_version_consistency() -> tuple[str, str]:
    """The bumper's own report, which covers the pins that broke the 1.0.0 build."""
    try:
        p = subprocess.run(  # noqa: S603
            [sys.executable, str(ROOT / "tools" / "bump.py")],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return UNKNOWN, f"could not run bump.py: {e}"
    last = (p.stdout or "").strip().splitlines()[-1] if p.stdout.strip() else ""
    if p.returncode == 0 and "consistent" in last:
        return OK, last
    return FAIL, last or "bump.py reported a version disagreement"


def check_workspace_builds() -> tuple[str, str]:
    """`cargo metadata` resolves the dependency graph without compiling.

    This is the check that would have caught the `0.1.0` caret pins in a second
    rather than after a tag was already pushed.

    **Without `--no-deps`, deliberately.** That flag means "do not resolve
    dependencies", which is precisely the work this check exists to do — the
    first version passed it and reported `ok` against the very pin mismatch it
    was written for. Resolution is the check; skipping it leaves a check that
    reads the manifests and confirms they parse.
    """
    cargo = shutil.which("cargo")
    if cargo is None:
        return UNKNOWN, "cargo is not on PATH"
    try:
        p = subprocess.run(  # noqa: S603
            [cargo, "metadata", "--format-version", "1"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        return UNKNOWN, str(e)
    if p.returncode == 0:
        return OK, "the Rust workspace resolves"
    first = (p.stderr or "").strip().splitlines()[0] if p.stderr.strip() else "failed"
    return FAIL, first


def check_publish_action_is_current() -> tuple[str, str]:
    """The pin that rejected our wheels for `Metadata-Version: 2.5`.

    Compares the pinned commit against the action's current release. A pin
    behind the latest is not automatically wrong — pinning is the point — so
    this is a warning that says what to check, not a failure.
    """
    wf = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    m = re.search(r"pypa/gh-action-pypi-publish@([0-9a-f]{40})", wf)
    if not m:
        return UNKNOWN, "no pinned publish action found"
    pinned = m.group(1)
    ok, tag = _gh("api", "repos/pypa/gh-action-pypi-publish/releases/latest", "--jq", ".tag_name")
    if not ok:
        return UNKNOWN, f"pinned {pinned[:8]}; could not reach GitHub to compare"
    ok, ref = _gh(
        "api", f"repos/pypa/gh-action-pypi-publish/git/ref/tags/{tag}", "--jq", ".object.sha"
    )
    if not ok:
        return UNKNOWN, f"pinned {pinned[:8]}; could not resolve {tag}"
    ok, commit = _gh(
        "api", f"repos/pypa/gh-action-pypi-publish/git/tags/{ref}", "--jq", ".object.sha"
    )
    current = commit if ok else ref
    if current.startswith(pinned) or pinned.startswith(current[:40]):
        return OK, f"publish action is at {tag}"
    return (
        WARN,
        f"publish action pinned at {pinned[:8]}, latest {tag} is {current[:8]} — "
        "an old twine here rejects new wheel metadata",
    )


#: (name, callable). Repo-scoped checks take the repo slug; local ones take nothing.


def check_homebrew_tap_secret(repo: str) -> tuple[str, str]:
    """The tap sync skips silently without this, and reports success anyway.

    Up to and including v1.1.1 the workflow read `HOMEBREW_TAP_PAT` while the
    secret in this repo is `HOMEBREW_TAP_TOKEN`. The name never matched, so the
    step took its skip path on every release and still exited 0 — which is why
    `brew install onpar` has never resolved. This checks the name that is
    actually configured, so a rename on either side is caught before a tag.
    """
    ok, out = _gh("api", f"repos/{repo}/actions/secrets")
    if not ok:
        return UNKNOWN, "could not list secrets (needs admin)"
    try:
        names = {s["name"] for s in json.loads(out).get("secrets", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return UNKNOWN, "could not parse the secrets list"
    if "HOMEBREW_TAP_TOKEN" in names:
        return OK, "HOMEBREW_TAP_TOKEN is set"
    return (
        FAIL,
        "HOMEBREW_TAP_TOKEN is not set — the tap sync will not update "
        "dshakes/homebrew-tap, so `brew install onpar` stays broken",
    )


def check_formula_pr_token(repo: str) -> tuple[str, str]:
    """Without a PAT the formula PR opens as a bot and never runs its checks.

    A PR opened with the default `GITHUB_TOKEN` is attributed to
    app/github-actions, and every workflow run on that branch lands in
    `action_required` waiting for a human. `main`'s ruleset requires `review`
    and `qa`, so the PR reports "no checks reported" and can never merge —
    while the job that opened it reports success. Both releases on 2026-08-13
    stalled exactly there and needed the runs approving by hand.
    """
    ok, out = _gh("api", f"repos/{repo}/actions/secrets")
    if not ok:
        return UNKNOWN, "could not list secrets (needs admin)"
    try:
        names = {s["name"] for s in json.loads(out).get("secrets", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return UNKNOWN, "could not parse the secrets list"
    if "SDLC_BOT_TOKEN" in names:
        return OK, "SDLC_BOT_TOKEN is set — the formula PR opens as a user"
    return (
        FAIL,
        "SDLC_BOT_TOKEN is not set — the Homebrew formula PR will open as a bot "
        "and its required checks will sit at action_required until someone "
        "approves them by hand",
    )


CHECKS = (
    ("version consistency", check_version_consistency, False),
    ("rust workspace resolves", check_workspace_builds, False),
    ("pypi publish action", check_publish_action_is_current, False),
    ("actions can open PRs", check_actions_can_open_prs, True),
    ("homebrew tap secret", check_homebrew_tap_secret, True),
    ("formula PR token", check_formula_pr_token, True),
    ("auto-merge", check_auto_merge, True),
    ("branch protection", check_branch_protection, True),
)


def main() -> int:
    strict = "--strict" in sys.argv
    repo = _repo()
    if repo is None:
        print("  note: `gh` unavailable or not authenticated — repo checks will be UNKNOWN\n")

    worst = OK
    rows = []
    for name, fn, needs_repo in CHECKS:
        if needs_repo:
            status, detail = (UNKNOWN, "no repo") if repo is None else fn(repo)
        else:
            status, detail = fn()
        rows.append((status, name, detail))
        if status == FAIL or (strict and status in (WARN, UNKNOWN)):
            worst = FAIL
        elif status in (WARN, UNKNOWN) and worst == OK:
            worst = WARN

    width = max(len(n) for _, n, _ in rows)
    print()
    for status, name, detail in rows:
        print(f"  {status:<8}{name:<{width + 2}}{detail}")
    print()
    if worst == FAIL:
        print("  Fix the FAILs before tagging. Each one of these has already cost a")
        print("  release round trip once.\n")
        return 1
    print("  Preflight clear. This checks preconditions, not the code — run the")
    print("  gate as well.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
