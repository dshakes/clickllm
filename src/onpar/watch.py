"""The proactive half: notice new models without being asked.

`onpar catalog-add` is the reactive path — you heard about a model, you add
it. That covers the case where you already knew. The case it does not cover is
the one that actually costs you: a model shipped three weeks ago that would have
been the better answer, and nobody looked.

So this runs on a schedule, reads the public index, derives the geometry from
each candidate's own `config.json`, and **stages** what it finds. Nothing is
installed, nothing is served, nothing is decided, and nothing reaches the solver
until a human promotes it.

## What a background job is allowed to do

A job that runs unattended and writes to your catalogue is a job that can
quietly corrupt the thing every sizing answer depends on. This is not
hypothetical — the first cut of this module wrote straight into `models.d/`, and
because a discovery has no parameter count and `ModelSpec` requires one, every
subsequent `onpar fit` failed on the file the robot had just written. The
guardrails below are the design, not decoration:

**It stages; it does not publish.** Discoveries land in `~/.config/onpar/
discovered/`, which [`catalog.sources`] does not read. `onpar watch --list`
shows them, `onpar catalog-add` promotes them. A background job may bring
something to your attention; it may not put it in the path of a sizing answer.

**It never marks anything verified.** `verified` means a human compared the
entry against the model's published config. A robot writing `verified: true`
would make the catalogue's `?` marker meaningless, which is worse than not
running at all.

**It never marks a licence clean.** `license_ok` gates commercial use. That is a
reading task, not a parsing task, and a job that guessed it would be inviting
somebody to ship on a licence nobody read. Every discovery lands with
`license_ok: false` and the licence string as the index reported it.

**It never guesses geometry.** A config missing a field we need is skipped and
counted, not filled in with a plausible number. A guessed head count produces a
confident, wrong memory figure — the exact failure the catalogue exists to stop.

**It never estimates parameter counts.** `config.json` does not carry one, and
deriving it from the file listing is a guess. Discoveries land with the count
absent, which makes them visible-but-unsizeable until a human supplies it. That
is deliberate: an invented parameter count is a wrong answer that looks right.

**It is bounded and it leaves a receipt.** At most [`MAX_PER_RUN`] additions per
run, and a JSON log of what it added, skipped and why — so "where did this model
come from" always has an answer.

**Nothing leaves the machine except the fetches you enabled.** The job is opt-in
per NFR-2, and the only egress is the public index and the config files it names.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import catalog
from . import catalog_update as cu

__all__ = [
    "MAX_PER_RUN",
    "Outcome",
    "WatchReport",
    "install_schedule",
    "pending",
    "run",
    "staging_dir",
]

#: Additions per run. A trending index can move a lot in a day, and a job that
#: could add forty models overnight is a job that can bury the catalogue in
#: unsizeable entries. Small enough to read, large enough to keep up.
MAX_PER_RUN = 5

#: Fields a discovery cannot supply, and which therefore make the entry
#: incomplete-by-design rather than wrong.
NEEDS_A_HUMAN = ("params_b", "license_ok", "verified")


def staging_dir() -> Path:
    """Where discoveries land — deliberately NOT a directory `catalog` reads.

    This was a real bug before it was a design: writing straight into
    ``models.d/`` broke the entire catalogue. A discovery has no parameter count
    (``config.json`` does not carry one) and ``ModelSpec`` requires it, so every
    subsequent ``onpar fit`` failed on the file the robot had just written.

    Staging is the fix and the honest model of what discovery is: a background
    job may bring something to your attention, and may not put it in the path of
    a sizing answer. Promotion is `onpar catalog-add`, which is where the
    parameter count and the licence reading come from.
    """
    base = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    return Path(base) / "onpar" / "discovered"


def pending(dest: Path | None = None) -> list[dict]:
    """Discoveries waiting on a human, newest first."""
    d = dest or staging_dir()
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            out.extend(json.loads(p.read_text()).get("models", []))
        except (OSError, json.JSONDecodeError) as e:
            # Not `continue`. A corrupt staging file must not *break* the
            # listing, which is why this is caught — but skipping it silently
            # made the file invisible in the one place a human looks, while it
            # went on blocking re-discovery of that repo. Reported as an entry
            # so it is seen and can be deleted.
            out.append({"id": f"<unreadable: {p.name}>", "name": str(p), "error": str(e)})
    return out


@dataclass(frozen=True, slots=True)
class Outcome:
    """One candidate and what happened to it."""

    repo: str
    added: bool
    reason: str = ""

    def render(self) -> str:
        mark = "added  " if self.added else "skipped"
        return f"  {mark} {self.repo}" + (f" — {self.reason}" if self.reason else "")


@dataclass(frozen=True, slots=True)
class WatchReport:
    """What one run did. Written to disk so the answer to "where did this come
    from" is never "no idea"."""

    checked: int = 0
    outcomes: tuple[Outcome, ...] = ()
    #: Set when the index itself was unreachable, which is a normal state for a
    #: laptop that was asleep — not a failure worth alarming about.
    offline: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def added(self) -> tuple[Outcome, ...]:
        return tuple(o for o in self.outcomes if o.added)

    def render(self) -> str:
        if self.offline:
            return "  index unreachable — nothing to do"
        if not self.outcomes:
            return f"  checked {self.checked}, nothing new"
        lines = [f"  checked {self.checked}, added {len(self.added)}"]
        lines += [o.render() for o in self.outcomes]
        if self.added:
            lines.append("")
            lines.append(
                "  These are STAGED, not in the catalogue: a discovery has no "
                "parameter count and cannot be sized. Promote the ones you want:"
            )
            for o in self.added:
                lines.append(f"    onpar catalog-add {o.repo} --params-b <N> --network")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "checked": self.checked,
                "offline": self.offline,
                "added": [o.repo for o in self.added],
                "skipped": [
                    {"repo": o.repo, "why": o.reason} for o in self.outcomes if not o.added
                ],
                "notes": list(self.notes),
            },
            indent=2,
        )


def staged_name(repo: str) -> str:
    """Filename for a staged discovery, keyed on the whole repo.

    It used to be `discovered-{spec['id']}.json`, and `id` is the last path
    segment — so `org/alpha` and `other/alpha` are two different models that
    produced one filename. The second was reported "already discovered", which
    was false: it had never been seen, and it was never staged, listed or
    mentioned again.

    Sanitising `/` to `-` fixed that case but opened another: `-` was already a
    character the sanitiser passed through unchanged, so `meta-llama/3` and
    `meta/llama-3` both collapse to the same filename and the second one is
    again reported "already discovered" without ever having been seen. A
    digest of the *whole* repo string, not just its sanitised characters, is
    what makes two different repos always produce two different filenames.

    The `id` inside the file stays short, because that is the catalogue's
    convention and a human renames it on curation. Only the file's identity
    needs to be as unique as the thing it describes.
    """
    safe = "".join(c if c.isalnum() or c in "-._" else "-" for c in repo.strip("/"))
    digest = hashlib.sha256(repo.encode("utf-8")).hexdigest()[:10]
    return f"discovered-{safe.lower()}-{digest}.json"


def _spec_from(repo: str, arch: cu.Architecture, license_name: str) -> dict:
    """A catalogue entry from a config, with every human-judgement field withheld.

    ``params_b`` is deliberately absent rather than estimated — see the module
    docstring. The entry is real, complete on geometry, and honest about the
    parts a machine cannot know.
    """
    spec = {
        "id": repo.split("/")[-1].lower(),
        "name": repo.split("/")[-1],
        "layers": arch.layers,
        "kv_heads": arch.kv_heads,
        "head_dim": arch.head_dim,
        "kv_scheme": arch.kv_scheme,
        "max_context": arch.max_context,
        "license": license_name or "unknown",
        "license_ok": False,
        "quants": ["q4", "q8"],
        "verified": False,
        "repo": repo,
    }
    if arch.kv_lora_rank:
        spec["kv_lora_rank"] = arch.kv_lora_rank
    return spec


def run(
    fetch: Callable[[str], str],
    *,
    dest: Path | None = None,
    limit: int = 40,
    max_add: int = MAX_PER_RUN,
) -> WatchReport:
    """One pass: discover, derive, write. Returns what it did.

    ``fetch`` is injected rather than imported so that importing this module
    never opens a socket, and so the whole path is testable without a network.
    """
    dest = dest or staging_dir()
    known = {m.repo for m in catalog.load() if m.repo}
    # `offline` now means offline. It used to also mean "the index was
    # malformed" and "we already know everything it listed" — three states
    # under one flag, so a corrupt index reported as "nothing new upstream",
    # which is precisely the drift this module exists to notice.
    try:
        found = cu.discover(known, fetch, limit=limit)
    except cu.Unreachable:
        return WatchReport(offline=True)
    if not found:
        # Reached the index, and it listed nothing we do not already carry.
        # That is a successful check with an empty result, not a failed one.
        return WatchReport(offline=False)

    outcomes: list[Outcome] = []
    added = 0
    for d in found:
        if added >= max_add:
            outcomes.append(Outcome(d.repo, False, f"run cap of {max_add} reached"))
            continue
        url = f"https://huggingface.co/{d.repo}/raw/main/config.json"
        try:
            arch = cu.parse_config(json.loads(fetch(url)))
        except cu.ConfigError as e:
            # The honest outcome. A model whose config we cannot read is a model
            # we cannot size, and inventing the missing field is the one thing
            # this must never do.
            outcomes.append(Outcome(d.repo, False, str(e)))
            continue
        except Exception as e:  # noqa: BLE001 — a bad fetch is data, not a crash
            outcomes.append(Outcome(d.repo, False, f"could not read config: {e}"))
            continue

        spec = _spec_from(d.repo, arch, d.license)
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / staged_name(d.repo)
        rewrite_reason = ""
        if path.exists():
            # Existence is not the question — *readability* is. A file that
            # cannot be parsed carries nothing a human curated, and treating it
            # as "already discovered" blocked re-discovery of that repo
            # permanently while `pending()` skipped it silently. Nothing then
            # mentioned the model again, in either direction.
            try:
                json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                rewrite_reason = f"staged file was unreadable, rewrote it ({e})"
            else:
                outcomes.append(Outcome(d.repo, False, "already discovered"))
                continue
        # `models` wrapper, matching the hand-written format exactly — a
        # discovery must be indistinguishable from something you wrote, so you
        # can edit it in place and keep it.
        path.write_text(json.dumps({"models": [spec]}, indent=2) + "\n")
        # One outcome per repo, always — a rewrite is still a single event, not
        # a skip followed by an add. Two outcomes for one repo made `checked`
        # and `len(outcomes)` disagree and printed both "skipped" and "added"
        # for the same model.
        outcomes.append(Outcome(d.repo, True, rewrite_reason))
        added += 1

    return WatchReport(
        checked=len(found),
        outcomes=tuple(outcomes),
        notes=("entries are unverified, licences unread, and carry no parameter count",),
    )


def install_schedule(*, interval_hours: int = 24) -> tuple[str, str]:
    """The scheduler fragment for this platform, and where it goes.

    Returned rather than written: installing a recurring background job that
    reaches the network is the user's decision, and a tool that silently
    scheduled itself would be doing exactly what this project promises not to.
    """
    import sys

    exe = f"{sys.executable} -m onpar.cli watch --network"
    if sys.platform == "darwin":
        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.onpar.watch</string>
  <key>ProgramArguments</key>
  <array>{"".join(f"<string>{a}</string>" for a in exe.split())}</array>
  <key>StartInterval</key><integer>{interval_hours * 3600}</integer>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>/tmp/onpar-watch.log</string>
  <key>StandardErrorPath</key><string>/tmp/onpar-watch.log</string>
</dict></plist>
"""
        return plist, "~/Library/LaunchAgents/com.onpar.watch.plist"

    timer = f"""# ~/.config/systemd/user/onpar-watch.service
[Unit]
Description=onpar model discovery

[Service]
Type=oneshot
ExecStart={exe}

# ~/.config/systemd/user/onpar-watch.timer
[Unit]
Description=Run onpar model discovery every {interval_hours}h

[Timer]
OnUnitActiveSec={interval_hours}h
# Catch up after the machine was asleep, but not all at once on every boot.
Persistent=true

[Install]
WantedBy=timers.target
"""
    return timer, "~/.config/systemd/user/onpar-watch.{service,timer}"


def demo() -> None:
    import tempfile

    index = json.dumps(
        [
            {
                "id": "acme/new-9b",
                "downloads": 90_000,
                "likes": 400,
                "pipeline_tag": "text-generation",
            },
            {
                "id": "acme/broken",
                "downloads": 80_000,
                "likes": 300,
                "pipeline_tag": "text-generation",
            },
        ]
    )
    good = json.dumps(
        {
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "hidden_size": 4096,
            "max_position_embeddings": 32768,
        }
    )

    def fetch(url: str) -> str:
        if "api" in url or "models?" in url:
            return index
        if "new-9b" in url:
            return good
        return json.dumps({"hidden_size": 4096})  # no layer count: unsizeable

    with tempfile.TemporaryDirectory() as d:
        dest = Path(d)
        r = run(fetch, dest=dest)

        added = [o.repo for o in r.added]
        assert added == ["acme/new-9b"], added
        # The unparseable one is skipped and *says why*, never invented.
        bad = next(o for o in r.outcomes if o.repo == "acme/broken")
        assert not bad.added and "layer count" in bad.reason, bad

        # `staged_name`, not a literal: the file is keyed on the whole repo, so
        # two orgs publishing the same short name stay two files.
        staged = dest / staged_name("acme/new-9b")
        assert staged.name.startswith("discovered-acme-new-9b-"), staged.name
        spec = json.loads(staged.read_text())["models"][0]
        assert spec["verified"] is False, "a robot must never mark an entry verified"
        assert spec["license_ok"] is False, "licence is a reading task, not a parsing task"
        assert "params_b" not in spec, "an invented parameter count looks right and is not"
        assert spec["kv_heads"] == 8 and spec["layers"] == 32

        # Idempotent: a second pass adds nothing.
        assert not run(fetch, dest=dest).added

        # Offline is a normal state, not an error — a laptop that was asleep
        # must not produce an alarm.
        def dead(url: str) -> str:
            raise OSError("no network")

        assert run(dead, dest=dest).offline

    frag, where = install_schedule()
    # The plist splits argv across <string> elements, so check the parts rather
    # than a contiguous command line.
    assert "onpar.cli" in frag and "watch" in frag and where

    print(f"watch: {r.checked} checked, {len(r.added)} added, cap {MAX_PER_RUN}")
    print(r.render())


if __name__ == "__main__":
    demo()
