"""`clickllm cache` — what proving N models left on your disk, and what to remove.

Proving is a sweep, not a download. You size six candidates, launch four of them,
and each one costs 30–45 GB that nobody is tracking. The engines fetch weights
themselves ([`launch`] deliberately does not), so the bytes land in **Hugging
Face's** cache — `~/.cache/huggingface/hub`, or wherever `HF_HOME` /
`HF_HUB_CACHE` point — and stay there. There is no budget, no eviction, and no
record of which of those repos is the one you are currently comparing against.

This module reads that cache and nothing else. It is deliberately *not* a second
store: a cache clickllm owned would be a cache nothing writes to, which is the
mistake [ADR-0010](../../docs/adr/0010-retire-the-weights-crate.md) records.

## The three rules that make deletion safe

**Nothing is removed without `--yes`.** [`plan_eviction`] computes;
[`apply_eviction`] refuses to act unless it is told twice, and `clickllm cache
evict` prints the whole list and the exact bytes freed first. A tool that
reclaimed 200 GB helpfully would eventually reclaim the wrong 200 GB.

**A pinned repo is never in a plan.** This is the point of the feature, not a
convenience: an eval sweep that evicts the incumbent you are measuring against
destroys the comparison, and the symptom is a re-download an hour later that
nobody connects to the sweep. If the budget cannot be met without touching a
pin, the plan says so and comes up short ([`Plan.short_by`]) rather than
quietly widening.

**Nothing outside the hub root is ever a target.** Every path is re-resolved
against the root and must still carry a `models--` / `datasets--` / `spaces--`
directory name before `rmtree` sees it. That check raises rather than skips: a
target outside the cache is a bug in this file, not a runtime condition.

## What "least recently used" can honestly mean here

The Hub records no access log. The best signal on disk is the newest of each
file's `atime` and `mtime`, which is what [`entries`] reports — and `atime` is a
calibration knob, not a truth: a `noatime` mount degrades it to "last written",
which for weights means "last downloaded". That is still the right ordering for
an eviction sweep, and it is why nothing here evicts without a human reading the
list.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Entry",
    "Plan",
    "Removal",
    "State",
    "Usage",
    "apply_eviction",
    "demo",
    "entries",
    "hub_dir",
    "load_state",
    "parse_bytes",
    "pin",
    "plan_eviction",
    "unpin",
    "usage",
]

GB = 1024**3

#: Hub directory prefixes, and what each one holds. A directory that does not
#: start with one of these is not ours to read and never ours to delete.
KINDS = {"models": "model", "datasets": "dataset", "spaces": "space"}

_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


def hub_dir() -> Path:
    """Where the engines actually put weights, honouring the Hub's own overrides.

    Resolution order is `huggingface_hub`'s: `HF_HUB_CACHE` wins outright,
    `HF_HOME` supplies the parent, and the default derives from
    `XDG_CACHE_HOME`. Read from the environment on every call rather than cached
    at import, because a test — and a user with two boxes' caches on one
    machine — changes it between calls.
    """
    if direct := os.environ.get("HF_HUB_CACHE"):
        return Path(direct)
    if home := os.environ.get("HF_HOME"):
        return Path(home) / "hub"
    base = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(base) / "huggingface" / "hub"


@dataclass(frozen=True, slots=True)
class Entry:
    """One repo in the hub cache, as it sits on disk."""

    repo: str
    kind: str
    path: Path
    bytes: int
    #: Unix seconds: the newest atime/mtime under the repo. See the module
    #: docstring — on a `noatime` mount this reads as "last downloaded".
    last_used: float

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.last_used) / 86400)

    def render(self, *, pinned: bool = False) -> str:
        mark = "PIN" if pinned else "   "
        return f"  {mark} {human(self.bytes):>9}  {self.age_days:>5.1f}d  {self.repo}"


def _repo_id(name: str) -> tuple[str, str]:
    """`models--mlx-community--Llama-3.1-8B-4bit` -> ('model', 'mlx-community/…').

    Split once for the kind, then the Hub's own flattening in reverse.
    ponytail: a repo name containing `--` round-trips wrong, exactly as it does
    in `huggingface_hub.scan_cache_dir`; matching their bug beats inventing a
    different one, and the fix is theirs to make in the directory layout.
    """
    prefix, _, rest = name.partition("--")
    return KINDS.get(prefix, ""), rest.replace("--", "/")


def _measure(root: Path) -> tuple[int, float]:
    """Bytes really on disk under `root`, and the newest access time seen.

    Snapshots are symlinks into `blobs/`, so following them would count every
    weight file twice — a 4 GB model reported as 8 GB is the first bug anyone
    writing this hits. Hardlinked blobs are counted once via (dev, ino).
    """
    total = 0
    newest = 0.0
    seen: set[tuple[int, int]] = set()
    for parent, _dirs, files in os.walk(root, followlinks=False):
        for name in files:
            try:
                st = os.lstat(os.path.join(parent, name))
            except OSError:
                continue  # a file that vanished mid-scan is not a scan failure
            if stat.S_ISLNK(st.st_mode):
                # A snapshot symlink is stamped once, when the download laid it
                # down, and never again — reading through it touches the blob.
                # Counting it would peg every repo's age to its first fetch.
                continue
            newest = max(newest, st.st_atime, st.st_mtime)
            if st.st_nlink > 1:
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
            total += st.st_size
    if not newest:
        try:
            newest = root.stat().st_mtime  # an empty repo dir still has an age
        except OSError:
            newest = 0.0
    return total, newest


def entries(root: Path | None = None) -> list[Entry]:
    """Every repo in the hub cache, largest first. A missing cache is empty."""
    root = root or hub_dir()
    if not root.is_dir():
        return []
    out: list[Entry] = []
    for d in sorted(root.iterdir()):
        kind, repo = _repo_id(d.name)
        if not kind or not repo or not d.is_dir() or d.is_symlink():
            continue
        size, used = _measure(d)
        out.append(Entry(repo=repo, kind=kind, path=d, bytes=size, last_used=used))
    return sorted(out, key=lambda e: e.bytes, reverse=True)


@dataclass(frozen=True, slots=True)
class Usage:
    """What the cache costs right now, against the budget if one is set."""

    root: Path
    entries: tuple[Entry, ...]
    free_bytes: int
    budget_bytes: int | None = None
    pinned: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return sum(e.bytes for e in self.entries)

    @property
    def over_by(self) -> int:
        """Bytes above budget, or 0. No budget set means never over."""
        if self.budget_bytes is None:
            return 0
        return max(0, self.total_bytes - self.budget_bytes)

    def render(self) -> str:
        if not self.entries:
            return f"\n  nothing cached under {self.root}\n"
        pins = {p.casefold() for p in self.pinned}
        lines = [f"\n  {self.root}\n"]
        lines += [e.render(pinned=e.repo.casefold() in pins) for e in self.entries]
        lines.append("")
        summary = (
            f"  {human(self.total_bytes)} in {len(self.entries)} repos"
            f" · {human(self.free_bytes)} free on disk"
        )
        if self.budget_bytes is not None:
            summary += f" · budget {human(self.budget_bytes)}"
        lines.append(summary)
        if self.over_by:
            lines.append(
                f"  {human(self.over_by)} over budget — "
                f"clickllm cache evict --budget {human(self.budget_bytes or 0).replace(' ', '')}"
            )
        lines.append("")
        return "\n".join(lines)


def usage(
    root: Path | None = None,
    *,
    budget_bytes: int | None = None,
    pinned: tuple[str, ...] = (),
) -> Usage:
    """Size, free space and budget for the hub cache."""
    root = root or hub_dir()
    try:
        free = shutil.disk_usage(root if root.exists() else root.anchor or "/").free
    except OSError:
        free = 0  # an unreadable mount is not a reason to refuse a listing
    return Usage(
        root=root,
        entries=tuple(entries(root)),
        free_bytes=free,
        budget_bytes=budget_bytes,
        pinned=pinned,
    )


@dataclass(frozen=True, slots=True)
class Plan:
    """What eviction would remove, and what it deliberately would not."""

    evict: tuple[Entry, ...]
    kept_pins: tuple[Entry, ...]
    budget_bytes: int
    before_bytes: int

    @property
    def freed_bytes(self) -> int:
        return sum(e.bytes for e in self.evict)

    @property
    def after_bytes(self) -> int:
        return self.before_bytes - self.freed_bytes

    @property
    def short_by(self) -> int:
        """Bytes still over budget once everything evictable is gone.

        Non-zero means the pins alone exceed the budget. The plan stops there —
        widening it would delete the model the sweep is measuring against.
        """
        return max(0, self.after_bytes - self.budget_bytes)

    def render(self) -> str:
        if not self.evict:
            head = (
                f"\n  nothing to evict — {human(self.before_bytes)}"
                f" is within {human(self.budget_bytes)}\n"
            )
            if self.short_by:
                head = (
                    f"\n  cannot reach {human(self.budget_bytes)} without evicting a pinned repo.\n"
                    f"  {human(self.short_by)} over, and every remaining repo is pinned.\n"
                )
            return head
        lines = [f"\n  would delete {len(self.evict)} repos, freeing {human(self.freed_bytes)}:\n"]
        lines += [e.render() for e in self.evict]
        lines.append("")
        lines.append(
            f"  {human(self.before_bytes)} -> {human(self.after_bytes)}"
            f" (budget {human(self.budget_bytes)})"
        )
        if self.kept_pins:
            names = ", ".join(e.repo for e in self.kept_pins)
            lines.append(f"  kept, pinned: {names}")
        if self.short_by:
            lines.append(f"  still {human(self.short_by)} over — the rest is pinned")
        lines.append("")
        return "\n".join(lines)


def plan_eviction(
    items: list[Entry] | tuple[Entry, ...],
    budget_bytes: int,
    pinned: tuple[str, ...] = (),
) -> Plan:
    """Least-recently-used repos to remove to get under `budget_bytes`.

    Pinned repos are excluded from the candidate list entirely, so they cannot
    be reached even when they are the oldest thing here and the budget is
    otherwise unreachable — that case reports as [`Plan.short_by`].

    Raises:
        ValueError: if the budget is negative, with the offending value.
    """
    if budget_bytes < 0:
        raise ValueError(f"negative cache budget: {budget_bytes}")
    pins = {p.casefold() for p in pinned}
    kept = tuple(e for e in items if e.repo.casefold() in pins)
    before = sum(e.bytes for e in items)

    remaining = before
    victims: list[Entry] = []
    for e in sorted((e for e in items if e.repo.casefold() not in pins), key=lambda e: e.last_used):
        if remaining <= budget_bytes:
            break
        victims.append(e)
        remaining -= e.bytes
    return Plan(
        evict=tuple(victims), kept_pins=kept, budget_bytes=budget_bytes, before_bytes=before
    )


@dataclass(frozen=True, slots=True)
class Removal:
    """What was actually deleted. `confirmed` false means nothing was."""

    removed: tuple[str, ...] = ()
    freed_bytes: int = 0
    confirmed: bool = False
    failed: tuple[tuple[str, str], ...] = ()

    def render(self) -> str:
        if not self.confirmed:
            return "  nothing deleted — re-run with --yes\n"
        lines = [f"  deleted {len(self.removed)} repos, freed {human(self.freed_bytes)}"]
        lines += [f"  could not remove {repo}: {why}" for repo, why in self.failed]
        return "\n".join(lines) + "\n"


def _guard(path: Path, root: Path) -> None:
    """Refuse any target that is not a repo directory directly under `root`.

    Raises rather than skipping. A path outside the cache reaching this point is
    a defect in this module, and the one response that cannot lose data is to
    stop.
    """
    resolved = path.resolve()
    if resolved.parent != root.resolve():
        raise ValueError(f"refusing to delete outside the hub cache: {path}")
    if not _repo_id(resolved.name)[0]:
        raise ValueError(f"refusing to delete a non-repo directory: {path}")


def apply_eviction(plan: Plan, *, confirm: bool = False, root: Path | None = None) -> Removal:
    """Delete the planned repos. Without `confirm=True`, deletes nothing.

    The default is the safe one on purpose: every caller has to say the word,
    and `clickllm cache evict` only says it when the user passed `--yes`.
    """
    if not confirm:
        return Removal()
    root = root or hub_dir()
    removed: list[str] = []
    failed: list[tuple[str, str]] = []
    freed = 0
    for e in plan.evict:
        _guard(e.path, root)
        try:
            shutil.rmtree(e.path)
        except OSError as err:
            failed.append((e.repo, str(err)))
            continue
        removed.append(e.repo)
        freed += e.bytes
    return Removal(tuple(removed), freed, True, tuple(failed))


# --------------------------------------------------------------------------- #
# Pins and budget — clickllm's state, never HF's cache
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class State:
    """Pins and the configured budget. Lives in clickllm's state dir.

    Not in the hub cache: writing bookkeeping into a directory another tool owns
    is how you end up with a file `huggingface_hub` deletes on its next scan.
    """

    pinned: tuple[str, ...] = ()
    budget_bytes: int | None = None


def state_path(path: Path | None = None) -> Path:
    from .launch import state_dir

    return path or state_dir() / "cache.json"


def _root_key(root: Path | None = None) -> str:
    """The cache root a pin belongs to, resolved and absolute.

    Pins used to be stored flat, so a pin made against one cache root applied to
    every other — including a scratch `HF_HOME` used for a test, which then wrote
    into the developer's real state and protected a repo they never pinned. A pin
    is a statement about one cache, so it is filed under that cache.
    """
    return str((root or hub_dir()).expanduser().resolve())


def _all_states(path: Path | None = None) -> dict:
    """Every root's state, migrating the old flat file if that is what is there.

    The format changed when pins became per-root. A file written before that has
    `pinned` at the top level; reading it as a root-keyed map would find nothing
    and silently drop protection somebody deliberately set — the one outcome a
    pin exists to prevent. So the old shape is recognised and attributed to the
    default cache root, which is the only cache it could have meant.
    """
    try:
        raw = json.loads(state_path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    if "pinned" in raw or "budget_bytes" in raw:
        return {_root_key(): raw}
    return raw


def load_state(path: Path | None = None, root: Path | None = None) -> State:
    """The saved pins and budget for one cache root. Corrupt or missing is empty."""
    raw = _all_states(path).get(_root_key(root))
    if not isinstance(raw, dict):
        return State()
    pins = raw.get("pinned")
    budget = raw.get("budget_bytes")
    return State(
        pinned=tuple(p for p in pins if isinstance(p, str)) if isinstance(pins, list) else (),
        budget_bytes=budget if isinstance(budget, int) and budget >= 0 else None,
    )


def save_state(state: State, path: Path | None = None, root: Path | None = None) -> None:
    """Write pins and budget, beside the old file and renamed.

    Raises:
        OSError: unlike `launch`'s resolution cache, a lost pin is not a
            recoverable optimisation — it is the difference between a sweep that
            keeps your incumbent and one that deletes it, so a failure is loud.
    """
    p = state_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = _all_states(path)
    body[_root_key(root)] = {
        "pinned": sorted(state.pinned),
        "budget_bytes": state.budget_bytes,
    }
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n")
    tmp.replace(p)


def pin(repo: str, path: Path | None = None, root: Path | None = None) -> State:
    """Protect `repo` from every future eviction plan."""
    if not repo.strip():
        raise ValueError(f"not a repo id: {repo!r}")
    st = load_state(path, root)
    if any(p.casefold() == repo.casefold() for p in st.pinned):
        return st
    st = State(pinned=(*st.pinned, repo), budget_bytes=st.budget_bytes)
    save_state(st, path, root)
    return st


def unpin(repo: str, path: Path | None = None, root: Path | None = None) -> State:
    """Drop a pin.

    Raises:
        ValueError: if it was not pinned — carrying the value, because the
            common cause is a typo and a silent no-op would leave someone
            believing a repo is evictable when it is still protected.
    """
    st = load_state(path, root)
    kept = tuple(p for p in st.pinned if p.casefold() != repo.casefold())
    if len(kept) == len(st.pinned):
        raise ValueError(f"not pinned: {repo}")
    st = State(pinned=kept, budget_bytes=st.budget_bytes)
    save_state(st, path, root)
    return st


# --------------------------------------------------------------------------- #
# Sizes
# --------------------------------------------------------------------------- #


def parse_bytes(text: str) -> int:
    """`'200G'` -> bytes. Accepts K/M/G/T, with or without a trailing `B`.

    Raises:
        ValueError: naming the offending string, since the whole point of a
            budget is that a misread one deletes the wrong amount.
    """
    t = text.strip().lower()
    if t.endswith("b"):
        t = t[:-1]
    unit = _UNITS.get(t[-1:], 1)
    number = t[:-1] if unit != 1 else t
    try:
        value = float(number)
    except ValueError:
        raise ValueError(f"not a size: {text!r} — try 200G, 1.5T, 500M") from None
    if value < 0:
        raise ValueError(f"negative size: {text!r}")
    return int(value * unit)


def human(n: int) -> str:
    """Bytes as the unit a person would have said."""
    for unit, scale in (("TB", 1024**4), ("GB", GB), ("MB", 1024**2), ("KB", 1024)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "hub"
        now = time.time()
        # Three repos: the incumbent is both the oldest and the largest, which
        # is what makes the pin worth testing.
        for name, size, age_days in (
            ("models--acme--incumbent-70b", 8000, 40),
            ("models--acme--candidate-8b", 4000, 1),
            ("datasets--acme--evalset", 1000, 20),
        ):
            snap = root / name / "snapshots" / "abc"
            snap.mkdir(parents=True)
            blob = root / name / "blobs" / "deadbeef"
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(b"\0" * size)
            # The snapshot is a symlink into blobs, as the Hub really lays it out.
            (snap / "model.safetensors").symlink_to(blob)
            when = now - age_days * 86400
            os.utime(blob, (when, when))

        found = entries(root)
        assert [e.repo for e in found] == [
            "acme/incumbent-70b",
            "acme/candidate-8b",
            "acme/evalset",
        ], found
        # The symlinked snapshot must not double-count the blob.
        assert found[0].bytes == 8000, found[0].bytes
        assert found[2].kind == "dataset"

        # LRU: with room for one repo, the two oldest go, newest survives.
        plan = plan_eviction(found, 4000)
        assert [e.repo for e in plan.evict] == ["acme/incumbent-70b", "acme/evalset"], plan.evict
        assert plan.after_bytes == 4000

        # Pinned: the incumbent is the oldest and the largest, and is still safe.
        pinned = plan_eviction(found, 4000, pinned=("acme/incumbent-70b",))
        assert "acme/incumbent-70b" not in [e.repo for e in pinned.evict]
        # Everything evictable is gone and it is still 8000 against a 4000
        # budget: the plan reports the shortfall instead of widening.
        assert pinned.short_by == 4000, pinned.short_by

        # Nothing is deleted without saying so twice.
        assert apply_eviction(plan, root=root).removed == ()
        assert (root / "models--acme--incumbent-70b").exists()

        done = apply_eviction(plan, confirm=True, root=root)
        assert done.removed == ("acme/incumbent-70b", "acme/evalset"), done
        assert not (root / "models--acme--incumbent-70b").exists()
        assert (root / "models--acme--candidate-8b").exists()

        # Nothing outside the cache is ever a target.
        outside = Entry("evil", "model", Path(d) / "models--evil--x", 1, now)
        try:
            _guard(outside.path, root)
        except ValueError as e:
            assert "outside the hub cache" in str(e)
        else:  # pragma: no cover - the guard failing is the bug this catches
            raise AssertionError("the guard let a path outside the cache through")

        state = Path(d) / "cache.json"
        pin("acme/candidate-8b", state)
        assert load_state(state).pinned == ("acme/candidate-8b",)
        unpin("ACME/candidate-8b", state)
        assert load_state(state).pinned == ()

    assert parse_bytes("200G") == 200 * GB
    assert parse_bytes("1.5tb") == int(1.5 * 1024**4)
    assert parse_bytes("4096") == 4096

    print(f"cache: {len(found)} repos, {human(sum(e.bytes for e in found))}")
    print(plan.render())


if __name__ == "__main__":
    demo()
