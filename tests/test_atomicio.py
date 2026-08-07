"""M?? — the write path: durability, and the lock that quietly is not there.

`atomicio.demo()` covers the visible contract — no partial file, no lost update,
a corrupt file recovering. These are the two properties you cannot see by
reading the file back afterwards, because both are about what survives a crash
that did not happen during the test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clickllm import atomicio
from clickllm.atomicio import atomic_write, atomic_write_json, update_json

# --- durability: the promise is about a crash, so the test is about ordering ---


def _record_calls(monkeypatch) -> list[str]:
    """Log fsync and rename in the order they happen."""
    calls: list[str] = []
    real_fsync, real_replace = os.fsync, Path.replace

    def fsync(fd):
        try:
            calls.append("fsync:dir" if os.fstat(fd).st_mode & 0o040000 else "fsync:file")
        except OSError:  # pragma: no cover — a closed fd would raise below anyway
            calls.append("fsync:?")
        return real_fsync(fd)

    def replace(self, target):
        calls.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(Path, "replace", replace)
    return calls


def test_the_data_is_flushed_before_the_rename_that_publishes_it(tmp_path, monkeypatch):
    """`replace` is atomic as a rename, which says nothing about whether the
    blocks it points at have reached the device.

    On a crash between the write and the kernel's own writeback, the rename can
    be durable while the data is not, and the file reads back empty — neither
    the old content nor the new one, the single outcome `atomic_write` promises
    cannot happen. Ordering is the whole property: an fsync *after* the rename
    would pass a test that only counted calls, and would not fix the bug.
    """
    calls = _record_calls(monkeypatch)
    atomic_write(tmp_path / "state.json", "hello")

    assert "fsync:file" in calls, "the scratch file was never fsynced"
    assert calls.index("fsync:file") < calls.index("replace"), (
        f"fsync must precede the rename that publishes the data, got {calls}"
    )


def test_the_rename_itself_is_made_durable(tmp_path, monkeypatch):
    """The other half, and the reason one fsync is not enough: on a filesystem
    that journals the directory entry separately from file data, the data can
    land while the rename is lost."""
    calls = _record_calls(monkeypatch)
    atomic_write(tmp_path / "state.json", "hello")

    assert "fsync:dir" in calls, f"the parent directory was never fsynced, got {calls}"
    assert calls.index("replace") < calls.index("fsync:dir"), (
        f"the directory fsync must follow the rename it is making durable, got {calls}"
    )


def test_a_directory_that_cannot_be_fsynced_still_writes(tmp_path, monkeypatch):
    """Best-effort by design. Windows will not open a directory read-only, and
    a write that refused over it would be worse than one that is merely less
    durable than intended."""
    monkeypatch.setattr(os, "open", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    atomic_write(tmp_path / "state.json", "hello")
    assert (tmp_path / "state.json").read_text() == "hello"


def test_the_content_is_still_exactly_what_was_asked_for(tmp_path):
    # The negative control for the two ordering tests above: a rewrite that
    # fsynced in the right order but corrupted the bytes would pass both.
    p = tmp_path / "state.json"
    atomic_write_json(p, {"a": 1, "b": [2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}
    assert p.read_text().endswith("\n")


# --- the lock that is not held, and used to say nothing about it ----------------


def _flock_fails(monkeypatch, err=None):
    """What an NFS mount without lock support does: EOPNOTSUPP from `flock`."""
    import fcntl

    err = err or OSError(45, "Operation not supported")

    def boom(fd, op):
        raise err

    monkeypatch.setattr(fcntl, "flock", boom)


def test_an_unlockable_path_says_so_rather_than_pretending(tmp_path, monkeypatch):
    """The fallback to an unlocked write is deliberate — a lock is not worth
    refusing to save state over — but it used to be indistinguishable from
    success.

    The module header justifies degrading when `fcntl` is *absent*, a platform
    fact. This branch also catches `flock` failing at runtime on a filesystem
    that does not implement it (NFS: EOPNOTSUPP, ENOLCK), where `update_json`'s
    promise that concurrent callers each see the other's result silently stops
    holding, and the "forty concurrent resolutions kept seven" bug the module
    exists to end is back with nothing to show for it.
    """
    _flock_fails(monkeypatch)
    p = tmp_path / "state.json"

    with pytest.warns(RuntimeWarning, match="could not lock"):
        assert update_json(p, lambda cur: {"a": 1}, default={})
    assert json.loads(p.read_text()) == {"a": 1}


def test_the_warning_names_the_path_and_the_reason(tmp_path, monkeypatch):
    # "Writing unlocked" with no path is not actionable when three modules use
    # this and only one of their state directories is on the bad filesystem.
    _flock_fails(monkeypatch)
    p = tmp_path / "weights.json"

    with pytest.warns(RuntimeWarning) as caught:
        update_json(p, lambda cur: {"a": 1}, default={})
    msg = str(caught[0].message)
    assert str(p) in msg
    assert "Operation not supported" in msg
    assert "may be lost" in msg


def test_a_lock_that_works_warns_about_nothing(tmp_path, recwarn):
    """The negative control: a guard that warned unconditionally would satisfy
    both tests above and make the warning worthless."""
    p = tmp_path / "state.json"
    assert update_json(p, lambda cur: {"a": 1}, default={})
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_an_unlocked_write_is_still_a_write_not_a_silent_no_op(tmp_path, monkeypatch):
    # Degraded must mean "less safe", never "did nothing" — the caller took the
    # warning as the whole cost.
    _flock_fails(monkeypatch)
    p = tmp_path / "state.json"
    with pytest.warns(RuntimeWarning):
        update_json(p, lambda cur: {**(cur or {}), "a": 1}, default={})
    with pytest.warns(RuntimeWarning):
        update_json(p, lambda cur: {**(cur or {}), "b": 2}, default={})
    assert json.loads(p.read_text()) == {"a": 1, "b": 2}


def test_the_module_self_check_still_passes():
    atomicio.demo()
