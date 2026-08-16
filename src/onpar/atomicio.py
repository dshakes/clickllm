"""Write a file so a concurrent writer cannot corrupt it or lose an update.

Three modules had grown their own copy of "write beside it and rename", and all
three shared the same two defects:

* **A fixed scratch name.** `path.with_suffix(".json.tmp")` is the same path in
  every process, so two writers write the *same* temp file and one renames what
  the other is still writing. The published file is then a splice of both.
* **An unlocked read-modify-write.** Both read, both add their key, both write
  the whole document — last writer wins. Measured on the weights cache: forty
  concurrent resolutions kept seven.

The launch cache was fixed first, in isolation, and the same bug stayed in
`cache.py` and `catalog_update.py` for another release. That is the pattern this
module exists to end: one implementation, so a fix reaches every caller.

Zero dependencies — `onpar fit` must keep working under `uvx` with nothing
installed. `fcntl` is POSIX-only and its absence degrades to unique-temp writes
rather than failing, because a lock is not worth refusing to save state over.
"""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["atomic_write", "atomic_write_json", "update_json"]


def _unique_tmp(path: Path) -> Path:
    """A scratch path no other process will pick.

    The pid alone is not quite enough — one process can write the same target
    twice concurrently from two threads — so the object id of the path
    disambiguates within a process as well.
    """
    return path.with_name(f".{path.name}.{os.getpid()}.{id(path):x}.tmp")


@contextlib.contextmanager
def _locked(path: Path):
    """Hold an exclusive lock for the whole read-modify-write, where possible."""
    handle = None
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        yield
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = (path.parent / f"{path.name}.lock").open("w")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError as e:
        # Degrading to an unlocked write is deliberate — a lock is not worth
        # refusing to save state over — but doing it *silently* is not. The
        # module header justifies the fallback for a missing `fcntl`, which is
        # a platform fact; this branch also catches `flock` failing at runtime
        # on a filesystem that does not implement it (NFS raising EOPNOTSUPP,
        # ENOLCK), where `update_json`'s promise that "two processes calling
        # this concurrently each see the other's result" quietly stops holding
        # and the lost-update bug this module exists to end comes back.
        #
        # `warnings.warn` rather than a return value: the default filter
        # already shows it once per call site, and changing the signature
        # would make every caller handle a case none of them can fix.
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        handle = None
        # Suppressed because a warning must not be able to prevent the thing it
        # is warning about. Under `PYTHONWARNINGS=error` — or any test harness
        # that turns warnings into errors — this call *raises*, which propagates
        # out of the context manager and skips the write entirely. That inverts
        # the module's stated design ("a lock is not worth refusing to save
        # state over") into "an unlockable filesystem cannot save state at all",
        # measured: the file did not exist afterwards.
        with contextlib.suppress(RuntimeWarning):
            warnings.warn(
                f"could not lock {path}: {e}. Writing unlocked — a concurrent "
                "writer's update may be lost.",
                RuntimeWarning,
                stacklevel=3,
            )
    try:
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


def _fsync_dir(directory: Path) -> None:
    """Make the rename durable, not just the bytes it points at.

    On filesystems that journal the directory entry separately from file data,
    a crash can otherwise land the new data and lose the rename, or the other
    way round. Best-effort: a platform that will not open a directory read-only
    (Windows) has no equivalent to offer, and there is nothing better to do
    than the write we already made.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, text: str) -> None:
    """Replace `path` with `text`, or leave it exactly as it was.

    A reader never sees a partial file: the content lands in a private scratch
    file first and `replace` is atomic within a filesystem.

    The `replace` is what makes it atomic; the `fsync` before it is what makes
    it *true after a crash*. Without it the rename can reach the disk while the
    data blocks it points at are still dirty in the page cache, and the file
    reads back empty — neither the old content nor the new one, which is the
    one outcome this function promises cannot happen.

    Ceiling: on macOS `fsync` flushes to the device but does not force the
    device's own write cache; `F_FULLFSYNC` does, at a large cost per call. The
    callers here write caches and a model catalog a few times per run, so the
    upgrade is affordable if a corrupted catalog is ever seen after a hard
    power loss on Apple hardware.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    try:
        with tmp.open("w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
        _fsync_dir(path.parent)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def atomic_write_json(path: Path, data: Any, *, indent: int = 2, sort_keys: bool = False) -> None:
    """`atomic_write` for a JSON document, with the trailing newline files want."""
    atomic_write(path, json.dumps(data, indent=indent, sort_keys=sort_keys) + "\n")


def update_json(
    path: Path,
    mutate: Callable[[Any], Any],
    *,
    default: Any = None,
    indent: int = 2,
    sort_keys: bool = False,
) -> bool:
    """Read, change, and write back under a lock. Returns whether it was written.

    `mutate` receives the current document (or `default` when the file is absent
    or unreadable) and returns the one to store. It runs INSIDE the lock, so two
    processes calling this concurrently each see the other's result rather than
    overwriting it.

    Failure is silent by design at the call sites that use this for caches: the
    answer is already in hand and a read-only home should not stop a server from
    starting. Callers that need to know get the boolean.
    """
    with _locked(path):
        try:
            current = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            current = default
        try:
            atomic_write_json(path, mutate(current), indent=indent, sort_keys=sort_keys)
        except OSError:
            return False
    return True


def demo() -> None:
    import tempfile
    from concurrent.futures import ThreadPoolExecutor

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.json"

        atomic_write_json(p, {"a": 1})
        assert json.loads(p.read_text()) == {"a": 1}

        # No scratch file survives a successful write.
        assert list(Path(d).glob("*.tmp")) == []

        # Concurrent updates all survive — the failure this module exists for.
        keys = [f"k{i}" for i in range(60)]

        def add(k: str) -> None:
            update_json(p, lambda cur, k=k: {**(cur or {}), k: k}, default={})

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(add, keys))
        got = json.loads(p.read_text())
        missing = sorted(set(keys) - set(got))
        assert not missing, f"{len(missing)} updates lost: {missing[:5]}"

        # A corrupt file reads as the default rather than raising.
        p.write_text("{not json")
        assert update_json(p, lambda cur: {"recovered": True}, default={})
        assert json.loads(p.read_text()) == {"recovered": True}

    print("ok")


if __name__ == "__main__":
    demo()
