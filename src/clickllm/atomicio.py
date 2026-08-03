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

Zero dependencies — `clickllm fit` must keep working under `uvx` with nothing
installed. `fcntl` is POSIX-only and its absence degrades to unique-temp writes
rather than failing, because a lock is not worth refusing to save state over.
"""

from __future__ import annotations

import contextlib
import json
import os
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
    except OSError:
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()
        handle = None
    try:
        yield
    finally:
        if handle is not None:
            with contextlib.suppress(OSError):
                handle.close()


def atomic_write(path: Path, text: str) -> None:
    """Replace `path` with `text`, or leave it exactly as it was.

    A reader never sees a partial file: the content lands in a private scratch
    file first and `replace` is atomic within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _unique_tmp(path)
    try:
        tmp.write_text(text)
        tmp.replace(path)
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
