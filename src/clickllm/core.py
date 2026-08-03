"""The Python side of the Rust seam.

`clickllm fit` is pure Python with zero dependencies, and that is a promise worth
keeping: `uvx clickllm fit` has to work on a laptop with nothing installed. A
compiled extension would trade that for a platform-specific wheel.

So the extension is **optional**, and this module is the boundary. Everything
that needs Rust goes through here, and if it is missing the caller gets one clear
sentence naming what to install — not an `ImportError` from three frames down.

    from clickllm import core

    if core.available():
        rows = core.read_captures(path, key)
    else:
        print(core.why_unavailable())

What lives behind this seam and why is documented on the Rust side, in
`clickllm-py/src/lib.rs`. The short version: only the capture format and the
redactor cross, because those are the two things that would otherwise be
implemented twice and drift apart.
"""

from __future__ import annotations

from typing import Any

from . import __version__

__all__ = [
    "CaptureError",
    "SeamError",
    "available",
    "load_or_create_key",
    "read_captures",
    "redact",
    "require",
    "version",
    "why_unavailable",
]

#: Extension versions this package knows how to talk to. A mismatch is refused
#: rather than trusted: an extension that imports fine and then misreads a
#: binary format is a far worse failure than one that will not load.
#:
#: Derived, not typed. This was the literal "0.1.0" while the package shipped
#: 0.1.9, and it happened to work only because `tools/bump.py` did not maintain
#: the Rust workspace version either — two frozen numbers agreeing by accident.
#: Bumping either one alone would have made a correctly-paired install refuse to
#: load, and the failure would have surfaced as "reinstall both together" on a
#: machine where both were already current. They now move as one.
COMPATIBLE = (__version__,)

INSTALL_HINT = "pip install clickllm-core"


class SeamError(RuntimeError):
    """The compiled extension is absent or unusable."""


try:  # pragma: no cover - the branch taken depends on the install
    import _clickllm_core as _ext

    _import_error: str | None = None
except ImportError as exc:  # pragma: no cover
    _ext = None  # type: ignore[assignment]
    _import_error = str(exc)


def _skew() -> str | None:
    """Describe a version mismatch, or None if the pair is compatible."""
    if _ext is None:
        return None
    got = getattr(_ext, "__version__", "unknown")
    if got in COMPATIBLE:
        return None
    return (
        f"_clickllm_core {got} does not match this package "
        f"(expects one of {', '.join(COMPATIBLE)}). "
        f"Reinstall both together: {INSTALL_HINT}"
    )


def available() -> bool:
    """Whether the compiled extension is present and version-compatible."""
    return _ext is not None and _skew() is None


def why_unavailable() -> str | None:
    """One sentence explaining why, or None when it is available.

    Returned rather than raised so a caller can degrade — show the reason in a UI,
    skip an optional column — instead of being forced into a try/except.
    """
    if _ext is None:
        return (
            f"the compiled extension is not installed ({INSTALL_HINT}). "
            f"Reading captured traffic needs it; `clickllm fit` does not."
        )
    return _skew()


def require() -> Any:
    """Return the extension module, or raise with the reason.

    For the paths that genuinely cannot proceed — reading a capture log has no
    pure-Python fallback, and inventing one would mean a second implementation of
    an encrypted format.
    """
    reason = why_unavailable()
    if reason is not None:
        raise SeamError(reason)
    return _ext


def version() -> str | None:
    """Version of the loaded extension, or None if there is none."""
    return None if _ext is None else getattr(_ext, "__version__", None)


class CaptureError(SeamError):
    """A capture log could not be read — usually the wrong key."""


def read_captures(path: str, key: bytes) -> list[dict[str, Any]]:
    """Decrypt every capture in the log at `path`.

    An absent log is an empty list: "nothing captured yet" is the normal state of
    a fresh install, and making callers distinguish it from a real failure only
    invites a bare `except`.

    Raises:
        SeamError: the extension is missing or mismatched.
        CaptureError: wrong key, or a tampered file.
    """
    ext = require()
    try:
        return list(ext.read_captures(path, key))
    except ext.CaptureError as exc:  # narrow: wrong key, not a missing file
        raise CaptureError(str(exc)) from exc


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Redact `text` with the same rules the gateway applies on the write path.

    Use this for anything entering the corpus from outside the gateway — an
    imported log, a hand-written eval case. A second set of patterns in Python
    would drift from the Rust ones, and the stale copy is the one that leaks.

    Returns the cleaned text and a count of what was removed by kind. The counts
    matter as much as the string: a redactor that silently does nothing looks
    identical to one that works.

    Raises:
        SeamError: the extension is missing or mismatched.
        ValueError: the text is too large to scan in full, so it must be dropped.
    """
    return require().redact(text)


def load_or_create_key(path: str) -> bytes:
    """Load the capture key at `path`, generating an owner-only one if absent.

    Raises:
        SeamError: the extension is missing or mismatched.
    """
    return require().load_or_create_key(path)
