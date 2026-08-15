# ADR-0009 — What crosses the Rust↔Python seam

**Status:** accepted · **Date:** 2026-07-27
**Extends:** [ADR-0007](0007-tech-stack.md)

## Context

ADR-0007 split the product on a line: **Rust holds the datapath, Python holds the judgment.** Routing, metering, SSE framing and capture must be fast and must not stall. Clustering, grading, equivalence and fit must be readable and cheap to change. That split has held.

It stops being free the moment the two halves need the same data. Three milestones now do:

- **M6** writes an encrypted capture log from Rust.
- **M9** must read that log, score candidate against baseline, and gate a promotion.
- **M10** must re-read it whenever a new model release lands.

So the seam has to exist. The question this ADR settles is not *whether* but **how narrow**, because every function that crosses becomes a compatibility surface that has to survive both languages changing independently.

There is also a promise at stake. `pyproject.toml` says the package has zero runtime dependencies so that `uvx onpar fit` works on a laptop with nothing installed. A compiled extension is platform-specific by construction. Making the whole package depend on one would trade that promise for a feature most `fit` users never touch.

## Decision

### 1. Two things cross, and only two

| Crosses | Why |
|---|---|
| **The capture format** (`read_captures`) | It is length-framed XChaCha20-Poly1305. A Python reader would be a *second implementation of a security-critical format*, and two implementations of a format drift. |
| **The redactor** (`redact`) | Its patterns and its Luhn check are policy. Two copies of a policy means one is quietly out of date — and the stale one is the one that leaks. |

Plus `load_or_create_key`, which is the key half of the first.

### 2. Three things deliberately do not

| Stays put | Why |
|---|---|
| **Sizing and fit** | Pure Python, zero dependencies. Moving it behind a compiled wheel would cost the `uvx` promise and buy nothing — it is arithmetic, not a hot path. |
| **Routing control** | A running gateway holds its router in memory. Writing a file from Python would change nothing. Phase changes go over HTTP to the live process, which is also the only form with an audit trail. |
| **Grading and clustering** | Judgment. It changes weekly and belongs where it is cheap to change. |

The test for admitting anything new: *would it otherwise be implemented twice?* If not, it does not cross.

### 3. The extension is optional, and its absence is a sentence, not a stack trace

- Distribution `onpar-core`, import name `_onpar_core`, built by maturin as an **abi3-py311** wheel — one wheel per platform covers every supported interpreter.
- Installed directly: `pip install onpar-core`. `onpar` itself keeps zero runtime dependencies and declares **no `core` extra**. An extra would read better, but `onpar-core` is not on PyPI yet, and uv resolves every extra before it will run anything — so a forward reference makes the whole package unresolvable for people who never wanted the extension. The extra goes in the day the wheel is published, not before.
- `src/onpar/core.py` is the only module that imports it. When it is missing, `core.available()` is `False` and `core.why_unavailable()` returns one line naming what to install *and what still works without it*. Callers that genuinely cannot proceed call `core.require()` and get a `SeamError` with the same text.

### 4. Version skew is refused, not tolerated

A Python package paired with a mismatched extension imports fine and then misreads a binary format. `core.COMPATIBLE` lists the versions this package knows how to talk to; anything else reports as unavailable with a reinstall instruction.

### 5. CI runs the suite twice

The `test` job runs **without** the extension — that is the `uvx onpar fit` path, and it must stay green. The `seam` job builds the wheel, asserts `core.available()` is true, *then* runs the tests. Without that assertion the seam tests would skip themselves into a passing build.

## Consequences

**Good**

- One implementation of the capture format and one of the redaction policy. No drift possible.
- `uvx onpar fit` still works on a bare machine.
- The seam is four functions. It can be read in a minute and reviewed in five.

**Costs, honestly**

- **Two wheels to release.** `onpar` (pure, universal) and `onpar-core` (per-platform). The release process gets a matrix build it did not have.
- **A platform without a prebuilt wheel needs a Rust toolchain.** For anything outside the CI matrix, `pip install onpar-core` compiles from source.
- **The install is two commands, not one, until publication.** A dependency an extra cannot yet name is a dependency the user has to name themselves.
- **The extension pulls the gateway crate in**, and with it tokio and reqwest. The wheel is larger than the exposed surface suggests. Acceptable today; if it becomes a problem the capture store moves to its own crate.
- **Data crosses as dicts, not typed objects.** Deliberate — the Python side reshapes freely without a binding change per field — but it means a renamed field fails at runtime in Python rather than at compile time. The round-trip test in `tests/test_core.py` reads a log written by the real Rust writer, which is what catches that.

## Alternatives considered

**HTTP instead of FFI.** Python asks the gateway for captures over an admin endpoint. Rejected for the read path: it requires a *running* gateway, and M10 re-proves against historical logs long after that process exited. Kept for the write path — phase changes do go over HTTP, per decision 2.

**Reimplement the format in Python.** Rejected outright. Two implementations of an encrypted format is the failure this ADR exists to prevent.

**Make the whole package a maturin build.** Simplest packaging, one wheel. Rejected because it makes `onpar` platform-specific and breaks the zero-dependency promise for everyone, to serve the minority who read capture logs.

**Ship the extension as `onpar._core` inside the main package.** Rejected: two distributions writing into one package directory, where the outcome depends on install order.
