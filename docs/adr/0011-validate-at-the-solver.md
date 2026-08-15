# ADR-0011 — Validate at the solver, not at the surface

**Status:** accepted · **Date:** 2026-08-04

## Context

The audit found the same defect at two entry points, days apart, in modules that
had never been read end to end:

| entry point | validated? | consequence |
|---|---|---|
| `cli.py` | yes, in argument parsing | correct |
| `mcp.py` | **no** ([#107](https://github.com/dshakes/onpar/issues/107)) | a model that does not fit reported FEASIBLE, with flattering headroom |
| `sdk.py` `fit()` | yes, hand-written again | correct |
| `sdk.py` `explain()` | **no** ([#108](https://github.com/dshakes/onpar/issues/108)) | same model, same lie, different door |

Three doors into one calculation. Two of them guarded, and the two that were not
produce a *wrong sizing verdict* rather than a crash — a non-positive `context`
or `concurrency` makes the KV term vanish, so the footprint shrinks and the
verdict flips to FEASIBLE.

The pattern is not that anyone was careless. It is that **the guard lived at the
surface, so every new entry point had to re-earn it**, and `sdk.fit()` shows what
that costs even when someone remembers: the bounds are written out a second time,
in a second place, where they can drift from the first.

This is the repo's most expensive recurring shape, recorded twice already in
`CLAUDE.md`: a fact duplicated across files, corrected only where it was noticed.
`solve()` had three copies of one contract and no owner.

## Decision

**Input bounds are enforced in `fit.solve()`, and nowhere else is authoritative.**

`context >= 1` and `concurrency >= 1` are checked at the top of `solve()`, which
raises `ValueError` naming the offending value — matching the house convention
that the CLI catches it and returns a nonzero exit with a sentence.

Surfaces may still validate *earlier* for a better message, and the CLI does, so
a user typing `--concurrency 0` is told at the point of the mistake. What changes
is that no surface is load-bearing: forgetting the check is now a worse error
message, not a wrong answer.

The rule generalises beyond these two bounds. **A constraint that protects a
calculation belongs to that calculation.** If a fourth entry point is added — a
web console, a gRPC service, another agent tool — it inherits the guard by
construction rather than by review.

## Consequences

**What gets better.** The FEASIBLE-on-a-model-that-does-not-fit class is closed
at the source rather than per-door. `sdk.explain()`, every MCP tool, and any
future caller are covered without touching them.

**What this costs.** `solve()` now raises where it previously returned a `Fit`
for any input. That is a behaviour change for library callers who passed
non-positive values and read the result — but the result they were reading was
wrong, so there is no correct caller to break.

`max_context()` and `max_concurrency()` clamp rather than raise, deliberately:
they answer "how much fits", where zero is a coherent question with a real
answer, not an invalid input.

**What this does not fix.** Argument *typing* — `mcp.py` skipping the coercion
`cli.py` performs (#107 finding 2) — is a separate defect with the same shape,
and this ADR does not address it. The same reasoning applies and the fix belongs
in the same place; it is not folded in here because it needs its own tests.

## Alternatives considered

**Fix each surface.** Rejected: it is what produced the current state. Two
surfaces were fixed this way already and the third was written without the guard.

**A shared `validate_inputs()` helper the surfaces call.** Rejected for the same
reason — it is still opt-in, and a helper nobody calls is a guard that does not
exist. The distinction that matters is not "shared code" but "unavoidable code".

**Clamp instead of raise.** Rejected. Silently sizing `concurrency=0` as `1`
answers a question nobody asked, and the caller cannot tell it happened. The
house rule is `?` rather than a fabricated number; refusing is the same
principle.
