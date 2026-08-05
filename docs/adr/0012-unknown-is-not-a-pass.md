# ADR-0012 — "Could not check" is not "checked and fine"

**Status:** accepted · **Date:** 2026-08-05
**Related:** [ADR-0011](0011-validate-at-the-solver.md)

## Context

The first full read of this codebase found the same defect in four unrelated
modules, written at different times, by different reasoning, none aware of the
others:

| where | the absence | how it rendered |
|---|---|---|
| `prove/gate.py` ([#84](https://github.com/dshakes/clickllm/issues/84)) | every scored cluster pinned, so nothing left to judge | **ADVANCE** — *"0% of traffic is proven … is supported"* |
| `prove/receipt.py` ([#83](https://github.com/dshakes/clickllm/issues/83)) | the `digest` key missing | tamper check skipped, forged receipt **accepted** |
| `plan.py` ([#97](https://github.com/dshakes/clickllm/issues/97)) | `bandwidth_gbps` unknown, so no throughput estimate | `meets_requirements = **True**` for an impossible budget |
| `watch.py` ([#113](https://github.com/dshakes/clickllm/issues/113)) | the index malformed | `offline=True` — *"nothing new upstream"* |

Four modules. One shape: **a state meaning *we do not know* was represented by
the same value as *we checked, and it is fine*.**

Every instance arrived the same way — an empty collection, a falsy field, or a
bare `return ()` — and in every instance the surrounding code was written by
someone who had thought carefully about the *known* cases. None was careless.
The failure is representational, not attentional: when "unknown" and "fine"
share a representation, the compiler, the tests and the reviewer all see one
value, and the difference exists only in someone's head.

It is also the failure this product can least afford. clickllm's entire claim is
that it hands you evidence rather than assurance — `?` rather than a fabricated
score, a confidence interval on every number, judge agreement disclosed. A
silent "fine" where the answer is "unknown" is the precise inversion of that
promise, and the four instances above landed on the cutover gate, the proof
artifact, the sizing verdict, and the drift watcher — the four things a user is
supposed to be able to trust without reading the source.

## Decision

**A function that can fail to determine something must not return the same value
it returns when the answer is favourable.**

Concretely, in order of preference:

1. **Raise.** If the caller cannot proceed meaningfully, refuse. `solve()` raises
   on a non-positive context (ADR-0011); `Receipt.from_json` raises when the
   digest is absent; `discover()` raises `Unreachable` rather than returning
   `[]` for a failed fetch.
2. **Return a distinct value.** Where refusing is too strong, the unknown state
   gets its own representation the caller must handle — `None` where the
   favourable answer is a number, a named enum member, or a separate field.
   `Interval.total == 0` renders `?`, and that is the pattern to copy.
3. **Carry the reason.** Where the unknown is reported to a human, say what could
   not be checked and why. `_budget_warnings` now emits *"could not be checked:
   this hardware's memory bandwidth is unknown … Not a pass — unmeasured"*
   rather than falling silent.

**Never:** an empty collection, a falsy default, or a silent `return` as the
signal for "could not determine", when that same value is what success looks
like.

### The test that goes with it

Any check with an *unknown* branch gets a case asserting the unknown state is
distinguishable from the favourable one. Not that it is handled well — that it
is not the same value. Each of the four fixes above carries exactly that test,
and each was verified by a negative control that restores the old collapse.

## Consequences

**What gets better.** The class closes at the point of representation rather than
per-module. A reviewer no longer has to notice the absent case; the type does.

**What this costs.** More refusals reach callers, and some of those callers will
need to handle a state they previously received as a cheerful default. That is
the point, and it is the trade CLAUDE.md already makes elsewhere: a refusal is a
bug someone fixes, a fabricated pass is a decision someone makes on bad evidence.

**What it does not cover.** This is a rule about representation, not about
coverage. It does nothing for a check that was never written — `demo()`
sensitivity is the ratchet for that
([#135](https://github.com/dshakes/clickllm/pull/135)), and nothing here replaces
running the product on real traffic.

## Alternatives considered

**Fix the four and move on.** Rejected for the reason ADR-0011 was written: four
independent instances is a property of how the code is shaped, not four
coincidences, and the fifth is already being written somewhere.

**A lint.** Attractive and insufficient — "returns `[]` on failure" is not
syntactically distinguishable from "returns `[]` legitimately", which is exactly
what made these four invisible. The rule has to be applied by the person
designing the signature.

**A `Result`/`Either` type throughout.** Rejected as disproportionate. Python's
exceptions and `None` already express this; the failure was not a missing
mechanism but an unstated rule, and a large refactor would trade a documented
convention for a novel one nobody here has used.
