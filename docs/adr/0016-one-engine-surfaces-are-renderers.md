# ADR-0016 — One engine; surfaces are renderers

**Status:** accepted · 2026-08-12

## Context

clickllm has five user-facing surfaces: the CLI (24 commands), the MCP server
(9 read-only tools), the Python SDK, the local browser workbench, and the agent
skill. The README describes them as *"four faces of one implementation"* and
promises *"an agent gets the same answer you do"*.

That promise is not currently kept, and it is checkable:

```
ui  feasible row: model_id name quant weights_gb kv_gb total_gb slow …
mcp feasible row: id       quant           total_gb …

ui  envelope: runtime              warnings
mcp envelope: recommended_runtime
```

The *answers* agree — the same five models in the same order, the same engine
with the same reason. The **shapes** do not. `model_id` versus `id`, `runtime`
versus `recommended_runtime`, and four fields the agent surface never sees:
`weights_gb`, `kv_gb`, `name`, `slow`.

That is subtler than a wrong answer and worse to live with. A caller reading
`runtime` gets `None` from the surface that spells it `recommended_runtime`,
and `None` renders as *"no recommendation"* rather than as an error. An agent
cannot see the weights/KV split a human sees in the browser, from the same
solver, for the same question.

**Both surfaces already import `fit.py`.** They shared the helper and diverged
anyway, because each re-assembled the result itself. That is the fact this
decision turns on: sharing a helper is not sharing a contract.

The same shape recurs elsewhere. `session.py` is a complete multi-turn engine
with a computable relevance filter; `workbench.html` reimplements "understand
the user" with six client-side regexes and never imports it. `cli.cmd_build`
and `mcp._build` each hand-chain `_apply_text` → `_apply_hardware` →
`_apply_fields` → exactly one `step()`, and getting that wrong silently loses
the best question.

## Decision

**One engine. Every surface is a renderer over it.**

1. `src/clickllm/engine.py` holds the product operations — `fit`, `where`,
   `explain`, `advise`, `build`, `observe`, `distill`, `prove`, `receipt`,
   `guard`, `migrate`, `measure` — each returning an existing typed dataclass.
2. A surface may **format, transport and authenticate**. It may not decide,
   validate, coerce, or assemble a result. Validation stays at the solver
   (ADR-0011).
3. Engine results carry a versioned contract, `clickllm.engine/v1`, following
   the existing `clickllm.receipt/v1` convention — so a breaking change to a
   shape is visible rather than silent.
4. **The anti-drift mechanism is a conformance test, not the shared code.**
   `tests/test_surface_conformance.py` asks every surface the same question and
   asserts the same answer *and* the same schema. A new surface, or a new field
   on an old one, fails until it agrees.

## Consequences

**A new UX becomes a renderer rather than a fork.** This is the point: the ask
that prompted this was "one backend, multiple UXes", and the current
architecture would have grown a sixth divergent shape.

**Converging the two existing shapes is a breaking change, and is deliberately
not bundled with the test that found it.** `id` versus `model_id` appears in
three published contracts at once — the MCP tool schema, `clickllm fit --json`,
and the SDK's `to_dict()` — all of which became contracts when 1.0.0 shipped.
The conformance cases for the schema and the runtime spelling are therefore
marked `xfail(strict=True)`: they document the defect, they fail loudly the
moment someone fixes it without removing the marker, and the fix lands as its
own versioned change with a note for anyone depending on the old spelling.

**The public contract wins where they disagree.** When converging, prefer the
MCP spelling (`id`, `recommended_runtime`) because that is the surface external
agents build against; the workbench is ours to update. The richer fields
(`weights_gb`, `kv_gb`, `name`, `slow`) are *added* to the shared shape rather
than dropped — additive for agents, and an agent should see the weights/KV
split that a human sees.

**Golden approval tests were recorded before any code moved.** A conformance
test proves the surfaces agree with each other; only a golden recorded
beforehand proves they still agree with what users saw last week. Both use one
synthetic machine, because a test that reads the host is partly a test of the
host — a mistake this repo has already made twice.

## What would falsify this

A surface that legitimately needs a different *answer*, not merely a different
rendering — for example a UI that must degrade to a cheaper computation to stay
interactive. If that arrives, the conformance test is the wrong shape and this
decision needs revisiting, rather than the test being weakened case by case.

## Alternatives rejected

**Leave it and rely on discipline.** They already shared `fit.py` and diverged.
Discipline is what produced the current state.

**Converge inside this change.** It would bundle a breaking change to three
published shapes with the test that discovered it, giving anyone bisecting a
release two unrelated reasons for a break.

**One shape enforced by a schema library.** Considered and rejected for now:
`clickllm fit` must keep working under `uvx` with zero runtime dependencies,
which is a promise worth more than the validation. The conformance test buys
most of the same safety at no dependency cost.
