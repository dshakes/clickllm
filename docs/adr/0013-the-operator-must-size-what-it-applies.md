# ADR-0013 — The operator must size what it applies

**Status:** proposed · **Date:** 2026-08-05
**Related:** [ADR-0012](0012-unknown-is-not-a-pass.md) · [ADR-0002](0002-runtime-abstraction.md)

## Context

`clickllm.k8s.reconcile` opens by stating why it exists at all:

> Nothing, if it asked for an image and a command. The reason to run an operator
> is that **the answer changes when the cluster changes**: a workload sized for
> an H100 node that gets rescheduled onto an L4 needs different flags, and a
> Deployment cannot know that. **Reconciling against real node capacity is the
> feature**; the CRD is just how you ask for it.

That is not implemented. At `reconcile.py:358`:

```python
p = plan(hw, _requirements(spec))
```

`plan()` is `plan(hw, req, model=None, quant=None, …)`. **No model is passed**, so
`Plan.fit` is `None` and no sizing is performed. `select_node` reads every node's
accelerator memory and picks the largest — and that capacity is then used for
nothing. A Deployment for a 70B model and one for an 8B model are planned
identically on the same hardware.

Two attempts to fix the visible symptom failed because of this, and both would
have shipped a guard that could never fire:

1. Gate the controller's apply loop on the pass's own `Ready` condition. Dead:
   nothing ever reports `Ready=False` while returning objects.
2. Make `ok` consult `p.fit.feasible`. Dead: `p.fit` is always `None`.

Probed against a real node with change (2) applied — an impossible 1 ms
inter-token budget and a 1,000,000-token context both plan clean:

```
{}                     objects=1  Ready=True  Planned
{'context': 1000000}   objects=1  Ready=True  Planned
{'concurrency': 512}   objects=1  Ready=True  Planned
{'itlMs': 1}           objects=1  Ready=True  Planned
```

### Why it was left this way

Not an oversight in the loop — a genuine seam. The CRD names a **Hugging Face
repo** (`Qwen/Qwen3-32B`), and the catalogue is keyed by **id**
(`qwen3-32b`). `plan()` wants a `ModelSpec`. Nothing in the operator resolves one
to the other, and the honest reason is that the answer for a repo we do not
carry is a policy question, not a lookup.

## Decision

**The operator resolves the model, sizes against the selected node, and refuses
to report Ready for a workload it could not size.**

Three parts:

**1. Resolution.** `ModelSpec` already carries `repo` (`catalog.py:42`), so
`spec.model` is matched against catalogue entries by repo, case-folded. Exact
match only — no prefix or fuzzy matching, because sizing the wrong model is
worse than sizing none.

**2. Sizing.** The resolved `ModelSpec` and the chosen quantisation are passed to
`plan()`, so `Plan.fit` exists and `deployment_for` emits flags derived from the
node the workload was actually scheduled against. This is the feature the
docstring already claims.

**3. The unresolvable case is a stated unknown, never a pass.** A repo not in the
catalogue yields `Ready=False` with reason `ModelNotInCatalogue`, and the
Deployment is **not** applied. Per ADR-0012, "we could not size this" must not
share a representation with "this fits" — and per this repo's most expensive
recurring shape, a wrong sizing verdict is silent while a refusal is loud.

Once (2) exists, the two dead fixes above become live and should land with it:
`ok` incorporates `fit.feasible`, and the controller declines to apply an object
whose own status says it does not fit.

## Consequences

**What gets better.** The operator does the thing it was built for. A workload
rescheduled from an H100 to an L4 is re-planned against the L4, which is the
entire argument for running a controller rather than committing a Deployment.

**What this costs.** Workloads naming a repo outside the catalogue stop being
applied. That is a behaviour change and it will surprise someone — but the
alternative is what exists today: applying a Deployment whose flags were derived
from no hardware at all, which fails later, in the cluster, as an OOM under a
status reading `Planned`.

**Migration.** `clickllm catalog-add` already exists for adding an entry from a
repo's `config.json`, so the remedy for `ModelNotInCatalogue` is a documented
one-command path rather than a dead end. The status message should name it.

**What this does not settle.** Which quantisation the operator should choose when
the CRD does not say. `best_quant` picks the largest that fits, which is the
right default, but a CRD field to pin it is likely wanted and is out of scope
here.

## Alternatives considered

**Size only when the model happens to resolve, and skip otherwise.** Rejected:
that is the current behaviour with extra steps, and it reintroduces exactly the
"unknown renders as fine" failure ADR-0012 exists to prevent.

**Fetch the config from Hugging Face and size from that.** Rejected for the
operator. It puts a network call in a reconcile loop that must work in an
air-gapped cluster, and it makes the plan depend on a remote whose availability
is unrelated to the workload's. `catalog_update` already does this offline, at
the point where a human is looking.

**Fuzzy-match the repo to a catalogue id.** Rejected. `Qwen/Qwen3-32B` and
`Qwen/Qwen3-32B-FP8` differ in exactly the way that matters for sizing, and a
near-match that sizes the wrong model produces a confident wrong number — the
failure mode this repo has spent the most effort removing.
