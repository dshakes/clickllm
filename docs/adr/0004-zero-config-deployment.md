# ADR-0004 — Deployment is zero-config. The generated file is a receipt, not an interface.

**Status:** accepted · **Date:** 2026-07-27 · **Amends:** [ADR-0001](0001-migration-not-platform.md), [ADR-0002](0002-runtime-abstraction.md)

## Context

A stated product goal: **make deploying open models easy, with the complex optimizations and configs abstracted away from the user.**

Two earlier decisions read as if they contradict this:

- [ADR-0001](0001-migration-not-platform.md) demoted deployment to "an output, not the product."
- [ADR-0002](0002-runtime-abstraction.md) requires emitting *native* engine config — which, read alone, sounds like "here's a 40-flag `vllm serve` command, good luck."

Both were right about **strategy** (don't compete with vLLM/KServe/llm-d) and wrong about **posture** (they made the user sound responsible for the config).

The abstraction is also where the real user pain is. Nobody knows, unprompted, that:
- EAGLE-3 gives ~1.3–1.8× at realistic serving concurrency, not the 2–3× single-stream headline — and *degrades* above batch ~32, where an untuned `num_speculative_tokens` makes you **slower**
- `kv_heads` is not attention heads (up to 8× KV overestimate), and MLA models need a different formula entirely (~50× off)
- SGLang's RadixAttention only pays off when your prompts share a prefix
- vLLM's `gpu-memory-utilization` default leaves headroom you may need back
- MoE sparsity cuts compute, not memory

That knowledge is the product. Handing someone a config file and expecting them to supply it is handing them the problem back.

## Decision

**Deployment is zero-config by default.** The user names a model and a target. onpar chooses every knob, from evidence it already has: hardware profile (③), observed workload shape (②), and proven quality (④).

Auto-tuned, never asked:

| Knob | Derived from |
|---|---|
| quantization | ③ memory solve, re-validated by ④ — never quantize without re-proving |
| speculative decoding: method, draft model, `num_speculative_tokens` | ② observed concurrency + ③ hardware; **disabled** when observed batch exceeds the acceptance cliff |
| prefix / radix caching | ② measured prefix-sharing rate across the traffic corpus |
| tensor & pipeline parallelism | ③ device count and topology |
| `max_model_len`, `max_num_seqs`, chunked prefill | ② p95 context and concurrency — **not** the model's advertised max |
| KV dtype, `gpu-memory-utilization` | ③ headroom after the memory solve |
| engine selection | ③ + ② (see `recommend_runtime`) |

The native config file still gets written — because [ADR-0002](0002-runtime-abstraction.md) and NFR-4 stand — but it is a **receipt**: proof of what was chosen and an exit door if you ever fire us. Reading it is optional. Editing it is optional. It is never a prerequisite.

**Resolution of the tension:** *kiosk outside, glass box inside* — the same principle as the rest of the product. The abstraction is the default path; the transparency is the escape hatch. ADR-0001's "deployment is an output" remains true as **strategy** (we don't build an orchestrator) and is now explicitly false as **user experience** (we own the whole knob surface).

## Consequences

**Good**
- Delivers the actual promise. "Deploy the best open model for my job" becomes one command with no LLM-infra expertise required.
- The auto-tuning is defensible *because* of stages ②③④. Nobody else can tune from observed traffic, because nobody else has it. **The abstraction is downstream of the moat, not a substitute for it.**
- Still no lock-in: the receipt runs with onpar uninstalled.

**Bad**
- **A wrong auto-tune is worse than no auto-tune.** Silently setting `num_speculative_tokens` at concurrency 64 makes a user slower and they will never know why. This is the same class of risk as a wrong equivalence verdict.
  → **Mitigation, and it changes the roadmap:** every generated config must be **measured, not just computed**. Phase 4 (Deploy) gains a mandatory benchmark step — start the engine, run the observed workload shape, compare against the unoptimized baseline, and *revert any optimization that doesn't help on this hardware*. Roofline estimates pick the candidate settings; measurement ratifies them.
- Raises Phase 4's cost and priority. Accepted.
- Every auto-tuned knob must be explainable on demand (`--explain`), or it's a black box we told users to trust.

## Follows from this

- `onpar deploy --model X` takes **no tuning flags**. Overrides exist (`--set`), are documented, and are never required.
- Every generated artifact carries a header comment: what was chosen, why, and the measurement that ratified it.
- "Fits but will be slow" and "optimization reverted, it didn't help here" are first-class outputs. Silence about a disabled optimization is a bug.
