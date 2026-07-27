# ADR-0003 — Don't build the fit layer. Adopt llmfit.

**Status:** superseded by [ADR-0008](0008-build-from-scratch.md) · **Date:** 2026-07-27 · **Supersedes:** Phase 0 of [`50-roadmap.md`](../50-roadmap.md)

## Context

Phase 0 was `clickllm fit` — a hardware profiler + model-sizing solver, planned as the free on-ramp. It was built and works ([`src/clickllm/fit.py`](../../src/clickllm/fit.py), 27 tests green).

Then [AlexsJones/llmfit](https://github.com/AlexsJones/llmfit) surfaced. It is:

- **30.8k stars, 1,014 commits, post-1.0**, MIT, Rust
- Hardware detection (`llmfit doctor`), MoE-aware sizing, dynamic quantization selection, multi-GPU
- **Real benchmarking** (`llmfit bench`) — measured tok/s and TTFT against a live provider, with community-submitted results merged upstream so others on matching hardware get *measured* rather than estimated numbers
- Provider integration: Ollama, llama.cpp, MLX, Docker Model Runner, LM Studio
- Transparent estimates — `llmfit info` ships the inputs behind every number
- CLI + TUI + web + desktop + Python bindings; JSON output for agents (`llmfit recommend --json`)

Our solver produces *roofline estimates*. Theirs produces estimates **corrected by a community measurement corpus we cannot replicate.** On the axis that matters — accuracy of the tok/s number — they win permanently, and the gap widens with every submitted benchmark.

## Decision

**Delete the fit layer as a product surface. Depend on llmfit.**

- `clickllm fit` becomes a thin adapter over `llmfit recommend --json` / `llmfit doctor`.
- Keep only the delta llmfit does not cover:
  1. **Serving-side targets** — llmfit's providers are local (Ollama, llama.cpp, MLX, LM Studio). It has no opinion on vLLM, SGLang, llm-d, KServe, or multi-node k8s fleets. Its sister project `llmserve` covers serving separately.
  2. **Solving at *observed* context and concurrency** — llmfit asks you for these. We read them from captured traffic (stage ②). "What fits at your actual p95 of 18K, not the model's advertised 1M" is only answerable if you have the traffic.
- Keep the existing solver code as the k8s/serving-side implementation and the offline fallback. It is no longer a headline feature and gets no further investment.

## Consequences

**Good**
- Removes a whole surface we would have lost. 30.8k stars of distribution and a measurement corpus is not a race worth entering.
- Sharpens the pitch. Phase 0 was always the weakest claim to differentiation; losing it forces the story onto stages ②④⑥⑦, which is where the moat actually was.
- Better product: llmfit's measured numbers beat our estimates, so users get a *more* accurate answer than we would have shipped.
- Natural integration partner rather than competitor. Adjacent, not overlapping.

**Bad**
- Loses the planned free/viral on-ramp. **Mitigation: the on-ramp moves to `clickllm observe`** — the cost dashboard over your existing closed-model spend. Arguably a better wedge anyway: it targets people already paying an API bill, which is exactly the audience for the pivot, whereas `fit` attracted people already running local models (already converted).
- A hard runtime dependency on a third-party binary. Mitigation: the retained solver is the offline fallback; llmfit is MIT and forkable.
- Some work is now dead. Small, and the honest cost of checking prior art late instead of first.

## Lesson

This should have been the first search, not the eighth. The competitive research covered engines, orchestrators, gateways, and eval frameworks and still missed a 30k-star tool sitting exactly on Phase 0 — because the search terms were about *categories* ("Kubernetes LLM serving") rather than about *the specific command we were about to write*. Search for the feature, not the market.
