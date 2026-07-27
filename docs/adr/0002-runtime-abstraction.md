# ADR-0002 — Runtime abstraction, and emit native config

**Status:** accepted · **Date:** 2026-07-27

## Context

Two forcing functions:

1. **Dev and prod run different engines.** The primary dev machine is an Apple M4 Max. vLLM (CUDA), SGLang, llm-d, and TensorRT-LLM do not run on it. The Metal-side stack is vllm-mlx, llama.cpp, mlx-lm, and MLC-LLM. Customer prod is CUDA. They share nothing but the OpenAI wire format.

2. **The runtime layer is not stable.** vllm-mlx was accepted at EuroMLSys '26. mlx-lm got speculative decoding in 0.21 (May 2026). Ollama moved from llama.cpp to MLX in 0.19. P-EAGLE landed in vLLM in March 2026. Anything hard-wired to one engine is refactoring debt with an 18-month fuse.

BentoML demonstrates the tempting wrong answer: wrap the engines behind one manifest. The observed cost is wrappers lagging upstream by one or more releases, plus rebuild-push-redeploy friction on every flag change — while the optimizations that matter (paged attention, prefix caching, continuous batching, spec-decode) are all engine-dependent anyway.

## Decision

A `Runtime` Protocol with three methods — `supports`, `plan`, `render` — and one hard rule:

> **`render` emits native engine configuration. We never wrap an engine at runtime.**

Output is a real `vllm serve` invocation, a real `InferencePool`, a real `mlx_lm.server` command — readable, forkable, and runnable with clickllm uninstalled (NFR-4).

No engine-specific type crosses above the Protocol boundary. The moment `prove` imports `vllm`, portability is gone.

## Consequences

**Good**
- Dogfooding works. The whole loop runs locally on Metal against the same code path that generates CUDA config.
- Zero upstream lag. New vLLM flags are a template change, not a wrapper release.
- No lock-in by construction — the correct posture for a product whose premise is escaping lock-in.
- New engines are additive: one Protocol implementation.

**Bad**
- We must track config schemas across seven engines and their releases. Mitigation: version templates per engine release; CI runs generated configs against real engines.
- We can't offer a uniform runtime API (no `clickllm serve` that abstracts everything). Accepted — that's what Ollama is for, and pretending otherwise recreates BentoML's problem.
- `supports()` must be honest across hardware we may not own. Mitigation: "yes, degraded" is a valid return; a runtime error on a customer cluster is not.

## Notes

- Recommendation logic must encode non-obvious tradeoffs, e.g. *llama.cpp Metal wins single-stream (~15%) but vllm-mlx wins ~27× on aggregate throughput at 32 concurrency* — the right pick depends entirely on workload shape from stage ②.
- Spec-decode config must be tuned to **observed** concurrency. EAGLE-3's 2–3× is a single-stream figure; realistic serving is ~1.3–1.8×, and acceptance degrades above batch ~32. Generating an untuned `num_speculative_tokens` is worse than generating none.
