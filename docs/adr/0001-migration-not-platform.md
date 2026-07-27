# ADR-0001 — Build the migration, not the platform

**Status:** accepted · **Date:** 2026-07-27

## Context

The original brief was a next-gen platform for deploying open LLMs: Kubernetes, SGLang/llm-d, speculative decoding, evals, routing, kiosk UX. Research ([`10-landscape.md`](../10-landscape.md)) found every one of those components already owned by a funded or CNCF-backed project — KServe, llm-d, vLLM production-stack, BentoML, KAITO, Ollama, LiteLLM, promptfoo.

The clarified goal is narrower and sharper: *help builders pivot off closed models.*

The blocker for that pivot is not deployment. Ollama solved deployment. The blocker is **evidence** — no team moves production off Claude because a leaderboard says a number.

## Decision

Build the **migration**, not the platform. Deployment becomes a generated output on the far side of a proven decision, not the product.

> **Amended by [ADR-0004](0004-zero-config-deployment.md):** this is a decision about *strategy* (we do not build an orchestrator), not about *user experience*. Deployment is zero-config — clickllm owns every tuning knob and the user is never asked for one.

The moat is four stages nobody owns: **distill** (traffic → private eval set), **prove** (equivalence matrix + regret), **cut over** (quality-gated canary), **guard** (drift watch). Everything else is composed from existing tools.

## Consequences

**Good**
- No head-to-head against CNCF projects or a $5B-valuation Baseten.
- Moat is customer-generated data (private eval sets, equivalence history) that compounds. Helm charts don't.
- Incumbents are structurally unable to follow: OpenRouter, Fireworks, Together, and Baseten all profit from you *not* leaving. LiteLLM has the traffic and no incentive to act on it.
- Deployment complexity becomes a feature we *emit* rather than a system we *operate*.

**Bad**
- Judge quality becomes existential. A wrong verdict breaks a customer's production system. Mitigations in [`30-architecture.md`](../30-architecture.md#grader-stack) — three-tier grading, position-swap, disclosed agreement rates, shadow mode as the real gate.
- Narrower initial TAM than "AI infra platform." Accepted — a narrow product that works beats a broad one that doesn't.
- Dependent on LiteLLM and Inspect AI. Both Apache-ish and forkable; acceptable.

**Reversible?** Partly. The runtime abstraction (ADR-0002) and config generation are reusable if we later pivot toward platform. The graders are not — that investment is bet-specific.

## Alternatives rejected

- **Another k8s serving platform** — losing race against four funded incumbents.
- **Better local kiosk than Ollama** — Ollama has 4,500+ models, a daemon, and MLX consolidation. Integrate with it instead.
- **An LLM gateway** — LiteLLM at ~10–20 ms overhead is good and free. Depend on it.
- **Eval framework** — Inspect AI and lm-eval-harness exist. The gap is eval-set *generation*, not the runner.
