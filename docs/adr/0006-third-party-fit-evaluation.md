# ADR-0006 — llmfit source evaluation: the overlap is larger than ADR-0003 assumed

**Status:** superseded by [ADR-0008](0008-build-from-scratch.md) · **Date:** 2026-07-27 · **Extends:** [ADR-0003](0003-adopt-third-party-fit.md)
**Basis:** read of `AlexsJones/llmfit` v1.1.6 — 45,450 LOC Rust, MIT

ADR-0003 adopted llmfit from its README. Reading the source changes the picture: it covers substantially more of our M1–M3 than assumed, and one module lands adjacent to the stage we called our hero. Correcting the record.

---

## What llmfit actually ships

| Module | LOC | What it does |
|---|---|---|
| `providers.rs` | 5,630 | **7 providers** behind a `ModelProvider` trait: Ollama, LM Studio, MLX, llama.cpp, Docker Model Runner, RamaLama, **vLLM**. Each exposes `start_pull()` with streamed progress. |
| `hardware.rs` | 4,585 | NVIDIA (multi-GPU, VRAM aggregation, name-based fallback), AMD via `rocm-smi`, Intel Arc via sysfs/`lspci`, Apple via `system_profiler`, **Ascend NPU** via `npu-smi`; backend detection drives speed estimation. |
| `fit.rs` | 3,875 | Fit levels (Perfect/Good/Marginal/Too Tight) × run modes (GPU/MoE-offload/CPU+GPU/CPU). MoE fixed-vs-active decomposition down to attention + router + shared experts + lm_head + embedding. |
| `models.rs` | 3,385 | HF-sourced catalog embedded at compile time; licence field + `matches_license_filter`; **known GGUF sources** (unsloth, bartowski repos); MLX variant detection. |
| `plan.rs` | 1,455 | Quantization hierarchy F32→Q2_K plus MLX and AWQ; KV-quant aware. |
| `share.rs` | 1,392 | **Community measurement flywheel** — bench results stored locally, then contributed upstream as a GitHub PR via OAuth device flow. No server, no `gh` needed, CI-usable. |
| `benchmarks.rs` + `bench.rs` | 1,869 | **Real measurement** — TTFT, TPS, total latency against live Ollama / vLLM / MLX endpoints. |
| `quality.rs` | 934 | **Role-based quality benchmarking** — 13 roles with scored rubrics, producing a **routing matrix mapping roles → best model**. |
| `claim.rs` | 364 | **Kubernetes DRA `ResourceClaim` generation** — CEL selector encoding the fit inequality so kube-scheduler places pods on nodes that can actually run the model. |

Plus `llmfit-python` (SDK), `llmfit-web`, `llmfit-tui`, `llmfit-desktop`, and `skills/llmfit-advisor` — an agentic skill packaged for OpenClaw.

---

## Where ADR-0003 was wrong

**1. "llmfit's providers are local; it has no opinion on vLLM."**
False. `VllmProvider` exists. *But* it is a read-only probe of `/v1/models` — llmfit **detects** a running vLLM, it never configures or launches one.

**2. "The gap is serving-side targets."**
Still true, and it is the larger half. There is no SGLang, no llm-d, no KServe, no serving-config generation anywhere in the tree. `claim.rs` addresses **scheduling** (which node can fit this), explicitly noting *"Nothing here runs in the serving path — the output is plain YAML."*

**3. "Nobody scores model quality." — the claim that needs the most correction.**
`quality.rs` scores models with rubrics and emits a routing matrix. That is genuinely adjacent to our stage ④.

The distinction that survives, and it is the whole thesis:

| | llmfit `quality.rs` | clickllm stage ④ |
|---|---|---|
| Test set | 13 fixed generic roles | **clustered from your captured production traffic** |
| Reference | rubric score, absolute | **your incumbent's actual responses to those exact prompts** |
| Question answered | "which open model is best at coding-in-general" | "does this match **what GPT-5 did for us**, per task cluster" |
| Output | routing matrix | equivalence matrix + regret + hybrid policy + cutover gate |

Both are useful; they answer different questions. Ours is not "better scoring" — it is **grounded in traffic and baselined against the incumbent**, which is what a migration decision actually requires. But "nobody evaluates model quality" was too strong and is now removed from our materials.

**4. Weight acquisition is more covered than assumed.** Provider `start_pull()` plus catalogued HF GGUF sources plus licence filtering covers much of what M1 was scoped to build.

---

## Decision

Deepen the dependency, and shrink our plan accordingly.

| Milestone | Was | Now |
|---|---|---|
| **M1 weights** | full acquisition stack, 2 wks | **~4 days.** Delegate discovery/pull to llmfit providers. Ours: direct-HF for non-GGUF, quant **conversion** (llmfit selects a quant, it does not produce one), and hard licence gating (llmfit *filters*; we must *refuse*). |
| **M2 runtimes** | 6 backends, 3 wks | **unchanged, and sharpened.** The delta is precisely `render()` + `launch()`. Detection is llmfit's; configuration and supervision are ours. Add SGLang, llm-d, KServe — absent upstream. |
| **M3 tune** | full knob solve + bench, 2 wks | **~1 wk.** Quant selection, bandwidth speed model, and measurement harness exist. Ours is the **serving** knob set llmfit has no concept of because it does not serve: spec-decode method + draft length, TP/PP, prefix/radix caching, chunked prefill, `max_num_seqs`, KV dtype. |
| **M4 box** | unchanged | no overlap |
| **M5 gateway** | unchanged | no overlap — no proxy, router, load balancing, or multi-model hosting upstream |
| **M8 prove** | unchanged in scope | **claim narrowed.** Position against generic-rubric scoring, not against a vacuum. |

**Also: contribute upstream rather than fork.** Our M3 benchmark-and-revert step produces exactly the (hardware, model, quant, measured tok/s) tuples `share.rs` collects. Feeding them back improves the corpus we depend on and costs us nothing — the incentives are aligned, and a fork would put us in a data race we would lose.

---

## Consequences

**Good**
- ~2.5 weeks removed from the critical path, and the removed work is the part someone else does better.
- MIT licence and a first-class Python SDK make the dependency genuinely safe.
- `claim.rs` is a better k8s placement answer than we had planned; we generate serving config *downstream* of their scheduling claim rather than duplicating it.
- Honest positioning is more durable. A reviewer who reads llmfit's source — as any serious one will — would have found `quality.rs` and discounted everything else we claimed.

**Bad**
- Deeper coupling to a fast-moving upstream. Mitigated: MIT, forkable, and our `Runtime` Protocol already isolates it behind one adapter.
- Our differentiation is narrower than ADR-0003 implied. That is a fact about the world, not a change in it — better known now.

**Watch:** if llmfit adds serving-config generation or traffic-grounded evaluation, the overlap becomes strategic rather than complementary. `quality.rs` is the module to monitor — a routing matrix is one short step from a gateway.
