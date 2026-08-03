---
name: clickllm
description: >-
  Work out which open-weight LLM can run on a machine, at what quantization,
  context and concurrency — and which serving runtime to use. Use when someone
  asks "what model can I run on this", "will X fit in Y GB", "why doesn't this
  fit", "should I use vLLM or llama.cpp", or is planning a move off a closed
  model API. Also use before recommending any specific open model for
  self-hosting, because the answer depends entirely on their hardware.
---

# clickllm

Answers **"what can I actually run, and should I?"** with arithmetic you can check.

## When to reach for this

- "Can I run GLM-5.2 on my Mac?" / "Will a 70B fit in 24 GB?"
- "Why doesn't Kimi K3 fit? It's only 50B active."
- "vLLM or llama.cpp?" — the answer depends on concurrency, not preference
- Anyone weighing a move from OpenAI/Anthropic to self-hosted

**Do not guess from parameter counts.** The three ways people get this wrong are
all silent, and all corrected below.

## Use it

```bash
clickllm fit --context 32k --concurrency 8     # what fits, with tok/s and licences
clickllm fit --explain qwen3-30b-a3b           # the full arithmetic
clickllm fit --json                            # machine-readable
clickllm models                                # catalogue with licences
```

From Python:

```python
from clickllm import sdk
report = sdk.fit(context="32k", concurrency=8)
best = report.best()
clean = report.commercially_clean()   # no licence review needed
print(sdk.explain("glm-5.2"))
```

## The three silent errors

1. **MoE sizes on TOTAL parameters.** Kimi K3 activates 50B of 2.8T per token,
   so people assume it needs 50B of memory. It needs all 2.8T resident —
   sparsity cuts *compute*, not *memory*. This is the most common mistake.
2. **GQA uses `kv_heads`, not attention heads.** Using attention heads
   overestimates KV cache by up to 8×.
3. **MLA (DeepSeek family) has a different formula entirely.** It compresses K
   and V into one low-rank latent. Applying the GQA formula overestimates by
   roughly 50×.

## Reading the output honestly

- **tok/s is a memory-bandwidth roofline estimate, not a measurement.** Say so
  when you report it. It is a ceiling, and real throughput lands below it.
- **`?` on a model means its architecture is unverified** — the KV figures carry
  error bars. Do not present those numbers as firm.
- **Licence column matters.** `Apache-2.0`/`MIT` are clean. Llama and Gemma
  carry caps or use restrictions that a human must read before production.
- **The NOT FEASIBLE section is the useful half.** It says *why*, which is
  usually what the person actually needed to know.

## Runtime choice depends on workload, not taste

| Situation | Runtime | Why |
|---|---|---|
| Apple Silicon, concurrency ≥ 4 | `mlx` | continuous batching; aggregate throughput dominates |
| Apple Silicon, single stream | `llama.cpp` Metal | best single-stream decode, widest model support |
| Apple Silicon, 64K+ context | `llama.cpp` | paged KV holds throughput steadiest |
| One NVIDIA GPU | `vLLM` | broadest support; enable EAGLE-3 |
| One GPU, concurrency ≥ 8, shared prefixes | `SGLang` | RadixAttention reuses the prefix |
| Multi-GPU | `llm-d` + GAIE | disaggregated prefill/decode, KV-cache-aware routing |

**vLLM, SGLang and llm-d do not run on Apple Silicon.** They are CUDA/ROCm-only.
If someone on a Mac asks for a vLLM setup, say that first.

## Speculative decoding is not free

EAGLE-3's headline 2–3× is a **single-stream** figure. At realistic serving
concurrency expect ~1.3–1.8×, and past batch ~32 acceptance falls off far enough
that it makes things **slower**. `clickllm` disables it above that cliff and says
so. Never recommend enabling it without knowing the concurrency.
