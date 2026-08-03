# Verdict: build vs. buy vs. don't

**Date:** 2026-07-27 · **Status:** decided · **Author:** research pass 1

You asked: *do I need this, or does it already exist?* Honest answer in three parts.

---

## 1. What you described mostly exists. Don't build it.

| What you asked for | Who already owns it | Verdict |
|---|---|---|
| Deploy open LLMs on k8s | [KServe](https://kserve.github.io) (CNCF incubating), [llm-d](https://llm-d.ai), vLLM production-stack, [KAITO](https://github.com/kaito-project/kaito), BentoML/Yatai | **Don't build.** Four funded projects, one CNCF-blessed. |
| High-perf inference engine | vLLM, SGLang, TensorRT-LLM, LMDeploy | **Don't build.** You will never beat these. |
| Speculative decoding / EAGLE-3 | Already shipped *inside* vLLM (`--speculative-config`), SGLang (`--speculative-algorithm EAGLE3`), and mlx-lm 0.21 | **Don't build. Configure it.** |
| KV-cache-aware routing | [Gateway API Inference Extension](https://github.com/kubernetes-sigs/gateway-api-inference-extension) + llm-d Endpoint Picker. 3× output tok/s, 2× TTFT reduction, measured. | **Don't build. Adopt it.** |
| Local one-click LLM | Ollama, LM Studio, Jan, GPT4All | **Don't build.** Ollama has 4,500+ models and a daemon. |
| Multi-provider gateway | LiteLLM (self-host), OpenRouter, Portkey | **Don't build the transport.** ~10–20ms overhead, solved. |
| Eval harness | lm-eval-harness, Inspect AI, promptfoo, DeepEval | **Don't build the runner.** |
| Chat UI / RAG | Open WebUI, AnythingLLM, Dify (140K stars) | **Don't build.** |

Building any one of these is a losing race. Building all of them is how a startup dies.

---

## 2. The gap nobody owns

Your clarification is the whole product:

> *"help builders leverage this to pivot from using closed source models and make their life easy"*

The blocker for that pivot **is not deployment**. Ollama solved deployment in 2023. The blocker is **evidence**.

A team running GPT-5 or Claude in production will not switch to Kimi K3 because a leaderboard says 76.8 on SWE-Bench. They switch when someone shows them: *on your actual traffic, this open model matches your closed model 94% of the time, costs 11× less, runs on hardware you already have, and here is the rollback button.*

Nobody produces that artifact. Look at the seams:

- **LiteLLM** sees all your traffic and does nothing with it. No eval, no equivalence, no migration.
- **promptfoo** evaluates prompts but has no idea what your production traffic looks like, and can't deploy.
- **lm-eval-harness** measures MMLU. Your users don't send MMLU.
- **Ollama** deploys the model but can't tell you if it's good enough for *your* job.
- **OpenRouter** is structurally disincentivized from helping you self-host.
- **llm-d / KServe** assume you already decided which model, and already believe.

Every tool covers one step. **The migration itself is unowned.** That's the wedge.

---

## 3. The reframe

**Not:** "a next-gen platform to deploy open LLMs on Kubernetes."
**But:** *"Point it at your OpenAI/Anthropic traffic. It proves which open model can replace it, on your hardware, and moves you there without a risky cutover."*

Deployment becomes an **output**, not the product. You still ship all the k8s/vLLM/SGLang/EAGLE-3 machinery you wanted — but as generated artifacts on the far side of a decision you *earned*, not as another orchestrator competing with CNCF.

This is defensible because the moat is data you generate and nobody else holds: a customer's private eval set distilled from their own traffic, plus the equivalence history behind it. That compounds. A Helm chart does not.

**Verdict: build it — but build the migration, not the platform.** Full spec in [`20-prd.md`](20-prd.md).

---

## 4. Hard constraint you need to know now

**Your machine is an Apple M4 Max (128 GB unified, 16 cores, no NVIDIA).**

Half the stack you named is CUDA-first and will not run locally:

| Component | On M4 Max |
|---|---|
| vLLM (CUDA) | ❌ |
| SGLang | ❌ |
| llm-d | ❌ (needs multi-GPU CUDA nodes) |
| TensorRT-LLM | ❌ |
| EAGLE-3 as documented in vLLM | ❌ (mlx-lm 0.21 has its own spec-decode since May 2026) |
| **mlx** | ✅ MLX-based vLLM plugin, continuous batching, ~525 tok/s small models on M4 Max, ~1,150 tok/s aggregate @ 32 concurrency |
| **llama.cpp Metal** | ✅ best single-stream, broadest model support |
| **MLX / mlx-lm** | ✅ highest throughput, spec-decode since 0.21 |
| **MLC-LLM** | ✅ best 64K–128K context (paged KV) |

This is not a footnote — it's an architectural forcing function. It means the product **must** treat the runtime as a swappable backend from commit one, because your dev machine and your customers' prod clusters run *different engines*. Which is exactly the abstraction that makes the product portable. See [ADR-0002](adr/0002-runtime-abstraction.md).

It also means: **you can dogfood the whole loop locally**, with 128 GB of unified memory, which is more than most GPUs. That's a genuine advantage.

---

## Sources

[vLLM vs SGLang 2026](https://www.yottalabs.ai/post/vllm-vs-sglang-which-inference-engine-should-you-use-in-2026) ·
[AI/ML on Kubernetes 2026 stack](https://kubernetesguru.com/ai-ml-on-kubernetes-2026-stack-guide/) ·
[KServe production serving](https://medium.com/@nonickedgr/kserve-production-ml-serving-on-kubernetes-from-sklearn-to-llms-6a0bbc923fd5) ·
[llm-d KV-cache aware routing](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference) ·
[Gateway API Inference Extension](https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/) ·
[EAGLE-3 / P-EAGLE in vLLM](https://vllm.ai/blog/2026-03-13-p-eagle) ·
[llama.cpp vs MLX vs Ollama vs vLLM on Apple Silicon](https://contracollective.com/blog/llama-cpp-vs-mlx-ollama-vllm-apple-silicon-2026) ·
[LLM gateway comparison](https://wavect.io/blog/llm-gateway-router-comparison-2026/) ·
[Eval framework comparison](https://machinelearningmastery.com/llm-evaluation-frameworks-compared-how-to-actually-measure-what-your-model-does/) ·
[Open-source LLM leaderboard July 2026](https://www.vellum.ai/open-llm-leaderboard) ·
[Open WebUI alternatives](https://onyx.app/insights/openwebui-alternatives) ·
[Self-hosted inference platforms](https://www.digitalocean.com/resources/articles/ai-inference-platforms)
