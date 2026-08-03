# Competitive landscape

Twelve products across four layers. For each: what it does, what to **steal**, and why it doesn't close the gap.

---

## Layer 1 — Inference engines (we integrate, never compete)

### 1. vLLM
The default. PagedAttention, continuous batching, widest hardware support (CUDA, ROCm, TPU, Trainium), OpenAI-compatible. HF Inference Endpoints default to it. ~12,500 tok/s on the common benchmark. Ships EAGLE-3 and P-EAGLE (`"parallel_drafting": true`, up to 1.69× over EAGLE-3 on B200).
**Steal:** the OpenAI-compatible surface as our universal contract. Their spec-decode config schema.
**Gap:** it's an engine. It won't tell you *which* model to run or whether it's good enough.

### 2. SGLang
RadixAttention prefix reuse — wins on agent workloads with shared system prompts and growing histories. ~16,200 tok/s. EAGLE-3 gives 1.81× at batch 2, 1.38× at batch 64.
**Steal:** RadixAttention is the reason agent workloads are cheaper self-hosted than via API. That's a *sales argument*, and we should compute it per customer.
**Gap:** same as vLLM. Also CUDA-first.

### 3. llama.cpp / MLX / mlx (Apple Silicon)
`llama.cpp` best single-stream + widest models; MLX highest throughput (~230 tok/s aggregate) with spec-decode since mlx-lm 0.21; **mlx** brings continuous batching to Metal (EuroMLSys '26, ~1,150 tok/s @ 32 concurrency on M4 Max).
**Steal:** all three. This is the entire local story and the reason we can dogfood.
**Gap:** three engines, three config formats, zero guidance on choosing.

---

## Layer 2 — Kubernetes serving (we generate for, never compete)

### 4. llm-d
Red Hat/Google/IBM. Disaggregated prefill/decode for 70B+ models. KV-cache-aware routing via a global cache indexer. Measured **3× output tok/s, 2× TTFT reduction** (Llama 3.1 70B, 4× MI300X). Prefix cache hit rates 75–80% in production (Snap).
**Steal:** we emit llm-d manifests as a deploy target. Their EPP scheduling is the right answer; don't reinvent it.
**Gap:** assumes you already chose the model and already trust it.

### 5. Gateway API Inference Extension (GAIE)
Kubernetes SIG. `InferencePool` API + Endpoint Picker Protocol. Routes on KV-cache utilization, queue depth, LoRA adapters. Core EPP has since merged into `llm-d-router`. Backs GKE Inference Gateway.
**Steal:** **do not write our own production router.** Our router is a *migration* router (shadow/canary/compare); once traffic is cut over, hand off to GAIE.
**Gap:** routing only. No opinion on models or quality.

### 6. KServe
CNCF incubating. `InferenceService` CRD, Knative scale-to-zero, canary, multi-model. Added `huggingfaceserver` + vLLM/TGI runtimes.
**Steal:** the CRD shape and canary primitives as a deploy target.
**Gap:** critics note the LLM features are visibly grafted onto a classical-ML core. And again — no model selection, no evals.

### 7. BentoML / OpenLLM / Yatai
One `bentofile.yaml` targets vLLM, TGI, or TensorRT-LLM — runtime swap is a build flag. Great Python ergonomics, local/prod parity.
**Steal:** **the single-manifest-many-runtimes idea is the best thing in this whole landscape.** It's the shape of our runtime abstraction.
**Gap:** wrappers lag upstream engines by a release or more; rebuild-push-redeploy friction on every flag change. We emit *native* config instead of wrapping.

### 8. KAITO
AKS operator, workspace CRD, node auto-provisioning for GPUs.
**Steal:** auto-provisioning UX — "declare the model, get the nodes."
**Gap:** Azure-locked.

---

## Layer 3 — Local kiosk (our UX bar, and our on-ramp)

### 9. Ollama
The one to beat on ergonomics. `ollama run <model>`, daemon on :11434, 4,500+ models, OpenAI-compatible. Consolidating onto MLX (0.19+), killing the "slow on Mac" critique. Zero telemetry.
**Steal:** the daemon model, the one-command bar, the OpenAI-compatible default. **Also: integrate it as a backend rather than replace it.** Many of our users already have it.
**Gap:** no eval, no routing, no fleet, no cost model, no k8s, no migration. It answers "can I run this?" — never "should I?"

### 10. LM Studio / Jan
LM Studio: best GUI, granular hardware controls, **side-by-side model comparison**, HF model browser. Free commercial since Jul 2025, telemetry on by default. Jan: Apache 2.0, MCP support, zero telemetry.
**Steal:** LM Studio's side-by-side comparison is the closest existing thing to our equivalence view — and it's manual, one prompt at a time. We automate it across a whole traffic corpus. Steal Jan's MCP-native posture.
**Gap:** desktop apps. Server dies when the window closes (LM Studio). No production path.

### 11. Open WebUI / AnythingLLM / Dify
Open WebUI: chat polish, RBAC, Functions framework, 9 vector DBs. AnythingLLM: RAG-first workspaces, `@agent` with web/SQL/FS tools, MIT. Dify: 140K stars, visual workflow builder.
**Steal:** Open WebUI's Functions extensibility model; AnythingLLM's workspace concept as our *project* boundary.
**Gap:** all three are consumption surfaces. None does model selection, evals, hardware fit, or migration.

---

## Layer 4 — Gateways & evals (we compose, partially compete)

### 12. LiteLLM / OpenRouter / Portkey
LiteLLM: OSS self-hosted proxy, 100+ providers, virtual keys, MCP, ~10–20ms overhead. OpenRouter: 300+ models hosted, 5.5% fee. Portkey: observability + guardrails.
**Steal:** LiteLLM's provider adapter layer — **we should run on top of LiteLLM, not rebuild it.** It's the cheapest correct transport.
**Gap:** LiteLLM logs your traffic and stops. It has every input needed to tell you "you could save $40K/mo on Kimi K3" and does nothing with it. **That inaction is our entire product.** OpenRouter is structurally opposed to helping you self-host.

### 13. promptfoo / lm-eval-harness / Inspect AI / DeepEval
promptfoo: YAML-first, LLM judge, CI, strong red-teaming — **now owned by OpenAI**, which is sunsetting its own Evals dashboard (read-only Oct 31 2026, shutdown Nov 30). lm-eval-harness: the academic standard, HF leaderboard backend. Inspect AI: UK AISI, agent audits.
**Steal:** promptfoo's assertion grammar and CI shape; run lm-eval-harness for capability screening. Use them as *libraries*, not as the product.
**Gap:** all start from a human-authored eval set. **Nobody generates the eval set from your production traffic, baselined against your incumbent's actual responses** — the step that actually blocks migrations. The nearest prior art anywhere is generic-rubric scoring against a fixed set of roles, which answers "best at coding in general" rather than "matches what GPT-5 did for us". Also: promptfoo being OpenAI-owned is a live strategic risk for a product whose purpose is leaving OpenAI. Plan for Inspect AI as the primary.

### 14. NVIDIA NIM / Baseten / Fireworks / Together / Modal
NIM: containerized microservices, max control, k8s ops burden. Baseten: self-hosted/hybrid in customer VPC, Truss bundles, $300M Series E at $5B (Feb 2026). Fireworks/Together: hosted APIs on vLLM/SGLang/TRT-LLM. Modal: IaC in Python, but all I/O routes through a us-east-1 control plane.
**Steal:** Baseten's hybrid/VPC posture is the enterprise shape. NIM's "model as a versioned container" packaging.
**Gap:** these are *destinations*, and expensive ones. They want you on their fleet. None helps you decide whether to leave your current provider — that's adversarial to their business model. **Nobody in this list is incentivized to build what we're building.** That's the strategic opening.

---

## Synthesis: the unowned seam

```
 model choice → hardware fit → proof of equivalence → deploy → cutover → drift watch
 ─────────────  ─────────────  ─────────────────────  ──────  ────────  ──────────
 leaderboards   nobody         nobody                 solved  nobody    nobody
 (unreliable)                                         (×5)
```

Four of six steps are unowned. Deployment — the one you originally centered — is the *only* solved step.

## Features worth stealing, ranked

1. **Single manifest → many runtimes** (BentoML) — but emit native config, don't wrap.
2. **KV-cache-aware routing** (GAIE/llm-d) — adopt wholesale for post-cutover.
3. **Side-by-side model comparison** (LM Studio) — automate it across a traffic corpus. This is the product's hero view.
4. **One-command bar + daemon** (Ollama) — the UX floor. Anything worse loses.
5. **Provider adapter layer** (LiteLLM) — depend on it.
6. **Assertion grammar + CI** (promptfoo) — but own the eval-set *generation*.
7. **Workspace/project isolation** (AnythingLLM) — multi-app tenancy.
8. **Node auto-provisioning** (KAITO) — declare model, get capacity.
9. **VPC/hybrid deploy** (Baseten) — enterprise requirement.
10. **Extensible functions** (Open WebUI) — customization escape hatch.

## Features nobody has (our surface area)

1. Eval sets **distilled from your own production traffic**.
2. **Equivalence matrix** — per-task-cluster, open vs. incumbent closed model.
3. **Hardware-fit solver** — what actually runs on the silicon you own, at what context and concurrency.
4. **Migration router** — shadow → canary → cut, with auto-rollback on quality regression, not just errors.
5. **Cost-delta ledger** — real dollars saved on real traffic, not list-price arithmetic.
6. **Drift watch** — new open model drops weekly; auto-re-run *your* evals and propose promotions.
7. **Regret analysis** — the tasks where the open model loses, clustered and named, so you know exactly what still needs a closed model.
