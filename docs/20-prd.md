# onpar — Product Requirements Document

**Version:** 0.1 · **Date:** 2026-07-27 · **Status:** draft for review
**Format:** BMAD (Goals → Requirements → UX → Technical Assumptions → Epics → Stories)

---

## 1. One-liner

> **onpar proves which open model can replace your closed one — on your traffic, your hardware, your budget — then moves you there without a risky cutover.**

Working name. Superseded by [70-naming.md](70-naming.md), which recommends renaming to **Parity** — motto *"Prove it, then move it."*

---

## 2. Goals

**G1.** A builder currently paying an OpenAI/Anthropic bill can, in **under 30 minutes and one line of config change**, get a defensible answer to *"which open model replaces this, and what breaks?"*

**G2.** Cutting over carries **less risk than staying**, via shadow traffic, quality-gated canary, and one-command rollback.

**G3.** Every recommendation **shows its work** — the eval set, the scores, the failures, the hardware math. No black-box "trust us."

**G4.** The generated deployment is **yours** — native vLLM/SGLang/llm-d/MLX config you can read, fork, and run without us. No lock-in, by construction.

**G5.** Staying current is automatic: when a new open model ships, your eval set re-runs and you get a *proposal*, not a newsletter.

**G6a.** **Inference in a box.** `onpar run ./model.box` serves an OpenAI-compatible endpoint on any hardware — container on Linux/NVIDIA/AMD/k8s, supervised native runtime on macOS/Metal, same command and same endpoint either way. The box re-profiles and re-tunes on the host it lands on. ([ADR-0005](adr/0005-inference-in-a-box.md))

**G6.** **Deploying is zero-config.** Name a model and a target; onpar chooses quantization, speculative decoding, caching, parallelism, and batch limits from your hardware and your observed traffic. No LLM-infra expertise required, and no tuning flag is ever mandatory. ([ADR-0004](adr/0004-zero-config-deployment.md))

### Non-goals (v1)

- ❌ Writing an inference engine. ❌ Writing a production load balancer (GAIE owns it). ❌ Being a chat UI (Open WebUI owns it). ❌ Fine-tuning / training. ❌ Hosting inference ourselves. ❌ RAG framework.

---

## 3. Users

| Persona | Pain | Success |
|---|---|---|
| **Ravi — staff eng at a Series B** ($18K/mo Anthropic bill, CFO asking) | Believes open models are "probably fine now" but can't prove it and won't risk the product. | Ships a memo with an equivalence matrix; 70% of traffic on GLM-5.2 in 3 weeks. |
| **Mei — solo founder** (M4 Max, $400/mo OpenAI) | Wants to run local, drowns in Ollama/vLLM/quant/context choices, doesn't know what fits 128 GB. | `onpar switch` → running locally the same day, knows the 3 task types that still need the API. |
| **Dana — platform lead, regulated enterprise** | Mandated to get data off third-party APIs. Has GPUs. Needs an audit trail for the decision. | Signed-off equivalence report + llm-d manifests + continuous drift evidence. |

**Wedge = Ravi.** Has budget pain, technical ability, and a decision to justify. Mei is the volume/community on-ramp. Dana is the revenue.

---

## 4. The Switch Loop (core product)

Seven stages. Each is an **agent with tools**, not a wizard step — conversational, resumable, and able to skip ahead when the answer is already known.

```
① OBSERVE  →  ② DISTILL  →  ③ FIT  →  ④ PROVE  →  ⑤ DEPLOY  →  ⑥ CUT OVER  →  ⑦ GUARD
   proxy in      traffic →      what runs   open vs      native      shadow →       new models →
   front of      private        on your     closed,      config      canary →       re-eval →
   your calls    eval set       silicon     per task     you own     cut, w/ RB     propose
```

**Stages ②, ④, ⑥, ⑦ are the moat.** ①, ③, ⑤ are table stakes that nobody has bundled with them.

### ① Observe — the trojan horse
Swap `base_url`. That's the whole integration. We proxy to your existing provider (via LiteLLM), record request/response/latency/cost/tool-calls, redact PII **locally before write**. Useful on day one as a cost dashboard even if you never switch — which is how it earns the right to stay installed.

### ② Distill — the asset
Cluster captured prompts by task shape (embedding + structural features: system prompt, tool schema, output format, context length). Sample representatively per cluster. The recorded closed-model output becomes the **incumbent baseline** — explicitly *not* ground truth. Output: a private, versioned eval set that is worth more than any public benchmark because it *is* your product.

### ③ Fit — the hardware solver
Profile the target (local Mac / GPU box / k8s cluster / cloud). Solve: weights + KV cache + activations vs. available memory, across quantization levels, for the context lengths **step ② observed you actually use**. Emit a feasible set with projected tok/s, concurrency, and cost. Rules out 80% of the catalog before you waste a GPU-hour.

### ④ Prove — the hero output
Run feasible candidates against the distilled eval set. Grade with a three-way stack: programmatic assertions (schema, tool-call correctness, format) → task-specific graders → pairwise LLM judge vs. the incumbent. Output: the **equivalence matrix**.

|  | codegen | tool-calling | long-ctx refactor | summarize | classify |
|---|---|---|---|---|---|
| GLM-5.2 (Q4, local) | **96%** | 91% | 62% ⚠ | **99%** | **100%** |
| Kimi K2.7 Code | **98%** | **94%** | 71% | 95% | **100%** |
| Qwen3.6-27B | 84% | 79% | 55% ⚠ | 97% | **100%** |

Plus **regret analysis**: the named, clustered tasks where open loses — so you keep a closed fallback for exactly those and nothing more.

### ⑤ Deploy — generated, not wrapped
Emit **native** config for the chosen target, tuned by ③:
- **Local Mac** → mlx (concurrency) or llama.cpp Metal (single-stream) or Ollama (zero-config), with mlx-lm spec-decode
- **Single GPU box** → vLLM + EAGLE-3 (`num_speculative_tokens` tuned to observed concurrency — flat gains ~1.3–1.8× at realistic load, not the 2–3× single-stream marketing number)
- **k8s** → KServe `InferenceService` or llm-d + GAIE `InferencePool` with prefix-cache-aware EPP
- Quantization chosen by ③'s memory solve and validated by ④ (never quantize without re-proving)

You can `git commit` the output and delete us.

### ⑥ Cut over — risk removal
Migration router mirrors live traffic to the open model, scores it against the closed response in real time using ④'s graders, and holds traffic at 0% until the equivalence bar sustains. Then 1% → 5% → 25% → 100%, each gated. **Auto-rollback on quality regression, not just on 5xx** — that's the part nobody does. Post-cutover, hand routing to GAIE/llm-d and get out of the data path.

### ⑦ Guard — compounding
Open models ship weekly (Kimi K3 landed 2026-07-16). Watch releases → auto-run *your* eval set → open a PR-shaped proposal: *"GLM-5.3 beats your current model on 4 of 5 clusters, +$1.9K/mo saved, one regression in tool-calling. Promote?"* Also catches silent degradation from your own prompt changes.

---

## 5. Requirements

### Functional

| ID | Requirement | Epic |
|---|---|---|
| FR-1 | Drop-in OpenAI-compatible proxy; integration is a `base_url` change | 1 |
| FR-2 | Capture request/response/latency/tokens/cost/tool-calls with local-first PII redaction | 1 |
| FR-3 | Cost dashboard over captured traffic, by model/route/task cluster | 1 |
| FR-4 | Cluster prompts into task types; representative sampling; versioned eval set | 2 |
| FR-5 | Eval sets are exportable (promptfoo/Inspect AI formats) and human-editable | 2 |
| FR-6 | Hardware profiler: Apple unified memory, NVIDIA/AMD VRAM, CPU/RAM, k8s node inventory | 3 |
| FR-7 | Fit solver: weights + KV + activations vs. memory, per quant, at observed context lengths | 3 |
| FR-8 | Curated model catalog with license, params, arch, quant availability, capability scores | 3 |
| FR-9 | Run candidates against eval set; assertions + graders + pairwise judge | 4 |
| FR-10 | Equivalence matrix with per-cluster scores and confidence intervals | 4 |
| FR-11 | Regret analysis: named clusters where open loses | 4 |
| FR-12 | Generate native config: mlx, llama.cpp, vLLM, SGLang, KServe, llm-d + GAIE | 5 |
| FR-13 | Auto-configure speculative decoding (EAGLE-3/P-EAGLE/mlx spec) tuned to observed concurrency; disable it when observed batch exceeds the acceptance cliff | 5 |
| FR-13a | Auto-tune quantization, prefix/radix caching, tensor+pipeline parallelism, `max_model_len`, `max_num_seqs`, chunked prefill, KV dtype, and memory utilization — none of them prompted | 5 |
| FR-13b | **Measure, don't just compute:** benchmark each generated config against the observed workload and automatically revert any optimization that fails to help on this hardware | 5 |
| FR-13c | `--explain` every auto-tuned knob; emit the choices and the ratifying measurement as a header comment in the artifact | 5 |
| FR-20 | `onpar pack` produces a portable **box**: manifest, weights lock, per-target bindings, bench evidence, human-readable README | 5 |
| FR-21 | `onpar run <box>` serves an OpenAI-compatible endpoint on Linux (container), macOS (supervised native — Docker cannot reach the Apple GPU), and CPU | 5 |
| FR-22 | On arrival, re-profile the host, re-solve if it differs from the pack-time class, re-benchmark, revert unhelpful optimizations, and report every change | 5 |
| FR-23 | Published, CI-tested support matrix (architecture × runtime × platform); unsupported architectures fail with a clear message, never a crash | 5 |
| FR-14 | Shadow mode: mirror traffic, score, never serve | 6 |
| FR-15 | Quality-gated canary with auto-rollback on eval regression | 6 |
| FR-16 | Hybrid routing: per-cluster policy (open for these, closed for regret set) | 6 |
| FR-17 | Watch model releases; auto-re-eval; emit promotion proposals | 7 |
| FR-18 | MCP server exposing the loop as tools for Claude Code / Cursor | 8 |
| FR-19 | Everything driveable from CLI, non-interactively, in CI | 8 |

### Non-functional

| ID | Requirement |
|---|---|
| NFR-1 | Proxy overhead **< 15 ms p95** added latency (LiteLLM's baseline is 10–20 ms; we must not double it). **Measured: +0.07 ms p95** with capture on, release build, 200 interleaved requests per arm — `onpar-gateway/tests/latency.rs`. Loopback, so a lower bound. |
| NFR-2 | **Local-first.** No traffic, prompt, or eval leaves the machine unless explicitly exported. Default: zero telemetry. |
| NFR-3 | Captured traffic encrypted at rest; redaction runs before persistence, never after |
| NFR-4 | Generated artifacts are readable, idiomatic, and runnable **without onpar installed** — a receipt, not an interface. Reading or editing it is never a prerequisite. |
| NFR-5 | Cutover rollback completes in **< 10 s** |
| NFR-6 | Loop is resumable — kill it at any stage, resume from disk state |
| NFR-7 | Runs fully offline after model download (air-gapped enterprise) |
| NFR-8 | Every score traceable to the exact prompt, response, grader, and model revision that produced it |
| NFR-9 | Apache 2.0 core. Nothing about leaving lock-in should create lock-in. |

---

## 6. Technical assumptions

- **Language:** **Rust** for the datapath and weights path (gateway, router, fleet memory budgeting, download/convert, runtime supervision); **Python 3.13** for the control plane (distill, prove, guard) where the ML ecosystem lives. Bridged with PyO3. See [ADR-0007](adr/0007-tech-stack.md) — Rust wins on p95 tail (no GC) and explicit fleet-memory accounting.
- **Depend, don't build:** LiteLLM (transport), Inspect AI (eval runner — primary, given promptfoo is now OpenAI-owned), lm-eval-harness (capability screen), the engines themselves.
- **Runtime abstraction is non-negotiable** — dev is Metal, prod is CUDA. See [ADR-0002](adr/0002-runtime-abstraction.md).
- **State on disk, not in a service.** SQLite + files. A migration must survive a laptop reboot and be `git`-inspectable.
- **Agentic ≠ chat wrapper.** Each stage is a tool-using agent with a deterministic fallback. Anything a script can decide, a script decides — the model handles judgment, not arithmetic.

---

## 7. Epics

| # | Epic | Ships | Why now |
|---|---|---|---|
| **1** | **Observe** — proxy + capture + cost dashboard | Standalone value: "where is my LLM money going" | Trojan horse. Earns the install. |
| **2** | **Distill** — traffic → private eval set | The asset | Nothing downstream works without it |
| **3** | **Fit** — hardware profiler + solver + catalog | "What can I even run?" | Cheapest useful artifact; validates local story |
| **4** | **Prove** — equivalence matrix + regret | **The hero** | This is the product |
| **5** | **Deploy** — native config generation | Runnable stack | Converts decision to reality |
| **6** | **Cut over** — shadow/canary/rollback router | Risk removal | The reason they actually switch |
| **7** | **Guard** — drift watch + promotion proposals | Retention | Turns a one-shot tool into a subscription |
| **8** | **Surfaces** — CLI, TUI, local console, MCP | Reach | Parallel to 1–7, not after |

**Ship order for a demo that sells: 3 → 1 → 2 → 4.** Epic 3 alone (`onpar fit`) is a shareable free tool that builds audience while 1/2/4 are built.

---

## 8. Epic detail — Epic 3 (first build) & Epic 4 (the hero)

### Epic 3 — Fit

**3.1 Hardware profiler**
*As a builder, I want onpar to know my machine so I stop guessing what fits.*
AC: detects Apple Silicon unified memory / NVIDIA / AMD VRAM / CPU+RAM; reports usable-vs-total headroom; k8s mode enumerates node GPU inventory; JSON output; degrades gracefully with no accelerator.

**3.2 Model catalog**
*As a builder, I want a curated catalog with licenses so I don't hit a legal wall after integrating.*
AC: params (total + active for MoE), layers, KV heads, head dim, context, license (**MIT/Apache flagged distinctly from Modified-MIT/Llama-style caps**), available quants, provenance. Updatable without a release.

**3.3 Fit solver**
*As a builder, I want to know which models run on my hardware at my context length.*
AC: computes weights + KV cache + activation overhead per quant; solves max context and max concurrency; ranks feasible models; **explains the arithmetic** (NFR-8); flags "fits but will be slow."

**3.4 Runtime recommendation**
AC: given hardware + workload shape, recommends engine with a stated reason — e.g. *"M4 Max, 32 concurrent agent requests → mlx: ~27× aggregate throughput over llama.cpp at this concurrency despite lower single-stream."*

### Epic 4 — Prove

**4.1 Grader stack**
AC: three tiers — programmatic assertions (JSON schema, tool-call name+args, regex/format), task graders (exact match, BLEU-ish, unit-test execution for codegen), pairwise LLM judge with position-swap debiasing. Judge model is configurable and **must be disclosed** in every report.

**4.2 Equivalence run**
AC: candidate × eval-set matrix; deterministic seeds where the engine allows; per-cluster score with CI; cost + latency recorded alongside quality; resumable; parallel across candidates.

**4.3 Equivalence matrix view**
AC: cluster × model grid, color-coded by equivalence band; drill into any cell to see the actual prompt, both responses, and the grader's reasoning; exportable as a shareable report (this is the artifact Ravi sends his CTO).

**4.4 Regret analysis**
AC: cluster and *name* the failure modes ("long-context multi-file refactor", "nested tool-call chains"); recommend a hybrid policy; quantify residual closed-model spend under that policy.

---

## 9. Metrics

| Metric | Target (12 mo) | Why |
|---|---|---|
| Time to first equivalence matrix | **< 30 min** | The core promise |
| % of installs reaching stage ④ | > 40% | Loop leakage is the risk |
| **$ of closed-model spend migrated** | **the north star** | Only number that matters |
| Cutovers rolled back for quality | < 5% | If high, graders are wrong |
| Post-cutover retention @ 90d | > 85% | Proves it actually held |

---

## 10. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| **Judge quality is the whole product.** Bad graders → wrong recommendation → a broken production system and a dead company. | 🔴 critical | Three-tier stack; position-swap debiasing; human-in-loop sampling; publish judge agreement rates; shadow mode as the safety net before any traffic moves. |
| Closed providers cut prices and kill the ROI | 🟠 high | Data sovereignty, latency, and rate-limit freedom survive price cuts. Don't lead with cost alone. |
| Frontier gap re-widens; open models stop being good enough | 🟠 high | Hybrid routing is a first-class outcome, not a failure. "Move 70%" is a win. |
| We become a thin wrapper on LiteLLM + Inspect AI | 🟡 medium | Moat is stages ②④⑥⑦. If those are thin, there's no company. Invest there, commodity elsewhere. |
| promptfoo is OpenAI-owned; eval tooling has a conflicted vendor | 🟡 medium | Inspect AI (UK AISI) primary. Abstract the runner. |
| Engines move fast; generated config rots | 🟡 medium | Config generation is versioned per engine release and CI-tested against real engines. |
| Apple Silicon dev / CUDA prod divergence | 🟡 medium | ADR-0002. CI must exercise both. |

---

## 11. Open questions

1. **Do we ever sit in the production datapath, or exit after cutover?** Exiting is honest and reduces risk; staying is where recurring revenue usually lives. *Leaning: exit the datapath, stay in the control plane (⑦).*
2. Team-scale evals need shared traffic — does that force a server mode that breaks NFR-2? *Leaning: opt-in self-hosted server, never our cloud.*
3. Open-core boundary. *Leaning: loop is Apache 2.0; enterprise = SSO, audit, multi-cluster, drift-watch-as-a-service.*
4. Local-only means no aggregate cross-customer intelligence ("847 teams migrated this workload successfully"). That's a real product loss. Opt-in anonymized benchmark sharing?
