# clickllm — pitch

---

## One-liner

**clickllm proves which open model can replace your closed one — on your traffic, your hardware, your budget — then moves you there without a risky cutover.**

Stripe made payments a `base_url`. We make *leaving OpenAI* a `base_url`.

---

## The problem

Open-weight models caught up in 2026. GLM-5.2 is MIT and scores 91.2 on GPQA Diamond. Kimi K3 leads Humanity's Last Exam. DeepSeek V4 Flash costs $0.11/M tokens.

**Almost nobody switches.**

Not because deployment is hard — Ollama solved that. Because a team paying $18K/month to Anthropic cannot answer one question:

> *Will this open model be good enough for **my** job?*

Public benchmarks don't answer it. MMLU is not your traffic. And the tools that *could* answer it won't:

- **LiteLLM** proxies every one of your requests and does nothing with them.
- **promptfoo / Inspect AI** evaluate a test set *you hand-write*. Yours doesn't exist.
- **LM Studio** compares two models side by side — one prompt at a time, eyeballed.
- **Every gateway** canaries on *errors*. None gates on *quality*.
- **OpenRouter, Fireworks, Together, Baseten** all profit from you not self-hosting.

So the migration never starts. The bill compounds. The dependency deepens.

---

## The solution

Point clickllm at your existing closed-model traffic. One `base_url` change.

It captures your real prompts, distills them into a **private eval set**, works out what your hardware can actually run, and produces the artifact nobody else produces:

```
                    codegen   tools   long-ctx   summar.   classify   ~cost
  GLM-5.2 Q4          96 ██   91 ██    62 ▓▒     99 ██     100 ██    $210/mo
  Kimi-K2.7-Code      98 ██   94 ██    71 ▓█     95 ██     100 ██    $180/mo
  gpt-5 (incumbent)  100      100      100       100       100     $2,847/mo

  ▸ REGRET: long-context refactor (11% of traffic) — keep gpt-5 for this.
    Hybrid saves $2,530/mo (89%) at zero quality loss.
```

Then it moves you: shadow traffic → quality-gated canary → cut, with rollback in under 10 seconds. And it keeps watching — when a better open model ships next week, your eval set re-runs automatically and you get a proposal, not a newsletter.

**Deployment is an output, not the product.**

---

## Why now

1. **Open models crossed the threshold** — mid-2026 is the first moment where "good enough for most production traffic" is broadly true, not aspirational.
2. **The serving layer finished.** vLLM, SGLang, llm-d, KServe, GAIE are mature and free. Five years ago this product would have had to build them; today it composes them.
3. **AI budgets are under review.** 2023–25 was land-grab spending. 2026 is the first year CFOs ask what the LLM line item buys.
4. **Model churn became continuous.** Kimi K3 shipped July 16. GLM-5.2, DeepSeek V4, MiniMax M3 within months. Manual re-evaluation cannot keep pace — which converts a one-time migration tool into a permanent control plane.
5. **Nobody incumbent can follow.** Every player positioned to build this makes money from you *not* switching. That's not a moat we built; it's one the market handed us.

---

## Business model

**Open core, Apache-2.0.** A product about escaping lock-in cannot create lock-in.

| Free forever | Paid |
|---|---|
| The whole Switch Loop, self-hosted | SSO, RBAC, audit trail |
| Local, zero telemetry | Multi-cluster / fleet management |
| Config generation, no restrictions | Drift-watch as a managed service |
| | Compliance reporting for regulated migrations |

Pricing anchors to **spend migrated**, not seats — we're worth a fraction of what we save. A team cutting $2.5K/month happily pays a few hundred for the thing that proved it was safe.

**North-star metric: dollars of closed-model spend migrated.** Not installs, not stars. It is the only number that means the product worked.

---

## Market

- **Wedge:** Series A–C engineering teams spending $5K–$100K/month on closed LLM APIs, with in-house infra ability and a CFO asking questions.
- **Expansion:** regulated enterprises under data-residency mandates — they *must* self-host and need an auditable record of the decision. Baseten's $300M Series E at $5B valuation on essentially "managed inference in your VPC" is the proof that this budget exists.
- **Bottom-up on-ramp:** solo builders on a Mac. 128 GB of unified memory is more than most GPUs. They convert to the paid tier when they get a job.

---

## Competition

Nobody is in this lane, and the reason is structural.

| | What they do | Why they can't do this |
|---|---|---|
| OpenRouter, Fireworks, Together | hosted inference | revenue dies if you self-host |
| Baseten, Modal | managed/VPC inference | you're still on their fleet |
| LiteLLM | gateway | has the traffic; acting on it isn't the business |
| promptfoo | evals | **acquired by OpenAI** — cannot credibly ship "leave OpenAI" |
| llm-d, KServe, vLLM | serving | assume you already decided and already believe |
| Ollama, LM Studio | local runners | answer "can I run this", never "should I" |
| **llmfit** (30.8k★) | hardware fit | best in class at it — **so we adopt it**, not compete |

The category isn't contested. It's unoccupied.

---

## Moat

Not the code. The code is composable and someone could rebuild it.

The moat is **data our customers generate and nobody else holds**: a private eval set distilled from their production traffic, plus the accumulated equivalence history behind every model they've tested. That compounds per customer and is worthless to a competitor.

Second-order: **switching costs run backwards.** Once ⑦ Guard is watching, we're the system of record for "which model should we be on" — a question that now has a new answer every few weeks.

---

## Status

**Pre-alpha, built in the open.**

- ✅ Full spec: [PRD](20-prd.md), [architecture](30-architecture.md), [UX](40-ux.md), [3 ADRs](adr/)
- ✅ Competitive research: 14 products across 4 layers
- ✅ Stage ③ (fit) shipping: hardware profiler, MoE/GQA/MLA-correct solver, runtime recommendation, 27 tests green, zero runtime deps
- 🔜 Stage ① (observe) — the on-ramp
- 🔜 Stage ④ (prove) — the hero

**Biggest open risk, stated plainly:** grader quality *is* the product. A wrong verdict breaks a customer's production system. Mitigations are designed in — three-tier grading, position-swap debiasing, published judge–human agreement rates, and shadow mode as the real gate before any traffic moves. If stage ④ can't produce verdicts people trust, there is no company, and we'd rather find that out in month 4 than month 24.

---

## The 30-second version

> Open models are good enough now. Nobody switches, because nobody can prove it for *their* workload — and every tool that could is paid by the incumbent. clickllm captures your real traffic, turns it into a private benchmark, tells you exactly which open model matches and which tasks still need the closed one, and moves your traffic across with a quality gate and a rollback button. Deployment is solved five times over; the migration is unowned. We're building the migration.
