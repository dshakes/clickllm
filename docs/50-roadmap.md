# Roadmap

Sequenced so each phase ships something **independently useful** — no phase is scaffolding for a later phase.

---

## Phase 0 — `clickllm fit` (weeks 1–3) ← *in progress*

**Ships:** a free, zero-config CLI that answers *"what can I actually run on this machine?"*

- Hardware profiler (Apple unified / NVIDIA / AMD / CPU)
- Model catalog with license flags and MoE-correct sizing
- Fit solver with `--explain`
- Runtime recommendation

**Why first:** smallest thing with real standalone value. Correct MoE and GQA math alone beats most advice online. It's shareable, it builds the audience, and it validates the hardware layer everything else depends on. Nothing else in the loop needs to exist for this to be useful.

**Done when:** a stranger with an M-series Mac or a 4090 runs one command and gets a correct, auditable answer.

---

## Phase 1 — Observe (weeks 4–7)

**Ships:** drop-in proxy + capture + cost dashboard. Useful even if you never switch.

- LiteLLM-backed OpenAI-compatible proxy
- Capture with **redaction-before-persistence, failing closed**
- Cost/latency/volume dashboard by model and route
- `< 15ms` p95 overhead verified under load (NFR-1)

**Done when:** someone leaves it running for a week because the cost dashboard alone is worth it.

---

## Phase 2 — Distill (weeks 8–11)

**Ships:** production traffic → private eval set.

- Task clustering (embedding + structural features)
- Representative sampling per cluster
- Versioned eval sets, human-editable, exportable to Inspect AI / promptfoo

**Risk:** clustering quality is subjective. Ship with manual cluster editing from day one — do not pretend the algorithm is right.

---

## Phase 3 — Prove (weeks 12–18) — **the hero**

**Ships:** the equivalence matrix + regret analysis + web console.

- Three-tier grader stack
- Position-swapped pairwise judging with disclosed agreement rates
- Equivalence matrix view with full drill-down
- Regret clustering + hybrid policy recommendation
- Shareable exported report

**Longest phase, and correctly so.** This is the product. Everything before it is setup; everything after is execution. Budget for grader calibration to take longer than the code.

**Done when:** a team makes a real migration decision from the report and it holds in production.

---

## Phase 4 — Deploy / "inference in a box" (weeks 19–25)

**Ships:** `clickllm pack` + `clickllm run` — a portable box that serves an OpenAI-compatible endpoint on any hardware, plus native config generation for every target.

- Local: vllm-mlx, llama.cpp Metal, MLC-LLM, mlx-lm
- GPU: vLLM + EAGLE-3/P-EAGLE, SGLang + RadixAttention
- K8s: KServe `InferenceService`, llm-d + GAIE `InferencePool`
- Spec-decode and quantization tuned from Phase 0's solve and Phase 3's proof
- CI that runs generated configs against real engines per release

- **Zero-config by default** — every knob auto-tuned from ③ hardware and ② observed traffic; no tuning flag is ever required ([ADR-0004](adr/0004-zero-config-deployment.md))
- **Benchmark-and-revert** — start the engine, run the observed workload shape, and drop any optimization that doesn't help on this hardware
- **The box** — one command, one endpoint, two delivery mechanisms: container on Linux/CUDA/ROCm/k8s, supervised native on macOS/Metal ([ADR-0005](adr/0005-inference-in-a-box.md))
- **Re-tune on arrival** — a box tuned on an A100 must re-solve when it lands on an L40S or a Mac, not apply stale settings

**Constraint:** generated artifacts must run with clickllm uninstalled (NFR-4). Test that explicitly.

**Risk:** a wrong auto-tune is worse than no auto-tune — an untuned `num_speculative_tokens` at high concurrency makes users *slower* and they never learn why. Roofline estimates pick candidate settings; **measurement ratifies them**. Never ship a computed optimization unmeasured.

---

## Phase 5 — Cut over (weeks 24–30)

**Ships:** the risk-removal layer.

- Shadow mode: mirror + score + serve nothing
- Quality-gated canary with auto-rollback on **eval** regression
- Hybrid per-cluster routing (open here, closed for the regret set)
- TUI cockpit; rollback < 10s
- Hand off to GAIE/llm-d post-cutover and exit the datapath

---

## Phase 6 — Guard (weeks 31+)

**Ships:** the reason it's a subscription and not a one-shot tool.

- Model release watcher
- Auto re-eval on your set
- Promotion proposals with cost/quality delta
- Regression detection on your own prompt changes

---

## Parallel: surfaces

MCP server and TUI are built alongside, not after. The CLI is P0 from Phase 0. The web console lands with Phase 3 because that's the first output that needs pixels.

---

## Sequencing logic

```
fit ──────────────────────────────────────────►  standalone, free, shareable
      observe ────────────────────────────────►  standalone, cost value
              distill ────────────────────────►  needs observe
                      PROVE ──────────────────►  needs distill + fit   ★ hero
                            deploy ───────────►  needs prove (don't deploy unproven)
                                   cutover ───►  needs deploy
                                           guard  needs prove
```

Two independent on-ramps (`fit`, `observe`) converge on the hero. If Phase 3 fails to produce trustworthy verdicts, **stop** — there is no company without it, and shipping Phase 4+ on bad graders actively harms users.

---

## What we are explicitly not building

Restating, because scope creep here is fatal: no inference engine, no production load balancer, no chat UI, no RAG framework, no fine-tuning, no hosted inference. Every one of these has a well-funded incumbent and none of them is the gap.
