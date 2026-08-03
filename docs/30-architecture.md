# Architecture

**Principle:** own the judgment, depend on the plumbing. Every box below is either *ours because it's the moat* or *someone else's because they're better at it*.

---

## System view

```
┌──────────────── your app (unchanged, base_url swapped) ─────────────────┐
└──────────────────────────────────┬──────────────────────────────────────┘
                                   ▼
              ┌────────── clickllm proxy (datapath) ──────────┐
              │  capture · redact · shadow · canary · gate    │   ← ours (thin)
              │  transport: LiteLLM adapters                  │   ← theirs
              └───────┬──────────────────────────┬────────────┘
                      ▼                          ▼
              closed provider            open model runtime
              (OpenAI/Anthropic)         (mlx/vLLM/SGLang/llm-d)
                      │                          │
                      └──────────┬───────────────┘
                                 ▼
        ┌──────────── control plane (off datapath) ────────────┐
        │                                                       │
        │  ┌─────────┐  ┌────────┐  ┌───────┐  ┌────────────┐  │
        │  │ DISTILL │→ │  FIT   │→ │ PROVE │→ │  GENERATE  │  │
        │  │ cluster │  │ solver │  │ judge │  │  config    │  │
        │  └─────────┘  └────────┘  └───────┘  └────────────┘  │
        │       ▲            ▲          ▲             │         │
        │       └────────────┴──────────┴─────────────┘         │
        │                    GUARD (drift watch)                │
        └───────────────────────────┬───────────────────────────┘
                                    ▼
                    ┌──────── store (local, on disk) ────────┐
                    │ SQLite: traffic, runs, scores          │
                    │ files: eval sets, reports, artifacts   │
                    └────────────────────────────────────────┘
                                    ▲
        surfaces ───────────────────┴───────────────────
        CLI  ·  TUI  ·  local web console  ·  MCP server
```

---

## Build vs. depend

| Component | Decision | Why |
|---|---|---|
| Provider transport | **LiteLLM** | 100+ providers, ~10–20 ms, solved |
| Eval runner | **Inspect AI** (primary) | UK AISI, agent-native. promptfoo is OpenAI-owned — conflicted for a product about leaving OpenAI |
| Capability screening | **lm-eval-harness** | The academic standard; use for catalog priors only |
| Inference engines | **all of them** | Never compete |
| Production routing | **GAIE / llm-d EPP** | KV-cache-aware, 3× tok/s, 2× TTFT — measured. Adopt. |
| K8s serving | **KServe / llm-d** | We *emit* their CRDs |
| — | — | — |
| Task clustering | **ours** | Moat |
| Fit solver | **ours** | Moat |
| Grader stack | **ours** | Moat — and the highest-risk component |
| Equivalence matrix | **ours** | Moat — the hero artifact |
| Migration router logic | **ours** | Moat — quality-gated canary is unowned |
| Config generation | **ours** | Moat — hardware-aware tuning is the differentiator |
| Drift watch | **ours** | Moat |

**Our datapath code is deliberately thin.** Everything expensive happens off-path in the control plane. NFR-1 (<15 ms) is only achievable if we resist putting judgment in the proxy.

---

## The runtime abstraction (the load-bearing decision)

Dev is Apple Metal. Prod is CUDA. They share nothing but the OpenAI wire format.

```python
class Runtime(Protocol):
    name: str
    def supports(self, hw: Hardware, model: ModelSpec) -> Feasibility: ...
    def plan(self, hw: Hardware, model: ModelSpec, wl: Workload) -> RuntimePlan: ...
    def render(self, plan: RuntimePlan, target: Target) -> list[Artifact]: ...
```

Implementations: `MlxRuntime`, `LlamaCppRuntime`, `OllamaRuntime`, `VllmCudaRuntime`, `SglangRuntime`, `LlmdRuntime`.

Three rules:
1. **`render` emits native config, never a wrapper.** BentoML's rebuild-push-redeploy friction and one-release upstream lag are the failure mode we're avoiding. The user gets a real `vllm serve` command or a real `InferencePool`.
2. **`supports` is honest.** Returning "yes, degraded" beats a runtime error on a customer's cluster.
3. **No engine type leaks above this Protocol.** The moment `PROVE` imports `vllm`, portability is gone.

---

## Data model (sketch)

```
Project            one app/workload boundary (cf. AnythingLLM workspaces)
 └ Capture         raw request/response, redacted, cost, latency, tool calls
     └ Cluster     task type — the unit everything else is measured in
         └ EvalItem   sampled prompt + incumbent baseline response
 HardwareProfile   detected target, versioned (hardware changes)
 ModelSpec         catalog entry: params, layers, kv_heads, head_dim, license, quants
 FitResult         model × hw × quant → feasible, max_ctx, max_concurrency, projected tok/s
 EvalRun           candidate × eval_set → scores, judge config, engine revision
 Equivalence       cluster × model → score, CI, regret notes
 Rollout           shadow/canary/cut state, gates, rollback history
```

**Cluster is the atomic unit.** Not "the model is 94% as good" — that's meaningless. *"The model is 96% on codegen and 62% on long-context refactor"* is actionable, and it's what makes hybrid routing and regret analysis possible.

---

## Fit solver math

```
weights_bytes  = params × bits_per_weight / 8
kv_bytes       = 2 × layers × kv_heads × head_dim × seq_len × dtype_bytes × batch
overhead       ≈ activations + fragmentation + runtime  (engine-specific factor)
required       = weights + kv + overhead
feasible       ⟺ required ≤ usable_memory
```

Notes that matter:
- **MoE** (Kimi K3: 2.8T total, ~16 of 896 experts active): weights need *total* params resident, compute scales with *active*. Getting this wrong is the most common sizing error in the wild.
- **GQA**: `kv_heads` ≪ `attn_heads`. Using attention heads overestimates KV by up to 8×.
- **Apple unified memory** is shared with the OS and GPU. Usable ≈ 75% of total, and `iogpu.wired_limit_mb` can raise it — surface that as a tunable, don't silently assume.
- Solve at **observed** context lengths from stage ②, not the model's advertised max. Nobody actually sends 1M tokens.

---

## Grader stack (highest-risk component)

Three tiers, cheapest first, escalating only when needed:

```
1. programmatic   JSON schema · tool name+args match · format/regex · exit code
                  → deterministic, free, catches most regressions
2. task graders   exact match · unit-test execution (codegen) · retrieval overlap
                  → cheap, high signal, cluster-specific
3. pairwise judge open vs. incumbent, position-swapped, judge model disclosed
                  → expensive, subjective, LAST RESORT
```

**Guardrails, because a wrong verdict here breaks a customer's production system:**
- Position-swap every pairwise comparison; disagreement between orderings = low confidence, flagged not averaged.
- Report judge–human agreement on a sampled subset. If we can't state it, we don't ship the verdict.
- Never let tier 3 alone authorize a cutover. Shadow mode (⑥) is the real gate — it scores against *live* traffic before anything is served.
- The judge model is disclosed in every report. Using a closed model to judge whether you can leave closed models is an obvious conflict; it must be visible and swappable.

---

## Security & privacy

- **Redaction before persistence.** PII detection runs in the proxy write path; unredacted text never touches disk. Redaction failures fail *closed* (drop the capture).
- **Local-first, zero telemetry default** (NFR-2). Ollama and Jan set this bar; LM Studio's opt-out default is the anti-pattern.
- **Encrypted at rest.** Captured traffic is the most sensitive data a customer has — it's their production prompts.
- **Captured content is data, never instructions.** Traffic is untrusted by definition; a prompt in the corpus that says "ignore previous instructions" is a row in a table, not a directive. This applies to the clustering agent, the judge, and every report renderer.
- **No egress without explicit action.** Export is a command, never a background sync.
- Air-gap support (NFR-7): everything after model download works offline.

---

## Surfaces

| Surface | For | Priority |
|---|---|---|
| **CLI** (`clickllm`) | primary; scriptable, CI-native, non-interactive flags on everything | P0 |
| **TUI** | live loop status, shadow-mode scoring, cutover gates | P1 |
| **Local web console** | equivalence matrix, drill-down, shareable report | P0 for stage ④ — the matrix needs pixels |
| **MCP server** | Claude Code / Cursor drive the loop conversationally | P1 |

Detail in [`40-ux.md`](40-ux.md).
