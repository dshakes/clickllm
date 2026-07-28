<div align="center">

<img src="docs/assets/loop-animated.svg" alt="Traffic flows through seven stages — observe, distill, fit, prove, deploy, cut over, guard — turning $2,847/mo of closed-model spend into $317/mo of proven open-model inference" width="100%">

# You don't have a model problem.<br>You have a proof problem.

**GLM-5.2 is MIT-licensed, 11× cheaper, and *probably* good enough.<br>
"Probably" is why you're still paying $18K a month.**

[![status](https://img.shields.io/badge/status-pre--alpha-22d3ee?style=flat-square)](docs/50-roadmap.md)
[![tests](https://img.shields.io/badge/tests-451-34d399?style=flat-square)](#verification)
[![license](https://img.shields.io/badge/license-Apache--2.0-a78bfa?style=flat-square)](LICENSE)
[![docs](https://img.shields.io/badge/docs-read-fbbf24?style=flat-square)](https://dshakes.github.io/clickllm/docs/)

**[Site](https://dshakes.github.io/clickllm/) · [Docs](https://dshakes.github.io/clickllm/docs/) · [Why](#why-this-exists) · [Quickstart](#try-it-in-ten-seconds) · [Roadmap](docs/50-roadmap.md)**

</div>

---

## Try it in ten seconds

```bash
uvx clickllm fit --context 32k --concurrency 8
```

```
  M4 Max · 16 cores · 128 GB · 546 GB/s          usable for inference: 96 GB

  model                     quant   weights      kv   total    free  ~tok/s  license
  ----------------------------------------------------------------------------------
  Qwen3 30B-A3B (MoE)       q8        28.4G   24.0G   56.2G   39.8G     119  Apache-2.0
  Phi-4 14B                 q8        13.7G   50.0G   66.3G   29.7G      27  MIT

  NOT FEASIBLE
  Kimi K3 (2.8T MoE)        weights alone need 1,467 GB at q4 — MoE sparsity
                            (50B of 2800B active) cuts compute, not memory
```

Then ask the inverse — *what would I need to run this?*

```bash
uvx clickllm where deepseek-v3 --context 16k
```

```
  hardware                  quant     total  ~tok/s    $/hr   $/Mtok
  ------------------------------------------------------------------
  8× NVIDIA H200            fp8      677.4G     649   28.80    12.32
  Apple M3 Ultra 512 GB     q4       382.1G      28       -        -

  WILL NOT RUN
  NVIDIA H100 80 GB SXM     weights alone need 352 GB at q4, 72 GB usable
```

Every number answers `--explain`, which prints the arithmetic that produced it.

---

## Why this exists

Ask any staff engineer whether an open model could handle most of their traffic and they'll say yes. Ask them to bet the product on it and they'll stop — correctly. **Nothing turns that instinct into evidence.**

| What you'd need | Closest tool | Why it stops short |
|---|---|---|
| A benchmark of *your* workload | promptfoo, Inspect AI | Both start from a test set you hand-write. Yours doesn't exist. |
| A verdict you'd act on | LM Studio side-by-side | One prompt at a time, judged by eye. |
| Your traffic, analysed | LiteLLM | Proxies every request. Does nothing with them. |
| A safe cutover | Any gateway | Canaries on **errors**. Not one gates on **quality**. |
| To stay current | — | You find out on Twitter, then redo the work by hand. |

<img src="docs/assets/gap-map.svg" alt="Bar chart: tools that solve each of six migration steps. Deployment has six; hardware fit has one; four steps have none." width="100%">

Everyone builds the one column that was already finished.

---

## The output is a decision, not a dashboard

```
REGRET — keep the incumbent for these:
  long-ctx refactor  (15% of traffic)  30% [22–40]

                        codegen  long-ctx refacto      rare-json
                          (60%)             (15%)          (25%)
  ──────────────────────────────────────────────────────────────
  glm-5.2           96% [90–98]       30% [22–40]  100% [44–100] ⚠   87% weighted
  ──────────────────────────────────────────────────────────────
  gpt-5 (incumbent)        100%              100%           100%

judge: claude-opus-5, position-swapped · human agreement 0.90 (n=40)
⚠ underpowered clusters (too few samples to conclude): rare-json
⚠ 2 items had no applicable grader and are excluded, not counted as passes

Move 60% of traffic to glm-5.2.
  Keep the incumbent for: long-ctx refactor
  Not yet proven (gather more evidence): rare-json
  Saving: $1,582/mo (56%) at zero measured quality loss
```

Four things there are deliberate, and each corrects how these comparisons usually get presented:

- **`100% [44–100] ⚠`** — a perfect score on three samples is not certainty. Wilson intervals, so a 95% gate can't open on noise.
- **Regret above the fold** — where the candidate loses is printed *first*. The honest failure is what makes the wins credible.
- **"Not yet proven" ≠ "regressed"** — thin evidence means *gather more*, not *give up*. Conflating them strands traffic on the incumbent forever.
- **No cost rate → no saving printed.** A fabricated saving is the most damaging number this report could contain.

---

## What runs today

| | Capability | |
|---|---|---|
| ① | **Observe** — capture, redaction that fails closed, encrypted store | ✅ |
| ② | **Distill** — structural clustering, representative sampling | ✅ |
| ③ | **Fit** — MoE/GQA/MLA-correct sizing, 17 hardware classes, `--explain` | ✅ |
| — | **Plan** — engine *and* flags derived from what the deployment is for | ✅ |
| — | **Intent** — a sentence in, a plan out; asks about what it cannot infer | ✅ |
| ④ | **Prove** — grader stack, position-swapped judge, equivalence matrix | ✅ |
| — | **Receipt** — a portable, reproducible proof you can hand to an auditor | ✅ |
| ⑤ | **Deploy** — native vLLM / SGLang / llm-d config, inference in a box | ✅ |
| ⑥ | **Gateway** — SSE streaming, metering, router, real shadow dispatch | ✅ |
| — | **Gate** — automatic rollback, human-gated advance, live control surface | ✅ |
| — | **Telemetry** — KV pressure, prefill/decode split, plan-vs-reality check | ✅ |
| ⑦ | **Guard** — model drift, traffic drift, re-prove proposals | ✅ |
| — | **Post-training** — distil from your own captured incumbent output | ✅ |
| — | **Surfaces** — CLI · MCP · Python SDK · agent skill · local console | ✅ |
| — | **TPU / host GPU stats** | 🔜 |

Full acceptance criteria and risk gates: **[implementation plan](docs/80-implementation-plan.md)**.

---

## Prove it, then move it

The motto is the control flow. Nothing skips a step, and each step can say *no*.

<img src="docs/assets/in-a-box.svg" alt="One artifact landing on three machines and producing three outcomes: run as packed, re-solved with the changes reported, or refused" width="100%">

**Proof is an artifact, not a dashboard.** A receipt is a file: every claim with
its confidence interval, the bar it was measured against, the judge and how much
it agreed with humans, and — required, never optional — the clusters that did
*not* pass. Re-run the same eval set and the digest must match, which is a
stronger claim than a signature. A signature says *we said this*; reproduction
says *and it is true*, and anyone holding the eval set can check it.

**Moving is asymmetric.** Rollback is automatic and deliberately easy to trigger.
Advancing is only ever a proposal — the gate says the evidence supports 25%, a
human moves it. The control surface re-derives that rule from the numbers alone,
so the automation can be wrong or bypassed and traffic still cannot escalate
unattended.

**Then it keeps checking.** The guard separates three things every other tool
collapses into one "stale" flag: the model changed behind its name (your proof is
void), your traffic moved (the eval set answers questions nobody asks now), or
something new was released (your proof is still true). Only the first two mean
you no longer know whether production is adequate.

---

## Three things everyone gets wrong

The docs teach the whole inference stack from first principles — [start here](https://dshakes.github.io/clickllm/docs/#edu-why) if you've never sized a KV cache. The short version:

**① MoE sizes on *total* parameters.** Kimi K3 activates 50B of 2.8T per token, so people assume it needs 50B of memory. All 2.8T must be resident. *Sparsity cuts compute, not memory.*

**② GQA uses `kv_heads`, not attention heads.** Using attention heads overestimates KV cache by up to 8×.

**③ MLA has a different formula entirely.** DeepSeek-family models compress K and V into one low-rank latent. Applying the GQA formula overestimates by ~50×.

And one that costs money in the other direction: **speculative decoding turns negative past batch ~32.** EAGLE-3's headline "2–3×" is a single-stream figure.

---

## Architecture

<img src="docs/assets/e2e.svg" alt="End-to-end: request path through the gateway and the control loop that decides what it may hit" width="100%">

Purple is the live request. Green is the control loop deciding what it's allowed to hit. **They never cross.**

**Rust** for the datapath and weights path — no GC pauses against a p95 budget, explicit accounting for GB-scale fleet memory. **Python** for the control plane, where the ML ecosystem lives. Reasoning and rejected alternatives in [ADR-0007](docs/adr/0007-tech-stack.md).

---

## Install

```bash
uvx clickllm fit                        # no install
pipx install clickllm                   # or
brew install dshakes/tap/clickllm       # or
docker run --rm ghcr.io/dshakes/clickllm:latest fit
curl -fsSL https://dshakes.github.io/clickllm/install.sh | sh
```

**Python SDK** — the same implementation the CLI and MCP server route through:

```python
from clickllm import sdk
report = sdk.fit(context="32k", concurrency=8)
report.best()                 # highest-capability candidate that isn't slow
report.commercially_clean()   # permissive licence AND verified architecture
```

**MCP server** — `clickllm-mcp`, JSON-RPC over stdio, zero dependencies. Deliberately read-only: an agent may analyse and recommend a migration; a human presses the button that moves production traffic.

---

## Verification

```bash
cargo test --all                                   # 229 Rust
cargo clippy --all-targets -- -D warnings
uv run --with pytest --python 3.13 pytest -q       # 222 Python
```

**451 tests.** The Rust core denies `unwrap`/`expect`/`panic!`/slice-indexing at the lint level — a sizing or licence bug must not be a panic. Gateway tests run over **real TCP** against a **real** upstream, because a test that calls the handler directly passes even when the response is buffered.

**Every engine flag is verified against published docs, never recalled.** That
rule exists because breaking it shipped a bug: `--guided-decoding-backend` had
been renamed in vLLM, so every structured-output config this repo generated was
unrunnable. Where a flag could not be confirmed — SGLang's grammar backend at the
time of writing — the adapter reports a gap rather than emitting a plausible
guess. A wrong flag fails loudly and costs an afternoon; a *right-looking* flag
with inverted meaning succeeds and quietly costs half your throughput.

Four defects caught by review or by rendering, all now regression tests:

- an SSE frame cap that was *detected* but never *enforced*
- shadow mirroring recorded and displayed without ever dispatching
- failover that would serve **unproven candidate output during shadow mode** —
  the phase whose entire contract is "scored, never served"
- concurrent capture appends interleaving into a log that decrypted as garbage

---

## Principles

1. **Kiosk outside, glass box inside.** One command; every number drillable to the raw prompt.
2. **Show the arithmetic.** `--explain` on everything.
3. **Lead with the regret.** Honest failures buy credibility for the wins.
4. **Never a number without its confidence.** `?` beats a fabricated score.
5. **Local-first, zero telemetry.** Your production prompts are the most sensitive data you have.
6. **No lock-in, by construction.** Eval sets export. Generated config runs standalone. *A product about escaping lock-in cannot create lock-in.*

---

## Docs

| | |
|---|---|
| [00 — Verdict](docs/00-verdict.md) | Build vs. buy, honestly. |
| [10 — Landscape](docs/10-landscape.md) | 14 competitors across 4 layers. |
| [20 — PRD](docs/20-prd.md) | Goals, personas, FR/NFR, epics, risks. |
| [30 — Architecture](docs/30-architecture.md) | Datapath/control-plane split, grader stack, fit math. |
| [40 — UX](docs/40-ux.md) | CLI, TUI, console, MCP. |
| [80 — Plan](docs/80-implementation-plan.md) | M0–M10, acceptance criteria, risk gates. |
| [ADRs](docs/adr/) | Eight decisions, including the two later reversed. |

## Prior art

**[LiteLLM](https://github.com/BerriAI/litellm)** provider transport · **[Inspect AI](https://inspect.aisi.org.uk)** eval runner · **[vLLM](https://github.com/vllm-project/vllm)** · **[SGLang](https://github.com/sgl-project/sglang)** · **[llama.cpp](https://github.com/ggerganov/llama.cpp)** · **[MLX](https://github.com/ml-explore/mlx)** · **[llm-d](https://llm-d.ai)** + **[GAIE](https://github.com/kubernetes-sigs/gateway-api-inference-extension)** · **[Ollama](https://ollama.com)**, the ergonomics bar everyone should be held to.

## License

Apache-2.0.
