<div align="center">

<img src="docs/assets/loop-animated.svg" alt="Traffic flows through seven stages — observe, distill, fit, prove, deploy, cut over, guard — turning $2,847/mo of closed-model spend into $317/mo of proven open-model inference" width="100%">

# You don't have a model problem.<br>You have a proof problem.

**GLM-5.2 is MIT-licensed, 11× cheaper, and *probably* good enough.<br>
"Probably" is why you're still paying $18K a month.**

[![status](https://img.shields.io/badge/status-pre--alpha-22d3ee?style=flat-square)](docs/50-roadmap.md)
[![tests](https://img.shields.io/badge/tests-163-34d399?style=flat-square)](#verification)
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
| ③ | **Fit** — MoE/GQA/MLA-correct sizing, runtime recommendation, `--explain` | ✅ |
| — | **Where** — 17 hardware classes, cost per Mtok, TP-aware bandwidth | ✅ |
| — | **Licence gate** — refuses before bytes move; unknown licences fail closed | ✅ |
| ② | **Distill** — structural clustering, representative sampling | ✅ |
| ④ | **Prove** — grader stack, position-swapped judge, equivalence matrix | ✅ |
| ⑥ | **Gateway** — SSE streaming, token metering, router, real shadow dispatch | ✅ |
| — | **Surfaces** — CLI · MCP · Python SDK · agent skill · local console | ✅ |
| ① | **Observe** — capture and redaction | 🔜 |
| ⑤ | **Deploy** — native config generation, inference in a box | 🔜 |
| ⑦ | **Guard** — drift watch, promotion proposals | 🔜 |

Full acceptance criteria and risk gates: **[implementation plan](docs/80-implementation-plan.md)**.

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
cargo test --all                                   # 110 Rust
cargo clippy --all-targets -- -D warnings
uv run --with pytest --python 3.13 pytest -q       # 53 Python
```

**163 tests.** The Rust core denies `unwrap`/`expect`/`panic!`/slice-indexing at the lint level — a sizing or licence bug must not be a panic. The gateway's streaming tests run over **real TCP** against a **real** upstream, because a test that calls the handler directly passes even when the response is buffered.

Two defects that shipped and were caught by review, both now regression tests: an SSE frame cap that was *detected* but never *enforced*, and shadow mirroring that was recorded and displayed without ever dispatching to the candidate.

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
