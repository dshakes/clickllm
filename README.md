<div align="center">

<img src="docs/assets/loop-animated.svg" alt="Traffic flows through seven stages — observe, distill, fit, prove, deploy, cut over, guard — turning $2,847/mo of closed-model spend into $317/mo of proven open-model inference" width="100%">

# Open models, served properly,<br>in one command.

### No config file. Ever. It reads your hardware, sizes the KV cache, picks the engine, sets the flags — and can show you the evidence that the model is good enough for *your* traffic.

```bash
clickllm run qwen3-30b-a3b
```

**Behind that one line: the KV cache sized without getting MoE, GQA or MLA
wrong; the right engine of seven chosen for the silicon you actually have; the
two dozen flags that matter set correctly; the weights resolved to a repo
confirmed to exist. You get an OpenAI-compatible endpoint and you never opened
an editor. When you need to know it is good enough — not on someone's
leaderboard, on your own captured requests — that is one more command, and it
answers per cluster with confidence intervals instead of a shrug.**

[![status](https://img.shields.io/badge/status-pre--alpha-22d3ee?style=flat-square)](docs/50-roadmap.md)
[![tests](https://img.shields.io/badge/tests-1519-34d399?style=flat-square)](#verification)
[![license](https://img.shields.io/badge/license-Apache--2.0-a78bfa?style=flat-square)](LICENSE)
[![docs](https://img.shields.io/badge/docs-read-fbbf24?style=flat-square)](https://dshakes.github.io/clickllm/docs/)

**[Site](https://dshakes.github.io/clickllm/) · [Docs](https://dshakes.github.io/clickllm/docs/) · [Agent-first](#agent-first-by-construction) · [Why](#why-this-exists) · [Quickstart](#try-it-in-ten-seconds) · [Roadmap](docs/50-roadmap.md)**

</div>

---

## Two things, done properly

**1 — Deploy open models without the project.** Between "the weights are on
Hugging Face" and "it is serving" sits a specialist skill: KV cache arithmetic
that goes wrong three different ways, seven engines with incompatible flag
dialects, quantisation that means something different on MLX than on vLLM,
memory maths that must saturate rather than silently wrap. clickllm does that
and prints the arithmetic, so you can check it rather than trust it.

```
  M4 Max · 16 cores · 128 GB · 546 GB/s          usable for inference: 96 GB

  model                     quant   weights      kv   total    free  ~tok/s  license
  Qwen3 30B-A3B (MoE)       q8        28.4G   24.0G   56.2G   39.8G     119  Apache-2.0
  Llama 3.1 8B              q8         7.5G   32.0G   41.6G   54.4G      49  Llama 3.1 !
```

It refuses rather than guesses. A model that will not fit says how far short and
in what unit. A flag the installed engine does not accept is a refusal, not a
command that fails on start-up. A repo it has not confirmed exists is never
printed as though it does.

**2 — Know where open is good enough.** Benchmarks are someone else's exam. The
model that tops MMLU may fail your extraction schema, and the one ranked
fortieth may be perfect at your four tasks. So the comparison runs on *your*
captured requests, per kind of task, against the closed model you are using
today:

```
                    Arithmetic  Ticket classific  Structured extra  One-line summari
llama-3.1-8b      98% [87–100]       90% [77–96]     100% [91–100]      98% [91–100]
gpt-4o-mini               100%              100%              100%              100%
```

Structured extraction is a solved problem for an 8B model you host — 40 of 40,
whole interval above the bar. Classification is not: 90% looks fine and the
interval says 77–96, which is not a number to run production on. Arithmetic
scored 98% and still does not clear, because 40 items is not enough evidence at
that rate — and the report says how many would be.

That is the whole answer: **which of your tasks an open model already does, with
a number you can defend.** What you do with it is yours.

**Why the interval and not the average.** 90% over 20 items and 90% over 400 are
the same number and completely different decisions. Every cell is a Wilson score
interval, and a task counts as proven only when its *whole* interval clears the
bar — so a small sample cannot promote itself by getting lucky. At a perfect
score you need 35 flawless items to clear 90%, and no fewer, however clean 12
looks.

**Why now.** The quality gap has closed — about 17.5 points of MMLU between the
best closed and best open model at the end of 2023, effectively zero on
knowledge benchmarks by 2026, with cost still running 6–62x apart. Open weights
stopped being the compromise and became the default for everything that does not
specifically need a frontier model. The remaining problem is not whether they
are good enough. It is that running them well is still a specialist skill, and
knowing *where* they are good enough is still guesswork.

<sub>Sources: benchmark convergence and the 6–62x cost spread — published 2026
comparisons of the Artificial Analysis index against provider pricing.
Promptfoo/OpenAI — announced 9 March 2026 by both parties with a commitment to
keep the project open source. Third-party facts, dated on purpose: if one has
changed since, this paragraph is wrong.</sub>

## What it is not

Naming this is faster than a feature matrix.

| | |
|---|---|
| **Not an inference engine.** | vLLM, SGLang and MLX exist and are excellent. clickllm chooses among them and configures them; it never competes with them. |
| **Not an eval platform.** | Braintrust and LangSmith watch production after you ship. clickllm answers one question before you ship, then gets out of the way. Promptfoo is the closest thing to this and it is good software — it was also acquired by OpenAI in March 2026. clickllm is independent and Apache-2.0, and has no model to sell you. |
| **Not hosted inference.** | Nothing runs on our machines. There is no account, no telemetry, and no egress you did not ask for. |
| **Not a router or a proxy.** | Nothing sits in your request path. clickllm sizes, configures, launches and measures; where the traffic goes afterwards is your call and your infrastructure. A proxy that must be up for your app to work is a liability you did not have before. |

The join between those categories is the product: **your traffic → which model →
will it fit → what it costs → is it good enough → deploy → roll back.** Nothing
else walks that whole line.

## Try it in ten seconds

> **The distribution is `clickllm-cli`; the command is `clickllm`.** PyPI refused the
> bare name as too similar to the existing `click-llm`, so `pip install clickllm` and
> `uvx clickllm` both fail — the `--from` is not optional. `uvx` runs it without
> installing anything. The bare `clickllm …` commands further down assume it is on
> your PATH; [Install](#install) puts it there.

```bash
uvx --from clickllm-cli clickllm fit --context 32k --concurrency 8
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
uvx --from clickllm-cli clickllm where deepseek-v3 --context 16k
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

Then run it. One command, no config file, no login:

```bash
clickllm run llama-3.1-8b --quant q4
```

```
  MODEL     Llama 3.1 8B @ q4   (22.0 GB of 96.0 GB usable)
  WEIGHTS   mlx-community/Llama-3.1-8B-Instruct-4bit   (confirmed; 4 candidates checked)
  ENGINE    mlx   Apple silicon: the CUDA engines cannot run here at all
  ENDPOINT  http://127.0.0.1:8000/v1
  SPEED     ~45 tok/s single-stream   (roofline estimate, not measured)

  NOT EXPRESSED
    · context_length: mlx_lm.server takes no context-length flag
    · quantization: mlx bakes precision into the weights — serve a -q4 repo
```

The weights repo is **confirmed to exist before it is used**, never constructed from a
pattern: `mlx-community/Llama-3.1-8B-Instruct-8bit` is a 404 while the 4-bit repo of the
same name is real, and the 8-bit one carries a `Meta-` prefix the base repo never had. A
name we have not checked is a name we do not print.

`NOT EXPRESSED` is the part most tools omit. Every flag emitted is verified against the
installed engine's own `--help`; anything the planner wanted that the engine cannot say
is reported rather than silently dropped or guessed at.

When it does not fit your machine, that is a routing decision, not a dead end:

```bash
clickllm host deepseek-v3 --context 128k
```

```
  provider              shape                quant     total  ~tok/s     $/hr   $/Mtok
  ------------------------------------------------------------------------------------
  Hugging Face Endpts   8x H200 1128 GB      fp8      706.9G     649   $40.00   $17.12

  NOT AVAILABLE
  Hugging Face Spaces   FREE TIER — 86 GB usable, needs 412 GB. Short by 325 GB
  RunPod (Pods)         largest shape is B200 180 GB. Short by 239 GB
  Google Colab (free)   excluded by Colab's own terms, not by its memory: the free
                        tier disallows "web service offerings not related to
                        interactive compute" — a tunnelled endpoint is that shape

  Prices read 2026-07-28 from each provider's own page. They move. Re-read the
  source before committing spend — nothing here checks it for you.
```

Free tiers are ranked first and excluded **with a reason**. Prices carry the date and
the URL they came from; an unpublished rate renders as `?` rather than a guess. clickllm
never touches your credentials — it writes the deploy artifact and prints the command
for you to run.

---

## Or just say what you're building

```bash
clickllm build "coding assistant for about 20 engineers, needs to feel snappy"
```
```
  Qwen3 30B-A3B (MoE) at q8 on M4 Max — 44.2 GB of 96.0 GB, 7 candidates fit. Engine: mlx.

  understood:
    · workload = interactive   (from "assistant")
    · concurrency = 4   (from "20 engineers — about a fifth in flight at once,
                         since people read and think between requests")

  assuming:
    · context = 32768

  ? How long are the prompts — a few thousand tokens, or tens of thousands?
```

It detects the machine, sizes it, picks the model, chooses the engine, critiques
its own plan, and ends with a command plus the eval to run against it.

**It asks one question, and only when the answer would change the deployment.**
That is computed, not curated — it re-plans under each candidate answer and stays
silent when they agree. A form that asks about concurrency when every candidate
fits at every concurrency is spending the only thing you brought: attention.

**It never blocks.** There is a usable answer before you answer anything, with
the assumptions listed. Questions refine; they do not gate. And when nothing
fits, it says what *would* — this context at that concurrency, or a machine that
is genuinely bigger on the same measure and actually solves the problem.

`--save` and `--resume` carry a session across restarts. An agent drives the
same flow over MCP by passing `state` back each turn.

---

## Agent-first, by construction

Most tools ship a CLI and bolt on an agent wrapper. Here the CLI, the MCP server,
the Python SDK and the agent skill are **four faces of one implementation** — an
agent gets the same answer you do, with the same arithmetic attached, because it
is calling the same code.

```jsonc
// clickllm-mcp — JSON-RPC over stdio, zero dependencies
clickllm_fit       // what runs on this machine, at this context and concurrency
clickllm_explain   // the full arithmetic behind one verdict — weights, KV, headroom
clickllm_prove     // run the eval suite: verdict, traffic split, and a receipt
clickllm_advise    // what to change unprompted, and where production diverged
clickllm_build     // the whole flow, multi-turn: pass state back to continue
clickllm_catalog   // parameters, MoE split, context, licence
```

**The read-only boundary is a test, not a promise.** The suite asserts that no
exposed tool name contains `cutover`, `apply`, `promote`, `advance`, `rollout`,
`deploy`, `serve` or `route` — so an agent can size, compare, *run the whole
evaluation* and recommend a migration, and **structurally cannot** be the thing
that moves production traffic. Add such a tool and the build fails. The
vocabulary is deliberately broad: the failure it guards against is someone
adding a helpful-looking `clickllm_promote` and nothing objecting.

**It tells you what you didn't ask.** A planner that only answers the question
asked is a form. Somebody deploys an agent fleet with a 2,000-token system prompt
on every request, never sets a prefix-sharing figure because no form demanded one,
and pays to recompute the same prefix a million times. The plan was correct — and
40% more expensive than it needed to be, with nothing saying so.

```bash
clickllm advise --context 128k --concurrency 16 --seen-concurrency 40
```
```
  PRODUCTION DIVERGED FROM THE PLAN

  [high] Re-plan for concurrency 40.
    because: the plan assumes 16 in-flight requests; production is running 40.
             KV cache, batch limits and the speculative decision were all
             derived from the smaller number.
    expect:  settings that match the traffic.
```

Feed it real telemetry and it reconciles what production *did* against what the
plan *assumed* — the self-healing seam. Every item carries the observation that
triggered it, so a wrong suggestion is dismissible rather than mysterious, and
every effect is labelled an estimate. It proposes; it never applies.

**You never write a config.** Not a YAML you fill in, not a template you fork —
you state the intent and it derives the engine and the flags. What it emits is a
real `vllm serve` or a real `InferencePool` that **runs with clickllm uninstalled**.
Abstraction that would trap you isn't abstraction, it's a dependency.

**Nothing leaves the machine.** Zero telemetry, zero egress, no account. Your
production prompts are the most sensitive data you have; export is a command you
run, never a sync that happens.

---

## Why this exists

The ecosystem is excellent at the last mile and absent at the first. vLLM and SGLang
serve brilliantly *once you know what to serve, on what, with which flags* — and
every one of those is a research task the docs assume you've already done. So the
work falls to whoever on the team has time, and it gets done from half-remembered
blog posts.

Ask any staff engineer whether an open model could handle most of their traffic and
they'll say yes. Ask them to bet the product on it and they'll stop — correctly.
**Nothing turns that instinct into evidence.**

| What you'd need | Closest tool | Why it stops short |
|---|---|---|
| To know what your hardware can run | spreadsheets, vibes | MoE, GQA and MLA each size differently; getting MLA wrong overestimates by ~50×. |
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

That output is one command over an eval set you never wrote:

```bash
clickllm prove evalset.json --candidate glm-5.2 --incumbent gpt-5 --out receipt.json
```

The eval set comes from your own captured traffic, clustered by task shape — which
is the whole reason it exists. Every other tool starts from a test set you author,
and nobody has one, because writing it *is* the work you were trying to avoid.

Four things there are deliberate, and each corrects how these comparisons usually get presented:

- **`100% [44–100] ⚠`** — a perfect score on three samples is not certainty. Wilson intervals, so a 95% gate can't open on noise.
- **Regret above the fold** — where the candidate loses is printed *first*. The honest failure is what makes the wins credible.
- **"Not yet proven" ≠ "regressed"** — thin evidence means *gather more*, not *give up*. Conflating them strands traffic on the incumbent forever.
- **No cost rate → no saving printed.** A fabricated saving is the most damaging number this report could contain.

And two the suite enforces underneath:

- **35 clean samples before a cluster can move.** A 0.90 bar needs the whole Wilson
  interval above it, and 34 perfect items do not clear it. The exact boundary is
  pinned by a test, because it is the number that decides whether traffic moves.
- **The judge is the last resort, not the first.** Deterministic graders run first,
  and an item they *disqualify* never reaches the judge — paying a model to
  re-confirm that malformed JSON is malformed is spend with no information in it.
  It also means a judge outage costs you graded items, not the run.

---

## What runs today

| | Capability | |
|---|---|---|
| ① | **Observe** — capture, redaction that fails closed, encrypted store | ✅ |
| ② | **Distill** — structural clustering, representative sampling | ✅ |
| ③ | **Fit** — MoE/GQA/MLA-correct sizing, 17 hardware classes, `--explain` | ✅ |
| — | **Plan** — engine *and* flags derived from what the deployment is for | ✅ |
| — | **Advise** — `clickllm advise`: what to change unprompted, and drift against real telemetry | ✅ |
| — | **Intent** — a sentence in, a plan out; asks about what it cannot infer | ✅ |
| — | **Build** — `clickllm build`: the whole flow multi-turn, resumable, agent-drivable | ✅ |
| ④ | **Prove** — `clickllm prove`: grader stack, position-swapped judge, equivalence matrix | ✅ |
| — | **Receipt** — a portable, reproducible proof you can hand to an auditor | ✅ |
| ⑤ | **Deploy** — native vLLM / SGLang / llm-d config, standalone by construction | ✅ |
| — | **Run** — `clickllm run`: resolve weights, start the engine, hand back an endpoint | ✅ |
| — | **Box** — `clickllm box`: ADR-0005's OCI artifact — manifest, weights lock, per-target launch specs | ✅ |
| — | **Host** — `clickllm host`: cost-ranked external hosting when the machine cannot fit it | ✅ |
| — | **Cache** — `clickllm cache`: budgeted weight cache with pinning, so a sweep cannot evict the incumbent | ✅ |
| ⑥ | **Gateway** — SSE streaming, metering, router, real shadow dispatch | ✅ |
| — | **Gate** — automatic rollback, human-gated advance, live control surface | ✅ |
| — | **Telemetry** — KV pressure, prefill/decode split, plan-vs-reality check | ✅ |
| ⑦ | **Guard** — model drift, traffic drift, re-prove proposals | ✅ |
| — | **Post-training** — distil from your own captured incumbent output | ✅ |
| — | **Surfaces** — CLI · MCP · Python SDK · agent skill · local console · native launcher | ✅ |
| — | **Targets** — systemd unit · `docker run` · Kubernetes Deployment, each standalone | ✅ |
| — | **Silicon** — NVIDIA · AMD · Apple · **TPU v5e/v6e/v5p**, sized per host | ✅ |
| — | **Host stats** — foreign GPU memory the engine cannot see | ✅ |
| — | **Kernel seam** — scaffold a vLLM plugin, and a plan that *proves* it helped | ✅ |

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

### Why any of this matters, on the silicon

<img src="docs/assets/edu-silicon.svg" alt="An H100 die with 132 SM squares, one lit, fed by a memory bus carrying all 32.8 GB of weights for every token; the arithmetic beside it — 65.6 GFLOP per token, 2.00 FLOP per byte read against the 295 the chip needs to break even; and a memory budget showing 18 concurrent sequences fitting in 72.0 GiB with 1.51 GiB spare while a 19th goes 0.49 GiB over and is refused" width="100%">

A decode step reads every weight once and does two operations with each, per sequence in the batch.
An H100 balances compute and memory at 295 operations per byte — so saturating its tensor cores
needs a batch of ~295. **At batch 1 you are using 0.3% of them: one lit square out of 132.**

Batching buys that back, and then stops: the KV cache fills the HBM at batch 18, or 6.1% of peak.
The other 94% is not idle because nobody batched hard enough — it is **unreachable** until something
gives up memory. That is why "will it fit" and "will it be fast" are the same question, and why
getting the three formulas above wrong costs hardware rather than just accuracy.

---

## Architecture

<img src="docs/assets/e2e.svg" alt="End-to-end: request path through the gateway and the control loop that decides what it may hit" width="100%">

Purple is the live request. Green is the control loop deciding what it's allowed to hit. **They never cross.**

**Rust** for the datapath — no GC pauses against a p95 budget, explicit accounting for GB-scale fleet memory. **Python** for the control plane, where the ML ecosystem lives. Reasoning and rejected alternatives in [ADR-0007](docs/adr/0007-tech-stack.md).

Weights are **not** on either path. The serving engines fetch them from the Hub
themselves, so clickllm resolves *which* repo and then gets out of the way —
`clickllm cache` manages the cache they fill rather than keeping a second one.
That reversed an earlier decision; [ADR-0010](docs/adr/0010-retire-the-weights-crate.md) records why.

---

## Install

```bash
uvx --from clickllm-cli clickllm fit      # run it, install nothing
npx clickllm fit                          # same, if node is what you have
uv tool install clickllm-cli              # or put clickllm on your PATH
pipx install clickllm-cli                 # same, via pipx
pip install clickllm-cli                  # into the current environment
```

All five give you the `clickllm` command. **The distribution is `clickllm-cli`, the
command is `clickllm`**, and that split is not cosmetic: PyPI refused the bare name as
too similar to the existing `click-llm`, so `pip install clickllm` and `uvx clickllm`
will never work. `--from` is what bridges the two names, which is why every `uvx` line
here carries one.

**`npx clickllm` works.** npm allowed the bare name PyPI refused, so there the package
*and* the command are both `clickllm` — three names in total. The npm package is a shim
rather than a second implementation: it execs `uvx --from clickllm-cli==<version>
clickllm`, falling back to `uv tool run` then `pipx run`. A Python runner is still
required underneath, so `npx` saves you naming the distribution, not installing Python.
The `==` is exact on purpose: `npx clickllm@0.1.9` runs `clickllm-cli` 0.1.9 and nothing
else, so the two registries cannot drift apart under you.

### Versions

The commands above are unpinned and fetch the newest release — currently **0.1.9**. Pin
when you need a build to stay put:

```bash
uvx --from clickllm-cli==0.1.9 clickllm fit   # exactly this build
npx clickllm@0.1.9 fit                        # same build, via npm
clickllm version                              # what you have, and where it came from
clickllm upgrade                              # how to move, for the way you installed it
```

`clickllm version` reads the installed metadata rather than a string someone typed —
through 0.1.4 the receipts it writes were stamped `0.1.0`, because that literal had been
hand-written once and never moved. Fixed in 0.1.5, so a receipt now names the build that
produced it.

There is no Homebrew formula — `dshakes/homebrew-tap` carries `compass.rb`, `distil.rb`
and `firstpass-proxy.rb` and nothing for clickllm. `tests/test_docs_lab.py` fails the
build if these docs ever name a package we have not actually published.

**Native launcher** — `clickllm desktop install` writes a real `.app` on macOS or a
`.desktop` entry on Linux. It launches `clickllm ui` rather than reimplementing it,
binds loopback only, and reopens the running instance instead of clashing if you
double-click twice. `clickllm desktop uninstall` removes it.

**Python SDK** — the same implementation the CLI and MCP server route through:

```python
from clickllm import sdk
report = sdk.fit(context="32k", concurrency=8)
report.best()                 # highest-capability candidate that isn't slow
report.commercially_clean()   # permissive licence AND verified architecture

result = sdk.prove(items, shares=shares, issued="2026-07-27")
result.policy.moved_share     # 0.75 — what the evidence supports moving
result.policy.regret_clusters # ('rare-json',) — where it loses, always named
result.receipt.digest()       # reproducible: same eval set, same digest
```

**MCP server** — `clickllm-mcp`, JSON-RPC over stdio, zero dependencies. Deliberately read-only: an agent may analyse and recommend a migration; a human presses the button that moves production traffic.

---

## Verification

```bash
cargo test --all                                   # 227 Rust
cargo clippy --all-targets -- -D warnings
uv run --with pytest --with pyyaml --python 3.13 pytest -q   # 1292 Python
```

**1,519 tests.** 1292 Python, 227 Rust. Ten of the Python tests skip on a bare
machine: eight exercise the PyO3 bridge (`maturin develop` in `clickllm-py/`
turns them on), and two ask vLLM and SGLang for their own flags, which needs
those engines installed. CI runs both inside the engines' published images, so
neither skip reaches a green tick unasked. The Rust core denies `unwrap`/`expect`/`panic!`/slice-indexing at the lint level — a sizing or licence bug must not be a panic. Gateway tests run over **real TCP** against a **real** upstream, because a test that calls the handler directly passes even when the response is buffered.

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
