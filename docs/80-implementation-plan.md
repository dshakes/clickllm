# Implementation plan

How all five ADRs and every named capability actually get built. Milestones are ordered so each one **ships something usable** and each one **de-risks the next**.

Traceability matrix at the bottom — every capability you've named maps to a milestone. Nothing is dropped.

---

## Stack

**Rust** for the datapath and weights path · **Python** for the control plane · PyO3 between them ([ADR-0007](adr/0007-tech-stack.md)). **No third-party fit dependency** — the whole stack is ours ([ADR-0008](adr/0008-build-from-scratch.md)).

## Package map

```
clickllm-core/          RUST — the datapath. No third-party fit dependency.
  src/weights/       M1  acquire · convert · cache · licence     (mmap, io_uring, SIMD hash)
  src/runtimes/      M2  mod.rs (Runtime trait) · vllm · sglang · llmd
                         · vllm_mlx · llamacpp · mlc            (render + launch)
  src/hw/            M0  detect · bandwidth · backend
  src/catalog/       M0  specs · quants · licences
  src/tune/          M3  solve · bench · revert
  src/box/           M4  pack · manifest · oci · run · reprofile
  src/gateway/       M5  proxy · router · fleet · providers · meter
  src/capture/       M6  record · redact · store                (redact before write)
  src/cutover/       M9  shadow · canary · gate · rollback
  src/ffi.rs             PyO3 boundary — narrow and typed

src/clickllm/           PYTHON — the control plane. ML ecosystem lives here.
  hardware.py        ✅ accelerator + memory + bandwidth detection
  catalog.py         ✅ model specs, licences, verified-architecture flags
  fit.py             ✅ MoE/GQA/MLA-correct solve (control-plane mirror of spec.rs)
  cli.py             ✅ command surface
  distill/           M7  cluster · sample · evalset
  prove/             M8  graders · judge · equivalence · regret
  guard/             M10 watch · reprove · propose
  surfaces/          ‖   tui · console · mcp · sdk_py · sdk_ts
```

`✅` ships today. Everything else below.

**Why the split lands here:** capture and cutover sit in Rust because they are *on* the datapath; distill, prove and guard sit in Python because they are off it and I/O-bound. See [ADR-0007](adr/0007-tech-stack.md).

---

## The two Protocols everything hangs off

```rust
// clickllm-core — ADR-0002: no engine-specific type escapes this trait
pub trait Runtime {
    fn name(&self) -> &str;
    fn supports(&self, hw: &Hardware, m: &ModelSpec) -> Feasibility;
    fn plan(&self, hw: &Hardware, m: &ModelSpec, wl: &Workload) -> RuntimePlan;
    fn render(&self, plan: &RuntimePlan, target: Target) -> Vec<Artifact>;  // config out
    fn launch(&self, plan: &RuntimePlan) -> Result<Endpoint>;               // the box runs it
}
```
```python
# clickllm (Python) — M8, the highest-risk component
class Grader(Protocol):
    tier: Literal["assert", "task", "judge"]
    def grade(self, item: EvalItem, cand: Response, base: Response) -> Score: ...
```

If an engine-specific type ever escapes `Runtime`, portability is gone and dogfooding on Metal stops working. That's the invariant most likely to be violated under deadline pressure — CI enforces it.

---

## Milestones

### M0 · Hardware + catalogue *(2 wks — restored by [ADR-0008](adr/0008-build-from-scratch.md))*
Detection across Apple / NVIDIA / AMD / Intel / CPU: memory, bandwidth, device count, backend. Catalogue with geometry, quant availability, licences. Bandwidth-roofline throughput estimation, **labelled as an estimate until a benchmark replaces it**.

**Done when:** detection is correct on hardware we do not own, or reports *unsupported* rather than guessing. CI exercises the non-Apple paths.

### M1 · Weights — acquire, convert, cache *(2 wks)*
Nothing downstream can run without local weights.

- Resolve `hf:org/model`, `s3://`, `oci://`, local path → canonical ref
- **Resumable** download, checksum verify, parallel shards
- **Licence gate before bytes move** — refuse or warn on Llama-style MAU caps and non-commercial terms, matching `catalog.license_ok`
- Quantize/convert once per (model, quant, format): GGUF, MLX, AWQ/GPTQ. Cache keyed by content hash, shared across boxes
- Disk budget with LRU eviction

**Done when:** `clickllm pull glm-5.2 --quant q4` resumes after `^C`, verifies, and a second box reuses the cache.

### M2 · Runtimes — the Protocol and six backends *(3 wks)*
- `Runtime` trait + `Feasibility`/`RuntimePlan`/`Endpoint` types
- CUDA: **vLLM**, **SGLang**; multi-node: **llm-d** + GAIE `InferencePool`
- Metal: **vllm-mlx**, **llama.cpp**, **MLC**; CPU fallback
- Both `render()` (config artifact) and `launch()` (supervised process/container)
- CI import-guard: no engine-specific type escapes `runtimes/`

**Done when:** the same `RuntimePlan` produces a working endpoint on an M4 Max and on a CUDA box, with no branching above the Protocol.

### M3 · Tune — auto-tune, then *prove the tune* *(2 wks)* — **ADR-0004**
- `solve` derives candidate knobs from hardware + workload: quantization, spec-decode method/draft length, prefix/radix caching, TP/PP, `max_model_len`, `max_num_seqs`, chunked prefill, KV dtype, memory utilization
- `bench` runs the observed workload shape against the candidate and a no-optimization baseline
- `revert` drops anything that didn't help **on this hardware**, and says so
- Every knob answers `--explain`

**Done when:** on a machine where EAGLE-3 hurts (batch past the acceptance cliff), spec-decode is disabled automatically and the log says why. That negative case is the acceptance test — a tuner that only ever adds optimizations hasn't been tested.

### M4 · Box — pack, push, run, re-profile *(3 wks)* — **ADR-0005**
- `pack` → OCI artifact: manifest, weights lock, per-target bindings, `bench.json`, human README
- `push`/`pull` via any OCI registry; `cosign`-signable; air-gap mirror
- `run`: container on Linux/Windows/k8s; supervised native on macOS
- `reprofile`: on arrival, detect → compare to pack-time class → re-solve → re-bench → revert → **report what changed**
- Published, CI-tested support matrix (architecture × runtime × platform); unsupported → clear message, never a crash

**Done when:** a box packed on CUDA runs on the M4 Max, re-quantizes itself, and prints the delta. And when a box for an unsupported architecture fails with one readable sentence.

### M5 · Gateway — router, fleet, providers *(3 wks)*
- OpenAI-compatible proxy; `<15ms` p95 verified under load (NFR-1)
- **All major providers** via LiteLLM adapters — OpenAI, Anthropic, Google, Bedrock, Azure, +100
- **Router**: model→backend, per-cluster policy, weighted split, failover
- **Multi-model hosting**: N models on one host with a shared memory budget, on-demand load, LRU eviction of idle models, per-model URLs
- Post-cutover: hand balancing to GAIE/llm-d and get out of the datapath
- `meter`: cost + latency + tokens per route

**Done when:** three models serve concurrently from one 128 GB host under a shared budget, an idle one evicts under pressure, and a dead backend fails over without a dropped request.

### M6 · Capture *(2 wks)*
- Record request/response/latency/tokens/cost/tool-calls
- **Redact before persistence, fail closed** — a redaction failure drops the capture (NFR-3)
- Encrypted at rest; zero telemetry; cost dashboard standalone-useful

### M7 · Distill *(3 wks)*
- Cluster by task shape (embeddings + structural: system prompt, tool schema, output format, context length)
- Representative sampling; versioned, human-editable eval sets
- Export to Inspect AI / promptfoo — **your eval set is portable on day one**
- Manual cluster editing from v1; don't pretend the algorithm is right

### M8 · Prove *(6 wks)* — **the hero, and the biggest risk**
- Three-tier `Grader` stack: programmatic assertions → task graders (incl. unit-test execution for codegen) → position-swapped pairwise judge
- Equivalence matrix, traffic-weighted, with confidence intervals
- Regret analysis: cluster and *name* the failure modes; recommend hybrid policy
- Publish judge–human agreement on a sampled subset; a cell with no confidence shows `?`
- Shareable exported report

**Longest milestone, deliberately.** Budget more time for grader calibration than for code.

### M9 · Cutover *(4 wks)*
Shadow → quality-gated canary → cut, auto-rollback `<10s` on **eval** regression, hybrid per-cluster routing, TUI cockpit.

### M10 · Guard *(3 wks)*
Release watcher → auto re-prove on your set → promotion proposal with cost/quality delta. Also catches regressions from your own prompt changes.

### ‖ Surfaces *(continuous, not a phase)*
CLI from day one · TUI at M3 · web console at M8 (the matrix needs pixels) · MCP server at M4 · Python + TypeScript SDKs at M5.

MCP stays **read/plan-heavy, write-light**: `cutover_advance` and `deploy_apply` are not tools. An agent recommends; a human pushes the button.

---

## Build order, and why

```
M1 weights ─→ M2 runtimes ─→ M3 tune ─→ M4 box ──────────────┐
                                 │                            ├─→ demo: "any model, any hardware, one command"
                    M5 gateway ──┴─→ M6 capture ─→ M7 distill ─→ M8 PROVE ─→ M9 cutover ─→ M10 guard
```

**M1–M4 first** delivers "inference in a box" as a standalone product — hardware scan, model suggestion, weights, optimization, portable artifact, endpoint URL. It is demoable, shareable, and useful to someone who never intends to run the migration.

**M5–M8** is the migration, and M8 is where the company is won or lost.

Two independently useful products, one shared spine. If M8's graders can't produce verdicts people trust, **stop and fix them** — M1–M5 still stand on their own, and shipping M9 on bad graders actively harms users.

---

## Risk gates

| Gate | Before proceeding to | Test |
|---|---|---|
| Auto-tune never makes things worse | M4 | The disable-spec-decode negative case passes |
| Box runs where it wasn't packed | M5 | CUDA-packed box re-solves and runs on Metal |
| Gateway overhead within budget | M6 | p95 `<15ms` under sustained load |
| Redaction fails closed | M7 | Fault injection: no unredacted byte reaches disk |
| **Graders agree with humans** | **M9** | **Published agreement rate on a sampled subset** |

The last gate is the one that matters. Everything downstream of it moves production traffic.

---

## Traceability — every capability you named

| Capability | Milestone |
|---|---|
| Download model weights | M1 |
| Optimizations abstracted from the user | M3 · ADR-0004 |
| vLLM · SGLang · llm-d | M2 |
| Speculative decoding (EAGLE-3 / P-EAGLE / mlx) | M3 |
| Scan hardware, suggest what fits | ✅ ships · broadened M0 |
| Deploy seamlessly, zero-config | M3 + M4 |
| Inference in a box, portable | M4 · ADR-0005 |
| Agentic proactive suggestions | M4 re-profile · MCP · M10 |
| Endpoint URLs | M4 + M5 |
| Router capability | M5 |
| Multi-model hosting | M5 |
| Acts as a gateway | M5 |
| Connect to all major providers | M5 |
| Kubernetes | M2 (llm-d/KServe) + M4 |
| Eval suite | M7 + M8 |
| Load balancing across models | M5, then GAIE post-cutover |
| Custom router | M5 + M9 hybrid policy |
| Best-in-class UX / web interface | ‖ surfaces, console at M8 |
| Self-service / kiosk | CLI `clickllm switch` from M4 |
| SDK support | ‖ M5 |
| Suggestions on what LLM fits | ✅ ships · + M8 evidence |
| Suggestions on what optimizations to apply | M3 |
| Eval set to validate | M7 + M8 |
| Multiple hardware infra | M2 + M4 support matrix |
| Customizable / guided | `--set` overrides, `--ask`, `--explain` throughout |

---

## Estimate

~33 engineering weeks to M10 for one focused engineer; **M0–M4 is ~10 weeks** and is the first thing worth showing anyone.

Treat M8 as elastic. It is the only milestone where "done" is a judgement about trustworthiness rather than a passing test, and the only one where shipping early does damage.
