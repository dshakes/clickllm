# ADR-0007 — Rust datapath, Python control plane

**Status:** accepted · **Date:** 2026-07-27 · **Supersedes:** the "Technical assumptions" section of [`20-prd.md`](../20-prd.md)

## Context

Stated requirement: use hardware — CPU, threads, memory — efficiently. That is the key concern.

It is also a requirement that must be aimed, not applied uniformly. Most of clickllm is not compute-bound, and rewriting an I/O-bound eval loop in a systems language buys nothing while giving up the ML ecosystem. So: where is the work actually?

| Component | Nature of the work | Efficiency actually matter? |
|---|---|---|
| **Gateway / router** | thousands of concurrent streaming connections, per-request tail latency, `<15ms` p95 budget | **yes — critically** |
| **Multi-model fleet** | GB-scale memory budgeting, eviction, mmap of weights | **yes** |
| **Weights** | parallel shard download, checksum, quant conversion | **yes — I/O + hashing throughput** |
| **Runtime supervision** | spawn, health, restart | modest |
| Fit solver | a few dozen multiplications | no — microseconds |
| Distill (clustering) | embeddings + k-means over ~10⁵ prompts | no — already native under NumPy/BLAS |
| Prove (eval runs) | blocked on model inference | no — I/O bound |
| Guard | periodic, low volume | no |

The efficiency-critical set is exactly the datapath and the weights path. Everything else is I/O-bound or already calls into native code.

## Decision

**Rust for the datapath and everything that touches bytes or memory budgets. Python for the control plane, where the ML ecosystem lives.**

```
clickllm-core   (Rust)     gateway · router · fleet · weights · runtime supervision · box run
      │ PyO3
clickllm        (Python)   distill · prove · guard · CLI orchestration
```

### Why Rust, not Go, for the datapath

- **No GC pauses.** NFR-1 is a *p95 tail latency* budget. Go's GC injects exactly the kind of tail that budget measures. This is the deciding argument.
- **Explicit memory control.** M5 budgets GB-scale memory across a multi-model fleet and evicts under pressure. That requires knowing what is resident, not asking a collector.
- **Zero-copy streaming.** Token streams pass through as `Bytes` without buffering a response body.
- **`llmfit-core` is a Rust crate.** A Rust datapath **links it as a library** instead of shelling out to a binary and parsing JSON per call — no process spawn, no serialization, direct access to its hardware/fit/provider types. This was not obvious until the source read ([ADR-0006](0006-llmfit-source-evaluation.md)) and it is a material win.

Go remains the better choice for anything we contribute *to* the k8s ecosystem (GAIE, llm-d are Go), but we consume those over the wire rather than extending them.

### Why Python stays for the control plane

Rewriting `distill`/`prove` in Rust would cost the embedding, clustering, tokenizer, and Inspect AI ecosystems to speed up code that spends its life awaiting HTTP. That is effort spent where the profile says there is nothing to win.

### Concrete efficiency commitments

| Concern | Approach |
|---|---|
| **Threads** | Tokio multi-threaded work-stealing runtime, sized to detected physical cores (not hyperthreads); blocking work confined to `spawn_blocking` |
| **CPU** | no per-request allocation in the hot path; header/route parsing on borrowed slices; SIMD-accelerated hashing on the weights path |
| **Memory** | `mimalloc` global allocator; **mmap weights** so the page cache is shared across models and processes; explicit per-model accounting with LRU eviction against a declared budget |
| **I/O** | `io_uring` on Linux, kqueue elsewhere; parallel range requests for shard download; streamed checksum so bytes are hashed once, in flight |
| **NUMA** | on multi-socket hosts, pin runtime threads and place model memory on the local node |
| **Backpressure** | bounded queues with explicit shed rather than unbounded buffering — a proxy that OOMs under load is worse than one that rejects |

**Every one of these is a claim to be measured, not asserted.** The gateway ships with a load harness, and the `<15ms` p95 gate ([`80-implementation-plan.md`](../80-implementation-plan.md) risk gates) is what decides whether the implementation is done — not the language choice.

### The existing Python fit solver

Stays as-is. Per [ADR-0006](0006-llmfit-source-evaluation.md) it shrinks to an adapter plus the serving-side solve, and it runs in microseconds on the control plane. Porting it would be motion, not progress.

## Consequences

**Good**
- Efficiency effort lands where the profile says it matters, and nowhere else.
- Direct `llmfit-core` linkage removes a process boundary from the hottest advisory path.
- One static binary for the datapath — which is also what [ADR-0005](0005-inference-in-a-box.md)'s macOS native execution binding wants to supervise.
- Rust's memory model is the honest tool for a component whose job is budgeting memory.

**Bad**
- **Two languages, one FFI seam.** Real, ongoing cost. Mitigated by keeping the boundary narrow and typed (PyO3, mirroring what `llmfit-python` already does successfully), and by never letting control-plane logic leak into the Rust side.
- Slower to write than Go or Python. Accepted for the datapath only — which is a small fraction of total LOC.
- Contributors need Rust. Mitigated: the control plane, where most feature work happens, stays Python.

**Rejected: all-Python.** Would miss the p95 budget under concurrency and cannot honestly budget fleet memory.
**Rejected: all-Rust.** Gives up the ML ecosystem to optimize code that is waiting on the network.
