# ADR-0008 — Build the full stack ourselves

**Status:** accepted · **Date:** 2026-07-27
**Supersedes:** [ADR-0003](0003-adopt-third-party-fit.md) and [ADR-0006](0006-third-party-fit-evaluation.md)

## Context

ADR-0003 and ADR-0006 concluded that the hardware-fit layer should be delegated to an existing third-party tool, and ADR-0007 leaned on linking that tool as a crate.

Direction from the product owner, after that analysis was presented: **build every capability ourselves, and build it better.** No third-party dependency for the fit, catalogue, hardware, or benchmarking layers.

This reverses a decision I recommended. The analysis behind it is unchanged and remains on file in the superseded ADRs — a third party has a community measurement corpus we start without. The owner has weighed that and chosen ownership of the full stack. That is a legitimate call: the corpus is a lead, not a moat, and a dependency at the centre of the product is a permanent constraint on how good the product can get.

## Decision

**Own the whole stack.** Nothing in the shipped product depends on, links, shells out to, or is positioned against a third-party fit tool.

This restores to our scope:

| Capability | Where it lands |
|---|---|
| Hardware detection — Apple / NVIDIA / AMD / Intel / CPU, memory, bandwidth, backend | **M0** (new), Rust |
| Model catalogue — params, geometry, quant availability, licences | **M0**, Rust + data file |
| Fit solve — MoE / GQA / MLA-correct sizing, quant selection, run modes | ✅ shipped (`spec.rs`, `runtime/`) |
| Throughput estimation — bandwidth roofline, per-backend constants | **M0** |
| Real measurement — TTFT, TPS, latency against live endpoints | **M3** |
| Measurement corpus — our own, from our own users, opt-in | **M3+** |

And it removes the crate-linkage argument from [ADR-0007](0007-tech-stack.md). Rust still wins the datapath on the two arguments that mattered most — **no GC pauses against a p95 budget** and **explicit accounting for GB-scale fleet memory**. The stack decision stands on those alone; it does not need the third argument.

## What "better" has to mean

Owning it is only worth the cost if the result is genuinely better. Three commitments:

1. **Estimates carry error bars, and say so.** A roofline projection labelled as measurement is the failure mode of every sizing tool. Ours prints `roofline estimate, not measured` until a real benchmark replaces it.
2. **Measurement beats estimation wherever we have it.** M3's benchmark-and-revert step already produces real (hardware, model, quant, tok/s) tuples. Those feed our own corpus, so our numbers improve with use rather than staying theoretical.
3. **The arithmetic is auditable.** `--explain` on every number, showing the formula and inputs. This is where a from-scratch implementation can beat an established one immediately: not on data volume, but on being checkable.

## Consequences

**Good**
- No dependency at the centre of the product. Nothing we need is gated on someone else's roadmap or licence.
- The fit layer becomes ours to differentiate. Serving-side sizing — KV at observed concurrency, tensor-parallel splits, disaggregated prefill/decode — is not something a local-first tool models, and we need it for M2/M4/M5 regardless.
- One language boundary fewer.

**Bad**
- **~2.5 weeks returns to the critical path**, plus M0. Total goes back to ~33 weeks; M1–M4 to ~10.
- **We start with zero measurement data.** Our tok/s figures are roofline estimates until users run benchmarks. This is the real cost, and it is why commitment (1) is not optional — overstating estimate quality would be the one way to lose credibility permanently.
- Hardware detection across five vendors is unglamorous, long-tail work that has to be right on machines we do not own. CI must exercise the non-Apple paths.

**Watch:** if M0 hardware detection starts consuming disproportionate time, the honest fallback is to narrow *supported* hardware rather than ship shallow detection for everything. A confident wrong answer about someone's GPU is worse than "unsupported, tell us about it".
