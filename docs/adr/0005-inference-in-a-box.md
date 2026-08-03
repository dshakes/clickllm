# ADR-0005 — "Inference in a box" is a contract, not a container image

**Status:** accepted · **Date:** 2026-07-27 · **Extends:** [ADR-0004](0004-zero-config-deployment.md)

## Context

Stated goal: *anyone should be able to deploy any open-source model seamlessly — agentic, proactively hardware- and software-aware, with the best optimizations. Inference in a box that works on any hardware; give them a container or similar.*

The intent is right and it is the product. Exactly one piece is not ours to build.

### Why the Linux-container-on-Metal path is not buildable

Not a Docker limitation, and not an engineering shortcut — the mechanism does not exist to call:

1. A Linux container on macOS runs inside a **Linux VM** (Virtualization.framework, or HyperKit/QEMU). Apple's own `container` tool, shipped 2025, does the same thing.
2. **Metal is a macOS-only API.** There is no Metal driver, and no Metal ABI, inside a Linux guest.
3. Bridging that would need **GPU passthrough**. On x86 that's VFIO/SR-IOV against a discrete card. Apple Silicon's GPU is **on-die and shared with the CPU across unified memory** — there is no discrete device to hand through, and Apple ships no IOMMU passthrough path for it.
4. Virtualization.framework's paravirtualized graphics device targets **display for macOS guests**, not GPGPU compute for Linux guests. There is no compute passthrough API to call.

So the missing piece is a **driver and passthrough interface only Apple can ship.** We could write a Metal shim, a virtio-GPU compute transport, and a guest driver — and it would still be blocked at the hypervisor, because the host-side interface it would bind to does not exist.

Scope it honestly, though — **it is one platform:**

| Platform | Container → GPU | Path |
|---|---|---|
| Linux + NVIDIA | ✅ NVIDIA Container Toolkit | container |
| Linux + AMD | ✅ ROCm device mounts | container |
| Windows + NVIDIA | ✅ WSL2 | container |
| Kubernetes | ✅ device plugins | container |
| **macOS + Apple Silicon** | ❌ **no mechanism exists** | native process |

One row. It happens to be the dev machine and the entire indie on-ramp, which is why it can't be waved off — a CPU-only container on an M4 Max is ~10× slower than the MLX path, and the failure is *invisible*: it works, it's just bad, and nobody learns why.

### What we can build — and should

The push-back is right that the *box* is buildable. Distribution and execution are separable, and only execution is constrained:

- **Package as an OCI artifact.** OCI images are content-addressed layers with a manifest; nothing requires a container runtime to *unpack* one. So the box ships through standard registries — `docker push`, `cosign` signing, layer dedup, air-gapped mirrors, existing enterprise supply-chain tooling — on **every** platform including macOS.
- **Bind execution per host.** On Linux/Windows/k8s the OCI artifact *is* the runtime image. On macOS the same artifact is pulled and unpacked, and clickllm supervises the native MLX/llama.cpp process against it.

The user gets one artifact, one registry, one `pull`, one command, one endpoint. Only the final exec binding differs — and that difference is invisible unless they ask.

If Apple ever ships Metal passthrough, this design absorbs it: the box gains a target and macOS moves to the container row. Nothing is bet against it.

Two further limits worth stating plainly:

- **"Any open-source model" is bounded by runtime support.** Coverage is broad (any GGUF via llama.cpp, any transformers-compatible architecture via vLLM, MLX conversions) but a brand-new architecture lands before its kernels do. Unsupported must produce *"no runtime supports this architecture yet"*, never a cryptic crash.
- **Tuning is not portable.** A config tuned on an A100 is wrong on an L40S and catastrophic on a Mac. Settings baked at build time and applied blindly elsewhere are exactly the "wrong auto-tune" failure ADR-0004 warns about.

## Decision

**Ship the box as an OCI artifact with a per-host execution binding.** One artifact, one registry, one command, one OpenAI-compatible endpoint. It runs *as* a container everywhere a container can reach the GPU, and as a supervised native runtime on macOS.

### The artifact — an OCI image, pushable to any registry

```
mymodel.box  (OCI manifest + layers)
  manifest.yaml     model · quant · tuned knobs · provenance · eval scores
  weights.lock      source + checksum (weights fetched, not vendored)
  targets/
    linux-cuda/     Dockerfile + compose + k8s manifests (vLLM / SGLang / llm-d)
    linux-rocm/     "
    darwin-metal/   native launch spec (mlx / llama.cpp / MLC)
    cpu/            llama.cpp fallback, honest about tok/s
  bench.json        the measurements that ratified the tuning, per host class
  README.md         what was chosen and why — human-readable
```

### The contract

| | |
|---|---|
| **One command** | `clickllm run ghcr.io/acme/triage:v3` — anywhere |
| **One endpoint** | OpenAI-compatible, same port, same shape, every platform |
| **One registry** | standard OCI — `cosign` signing, layer dedup, air-gapped mirrors |
| **One manifest** | portable across hosts; the *bindings* differ, the declaration doesn't |
| **Zero knobs** | ADR-0004 holds — nothing is asked |

`clickllm pack` builds it, `clickllm push`/`pull` moves it, `clickllm run` executes it. On Linux/Windows/k8s that resolves to a container; on macOS the same artifact is unpacked and a native engine is supervised against it. **Same command, same endpoint, same artifact** — which is what "in a box" has to mean.

### Proactive, not baked

The agent re-profiles **at run time, on the destination host** — not only at pack time:

1. **Detect** — accelerator, memory, bandwidth, driver/CUDA/ROCm version, engine availability
2. **Compare** against the host class the box was tuned on
3. **Re-solve** if they differ — a box tuned for 80 GB landing on 24 GB re-quantizes and re-fits rather than OOMing
4. **Re-benchmark** the candidate settings on *this* machine
5. **Revert** any optimization that doesn't help here (ADR-0004)
6. **Report** what changed and why — in the run log, not buried

A box is a *tuned starting point plus the evidence behind it*, never a frozen command line.

## Consequences

**Good**
- Delivers the goal honestly on every platform, including the one a literal container would silently ruin.
- Mac developers get the fast path (~119 tok/s on a 30B MoE via MLX) instead of CPU-only inference.
- Re-tuning on arrival is a genuine differentiator: nobody else's artifact adapts to the host it lands on. It also solves the "worked on my GPU" class of bug outright.
- Boxes are shareable and inspectable — a team publishes `support-triage.box` and colleagues run it on whatever they have.

**Bad**
- Two **execution** bindings to build and test. Distribution stays single-path (OCI), which removes most of the duplication — registries, signing, and mirroring are written once. Mitigated further by [ADR-0002](0002-runtime-abstraction.md): the `Runtime` Protocol already models this split, so it's one more `render` target, not a second architecture.
- Run-time re-benchmarking adds first-run latency (seconds to a minute). Mitigate with a cached host-class fingerprint: pay it once per machine, then reuse.
- "Any model" needs an honest, maintained support matrix, and a clear *no* when an architecture has no runtime yet.

**Rejected: ship one CUDA container and tell Mac users to use CPU.**
Simpler, and it silently guts the on-ramp persona and the dogfooding machine. The whole product's credibility rests on measuring things honestly; shipping a knowingly 10×-slow default contradicts that on day one.

## Follows from this

- `clickllm run` must work with **no clickllm-specific runtime present** on the target beyond the CLI itself — pulling the container or the native engine as needed.
- The support matrix (architecture × runtime × platform) is published, tested in CI, and treated as a first-class artifact.
- Every run emits a one-line provenance summary: model, quant, engine, knobs changed on arrival, measured tok/s. Silence about a re-tune is a bug.
