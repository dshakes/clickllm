# ADR-0010 — Retire `clickllm-weights`; manage the cache the engines actually fill

**Status:** accepted · **Date:** 2026-07-28
**Relates to:** [ADR-0005](0005-inference-in-a-box.md) · [ADR-0009](0009-rust-python-seam.md)

## Context

`clickllm-weights/` was a complete Rust crate: resumable HTTP fetch with `.part` files beside their target, SHA-256 verification, a rename as the commit point, a content-addressed store keyed on the canonical `ModelRef`, an LRU budget with eviction, and pinning so a running engine never has weights deleted underneath it. 22 tests, all passing, all green in CI since the day it landed.

**Nothing called it.** Not the gateway, not the Python control plane, not a binary, not a test outside its own directory. It compiled, it was correct, and it was dead.

That is not a quality problem. The crate was the right thing to build for the design of the time — M1 in [ADR-0007](0007-tech-stack.md)'s milestone map is *"weights: acquire · convert · cache · licence"*, and acquisition on the Rust side of the datapath is exactly what that called for. Two things changed under it.

**The engines fetch their own weights, and `launch` deliberately does not.** mlx_lm's `_download` is documented as *"ensures the model is available locally. If the path does not exist locally, it is downloaded from the Hugging Face Hub"*; vLLM and SGLang behave the same. So `src/clickllm/launch.py` refuses to download, and says why in its module docstring: a downloader beside the engine's is *a second, worse cache*, whose first symptom is 30 GB fetched twice. [ADR-0005](0005-inference-in-a-box.md) had already put this in writing on the packaging side — the box carries a `weights.lock` recording *"source + checksum (weights fetched, not vendored)"*. Weights were never ours to ship; once launch existed, they were not ours to fetch either.

**It managed a cache nothing writes to.** This is the part that makes "unused" into "unusable". `clickllm-weights` owned its own directory with its own on-disk format (`.clickllm-entry.json` per entry). The engines populate **Hugging Face's** cache — `~/.cache/huggingface/hub`, or `$HF_HOME` / `$HF_HUB_CACHE` — in the Hub's own `models--org--name/{blobs,snapshots,refs}` layout. Two stores, disjoint, and the empty one was the one with the eviction policy.

Meanwhile the need the crate was reaching for got sharper, not softer: proving N candidate models means N × 30–45 GB accumulating on a laptop with no budget, no eviction, and no record of which repo is the incumbent you are comparing against.

## Decision

**Delete `clickllm-weights/` and its workspace membership. Replace it with `clickllm cache` (`src/clickllm/cache.py`), which reads and prunes the Hugging Face cache the engines really fill.**

The Rust suite goes from 249 tests to 227; the 22 that leave were testing a downloader we do not run.

### Why not repurpose it

The obvious move — point the existing crate at `~/.cache/huggingface/hub` — was rejected on three counts:

1. **The valuable half is the half that no longer applies.** Resumable fetch, digest verification and the atomic-rename commit point are the crate's real content, and all of it is download machinery. Reading a directory and deleting subtrees is what remains, and it is a hundred lines.
2. **The format is not ours.** The Hub's layout — content-addressed `blobs/`, symlinked `snapshots/<sha>/`, `refs/main` — is `huggingface_hub`'s to change. Encoding it in a compiled crate makes every upstream layout tweak a release of ours; encoding it in a readable Python module makes it an afternoon.
3. **It fails [ADR-0009](0009-rust-python-seam.md)'s admission test.** The test for crossing the seam is *"would it otherwise be implemented twice?"* Cache reporting would be implemented once, in the language the CLI already lives in, with no format shared with Rust. Nothing about it earns a compiled dependency — and a `uvx clickllm` user with no wheel for their platform would lose the feature entirely.

### What replaces it

`clickllm cache` — stdlib-only Python, same zero-dependency rule as `fit`:

| Command | Does |
|---|---|
| `clickllm cache` | every repo in the hub cache: size on disk, age, pin marker, free space, budget |
| `clickllm cache evict --budget 200G` | prints the exact repos and bytes it *would* free. Deletes nothing |
| `clickllm cache evict --budget 200G --yes` | performs that plan |
| `clickllm cache pin <repo>` / `unpin` | protects a repo from every future plan |

Three properties are load-bearing, and each is a test:

- **Deletion is confirmed twice.** `plan_eviction` computes, `apply_eviction` refuses without `confirm=True`, and the CLI only confirms on `--yes`. A tool that reclaims 200 GB helpfully will eventually reclaim the wrong 200 GB.
- **A pinned repo is never in a plan.** Not "evicted last" — absent from the candidate list. An eval sweep that evicts the incumbent destroys the comparison it exists for, and the symptom is a silent re-download nobody connects to the sweep. When the budget cannot be met without touching a pin, the plan reports the shortfall instead of widening.
- **Nothing outside the hub root is a target.** Every path is re-resolved against the root and must still be a `models--` / `datasets--` / `spaces--` directory *directly* under it. The check raises rather than skips: a path that gets that far is a defect in the module, and stopping is the only response that cannot lose data.

Pins and the budget live in clickllm's state directory (`launch.state_dir()`), not in the Hub's cache — writing bookkeeping into a directory another tool owns is how you get a file its next scan deletes.

Sizes are measured off `blobs/` with symlinks skipped and hardlinks counted once. Following the snapshot symlinks reports a 4 GB model as 8 GB, which is the first bug anyone writing this hits.

## Consequences

**Good**

- The eviction policy now applies to the bytes that actually exist. That was the whole gap.
- One less crate, one less dead-code surface, and 22 fewer tests whose green tells nobody anything.
- The feature works under `uvx clickllm` on a machine with no Rust toolchain and no wheel.
- `launch`'s refusal to download is now the *complete* position rather than half of one: we do not fetch weights, and we do take responsibility for what fetching leaves behind.

**Costs, honestly**

- **A genuinely good implementation is gone.** The resumable-fetch and verify-then-rename logic was careful work. It is in git history at the commit before this ADR, and if we ever need to download weights ourselves — an air-gapped mirror is the plausible case — that is where to start rather than from nothing.
- **`last_used` is a proxy, not a log.** The Hub records no access time, so LRU ordering comes from the newest `atime`/`mtime` under each repo. On a `noatime` mount that degrades to "last downloaded". It is still the right ordering for a sweep, and it is a further reason nothing is evicted without a human reading the list.
- **We now depend on someone else's directory layout.** If `huggingface_hub` changes it, `clickllm cache` reports nothing until we follow. Contained by keeping the parsing in one readable module and refusing to act on any directory it does not recognise.
- **`ModelRef`'s cache-key normalisation loses its only consumer.** It stays in `clickllm-core` because licence gating and spec resolution use the same canonicalisation; no other crate is affected.

## Alternatives considered

**Leave it in place, unused.** Rejected. Dead code that compiles and passes tests is worse than no code: it reads as a supported path, and the next person to need weight management will extend it rather than notice that nothing calls it. Deleting it is the only honest way to record that the design moved.

**Write `clickllm cache` in Rust, reusing the crate's eviction logic.** Rejected per decisions 2 and 3 above — it buys nothing over stdlib Python, ties an upstream directory layout to our release cycle, and costs the `uvx` path.

**Have clickllm point the engines at a cache we own, via `HF_HUB_CACHE`.** Superficially attractive: one store, ours, with a real budget. Rejected because it takes over an environment variable the user's other tools also read, and silently relocating someone's 200 GB cache — orphaning every model their existing scripts expect to find — is exactly the kind of helpful, irreversible act this project does not do. Reading their cache costs us nothing and surprises no one.
