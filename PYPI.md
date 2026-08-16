# onpar

**Prove which open model can replace your closed one — on your traffic, your hardware, your budget.**

No config file. It reads your machine, sizes the KV cache per model, applies the
licence gate, picks the engine, and sets the flags. Then it tells you whether the
open model is actually good enough for *your* traffic, and hands you a receipt
that says so.

```bash
uvx --from onpar onpar fit --context 32k --concurrency 8
```

```
  M4 Max · 16 cores · 128 GB · 546 GB/s
  usable for inference: 96 GB

  FEASIBLE at 32,768 context, concurrency 8

  model                     quant   weights      kv   total    free  ~tok/s  license
  ----------------------------------------------------------------------------------
  Qwen3 30B-A3B (MoE)       q8        28.4G   24.0G   56.2G   39.8G      60  Apache-2.0 OK
  Phi-4 14B                 q8        13.7G   50.0G   66.3G   29.7G      18  MIT OK
```

Every number answers `--explain`, which prints the arithmetic that produced it.
Throughput is a memory-bandwidth roofline estimate, not a measurement, and says
so wherever it appears.

## Or just say what you're building

`onpar` with no arguments opens a conversation — one question at a time, asked
only when the answer would change the plan. Stop whenever you like and you still
leave with the best answer so far. In a script or a CI step it stays a usage
error rather than waiting on input.

## Then prove it, on your own traffic

Sizing tells you what fits. It cannot tell you whether the model is good enough,
and no public benchmark can either, because it has never seen your traffic.

```bash
onpar observe --upstream https://api.openai.com/v1   # record, redacted before storage
onpar distill                                        # cluster it into an eval set
onpar prove evalset.json --incumbent-cost 2847 --candidate-cost 317 \
    --traffic-window '14 days' --resume run
onpar brief receipt.json --out brief.html            # one page for whoever signs off
```

The verdict is per task shape, never an average — a candidate can pass
tool-calling and fail long-context, and one number hides that. Every claim
carries a Wilson interval, and a cluster is proven only when its **whole**
interval clears the bar.

The saving is a range, never a point: the share of traffic that moves is
*measured*, so the dollars inherit its uncertainty. With no rate, no capture
count, or under a week of traffic, it says the saving is unknown and names what
would fix it.

## For agents

`onpar-mcp` speaks MCP over stdio with zero dependencies: nine read-only
tools, receipts and eval sets as resources confined to one root, and three
pre-built workflows. An agent can size, prove and guard a migration. It cannot
cut over — not by policy, but because no such tool exists to call.

## Install

```bash
uv tool install onpar
pipx install onpar
```

The PyPI name is `onpar`; the command is `onpar`. `pip install
onpar` installs an unrelated package.

## Sizing, and the three ways it goes wrong

- **MoE** needs **total** parameters resident. Sparsity cuts compute, not memory.
- **GQA** must use `kv_heads`, not attention heads — up to 8× out otherwise.
- **MLA** stores a compressed latent — roughly 50× out if you use the GQA formula.

## Status

Pre-alpha, and honest about it: every number that is a projection says so, and a
cell shows `?` rather than a fabricated score.

- **Source, docs and the full README:** https://github.com/dshakes/onpar
- **Site:** https://dshakes.github.io/onpar/
- **Licence:** Apache-2.0
