# ADR-0018 — The product is renamed to onpar, and every identifier moves with it

**Status:** accepted · 2026-08-15

## Context

The product was called `clickllm`. The name was wrong in three ways, and
[docs/70-naming.md](../70-naming.md) had already recorded that it should go:

- **It mis-signalled the category.** "Click" sells convenience — a wizard, a button,
  one-click deploy. This product sells evidence: an eval set distilled from captured
  traffic, a score per task shape, a `?` where the sample is too thin, and a refusal to
  promote on a number it cannot defend. The name advertised the commodity half.
- **`-llm` is a saturated suffix**, dating the product to the moment it was named.
- **It sold the wrong half.** "Will it fit" is a free web calculator with a dozen
  incumbents. The defensible product is the *join* — hardware truth and quality truth in
  one sentence — and the name pointed nowhere near it.

That document recommended **Parity**, which never shipped: [Parity
Technologies](https://www.parity.io/) is a live funded software company, and the word does
not verb. The thesis was right and the lexical form was unusable.

Two further facts made this the moment to move rather than later:

1. **Nobody is using the product yet.** No PyPI installs to strand, no `~/.clickllm` state
   to migrate, no deployed `InferenceWorkload` CRs to convert. Every compatibility
   obligation that would normally make a rename expensive is absent. That is a window that
   closes permanently at the first real user.
2. **A direct competitor named the adjacent promise.** [Understudy Labs
   (YC S26)](https://www.ycombinator.com/companies/understudy-labs) launched with
   *"effortlessly move to open weight models"* — this product's thesis, funded. Positioning
   against it now matters more than it did a month ago.

## Decision

**The product, and every identifier derived from it, is renamed to `onpar`.**
Motto: *Nothing moves until it's on par.*

`onpar` is the question the buyer asks and the answer the tool gives, in the same words:
`extraction 96% ✓ on par / codegen 71% ✗ off par`. It verbs, it carries quality parity and
cost parity (finance: *at par* = break-even) in one word, and it gives the failing cell a
name — **off par** — which matters because the failing cell is the differentiating half of
this product. Full rationale and the rejected shortlist are in
[docs/70-naming.md](../70-naming.md).

**The rename is total.** No dual-name compatibility layer, no deprecation shim, no
dual-prefix env-var reads. Specifically:

| Surface | From | To |
|---|---|---|
| Distribution | `clickllm-cli` | `onpar` |
| Console scripts | `clickllm`, `clickllm-mcp` | `onpar`, `onpar-mcp` |
| Python package | `src/clickllm/` | `src/onpar/` |
| Rust crates | `clickllm-{core,gateway,py}` | `onpar-{core,gateway,py}` |
| Env vars (11) | `CLICKLLM_*` | `ONPAR_*` |
| State dir | `~/.clickllm/` | `~/.onpar/` |
| K8s API group | `clickllm.dev` | `onpar.dev` |
| K8s managed-by | `clickllm` | `onpar` |
| Box media type | `application/vnd.clickllm.box.v1+json` | `application/vnd.onpar.box.v1+json` |
| MCP tools (14) | `clickllm_*` | `onpar_*` |
| HTTP headers | `x-clickllm-{cluster,backend}` | `x-onpar-{cluster,backend}` |
| systemd units | `clickllm-{vllm,sglang}.service` | `onpar-{vllm,sglang}.service` |

**The `-cli` suffix is dropped.** It existed only because PyPI rejected bare `clickllm` as
too close to the existing `click-llm` (recorded at `.github/workflows/release.yml:137-138`).
`onpar`, `on-par` and `on_par` — one name under PEP 503 normalisation — are all free, so
package name, command name and import name are finally the same string.

## Consequences

**Breaking, and deliberately so:**

- Any exported `CLICKLLM_*` variable stops being read. No fallback.
- `~/.clickllm/` is orphaned, including the capture key, which is unrecoverable. Acceptable
  only because no such directory exists in the wild.
- Renaming the K8s API group changes the CRD name, so `kubectl apply` would create a
  *second* CRD rather than update the first; CRs under the old group become invisible to the
  controller. Likewise `app.kubernetes.io/managed-by` is selector-relevant, so changing it
  orphans previously-created Deployments. Both are safe here and would not be later.
- The GitHub Pages path is derived from the repo name, so the site moved to
  `dshakes.github.io/onpar/` and the old install one-liner now 404s. A 404 is the honest
  failure; a silently-wrong install is not.

**Silent-breakage hazards this rename introduced**, each of which required a targeted check
rather than a find-replace — recorded because they are the shape that recurs here (see
[the four defect shapes](../../CLAUDE.md)):

- `onpar-py/Cargo.toml` `[lib] name` and the `#[pymodule]` init symbol must rename
  *together*; mismatched, the extension compiles clean and fails at **import** time.
- `Dockerfile` copies `libclickllm_core.rlib`, a filename derived from the crate name.
- `tools/bump.py` hardcodes `../clickllm-core` inside a regex; unmatched, it **silently
  no-ops** and ships a mis-versioned release.
- **Two negative assertions pass vacuously after a rename** —
  `tests/test_box.py` (boxes stay dependency-free) and
  `clickllm-core/src/runtime/vllm.rs` (generated config never shells back into the tool,
  NFR-4). Renaming the string alone would retire both guarantees while staying green. Each
  was re-verified by injecting a violation and confirming the test fails.

**Non-breaking:** the argparse program name is derived from `sys.argv[0]`, so usage and
error strings follow the console-script rename for free.

## What would falsify this

- If `onpar` turns out to be unregistrable as a trademark in the relevant classes, or an
  existing holder objects. Clearance is a register search and has not been done; the three
  known collisions ([On Par Analytics](https://github.com/onpar),
  [poy/onpar](https://github.com/poy/onpar), [OnPar
  Technologies](https://www.onpartech.com/)) are documented in the naming doc as accepted.
- If users consistently read "on par" as *merely adequate* rather than *equal to*. The word
  is meant to claim equivalence; if it reads as faint praise for the open model, the name
  undercuts the product.
- If the product's centre of gravity moves off the equivalence verdict — if it becomes a
  serving platform or a router, "on par" stops describing it.

## Alternatives rejected

- **Keep `clickllm`.** Rejected: the naming doc's case against it stands, and the cost of
  renaming only rises with adoption. This was the last cheap moment.
- **Parity** — the previous recommendation. Rejected: live company, and does not verb.
- **Understudy** — the best story of any candidate. Rejected: [Understudy Labs
  (YC S26)](https://www.ycombinator.com/companies/understudy-labs) took both the name and
  the pitch.
- **Bare `par`.** Rejected: three letters is unregistrable, unsearchable, and collides with
  `$PATH` binaries.
- **Rename the product but keep `clickllm.dev` as the API group**, to avoid a K8s break.
  Rejected: it buys nothing when there are no deployed CRs, and leaves an `onpar` tool
  permanently emitting `clickllm.dev` resources — the same duplicated-fact shape that has
  bitten this repo before.
- **Ship a compatibility layer** (dual-prefix env vars, a `clickllm-cli` shim on PyPI,
  dual-served CRD groups). Rejected: every line of it is dead code written for users who do
  not exist, and it would have to be deleted later anyway.
