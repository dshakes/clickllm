# ADR-0017 — Money is a range or a refusal, and the receipt is versioned to carry it

**Status:** accepted · 2026-08-12

## Context

clickllm's entire proposition is that a team can replace a closed-model API
with an open-weight one and know what it costs them in quality and saves them
in money. The quality half was rigorous: Wilson intervals, a bar a whole
interval must clear, `?` rather than a fabricated score, and invariant 6 —
*never report a number without its confidence.*

The money half was not computed at all. `suite()` had accepted
`incumbent_cost` and `monthly_cost` since it was written and `Policy` computed
a blended figure from them, but **no surface supplied either**, so every run
printed `Saving: unknown — no cost rate configured`. `Receipt` — the artifact a
stakeholder actually receives, and the only part of this a CFO reads — had
eighteen fields and none about money.

Three further findings came out of wiring it up, and each changed the design.

**The saving line was making a claim that was false.** It read:

```
Saving: $2,530/mo (89%) at zero measured quality loss
```

A cluster moves when its *whole interval* clears the bar. At a 0.90 bar, a
cluster scoring 98% [95–99] moves — and in a 200-item run that is four items
whose answers measurably differed. "Zero loss" was the strongest possible claim
attached to the one number a reader most wants to believe, on the line most
likely to be pasted into a slide.

**A point estimate is the same defect invariant 6 exists to prevent.**
`$2,530/mo` reads as measured. The rates *are* given, but the moved share is
**measured**, and measuring 60% on 40 requests is a different claim from
measuring 60% on 4,000. The dollars have to inherit that.

**Adding any field to `Receipt` invalidated every receipt 1.0.0 issued.** The
digest is `sha256` over `asdict(self)`, so two new fields changed the digest of
documents that already exist on other people's disks — and `from_json` reports
a mismatch as *"it has been altered since it was issued"*. Not a parse error: a
**false accusation of forgery**, on the artifact whose only job is to be
trusted, in a product where invariant 8 says nothing authorises a cutover
except evidence.

## Decision

**1. One module owns the arithmetic.** `prove/cost.py` holds `blended()`,
`saving()` and the refusal rules. `Policy` and `Receipt` both call it. This is a
direct response to the most common defect in this repo's history — a fact
duplicated is a fact fixed in one place — applied to the fact you can least
afford two answers to.

**2. A saving is a range.** The moved share carries a Wilson interval at the
sample size it was measured on; the money is that interval through the blended
cost. Less evidence reads as a wider range, which is the property that makes it
honest rather than decorative.

**3. Refuse rather than flatter.** Four refusals, each naming what would fix it:

| Missing | Refusal |
|---|---|
| either rate | `pass --incumbent-cost and --candidate-cost` |
| captured traffic | `run clickllm observe first` |
| the window | `pass --traffic-window` |
| a window under 7 days | `capture at least 7 days so a full week is represented` |

The seven-day floor is a calibration knob, not a truth: traffic has a weekly
shape, so scaling three days to thirty multiplies whichever part of the week
happened to be captured. A workload with no weekly shape could justify less.

The window is a **flag rather than a derivation** because captures carry no
timestamps. The tool genuinely does not know how long you watched, and
inventing a default here would fabricate the exact fact the refusal protects.

**4. `clickllm.receipt/v2`, and v1 stays readable.** A v1 document is digested
over the fields v1 had; v2 over all of them. Neither is weakened — each covers
everything its own document contains — and a receipt issued by 1.0.0 verifies
unchanged. A v1 receipt is **refused at construction** if it carries a cost,
because a v1 digest would not cover it, and money outside the seal is the one
field where that is unacceptable.

## Consequences

**Every receipt now states the money or says why it cannot.** Silence was the
worst option: a receipt that omits cost when it cannot compute one reads as a
migration nobody costed.

**The policy line and the receipt line are the same claim**, from the same
function, over the same denominator — rather than two computations that happen
to agree today.

**`clickllm fit` is untouched.** `cost.py` imports only `stats`, so the
zero-runtime-dependency promise for `fit` under `uvx` still holds.

**The capture denominator can understate confidence, never overstate it.**
`suite` falls back to the eval-set size when the real capture count is not
supplied. An eval set is drawn *from* captures, so the fallback is never larger
than the truth, and a smaller denominator widens the interval — the one
direction this number is allowed to be wrong in.

## What would falsify this

A workload where the seven-day floor blocks a legitimate claim — a batch system
with no weekly shape, or a monthly billing cycle that makes seven days
meaningless in the other direction. The floor is a constant with a comment, not
a law; what must not move is that the refusal is explicit and names its fix.

## Alternatives rejected

**A point estimate with a footnote.** The footnote does not survive the
screenshot. The range is the number.

**Deriving the window from capture timestamps.** They do not exist. Adding them
is a change to what gets written to disk about a customer's traffic (NFR-2,
NFR-3) and is not worth making as a side effect of a costing feature.

**Bumping to v2 and dropping v1.** Every receipt already issued would fail to
parse. The point of a content-addressed artifact is that it outlives the tool
that wrote it.

**Silently changing the digest.** The break would be invisible until someone
tried to verify an old receipt, and it would report as tampering.
