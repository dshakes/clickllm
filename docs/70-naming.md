# Name & motto

## Recommendation

# **onpar**

### *Nothing moves until it's on par.*

**onpar is the question the product answers, and the answer it gives.** "Is Qwen on par
with GPT-5 on our workload?" is the sentence a staff engineer says in the meeting where
this gets decided. The tool's output is the reply, in the same words: `extraction 96% ✓ on
par / codegen 71% ✗ off par`. Owning that phrase means owning the conversation.

This supersedes an earlier recommendation in this document for **Parity**. It is not a
change of thesis — *on par* and *parity* are the same idea — only of lexical form. Parity
was blocked by [Parity Technologies](https://www.parity.io/), a real funded software
company, and it never verbed: you cannot "parity" a model. `onpar` keeps the positioning
and fixes both problems.

- **One real phrase, closed up.** No portmanteau, no dropped vowel, no `.ai` gimmick.
- **It verbs and adjectives.** `onpar qwen3-32b`. "Is it onpar yet?" "The onpar run says no."
- **The failure case has a name.** *Off par* is the vocabulary for the cell that fails, and
  the failing cell is the differentiating half of this product — onpar's job is to say no,
  in public, with the arithmetic attached. No competitor has a word for that.
- **The second meaning is also ours.** In finance, *at par* is break-even. Quality parity
  and cost parity are the two axes the equivalence matrix computes, and one word carries both.
- **Package name == command name == import name.** `pip install onpar` → `onpar`. The old
  distribution carried a `-cli` suffix only because PyPI rejected bare `clickllm` as too
  close to the existing `click-llm`; `onpar` has no such neighbour, so the workaround goes.

## Why `clickllm` had to go

- **Mis-signals the category.** "Click" sells convenience — one-click deploy, a wizard, a
  button. This product sells *evidence*: an eval set distilled from your traffic, a score
  per task shape, a `?` where the sample is too thin, and a refusal to promote on a number
  it cannot defend. The name promised the commodity half and hid the load-bearing half.
- **Saturated suffix.** `-llm` is the `-ify` of this cycle. It dates the product to the
  moment it was named and puts it on a shelf with a hundred wrappers.
- **Sold the wrong half.** Sizing "will it fit" is a free web calculator with a dozen
  incumbents. The defensible product is the *join* — hardware truth and quality truth in
  one sentence — and nothing in the old name pointed at it.

## The shortlist

| Name | Motto | Verdict |
|---|---|---|
| **onpar** | Nothing moves until it's on par. | **chosen** — the buyer's question verbatim; verbs; *off par* names the failure; carries quality + cost parity in one word |
| **Assay** | Prove it, then move it. | strong runner-up — the metallurgical test that establishes what ore is worth, which is exactly the procedure; extends free (ore → sample → assay → mine); loses on the "essay" homophone and biotech SEO |
| **Vouch** | Nothing ships unvouched. | best trust framing — a vouch is personal accountability, which is *why* the `?` cell and the disclosed judge model exist; loses on reading warm/social in a category of instruments |
| **Muster** | Pass muster, then switch. | "pass muster" is the exact idiom for inspected-and-fit-to-serve, and a muster is literally the pre-deployment inspection; loses to onpar on being one step less direct |
| **Par** | Nothing moves below par. | same thesis, better length — but three letters is unregistrable, unsearchable, and collides with `$PATH` binaries and [PaR Systems](https://en.wikipedia.org/wiki/PaR_Systems) / [3PAR](https://en.wikipedia.org/wiki/3PAR) |
| **Parbench** | A benchmark of your traffic, not someone else's exam. | clean namespace, but "bench" imports the baggage of generic benchmarking — the precise thing this product argues against |
| **Touchstone** | — | the stone you rub gold against to grade it: the most exact fit for an equivalence matrix; reads consultative rather than next-gen, and Disney holds the mark in media |
| **Aliquot** | Measure the sample. Trust the whole. | the most intellectually honest option — it *is* the statistical method behind `distill`; unpronounceable at a conference, a name to read rather than say |
| **Grade** | — | ore has a grade, report cards have grades; loses to edtech search noise |
| **Fiducial** | The reference in frame. | sounds the most credible and appears unclaimed; four syllables and needs explaining every time |
| **Datum** | The point everything else is measured from. | clean metrology meaning; users hear "data" and drift to the crowded data-tooling shelf |
| **Ringer** | Bring in a ringer. | instantly understood, maximum memorability; implies a deception, which cuts against a product whose whole claim is honesty |
| ~~**Parity**~~ | Parity, proven. | *the previous recommendation in this document* — **disqualified**: [Parity Technologies](https://www.parity.io/) is a live funded company, and the word does not verb |
| ~~**Understudy**~~ | The one who can go on tonight. | best story of any candidate — **disqualified**: [Understudy Labs (YC S26)](https://www.ycombinator.com/companies/understudy-labs) launched with the pitch *"effortlessly move to open weight models"*, which is this product's thesis with a batch behind it |
| ~~**Defect**~~ | Defect from closed models. | most viral framing, best double meaning — **disqualified**: "defect" means *bug*, fatal collision for a product whose job is quality assurance |
| ~~**Swap**~~ | — | **disqualified**: [llama-swap](https://github.com/mostlygeek/llama-swap) does model swapping for local servers — same neighbourhood, same word |
| ~~**Twin**~~ | The open twin of your closed model. | **disqualified**: [Twin Labs](https://techcrunch.com/2024/01/31/twin-labs-automates-repetitive-tasks-by-letting-ai-take-over-your-mouse-cursor) |
| ~~**Yardstick**~~ | — | **disqualified**: an existing YC company, plus several firms on the name |
| ~~**Dyno**~~ / ~~**Crucible**~~ / ~~**Endpoint**~~ | — | **disqualified**: Heroku, Perforce/Cisco, and the entire security industry respectively |
| ~~**clickllm**~~ | — | **disqualified**: mis-signals category, saturated suffix, sells the commodity half |

## Collisions accepted

`onpar` is a common English phrase, so it is not a clean sweep. These are known and judged
acceptable — recorded here so the decision reads honestly rather than as a name nobody else
had thought of:

- **[On Par Analytics](https://github.com/onpar)** holds the `github.com/onpar` account. We
  publish from `github.com/dshakes/onpar`, so there is no path conflict.
- **[poy/onpar](https://github.com/poy/onpar)** is a Go parallel-testing framework. Different
  ecosystem and different registry — we publish to PyPI, crates.io, npm and Homebrew, none
  of which it occupies. It does mean we are the second `onpar` in a GitHub search.
- **[OnPar Technologies](https://www.onpartech.com/)** is an IT-services firm holding
  `onpartech.com`. Different trade class, no product overlap.

Registry availability was verified before committing: `onpar`, `on-par` and `on_par` (one
name under PEP 503 normalisation) are all free on PyPI, `onpar` is free on npm, and
`onpar-core` / `onpar-gateway` are unclaimed on crates.io. **Trademark clearance is a
register search, not a web search, and has not been done.**

## Where this sits against the field

Two companies sell an adjacent promise, and the contrast is the positioning:

- **[Understudy Labs (YC S26)](https://www.ycombinator.com/companies/understudy-labs)** sells
  *effortless* — "don't use their models, use yours."
- **[Nadir](https://getnadir.com/)** sells *cheaper* — a verifier-gated cascade router.
- **onpar sells *proven*.** Not the swap and not the saving, but the evidence that licenses
  either one. The word "effortless" is a promise about the migration; "on par" is a claim
  about the model, and a claim is a thing you can be wrong about in public.

## Positioning

1. **The verdict is the product.** Not the sizing arithmetic, not the eval harness — the
   sentence that joins them: *this model, on your hardware, at this concurrency, scores this
   on your traffic.* Nobody else produces that sentence, because it needs both halves.
2. **Lead with the regret.** Publish where open models lose. The honest failure is what
   makes the wins credible — and it's the thing no vendor-authored comparison will ever do.
3. **`?` is a feature, and says so out loud.** A cell with too little evidence prints `?`
   rather than a fabricated score. Every competitor prints a number. Being the tool that
   refuses is the whole brand.
4. **Neutral by construction.** Apache-2.0, zero telemetry, zero egress, runs local. Every
   other answer to "should you self-host" is sold by someone with a stake in the answer —
   a GPU vendor or a token vendor. That is an unoccupied trust position and the only
   coherent one for this product.
5. Migration, with receipts.
