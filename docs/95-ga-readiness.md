# GA readiness

**Assessed 2026-08-17 against 1.3.3.** Written by running the product as a
consumer would, from the published packages — not by reading the source.

The badge moved from `pre-alpha` to `beta` at 1.3.3. That is a claim about the
**install**, not the code: every stage command ships, all four registries carry
matching artifacts, and the three-package install is documented and verified.

It is deliberately not `stable`. Blocker #2 below — `observe` has never been
pointed at real traffic — is the one that would have to close first, because it
is the invariant with the highest cost of being wrong. A badge is a published
claim like any other number here, and it gets the same rule: say what is true,
and say the confidence.

---

## The headline

Every stage of the seven-stage loop **exists and runs**. What is not GA is the
*install*: the product is three PyPI distributions, and a consumer who follows
the README gets one of them.

```
$ pip install onpar && onpar observe --upstream https://api.openai.com/v1

  No gateway binary found.
  Install it:    pip install onpar-gateway
```

Install that, and the next thing you are told is:

```
  Heads up: captures will be written but cannot be read back yet.
  the compiled extension is not installed (pip install onpar-core).
```

Install all three and the warnings go to zero and the gateway serves. So the
product works — it just takes three commands nobody is told about up front, and
the failure between them is discovered one stage at a time.

**Both messages are good errors.** They name the exact fix, they refuse to
proceed silently, and the second explicitly says *"install it before you rely on
this run"* rather than letting a user capture traffic they cannot read. Nothing
here is broken. It is a packaging story that has not been finished.

---

## Stage by stage

Verified by invoking each command from a clean environment.

| Stage | Command | State |
|---|---|---|
| Observe | `onpar observe` | **runs** — needs `onpar-gateway` |
| Distill | `onpar distill` | **runs** — needs `onpar-core` |
| Fit | `onpar fit` | **runs, zero deps** — works under `uvx` with nothing installed |
| Prove | `onpar prove` | **runs** |
| Deploy | `onpar build` / `run` / `host` | **runs** |
| Cut over | `onpar migrate` | **runs** |
| Guard | `onpar guard` | **runs** |

Also shipping: `measure`, `receipt`, `brief`, `advise`, `box`, `watch`, `ui`,
`desktop`, `kernel`, and the catalog commands.

`docs/50-roadmap.md` still marks **Phase 0 as "in progress"** while Phases 1–6
carry no marker. That is stale: the commands for every phase exist. The roadmap
is describing an earlier repo.

---

## What is genuinely ready

- **`onpar fit`** is the strongest surface. Zero runtime dependencies, correct on
  the three sizing errors that are common in the wild (MoE totals, GQA `kv_heads`,
  MLA `kv_lora_rank`), and `--explain` prints the arithmetic for any row. It runs
  under `uvx` with nothing installed.
- **The refusal behaviour.** A cell with too little evidence prints `?`, and since
  #259 says **"not proven"** rather than borrowing a verdict. A perfect score on
  twelve items does not clear a 90% bar and the tool says so.
- **The four publish channels.** PyPI, npm, Homebrew and GitHub Releases all carry
  1.3.3, verified against the registries rather than a workflow's green tick.
- **Provenance.** Generated config carries a header saying what was chosen and
  that it runs with onpar uninstalled; a test asserts the generated artifact never
  shells back into the tool (NFR-4).

---

## Blockers, ranked

### 1. ~~Three distributions, one documented install~~ — CLOSED in 1.3.3

The README's quickstart was `uvx --from onpar onpar fit`, honest for the sizing
half and silent about the rest, so a user reaching `observe` or `distill`
discovered the other two packages by hitting the error.

The install section now names all three up front. Deliberately **not** solved by
making `onpar` depend on the other two: `uvx onpar fit` working on a machine with
nothing installed is a real guarantee, and a default dependency would spend it to
save a line of documentation.

Verified from the published 1.3.3 packages — all three install, and `onpar
observe` starts with zero warnings.


### 2. `observe` has never been pointed at real traffic

The gateway starts, serves, and captures. Nothing in this repo demonstrates it
against a live provider under load with real prompts. The capture path is where
NFR-3 lives; it is the least exercised code with the most sensitive job.

**What would make it GA:** one recorded end-to-end run against a real upstream,
with the capture log inspected to confirm no readable prompt text on disk.

### 3. ~~Phase markings in the roadmap are stale~~ — CLOSED in 1.3.3

Phase 0 read "in progress" while all seven stages shipped, so the document whose
job is to say what is finished described an earlier repo. Corrected, with a dated
note explaining what is and is not done.


### 4. The Intel-macOS wheel is built but never run on Intel

Cross-compiled on the arm64 runner because `macos-13` has never been assigned a
runner (three consecutive releases, `runner=NEVER ASSIGNED`). An Intel-only
regression would ship undetected. Recorded in the `build-compiled` matrix comment;
acceptable for now, not acceptable silently.

### 5. Human-agreement rate is claimed but not produced

Invariant 6 requires the judge model *and* the human-agreement rate in every
report. The judge is disclosed. The agreement rate needs a human-labelling loop
that does not exist, so that field can only print `?` today.

---

## Defects closed on the way to this assessment

Recorded because they share one shape — **a green signal over something that
didn't happen** — and that shape is the thing to keep watching:

| Defect | Why it stayed hidden |
|---|---|
| `tracing_subscriber_init()` was empty | Nine `warn!`/`error!` sites discarded, including `redaction pattern failed to compile` — an NFR-3 event reporting to nobody. Fixed in ADR-0019. |
| A capture **key** was committed | Swept in by `git add -A`; `.gitignore` covered neither the key nor the log, two lines below a comment promising captured traffic is never committed. Now gated by a test. |
| Published test count drifted | The guard's tolerance was `max(5, actual // 50)` — proportional, so it widened as the claim grew. Now fixed at 10. |
| Compiled wheels never published | One permanently-queued matrix leg blocked `publish-compiled` for three releases while reporting no failure. |
| Two negative assertions | Would have passed vacuously after the rename, silently retiring the guarantees they enforce. |

Each fix was verified by **making it fail** — injecting the violation, watching
the gate go red, reverting. A guard that has never been seen to fail is a guard
nobody has tested.

---

## The honest summary

**onpar 1.3.3 is a working tool that is honest about what it has not proven.**
The sizing half is genuinely good and genuinely dependency-free. The evidence half
is built, runs, and now installs in one documented command — but it has not been
exercised against real traffic.

That last clause is why the badge says `beta` and not `stable`. Blockers 1 and 3
closed in 1.3.3. Blocker 2 is the one that matters, and it cannot be closed from
inside the repo: it needs a real customer, a real upstream, and a willingness to
look at what lands on disk. Blockers 4 and 5 are known, bounded, and written down
where a reader will find them.

The rule this document holds itself to is the one the product holds its own output
to: **say what is true, and say the confidence.** A status badge is a published
claim like any other number here.
