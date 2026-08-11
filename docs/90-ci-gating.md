# Gating a deploy on a proof that still holds

A receipt is a claim about a moment: *on this eval set, against this incumbent,
these clusters cleared this bar*. Moments end. The model behind a name changes,
the traffic moves, and the proof that justified running an open model in
production quietly stops being about production.

`clickllm guard` is the check, and it exits nonzero, so it works as a CI step
without anything parsing its output.

```bash
clickllm guard receipt.json
echo $?     # 0 — still holds
```

## Exit codes

| code | meaning |
|---|---|
| `0` | the receipt still describes production |
| `1` | it does not, or `--fail-on any` and there is any finding at all |
| `2` | the receipt could not be read, or an input was malformed |

`2` is deliberately distinct. A gate that treats "could not read the receipt" as
"the receipt failed" is honest but noisy; one that treats it as a pass is a
gate that opens when it breaks. Distinguish them and you can alert on `2`
without blocking on it.

## Which findings should stop a deploy

`guard` separates three things that most tools collapse into one *stale* flag,
because only two of them mean you no longer know whether production is adequate:

| finding | what happened | voids the proof |
|---|---|---|
| `model_changed` | the model behind the name is not the one you proved | **yes** |
| `traffic_moved` | real traffic drifted outside the eval set | **yes** |
| `traffic_uncovered` | a share of traffic the eval set never covered | **yes** |
| `aged` | nothing observed changed; nobody has re-checked in a while | no |
| `new_candidate` | something new was released | no |
| `uncheckable` | you supplied fingerprints and the receipt records none | no |

A new model existing does not make an old proof false. That is worth stating
because the instinct is to treat any news as staleness, and doing so trains
people to ignore the alert.

**`--fail-on` is a policy choice and we do not make it for you.** The default
(`invalidating`) fails only on the first three. `--fail-on any` also fails on age
and new releases, which is usually what a release gate wants: *nothing observed
has changed, but nobody has looked in eleven months* is not a thing to deploy on
silently.

## GitHub Actions

```yaml
name: proof
on: [pull_request]

jobs:
  receipt-still-holds:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6

      # The receipt is committed to the repo. That is the point of it being a
      # file: the thing that justifies running this model in production lives
      # beside the code it justifies, and moves under review like anything else.
      - name: does the proof still hold
        run: |
          uvx --from clickllm-cli clickllm guard proof/receipt.json \
            --fingerprints proof/fingerprints.json \
            --traffic proof/traffic.json \
            --fail-on any
```

`fingerprints.json` maps a model name to whatever identifies the weights you are
actually serving — a digest, a revision, a build id.

**It takes two of them, and both must exist.** The receipt records what it was
issued against; the guard reads what is running now; a mismatch is the finding.
So the receipt has to have been issued with them:

```bash
clickllm prove evalset.json --candidate-endpoint $CAND \
                            --fingerprints proof/issued-against.json \
                            --out proof/receipt.json
```

A receipt without them cannot answer the question. Until recently `guard` handled
that by reporting *still holds* — the loop over the receipt's fingerprints simply
ran zero times, so a caller who had gone to the trouble of collecting current
ones was told everything was fine. That now surfaces as a finding of its own:

```
[note] fingerprints: you supplied 2 current fingerprint(s) and this receipt
       records none, so nothing was compared.
→ re-issue the receipt with --fingerprints: nothing here checked whether the
  model changed
```

It does **not** void the receipt — *I could not tell whether the model changed*
is not evidence that it did. But it is a finding, so `--fail-on any` stops a
deploy on it, which is the correct behaviour for a release gate: a proof nobody
can check is not a proof you should ship behind.

`traffic.json` is the current cluster shares. Produce it from a fresh capture
window:

```bash
clickllm observe --upstream $PROVIDER    # for a while
clickllm distill --out fresh.json
jq '.shares' fresh.json > proof/traffic.json
```

## Re-verifying the receipt itself

`guard` asks whether the claim is still true. A different question is whether
the receipt is the one that was issued — for that, re-run the eval set and
compare:

```bash
clickllm prove evalset.json --candidate-endpoint $CAND --out rerun.json
clickllm receipt rerun.json --against proof/receipt.json
echo $?     # 0 — the evidence agrees
```

This compares **evidence, not identity**. A rerun carries a later date and often
a newer build, so the digests differ while the claims match, and the output says
which fields differ so a reader does not go looking for a discrepancy that is
not one.

That is a stronger property than a signature. A signature says *we said this*;
reproduction says *and it is true*, and anyone holding the eval set can check it
without trusting whoever issued it.

## What this cannot do

It cannot tell you the proof was ever any good. A receipt over twelve items with
a wide interval is still a receipt, and `guard` will happily report that it still
holds. Read the receipt — `clickllm receipt receipt.json` prints what is proven,
what must stay on the incumbent, and what is unproven either way, in that order.

Nothing here moves traffic. `guard` reports; a human decides. See
[ADR-0015](adr/0015-in-the-path-only-while-migrating.md) for the datapath
boundary and invariant 8 for why a cutover is never automatic.
