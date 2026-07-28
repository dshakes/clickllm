# Deploying clickllm on Kubernetes

Two pieces, useful independently.

## `kubectl clickllm` — the plugin

Put `kubectl-clickllm` on your `PATH`. kubectl discovers any executable named
`kubectl-*` and exposes it as a subcommand; there is no manifest to register and
nothing to compile.

```bash
kubectl clickllm nodes                    # what this cluster can actually run
kubectl clickllm fit --context 32k        # which models fit the best node
kubectl clickllm plan -f workload.yaml    # the Deployment an InferenceWorkload produces
```

`plan` runs **the same reconcile the controller runs**, so what it prints is what
would be applied. It is a dry run that cannot drift from the real path, because
it is the real path.

## `kind: InferenceWorkload` — the CRD

```bash
kubectl apply -f deploy/crd.yaml
```

Declare what the workload *is*, not how to run it:

```yaml
apiVersion: clickllm.dev/v1alpha1
kind: InferenceWorkload
metadata: {name: triage, namespace: ml}
spec:
  model: Qwen/Qwen3-32B
  workload: interactive       # or batch / realtime
  concurrency: 8
  context: 32768
  prefixSharing: 0.9          # agent fleets on one system prompt are routinely 0.8+
  structuredOutput: true
```

No image. No command. No flags. Those are derived from this plus the real
capacity of your nodes — which is the only reason to run a controller rather
than write a Deployment.

`kubectl describe iw triage` then tells you **why** each flag is what it is, and
what the chosen engine could not express:

```
Status:
  Engine:         sglang
  Engine Reason:  90% of prompt tokens are shared across requests. RadixAttention
                  reuses that prefix's KV instead of recomputing it per request…
  Knobs:
    Setting:  prefix_reuse    Value: True
    Why:      90% of prompt tokens repeat across requests; caching them skips
              prefill entirely for that span.
  Gaps:
    structured_output: the grammar-backend flag was not on the verified page.
```

### Two things worth knowing before you apply it

**`spec.phase` is yours.** The controller lowers it on a regression — a rollback
has to work at 3am with nobody watching — and it will never raise it. Moving more
production traffic onto a candidate is a human decision. That rule is enforced
independently here, in `prove.gate`, and in the gateway's control surface.

The controller learns about a regression from an annotation the quality gate
writes onto the resource:

```bash
kubectl annotate iw triage clickllm.dev/regressed=true   clickllm.dev/regression-reason="extract fell to 61% [54-68]"
```

Two processes rather than one, deliberately: the gate needs the eval corpus and
the controller needs cluster credentials, and giving either one both is a larger
blast radius than the feature is worth. Only a literal `true` triggers a
rollback — a typo or an `unknown` is treated as *not regressed*, because rolling
back on a value nobody understood would make a stray annotation an incident.

**It polls, it does not watch.** The controller shells out to `kubectl` rather
than embedding a Kubernetes client, which keeps `clickllm fit` dependency-free
and inherits your kubeconfig, contexts and cloud exec plugins for free. The cost
is that it reacts within an interval instead of milliseconds. For an input that
changes when someone edits YAML or a node joins, that is the right trade — and if
it stops being one, only `controller.py` changes.

## Running the controller

```bash
python -m clickllm.k8s.controller                    # loop, all namespaces
python -m clickllm.k8s.controller -n ml --interval 15
python -m clickllm.k8s.controller --once             # one pass — for a CronJob
python -m clickllm.k8s.controller --dry-run          # applies nothing
python -m clickllm.k8s.controller --self-check       # built-in self-check
```

## A node it cannot size

Kubernetes reports GPU *counts*, never capacities. Memory comes from a label that
NVIDIA's GPU Feature Discovery publishes, so a cluster without it produces:

```
mystery-gpu   nvidia   4   —   the cluster does not publish nvidia.com/gpu.memory
```

That node is skipped and the reason is on the resource. clickllm will not invent
80 GB because the product string says H100.
