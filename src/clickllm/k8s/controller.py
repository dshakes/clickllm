"""The reconcile loop — thin glue around a function that is already tested.

Everything that decides anything lives in [`clickllm.k8s.reconcile`], as a pure
function of a spec and a list of nodes. This file is the part that talks to the
API server, and it is deliberately the smallest thing that can.

## Why `kubectl` instead of a client library

A Kubernetes client is a real dependency with a real release treadmill, and it
would have to re-implement kubeconfig parsing, context selection, and the cloud
exec-credential plugins that EKS and GKE rely on. `kubectl` already does all of
that, is already installed wherever this is useful, and keeps `clickllm fit` free
of runtime dependencies.

The cost is honest and worth naming: **this polls, it does not watch.** A watch
would react in milliseconds; this reacts within an interval. For a controller
whose input changes when someone edits a YAML or a node joins a cluster, that is
the right trade — and if it ever is not, the reconcile function does not change,
only this file does.

## What it will and will not do

It applies Deployments and writes status. It **lowers** `spec.phase` on a
regression, because a rollback has to work at 3am with nobody watching. It never
raises it. That rule is enforced independently here, in `prove.gate`, and in the
gateway's control surface — three implementations, because the cost of getting it
wrong is production traffic on an unproven model.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

from clickllm.k8s.nodes import read_cluster
from clickllm.k8s.reconcile import Reconciled, reconcile

__all__ = ["Kubectl", "RegressionCheck", "demo", "gate_from_annotation", "reconcile_once", "run"]

#: A function the loop calls per workload to ask "has this regressed?".
#:
#: Returns `(regressed, reason)`. Injected rather than computed here because
#: judging a regression needs eval data this package deliberately does not
#: reach for — but the loop MUST consult something, or the rollback path is
#: documentation rather than behaviour.
RegressionCheck = Callable[[dict], tuple[bool, str]]

#: Seconds between passes. Long enough not to hammer the API server, short
#: enough that a node joining is noticed before anyone files a ticket.
DEFAULT_INTERVAL = 30


#: kubectl verbs that only read. Everything else is treated as mutating under
#: `--dry-run`, so a verb nobody thought about is suppressed rather than run.
_READ_ONLY_VERBS = frozenset({"get", "describe", "explain", "logs", "version", "api-resources"})


@dataclass(frozen=True, slots=True)
class Kubectl:
    """A `kubectl` invoker.

    A struct rather than bare functions so tests can substitute one that records
    calls instead of making them — which is the only way the apply path gets
    exercised without a cluster.
    """

    context: str | None = None
    timeout: int = 30
    dry_run: bool = False

    def run(self, args: list[str], stdin: str | None = None) -> str:
        """Invoke kubectl, returning stdout.

        Raises:
            RuntimeError: carrying kubectl's own stderr, so a permissions or
                context failure is diagnosable rather than generic.
        """
        cmd = ["kubectl", *args]
        if self.context:
            cmd += ["--context", self.context]
        # Every mutating verb, not just `apply`. This was `args[0] == "apply"`,
        # so `patch` slipped through — and `patch_status()` and `demote()` both
        # issue `patch`. An operator running `--dry-run --once` to preview a
        # pass against a live cluster really patched `status.conditions`, and
        # really demoted `spec.phase` (canary -> shadow) when the gate saw a
        # regression. A flag whose whole contract is "changes nothing" was
        # performing the one action in this system that moves production
        # traffic.
        #
        # Allow-list the read-only verbs instead of naming the mutating ones:
        # the failure was an unlisted mutation, so the default must be to
        # suppress. A verb added later is dry-run-safe until someone says
        # otherwise, which is the direction that fails safe.
        if self.dry_run and args and args[0] not in _READ_ONLY_VERBS:
            cmd += ["--dry-run=server"]
        try:
            r = subprocess.run(
                cmd, input=stdin, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except FileNotFoundError as e:
            raise RuntimeError("kubectl is not installed or not on PATH") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"kubectl {args[0]} timed out after {self.timeout}s") from e
        if r.returncode != 0:
            raise RuntimeError(f"kubectl {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout

    def workloads(self, namespace: str | None = None) -> list[dict]:
        """Every InferenceWorkload, cluster-wide or in one namespace."""
        scope = ["-n", namespace] if namespace else ["--all-namespaces"]
        out = self.run(["get", "inferenceworkloads.clickllm.dev", *scope, "-o", "json"])
        return json.loads(out).get("items", [])

    def apply(self, obj: dict) -> str:
        """Apply one object from stdin."""
        return self.run(["apply", "-f", "-"], stdin=json.dumps(obj))

    def patch_status(self, name: str, namespace: str, status: dict) -> str:
        """Write the status subresource."""
        return self.run(
            [
                "patch",
                "inferenceworkloads.clickllm.dev",
                name,
                "-n",
                namespace,
                "--subresource=status",
                "--type=merge",
                "-p",
                json.dumps({"status": status}),
            ]
        )

    def demote(self, name: str, namespace: str, phase: str) -> str:
        """Lower `spec.phase`. The only spec field the controller ever writes."""
        return self.run(
            [
                "patch",
                "inferenceworkloads.clickllm.dev",
                name,
                "-n",
                namespace,
                "--type=merge",
                "-p",
                json.dumps({"spec": {"phase": phase}}),
            ]
        )


#: Annotation the quality gate writes when it finds a regression.
#:
#: The gate runs where the eval data is — it reads the capture store, scores
#: candidate against baseline, and stamps its verdict here. The controller then
#: acts on it. Two processes rather than one because the gate needs the eval
#: corpus and the controller needs cluster credentials, and giving either one
#: both is a larger blast radius than the feature is worth.
REGRESSION_ANNOTATION = "clickllm.dev/regressed"
REGRESSION_REASON_ANNOTATION = "clickllm.dev/regression-reason"


def gate_from_annotation(wl: dict) -> tuple[bool, str]:
    """Default regression check: read the gate's verdict off the resource.

    Treats anything other than a literal `"true"` as *not* regressed — including
    a missing annotation, a typo, or `"unknown"`. Rolling back on a value we do
    not understand would make a stray annotation a production incident.
    """
    ann = (wl.get("metadata", {}) or {}).get("annotations", {}) or {}
    regressed = str(ann.get(REGRESSION_ANNOTATION, "")).strip().lower() == "true"
    reason = ann.get(REGRESSION_REASON_ANNOTATION, "") or "the quality gate reported a regression"
    return regressed, reason


def _ready(r: Reconciled) -> bool:
    """Whether this pass's own status says the workload is fit to apply.

    No Ready condition means ready — a defensive default for a `Reconciled`
    built without one.

    The previous version of this docstring claimed the rollback path is the
    no-conditions case. It is not: that path sets `Ready=True` with reason
    `RolledBack`, so this predicate passed and the fit gate never fired there.
    A comment stating an assumption nobody had checked, in the fix for exactly
    that shape. The rollback now returns no objects at all, so it does not reach
    this predicate — but the docstring is corrected rather than deleted, because
    the wrong version is the more instructive artifact.
    """
    for c in r.status.get("conditions", ()):
        if c.get("type") == "Ready":
            return str(c.get("status", "True")).lower() == "true"
    return True


def reconcile_once(
    kc: Kubectl,
    namespace: str | None = None,
    nodes: list | None = None,
    regression_check: RegressionCheck | None = None,
) -> list[tuple[str, Reconciled]]:
    """One pass over every workload.

    `nodes` may be supplied to skip the cluster read — which is how this gets
    tested without a cluster, and cheaper than patching a module global that
    `python -m` would hand you a second copy of anyway.

    Failures are per-workload: one broken resource must not stop the others from
    reconciling, because the usual cause is a typo in someone's YAML and the
    blast radius should be their workload, not the cluster's.
    """
    nodes = read_cluster(kc.context) if nodes is None else nodes
    check = regression_check or gate_from_annotation
    results: list[tuple[str, Reconciled]] = []

    for wl in kc.workloads(namespace):
        meta = wl.get("metadata", {})
        name, ns = meta.get("name", "?"), meta.get("namespace", "default")
        ref = f"{ns}/{name}"
        try:
            # The rollback path is only real if something consults it. An
            # earlier version of this loop called reconcile() with no regression
            # input at all, which meant `regressed` was always False and the
            # automatic rollback — three paragraphs of documentation — could
            # never once have fired in production.
            regressed, reason = check(wl)
            r = reconcile(wl, nodes, regressed=regressed, regression_reason=reason)
            # Do not apply what this pass just declared unfit. Dead code until
            # ADR-0013 made `reconcile` actually size the model — two earlier
            # attempts at this gate were discarded precisely because nothing
            # ever reported Ready=False while returning objects (#114).
            #
            # The status is still written: "a workload that cannot be sized
            # still produces a status explaining why" is the documented and
            # useful half. What changes is that reporting a problem no longer
            # coexists with shipping it.
            if _ready(r):
                for obj in r.objects:
                    kc.apply(obj)
            if r.status:
                kc.patch_status(name, ns, r.status)
            if r.demote_to:
                kc.demote(name, ns, r.demote_to)
            results.append((ref, r))
        except Exception as e:  # noqa: BLE001 - deliberate; see below
            # Deliberately broad. This is a cluster-wide daemon, and any
            # exception escaping here takes down reconciliation for *every*
            # workload — so one person's malformed YAML would crash-loop the
            # controller for everybody. The blast radius of a bad resource must
            # be that resource. Reported on the object itself, because a
            # controller that only logs is one whose failures nobody sees.
            failed = Reconciled(
                status={
                    "conditions": [
                        {
                            "type": "Ready",
                            "status": "False",
                            "reason": "ReconcileFailed",
                            "message": str(e)[:400],
                        }
                    ]
                }
            )
            # The API server being unreachable is normal during a rolling
            # restart; the next pass retries. Failing here would turn a blip
            # into a crash loop.
            with contextlib.suppress(RuntimeError):
                kc.patch_status(name, ns, failed.status)
            results.append((ref, failed))
    return results


def run(
    kc: Kubectl | None = None,
    namespace: str | None = None,
    interval: int = DEFAULT_INTERVAL,
    max_passes: int | None = None,
    nodes: list | None = None,
    regression_check: RegressionCheck | None = None,
) -> None:
    """Reconcile forever.

    `max_passes` exists so this is runnable as a one-shot in CI or a CronJob —
    an operator you cannot invoke once is an operator you cannot test.
    """
    kc = kc or Kubectl()
    passes = 0
    while max_passes is None or passes < max_passes:
        try:
            for ref, r in reconcile_once(kc, namespace, nodes, regression_check):
                state = "ok" if r.ready else "blocked"
                extra = f" → demoted to {r.demote_to}" if r.demote_to else ""
                print(f"[clickllm] {ref}: {state}{extra}", flush=True)
        except RuntimeError as e:
            # The cluster being briefly unreachable is normal. Log and retry;
            # exiting would make a rolling API-server restart look like a crash.
            print(f"[clickllm] pass failed: {e}", flush=True)
        passes += 1
        if max_passes is not None and passes >= max_passes:
            break
        time.sleep(interval)


def demo() -> None:
    """Self-check. Run with `python -m clickllm.k8s.controller`."""
    from clickllm.k8s.nodes import GPU_MEMORY_LABEL

    NODES = {
        "items": [
            {
                "metadata": {"name": "h100", "labels": {GPU_MEMORY_LABEL: "81559"}},
                "status": {"allocatable": {"nvidia.com/gpu": "8", "memory": "2Ti", "cpu": "192"}},
            }
        ]
    }
    WORKLOADS = {
        "items": [
            {
                "metadata": {"name": "triage", "namespace": "ml"},
                "spec": {"model": "Qwen/Qwen3-32B", "workload": "interactive", "phase": "shadow"},
            },
            {
                "metadata": {"name": "broken", "namespace": "ml"},
                "spec": {"workload": "batch"},  # no model
            },
        ]
    }

    class Fake(Kubectl):
        """Records calls instead of making them."""

        def run(self, args, stdin=None):  # type: ignore[override]
            calls.append((tuple(args[:2]), stdin))
            if args[0] == "get" and "nodes" in args:
                return json.dumps(NODES)
            if args[0] == "get":
                return json.dumps(WORKLOADS)
            return ""

    from clickllm.k8s.nodes import from_json

    calls: list = []
    nodes = from_json(NODES)
    results = reconcile_once(Fake(), nodes=nodes)
    got = dict(results)
    assert "ml/triage" in got and got["ml/triage"].ready
    # The broken one is reported on the resource, not merely logged...
    assert not got["ml/broken"].ready
    assert got["ml/broken"].status["conditions"][0]["reason"] == "NoModel"
    # ...and it did not stop the healthy one from reconciling.
    assert got["ml/triage"].status["engine"] == "vllm"

    verbs = [c[0][0] for c in calls]
    assert "apply" in verbs, verbs
    assert "patch" in verbs, verbs

    # Nothing in a healthy pass writes spec.
    spec_writes = [s for a, s in calls if a[0] == "patch" and s and '"spec"' in str(s)]
    assert not spec_writes, "a healthy reconcile must never touch spec"

    # `run` is invocable once, so it is testable and CronJob-able.
    run(Fake(), max_passes=1, interval=0, nodes=nodes)

    print("k8s.controller: ok")


def main(argv: list[str] | None = None) -> int:
    """Entrypoint. `python -m clickllm.k8s.controller` runs the loop.

    It previously ran `demo()`, which meant the command `deploy/README.md` told
    operators to run started a self-check and exited — a documented entrypoint
    that did not do the documented thing.
    """
    import argparse

    p = argparse.ArgumentParser(prog="clickllm-controller")
    p.add_argument("-n", "--namespace", help="watch one namespace (default: all)")
    p.add_argument("--context", help="kubectl context")
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    p.add_argument("--once", action="store_true", help="one pass, then exit — for a CronJob")
    p.add_argument("--dry-run", action="store_true", help="server-side dry run; applies nothing")
    p.add_argument("--self-check", action="store_true", help="run the built-in self-check and exit")
    a = p.parse_args(argv)

    if a.self_check:
        demo()
        return 0
    try:
        run(
            Kubectl(context=a.context, dry_run=a.dry_run),
            namespace=a.namespace,
            interval=a.interval,
            max_passes=1 if a.once else None,
        )
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
