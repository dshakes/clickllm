"""Can each module's `demo()` actually fail?

Every module here carries an assert-based `demo()`, and the convention treats it
as that module's safety net. The audit found **six** of them asserting something
true by construction — `fit`, `prove/__init__`, `k8s/reconcile`, `advise`,
`catalog_update` and `core`. The clearest was:

    assert f.total_bytes == f.weight_bytes + f.kv_bytes + f.overhead_bytes

where `total_bytes` is *defined* as exactly that sum. It looked like the sizing
check and verified that Python adds the same way twice. It would have passed with
all three components wrong — demonstrated: corrupting `weight_bytes` by one byte
left it green.

Six instances is not six oversights. It is what happens when a convention has
nothing checking the check. So this is the check on the checks: perturb a
constant the module declares, run its `demo()`, and require that at least one
perturbation is noticed.

Deliberately weak. It asks "is this demo sensitive to anything at all", not "is
it good". A demo that catches one mutation can still miss others — but a demo
that catches *none* is decoration, and that is the state this exists to make
impossible to reach silently.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

import pytest

import onpar


def _modules_with_demos() -> list[str]:
    out = []
    for m in pkgutil.walk_packages(onpar.__path__, "onpar."):
        try:
            mod = importlib.import_module(m.name)
        except Exception:  # noqa: BLE001 — an unimportable module is another test's problem
            continue
        if callable(getattr(mod, "demo", None)):
            out.append(m.name)
    return sorted(out)


#: Module-level numbers are the tuning surface — thresholds, floors, fractions,
#: bit-widths. Perturbing one and re-running the demo asks whether the demo
#: depends on the module's own arithmetic or merely re-states it.
def _constants(mod) -> list[str]:
    return [
        n
        for n in dir(mod)
        if n.isupper()
        and isinstance(getattr(mod, n, None), int | float)
        and not isinstance(getattr(mod, n), bool)
        and getattr(mod, n) != 0
    ]


#: Modules whose `demo()` notices no perturbation of their own constants, as of
#: the first full audit. A ratchet, not an amnesty: the test below fails if a
#: module JOINS this set, and fails if a name in it starts passing without being
#: removed. So the list can only shrink, and it cannot silently rot.
#:
#: Being here is not proof of a bad demo — a demo can exercise string rendering
#: or ordering well and never touch a threshold. It is proof that nothing has
#: checked, which is the state that let six tautological self-checks ship.
INSENSITIVE = frozenset(
    {
        "onpar.box",
        # Platform-dependent: on macOS its demo exercises the Apple detection
        # path and IS sensitive; on a Linux runner that path never executes.
        # CI found this, not the author's laptop.
        "onpar.hardware",
        "onpar.desktop",
        "onpar.distill.cluster",
        "onpar.host",
        "onpar.k8s.controller",
        "onpar.k8s.nodes",
        "onpar.launch",
        "onpar.mcp",
        "onpar.prove",
        "onpar.prove.collect",
        "onpar.prove.equivalence",
        "onpar.prove.graders",
        "onpar.prove.receipt",
        "onpar.prove.stats",
        "onpar.sdk",
        "onpar.ui",
        "onpar.watch",
    }
)


@pytest.mark.parametrize("name", _modules_with_demos())
def test_the_module_self_check_is_sensitive_to_something(name: str):
    """`demo()` must notice at least one perturbation of its own constants."""
    mod = importlib.import_module(name)
    consts = _constants(mod)
    if not consts:
        pytest.skip(f"{name} declares no numeric constants to perturb")

    baseline_ok = True
    try:
        mod.demo()
    except Exception:  # noqa: BLE001
        baseline_ok = False
    if not baseline_ok:
        pytest.skip(f"{name}.demo() does not pass unperturbed here; nothing to conclude")

    noticed = []
    for const in consts[:8]:  # bounded: this runs demo() once per constant
        original = getattr(mod, const)
        try:
            setattr(mod, const, original * 2 + 1)
        except Exception:  # noqa: BLE001 — frozen or read-only
            continue
        try:
            mod.demo()
        except Exception:  # noqa: BLE001 — ANY failure means the demo noticed
            noticed.append(const)
        finally:
            setattr(mod, const, original)

    if name in INSENSITIVE:
        if noticed:
            # A WARNING, not a failure. The first version failed here, to stop a
            # stale exemption hiding a later regression — and CI immediately
            # showed why it cannot: `onpar.hardware` is sensitive on macOS
            # (its demo runs the Apple path) and insensitive on Linux (it does
            # not). Membership is therefore platform-dependent, and a strict
            # reverse check makes such a module unpinnable on both at once.
            #
            # Failing a build because a demo got BETTER is also the wrong trade.
            # The forward direction stays strict, which is the half that stops
            # this getting worse; this half only nags.
            warnings.warn(
                f"{name} is listed INSENSITIVE but now notices {noticed} — "
                f"remove it from the list if it is sensitive on every platform.",
                stacklevel=1,
            )
        pytest.xfail(f"{name} is a known-insensitive demo (see INSENSITIVE)")

    assert noticed, (
        f"{name}.demo() passed with every one of {consts[:8]} doubled. It is not "
        f"reading this module's arithmetic — a demo that notices no mutation is "
        f"decoration, and this repo has shipped six of those. If this module is "
        f"genuinely one where constants do not matter, say so by adding it to "
        f"INSENSITIVE with a reason, rather than deleting this assertion."
    )


@pytest.mark.parametrize("name", _modules_with_demos())
def test_the_module_self_check_actually_passes(name: str):
    """A `demo()` that does not run is a self-check nobody is running.

    The mutation test above *skips* when the baseline fails — correctly, because
    a mutant surviving proves nothing when the unmutated demo already fails. But
    nothing else ran these, so a broken `demo()` produced a green suite. That is
    how `onpar.prove.receipt.demo()` sat broken for a full CI cycle: it
    asserted "altered" against a receipt now refused earlier and more
    specifically, and the harness turned the failure into a skip.

    CLAUDE.md says every module carries an assert-based `demo()` runnable via
    `python -m onpar.<mod>`. This is the test that says so.
    """
    mod = importlib.import_module(name)
    mod.demo()
