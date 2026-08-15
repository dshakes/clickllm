"""Every surface must answer the same question with the same answer.

The README says the CLI, the MCP server, the SDK and the workbench are "four
faces of one implementation" and that "an agent gets the same answer you do".
They were not, and it was checkable:

    ui  feasible row: model_id name quant weights_gb kv_gb total_gb slow …
    mcp feasible row: id       quant           total_gb …

    ui  envelope: runtime              warnings
    mcp envelope: recommended_runtime

Same *answer* — the same five models in the same order, the same engine with
the same reason — in two incompatible **shapes**. `model_id` vs `id`,
`runtime` vs `recommended_runtime`, and four fields the agent surface simply
never sees (`weights_gb`, `kv_gb`, `name`, `slow`).

That is subtler than a wrong answer and worse to live with: a caller reading
`runtime` gets `None` from the surface that spells it `recommended_runtime`,
and `None` renders as "no recommendation" rather than as an error. Both
surfaces already imported `fit.py`. **Sharing helpers is not sharing a
contract** — each re-assembled the result its own way, and nothing compared
them.

So the anti-drift mechanism is a test, not the shared code. A new surface, or a
new field on an old one, fails here until it agrees.

What is deliberately *not* asserted: prose. The CLI renders a table and MCP
returns JSON, and requiring them to match character-for-character would forbid
the thing surfaces exist to do. What must match is the *answer* — the facts a
user would act on.
"""

from __future__ import annotations

import pytest

from onpar import hardware, mcp, sdk, ui
from onpar.hardware import Hardware

GB = 1024**3

#: The same synthetic machine the goldens use, for the same reason: a
#: conformance test that reads the host is partly a test of the host.
MACHINE = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * GB,
    usable_bytes=96 * GB,
    bandwidth_gbps=546.0,
    cores=16,
)


@pytest.fixture(autouse=True)
def _fixed_machine(monkeypatch):
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)


# --- the facts each surface must agree on ----------------------------------------


def _feasible_ids(payload: dict) -> list[str]:
    """Model ids reported as fitting, however this surface spells the field.

    Tolerant on purpose, and only here: the *values* agreeing is a separate
    claim from the *schema* agreeing, and `test_feasible_rows_share_one_schema`
    is what fails on the spelling. A strict lookup in this helper would raise a
    KeyError and report a missing field where the real finding is a rename —
    which is how the first version of this file reported "ui says the best
    model is None" when ui simply called it `model_id`.
    """
    rows = payload.get("feasible") or []
    out = []
    for r in rows:
        if isinstance(r, dict):
            out.append(str(r.get("id") or r.get("model") or r.get("model_id") or ""))
        else:
            out.append(str(r))
    return [x for x in out if x]


def _runtime(payload: dict) -> str | None:
    """The recommended runtime, under whichever key this surface used.

    Both spellings are accepted *here* and nowhere else: the point of this test
    is to fail while they differ, and the assertion below is what fails. If this
    helper were strict it would raise a KeyError and report a missing key rather
    than a disagreement.
    """
    v = payload.get("runtime") or payload.get("recommended_runtime")
    if isinstance(v, dict):
        return str(v.get("name") or v.get("engine") or "")
    return str(v) if v else None


def test_every_surface_reports_the_same_feasible_models():
    """The core question — what can I run — asked of each face."""
    answers = {
        "sdk": sdk.fit(context="32k", concurrency=8).to_dict(),
        "ui": ui._fit("32k", 8),
        "mcp": mcp._fit(context="32k", concurrency=8),
    }
    ids = {name: _feasible_ids(payload) for name, payload in answers.items()}
    distinct = {tuple(v) for v in ids.values()}
    assert len(distinct) == 1, (
        "surfaces disagree about what fits on the same machine:\n  "
        + "\n  ".join(f"{k}: {v[:4]}" for k, v in ids.items())
    )


def test_every_surface_recommends_the_same_runtime():
    """The values already agree — both said `mlx`, with the same reason.

    Asserted anyway, because that agreement is currently an accident of both
    calling the same helper, not a guarantee. The *shape* is what diverged, and
    the two tests below are what fail on it.
    """
    answers = {
        "ui": ui._fit("32k", 8),
        "mcp": mcp._fit(context="32k", concurrency=8),
    }
    runtimes = {k: _runtime(v) for k, v in answers.items()}
    assert all(runtimes.values()), f"a surface reported no runtime at all: {runtimes}"
    assert len(set(runtimes.values())) == 1, f"surfaces recommend different engines: {runtimes}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known and recorded in ADR-0016. Converging these renames a field in "
        "three published shapes at once — the MCP tool schema, `onpar fit "
        "--json`, and the SDK's to_dict — all of which became contracts when "
        "1.0.0 shipped. That is a deliberate, versioned change, not a drive-by "
        "fix inside the PR that adds the test. strict=True so this fails loudly "
        "the moment it is fixed and the marker must be removed."
    ),
)
def test_feasible_rows_share_one_schema():
    """The rows agree on which models fit and disagree on how to say so.

    `ui` spells the model `model_id` and carries `weights_gb`, `kv_gb`, `name`
    and `slow`; `mcp` spells it `id` and carries none of those. So an agent
    cannot see the weights/KV split that a human sees in the browser, over the
    same question, from the same solver.
    """
    u = ui._fit("32k", 8)["feasible"]
    m = mcp._fit(context="32k", concurrency=8)["feasible"]
    assert u and m, "nothing fits; this test would be vacuous"
    only_ui = sorted(set(u[0]) - set(m[0]))
    only_mcp = sorted(set(m[0]) - set(u[0]))
    assert not (only_ui or only_mcp), (
        f"one row, two schemas — ui-only {only_ui}, mcp-only {only_mcp}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known and recorded in ADR-0016. Converging these renames a field in "
        "three published shapes at once — the MCP tool schema, `onpar fit "
        "--json`, and the SDK's to_dict — all of which became contracts when "
        "1.0.0 shipped. That is a deliberate, versioned change, not a drive-by "
        "fix inside the PR that adds the test. strict=True so this fails loudly "
        "the moment it is fixed and the marker must be removed."
    ),
)
def test_every_surface_uses_one_name_for_the_recommended_runtime():
    """The shape, not just the value. Two spellings of one field is how the
    values drifted apart in the first place — a caller reading `runtime` got
    `None` from the surface that called it `recommended_runtime`, and `None`
    renders as "no recommendation" rather than as an error.
    """
    keys = {
        "ui": set(ui._fit("32k", 8)),
        "mcp": set(mcp._fit(context="32k", concurrency=8)),
    }
    spellings = {k for keyset in keys.values() for k in keyset if "runtime" in k}
    assert len(spellings) <= 1, (
        f"the recommended runtime is spelled {sorted(spellings)} across surfaces — "
        "pick one and let the others render it"
    )


def test_the_answer_envelope_agrees_on_what_was_asked():
    """Echoing the question back is how a caller knows which answer it holds.
    A surface that drops it makes its reply unattributable."""
    for name, payload in (
        ("ui", ui._fit("32k", 8)),
        ("mcp", mcp._fit(context="32k", concurrency=8)),
    ):
        assert payload.get("context") == 32768, f"{name} did not echo the context"
        assert payload.get("concurrency") == 8, f"{name} did not echo the concurrency"


def test_every_surface_agrees_on_where_a_model_runs():
    """`where` is the inverse question, and it is asked of two surfaces."""
    u = ui._where("llama-3.1-8b", "32k", 1)
    m = mcp._where("llama-3.1-8b", context="32k", concurrency=1)

    def verdicts(payload: dict) -> dict[str, bool]:
        """profile id -> whether that surface says the model fits there.

        Comparing verdicts, not just which profiles were considered — a
        regression that flips one surface's fit/no-fit answer for a profile
        while leaving the profile list untouched would still pass a
        set-of-ids-only check. ui spells this `feasible`, mcp spells it
        `fits`; both are `Placement.fit is not None`.
        """
        rows = payload.get("placements") or payload.get("hardware") or []
        out = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            profile = str(r.get("profile") or r.get("id") or "")
            out[profile] = bool(r.get("feasible", r.get("fits")))
        return out

    vu, vm = verdicts(u), verdicts(m)
    assert vu == vm, (
        f"ui says {sorted(p for p, ok in vu.items() if ok)[:3]} fit, "
        f"mcp says {sorted(p for p, ok in vm.items() if ok)[:3]} fit"
    )


def test_a_surface_cannot_quietly_stop_answering():
    """The control. Every assertion above compares surfaces to each other, so
    they would all pass if every surface returned the same empty dict."""
    for name, payload in (
        ("sdk", sdk.fit(context="32k", concurrency=8).to_dict()),
        ("ui", ui._fit("32k", 8)),
        ("mcp", mcp._fit(context="32k", concurrency=8)),
    ):
        assert _feasible_ids(payload), f"{name} reported nothing fits on a 96 GB machine"


# --- invariant 6, across every machine-readable surface --------------------------


def _throughput_fields(row: dict) -> tuple[str | None, object]:
    """The tok/s field and its value, whatever this surface calls it."""
    for k in ("tokens_per_sec_estimate", "tokens_per_sec"):
        if k in row:
            return k, row[k]
    return None, None


def test_every_machine_readable_surface_says_the_throughput_is_an_estimate():
    """Invariant 6: never report a number without its confidence.

    Every throughput figure in this project is a memory-bandwidth roofline.
    `onpar fit --explain` says so — "roofline estimate, not measured" — and
    `mcp`, `sdk` and `ui` all carry `estimate_basis` beside the number.

    `onpar fit --json` carried neither. A human reading the table sees a `~`
    and a pointer to `--explain`; a program reading the JSON sees
    `tokens_per_sec: 15` and has no way to know it is a projection. That is the
    surface most likely to be piped into someone's capacity planning.

    This is the reason the four shapes are worth converging at all: while each
    surface assembles its own result, a disclosure is per-surface and one of
    them will always be missing it.
    """
    import contextlib
    import io
    import json as _json

    from onpar import cli

    def _cli_json() -> dict:
        """Run the command; do not read its golden.

        The first version of this read `tests/golden/fit_json.txt`, which made
        it a test of a recorded file rather than of the code — deleting the
        disclosure from `cli.py` left it passing, and the control caught that.
        A conformance test that reads an artifact is only checking that the
        artifact still says what it said.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.main(["fit", "--context", "32k", "--concurrency", "8", "--json"])
        return _json.loads(buf.getvalue())

    surfaces = {
        "ui": ui._fit("32k", 8),
        "mcp": mcp._fit(context="32k", concurrency=8),
        "sdk": sdk.fit(context="32k", concurrency=8).to_dict(),
        "cli --json": _cli_json(),
    }
    missing = []
    for name, payload in surfaces.items():
        rows = payload.get("feasible") or []
        assert rows, f"{name} reported nothing feasible"
        row = rows[0]
        field, value = _throughput_fields(row)
        assert field, f"{name} reports no throughput at all"
        if value is None:
            continue  # a model with no estimate is already honest
        discloses = "estimate" in field or bool(row.get("estimate_basis"))
        if not discloses:
            missing.append(f"{name} ({field}={value!r}, no estimate_basis)")

    assert not missing, "these hand a program a roofline projection as a bare number: " + "; ".join(
        missing
    )
