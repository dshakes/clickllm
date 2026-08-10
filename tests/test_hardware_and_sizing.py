"""Detection that must not invent a machine, a rented figure that must be the
rented one, and a solver that must not answer a question the model will refuse.

Three of these understate or overstate silently. The direction differs; being
silent is what they share.
"""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from clickllm import fit as F
from clickllm import hardware as H
from clickllm.catalog import load
from clickllm.hardware import Hardware

GB = 1024**3
_CARD = {"VRAM Total Memory (B)": "68702699520", "Card Series": "MI300X"}


def _rocm(payload: dict):
    """Run `_detect_amd` against a canned `rocm-smi --json` document."""

    class Result:
        stdout = json.dumps(payload)

    with (
        patch.object(H.shutil, "which", lambda _n: "/usr/bin/rocm-smi"),
        patch.object(H.subprocess, "run", lambda *_a, **_k: Result()),
    ):
        return H._detect_amd()


# --- a machine that is not the machine -------------------------------------------


def test_a_card_whose_vram_key_is_unrecognised_declines_rather_than_shrinking():
    """The skip could not tell "this entry is not a card" from "this entry IS a
    card whose key I no longer recognise".

    A ROCm release renaming the key for one card on a four-card box reported a
    three-card machine. It understates capacity, which is the safe direction,
    and it is silent, which is not — the same shape as the mixed-card bug in
    #146, where a note and the arithmetic disagreed and nothing said so.
    """
    payload = {f"card{i}": dict(_CARD) for i in range(3)}
    payload["card3"] = {"HBM Capacity (B)": "68702699520", "Card Series": "MI300X"}

    assert _rocm(payload) is None


def test_every_card_readable_is_the_whole_machine():
    hw = _rocm({f"card{i}": dict(_CARD) for i in range(4)})
    assert hw is not None
    assert hw.devices == 4
    assert round(hw.total_bytes / GB) == 256


def test_metadata_blocks_are_still_skipped_silently():
    """The negative control, and the reason the skip existed: `rocm-smi` emits
    non-card entries, and failing closed on those would mean never detecting an
    AMD box at all."""
    payload = {f"card{i}": dict(_CARD) for i in range(2)}
    payload["system"] = {"Driver version": "6.2.0"}
    hw = _rocm(payload)
    assert hw is not None and hw.devices == 2


def test_a_vram_figure_that_is_not_a_number_still_declines():
    payload = {"card0": {"VRAM Total Memory (B)": "lots", "Card Series": "MI300X"}}
    assert _rocm(payload) is None


# --- a rented figure that is not the rented one ----------------------------------


def test_the_v5e_bandwidth_is_the_primary_source_converted_not_a_secondary_quote():
    """#131 finding 2 said this was inflated ~5% and should read 819. It should
    not, and this test exists because I changed it and had to change it back.

    Google's v5e table says "HBM bandwidth per chip: 800 GiBps" and, in the same
    table, "ICI bandwidth: 400 GBps" — it distinguishes the units deliberately.
    800 GiB/s is 859 GB/s, which is what every other row here is in. The 819
    figure is what secondary sources quote, and 819 GB/s is 763 GiB/s, which
    contradicts the primary table.
    """
    from clickllm.hardware_catalog import PROFILES

    v5e = next(p for p in PROFILES if p.id == "tpu-v5e-8")
    assert v5e.bandwidth_gbps == 859
    assert round(800 * 1.073741824) == 859, "the conversion this row encodes"


def test_the_other_tpu_rows_were_left_alone():
    """The negative control: these match their published figures directly and a
    sweeping "fix" of the conversion would have moved them too."""
    from clickllm.hardware_catalog import PROFILES

    by_id = {p.id: p for p in PROFILES}
    assert by_id["tpu-v5p-4"].bandwidth_gbps == 2765
    assert by_id["tpu-v6e-8"].bandwidth_gbps == 1638


def test_every_multi_device_profile_prices_the_whole_shape():
    """The finding that is already fixed, kept as a guard: these three rows
    carried the per-*chip* list price while every NVIDIA multi-device row
    carried the whole-shape price, and `Placement` divides by aggregate
    throughput — so a 1-chip numerator over an 8-chip denominator understated
    $/Mtok eightfold."""
    from clickllm.hardware_catalog import PROFILES

    for p in PROFILES:
        if p.devices > 1 and p.hourly_usd is not None:
            per_device = p.hourly_usd / p.devices
            assert per_device < p.hourly_usd, p.id
            assert p.hourly_usd > 1.0, f"{p.id} looks like a per-device rate"


# --- a question the model will refuse --------------------------------------------


def _hw() -> Hardware:
    return Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=96 * GB,
        bandwidth_gbps=546.0,
        cores=16,
    )


def _capped(ctx: int = 32768):
    spec = next(m for m in load() if m.max_context >= 131072)
    return replace(spec, max_context=ctx)


def test_a_context_above_the_models_own_ceiling_is_disclosed():
    """`max_context()` applies `min(model.max_context, ...)`; `solve()` never
    did, so the two entry points disagreed about the same model — one answering
    "32,768 is the most that fits" while the other accepted a request for twice
    that and reported it feasible.
    """
    fit = F.solve(_capped(), "q8", _hw(), 65536, 8)
    assert fit.beyond_published_context
    text = fit.explain()
    assert "above this model's published limit" in text
    assert "32,768" in text


def test_it_is_disclosed_and_not_refused():
    """The decision, recorded as a test: `max_position_embeddings` is not always
    the operational limit — a deployment served with RoPE scaling runs above it
    deliberately — so refusing would make the solver wrong about a real case.

    The memory arithmetic stays correct and stays reported.
    """
    fit = F.solve(_capped(), "q8", _hw(), 65536, 8)
    assert fit.kv_bytes > 0
    assert fit.total_bytes > fit.weight_bytes


@pytest.mark.parametrize("ctx", [1024, 16384, 32768])
def test_a_context_within_the_ceiling_says_nothing(ctx):
    fit = F.solve(_capped(), "q8", _hw(), ctx, 8)
    assert not fit.beyond_published_context
    assert "published limit" not in fit.explain()


def test_the_two_entry_points_now_agree_about_the_ceiling():
    """The disagreement itself, asserted directly: whatever `max_context()`
    returns must be a context `solve()` does not flag."""
    model = _capped()
    largest = F.max_context(model, "q8", _hw(), 8)
    assert not F.solve(model, "q8", _hw(), largest, 8).beyond_published_context
    assert F.solve(model, "q8", _hw(), largest + 1, 8).beyond_published_context
