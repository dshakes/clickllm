"""The docs' interactive calculator must agree with the solver.

The learning module now lets you drag context and concurrency and watch memory
move. That is only worth having if the numbers are the product's numbers — a
teaching tool that contradicts the tool teaches the wrong lesson twice, and it
would drift silently because nothing else reads the page.

So this parses the model specs and the arithmetic out of the page and checks
both against `clickllm.catalog` and `clickllm.fit`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from clickllm import catalog

PAGE = Path(__file__).resolve().parents[1] / "site" / "docs" / "index.html"
GB = 1024**3


def page() -> str:
    return PAGE.read_text()


def lab_models() -> list[dict]:
    """The MODELS array the widget is built from."""
    block = re.search(r"const MODELS = \[(.*?)\n    \];", page(), re.S)
    assert block, "the memory lab's MODELS array moved or vanished"
    rows = []
    for line in block.group(1).strip().splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        # JS object literal -> JSON: quote the bare keys, swap quote style.
        rows.append(json.loads(re.sub(r"(\w+):", r'"\1":', line).replace("'", '"')))
    return rows


def lab_hardware() -> list[dict]:
    block = re.search(r"const HW = \[(.*?)\n    \];", page(), re.S)
    assert block, "the memory lab's HW array moved or vanished"
    return [
        json.loads(re.sub(r"(\w+):", r'"\1":', ln.strip().rstrip(",")).replace("'", '"'))
        for ln in block.group(1).strip().splitlines()
        if ln.strip().startswith("{")
    ]


def test_the_lab_carries_at_least_one_of_each_shape():
    """Dense, MoE and MLA behave differently, and the whole point of the widget
    is that you can see the difference."""
    kinds = {(m.get("moe", False), m["kv"]) for m in lab_models()}
    assert (False, "gqa") in kinds, "no plain dense model to compare against"
    assert (True, "gqa") in kinds, "no MoE model — the total-vs-active lesson needs one"
    assert (True, "mla") in kinds, "no MLA model — the ~50x lesson needs one"


@pytest.mark.parametrize("row", lab_models(), ids=lambda r: r["id"])
def test_every_lab_model_matches_the_catalogue(row):
    """The widget hard-codes specs so it can run offline in a browser. This is
    the check that stops them going stale."""
    m = catalog.get(row["id"])
    assert row["p"] == pytest.approx(m.params_b), "parameter count"
    assert row["layers"] == m.layers
    assert row["kvh"] == m.kv_heads
    assert row["hd"] == m.head_dim
    assert row["kv"] == m.kv_scheme
    if m.kv_scheme == "mla":
        assert row.get("rank") == m.kv_lora_rank, "MLA without kv_lora_rank is off by ~50x"
    if m.is_moe:
        assert row.get("moe") is True, f"{m.id} is MoE and the widget does not say so"
        assert row["a"] == pytest.approx(m.active_b)


@pytest.mark.parametrize("row", lab_models(), ids=lambda r: r["id"])
def test_the_lab_kv_formula_matches_the_solver(row):
    """Per-token KV, computed the way the page computes it."""
    per_tok = (
        (row["rank"] if row["kv"] == "mla" else 2 * row["kvh"] * row["hd"]) * row["layers"] * 2
    )
    assert per_tok == catalog.get(row["id"]).kv_bytes_per_token()


@pytest.mark.parametrize("quant,bits", [("fp16", 16), ("q8", 8), ("q4", 4.5)])
def test_the_lab_quant_bits_match_the_catalogue(quant, bits):
    """q4 is 4.5 bits, not 4 — the widget must not round it into a nicer number."""
    assert catalog.QUANT_BITS[quant] == bits
    block = re.search(r"const GB = .*?BITS = \{(.*?)\}", page(), re.S)
    assert block and f"{quant}: {bits}" in block.group(1)


def test_the_lab_totals_match_a_real_solve():
    """End to end, against the actual solver rather than a re-derivation."""
    from clickllm.fit import solve
    from clickllm.hardware import Hardware

    hw_row = next(h for h in lab_hardware() if "M4 Max" in h["name"])
    hw = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=hw_row["usable"] * GB,
        bandwidth_gbps=546.0,
        cores=16,
    )
    for row in lab_models():
        m = catalog.get(row["id"])
        if "q8" not in m.quants:
            continue
        f = solve(m, "q8", hw, 32_768, 1)
        weights = row["p"] * 1e9 * 8 / 8
        per_tok = (
            (row["rank"] if row["kv"] == "mla" else 2 * row["kvh"] * row["hd"]) * row["layers"] * 2
        )
        kv = per_tok * 32_768
        over = weights * 0.08 + 1.5 * GB
        assert weights == pytest.approx(f.weight_bytes, rel=1e-6), row["id"]
        assert kv == pytest.approx(f.kv_bytes, rel=1e-6), row["id"]
        assert over == pytest.approx(f.overhead_bytes, rel=1e-6), row["id"]


def test_the_lab_is_reachable_and_labelled():
    """It is a teaching device; if it is not announced it will not be used."""
    html = page()
    assert 'id="memlab"' in html
    assert "Try it" in html
    # Accessible: the bar is an image with a label, not a decorative div.
    assert 'role="img"' in html and 'aria-labelledby="m-verdict"' in html
