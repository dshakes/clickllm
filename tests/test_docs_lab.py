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


# --- the silicon diagram's numbers must stay the solver's ----------------------


def silicon_src() -> str:
    """Just silicon()'s body.

    Scoped deliberately: `rows = [` appears in several diagram functions, and a
    whole-file search picked up tpus()' table instead — which parsed to zero
    entries and left the roofline check passing while asserting nothing.
    """
    src = (Path(__file__).resolve().parents[1] / "tools" / "diagrams.py").read_text()
    body = re.search(r"\ndef silicon\(\).*?\n    save\(\"edu-silicon\.svg\"", src, re.S)
    assert body, "silicon() moved or vanished"
    return body.group(0)


def silicon_consts() -> dict[str, float]:
    """The hard-coded figures in tools/diagrams.py's silicon()."""
    src = silicon_src()
    a = re.search(r"WEIGHTS_GB, SMS, LIT = ([\d.]+), (\d+), (\d+)", src)
    b = re.search(r"GOOD_BATCH, OVER_BATCH = (\d+), (\d+)", src)
    c = re.search(r"NEED_GB, HAVE_GB = ([\d.]+), ([\d.]+)", src)
    assert a and b and c, "silicon()'s constants moved or vanished"
    return {
        "weights": float(a.group(1)),
        "sms": int(a.group(2)),
        "lit": int(a.group(3)),
        "good_batch": int(b.group(1)),
        "over_batch": int(b.group(2)),
        "need": float(c.group(1)),
        "have": float(c.group(2)),
    }


def test_the_silicon_diagram_matches_a_real_solve():
    """The diagram states 30.5 GB of weights and "19 people will not fit" as
    fact. Both came from `fit.solve`, and nothing else reads a picture — so
    without this they go stale the first time the catalogue moves, and the
    teaching artefact quietly starts contradicting the tool it teaches."""
    from clickllm.fit import solve
    from clickllm.hardware_catalog import get

    c = silicon_consts()
    hw = get("h100").to_hardware()
    m = catalog.get("qwen3-32b")

    ok = solve(m, "q8", hw, 8192, c["good_batch"])
    assert c["weights"] == pytest.approx(ok.weight_bytes / GB, abs=0.1)
    assert ok.feasible, "the diagram shows this batch fitting"

    over = solve(m, "q8", hw, 8192, c["over_batch"])
    assert not over.feasible, "the diagram shows this batch NOT fitting"
    assert c["need"] == pytest.approx(over.total_bytes / GB, abs=0.1)
    assert c["have"] == pytest.approx(hw.usable_bytes / GB, abs=0.5)


def test_the_batch_ceiling_is_where_the_solver_says_it_is():
    """18 is not a round number someone liked — it is max_concurrency, and the
    19 beside it is the first batch that genuinely does not fit."""
    from clickllm.fit import max_concurrency
    from clickllm.hardware_catalog import get

    c = silicon_consts()
    ceiling = max_concurrency(catalog.get("qwen3-32b"), "q8", get("h100").to_hardware(), 8192)
    assert c["good_batch"] == ceiling
    assert c["over_batch"] == ceiling + 1


def test_the_silicon_diagram_speaks_plainly():
    """It replaced a version that said "arithmetic intensity", "FLOP per byte"
    and "the ridge point" — all true, none of it informative to the reader this
    module is written for. This keeps the jargon out."""
    src = silicon_src()
    for word in ("arithmetic intensity", "FLOP/byte", "ridge point", "roofline"):
        assert word.lower() not in src.lower().split('"""')[2], f"jargon is back: {word}"


# --- motion is opt-out-able, and degrades safely -------------------------------


def test_every_animated_diagram_honours_reduced_motion():
    """Motion a reader cannot stop is an accessibility failure."""
    for name in ("edu-prefill-decode", "edu-batching", "edu-silicon"):
        svg = (Path(__file__).resolve().parents[1] / "docs" / "assets" / f"{name}.svg").read_text()
        assert "prefers-reduced-motion" in svg, name


def test_animated_marks_stay_visible_without_css_animation():
    """The `.pop` class must not carry a bare `opacity:0`. With one, any renderer
    that ignores CSS animation blanks the marks entirely — turning "not animated"
    into "half the diagram is missing"."""
    for name in ("edu-prefill-decode", "edu-batching"):
        svg = (Path(__file__).resolve().parents[1] / "docs" / "assets" / f"{name}.svg").read_text()
        rule = re.search(r"\.pop\{([^}]*)\}", svg).group(1)
        assert "opacity:0" not in rule, f"{name}: {rule}"
        assert "both" in rule, f"{name}: needs backwards fill for the hidden state"


# --- a CSS class silently beats an SVG fill attribute --------------------------


def test_no_diagram_sets_a_fill_attribute_that_its_class_overrides():
    """`<text class="lb" fill="var(--bg)">` renders in the CLASS's colour, not the
    attribute's — a CSS rule beats a presentation attribute. Every such pairing
    was silently discarded, and 11 labels in one diagram alone were rendering in
    the wrong colour: dark-for-contrast text on a mid-tone bar came out light
    grey and unreadable. Nothing errors, so only looking at it finds this.

    The fix is an inline `style`, which does beat the class. This keeps it fixed.
    """
    styled = re.compile(r'class="(?:lb|ax|m|note|sub)"')
    assets = (Path(__file__).resolve().parents[1] / "docs" / "assets").glob("*.svg")
    offenders = []
    for f in assets:
        for tag in re.finditer(r"<text[^>]*>", f.read_text()):
            s = tag.group(0)
            if styled.search(s) and re.search(r'\sfill="', s):
                offenders.append(f"{f.name}: {s[:70]}")
    assert not offenders, "fill attribute will be ignored — use style=:\n" + "\n".join(offenders)


# --- animated diagrams must not be embedded with <img> -------------------------


def test_animated_diagrams_are_inlined_not_referenced_with_img():
    """An SVG loaded via <img> becomes its own image/svg+xml document, and that
    document's animation timeline never starts — measured at 0 while the host
    page's advanced normally. Neither CSS keyframes nor SMIL survive it.

    So any diagram carrying animation classes must be inlined (a `.anim
    [data-svg]` host), never referenced with <img> outside a <noscript>
    fallback. Getting this wrong produces a page that looks completely fine and
    silently shows a still frame, which is how it went unnoticed the first time.
    """
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    animated = {
        f.stem
        for f in assets.glob("*.svg")
        if re.search(r'class="(pop|stream|light|flow|pulse)"', f.read_text())
    }
    assert animated, "no animated diagrams found — has the motion been removed?"

    html = (Path(__file__).resolve().parents[1] / "site" / "docs" / "index.html").read_text()
    # Strip <noscript> blocks: a static <img> in there is the correct fallback.
    live = re.sub(r"<noscript>.*?</noscript>", "", html, flags=re.S)

    offenders = [
        name for name in animated if re.search(rf'<img[^>]+src="[^"]*{re.escape(name)}\.svg', live)
    ]
    assert not offenders, (
        "animated diagrams embedded with <img> will render as a frozen first "
        f"frame: {sorted(offenders)}"
    )
