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
import os
import re
import shutil
import subprocess
import sys
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


def test_the_silicon_diagram_glosses_every_technical_term():
    """The diagram has been wrong in both directions.

    First it said "arithmetic intensity" and "the ridge point" unexplained, to
    a reader meeting both for the first time. The correction over-swung and
    deleted the terms *and the numbers with them*, which taught nothing either:
    a claim with no quantities behind it cannot be checked.

    So the rule is not "no jargon" but "no unglossed jargon". Each term may
    appear only if its plain-language gloss appears too — and the gloss is the
    thing that rots first, because a later edit trims the long line and leaves
    the short one. This is what holds the pairing together.

    Read from the rendered SVG rather than the source: a gloss wrapped across
    two f-strings is present for the reader and absent from any one source
    line, so checking the source would fail on formatting rather than on
    substance.
    """
    svg = (Path(__file__).resolve().parents[1] / "docs" / "assets" / "edu-silicon.svg").read_text()
    body = svg.split("</style>")[-1].lower()
    for term, gloss in (
        ("arithmetic intensity", "flops computed per byte read"),
        ("ridge point", "break even"),
        ("flop/byte", "one floating-point operation"),
        ("sms", "streaming multiprocessor"),
        ("kv cache", "keys and values"),
    ):
        if term in body:
            assert gloss in body, f"'{term}' appears with no gloss — expected '{gloss}'"


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

    `font-size` is checked the same way and for the same reason: `.t` sets one,
    so `<text class="t" font-size="32">` renders at 15px. That one is quieter
    than the colour bug — nothing is unreadable, the emphasis simply never
    arrives — which is exactly why it survived a visual review.

    Scoped to the files `head()` generates, identified by the `--s0` custom
    property only it emits. The class names are only load-bearing under that
    stylesheet: gap-map.svg is hand-authored and defines its own `.t` with
    nothing but a font-family, so its fill attribute is honoured and flagging
    it would be a false alarm.
    """
    # (classes that set the property, the attribute they would silently beat)
    traps = (
        (r'class="(?:lb|ax|m|note|sub|t)"', "fill"),
        (r'class="(?:t|sub|lb|ax|m|note)"', "font-size"),
    )
    assets = sorted((Path(__file__).resolve().parents[1] / "docs" / "assets").glob("*.svg"))
    generated = [f for f in assets if "--s0:" in f.read_text()]
    assert generated, "no generated diagrams found — has head() changed?"
    offenders = []
    for f in generated:
        text = f.read_text()
        for tag in re.finditer(r"<text[^>]*>", text):
            s = tag.group(0)
            for styled, attr in traps:
                if re.search(styled, s) and re.search(rf'\s{attr}="', s):
                    offenders.append(f"{f.name}: {attr} on {s[:70]}")
    assert not offenders, "presentation attribute will be ignored — use style=:\n" + "\n".join(
        offenders
    )


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

    # BOTH pages. The first version of this test checked only the docs page, and
    # the landing page went on serving an animated diagram through <img> — the
    # same defect, sitting in the file nobody thought to re-check.
    offenders = []
    for page in (
        Path(__file__).resolve().parents[1] / "site" / "index.html",
        Path(__file__).resolve().parents[1] / "site" / "docs" / "index.html",
    ):
        # Strip <noscript> blocks: a static <img> in there is the correct fallback.
        live = re.sub(r"<noscript>.*?</noscript>", "", page.read_text(), flags=re.S)
        offenders += [
            f"{page.parent.name}/{page.name}: {name}"
            for name in animated
            if re.search(rf'<img[^>]+src="[^"]*{re.escape(name)}\.svg', live)
        ]
    assert not offenders, (
        "animated diagrams embedded with <img> will render as a frozen first "
        f"frame: {sorted(offenders)}"
    )


# --- diagrams must not be stretched out of proportion --------------------------

SITE = Path(__file__).resolve().parents[1] / "site"
PAGES = (SITE / "index.html", SITE / "docs" / "index.html")


def figure_media_rules(css: str) -> list[tuple[str, str]]:
    """(selector, declarations) for rules that size figure images or inline SVG.

    Comments are stripped first: leaving them in makes a failure message that
    quotes an entire paragraph of prose as if it were a selector.
    """
    css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
    out = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        s = " ".join(sel.split())
        if re.search(r"figure\s+(img|svg|\.anim svg)|\.hero-anim\s+(img|svg)", s):
            out.append((s, " ".join(body.split())))
    return out


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.parent.name + "/" + p.name)
def test_no_rule_forces_a_width_on_a_height_capped_diagram(page):
    """This shipped, and it is nastier than it sounds.

    A rule setting `width:100%` alongside a `max-height` pins BOTH axes of an
    <img>, and the default `object-fit:fill` then STRETCHES rather than scales.
    A 900x690 diagram in a 1112px column became a 1112x526 box — aspect 2.11
    against a source aspect of 1.30, a 38% vertical crush.

    "Too big" is obvious. "Subtly squashed" just reads as the graphics looking
    wrong, which is exactly how it was reported.
    """
    rules = figure_media_rules(page.read_text())
    assert rules, f"{page.name}: no figure media rules found — has the markup moved?"

    capped = [r for r in rules if "max-height" in r[1]]
    assert capped, f"{page.name}: nothing caps diagram height; one will tower over the fold"

    forcing = [s for s, b in rules if re.search(r"(?<!max-)\bwidth\s*:\s*100%", b)]
    assert not forcing, (
        f"{page.name}: these force a width while another rule caps the height, "
        f"which stretches the image instead of scaling it: {forcing}"
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.parent.name + "/" + p.name)
def test_height_capped_diagrams_declare_object_fit_contain(page):
    """The belt to the braces above: with `contain`, deformation is impossible
    no matter what a future rule pins, so the picture letterboxes instead."""
    capped = [(s, b) for s, b in figure_media_rules(page.read_text()) if "max-height" in b]
    for sel, body in capped:
        assert "object-fit:contain" in body.replace(" ", ""), (
            f"{page.name}: '{sel}' caps height without object-fit:contain — "
            f"any forced width will squash the diagram"
        )


def test_every_generated_svg_is_intrinsically_proportional():
    """width/height attributes must agree with the viewBox, or the picture is
    born distorted before any CSS touches it."""
    bad = []
    for f in sorted((Path(__file__).resolve().parents[1] / "docs" / "assets").glob("*.svg")):
        head = f.read_text(errors="ignore")[:600]
        vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head)
        wh = re.search(r'width="(\d+)"\s+height="(\d+)"', head)
        if not (vb and wh):
            continue  # hand-authored files need not declare both
        vw, vh = float(vb.group(1)), float(vb.group(2))
        w, h = float(wh.group(1)), float(wh.group(2))
        if abs((w / h) - (vw / vh)) > 0.01:
            bad.append(f"{f.name}: attrs {w}x{h} vs viewBox {vw}x{vh}")
    assert not bad, "intrinsically distorted: " + "; ".join(bad)


# --- published claims ------------------------------------------------------
# The test count appears on three surfaces that drifted apart unnoticed: the
# site said 478, the README badge said 668, and the suite was actually 675.
# Every one of them was green the whole time, because nothing compared them.


def _published_test_counts() -> dict[str, int]:
    """The test count as each published surface states it."""
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    site = (root / "site" / "index.html").read_text()
    found = {}
    for name, text, pattern in (
        ("readme badge", readme, r"badge/tests-(\d+)-"),
        ("readme prose", readme, r"\*\*(\d[\d,]*) tests\.\*\*"),
        ("site stat", site, r'<div class="n grad">(\d+)</div><div class="l">tests'),
    ):
        m = re.search(pattern, text)
        assert m, f"{name}: no test count found — did the markup change?"
        found[name] = int(m.group(1).replace(",", ""))
    return found


def test_every_published_test_count_agrees():
    """Three surfaces, one number. They have disagreed before."""
    counts = _published_test_counts()
    assert len(set(counts.values())) == 1, "published test counts disagree: " + ", ".join(
        f"{k}={v}" for k, v in counts.items()
    )


def test_published_test_count_is_not_stale(request):
    """The published number must still describe the suite that exists.

    Checks the Python half against what pytest actually collected this run, so
    the figure cannot rot quietly. The Rust half is read from `cargo test` only
    when cargo is available; without it the check reports what it could not
    verify rather than passing silently.
    """
    # Only meaningful on a full run. Under `-k`, `-m`, or a single-file
    # invocation `testscollected` counts the selection, not the suite, and this
    # would fail for a reason that has nothing to do with a stale number.
    opt = request.config.option
    filtered = bool(getattr(opt, "keyword", "") or getattr(opt, "markexpr", ""))
    root = Path(__file__).resolve().parents[1]
    args = [Path(a).resolve() for a in request.config.args]
    partial = any(a != root and a != root / "tests" for a in args)
    if filtered or partial:
        pytest.skip("partial run — the collected count is a selection, not the suite")

    published = next(iter(_published_test_counts().values()))
    python_collected = request.session.testscollected
    assert python_collected > 0, "collected nothing; the check would be vacuous"

    cargo = shutil.which("cargo")
    if cargo is None:
        pytest.skip("cargo unavailable — cannot verify the Rust half of the count")
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [cargo, "test", "--no-run", "--message-format=short"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.skip(f"cargo test --no-run failed; cannot verify: {proc.stderr[-200:]}")
    rust = _rust_test_count(root, cargo)
    if rust is None:
        pytest.skip("could not read a Rust test count")
    assert published == rust + python_collected, (
        f"published {published} but the suite is {rust} Rust + "
        f"{python_collected} Python = {rust + python_collected}"
    )


def _rust_test_count(root: Path, cargo: str) -> int | None:
    """Rust tests, counted by asking each test binary to list them."""
    proc = subprocess.run(
        [cargo, "test", "--", "--list"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        return None
    # Each binary prints "N tests, M benchmarks"; sum the N across binaries.
    totals = [int(n) for n in re.findall(r"^(\d+) tests?, \d+ benchmark", proc.stdout, re.M)]
    return sum(totals) if totals else None


# --- the docs must teach commands that exist -----------------------------------


def _cli_commands() -> set[str]:
    """Every subcommand the CLI actually registers."""
    from clickllm import cli

    parser = cli.build_parser() if hasattr(cli, "build_parser") else None
    if parser is None:  # fall back to asking the CLI itself
        out = subprocess.run(
            [sys.executable, "-m", "clickllm.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONPATH": "src"},
            timeout=120,
        ).stdout
        m = re.search(r"\{([a-z,-]+)\}", out)
        assert m, f"could not read the command list:\n{out[:300]}"
        return set(m.group(1).split(","))
    for action in parser._subparsers._group_actions:  # noqa: SLF001
        if hasattr(action, "choices"):
            return set(action.choices)
    raise AssertionError("no subparsers found")


@pytest.mark.parametrize(
    "page", [Path("site/docs/index.html"), Path("site/index.html"), Path("README.md")]
)
def test_every_command_the_docs_tell_you_to_run_exists(page):
    """A `$ clickllm X` in a code block is an instruction, not prose.

    The docs shipped eight that were never implemented — `switch`, `cutover`,
    `deploy`, `tune`, `pack`, `pull`, `push`, and `explain` (which is a flag on
    `fit`, not a subcommand). The kiosk walkthrough *opened* with `clickllm
    switch`, so the first thing a reader was told to type did not exist.

    Only shell-prompt occurrences count. Prose like "clickllm picks the highest
    precision" is English, and matching it would make this unfixable noise.
    """
    root = Path(__file__).resolve().parents[1]
    text = (root / page).read_text()
    real = _cli_commands()

    # `$ clickllm <cmd>` — the shell-prompt form, in HTML or fenced markdown.
    told = {m.group(1) for m in re.finditer(r"\$\s*(?:uvx\s+)?clickllm\s+([a-z][a-z0-9-]*)", text)}
    unknown = sorted(told - real)
    assert not unknown, (
        f"{page} tells the reader to run commands that do not exist: {unknown}. "
        f"registered: {sorted(real)}"
    )


# --- ...and install commands that name something obtainable --------------------

#: Names of OUR OWN distributions that have actually been published, anywhere a
#: reader could install them from — PyPI, a Homebrew tap, a registry.
#:
#: One entry, and it is the whole list. `clickllm-cli` is on PyPI;
#: `https://pypi.org/pypi/clickllm/json` is still a **404** and so is
#: `clickllm-core`, and `dshakes/homebrew-tap` still carries only `compass.rb`,
#: `distil.rb` and `firstpass-proxy.rb` — no clickllm formula. Publishing
#: something means adding it here, which is the point: the constant is the claim,
#: and the test is what makes stating it falsely cost something.
PUBLISHED_PACKAGE_NAMES: frozenset[str] = frozenset({"clickllm-cli"})
#: Published 2026-07-30 and verified by installing it: `uvx --from clickllm-cli
#: clickllm fit` ran. `clickllm` itself is NOT in here and cannot be — PyPI
#: refused it as too similar to the existing `click-llm`. The distribution is
#: `clickllm-cli`; the console script is `clickllm`.

#: Surfaces that tell a reader how to install. install.sh is included because it
#: is served live from the docs site and is executed verbatim by a `curl | sh`.
INSTALL_SURFACES = (
    Path("README.md"),
    Path("site/index.html"),
    Path("site/docs/index.html"),
    Path("install.sh"),
    # The one the curl URL actually serves. It was a stale copy of install.sh
    # and kept every unpublished command after install.sh was fixed — so the
    # served installer stayed broken while the repo's looked correct, and this
    # list not covering it is why nothing said so.
    Path("site/install.sh"),
)

#: `pip install X`, `pipx install X`, `uvx X`, `uv tool install X`, `brew install X`,
#: `docker run … IMAGE`. The verb is what makes the line an instruction.
_INSTALL_VERB = re.compile(
    r"\b(?:pip3?\s+install|python3?\s+-m\s+pip\s+install|pipx\s+install"
    r"|uv\s+tool\s+install|uvx|brew\s+install|docker\s+run)\b"
)

#: Anything that looks like one of our distribution or image names, including the
#: tap-qualified (`dshakes/tap/clickllm`) and registry-qualified
#: (`ghcr.io/dshakes/clickllm:latest`) forms.
_OUR_NAME = re.compile(r"[\w./-]*clickllm[\w.-]*")


def _install_targets(line: str) -> set[str]:
    """The names an install line actually asks a package manager to fetch.

    Only the token in *package position* counts. Everything else on the line is
    annotation, and this distinction became load-bearing the moment the two names
    diverged: the distribution is `clickllm-cli`, the command it installs is
    `clickllm`, so `uvx --from clickllm-cli clickllm fit` and
    `uv tool install clickllm-cli  # puts clickllm on your PATH` are both correct
    while containing the bare name. Scanning the whole line called them broken and
    made the correct instruction unwritable.

    Package position is the `--from` argument where there is one, otherwise the
    first non-flag token after the verb. `docker run` is the exception — its image
    can sit behind any number of flags — so every token is considered there.
    """
    frm = re.search(r"--from[=\s]+(\S+)", line)
    if frm:
        return set(_OUR_NAME.findall(frm.group(1)))
    verb = _INSTALL_VERB.search(line)
    if not verb:
        return set()
    tokens = [t for t in line[verb.end() :].split("#")[0].split() if not t.startswith("-")]
    if verb.group(0).startswith("docker"):
        return {n for t in tokens for n in _OUR_NAME.findall(t)}
    return set(_OUR_NAME.findall(tokens[0])) if tokens else set()


def _instruction_text(path: Path) -> str:
    """The parts of a page that are instructions rather than prose.

    Markdown: fenced blocks only. HTML: everything except inline `<code>` spans.
    Shell: everything except `#` comments. All three follow the rule the sibling
    test above settled on — "`pip install clickllm` fails" is a *sentence about* a
    broken command, and a check that cannot tell it from the command itself makes
    the failure unfixable, because the fix is to write the sentence. The shell
    carve-out is the one that was missing: install.sh's header comment explains
    why the distribution is `clickllm-cli`, and that explanation has to name the
    bare form it is warning you off.
    """
    text = path.read_text()
    if path.suffix == ".md":
        return "\n".join(re.findall(r"^```[a-z]*\n(.*?)^```", text, re.S | re.M))
    if path.suffix == ".sh":
        return re.sub(r"#.*$", " ", text, flags=re.M)
    return re.sub(r"<code>.*?</code>", " ", text, flags=re.S)


@pytest.mark.parametrize("page", INSTALL_SURFACES, ids=str)
def test_every_install_command_the_docs_publish_names_something_obtainable(page):
    """Every install instruction we ship was unrunnable, on every surface.

    The README, the landing page's install-method tabs (uvx / Homebrew / pip /
    git) and `install.sh` all told the reader to install a package name that has
    never existed: `https://pypi.org/pypi/clickllm/json` returns **404**, and so
    does `clickllm-core`. `brew install dshakes/tap/clickllm` resolves to no
    formula. `ghcr.io/dshakes/clickllm` is not a published image. A reader's
    first contact with this project was a command that failed, and nothing in a
    green suite objected — the sibling test above checks that `clickllm fit` is a
    real *subcommand*, never that `clickllm` is a real *package*.

    A name is only allowed if we have actually published it
    (`PUBLISHED_PACKAGE_NAMES` — `clickllm-cli`, and nothing else; the bare
    `clickllm` is not on PyPI and cannot be, because PyPI refused it as too close
    to `click-llm`). A `git+…` source is always allowed, because it resolves
    against this repository and cannot go stale while the repository exists.

    Deliberately offline: a test that reaches PyPI to decide would be green in CI
    and red on a plane, and the fact it needs — what we have published — is
    something we know without asking anyone.

    Third-party names (`pip install vllm`, `pip install mlx-lm`) are out of
    scope. Whether *those* are on PyPI is not ours to assert or to fix.
    """
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for line in _instruction_text(root / page).splitlines():
        if not _INSTALL_VERB.search(line):
            continue
        ours = _install_targets(line)
        if not ours or "git+" in line:
            continue
        unpublished = sorted(n for n in ours if n not in PUBLISHED_PACKAGE_NAMES)
        if unpublished:
            offenders.append(f"{unpublished} in: {line.strip()[:100]}")
    assert not offenders, (
        f"{page} tells the reader to install something we have never published. "
        f"Use the git form (`--from git+https://github.com/dshakes/clickllm`), or add "
        f"the name to PUBLISHED_PACKAGE_NAMES once it is genuinely on PyPI:\n  "
        + "\n  ".join(offenders)
    )


def test_the_served_installer_is_the_repo_installer():
    """`site/install.sh` is what `curl … | sh` fetches; `install.sh` is what gets
    edited. They drifted, and the served copy kept `pipx install clickllm`,
    `brew install dshakes/tap/clickllm` and `pip install --user clickllm` after
    the real one was fixed — so the installer people actually run stayed broken
    while the repo's looked correct.

    Identical, or the copy is stale by definition.
    """
    root = Path(__file__).resolve().parents[1]
    a = (root / "install.sh").read_text()
    b = (root / "site" / "install.sh").read_text()
    assert a == b, (
        "site/install.sh has drifted from install.sh — the served installer is "
        "not the one that gets reviewed. Copy install.sh over it."
    )
