#!/usr/bin/env python3
"""Generate the learning-module diagrams.

One generator rather than eight hand-drawn files, because consistency across a
teaching set matters more than any individual picture: the same hue must mean the
same thing in every diagram, or the reader has to relearn the legend each time.

## The palette is computed, not chosen

Both modes were validated with the dataviz skill's checker against the real
surfaces, all-pairs (not merely adjacent):

    dark  #0e9bbd #8b5cf6 #c2820b #e0405f   on #0d1320  → all checks pass
    light #0369a1 #7e22ce #b45309 #15803d   on #fcfcfb  → all checks pass

Four slots, not five. A fifth passed on adjacent pairs and failed all-pairs
(emerald↔rose ΔE 4.1 under deuteranopia), and in a teaching diagram every series
is compared against every other, not just its neighbour. So the categorical set
is capped at four and anything beyond that is faceted instead.

Run: `python tools/diagrams.py` — writes into docs/assets/.
"""

from __future__ import annotations

import math
import pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"

# --- the design system, in one place ------------------------------------------

SERIES = ("#0e9bbd", "#8b5cf6", "#c2820b", "#e0405f")
SERIES_LIGHT = ("#0369a1", "#7e22ce", "#b45309", "#15803d")
SURFACE, PANEL = "#0d1320", "#111a2b"
INK, INK_2, MUTED, GRID = "#eef4ff", "#9fb0cc", "#63758f", "#1c2534"
FONT = "ui-sans-serif,-apple-system,'Segoe UI',Inter,system-ui,sans-serif"
MONO = "ui-monospace,'SF Mono',Menlo,monospace"


def head(w: int, h: int, title: str, desc: str) -> list[str]:
    """Open an SVG with the theme block and an accessible title.

    Dark is the designed mode (the site is dark-first); the light block is a
    *selected* set of steps from the same hues, validated separately — never an
    automatic inversion, which would put unvalidated colours on a surface nobody
    checked.
    """
    light = "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_LIGHT))
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}" role="img" '
        f'aria-labelledby="t d" font-family="{FONT}">',
        f"<title id='t'>{title}</title><desc id='d'>{desc}</desc>",
        "<style>",
        ":root{"
        + "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES))
        + f"--bg:{SURFACE};--panel:{PANEL};--ink:{INK};--ink2:{INK_2};"
        f"--muted:{MUTED};--grid:{GRID};}}",
        "@media (prefers-color-scheme:light){:root:not([data-theme=dark]){"
        + light
        + "--bg:#fcfcfb;--panel:#f4f4f2;--ink:#0b0b0b;--ink2:#52514e;"
        "--muted:#7a7975;--grid:#e3e3e0;}}",
        ":root[data-theme=light]{"
        + light
        + "--bg:#fcfcfb;--panel:#f4f4f2;--ink:#0b0b0b;--ink2:#52514e;"
        "--muted:#7a7975;--grid:#e3e3e0;}",
        ".t{fill:var(--ink);font-size:15px;font-weight:650}"
        ".sub{fill:var(--ink2);font-size:11.5px}"
        ".lb{fill:var(--ink2);font-size:11px}"
        ".ax{fill:var(--muted);font-size:10.5px}"
        f".m{{font-family:{MONO};font-size:10.5px;fill:var(--ink2)}}"
        ".g{stroke:var(--grid);stroke-width:1}"
        ".note{fill:var(--muted);font-size:10.5px;font-style:italic}",
        "</style>",
        f'<rect width="{w}" height="{h}" fill="var(--bg)"/>',
        f'<text x="28" y="30" class="t">{title}</text>',
    ]


def legend(x: int, y: int, items: list[tuple[int, str]]) -> str:
    """Legend swatches. Always present for >=2 series — identity is never colour
    alone, and every diagram here also direct-labels."""
    out, dx = [], 0
    for slot, label in items:
        out.append(
            f'<rect x="{x + dx}" y="{y - 7}" width="9" height="9" rx="2.5" '
            f'fill="var(--s{slot})"/>'
            f'<text x="{x + dx + 14}" y="{y + 1}" class="lb">{label}</text>'
        )
        dx += 26 + int(len(label) * 6.6)
    return "".join(out)


def bar(x: float, y: float, w: float, h: float, slot: int, r: float = 4) -> str:
    """A data mark. 4px rounded end, 2px surface gap handled by the caller."""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{max(h, 0.1):.1f}" '
        f'rx="{r}" fill="var(--s{slot})"/>'
    )


def note(parts: list[str], x: int, y: int, text: str, per_line: int = 116) -> None:
    """Wrap a footnote to the canvas width.

    Every overflow bug in the first pass was a long note running off the right
    edge, so wrapping lives here rather than in each caller's head.
    """
    words, line, lines = text.split(), "", []
    for w in words:
        if len(line) + len(w) + 1 > per_line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    for i, ln in enumerate(lines):
        parts.append(f'<text x="{x}" y="{y + i * 16}" class="note">{ln}</text>')


def save(name: str, parts: list[str]) -> None:
    parts.append("</svg>")
    p = OUT / name
    p.write_text("\n".join(parts))
    print(f"  {name}  {p.stat().st_size / 1024:.1f} KB")


# --- 1. where the memory goes --------------------------------------------------


def memory_breakdown() -> None:
    """Stacked bars: the KV cache is the term that surprises people.

    Magnitude across a few categories at four concurrency levels — a stacked bar
    is the right form because the *total* crossing the ceiling is the point, and
    the composition explains why.
    """
    W, H = 860, 452
    p = head(
        W,
        H,
        "Where the memory goes",
        "Stacked memory use for a 32B model at 32k context as concurrency rises "
        "from 1 to 32. Weights stay constant; the KV cache grows linearly and is "
        "what eventually exceeds the 80 GB ceiling.",
    )
    p.append(
        '<text x="28" y="50" class="sub">32B model · 8-bit weights · 32k context '
        "· the KV cache is the term that moves</text>"
    )

    weights, overhead = 32.0, 4.0
    kv_per = 2.0  # GiB per concurrent stream at 32k, GQA
    levels = [1, 4, 16, 32]
    ceiling = 80.0

    x0, y0, plot_h, bw, gap = 90, 92, 232, 74, 46
    scale = plot_h / 96.0

    def ypx(gib: float) -> float:
        return y0 + plot_h - gib * scale

    for gib in range(0, 97, 24):
        y = ypx(gib)
        p.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{W - 250}" y2="{y:.1f}" class="g"/>')
        p.append(f'<text x="{x0 - 10}" y="{y + 3.5:.1f}" class="ax" text-anchor="end">{gib}</text>')
    p.append(f'<text x="{x0 - 30}" y="{y0 - 14}" class="ax" text-anchor="start">GiB</text>')

    yc = ypx(ceiling)
    p.append(
        f'<line x1="{x0}" y1="{yc:.1f}" x2="{W - 250}" y2="{yc:.1f}" '
        f'stroke="var(--s3)" stroke-width="2" stroke-dasharray="7 4"/>'
        f'<text x="{W - 246}" y="{yc + 3.5:.1f}" class="lb" fill="var(--s3)">80 GB card</text>'
    )

    for i, c in enumerate(levels):
        x = x0 + i * (bw + gap)
        kv = kv_per * c
        # 2px surface gap between stacked segments.
        p.append(bar(x, ypx(weights), bw, weights * scale - 2, 0))
        p.append(bar(x, ypx(weights + overhead), bw, overhead * scale - 2, 1))
        p.append(bar(x, ypx(weights + overhead + kv), bw, kv * scale - 2, 2))

        total = weights + overhead + kv
        over = total > ceiling
        p.append(
            f'<text x="{x + bw / 2:.1f}" y="{ypx(total) - 10:.1f}" '
            f'class="lb" text-anchor="middle" '
            f'fill="{"var(--s3)" if over else "var(--ink2)"}">{total:.0f} GiB</text>'
        )
        p.append(
            f'<text x="{x + bw / 2:.1f}" y="{y0 + plot_h + 18:.1f}" class="ax" '
            f'text-anchor="middle">{c} user{"s" if c > 1 else ""}</text>'
        )
        # Direct-label the KV segment: it is the one carrying the lesson.
        if c >= 16:
            p.append(
                f'<text x="{x + bw / 2:.1f}" y="{ypx(weights + overhead + kv / 2) + 4:.1f}" '
                f'class="m" text-anchor="middle" fill="var(--bg)">KV {kv:.0f}</text>'
            )

    p.append(
        legend(x0, y0 + plot_h + 46, [(0, "weights"), (1, "runtime overhead"), (2, "KV cache")])
    )
    note(
        p,
        28,
        H - 40,
        "Weights are a one-time cost you can look up. The KV cache is the one that "
        "decides how many people you can serve — and it is linear in context × "
        "concurrency.",
    )
    save("edu-memory.svg", p)


# --- 2. prefill vs decode ------------------------------------------------------


def prefill_decode() -> None:
    """A timeline, because the lesson is about *when*, not how much."""
    W, H = 860, 386
    p = head(
        W,
        H,
        "Prefill and decode are different machines",
        "A timeline of one request: a short, wide, compute-bound prefill reading "
        "the whole prompt in parallel, followed by a long sequence of narrow "
        "memory-bound decode steps emitting one token each.",
    )
    p.append(
        '<text x="28" y="50" class="sub">One request, two phases with opposite '
        "bottlenecks — which is why they are tuned, and increasingly served, "
        "separately</text>"
    )

    x0, y = 100, 108
    p.append(f'<text x="{x0 - 12}" y="{y + 22}" class="ax" text-anchor="end">time →</text>')

    pw = 128
    p.append(bar(x0, y, pw, 44, 0, r=6))
    p.append(
        f'<text x="{x0 + pw / 2:.0f}" y="{y + 28}" class="m" text-anchor="middle" '
        f'fill="var(--bg)">prefill</text>'
    )

    tx = x0 + pw + 14
    for i in range(22):
        p.append(bar(tx + i * 26, y, 22, 44, 1, r=5))
    p.append(
        f'<text x="{tx + 11 * 26:.0f}" y="{y - 12}" class="lb" text-anchor="middle" '
        f'fill="var(--s1)">decode — one token per step, {22} steps shown</text>'
    )

    box = y + 84
    cols = [
        (
            0,
            "prefill",
            [
                "reads the whole prompt at once",
                "all tokens in parallel → compute-bound",
                "sets time-to-first-token",
                "wants: big chunks, tensor cores busy",
            ],
        ),
        (
            1,
            "decode",
            [
                "emits one token, then the next",
                "re-reads all weights per token → bandwidth-bound",
                "sets the gap between tokens",
                "wants: big batches, fast memory",
            ],
        ),
    ]
    for i, (slot, name, lines) in enumerate(cols):
        bx = 28 + i * 412
        p.append(
            f'<rect x="{bx}" y="{box}" width="392" height="112" rx="10" '
            f'fill="var(--panel)" stroke="var(--grid)"/>'
        )
        p.append(f'<rect x="{bx}" y="{box}" width="4" height="112" rx="2" fill="var(--s{slot})"/>')
        p.append(
            f'<text x="{bx + 18}" y="{box + 24}" class="lb" fill="var(--s{slot})">{name}</text>'
        )
        for j, line in enumerate(lines):
            p.append(f'<text x="{bx + 18}" y="{box + 46 + j * 17}" class="ax">{line}</text>')

    note(
        p,
        28,
        H - 40,
        "A long prompt with a short answer is a prefill problem. A short prompt with "
        "a long answer is a decode problem. Tuning for the wrong one is the "
        "commonest way to buy hardware that does not help.",
    )
    save("edu-prefill-decode.svg", p)


# --- 3. KV growth --------------------------------------------------------------


def kv_growth() -> None:
    """Change over two variables — lines, faceted by attention scheme.

    Log-scaled y because the whole point is that MLA is ~50× smaller, and a
    linear axis would render three of the four series as one flat line.
    """
    W, H = 860, 470
    p = head(
        W,
        H,
        "Why the same model needs wildly different memory",
        "KV cache size per stream against context length, for MHA, GQA and MLA "
        "attention. A logarithmic axis, because MLA is roughly fifty times "
        "smaller than MHA and a linear axis would flatten the others to nothing.",
    )
    p.append(
        '<text x="28" y="50" class="sub">KV bytes per stream · 32B-class model · '
        "log scale, because the spread is two orders of magnitude</text>"
    )

    x0, y0, pw, ph = 92, 92, 520, 220
    ctxs = [4096, 16384, 65536, 262144]
    # Per-token KV bytes: MHA stores every head; GQA shares K/V across groups;
    # MLA stores a compressed latent. Mirrors clickllm's own sizing rules.
    schemes = [(0, "MHA", 640_000), (1, "GQA (8 groups)", 80_000), (2, "MLA (latent)", 13_000)]

    # Floor well below the smallest series so MLA does not sit on the axis.
    lo, hi = 1e6, 2e12

    def xpx(i: int) -> float:
        return x0 + i * (pw / (len(ctxs) - 1))

    def ypx(v: float) -> float:
        return y0 + ph - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * ph

    for dec in range(6, 13):
        v = 10.0**dec
        yy = ypx(v)
        if y0 <= yy <= y0 + ph:
            p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" class="g"/>')
            lab = f"{v / 1e9:.0f} GB" if v >= 1e9 else f"{v / 1e6:.0f} MB"
            p.append(
                f'<text x="{x0 - 10}" y="{yy + 3.5:.1f}" class="ax" text-anchor="end">{lab}</text>'
            )

    for i, c in enumerate(ctxs):
        p.append(
            f'<text x="{xpx(i):.1f}" y="{y0 + ph + 18:.1f}" class="ax" '
            f'text-anchor="middle">{c // 1024}k</text>'
        )
    p.append(
        f'<text x="{x0 + pw / 2:.0f}" y="{y0 + ph + 38}" class="ax" '
        f'text-anchor="middle">context length</text>'
    )

    for slot, name, per_tok in schemes:
        pts = " ".join(f"{xpx(i):.1f},{ypx(per_tok * c / 1000):.1f}" for i, c in enumerate(ctxs))
        p.append(
            f'<polyline points="{pts}" fill="none" stroke="var(--s{slot})" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )
        for i, c in enumerate(ctxs):
            # 2px surface ring so overlapping markers stay separable.
            p.append(
                f'<circle cx="{xpx(i):.1f}" cy="{ypx(per_tok * c / 1000):.1f}" r="4.5" '
                f'fill="var(--s{slot})" stroke="var(--bg)" stroke-width="2"/>'
            )
        ey = ypx(per_tok * ctxs[-1] / 1000)
        p.append(
            f'<text x="{x0 + pw + 14}" y="{ey + 4:.1f}" class="lb" '
            f'fill="var(--s{slot})">{name}</text>'
        )

    p.append(legend(x0, y0 + ph + 76, [(s, n) for s, n, _ in schemes]))
    p.append(
        f'<text x="28" y="{H - 22}" class="note">Using the MHA formula on an MLA '
        f"model overestimates its KV cache by around fifty times — which is how a "
        f"model that fits comfortably gets rejected as too big.</text>"
    )
    save("edu-kv-growth.svg", p)


# --- 4. continuous batching ----------------------------------------------------


def batching() -> None:
    """Two timelines stacked. The lesson is the *gaps*, so draw the gaps."""
    W, H = 860, 448
    p = head(
        W,
        H,
        "Continuous batching: stop waiting for the slowest request",
        "Two timelines. Static batching runs four requests in lockstep and idles "
        "until the longest finishes. Continuous batching admits a new request "
        "into the slot the moment one frees, leaving almost no idle time.",
    )
    p.append(
        '<text x="28" y="50" class="sub">Four slots, same requests, same '
        "hardware — the difference is entirely scheduling</text>"
    )

    lens = [3, 7, 4, 9]
    unit, x0 = 46, 128

    def row(
        y: int, label: str, runs: list[tuple[int, int, int]], idle: list[tuple[int, int]]
    ) -> None:
        p.append(f'<text x="{x0 - 14}" y="{y + 16}" class="ax" text-anchor="end">{label}</text>')
        for a, b in idle:
            p.append(
                f'<rect x="{x0 + a * unit:.0f}" y="{y}" width="{(b - a) * unit:.0f}" '
                f'height="24" rx="4" fill="var(--grid)" opacity="0.55"/>'
            )
        for start, length, slot in runs:
            p.append(bar(x0 + start * unit + 1, y, length * unit - 3, 24, slot, r=4))

    p.append('<text x="28" y="88" class="lb">static batching</text>')
    for i, ln in enumerate(lens):
        y = 100 + i * 30
        row(y, f"slot {i + 1}", [(0, ln, i % 4)], [(ln, 9)])
    p.append(
        f'<text x="{x0 + 9 * unit + 12}" y="{100 + 45}" class="lb" fill="var(--s3)">'
        f"idle until the slowest finishes</text>"
    )

    p.append('<text x="28" y="252" class="lb">continuous batching</text>')
    # Each slot refills as soon as it frees.
    plans = [
        [(0, 3, 0), (3, 4, 1)],
        [(0, 7, 1)],
        [(0, 4, 2), (4, 3, 3)],
        [(0, 9, 3)],
    ]
    for i, runs in enumerate(plans):
        y = 264 + i * 30
        end = max(s + ln for s, ln, _ in runs)
        row(y, f"slot {i + 1}", runs, [(end, 9)] if end < 9 else [])

    for i, line in enumerate(
        [
            "Same tokens, same GPU. Static batching leaves the shaded time unused;",
            "continuous batching sells it — the single biggest throughput win in modern",
            "serving, and on by default in vLLM and SGLang.",
        ]
    ):
        p.append(f'<text x="28" y="{H - 58 + i * 16}" class="note">{line}</text>')
    save("edu-batching.svg", p)


# --- 5. speculative decoding win/loss ------------------------------------------


def spec_decode() -> None:
    """Diverging: the quantity has a meaningful zero and changes sign.

    The only honest way to draw this. A bar chart of "speedup" would hide that
    the technique goes *negative*, which is the entire lesson.
    """
    W, H = 860, 468
    p = head(
        W,
        H,
        "Speculative decoding is a bet on idle compute",
        "Throughput change from speculative decoding against batch size. "
        "Strongly positive at batch 1, decaying as batch grows, and negative "
        "beyond roughly batch 32 where the draft pass competes with real tokens.",
    )
    p.append(
        '<text x="28" y="50" class="sub">It pays when the accelerator has spare '
        "cycles between tokens. At high batch there are none.</text>"
    )

    x0, y0, pw, ph = 92, 96, 600, 230
    batches = [1, 2, 4, 8, 16, 32, 64, 128]
    delta = [0.78, 0.66, 0.48, 0.30, 0.12, -0.02, -0.18, -0.31]

    lo, hi = -0.45, 0.9

    def xpx(i: int) -> float:
        return x0 + 34 + i * ((pw - 68) / (len(batches) - 1))

    def ypx(v: float) -> float:
        return y0 + ph - (v - lo) / (hi - lo) * ph

    zero = ypx(0.0)
    for v in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8):
        yy = ypx(v)
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + pw}" y2="{yy:.1f}" class="g"/>')
        p.append(
            f'<text x="{x0 - 10}" y="{yy + 3.5:.1f}" class="ax" text-anchor="end">{v:+.0%}</text>'
        )
    p.append(
        f'<line x1="{x0}" y1="{zero:.1f}" x2="{x0 + pw}" y2="{zero:.1f}" '
        f'stroke="var(--ink2)" stroke-width="1.5"/>'
    )

    bw = 30
    for i, (b, d) in enumerate(zip(batches, delta, strict=True)):
        x = xpx(i) - bw / 2
        # Diverging: two poles, neutral midpoint. Slot 0 for gain, slot 3 for loss.
        slot = 0 if d >= 0 else 3
        top = ypx(max(d, 0.0))
        p.append(bar(x, top, bw, abs(ypx(d) - zero), slot, r=4))
        p.append(
            f'<text x="{xpx(i):.1f}" y="{(top - 8) if d >= 0 else (ypx(d) + 15):.1f}" '
            f'class="ax" text-anchor="middle" '
            f'fill="var(--s{slot})">{d:+.0%}</text>'
        )
        p.append(
            f'<text x="{xpx(i):.1f}" y="{y0 + ph + 18:.1f}" class="ax" '
            f'text-anchor="middle">{b}</text>'
        )
    p.append(
        f'<text x="{x0 + pw / 2:.0f}" y="{y0 + ph + 38}" class="ax" '
        f'text-anchor="middle">batch size (concurrent requests)</text>'
    )

    xc = xpx(4.6)
    p.append(
        f'<line x1="{xc:.1f}" y1="{y0}" x2="{xc:.1f}" y2="{y0 + ph}" '
        f'stroke="var(--s3)" stroke-width="2" stroke-dasharray="6 4"/>'
        f'<text x="{xc + 8:.1f}" y="{y0 + 14}" class="lb" fill="var(--s3)">'
        f"break-even</text>"
    )

    p.append(legend(x0, y0 + ph + 58, [(0, "throughput gained"), (3, "throughput lost")]))
    note(
        p,
        28,
        H - 44,
        "Illustrative shape, not measured numbers — the crossover moves with the "
        "model, the drafter and the hardware. The lesson is that the quantity has a "
        "sign, which is why clickllm derives it from your concurrency instead of "
        "offering it as a checkbox.",
    )
    save("edu-spec-decode.svg", p)


# --- 6. quantization -----------------------------------------------------------


def quantization() -> None:
    """Magnitude across one dimension. Bars, one hue, direct-labelled."""
    W, H = 860, 392
    p = head(
        W,
        H,
        "Quantization buys bandwidth, not just space",
        "Weight size for a 32B model at four precisions, from 16-bit down to "
        "4-bit. Decode re-reads every weight per token, so a smaller format is "
        "directly faster as well as smaller.",
    )
    p.append(
        '<text x="28" y="50" class="sub">32B model · decode re-reads every weight '
        "for every token, so halving the bytes roughly halves the time</text>"
    )

    rows = [
        ("16-bit (fp16/bf16)", 64.0, "reference quality"),
        ("8-bit (fp8/int8)", 32.0, "no measurable loss, most tasks"),
        ("6-bit", 24.0, "small loss, rarely noticed"),
        ("4-bit (Q4_K_M/AWQ)", 16.0, "hurts hard reasoning; fine to extract"),
    ]
    x0, y0, bwid, rowh = 196, 92, 356, 46
    scale = bwid / 64.0

    for i, (name, gib, qual) in enumerate(rows):
        y = y0 + i * rowh
        p.append(f'<text x="{x0 - 14}" y="{y + 20}" class="lb" text-anchor="end">{name}</text>')
        # Sequential magnitude, one hue — never a rainbow across an ordered scale.
        p.append(bar(x0, y, gib * scale, 26, 0))
        p.append(
            f'<text x="{x0 + gib * scale + 12:.0f}" y="{y + 19}" class="m">{gib:.0f} GiB</text>'
        )
        p.append(f'<text x="{x0 + bwid + 84}" y="{y + 19}" class="ax">{qual}</text>')

    note(
        p,
        28,
        H - 46,
        "clickllm caps its own recommendation at 8-bit: below that the quality "
        "question stops being free, and it is a decision to make deliberately "
        "rather than a default to inherit.",
    )
    save("edu-quantization.svg", p)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing diagrams:")
    memory_breakdown()
    prefill_decode()
    kv_growth()
    batching()
    spec_decode()
    quantization()
    print("done")
