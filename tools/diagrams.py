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

ROOT = pathlib.Path(__file__).resolve().parents[1]
# Both trees, because they are served by different things: docs/assets/ is what
# the repo markdown links to, site/assets/ is what the published page fetches.
# Writing to only one is how a diagram exists and still 404s.
OUTS = (ROOT / "docs" / "assets", ROOT / "site" / "assets")

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
        ".note{fill:var(--muted);font-size:10.5px;font-style:italic}"
        # --- motion -------------------------------------------------------
        # Animation is used only where the thing being taught IS a process over
        # time — a token stream, a slot refilling, a saturated pipe. On a static
        # magnitude it would be decoration competing with the data, so most of
        # these diagrams stay still on purpose.
        #
        # CSS keyframes rather than SMIL, for one reason that matters: a media
        # query can switch them off. Motion a reader cannot stop is an
        # accessibility failure, and `prefers-reduced-motion` is honoured here
        # in the shared header so no individual diagram can forget it.
        "@keyframes clkpulse{0%,100%{opacity:.45}50%{opacity:1}}"
        "@keyframes clkflow{to{stroke-dashoffset:-28}}"
        "@keyframes clkpop{0%{opacity:0}3%{opacity:1}92%{opacity:1}100%{opacity:0}}"
        ".pulse{animation:clkpulse 2.4s ease-in-out infinite}"
        ".flow{animation:clkflow 1.1s linear infinite}"
        # No `opacity:0` on the class itself. The hidden-until-my-turn state
        # comes from the backwards fill of `both`, so a renderer that ignores
        # CSS animation entirely leaves these marks VISIBLE rather than blanking
        # them — the difference between "not animated" and "half the diagram is
        # missing" in whatever strips CSS from an <img> next.
        ".pop{animation:clkpop 6s linear infinite both}"
        # Weights streaming from memory onto the die, and the die lighting as
        # they land. SVG needs transform-box:fill-box or the transform origin is
        # the viewport, not the element.
        "@keyframes clkstream{from{transform:translateX(0)}to{transform:translateX(46px)}}"
        "@keyframes clklight{0%,12%{opacity:.10}26%,68%{opacity:1}84%,100%{opacity:.10}}"
        ".stream{transform-box:fill-box;animation:clkstream 1.15s linear infinite}"
        ".light{animation:clklight 3.45s ease-in-out infinite}"
        "@media (prefers-reduced-motion:reduce){"
        ".pulse,.flow,.pop,.stream,.light{animation:none;opacity:1;transform:none}}",
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


def chip(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    pins_per_side: int = 4,
    accent: int | None = None,
    net: bool = True,
    proven: bool = False,
    anim: bool = False,
) -> str:
    """A silicon package with a neural graph on the die.

    The motif for this product in one object: a chip that visibly has a *model*
    on it. Pins on all four sides and a pin-1 notch are what make a rounded
    rectangle read as hardware; the node graph on the die is what makes it read
    as an LLM rather than any old processor.

    ``proven=True`` adds the check badge, which is not decoration — it is the
    one claim this product exists to make about a piece of silicon: that what
    runs on it was measured, not assumed.
    """
    body = f"var(--s{accent})" if accent is not None else "var(--grid)"
    out: list[str] = []

    # Contacts on every side, drawn first so the body covers their inner ends.
    for i in range(1, pins_per_side + 1):
        fx = x + i * w / (pins_per_side + 1) - 2.5
        fy = y + i * h / (pins_per_side + 1) - 2.5
        for px, py, pw, ph in (
            (fx, y - 8, 5, 11),  # top
            (fx, y + h - 3, 5, 11),  # bottom
            (x - 8, fy, 11, 5),  # left
            (x + w - 3, fy, 11, 5),  # right
        ):
            out.append(
                f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw}" height="{ph}" rx="2" '
                f'fill="{body}" opacity="0.55"/>'
            )

    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(w, h) * 0.17:.0f}" '
        f'fill="{body}" opacity="0.30"/>'
    )
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{min(w, h) * 0.17:.0f}" '
        f'fill="none" stroke="{body}" stroke-width="1.5" opacity="0.9"/>'
    )

    # The die: an inset panel, where the model actually sits.
    inset = min(w, h) * 0.19
    dx, dy = x + inset, y + inset
    dw, dh = w - 2 * inset, h - 2 * inset
    out.append(
        f'<rect x="{dx:.1f}" y="{dy:.1f}" width="{dw:.1f}" height="{dh:.1f}" '
        f'rx="{min(dw, dh) * 0.16:.0f}" fill="var(--bg)" opacity="0.55"/>'
    )

    if net:
        # Centre node plus a ring of six: enough to read as a network, few
        # enough to stay legible when this is 40px wide in a heading.
        cx, cy = dx + dw / 2, dy + dh / 2
        r = min(dw, dh) * 0.30
        pts = [
            (cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a)))
            for a in range(-90, 270, 60)
        ]
        nr = max(1.8, min(dw, dh) * 0.055)
        for i, (px, py) in enumerate(pts):
            qx, qy = pts[(i + 1) % len(pts)]
            out.append(
                f'<path d="M {cx:.1f} {cy:.1f} L {px:.1f} {py:.1f}" '
                f'stroke="{body}" stroke-width="1.1" opacity="0.75"/>'
                f'<path d="M {px:.1f} {py:.1f} L {qx:.1f} {qy:.1f}" '
                f'stroke="{body}" stroke-width="1.1" opacity="0.45"/>'
            )
        # Not a decorative graph in one colour. The ring runs through this
        # product's own two poles — warm for the incumbent you are paying for,
        # cool for the candidate you are proving — so the die shows a migration
        # *happening on it* rather than a generic neural-network sticker. The
        # centre node is the gate the traffic crosses.
        for i, (px, py) in enumerate(pts):
            moved = i >= len(pts) // 2
            paint = "var(--s0)" if moved else "var(--s2)"
            a = f' class="light" style="animation-delay:{0.9 + i * 0.12:.2f}s"' if anim else ""
            out.append(f'<circle{a} cx="{px:.1f}" cy="{py:.1f}" r="{nr:.1f}" fill="{paint}"/>')
        ac = ' class="light" style="animation-delay:1.6s"' if anim else ""
        out.append(f'<circle{ac} cx="{cx:.1f}" cy="{cy:.1f}" r="{nr * 1.5:.1f}" fill="var(--s0)"/>')

    if proven:
        bx, by, br = x + w - 4, y + h - 4, min(w, h) * 0.19
        out.append(
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{br:.1f}" fill="var(--bg)"/>'
            f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{br - 2:.1f}" fill="#2ea567"/>'
            f'<path d="M {bx - br * 0.42:.1f} {by:.1f} l {br * 0.28:.1f} {br * 0.30:.1f} '
            f'l {br * 0.58:.1f} {-br * 0.62:.1f}" fill="none" stroke="#fff" '
            f'stroke-width="{max(1.6, br * 0.17):.1f}" stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
    return "".join(out)


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
    body = "\n".join(parts)
    for out in OUTS:
        out.mkdir(parents=True, exist_ok=True)
        (out / name).write_text(body)
    print(f"  {name}  {len(body) / 1024:.1f} KB  →  {len(OUTS)} trees")


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
    # Animated, because "one token per step" is a claim about *time*. A row of
    # 22 identical squares states it; watching them arrive one at a time after a
    # single wide prefill is the same fact learned instead of read.
    for i in range(22):
        p.append(
            f'<rect class="pop" style="animation-delay:{0.5 + i * 0.15:.2f}s" '
            f'x="{tx + i * 26}" y="{y}" width="22" height="44" rx="5" '
            f'fill="var(--s1)"/>'
        )
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
            # Both timelines share one clock: a run appears at its own start
            # time. That is the entire lesson made visible — in the static row
            # nothing ever appears after t=0, while in the continuous row a new
            # request drops into a slot the instant it frees.
            p.append(
                f'<rect class="pop" style="animation-delay:{start * 0.45:.2f}s" '
                f'x="{x0 + start * unit + 1:.0f}" y="{y}" '
                f'width="{length * unit - 3:.0f}" height="24" rx="4" '
                f'fill="var(--s{slot})"/>'
            )

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


# --- 7. inference in a box -----------------------------------------------------


def in_a_box() -> None:
    """The artifact, and the three things that can happen when it lands.

    The naive mental model is "it is a Docker container". It cannot be: a Linux
    container has no path to Apple's GPU, so one image cannot span a CUDA server
    and a laptop. The box is therefore an *artifact plus a binding decision made
    on arrival* — which is exactly what makes it worth drawing, because the
    interesting content is the three outcomes, not the packaging.

    Mirrors `clickllm_core::pack::Arrival` so the picture cannot drift from the
    code: AsPacked / Resolved / Unsupported, with the re-solve reporting what it
    gave up.
    """
    W, H = 900, 580
    p = head(
        W,
        H,
        "Inference in a box",
        "One artifact carrying a model reference, a tuned plan, a provenance "
        "record and an optional benchmark. It lands on three different machines "
        "and produces three different outcomes: run as packed, re-solved with "
        "the changes reported, or refused.",
    )
    p.append(
        '<text x="28" y="50" class="sub">One artifact · three machines · three '
        "honest outcomes. Not a container — a container cannot reach Apple's "
        "GPU.</text>"
    )

    # --- the artifact ---------------------------------------------------------
    bx, by, bw, bh = 300, 78, 300, 112
    p.append(
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="12" '
        f'fill="var(--panel)" stroke="var(--s1)" stroke-width="2"/>'
    )
    p.append(
        f'<text x="{bx + bw / 2:.0f}" y="{by + 24}" class="lb" '
        f'text-anchor="middle" fill="var(--s1)">the box · OCI artifact</text>'
    )
    for i, line in enumerate(
        [
            "model reference + digest",
            "the plan it was tuned with",
            "the host class it was tuned on",
            "benchmark, or an explicit none",
        ]
    ):
        p.append(f'<text x="{bx + 20}" y="{by + 46 + i * 16}" class="ax">· {line}</text>')

    # --- three hosts ----------------------------------------------------------
    hosts = [
        (
            0,
            "same host class",
            "H100 80GB",
            "AS PACKED",
            ["plan used unchanged", "benchmark still valid"],
        ),
        (
            1,
            "different silicon",
            "M4 Max 128GB",
            "RE-SOLVED",
            [
                "CUDA engines cannot run here",
                "→ llama.cpp on Metal",
                "benchmark invalidated",
            ],
        ),
        (
            2,
            "smaller machine",
            "RTX 4090 24GB",
            "RE-SOLVED, DEGRADED",
            ["q8 → q4", "context 32k → 8k", "and it says so"],
        ),
    ]
    cw, gap = 262, 20
    x0 = (W - (len(hosts) * cw + (len(hosts) - 1) * gap)) / 2
    top = 258

    for i, (slot, why, host, verdict, changes) in enumerate(hosts):
        x = x0 + i * (cw + gap)
        mid = x + cw / 2
        # Connector from the artifact down to this host.
        p.append(
            f'<path d="M {bx + bw / 2:.0f} {by + bh} V {top - 46:.0f} '
            f'H {mid:.0f} V {top:.0f}" fill="none" stroke="var(--grid)" '
            f'stroke-width="1.5"/>'
        )
        p.append(
            f'<text x="{mid:.0f}" y="{top - 12:.0f}" class="ax" text-anchor="middle">{why}</text>'
        )
        p.append(
            f'<rect x="{x:.0f}" y="{top}" width="{cw}" height="132" rx="10" '
            f'fill="var(--panel)" stroke="var(--grid)"/>'
        )
        p.append(
            f'<rect x="{x:.0f}" y="{top}" width="{cw}" height="4" rx="2" fill="var(--s{slot})"/>'
        )
        p.append(f'<text x="{x + 18:.0f}" y="{top + 30}" class="lb">{host}</text>')
        p.append(
            f'<text x="{x + 18:.0f}" y="{top + 52}" class="ax" '
            f'fill="var(--s{slot})">{verdict}</text>'
        )
        for j, c in enumerate(changes):
            p.append(f'<text x="{x + 18:.0f}" y="{top + 78 + j * 18}" class="ax">{c}</text>')

    # --- the refusal ----------------------------------------------------------
    ry = top + 152
    p.append(
        f'<rect x="{x0:.0f}" y="{ry}" width="{len(hosts) * cw + (len(hosts) - 1) * gap}" '
        f'height="52" rx="10" fill="var(--panel)" stroke="var(--s3)"/>'
    )
    p.append(f'<rect x="{x0:.0f}" y="{ry}" width="4" height="52" rx="2" fill="var(--s3)"/>')
    p.append(
        f'<text x="{x0 + 20:.0f}" y="{ry + 22}" class="lb" fill="var(--s3)">'
        f"UNSUPPORTED — below 2k context, the honest answer is no</text>"
    )
    p.append(
        f'<text x="{x0 + 20:.0f}" y="{ry + 40}" class="ax">A box that quietly '
        f"served a quarter of the context it was built for would be worse than "
        f"one that refuses: nobody finds out until a long request truncates.</text>"
    )

    p.append(legend(28, ry + 84, [(0, "as packed"), (1, "re-solved"), (3, "refused")]))
    note(
        p,
        28,
        ry + 112,
        "The re-solve is the product. Anyone can ship an artifact; the value is "
        "that it re-derives its own plan on arrival and reports every single "
        "thing it had to give up.",
    )
    save("in-a-box.svg", p)


# --- 8. the degradation ladder -------------------------------------------------


def degradation() -> None:
    """What "it fits" costs, step by step.

    A magnitude comparison across an ordered sequence, so: one hue, descending.
    The floor is drawn as a hard line because it is one — `MIN_DEGRADED_CONTEXT`.
    """
    W, H = 900, 366
    p = head(
        W,
        H,
        "When the box lands somewhere smaller",
        "The degradation ladder: quantisation is reduced first, then context is "
        "halved repeatedly, until either the workload fits or it drops below the "
        "2k floor and is refused.",
    )
    p.append(
        '<text x="28" y="50" class="sub">Give something up rather than nothing — '
        "and name what was given up</text>"
    )

    steps = [
        ("as packed", "q8 · 32k context", 100),
        ("re-quantise", "q4 · 32k context", 62),
        ("halve context", "q4 · 16k context", 40),
        ("halve again", "q4 · 8k context", 28),
        ("floor reached", "below 2k — refuse", 0),
    ]
    x0, y0, bwid, rowh = 200, 92, 380, 44

    for i, (name, detail, frac) in enumerate(steps):
        y = y0 + i * rowh
        refused = frac == 0
        p.append(f'<text x="{x0 - 14}" y="{y + 20}" class="lb" text-anchor="end">{name}</text>')
        if refused:
            p.append(
                f'<rect x="{x0}" y="{y}" width="{bwid}" height="26" rx="4" '
                f'fill="none" stroke="var(--s3)" stroke-dasharray="6 4"/>'
            )
            p.append(
                f'<text x="{x0 + 14}" y="{y + 18}" class="ax" fill="var(--s3)">{detail}</text>'
            )
        else:
            p.append(bar(x0, y, bwid * frac / 100, 26, 0))
            p.append(f'<text x="{x0 + bwid + 16}" y="{y + 18}" class="m">{detail}</text>')

    note(
        p,
        28,
        H - 46,
        "Every rung is reported, never applied silently. A deployment that serves "
        "8k when it was built for 32k is fine if you know; it is an incident if "
        "you do not.",
    )
    save("degradation.svg", p)


# --- 9. TPUs ------------------------------------------------------------------


def tpus() -> None:
    """Memory per host, because that is the number that decides what fits.

    Per-*chip* memory is the number everyone quotes and the wrong one to plan
    with: a TPU is rented by the host, and vLLM does not support multi-host
    serving on GKE — so one host is a hard ceiling, not a starting point.

    Specs verified against Google Cloud's per-generation pages, 2026-07-27.
    """
    W, H = 900, 400
    p = head(
        W,
        H,
        "TPUs: plan by the host, not the chip",
        "Memory available per TPU host for v5e, v6e and v5p, against an H100 and "
        "an 8×H100 node for scale. Per-chip figures are the ones usually quoted "
        "and the wrong ones to size against.",
    )
    p.append(
        '<text x="28" y="50" class="sub">Multi-host serving is unsupported on '
        "GKE, so one host is a ceiling rather than a starting point</text>"
    )

    rows = [
        ("TPU v5e · 8 chips", 16, 8, 128, 0),
        ("TPU v6e · 8 chips", 32, 8, 256, 0),
        ("TPU v5p · 4 chips", 95, 4, 380, 0),
        ("NVIDIA H100", 80, 1, 80, 1),
        ("8× H100 node", 80, 8, 640, 1),
    ]
    x0, y0, bwid, rowh = 210, 92, 400, 44
    scale = bwid / 640.0

    for i, (name, per_chip, chips, total, slot) in enumerate(rows):
        y = y0 + i * rowh
        p.append(f'<text x="{x0 - 14}" y="{y + 20}" class="lb" text-anchor="end">{name}</text>')
        p.append(bar(x0, y, max(total * scale, 3), 26, slot))
        p.append(
            f'<text x="{x0 + total * scale + 14:.0f}" y="{y + 19}" class="m">{total:,} GB</text>'
        )
        p.append(
            f'<text x="{x0 + bwid + 130}" y="{y + 19}" class="ax">{per_chip} GB × {chips}</text>'
        )

    p.append(legend(x0, y0 + len(rows) * rowh + 22, [(0, "TPU host"), (1, "NVIDIA")]))
    note(
        p,
        28,
        H - 62,
        "A v6e host holds 256 GB without leaving one machine, which is more than "
        "an H100 and less than an 8×H100 node — and unlike the node, it cannot be "
        "extended by adding a second one. vLLM lists v5e, v6e and v7x as "
        "recommended; v3, v4 and v5p are experimental, which is a different "
        "promise.",
    )
    save("edu-tpu.svg", p)


# --- 10. where a kernel sits, and what it actually moves -----------------------


def kernels() -> None:
    """Two questions in one picture: where a kernel sits, and what it moves.

    The top row is the call stack, because the common mental model is that a
    kernel is somewhere "inside vLLM" and unreachable without forking it. It is
    reachable, at exactly one seam, and the seam is an entry point.

    The bottom-left is the part that changes behaviour. Decode is memory-bound,
    so the honest unit for "where does the time go" is *bytes moved*, not FLOPs —
    and once it is drawn in bytes, the elementwise ops everyone wants to fuse are
    visibly ~0% of the step at batch 1. That is the same batch-1 trap as
    speculative decoding, which is why it is worth drawing rather than asserting.

    Arithmetic (7B-class dense layer, d=4096, 32 heads, 8 kv-heads, head_dim=128,
    intermediate=11008, fp16, 32k context), per layer:
        QKV   4096·(4096+1024+1024) = 25.17M params →  50.3 MB
        O     4096·4096             = 16.78M        →  33.6 MB
        gate+up 2·4096·11008        = 90.18M        → 180.4 MB
        down  11008·4096            = 45.09M        →  90.2 MB
        KV    8·128·2·32768·2 bytes                 → 134.2 MB
        total                                       ≈ 488.6 MB
    """
    W, H = 900, 664
    p = head(
        W,
        H,
        "Where a kernel sits, and what it moves",
        "Top: the request path from HTTP down to HBM, with the kernel layer "
        "highlighted and the vLLM plugin entry point marked as the only seam "
        "into it. Bottom left: one decode step decomposed by bytes moved. "
        "Bottom right: what fusing two elementwise kernels actually saves.",
    )
    p.append(
        '<text x="28" y="50" class="sub">You do not fork the engine to change a '
        "kernel — you register one. The hard part is proving it did not change "
        "the output.</text>"
    )

    # --- the stack ------------------------------------------------------------
    stages = [
        ("request", "OpenAI-shaped", "HTTP"),
        ("API server", "tokenize", "admit"),
        ("scheduler", "paged KV", "form the batch"),
        ("model forward", "torch graph", "layer by layer"),
        ("KERNELS", "attention · GEMM", "norm · sampling"),
        ("GPU", "SMs", "read / write HBM"),
    ]
    bx0, bw, gap, by, bh = 27, 126, 18, 78, 84
    kernel_i = 4

    for i, (name, l1, l2) in enumerate(stages):
        x = bx0 + i * (bw + gap)
        mid = x + bw / 2
        hot = i == kernel_i
        p.append(
            f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="10" '
            f'fill="var(--panel)" stroke="var(--s0)" stroke-width="2"/>'
            if hot
            else f'<rect x="{x}" y="{by}" width="{bw}" height="{bh}" rx="10" '
            f'fill="var(--panel)" stroke="var(--grid)"/>'
        )
        p.append(
            f'<text x="{mid:.0f}" y="{by + 26}" class="lb" text-anchor="middle"'
            + (' fill="var(--s0)">' if hot else ">")
            + f"{name}</text>"
        )
        p.append(f'<text x="{mid:.0f}" y="{by + 48}" class="ax" text-anchor="middle">{l1}</text>')
        p.append(f'<text x="{mid:.0f}" y="{by + 65}" class="ax" text-anchor="middle">{l2}</text>')
        if i:
            ax = x - gap / 2
            p.append(
                f'<path d="M {ax - 4:.0f} {by + bh / 2 - 5:.0f} l 5 5 l -5 5" '
                f'fill="none" stroke="var(--muted)" stroke-width="1.5"/>'
            )

    # --- the seam -------------------------------------------------------------
    kx = bx0 + kernel_i * (bw + gap) + bw / 2
    cx, cy, cw2, ch = 440, 214, 433, 64
    p.append(
        f'<path d="M {kx:.0f} {by + bh} V {cy:.0f}" fill="none" '
        f'stroke="var(--s0)" stroke-width="1.5" stroke-dasharray="4 3"/>'
    )
    p.append(
        f'<rect x="{cx}" y="{cy}" width="{cw2}" height="{ch}" rx="10" '
        f'fill="var(--panel)" stroke="var(--s0)"/>'
    )
    p.append(
        f'<text x="{cx + 18}" y="{cy + 25}" class="m" fill="var(--s0)">vllm.general_plugins</text>'
    )
    p.append(
        f'<text x="{cx + 18}" y="{cy + 46}" class="ax">An entry point vLLM '
        f"imports in every process it spawns.</text>"
    )
    for i, ln in enumerate(
        [
            "clickllm generates the config behind boxes 2–3. Box 4 is",
            "vLLM's, and forking it is a permanent tax. Box 5 is the only",
            "sanctioned seam — and register() runs once per worker, so it",
            "must be idempotent or tensor parallelism breaks it on rank 1.",
        ]
    ):
        p.append(f'<text x="28" y="{cy + 12 + i * 17}" class="ax">{ln}</text>')

    # --- panels ---------------------------------------------------------------
    py, ph = 300, 248
    for x, w in ((28, 496), (548, 325)):
        p.append(
            f'<rect x="{x}" y="{py}" width="{w}" height="{ph}" rx="10" '
            f'fill="var(--panel)" stroke="var(--grid)"/>'
        )

    p.append(f'<text x="46" y="{py + 26}" class="lb">One decode step, by bytes moved</text>')
    p.append(
        f'<text x="46" y="{py + 44}" class="ax">7B-class layer · 8 kv-heads · '
        f"32k context · fp16 · batch 1</text>"
    )

    # (label, MB, slot) — slot 0 is the KV read, 1 the weight reads, 3 elementwise.
    rows = [
        ("QKV projections", 50.3, 1),
        ("attention · KV read", 134.2, 0),
        ("O projection", 33.6, 1),
        ("gate + up", 180.4, 1),
        ("down", 90.2, 1),
        ("RMSNorm + SiLU", 0.1, 3),
    ]
    total = sum(mb for _, mb, _ in rows)
    lx, rx0, rw, ry0, rh = 170, 182, 210, 358, 27
    scale = rw / max(mb for _, mb, _ in rows)

    for i, (name, mb, slot) in enumerate(rows):
        y = ry0 + i * rh
        pct = 100 * mb / total
        p.append(f'<text x="{lx}" y="{y + 14}" class="ax" text-anchor="end">{name}</text>')
        p.append(bar(rx0, y, max(mb * scale, 2), 18, slot))
        label = "&lt;0.1%" if pct < 0.1 else f"{pct:.1f}%"
        p.append(
            f'<text x="{rx0 + max(mb * scale, 2) + 10:.0f}" y="{y + 13}" class="m">{label}</text>'
        )

    p.append(
        f'<text x="{lx}" y="{ry0 + len(rows) * rh + 14}" class="ax" text-anchor="end">total</text>'
    )
    p.append(
        f'<text x="{rx0}" y="{ry0 + len(rows) * rh + 14}" class="m" '
        f'fill="var(--ink)">{total:.0f} MB moved per layer</text>'
    )

    # --- what fusion buys -----------------------------------------------------
    p.append(f'<text x="566" y="{py + 26}" class="lb">What fusion actually saves</text>')
    p.append(f'<text x="566" y="{py + 46}" class="m">y = silu(gate) * up</text>')

    unit, ugap = 50, 4
    for i, (name, trips) in enumerate((("unfused · 2 kernels", 5), ("fused · 1 kernel", 3))):
        y = py + 74 + i * 54
        p.append(f'<text x="566" y="{y}" class="ax">{name} · {trips} HBM trips</text>')
        for t in range(trips):
            p.append(bar(566 + t * (unit + ugap), y + 8, unit, 16, 3, r=3))

    p.append(
        f'<text x="566" y="{py + 196}" class="ax">Three HBM round trips instead of five —</text>'
    )
    p.append(
        f'<text x="566" y="{py + 213}" class="ax">a 40% cut, on the one term that is under</text>'
    )
    p.append(f'<text x="566" y="{py + 230}" class="ax">0.1% of the step at batch 1.</text>')

    p.append(
        legend(
            28,
            py + ph + 30,
            [(1, "weight reads"), (0, "KV cache read"), (3, "elementwise / activations")],
        )
    )
    note(
        p,
        28,
        py + ph + 58,
        "This is why a kernel is measured at real concurrency or not at all. At "
        "batch 1 the weights dominate and fusion is noise; at batch 64 the weight "
        "read is amortised across every sequence, the activation traffic scales "
        "with the batch, and the same fusion becomes the whole win. Benchmark at "
        "the wrong batch size and you will ship the opposite conclusion — the "
        "identical trap speculative decoding sets.",
    )
    save("edu-kernels.svg", p)


# --- 11. what the silicon is actually doing ------------------------------------


def silicon() -> None:
    """How much of the machine you are actually using, in plain words.

    The first version of this taught nothing. It had 132 tiny squares, five
    memory stacks, a bandwidth pipe, a six-row roofline chart and a legend, and
    it spoke in "arithmetic intensity" and "FLOP per byte" and "the ridge
    point". All of that is true and none of it is *informative* to someone who
    has not already met the concept — it was an expert's artefact wearing a
    teaching diagram's clothes, in a module whose whole brief is to take a
    non-specialist to competence.

    So this is the same fact told as three plain beats, with no jargon and no
    second chart bolted underneath:

        1. The machine has 132 engines. About one of them is busy.
        2. Why: to write a single word it re-reads all 30.5 GB of the model.
           The maths it does with those bytes is trivial by comparison.
        3. The fix and its ceiling: serve 18 people and that same one read
           produces 18 words. Ask for 19 and the cache has nowhere to live.

    Every figure is `clickllm fit`'s for Qwen3 32B at q8 and 8k context on an
    H100 — the 30.5 / 36.0 / 72.5-vs-72 numbers are a real solve, not a
    illustration, and a test holds them to it.
    """
    W, H = 900, 690
    p = head(
        W,
        H,
        "You bought 132 engines. Your memory feeds about one.",
        "An H100 serving a 32B model. Only one of its 132 compute engines has "
        "work at any moment, because producing a single word means re-reading "
        "the entire 30.5 GB model from memory. Serving 18 people at once shares "
        "that one read across 18 words; a 19th does not fit, because the "
        "conversation cache has nowhere left to live.",
    )
    p.append(
        '<text x="28" y="50" class="sub">Why a fast GPU can still feel slow — '
        "and the one thing that actually fixes it</text>"
    )

    WEIGHTS_GB, SMS, LIT = 30.5, 132, 1
    GOOD_BATCH, OVER_BATCH = 18, 19
    NEED_GB, HAVE_GB = 72.5, 72.0

    # --- beat 1: the machine --------------------------------------------------
    dx, dy, dw, dh = 28, 84, 336, 322
    p.append(chip(dx, dy, dw, dh, pins_per_side=6, net=False))
    p.append(f'<text x="{dx + 20}" y="{dy + 28}" class="lb">the GPU, actual size</text>')
    # Cells big enough to count at a glance. The first version made them 17px in
    # a cramped box, which read as texture rather than as 132 separate things.
    cols, cell, pitch = 12, 22, 24
    gx, gy = dx + 22, dy + 46
    for i in range(SMS):
        cx, cy = gx + (i % cols) * pitch, gy + (i // cols) * pitch
        if i < LIT:
            p.append(bar(cx, cy, cell, cell, 2, r=4))
        else:
            p.append(
                f'<rect x="{cx}" y="{cy}" width="{cell}" height="{cell}" rx="4" '
                f'fill="none" stroke="var(--grid)"/>'
            )

    tx = dx + dw + 34
    p.append(
        f'<text x="{tx}" y="{dy + 96}" class="t" fill="var(--s2)" font-size="34">1 of {SMS}</text>'
    )
    p.append(f'<text x="{tx}" y="{dy + 124}" class="lb">engines busy</text>')
    for i, line in enumerate(
        [
            "The other 131 are not working on",
            "anything. They are waiting for the",
            "memory to hand them the next piece",
            "of the model.",
        ]
    ):
        p.append(f'<text x="{tx}" y="{dy + 164 + i * 19}" class="ax">{line}</text>')
    p.append(
        f'<text x="{tx}" y="{dy + 258}" class="lb" fill="var(--s0)">'
        f"So the GPU is not the slow part.</text>"
    )
    p.append(f'<text x="{tx}" y="{dy + 278}" class="lb" fill="var(--s0)">The memory is.</text>')

    # --- beat 2: why ----------------------------------------------------------
    by = 442
    p.append(f'<text x="28" y="{by}" class="lb">Because one word costs a whole read</text>')

    ry, rh2 = by + 18, 46
    # The widths carry the argument: an enormous read, a sliver of maths, one
    # word out. Drawn to scale against each other rather than labelled as such.
    p.append(bar(28, ry, 470, rh2, 1))
    p.append(
        f'<text x="{28 + 235}" y="{ry + 28}" class="lb" text-anchor="middle" '
        f'fill="var(--bg)">read all {WEIGHTS_GB} GB of the model</text>'
    )
    p.append(
        f'<path d="M 508 {ry + 17} l 10 6 l -10 6" fill="none" stroke="var(--muted)" '
        f'stroke-width="1.5"/>'
    )
    p.append(bar(530, ry, 66, rh2, 3))
    p.append(
        f'<text x="563" y="{ry + 28}" class="ax" text-anchor="middle" '
        f'fill="var(--bg)">a little</text>'
    )
    p.append(
        f'<path d="M 606 {ry + 17} l 10 6 l -10 6" fill="none" stroke="var(--muted)" '
        f'stroke-width="1.5"/>'
    )
    p.append(bar(628, ry, 54, rh2, 2))
    p.append(
        f'<text x="655" y="{ry + 28}" class="ax" text-anchor="middle" '
        f'fill="var(--bg)">1 word</text>'
    )
    p.append(f'<text x="700" y="{ry + 21}" class="ax">and then it starts</text>')
    p.append(f'<text x="700" y="{ry + 38}" class="ax">again for word two.</text>')

    # --- beat 3: the fix, and where it stops ----------------------------------
    fy = ry + 88
    p.append(f'<text x="28" y="{fy}" class="lb">Which is why you serve people in batches</text>')

    rows = [
        (f"{1} person", f"read {WEIGHTS_GB} GB", "1 word", True, ""),
        (
            f"{GOOD_BATCH} people",
            f"read {WEIGHTS_GB} GB",
            f"{GOOD_BATCH} words",
            True,
            f"the same read, {GOOD_BATCH}x the output",
        ),
        (
            f"{OVER_BATCH} people",
            "no room left",
            "—",
            False,
            f"needs {NEED_GB} GB, the card has {HAVE_GB:.0f}",
        ),
    ]
    y0, rh3 = fy + 20, 34
    for i, (who, what, out, ok, why) in enumerate(rows):
        y = y0 + i * rh3
        p.append(f'<text x="132" y="{y + 16}" class="ax" text-anchor="end">{who}</text>')
        if ok:
            p.append(bar(146, y, 200, 22, 1))
            p.append(
                f'<text x="246" y="{y + 16}" class="ax" text-anchor="middle" '
                f'fill="var(--bg)">{what}</text>'
            )
            p.append(bar(358, y, 96, 22, 2))
            p.append(
                f'<text x="406" y="{y + 16}" class="ax" text-anchor="middle" '
                f'fill="var(--bg)">{out}</text>'
            )
        else:
            p.append(
                f'<rect x="146" y="{y}" width="308" height="22" rx="4" fill="none" '
                f'stroke="var(--s3)" stroke-dasharray="5 4"/>'
            )
            p.append(
                f'<text x="300" y="{y + 16}" class="ax" text-anchor="middle" '
                f'fill="var(--s3)">{what}</text>'
            )
        if why:
            p.append(f'<text x="470" y="{y + 16}" class="ax">{why}</text>')

    note(
        p,
        28,
        y0 + len(rows) * rh3 + 26,
        "Batching is the whole game, and the conversation cache is what ends it: every "
        "extra person needs their own copy, so you run out of memory long before you run "
        "out of engines. Making that cache smaller — a lighter quantisation, a shorter "
        "context, an MLA model — is the same thing as buying more of the GPU you already "
        "own. Figures are a real `clickllm fit` solve, not an illustration.",
    )
    save("edu-silicon.svg", p)


# --- 12. weights streaming onto silicon ----------------------------------------


def weights_flow() -> None:
    """The decode loop, animated: weights leave memory, the die lights, one word.

    This one earns its motion. The product's whole argument is that memory —
    not maths — decides how a deployment feels, and that argument is a *loop*:
    every single token drags the entire model across the bus again. A still
    picture can state that; only a moving one shows the "again".

    It replaced a field of drifting particles on the landing page. Those had no
    referent — they were dots, and dots do not mean anything. These blocks are
    the weights, the lane is the memory bus, and the grid is the die.
    """
    W, H = 900, 254
    p = head(
        W,
        H,
        "Every token re-reads the whole model",
        "Animated: blocks of model weights stream out of memory, across the bus, "
        "onto the compute die, which lights up and emits a single word — then the "
        "whole read starts again for the next word.",
    )
    p.append(
        '<text x="28" y="50" class="sub">Which is why memory, not maths, decides '
        "how fast this feels</text>"
    )

    lane_y, lane_h = 108, 44

    # --- memory ---------------------------------------------------------------
    p.append(
        f'<rect x="28" y="{lane_y - 16}" width="118" height="{lane_h + 32}" rx="9" '
        f'fill="var(--panel)" stroke="var(--grid)"/>'
    )
    for i in range(4):
        p.append(bar(40, lane_y - 6 + i * 17, 94, 12, 1, r=3))
    p.append(f'<text x="87" y="{lane_y - 24}" class="ax" text-anchor="middle">memory</text>')
    p.append(
        f'<text x="87" y="{lane_y + lane_h + 32}" class="m" text-anchor="middle">30.5 GB</text>'
    )

    # --- the bus --------------------------------------------------------------
    bx0, bx1 = 160, 584
    p.append(
        f'<rect x="{bx0}" y="{lane_y}" width="{bx1 - bx0}" height="{lane_h}" rx="8" '
        f'fill="var(--panel)" stroke="var(--grid)"/>'
    )
    p.append(
        f'<clipPath id="lane"><rect x="{bx0 + 2}" y="{lane_y + 2}" '
        f'width="{bx1 - bx0 - 4}" height="{lane_h - 4}" rx="7"/></clipPath>'
    )
    # One extra block beyond the lane, and a translate of exactly one pitch, so
    # the loop is seamless rather than visibly snapping back.
    p.append('<g clip-path="url(#lane)">')
    pitch = 46
    for i in range(-1, (bx1 - bx0) // pitch + 2):
        p.append(
            f'<g class="stream"><rect x="{bx0 + i * pitch + 6}" y="{lane_y + 11}" '
            f'width="30" height="22" rx="4" fill="var(--s1)" opacity="0.85"/></g>'
        )
    p.append("</g>")
    p.append(
        f'<text x="{(bx0 + bx1) / 2:.0f}" y="{lane_y - 24}" class="ax" '
        f'text-anchor="middle">the memory bus — saturated, every token</text>'
    )

    # --- the silicon ----------------------------------------------------------
    # The chip is the centrepiece: pins on every side and a model on the die,
    # its nodes lighting as the weights land. It is the same object the site
    # uses as its hardware mark, at a size where the graph is legible.
    csz = 126
    dx, dy = 604, lane_y + lane_h / 2 - csz / 2
    p.append(chip(dx, dy, csz, csz, pins_per_side=4, accent=1, anim=True, proven=True))
    p.append(
        f'<text x="{dx + csz / 2:.0f}" y="{dy - 18}" class="ax" '
        f'text-anchor="middle">the model, on the silicon</text>'
    )

    # --- one word out ---------------------------------------------------------
    tx = dx + csz + 26
    p.append(
        f'<path d="M {tx - 14} {lane_y + 16} l 9 6 l -9 6" fill="none" '
        f'stroke="var(--muted)" stroke-width="1.5"/>'
    )
    p.append(
        f'<rect class="light" style="animation-delay:1.6s" x="{tx}" y="{lane_y + 8}" '
        f'width="52" height="26" rx="5" fill="var(--s2)"/>'
    )
    p.append(
        f'<text x="{tx + 26}" y="{lane_y + 25}" class="ax" text-anchor="middle" '
        f'fill="var(--bg)">1 word</text>'
    )

    note(
        p,
        28,
        H - 34,
        "Then it happens again, from the start, for the next word. That loop is the "
        "product: the bus is the ceiling, and everything clickllm tunes — batching, "
        "quantisation, KV layout — is a way of getting more words out of one trip across it.",
    )
    save("hero-weights.svg", p)


if __name__ == "__main__":
    print("writing diagrams:")
    memory_breakdown()
    prefill_decode()
    kv_growth()
    batching()
    spec_decode()
    quantization()
    in_a_box()
    degradation()
    tpus()
    kernels()
    silicon()
    weights_flow()
    print("done")
