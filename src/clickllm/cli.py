"""clickllm CLI. Stdlib only — `uv run clickllm fit` must work with zero install."""

from __future__ import annotations

import argparse
import json
import sys

from . import catalog, fit, hardware

GB = 1024**3


def _parse_size(s: str) -> int:
    """'32k' -> 32768, '128000' -> 128000."""
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    if s.endswith("m"):
        return int(float(s[:-1]) * 1024 * 1024)
    return int(s)


def _lic(m: catalog.ModelSpec) -> str:
    return f"{m.license} {'OK' if m.license_ok else '!'}"


def cmd_fit(args: argparse.Namespace) -> int:
    hw = hardware.detect()
    ctx, conc = _parse_size(args.context), args.concurrency

    if args.explain:
        m = catalog.get(args.explain)
        f = fit.best_quant(m, hw, ctx, conc) or fit.solve(
            m, min(m.quants, key=lambda q: catalog.QUANT_BITS[q]), hw, ctx, conc
        )
        print(f"\n{f.explain()}\n")
        print(f"  verdict: {'FITS' if f.feasible else 'DOES NOT FIT'}\n")
        return 0 if f.feasible else 1

    feasible, rejected = fit.rank(hw, ctx, conc)

    if args.json:
        print(
            json.dumps(
                {
                    "hardware": hw.to_dict(),
                    "context": ctx,
                    "concurrency": conc,
                    "feasible": [
                        {
                            "id": f.model.id,
                            "quant": f.quant,
                            "verified": f.model.verified,
                            "weights_gb": round(f.weight_bytes / GB, 1),
                            "kv_gb": round(f.kv_bytes / GB, 1),
                            "total_gb": round(f.total_bytes / GB, 1),
                            "headroom_gb": round(f.headroom_bytes / GB, 1),
                            "tokens_per_sec": round(f.tokens_per_sec) if f.tokens_per_sec else None,
                            "license": f.model.license,
                            "license_ok": f.model.license_ok,
                        }
                        for f in feasible
                    ],
                    "rejected": [{"id": m.id, "reason": why} for m, why in rejected],
                    "runtime": dict(
                        zip(("name", "why"), fit.recommend_runtime(hw, ctx, conc), strict=True)
                    ),
                },
                indent=2,
            )
        )
        return 0

    bw = f" · {hw.bandwidth_gbps:g} GB/s" if hw.bandwidth_gbps else ""
    dev = f" · {hw.devices}x" if hw.devices > 1 else ""
    print(f"\n  {hw.name} · {hw.cores or '?'} cores · {hw.total_gb:.0f} GB{bw}{dev}")
    print(f"  usable for inference: {hw.usable_gb:.0f} GB")
    if hw.note:
        print(f"  {hw.note}")

    print(f"\n  FEASIBLE at {ctx:,} context, concurrency {conc}\n")
    if not feasible:
        print("  nothing in the catalog fits. try a smaller context or --concurrency 1\n")
    else:
        print(
            f"  {'model':<26}{'quant':<7}{'weights':>8}{'kv':>8}{'total':>8}"
            f"{'free':>8}{'~tok/s':>8}  license"
        )
        print(f"  {'-' * 88}")
        for f in feasible:
            flag = " ?" if not f.model.verified else ""
            tps = f"{f.tokens_per_sec:.0f}" if f.tokens_per_sec else "-"
            slow = " slow" if f.slow else ""
            print(
                f"  {f.model.name[:24] + flag:<26}{f.quant:<7}"
                f"{f.weight_bytes / GB:>7.1f}G{f.kv_bytes / GB:>7.1f}G"
                f"{f.total_bytes / GB:>7.1f}G{f.headroom_bytes / GB:>7.1f}G"
                f"{tps:>8}{slow}  {_lic(f.model)}"
            )

    if rejected and not args.quiet:
        print("\n  NOT FEASIBLE\n")
        for m, why in rejected:
            print(f"  {m.name[:24]:<26}{why}")

    name, why = fit.recommend_runtime(hw, ctx, conc)
    print(f"\n  runtime -> {name}")
    for line in _wrap(why, 76):
        print(f"             {line}")

    unverified = [f for f in feasible if not f.model.verified]
    if unverified:
        print(
            f"\n  ? = architecture unverified; KV figures are estimates ({len(unverified)} shown)"
        )
    print("\n  clickllm fit --explain <model-id>   # show the arithmetic\n")
    return 0


def _wrap(text: str, width: int) -> list[str]:
    out, line = [], ""
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def cmd_models(args: argparse.Namespace) -> int:
    print(f"\n  {'id':<22}{'params':>9}{'active':>9}{'ctx':>10}  license")
    print(f"  {'-' * 66}")
    for m in sorted(catalog.load(), key=lambda m: m.params_b):
        act = f"{m.active_b:g}B" if m.is_moe else "-"
        print(f"  {m.id:<22}{m.params_b:>8g}B{act:>9}{m.max_context:>10,}  {_lic(m)}")
    print()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="clickllm",
        description="Prove which open model can replace your closed one.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="what runs on this machine")
    f.add_argument("--context", default="32k", help="context length, e.g. 8k, 32k, 128000")
    f.add_argument("--concurrency", type=int, default=1, help="simultaneous requests")
    f.add_argument("--explain", metavar="MODEL_ID", help="show the arithmetic for one model")
    f.add_argument("--json", action="store_true")
    f.add_argument("--quiet", action="store_true", help="hide the NOT FEASIBLE section")
    f.set_defaults(fn=cmd_fit)

    m = sub.add_parser("models", help="list the catalog")
    m.set_defaults(fn=cmd_models)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
