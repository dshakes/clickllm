"""clickllm CLI. Stdlib only — `uv run clickllm fit` must work with zero install."""

from __future__ import annotations

import argparse
import json
import pathlib
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


def cmd_where(args: argparse.Namespace) -> int:
    """Which hardware classes can serve this model."""
    m = catalog.get(args.model)
    ctx, conc = _parse_size(args.context), args.concurrency
    placements = fit.where(m, ctx, conc)

    if args.json:
        print(
            json.dumps(
                {
                    "model": m.id,
                    "context": ctx,
                    "concurrency": conc,
                    "placements": [
                        {
                            "profile": p.profile_id,
                            "name": p.profile_name,
                            "feasible": p.feasible,
                            "quant": p.fit.quant if p.fit else None,
                            "total_gb": round(p.fit.total_bytes / GB, 1) if p.fit else None,
                            "tokens_per_sec": (
                                round(p.tokens_per_sec) if p.tokens_per_sec else None
                            ),
                            "hourly_usd": p.hourly_usd,
                            "usd_per_mtok": (
                                round(p.cost_per_mtok_usd, 2) if p.cost_per_mtok_usd else None
                            ),
                            "reason": p.reason or None,
                        }
                        for p in placements
                    ],
                },
                indent=2,
            )
        )
        return 0

    ok = [p for p in placements if p.feasible]
    no = [p for p in placements if not p.feasible]

    print(f"\n  {m.name} · {m.params_b:g}B", end="")
    if m.is_moe:
        print(f" ({m.active_b:g}B active, MoE)", end="")
    print(f" · {_lic(m)}")
    print(f"  at {ctx:,} context, concurrency {conc}\n")

    if ok:
        print(f"  {'hardware':<26}{'quant':<7}{'total':>8}{'~tok/s':>8}{'$/hr':>8}{'$/Mtok':>9}")
        print(f"  {'-' * 66}")
        for p in ok:
            f = p.fit
            tps = f"{p.tokens_per_sec:.0f}" if p.tokens_per_sec else "-"
            hr = f"{p.hourly_usd:.2f}" if p.hourly_usd is not None else "-"
            mt = f"{p.cost_per_mtok_usd:.2f}" if p.cost_per_mtok_usd else "-"
            slow = " slow" if f and f.slow else ""
            print(
                f"  {p.profile_name[:24]:<26}{f.quant:<7}"
                f"{f.total_bytes / GB:>7.1f}G{tps:>8}{hr:>8}{mt:>9}{slow}"
            )
    else:
        print("  Nothing in the hardware catalogue can serve this model at that shape.\n")

    if no and not args.quiet:
        print("\n  WILL NOT RUN")
        for p in no:
            print(f"  {p.profile_name[:24]:<26}{p.reason}")

    print(
        "\n  $/Mtok assumes the machine is saturated single-stream; real cost is higher."
        "\n  Throughput figures are roofline estimates, not measurements.\n"
    )
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    """Verify catalogue entries against each model's published config."""
    from . import catalog_update as cu

    specs = [m for m in catalog.load() if m.repo]
    skipped = [m for m in catalog.load() if not m.repo]
    if args.model:
        specs = [m for m in specs if m.id == args.model]
        if not specs:
            print(f"error: {args.model} has no known repo to verify against", file=sys.stderr)
            return 2

    if not args.network:
        print(
            "\n  Catalogue verification needs network access, which is opt-in.\n"
            f"  {len(specs)} entries have a known repo; {len(skipped)} do not.\n\n"
            "  Re-run with --network to fetch each model's config.json.\n"
            "  Nothing is written without --apply.\n"
        )
        return 0

    print(f"\n  Checking {len(specs)} entries against their published configs…\n")
    proposals = [cu.propose(m, m.repo or "", cu.http_fetch) for m in specs]
    report = cu.UpdateReport(proposals)
    print(report.render())

    if args.apply:
        n = cu.apply_proposals(proposals)
        print(f"\n  Applied {n} update(s) to the catalogue.")
        if any(p.significant for p in report.changed):
            print("  Memory figures changed — re-run `clickllm fit` before deploying.")
    print()
    return 0


def cmd_discover(args: argparse.Namespace) -> int:
    """Models trending in the wild that this catalogue does not carry."""
    from . import catalog_update as cu

    if not args.network:
        print("\n  Discovery needs network access, which is opt-in.\n  Re-run with --network.\n")
        return 0

    known = {m.repo for m in catalog.load() if m.repo}
    found = cu.discover(known, cu.http_fetch, limit=args.limit)
    if not found:
        print("\n  Nothing new, or the index was unreachable.\n")
        return 0

    print(f"\n  {len(found)} trending models not in the catalogue:\n")
    for d in found[:20]:
        print(d.render())
    print(
        "\n  This is a shortlist, not a recommendation. A model trending publicly"
        "\n  says nothing about whether it fits your hardware or your workload —"
        "\n  run `clickllm where <model>` and prove it on your traffic first.\n"
    )
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    """Launch the local workbench."""
    from . import ui

    return ui.serve(host=args.host, port=args.port, open_browser=not args.no_open)


def cmd_models(args: argparse.Namespace) -> int:
    print(f"\n  {'id':<22}{'params':>9}{'active':>9}{'ctx':>10}  license")
    print(f"  {'-' * 66}")
    for m in sorted(catalog.load(), key=lambda m: m.params_b):
        act = f"{m.active_b:g}B" if m.is_moe else "-"
        print(f"  {m.id:<22}{m.params_b:>8g}B{act:>9}{m.max_context:>10,}  {_lic(m)}")
    print()
    return 0


def cmd_guard(args: argparse.Namespace) -> int:
    """Check whether a receipt still describes production.

    Heavy imports stay inside the handler so `clickllm fit` remains stdlib-only.
    """
    from datetime import date

    from .guard import check
    from .prove.receipt import Receipt

    receipt = Receipt.from_json(pathlib.Path(args.receipt).read_text())

    traffic = None
    if args.traffic:
        raw = json.loads(pathlib.Path(args.traffic).read_text())
        traffic = {str(k): float(v) for k, v in raw.items()}

    fingerprints = None
    if args.fingerprints:
        raw = json.loads(pathlib.Path(args.fingerprints).read_text())
        fingerprints = {str(k): str(v) for k, v in raw.items()}

    today = date.fromisoformat(args.today) if args.today else date.today()
    proposal = check(
        receipt,
        today=today,
        fingerprints=fingerprints,
        traffic=traffic,
        available=frozenset(args.available or ()),
    )

    if args.json:
        print(
            json.dumps(
                {
                    "receipt": proposal.receipt_digest,
                    "valid": proposal.valid,
                    "action": proposal.action,
                    "findings": [
                        {
                            "kind": f.kind.value,
                            "subject": f.subject,
                            "detail": f.detail,
                            "invalidates": f.invalidates,
                        }
                        for f in proposal.findings
                    ],
                },
                indent=2,
            )
        )
    else:
        print()
        print(proposal.render())
        print()

    # Nonzero when the receipt no longer describes production, so this is usable
    # as a cron job or a CI step without parsing the output.
    return 0 if proposal.valid else 1


def cmd_receipt(args: argparse.Namespace) -> int:
    """Render or verify a receipt someone handed you."""
    from .prove.receipt import Receipt, verify

    r = Receipt.from_json(pathlib.Path(args.receipt).read_text())
    if not args.against:
        print()
        print(r.render())
        print()
        return 0

    other = Receipt.from_json(pathlib.Path(args.against).read_text())
    ok, diffs = verify(r, other)
    print()
    if ok:
        print(f"verified · both receipts agree · {r.digest()[:12]}")
        print()
        return 0
    print(f"DOES NOT VERIFY · {r.digest()[:12]} vs {other.digest()[:12]}")
    for d in diffs:
        print(f"  · {d.render()}")
    print()
    return 1


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

    w = sub.add_parser("where", help="which hardware can run a given model")
    w.add_argument("model", help="catalogue model id, e.g. glm-5.2")
    w.add_argument("--context", default="32k", help="context length, e.g. 8k, 32k")
    w.add_argument("--concurrency", type=int, default=1, help="simultaneous requests")
    w.add_argument("--json", action="store_true")
    w.add_argument("--quiet", action="store_true", help="hide the WILL NOT RUN section")
    w.set_defaults(fn=cmd_where)

    c = sub.add_parser("catalog", help="verify catalogue entries against published configs")
    c.add_argument("--model", help="check one entry instead of all")
    c.add_argument("--network", action="store_true", help="allow network access (required)")
    c.add_argument("--apply", action="store_true", help="write the proposed changes")
    c.set_defaults(fn=cmd_catalog)

    d = sub.add_parser("discover", help="trending models not yet in the catalogue")
    d.add_argument("--network", action="store_true", help="allow network access (required)")
    d.add_argument("--limit", type=int, default=40)
    d.set_defaults(fn=cmd_discover)

    u = sub.add_parser("ui", help="launch the local workbench")
    u.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    u.add_argument("--port", type=int, default=7171)
    u.add_argument("--no-open", action="store_true", help="don't open a browser")
    u.set_defaults(fn=cmd_ui)

    m = sub.add_parser("models", help="list the catalog")
    m.set_defaults(fn=cmd_models)

    g = sub.add_parser("guard", help="check whether a receipt still holds")
    g.add_argument("receipt", help="path to a receipt JSON file")
    g.add_argument("--traffic", help="JSON file of current cluster shares")
    g.add_argument("--fingerprints", help="JSON file of current model fingerprints")
    g.add_argument(
        "--available", action="append", help="a candidate model that exists now (repeatable)"
    )
    g.add_argument("--today", help="ISO date to evaluate against (default: today)")
    g.add_argument("--json", action="store_true")
    g.set_defaults(fn=cmd_guard)

    rc = sub.add_parser("receipt", help="render or verify a migration receipt")
    rc.add_argument("receipt", help="path to a receipt JSON file")
    rc.add_argument("--against", help="a second receipt to verify this one against")
    rc.set_defaults(fn=cmd_receipt)

    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except (KeyError, ValueError, OSError) as e:
        # Convention: a bad path, an altered receipt or an unparseable date is a
        # nonzero exit with a sentence, never a traceback. `json.JSONDecodeError`
        # is a `ValueError`, so malformed input is covered here too.
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
