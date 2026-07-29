"""clickllm CLI. Stdlib only — `uv run clickllm fit` must work with zero install."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import TYPE_CHECKING

from . import catalog, fit, hardware

if TYPE_CHECKING:  # imported lazily at the call sites — `clickllm fit` must stay cheap
    from .prove.collect import Collection
    from .prove.graders import EvalItem

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


def cmd_catalog_add(args: argparse.Namespace) -> int:
    """Add a model to the catalogue from its own published config.

    The point is that new models arrive constantly and none of them should need
    a release of this tool. The geometry is read from the model's `config.json`
    — the same ground truth `clickllm catalog` verifies against — and written to
    a drop-in file, so the model is available on the very next command.

    Nothing is guessed. A config missing a field we need is an error, because a
    guessed head count produces a confident and wrong memory figure.
    """
    from . import catalog_update as cu

    if not args.network:
        print(
            f"\n  Reading {args.repo}'s config.json needs network access, which is opt-in."
            "\n  Re-run with --network.\n"
        )
        return 0

    url = f"https://huggingface.co/{args.repo}/raw/main/config.json"
    try:
        arch = cu.parse_config(json.loads(cu.http_fetch(url)))
    except cu.ConfigError as e:
        print(f"error: {args.repo}: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — network and JSON, reported not raised
        print(f"error: could not read {url}: {e}", file=sys.stderr)
        return 2

    model_id = args.id or args.repo.split("/")[-1].lower()
    params = args.params_b
    spec = {
        "id": model_id,
        "name": args.name or args.repo.split("/")[-1],
        "params_b": params,
        "active_b": args.active_b if args.active_b is not None else params,
        "layers": arch.layers,
        "kv_heads": arch.kv_heads,
        "head_dim": arch.head_dim,
        "kv_scheme": arch.kv_scheme,
        "max_context": arch.max_context,
        "license": args.license,
        "license_ok": args.license_ok,
        "quants": list(args.quants.split(",")),
        # False on purpose. The geometry came from the config, but nobody has
        # checked the parameter count or the licence, and the catalogue's `?`
        # marker is how the tool stays honest about that.
        "verified": False,
        "repo": args.repo,
    }
    if arch.kv_lora_rank:
        spec["kv_lora_rank"] = arch.kv_lora_rank

    dest = pathlib.Path(args.out).expanduser() if args.out else _drop_in_dir() / f"{model_id}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"models": [spec]}, indent=2) + "\n")

    print(f"\n  wrote {dest}")
    print(f"  {model_id}: {arch.layers} layers · {arch.kv_heads} kv-heads · {arch.kv_scheme}")
    print(f"  marked unverified — the geometry is from {args.repo}'s config, the")
    print("  parameter count and licence are what you passed in.\n")
    print(f"  clickllm fit --explain {model_id}\n")
    return 0


def _drop_in_dir() -> pathlib.Path:
    """Where `catalog add` writes, and `catalog.sources()` reads."""
    base = os.environ.get("XDG_CONFIG_HOME", pathlib.Path.home() / ".config")
    return pathlib.Path(base) / "clickllm" / "models.d"


def cmd_build(args: argparse.Namespace) -> int:
    """Say what you are building; get a deployment and how to prove it.

    The whole flow in one command: read the sentence, size the machine, choose,
    configure, critique, and hand over. It stops at the door — nothing here runs
    anything.
    """
    from .session import Session

    s = Session()
    if args.resume:
        s = Session.from_json(pathlib.Path(args.resume).read_text())
    # Apply every input as a state mutation, WITHOUT calling step() in between —
    # step() is where a worth-asking question gets committed to `s.asked`, and
    # committing it on an intermediate call whose Turn we then discard is how
    # the single most valuable question got silently lost before a user ever
    # saw it. One step() call at the end means exactly one commit, in the
    # correct priority order.
    if args.description:
        s._apply_text(" ".join(args.description))
    # Only touch hardware when a profile was named or none is known yet — a
    # bare --resume must not silently re-detect the local machine and discard
    # a previously saved --on profile (e.g. a remote "h100"). mcp._build guards
    # the same way; this was a real inconsistency between the two surfaces.
    if args.on or s.hw is None:
        s._apply_hardware(args.on)

    for name in ("concurrency", "context", "ttft_ms", "itl_ms", "prefix_sharing"):
        v = getattr(args, name, None)
        if v is not None:
            s._apply_fields(**{name: _parse_size(v) if name == "context" else v})

    turn = s.step()
    if args.json:
        print(
            json.dumps(
                {
                    "stage": turn.stage.value,
                    "said": turn.said,
                    "question": turn.question,
                    "evidence": list(turn.evidence),
                    "assuming": list(turn.assuming),
                    "answer": s.answer(),
                    "state": json.loads(s.to_json()),
                },
                indent=2,
            )
        )
        return 0

    print()
    print(turn.render())
    print()
    print(s.answer())
    print()
    if args.save:
        pathlib.Path(args.save).write_text(s.to_json())
        print(f"  session saved to {args.save} — resume with --resume {args.save}\n")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Serve a model on this machine. The only command here that starts anything.

    Still no cutover: this binds loopback and hands you a URL. Moving production
    traffic onto it is a separate, human decision — see `clickllm prove`.
    """
    from . import launch

    p = launch.plan(
        args.model,
        quant=args.quant,
        context=_parse_size(args.context),
        concurrency=args.concurrency,
        port=args.port,
        host=args.host,
    )
    print()
    print(p.render())
    print()
    if isinstance(p, launch.Refusal):
        return 1
    if args.dry_run:
        print("  --dry-run: nothing was started.\n")
        return 0

    ep = launch.serve(p)
    print(f"\n  ready in {ep.ready_seconds:.0f}s — {ep.base}/v1  (pid {ep.pid})")
    print(f"  curl {ep.base}/v1/models")
    print("  ctrl-c to stop\n")
    return ep.wait()


def cmd_host(args: argparse.Namespace) -> int:
    """Where to run a model this machine cannot, cheapest first.

    The other side of `run`: that one refuses when the model does not fit and
    names this. Nothing here spends money or holds a credential — it ranks the
    options against a dated price registry and, with `--write`, emits a deploy
    artifact you run yourself.
    """
    from . import host

    m = catalog.get(args.model)
    s = host.survey(
        m,
        quant=args.quant,
        context=_parse_size(args.context),
        concurrency=args.concurrency,
        free_only=args.free_only,
    )

    if args.json:
        print(json.dumps(host.to_dict(s), indent=2))
        return 0

    print()
    print(s.render())

    if not s.options:
        return 1

    chosen = host.find(s, args.provider) if args.provider else s.cheapest
    if chosen is None:
        known = ", ".join(sorted({o.provider.id for o in s.options}))
        raise ValueError(f"no option at provider {args.provider!r} fits this model. Fits: {known}")

    print("  PICK")
    print(chosen.render())
    for note in chosen.provider.notes:
        for line in _wrap(note, 72):
            print(f"    {line}")
    print()

    if not args.write:
        print(f"  clickllm host {m.id} --write ./deploy   # emit the deploy files")
        print(f"  clickllm host {m.id} --free-only        # only what costs nothing")
        print("  Nothing was written, nothing was started, no provider was contacted.\n")
        return 0

    art = host.artifact(chosen, context=_parse_size(args.context), concurrency=args.concurrency)
    written = art.write(args.write)
    print(f"  WROTE  ({len(written)} file(s) to {args.write})")
    for path in written:
        print(f"    {path}")
    print("\n  NEXT")
    for i, step in enumerate(art.how, 1):
        for j, line in enumerate(_wrap(step, 70)):
            print(f"    {f'{i}.' if j == 0 else '  ':<4}{line}")
    print(
        "\n  Run these yourself. clickllm never authenticates to a provider, and\n"
        "  these files bill your account, not ours.\n"
    )
    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """What proving N models left on disk, and — only when asked twice — less of it.

    Reads the engines' cache, never a cache of ours. Deletion needs both an
    explicit `evict` and `--yes`, and a pinned repo is never in the plan: an
    eval sweep that evicts the incumbent destroys the comparison it exists for.
    """
    from . import cache

    state = cache.load_state()
    budget = cache.parse_bytes(args.budget) if args.budget else state.budget_bytes

    if args.action in ("pin", "unpin"):
        if not args.repo:
            raise ValueError(f"{args.action} needs a repo, e.g. mlx-community/Llama-3.1-8B-4bit")
        st = cache.pin(args.repo, None) if args.action == "pin" else cache.unpin(args.repo, None)
        print(f"\n  pinned: {', '.join(st.pinned) or 'nothing'}\n")
        return 0

    root = cache.hub_dir()
    found = cache.entries(root)

    if args.action == "evict":
        if budget is None:
            raise ValueError("evict needs a budget, e.g. --budget 200G")
        plan = cache.plan_eviction(found, budget, state.pinned)
        print(plan.render())
        if not plan.evict:
            return 0
        result = cache.apply_eviction(plan, confirm=args.yes, root=root)
        print(result.render())
        return 0

    remembered = bool(args.budget) and budget != state.budget_bytes
    if remembered:
        cache.save_state(cache.State(pinned=state.pinned, budget_bytes=budget))
    u = cache.usage(root, budget_bytes=budget, pinned=state.pinned)
    print(u.render())
    if remembered:
        print(f"  budget remembered: {cache.human(budget or 0)} — evict defaults to it now\n")
    if u.entries:
        print("  clickllm cache pin <repo>              # never evict this one")
        print("  clickllm cache evict --budget 200G     # show what would go\n")
    return 0


def cmd_box(args: argparse.Namespace) -> int:
    """Assemble the OCI artifact of ADR-0005. Writes files; pushes nothing.

    The push commands are printed for you to run. clickllm holds no registry
    credential, reads no token, and contacts no registry — the same boundary
    `host` keeps.
    """
    from . import box
    from .prove.receipt import Receipt

    receipt = None
    if args.receipt:
        receipt = Receipt.from_json(pathlib.Path(args.receipt).read_text())

    b = box.build(
        args.model,
        quant=args.quant,
        context=_parse_size(args.context),
        concurrency=args.concurrency,
        receipt=receipt,
        ref=args.ref,
    )

    print()
    print(b.render())
    print()

    if not args.write:
        print(f"  clickllm box {b.model.id} --write ./{b.model.id}-box   # emit the artifact")
        print(f"  clickllm box {b.model.id} --receipt receipt.json       # attach the proof")
        print("  Nothing was written, nothing was pushed, no registry was contacted.\n")
        return 0

    written = b.write(args.write)
    print(f"  WROTE  ({len(written)} file(s) to {args.write})")
    for path in written:
        print(f"    {path}")
    print("\n  PUSH")
    for command in b.push_commands(args.write):
        print(f"    {command}")
    print(
        "\n  Run these yourself. clickllm holds no registry credential, reads no\n"
        "  token, and has contacted no registry.\n"
    )
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Notice new models without being asked. Stages; never publishes."""
    from . import watch

    if args.list:
        rows = watch.pending()
        if not rows:
            print("\n  nothing staged\n")
            return 0
        print(f"\n  {len(rows)} staged, waiting on a parameter count and a licence read:\n")
        for m in rows:
            print(f"  {m.get('id', '?'):<28}{m.get('layers', '?')} layers  {m.get('repo', '')}")
        print("\n  clickllm catalog-add <repo> --params-b <N> --network   # promote\n")
        return 0

    if args.install:
        fragment, where = watch.install_schedule(interval_hours=args.every_hours)
        print(f"\n  Save this as {where}\n")
        print(fragment)
        print(
            "  Not installed for you: a recurring job that reaches the network is\n"
            "  your decision, and a tool that scheduled itself would be doing the\n"
            "  opposite of what this project promises.\n"
        )
        return 0

    if not args.network:
        print(
            "\n  Discovery reads a public index, which needs network access.\n"
            "  Re-run with --network. Nothing is ever sent — only fetched.\n"
        )
        return 0

    from . import catalog_update as cu

    report = watch.run(cu.http_fetch, limit=args.limit, max_add=args.max_add)
    if args.json:
        print(report.to_json())
        return 0
    print()
    print(report.render())
    print()
    return 0


def cmd_catalog_sources(args: argparse.Namespace) -> int:
    """Where the catalogue's models came from."""
    models = catalog.load()
    print()
    for p in catalog.sources():
        mark = "built-in" if p == catalog.CATALOG_PATH else "drop-in "
        exists = "" if p.exists() else "  (missing)"
        print(f"  {mark}  {p}{exists}")
    print(f"\n  {len(models)} models · later files override earlier ones by id")
    print(f"  drop new ones into {_drop_in_dir()}/ — no reinstall\n")
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


def cmd_advise(args: argparse.Namespace) -> int:
    """What to change about a deployment, unprompted — and what production says.

    Prints proposals with their evidence. Applies nothing: the whole point is
    that a suggestion you cannot argue with is one you cannot trust.
    """
    from .advise import Observed, reconcile, suggest
    from .plan import Requirements, Workload, plan

    hw = hardware.detect()
    req = Requirements(
        workload=Workload(args.workload),
        concurrency=args.concurrency,
        context=_parse_size(args.context),
        ttft_ms=args.ttft_ms,
        itl_ms=args.itl_ms,
        prefix_sharing=args.prefix_sharing,
        structured_output=args.structured_output,
    )
    p = plan(hw, req)

    seen = Observed(
        concurrency=args.seen_concurrency,
        prefix_sharing=args.seen_prefix_sharing,
        ttft_ms=args.seen_ttft_ms,
        peak_context=_parse_size(args.seen_peak_context) if args.seen_peak_context else None,
        kv_utilisation=args.seen_kv_utilisation,
    )
    drift = reconcile(req, p, seen)
    ideas = suggest(req, p)

    if args.json:
        print(
            json.dumps(
                {
                    "engine": p.engine.value,
                    "suggestions": [s.__dict__ | {"impact": s.impact.value} for s in ideas],
                    "drift": [s.__dict__ | {"impact": s.impact.value} for s in drift],
                },
                default=str,
                indent=2,
            )
        )
        return 0

    print(f"\n  {hw.name} · {p.engine.value}\n")
    if drift:
        print("  PRODUCTION DIVERGED FROM THE PLAN\n")
        for s in drift:
            print(f"  {s.render()}\n")
    if ideas:
        print("  WORTH CHANGING\n")
        for s in ideas:
            print(f"  {s.render()}\n")
    if not drift and not ideas:
        print("  Nothing to suggest — the plan matches the requirements as stated.\n")

    # Nonzero only when production has diverged, so this is usable as a probe.
    return 1 if drift else 0


def _collect_replies(
    items: list[EvalItem], args: argparse.Namespace
) -> tuple[list[EvalItem], list[Collection]]:
    """Fill in whichever side has a live endpoint, and say what did not come back.

    The two sides are collected in sequence rather than independently, so the
    second one is only asked about items the first one answered — the funnel is
    visible in the report and nothing is spent fetching a reply for an item that
    is already unscoreable.
    """
    from .prove.collect import collect

    sides = (
        ("candidate", args.candidate_endpoint, args.candidate_model or args.candidate),
        ("baseline", args.incumbent_endpoint, args.incumbent_model or args.incumbent),
    )
    collections = []
    for side, endpoint, model_id in sides:
        if not endpoint:
            continue
        if not items:
            break
        got = collect(
            items,
            base=endpoint,
            model=model_id,
            side=side,
            workers=args.collect_workers,
            timeout=args.collect_timeout,
        )
        collections.append(got)
        items = list(got.items)
    return items, collections


def cmd_prove(args: argparse.Namespace) -> int:
    """Run the eval suite over an eval set and print the verdict.

    The kiosk over `clickllm.prove`. Deliberately has no flag that moves traffic:
    it prints a proposal, and a human runs the thing that acts on it. That holds
    with `--candidate-endpoint` too — collecting from a live endpoint gathers
    replies, it does not put anything in front of users.
    """
    from datetime import date

    from .mcp import SERVER_INFO
    from .prove import EvalItem, suite

    raw = json.loads(pathlib.Path(args.evalset).read_text())
    if isinstance(raw, dict):
        rows, shares, names = raw.get("items", []), raw.get("shares", {}), raw.get("names", {})
    else:  # a bare list of items — shares fall back to equal weight
        rows, shares, names = raw, {}, {}

    if not rows:
        raise ValueError(f"{args.evalset} contains no eval items")

    items = [
        EvalItem(
            item_id=str(r.get("item_id", i)),
            cluster=str(r.get("cluster", "all")),
            prompt=str(r.get("prompt", "")),
            baseline=str(r.get("baseline", "")),
            candidate=str(r.get("candidate", "")),
            baseline_tool_calls=tuple(r.get("baseline_tool_calls", ()) or ()),
            candidate_tool_calls=tuple(r.get("candidate_tool_calls", ()) or ()),
            response_format=r.get("response_format"),
        )
        for i, r in enumerate(rows)
    ]

    asked = len(items)
    items, collections = _collect_replies(items, args)
    if collections and not items:
        raise ValueError(
            f"no item in {args.evalset} got a reply — nothing can be scored. "
            f"{collections[0].failures[0].reason}"
        )

    # No shares given means every cluster weighs the same. Said out loud, because
    # an unweighted verdict on unevenly-distributed traffic is a different claim.
    if not shares:
        keys = sorted({i.cluster for i in items})
        shares = {k: 1 / len(keys) for k in keys}
        print(f"\n  no traffic shares supplied — weighting {len(keys)} clusters equally")

    result = suite(
        items,
        shares=shares,
        names=names,
        issued=args.issued or date.today().isoformat(),
        candidate=args.candidate,
        incumbent=args.incumbent,
        bar=args.bar,
        # What the eval set was drawn from, not what survived collection. The
        # receipt then states 400 alongside claims totalling 388, rather than
        # quietly presenting the smaller number as the whole thing.
        traffic_captures=asked,
        tool_version=SERVER_INFO["version"],
    )

    # Collection failures are reported apart from eval failures, and never in
    # place of them: "12 items never answered" and "12 items answered badly" are
    # different facts, and only the second one is about the model. In `--json`
    # this goes to stderr so the receipt stays pipeable and the caveat still
    # reaches a human.
    if collections:
        out = sys.stderr if args.json else sys.stdout
        print("\n  COLLECTED", file=out)
        for c in collections:
            print(c.render(), file=out)
        if len(items) < asked:
            print(
                f"    scoring {len(items)} of {asked} items — the intervals below are "
                f"over what answered, and the eval-set digest is over those items too",
                file=out,
            )

    if args.json:
        print(result.receipt.to_json())
        return 0

    print()
    print(result.render())
    print()
    if args.out:
        pathlib.Path(args.out).write_text(result.receipt.to_json())
        print(f"  receipt written to {args.out}\n")

    # Nonzero when nothing can move, so this is usable as a CI gate.
    return 0 if result.policy.moved_share > 0 else 1


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


def cmd_desktop(args: argparse.Namespace) -> int:
    """Install a double-clickable launcher for the workbench."""
    from . import desktop

    lz = desktop.install(port=args.port)
    print(f"\n  installed {lz.path}")
    print(f"  uninstall with:  {lz.uninstall}\n")
    print("  It starts `clickllm ui` and opens it. Nothing is bundled and")
    print("  nothing auto-updates — this only adds the icon.\n")
    return 0


def _plugin_kinds() -> list[str]:
    """vLLM entry-point group names, imported lazily to keep `fit` stdlib-only."""
    from .kernels import PluginKind

    return [k.value for k in PluginKind]


def cmd_kernel(args: argparse.Namespace) -> int:
    """Scaffold a vLLM plugin package, and the plan for proving it."""
    from .kernels import KernelClaim, Plugin, PluginKind, scaffold

    plugin = Plugin(
        name=args.name,
        kind=PluginKind(args.kind),
        target=f"{args.name.replace('-', '_')}:register",
    )
    claim = KernelClaim(
        name=args.name,
        claimed_speedup=args.speedup,
        bit_identical=args.bit_identical,
        expected_drift=args.drift or "",
    )
    out = pathlib.Path(args.out or args.name)
    if out.exists() and any(out.iterdir()):
        print(f"error: {out} exists and is not empty", file=sys.stderr)
        return 2
    for rel, body in scaffold(plugin, claim).items():
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body)
        print(f"  wrote {dest}")
    print(f"\n  Loading is the easy half. {out}/PROVING.md is the other one.\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="clickllm",
        description="Run open models properly on your own hardware — and prove they hold.",
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

    ca = sub.add_parser("catalog-add", help="add a model from its published config, no reinstall")
    ca.add_argument("repo", help="Hugging Face repo, e.g. meta-llama/Llama-3.1-8B")
    ca.add_argument(
        "--params-b",
        type=float,
        required=True,
        dest="params_b",
        help="total parameters in billions — not in config.json, so you supply it",
    )
    ca.add_argument(
        "--active-b",
        type=float,
        dest="active_b",
        help="active parameters for MoE; defaults to total (dense)",
    )
    ca.add_argument("--id", help="catalogue id; defaults to the repo name, lowercased")
    ca.add_argument("--name", help="display name; defaults to the repo name")
    ca.add_argument("--license", default="unknown")
    ca.add_argument(
        "--license-ok",
        action="store_true",
        dest="license_ok",
        help="mark the licence as cleanly commercial (read it first)",
    )
    ca.add_argument("--quants", default="q4,q8", help="comma-separated, default q4,q8")
    ca.add_argument("--out", help="write here instead of the drop-in directory")
    ca.add_argument("--network", action="store_true", help="allow network access (required)")
    ca.set_defaults(fn=cmd_catalog_add)

    cs = sub.add_parser("catalog-sources", help="where the catalogue's models come from")
    cs.set_defaults(fn=cmd_catalog_sources)

    b = sub.add_parser("build", help="say what you're building; get a deployment and a proof plan")
    b.add_argument("description", nargs="*", help='e.g. "coding assistant for 20 engineers"')
    b.add_argument("--on", help="hardware profile id; omit to detect this machine")
    b.add_argument("--concurrency", type=int)
    b.add_argument("--context")
    b.add_argument("--ttft-ms", type=int, dest="ttft_ms")
    b.add_argument("--itl-ms", type=int, dest="itl_ms")
    b.add_argument("--prefix-sharing", type=float, dest="prefix_sharing")
    b.add_argument("--save", help="write the session state here")
    b.add_argument("--resume", help="continue a saved session")
    b.add_argument("--json", action="store_true")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="serve a model here: resolve weights, launch, wait for healthy")
    r.add_argument("model", help="catalogue model id, e.g. qwen3-30b-a3b")
    r.add_argument("--quant", help="force a quantisation; default is the best that fits")
    r.add_argument("--context", default="32k", help="context length, e.g. 8k, 32k")
    r.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="simultaneous requests to tune for (a server is not a single-stream benchmark)",
    )
    r.add_argument("--host", default="127.0.0.1", help="bind address (loopback by default)")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="print the plan and the exact command; start nothing",
    )
    r.set_defaults(fn=cmd_run)

    ho = sub.add_parser("host", help="where to run a model this machine cannot; free tiers first")
    ho.add_argument("model", help="catalogue model id, e.g. deepseek-v3")
    ho.add_argument("--context", default="32k", help="context length, e.g. 8k, 32k")
    ho.add_argument("--concurrency", type=int, default=4, help="simultaneous requests")
    ho.add_argument(
        "--quant", help="force a quantisation; default is the best that fits each shape"
    )
    ho.add_argument(
        "--free-only",
        action="store_true",
        dest="free_only",
        help="only shapes that cost nothing",
    )
    ho.add_argument("--provider", help="pick this provider instead of the cheapest that fits")
    ho.add_argument("--write", metavar="DIR", help="emit the deploy artifact here")
    ho.add_argument("--json", action="store_true")
    ho.set_defaults(fn=cmd_host)

    cc = sub.add_parser("cache", help="what the engines downloaded, and what to evict")
    cc.add_argument(
        "action",
        nargs="?",
        choices=["evict", "pin", "unpin"],
        help="omit to report; evict needs --yes before anything is deleted",
    )
    cc.add_argument("repo", nargs="?", help="repo to pin or unpin, e.g. org/Model-4bit")
    cc.add_argument("--budget", help="cache budget, e.g. 200G; remembered once set")
    cc.add_argument("--yes", action="store_true", help="actually delete (evict prints otherwise)")
    cc.set_defaults(fn=cmd_cache)

    bx = sub.add_parser("box", help="build the OCI box: one artifact, every platform")
    bx.add_argument("model", help="catalogue model id, e.g. qwen3-30b-a3b")
    bx.add_argument("--write", metavar="DIR", help="emit the box here")
    bx.add_argument("--context", default="32k", help="context length to tune for, e.g. 8k, 32k")
    bx.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="simultaneous requests to tune for (a box is published for a fleet)",
    )
    bx.add_argument("--quant", help="force a quantisation; default is the best that fits each host")
    bx.add_argument("--receipt", help="attach this proof; without it the box claims nothing")
    bx.add_argument("--ref", help="registry reference the printed push commands use")
    bx.set_defaults(fn=cmd_box)

    w = sub.add_parser("watch", help="discover new models on a schedule; stages, never publishes")
    w.add_argument("--list", action="store_true", help="show what is staged and waiting")
    w.add_argument("--install", action="store_true", help="print the scheduler fragment")
    w.add_argument("--every-hours", type=int, default=24, dest="every_hours")
    w.add_argument("--limit", type=int, default=40, help="how many index entries to consider")
    w.add_argument("--max-add", type=int, default=5, dest="max_add", help="cap per run")
    w.add_argument("--network", action="store_true", help="allow network access (required)")
    w.add_argument("--json", action="store_true")
    w.set_defaults(fn=cmd_watch)

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

    ad = sub.add_parser("advise", help="what to change, and where production diverged")
    ad.add_argument("--context", default="32k")
    ad.add_argument("--concurrency", type=int, default=1)
    ad.add_argument(
        "--workload", default="interactive", choices=["interactive", "realtime", "batch"]
    )
    ad.add_argument("--ttft-ms", type=int, dest="ttft_ms")
    ad.add_argument("--itl-ms", type=int, dest="itl_ms")
    ad.add_argument("--prefix-sharing", type=float, default=0.0, dest="prefix_sharing")
    ad.add_argument("--structured-output", action="store_true", dest="structured_output")
    # Observed reality — any subset. Telemetry arrives in pieces.
    ad.add_argument("--seen-concurrency", type=int, dest="seen_concurrency")
    ad.add_argument("--seen-prefix-sharing", type=float, dest="seen_prefix_sharing")
    ad.add_argument("--seen-ttft-ms", type=int, dest="seen_ttft_ms")
    ad.add_argument("--seen-peak-context", dest="seen_peak_context")
    ad.add_argument("--seen-kv-utilisation", type=float, dest="seen_kv_utilisation")
    ad.add_argument("--json", action="store_true")
    ad.set_defaults(fn=cmd_advise)

    pv = sub.add_parser("prove", help="run the eval suite and print the verdict")
    pv.add_argument("evalset", help="path to an eval-set JSON file")
    pv.add_argument("--candidate", default="candidate", help="the open model under test")
    pv.add_argument("--incumbent", default="incumbent", help="the model being replaced")
    pv.add_argument("--issued", default="", help="ISO date stamped on the receipt")
    pv.add_argument(
        "--bar",
        type=float,
        default=0.90,
        help="equivalence bar; a cluster moves only when the whole interval clears it",
    )
    pv.add_argument("--out", help="write the receipt to this path")
    pv.add_argument("--json", action="store_true", help="print the receipt as JSON")
    # Live collection. Without these the eval set must already carry both
    # answers, which is the original behaviour and still the default.
    pv.add_argument(
        "--candidate-endpoint",
        dest="candidate_endpoint",
        help="OpenAI-compatible base URL to collect the candidate's replies from, "
        "e.g. http://127.0.0.1:8000/v1 (from `clickllm run`)",
    )
    pv.add_argument(
        "--candidate-model",
        dest="candidate_model",
        help="model id to ask that endpoint for; defaults to --candidate",
    )
    pv.add_argument(
        "--incumbent-endpoint",
        dest="incumbent_endpoint",
        help="same, for the model being replaced",
    )
    pv.add_argument(
        "--incumbent-model",
        dest="incumbent_model",
        help="model id to ask that endpoint for; defaults to --incumbent",
    )
    pv.add_argument(
        "--collect-workers",
        dest="collect_workers",
        type=int,
        default=8,
        help="parallel requests while collecting",
    )
    pv.add_argument(
        "--collect-timeout",
        dest="collect_timeout",
        type=float,
        default=120.0,
        help="seconds to wait for one reply",
    )
    pv.set_defaults(fn=cmd_prove)

    rc = sub.add_parser("receipt", help="render or verify a migration receipt")
    rc.add_argument("receipt", help="path to a receipt JSON file")
    rc.add_argument("--against", help="a second receipt to verify this one against")
    rc.set_defaults(fn=cmd_receipt)

    d = sub.add_parser("desktop", help="install a double-clickable launcher")
    d.add_argument("--port", type=int, default=7171)
    d.set_defaults(fn=cmd_desktop)

    kn = sub.add_parser("kernel", help="scaffold a vLLM plugin and its proof plan")
    kn.add_argument("name", help="plugin name, e.g. fused-rmsnorm")
    kn.add_argument(
        "--kind",
        default="vllm.general_plugins",
        choices=_plugin_kinds(),
        help="vLLM entry-point group",
    )
    kn.add_argument("--speedup", type=float, default=1.0, help="the speedup you claim")
    kn.add_argument("--bit-identical", action="store_true", help="claims byte-for-byte output")
    kn.add_argument("--drift", help="what you expect to change, if not bit-identical")
    kn.add_argument("--out", help="output directory (default: the plugin name)")
    kn.set_defaults(fn=cmd_kernel)

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
