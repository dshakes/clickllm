"""clickllm CLI. Stdlib only — `uv run clickllm fit` must work with zero install."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from . import catalog, engine, fit, hardware

if TYPE_CHECKING:  # imported lazily at the call sites — `clickllm fit` must stay cheap
    from .prove.collect import Collection
    from .prove.graders import EvalItem
    from .prove.judge import Agreement, JudgeFn

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


#: What every throughput figure in this project actually is. Stated once and
#: emitted beside the number on every machine-readable surface, because a
#: disclosure that each surface re-decides is a disclosure one of them will
#: eventually omit — which is exactly what happened here.
ROOFLINE_BASIS = "memory-bandwidth roofline, not measured"


def cmd_fit(args: argparse.Namespace) -> int:
    hw = hardware.detect()
    ctx, conc = _parse_size(args.context), args.concurrency

    if args.explain:
        m = catalog.get(args.explain)
        f = fit.best_quant(m, hw, ctx, conc) or fit.solve(
            m, min(m.quants, key=lambda q: catalog.QUANT_BITS[q]), hw, ctx, conc
        )
        # `--explain` returned before the `--json` branch below, so
        # `fit --explain X --json` printed the human block and ignored the flag
        # entirely — a flag that silently does nothing is worse than one that
        # errors, because a script consuming it parses prose as JSON.
        if args.json:
            print(
                json.dumps(
                    {
                        "model": m.id,
                        "quant": f.quant,
                        "context": ctx,
                        "concurrency": conc,
                        "weight_bytes": f.weight_bytes,
                        "kv_bytes": f.kv_bytes,
                        "overhead_bytes": f.overhead_bytes,
                        "total_bytes": f.total_bytes,
                        "usable_bytes": f.usable_bytes,
                        "headroom_bytes": f.headroom_bytes,
                        "tokens_per_sec": f.tokens_per_sec,
                        # Invariant 6, and the reason this is here rather than
                        # only in `--explain`: a human sees the `~` in the table
                        # and the pointer to `--explain`; a program reading this
                        # JSON sees a bare integer. `mcp`, `sdk` and `ui` all
                        # carried this basis and only the CLI's machine-readable
                        # surface — the one most likely to be piped into someone
                        # else's capacity planning — did not.
                        "estimate_basis": ROOFLINE_BASIS,
                        "feasible": f.feasible,
                        "explain": f.explain(),
                    },
                    indent=2,
                )
            )
            return 0 if f.feasible else 1
        # Unconditional. `--quiet` is documented as "hide the NOT FEASIBLE
        # section", and `--explain` has no such section — it is one model. The
        # gate here meant `fit --explain X --quiet` printed *nothing* and exited
        # 0, which reads as "no answer" rather than as a suppressed list.
        print(f"\n{f.explain()}\n")
        print(f"  verdict: {'FITS' if f.feasible else 'DOES NOT FIT'}\n")
        return 0 if f.feasible else 1

    # Through the engine — ADR-0016. This called `fit.rank()` and assembled its
    # own shape, which is how the CLI's JSON came to omit the roofline
    # disclosure every other surface carried. The wire format below is
    # unchanged; only the source of the numbers moved.
    report = engine.fit(context=ctx, concurrency=conc, hw=hw)
    feasible, rejected = report.feasible, report.rejected

    if args.json:
        print(
            json.dumps(
                {
                    "hardware": hw.to_dict(),
                    "context": ctx,
                    "concurrency": conc,
                    "feasible": [
                        {
                            "id": c.model_id,
                            "quant": c.quant,
                            "verified": c.architecture_verified,
                            "weights_gb": round(c.weights_gb, 1),
                            "kv_gb": round(c.kv_gb, 1),
                            "total_gb": round(c.total_gb, 1),
                            "headroom_gb": round(c.headroom_gb, 1),
                            # This surface has always rounded to a whole
                            # number here, and rounding is formatting.
                            "tokens_per_sec": (
                                round(c.tokens_per_sec_estimate)
                                if c.tokens_per_sec_estimate
                                else None
                            ),
                            # See the note on the other emission site above.
                            "estimate_basis": ROOFLINE_BASIS,
                            "license": c.license,
                            "license_ok": c.license_clean_commercial,
                        }
                        for c in feasible
                    ],
                    "rejected": [{"id": r.model_id, "reason": r.reason} for r in rejected],
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
        for c in feasible:
            flag = " ?" if not c.architecture_verified else ""
            tps = f"{c.tokens_per_sec_estimate:.0f}" if c.tokens_per_sec_estimate else "-"
            slow = " slow" if c.slow else ""
            print(
                f"  {c.name[:24] + flag:<26}{c.quant:<7}"
                f"{c.weights_gb:>7.1f}G{c.kv_gb:>7.1f}G"
                f"{c.total_gb:>7.1f}G{c.headroom_gb:>7.1f}G"
                f"{tps:>8}{slow}  {c.license} {'OK' if c.license_clean_commercial else '!'}"
            )

    if rejected and not args.quiet:
        print("\n  NOT FEASIBLE\n")
        for r in rejected:
            print(f"  {r.name[:24]:<26}{r.reason}")

    name, why = fit.recommend_runtime(hw, ctx, conc)
    print(f"\n  runtime -> {name}")
    for line in _wrap(why, 76):
        print(f"             {line}")

    unverified = [c for c in feasible if not c.architecture_verified]
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
                            # How optimistic the two figures above are for an
                            # MoE at this batch. Absent for dense models and at
                            # batch 1, where they carry no routing risk.
                            "moe_batch_optimism": (
                                round(p.moe_batch_optimism, 1) if p.moe_batch_optimism else None
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

    # "saturated single-stream" was true of an older `cost_per_mtok_usd` that
    # divided by one stream's output. It uses aggregate throughput now, and its
    # own docstring records that "real cost is higher" was wrong by more in the
    # other direction. A stale caveat is worse than none: it tells the reader to
    # correct in a direction that no longer applies.
    notes = [
        "\n  $/Mtok assumes the machine is busy at the requested concurrency;",
        "  a box at 10% utilisation costs about ten times this per token.",
        "  Throughput figures are roofline estimates, not measurements.",
    ]
    spread = max((p.moe_batch_optimism or 1.0) for p in placements) if placements else 1.0
    if spread > 1.0:
        notes.append(
            f"  MoE: the weight read is held at the active-parameter count, exact only at"
            f" batch 1. Throughput and $/Mtok are the optimistic end of a band up to"
            f" {spread:.1f}x wide — see `clickllm fit --explain`."
        )
    print("\n".join(notes) + "\n")
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
    # `Session.turn()` — one step per external turn, which is where a
    # worth-asking question gets committed to `s.asked`. Committing on an
    # intermediate call whose Turn is then discarded is how the single most
    # valuable question got silently lost before a user ever saw it. That rule
    # used to be written out here and again in `mcp._build`; it now belongs to
    # the method, so a third caller cannot get it subtly wrong.
    fields = {}
    for name in ("concurrency", "context", "ttft_ms", "itl_ms", "prefix_sharing"):
        v = getattr(args, name, None)
        if v is not None:
            fields[name] = _parse_size(v) if name == "context" else v

    turn = s.turn(
        description=" ".join(args.description) if args.description else "",
        machine=args.on or None,
        # Only detect when nothing is known yet: a bare `--resume` must not
        # silently re-detect the local machine and discard a saved `--on`
        # profile (e.g. a remote "h100").
        detect_hardware=s.hw is None,
        **fields,
    )
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
        engine=args.engine,
        tag=args.tag,
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
    print("  ctrl-c to stop\n", flush=True)
    return _serve_until_signalled(ep)


def _serve_until_signalled(ep) -> int:
    """Block until the engine exits or we are told to stop — then actually stop it.

    `stop()` was correct and nothing called it. This function exited straight
    into `ep.wait()`, so a signal killed clickllm and left the engine serving,
    holding the GPU, with the port still open. Measured against a real run:
    clickllm gone, `mlx_lm` alive, endpoint still answering.

    Interactive Ctrl-C hid it for as long as it existed. A tty sends SIGINT to
    the whole foreground process group, so the engine received its own signal
    from the terminal rather than from us — and `serve()` deliberately keeps the
    child in that group for exactly this reason. Every other way of stopping a
    process reaches clickllm alone: `kill`, systemd, `docker stop`, a Kubernetes
    pod eviction. Those are also the ways it gets stopped when nobody is
    watching, which is when an engine still holding a GPU costs the most.

    Signal policy lives here rather than in `launch`, because a library that
    installs process-wide handlers surprises whoever imports it.
    """
    import signal as _signal

    stopping = False

    def teardown(signum, _frame):
        nonlocal stopping
        if stopping:  # a second Ctrl-C: let the default handler have it
            raise KeyboardInterrupt
        stopping = True
        name = _signal.Signals(signum).name
        print(f"\n  {name} — stopping {ep.model}...", flush=True)
        survivors = ep.stop()
        if survivors:
            pids = ", ".join(str(x) for x in survivors)
            print(f"  WARNING  {len(survivors)} process(es) would not die: {pids}")
            print("           they may still hold the device.", flush=True)
        else:
            print("  stopped.", flush=True)

    previous = {}
    for sig in (_signal.SIGINT, _signal.SIGTERM):
        # Not the main thread → signal() raises; the handler is a nicety, not a
        # requirement, and failing to install it must not stop a server running.
        with contextlib.suppress(ValueError, OSError):
            previous[sig] = _signal.signal(sig, teardown)
    try:
        return ep.wait()
    except KeyboardInterrupt:
        ep.stop()
        return 130
    finally:
        for sig, handler in previous.items():
            with contextlib.suppress(ValueError, OSError):
                _signal.signal(sig, handler)


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


def cmd_observe(args: argparse.Namespace) -> int:
    """Put the gateway in front of a provider and record what goes through it.

    This is the one command that enters the request path, so it says so, names
    where the prompts will land, and never moves traffic — see ADR-0015 and the
    gateway's own refusal of a startup `--percent`.
    """
    from . import observe

    try:
        binary = observe.find_gateway()
    except FileNotFoundError as exc:
        print(f"\n  {exc}\n")
        return 2
    if binary is None:
        print(
            "\n  No gateway binary found.\n\n"
            "  Install it:    pip install clickllm-gateway\n"
            "  From source:   cargo build --release -p clickllm-gateway\n"
            "  Or point at one: CLICKLLM_GATEWAY_BIN=/path/to/clickllm-gateway\n\n"
            "  `clickllm fit` needs none of this — the gateway is only for capture.\n"
        )
        return 2

    home = observe.state_dir()
    capture = pathlib.Path(args.capture) if args.capture else home / "captures.log"
    argv = observe.gateway_argv(
        binary,
        args.upstream,
        port=args.port,
        candidate=args.candidate,
        capture=capture,
        key=home / "capture.key",
        no_capture=args.no_capture,
    )
    # One line, and only what the binary does not already print — it announces
    # the capture path, the upstream and the bound address itself, and saying
    # them twice reads as two different things happening.
    #
    # Flushed explicitly: Python's stdout is block-buffered when it is not a
    # terminal, and `run_gateway` hands the same fd to a child that writes
    # immediately. Without this the banner arrives after the gateway's, or —
    # when the process is killed rather than exiting — not at all, which is
    # exactly how it behaved the first time it was piped to a file.
    print("\n  In your request path from now until Ctrl-C, and not after (ADR-0015).", flush=True)
    return observe.run_gateway(argv)


def cmd_distill(args: argparse.Namespace) -> int:
    """Turn a capture log into the eval set `clickllm prove` reads."""
    from . import core, observe

    if not core.available():
        print(f"\n  {core.why_unavailable()}\n")
        return 2

    home = observe.state_dir()
    log = pathlib.Path(args.capture) if args.capture else home / "captures.log"
    key_path = pathlib.Path(args.key) if args.key else home / "capture.key"
    if not log.exists():
        print(f"\n  No capture log at {log}. Run `clickllm observe` first.\n")
        return 2
    if not key_path.exists():
        # Never `load_or_create_key` here: a fresh key would decrypt nothing and
        # report an empty log, which reads as "no traffic" rather than "wrong
        # key" — the most misleading answer available.
        print(f"\n  No key at {key_path}. It is written when the gateway first records.\n")
        return 2

    rows = core.read_captures(str(log), key_path.read_bytes())
    if not rows:
        print(f"\n  {log} holds no captures yet.\n")
        return 2

    doc, report = observe.distill(rows, budget=args.budget, min_per_cluster=args.min_per_cluster)
    out = observe.write_eval_set(doc, pathlib.Path(args.out))
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0
    print()
    print(report.render())
    print(f"\n  wrote {out}")
    print(
        f"  next: clickllm prove {out} "
        "--candidate-endpoint <url> --candidate <model>\n"
        "        (the endpoint is required — without it every candidate answer "
        "in this file is blank)\n"
    )
    return 0


def cmd_migrate(args: argparse.Namespace) -> int:
    """Walk the chain, record what the evidence permits, and stop at the gate.

    The only command that runs the whole loop, and the one most worth being
    clear about what it does not do: it never moves traffic. `gate.decide`
    returns `ADVANCE` as a proposal, this prints it, and a human applies it
    through the gateway's control surface — which records a reason and refuses
    an unconfirmed increase (invariant 8).
    """
    from datetime import date

    from . import core, migrate, observe

    path = migrate.state_path(args.state)

    if args.status:
        if not path.exists():
            print(f"\n  No migration in progress ({path} does not exist).\n")
            return 2
        st = migrate.load_state(path)
        print(f"\n  migration to {st.candidate}, started {st.started}")
        print(f"  currently {st.stage().render()}  ·  {len(st.runs)} runs recorded\n")
        for r in st.runs[-5:]:
            print(f"  {r.when}  {r.action:<9} {r.items:>4} items  {r.reason[:60]}")
        print()
        return 0

    if args.install:
        fragment, where = migrate.install_schedule(
            interval_hours=args.every_hours, state=args.state or ""
        )
        print(f"\n  Save this in {where}\n")
        print(f"  {fragment}")
        print(
            "  Not installed for you: a recurring job that reads your captured\n"
            "  traffic is your decision.\n"
        )
        return 0

    if not args.candidate:
        raise ValueError("--candidate is required: name the model being evaluated")
    if not core.available():
        print(f"\n  {core.why_unavailable()}\n")
        return 2

    home = observe.state_dir()
    log = pathlib.Path(args.capture) if args.capture else home / "captures.log"
    key = pathlib.Path(args.key) if args.key else home / "capture.key"
    if not log.exists() or not key.exists():
        print(f"\n  No captures at {log}. Run `clickllm observe` first.\n")
        return 2

    rows = core.read_captures(str(log), key.read_bytes())
    st = migrate.load_state(path, args.candidate)

    def prove_it(doc: dict) -> object:
        from .prove import EvalItem, suite

        items = [
            EvalItem(
                item_id=str(r["item_id"]),
                cluster=str(r["cluster"]),
                prompt=str(r["prompt"]),
                baseline=str(r["baseline"]),
                candidate=str(r.get("candidate") or ""),
                baseline_tool_calls=tuple(r.get("baseline_tool_calls") or ()),
                response_format=r.get("response_format"),
            )
            for r in doc["items"]
        ]
        _refuse_ungraded(items, endpoint=args.candidate_endpoint, source="the distilled eval set")
        items, _ = _collect_replies(items, args)
        if not items:
            raise ValueError(
                "no candidate replies were collected, so there is nothing to "
                "grade. Check --candidate-endpoint."
            )
        return suite(
            items,
            shares=doc["shares"],
            names=doc["names"],
            candidate=args.candidate,
            incumbent=args.incumbent,
            issued=date.today().isoformat(),
            bar=args.bar,
        )

    st, run, decision = migrate.step(
        st, rows=rows, prove=prove_it, budget=args.budget, min_per_cluster=args.min_per_cluster
    )
    migrate.save_state(st, path)
    print(migrate.render(st, run, decision))
    print(f"  state: {path}\n")
    # Zero when the loop ran, whatever it decided. A scheduled job that exits
    # nonzero because the answer was "hold" would page someone nightly for the
    # normal case.
    return 0


#: Mirrored from `clickllm.measure` so building the parser does not import it.
#: A drift here is caught by a test rather than by a user getting a different
#: default from `--help` than the module applies.
M_DEFAULT_SAMPLES = 5
M_DEFAULT_MAX_TOKENS = 128


def cmd_measure(args: argparse.Namespace) -> int:
    """Measure decode throughput against a running endpoint, or refuse to.

    The one command that can turn a roofline estimate into an observation, and
    the one most able to make this product *less* trustworthy if it is careless:
    a measured number carries more authority than an estimate and outlives the
    conditions that produced it. See issue #80 and `clickllm.measure`.
    """
    from . import measure as M

    hw = hardware.detect()
    roofline = None
    if args.model:
        try:
            spec = catalog.get(args.model)
        except KeyError:
            spec = None
        if spec is not None:
            fit_result = fit.best_quant(spec, hw, _parse_size(args.context), args.concurrency)
            if fit_result is not None:
                # The like-for-like comparison: single-stream decode against a
                # single-stream roofline. `aggregate_tokens_per_sec` is the batch
                # figure and would flatter the measurement at concurrency > 1.
                roofline = fit_result.tokens_per_sec

    result = M.measure(
        args.endpoint,
        args.served_model or args.model,
        cores=hw.cores,
        samples=args.samples,
        max_tokens=args.max_tokens,
        roofline=roofline,
        api_key=os.environ.get("CLICKLLM_API_KEY", ""),
    )

    if args.json:
        print(result.to_json())
    else:
        print(result.render())
        if roofline is None and args.model:
            print(
                f"  No roofline to compare against: {args.model!r} is not in the\n"
                "  catalogue, or nothing fits on this machine at that context.\n"
            )

    if args.out:
        from .atomicio import atomic_write

        atomic_write(pathlib.Path(args.out), result.to_json())
        # In `--json` this goes to stderr so stdout stays parseable — the
        # status line would otherwise trail the JSON object and break a
        # script piping stdout straight into a JSON parser.
        print(f"  wrote {args.out}\n", file=sys.stderr if args.json else sys.stdout)

    # Nonzero when the samples are not usable as a measurement, so a script that
    # wanted a number knows it did not get one. This is not an error — "the
    # machine was too busy" is a legitimate outcome, and the report says so.
    return 0 if result.usable else 1


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
    #
    # `--fail-on any` exists because the default is a *judgement*, not a fact:
    # an aged proof and a newly released model both leave `valid` true, and
    # whether that should stop a deploy is the deploying team's policy rather
    # than ours. Making them pick it is better than picking for them and being
    # quietly wrong for half of them — and the strict mode is the one a release
    # gate usually wants, since "nothing observed has changed, but nobody has
    # checked in eleven months" is not a thing to deploy on silently.
    if args.fail_on == "any" and proposal.findings:
        if not args.json:
            if proposal.valid:
                print(
                    f"  --fail-on any: {len(proposal.findings)} finding(s), none of which "
                    "voids the receipt on its own.\n"
                )
            else:
                print(f"  --fail-on any: failing on {len(proposal.findings)} finding(s).\n")
        return 1
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


def _refuse_ungraded(items: list, *, endpoint: str | None, source: str) -> None:
    """Refuse to grade a set where nothing was ever collected.

    Every candidate answer empty means every grader fails, and the verdict comes
    back 0% — "keep the incumbent", stated with exactly the confidence of a real
    result. It fails closed, so nobody ships a bad model on it; they abandon a
    good one instead, which is the same defect pointing the other way.

    `clickllm distill` writes precisely this file, deliberately: the candidate
    column is for `--candidate-endpoint` to fill. So the refusal belongs at the
    solver rather than in the hint distill prints (ADR-0011) — and in *one*
    place, because it was written for `clickllm prove` and `clickllm migrate`
    then reached the same grader through its own prover without it, scoring a
    verdict over answers nobody gave and recording it as a real run.
    """
    if endpoint or any(i.candidate for i in items):
        return
    raise ValueError(
        f"{source}: every candidate answer is blank and no --candidate-endpoint "
        "was given, so there is nothing to grade. Pass --candidate-endpoint <url> "
        "to collect the candidate's replies, or fill the `candidate` field in the "
        "file yourself. Grading it as-is would report 0% and read as a verdict."
    )


def _add_collection_flags(parser: argparse.ArgumentParser) -> None:
    """Declare the flags `_collect_replies` reads.

    One definition, every parser that collects. `prove` had them and `migrate`
    did not, so the loop reached `_collect_replies` and died on
    `AttributeError: 'Namespace' object has no attribute 'collect_workers'` —
    a missing flag surfacing as a crash inside the collector rather than as a
    usage error at the boundary. Two copies would have fixed today and drifted
    by the next flag.
    """
    parser.add_argument(
        "--collect-workers",
        dest="collect_workers",
        type=int,
        default=8,
        help="parallel requests while collecting",
    )
    parser.add_argument(
        "--collect-timeout",
        dest="collect_timeout",
        type=float,
        default=120.0,
        help="seconds to wait for one reply",
    )


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
            on_progress=_progress_line(side, quiet=getattr(args, "json", False)),
        )
        # Leave the finished count on its own line, so the report that follows
        # does not land on top of a half-drawn progress line.
        if not getattr(args, "json", False):
            print(file=sys.stderr)
        collections.append(got)
        items = list(got.items)
    return items, collections


def _progress_line(side: str, *, quiet: bool) -> Callable[[object], None] | None:
    """A one-line progress indicator for a collection, or nothing.

    Collection is the longest thing this tool does — `collect`'s own docstring
    says a real eval set takes hours — and it printed nothing until it finished.
    Silence and a hang look identical, and the reasonable response to a tool
    that looks hung is to kill it, which on a paid endpoint discards every reply
    already bought.

    **stderr, not stdout.** `--json` output must stay a parseable document, and
    a progress line on stdout would sit inside it. `--json` suppresses this
    entirely anyway; writing to stderr means even that is belt and braces.
    """
    if quiet:
        return None

    def show(p: object) -> None:
        # `\r` and no newline: one line that updates in place rather than a
        # thousand lines of scrollback for a thousand-item eval set.
        print(f"\r  {side}: {p.render().strip()}   ", end="", file=sys.stderr, flush=True)

    return show


def _judge_from(args: argparse.Namespace) -> tuple[JudgeFn | None, str, Agreement | None]:
    """The judge, its name, and how far it is trusted — or nothing at all.

    Returns `(judge, judge_model, agreement)`. The agreement is *always* built
    when a judge is used, even with nothing sampled: `Agreement(0, 0, model)`
    renders as UNMEASURED and reads as untrustworthy, whereas passing `None`
    would make the receipt state "no judge was used" about a run that used one.

    Raises:
        ValueError: a judge endpoint with no model to disclose, a model with no
            endpoint (which would silently run without a judge), or an agreement
            that claims more agreements than samples.
    """
    from .prove.collect import endpoint_judge
    from .prove.judge import Agreement

    model = (args.judge_model or "").strip()
    if not args.judge_endpoint:
        if model:
            raise ValueError(
                "--judge-model was given without --judge-endpoint; the run would "
                "have used no judge at all while reporting one"
            )
        return None, "", None
    if not model:
        raise ValueError(
            "--judge-model is required with --judge-endpoint: an anonymous judge "
            "makes every score it touched unauditable"
        )
    if args.judge_agreed < 0 or args.judge_samples < 0:
        raise ValueError("judge agreement counts cannot be negative")
    if args.judge_agreed > args.judge_samples:
        raise ValueError(
            f"--judge-agreed {args.judge_agreed} exceeds --judge-samples {args.judge_samples}"
        )

    judge = endpoint_judge(
        base=args.judge_endpoint,
        model=model,
        timeout=args.collect_timeout,
    )
    return judge, model, Agreement(agreed=args.judge_agreed, total=args.judge_samples, model=model)


def _fingerprints_from(path: str | None) -> dict[str, str]:
    """Read a `{model: fingerprint}` file, or return nothing if none was given.

    Values are stringified rather than trusted: a JSON number here would be
    compared against a string later and always differ, reporting a model change
    that never happened — the one false alarm guaranteed to make someone stop
    believing the alert.
    """
    if not path:
        return {}
    raw = json.loads(pathlib.Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object mapping model name to fingerprint")
    return {str(k): str(v) for k, v in raw.items()}


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

    # Built before anything is collected: a misspelled judge endpoint should cost
    # nothing, not several hundred round trips followed by a usage error.
    judge, judge_model, agreement = _judge_from(args)

    # Parsed here too, for the same reason: a missing or malformed
    # --fingerprints file should fail before collection spends anything, not
    # after `_collect_replies()` has already paid for every round trip.
    fingerprints = _fingerprints_from(args.prove_fingerprints)

    # A file where nothing was ever collected is not a proof of anything, and
    # it does not read as one: every candidate answer is empty, every grader
    # fails it, and the verdict comes back 0% — "keep the incumbent", stated
    # with the same confidence as a real result. It fails closed, so nobody
    # ships a bad model on it; they abandon a good one instead, which is the
    # same defect pointing the other way.
    #
    # `clickllm distill` writes exactly this file, deliberately: the candidate
    # column is for `--candidate-endpoint` to fill. So the refusal belongs here,
    # at the solver, not only in the hint distill prints (ADR-0011).
    _refuse_ungraded(items, endpoint=args.candidate_endpoint, source=args.evalset)

    asked = len(items)
    items, collections = _collect_replies(items, args)
    if collections and not items:
        # Take the first failure ACROSS collections, not `collections[0]`'s.
        # When the candidate endpoint answers everything and the incumbent
        # answers nothing, collection zero has no failures at all, and indexing
        # into it raised an IndexError over the top of the message explaining
        # what actually went wrong.
        reason = next(
            (f.reason for c in collections for f in c.failures),
            "no endpoint reported a reason",
        )
        raise ValueError(f"no item in {args.evalset} got a reply — nothing can be scored. {reason}")

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
        judge=judge,
        judge_model=judge_model,
        agreement=agreement,
        # What the eval set was drawn from, not what survived collection. The
        # receipt then states 400 alongside claims totalling 388, rather than
        # quietly presenting the smaller number as the whole thing.
        traffic_captures=asked,
        tool_version=SERVER_INFO["version"],
        # What the receipt was issued against, so `clickllm guard` can later
        # tell whether the model behind the name is still that model. Absent,
        # that check has nothing to compare and silently does not run.
        fingerprints=fingerprints,
        # The cost the whole product is about. `suite` has accepted these since
        # it was written and `Policy` has computed the blended figure all along;
        # no surface supplied them, so the saving was permanently "unknown".
        incumbent_cost=args.incumbent_cost,
        monthly_cost=args.candidate_cost,
        traffic_window=args.traffic_window,
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
    from .prove.receipt import _NON_EVIDENTIARY, Receipt, verify

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
        if r.digest() == other.digest():
            print(f"verified · both receipts agree · {r.digest()[:12]}")
        else:
            # `verify` compares evidence, not identity — a rerun carries a
            # later date and often a newer build. Saying "both receipts agree"
            # over one digest invites the reader to check it against the other
            # file, find a different string, and distrust the verification.
            # Name what actually matched, and show both.
            #
            # Read the differing fields rather than assuming which one moved:
            # `tool_version` alone can differ, and asserting "differ in when
            # they were issued" would then be a confident, checkable falsehood
            # in the line that exists to stop exactly that.
            differing = [
                name
                for name in _NON_EVIDENTIARY
                if getattr(r, name, None) != getattr(other, name, None)
            ]
            print(
                f"verified · the evidence agrees · {r.digest()[:12]} vs "
                f"{other.digest()[:12]} (the receipts differ in "
                f"{' and '.join(differing) or 'no evidentiary field'}, "
                f"not in what they claim)"
            )
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
    from .kernels import KernelClaim, Plugin, PluginKind, entry_point_target, scaffold

    kind = PluginKind(args.kind)
    plugin = Plugin(
        name=args.name,
        kind=kind,
        # Not a hardcoded `:register` — STAT_LOGGER's entry point is the class
        # itself, so a fixed attribute here generated a package whose entry
        # point named something its own module did not define.
        target=entry_point_target(args.name, kind),
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


def cmd_version(args: argparse.Namespace) -> int:
    """`clickllm version`. Prints what is installed, and where it came from.

    The distribution and the command differ — PyPI refused `clickllm` as too
    similar to the existing `click-llm` — and someone reading a bug report needs
    to know which package to ask about, so both names are printed.
    """
    from . import __version__

    print(f"clickllm {__version__}")
    print("  distribution  clickllm-cli")
    print(f"  python        {sys.version.split()[0]}")
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    """`clickllm upgrade`. Prints how to upgrade; does not do it.

    Deliberately not self-updating. clickllm may be installed by uv, pipx or
    pip, into a tool environment this process cannot see, and a command that
    guesses wrong either fails confusingly or upgrades a different copy from the
    one the user is running. Detecting the installer and naming the exact command
    is honest and cannot corrupt anything.
    """
    from . import __version__

    print(f"clickllm {__version__} (clickllm-cli)")
    print()
    print("  Upgrade with whichever installed it:")
    print("    uv tool upgrade clickllm-cli")
    print("    pipx upgrade clickllm-cli")
    print("    python3 -m pip install --upgrade clickllm-cli")
    print()
    print("  Running it once, without installing:")
    print("    uvx --from clickllm-cli clickllm fit")
    print("    npx clickllm fit                        # if node is what you have")
    print()
    # npx has no upgrade step — the shim resolves a version per invocation, so
    # `npx clickllm@latest` IS the upgrade. Saying so matters because this text
    # listed three Python installers and no npm one for four releases after the
    # npm package went live, which reads as "npx is not supported".
    print("  Installed via npx? There is nothing to upgrade — the shim resolves")
    print("  per run, so `npx clickllm@latest fit` is already the newest, and")
    print(f"  `npx clickllm@{__version__} fit` pins one build.")
    print()
    print("  This prints the commands rather than running one: clickllm cannot")
    print("  see which tool owns its environment, and upgrading the wrong copy")
    print("  is worse than telling you.")
    return 0


def main(argv: list[str] | None = None) -> int:
    from . import __version__

    p = argparse.ArgumentParser(
        prog="clickllm",
        description="Run open models properly on your own hardware — and prove they hold.",
    )
    # Both spellings. `--version` is the convention; `clickllm version` is what
    # people actually type, and answering it with an argparse error over a list
    # of twenty subcommands is a bad first minute after installing.
    p.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"clickllm {__version__} (clickllm-cli)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version", help="print the installed version").set_defaults(fn=cmd_version)
    sub.add_parser("upgrade", help="how to upgrade this install").set_defaults(fn=cmd_upgrade)

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
        "--engine",
        help="force this engine rather than letting the planner pick one "
        "(e.g. ollama, which has no hardware signature to be auto-selected by)",
    )
    r.add_argument(
        "--tag",
        help="the Ollama tag to pull, e.g. llama3.1:8b-instruct-q8_0 — required "
        "with --engine ollama, since its registry names are not the catalogue's ids",
    )
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

    ob = sub.add_parser("observe", help="record real traffic through a capture gateway")
    ob.add_argument("--upstream", required=True, help="the provider you use today")
    ob.add_argument("--port", type=int, default=8787)
    ob.add_argument("--candidate", help="a second backend to shadow against")
    ob.add_argument("--capture", help="where the log goes (default: ~/.clickllm/captures.log)")
    ob.add_argument("--no-capture", action="store_true", dest="no_capture")
    ob.set_defaults(fn=cmd_observe)

    di = sub.add_parser("distill", help="turn a capture log into an eval set")
    di.add_argument("--capture", help="the log to read (default: ~/.clickllm/captures.log)")
    di.add_argument("--key", help="the capture key (default: ~/.clickllm/capture.key)")
    di.add_argument("--out", default="evalset.json")
    di.add_argument("--budget", type=int, default=200, help="total eval items to sample")
    di.add_argument("--min-per-cluster", type=int, default=3, dest="min_per_cluster")
    di.add_argument("--json", action="store_true")
    di.set_defaults(fn=cmd_distill)

    mg = sub.add_parser("migrate", help="walk the loop: distill, prove, and what the gate permits")
    mg.add_argument("--candidate", help="the model being evaluated")
    mg.add_argument("--incumbent", default="incumbent")
    mg.add_argument("--candidate-endpoint", dest="candidate_endpoint")
    mg.add_argument("--candidate-model", dest="candidate_model")
    mg.add_argument("--incumbent-endpoint", dest="incumbent_endpoint")
    mg.add_argument("--incumbent-model", dest="incumbent_model")
    mg.add_argument("--capture", help="capture log (default: ~/.clickllm/captures.log)")
    mg.add_argument("--key", help="capture key (default: ~/.clickllm/capture.key)")
    mg.add_argument("--state", help="migration state file (default: ~/.clickllm/migration.json)")
    mg.add_argument("--budget", type=int, default=200)
    mg.add_argument("--min-per-cluster", type=int, default=3, dest="min_per_cluster")
    mg.add_argument("--bar", type=float, default=0.9)
    _add_collection_flags(mg)
    mg.add_argument("--status", action="store_true", help="where the migration is; runs nothing")
    mg.add_argument("--step", action="store_true", help="run one pass (the default)")
    mg.add_argument("--install", action="store_true", help="print a crontab fragment")
    mg.add_argument("--every-hours", type=int, default=24, dest="every_hours")
    mg.set_defaults(fn=cmd_migrate)

    ms = sub.add_parser("measure", help="measure real decode throughput, or refuse to")
    ms.add_argument("--endpoint", required=True, help="an OpenAI-compatible base URL")
    ms.add_argument("--model", default="", help="catalogue id, for the roofline comparison")
    ms.add_argument(
        "--served-model",
        dest="served_model",
        default="",
        help="the name the endpoint knows it by, when it differs from the catalogue id",
    )
    ms.add_argument("--samples", type=int, default=M_DEFAULT_SAMPLES)
    ms.add_argument("--max-tokens", type=int, default=M_DEFAULT_MAX_TOKENS, dest="max_tokens")
    ms.add_argument("--context", default="32k", help="context for the roofline comparison")
    ms.add_argument("--concurrency", type=int, default=1)
    ms.add_argument("--out", help="write the measurement as JSON")
    ms.add_argument("--json", action="store_true")
    ms.set_defaults(fn=cmd_measure)

    g = sub.add_parser("guard", help="check whether a receipt still holds")
    g.add_argument("receipt", help="path to a receipt JSON file")
    g.add_argument("--traffic", help="JSON file of current cluster shares")
    g.add_argument("--fingerprints", help="JSON file of current model fingerprints")
    g.add_argument(
        "--available", action="append", help="a candidate model that exists now (repeatable)"
    )
    g.add_argument("--today", help="ISO date to evaluate against (default: today)")
    g.add_argument(
        "--fail-on",
        dest="fail_on",
        choices=("invalidating", "any"),
        default="invalidating",
        help=(
            "which findings exit nonzero. 'invalidating' (default): only what "
            "voids the proof — the model changed, or traffic moved outside the "
            "eval set. 'any': also age and new releases, for a release gate "
            "that should not deploy on an unreviewed proof."
        ),
    )
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
    # The judge. Optional, and a run without one is the stronger result — it says
    # every verdict came from a deterministic grader.
    pv.add_argument(
        "--judge-endpoint",
        dest="judge_endpoint",
        help="OpenAI-compatible base URL for the tier-3 judge. Only consulted "
        "where the deterministic graders could not decide, and never enough on "
        "its own to move traffic",
    )
    pv.add_argument(
        "--judge-model",
        dest="judge_model",
        help="the judge's id. Required with --judge-endpoint and stamped on the "
        "receipt — an undisclosed judge makes every score it touched unauditable",
    )
    pv.add_argument(
        "--judge-agreed",
        dest="judge_agreed",
        type=int,
        default=0,
        help="how many of the sampled comparisons a human agreed with",
    )
    pv.add_argument(
        "--judge-samples",
        dest="judge_samples",
        type=int,
        default=0,
        help="how many comparisons a human checked. Zero means the judge is "
        "UNMEASURED, which the report says out loud and the gate refuses to "
        "advance on",
    )
    _add_collection_flags(pv)
    pv.add_argument(
        "--incumbent-cost",
        dest="incumbent_cost",
        type=float,
        help=(
            "what you spend on the incumbent per month, in dollars. Without it "
            "the report says the saving is unknown rather than guessing one — a "
            "fabricated saving is the single most damaging number here."
        ),
    )
    pv.add_argument(
        "--candidate-cost",
        dest="candidate_cost",
        type=float,
        help=(
            "what the candidate would cost per month. `clickllm fit` and "
            "`clickllm host` both estimate this for self-hosting; pass the "
            "figure you believe."
        ),
    )
    pv.add_argument(
        "--traffic-window",
        dest="traffic_window",
        default="",
        metavar="PERIOD",
        help=(
            "how long the traffic was captured over, e.g. '14 days'. Captures "
            "carry no timestamps, so this is not derivable — and without it a "
            "monthly saving would be an extrapolation from an unknown period, "
            "which the receipt refuses to make."
        ),
    )
    pv.add_argument(
        "--fingerprints",
        dest="prove_fingerprints",
        help=(
            "JSON file mapping model name to whatever identifies the weights you "
            "served — a digest, a revision, a build id. Recorded in the receipt so "
            "`clickllm guard` can later detect a silent provider-side swap. Without "
            "it that check has nothing to compare against and cannot run."
        ),
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
    # RuntimeError included: `desktop.install()` documents and raises it on any
    # unsupported platform, and it was not in this tuple — so `clickllm desktop`
    # on Linux printed a traceback, breaking the convention stated two lines
    # down. The modules raise it deliberately as "this environment cannot do
    # what you asked", which is a sentence-and-exit-2, not a crash.
    except (KeyError, ValueError, OSError, RuntimeError) as e:
        # Convention: a bad path, an altered receipt or an unparseable date is a
        # nonzero exit with a sentence, never a traceback. `json.JSONDecodeError`
        # is a `ValueError`, so malformed input is covered here too.
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
