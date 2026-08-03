"""`clickllm run` — from a catalogue id to a served endpoint you can curl.

Everything else in this package stops at the door: `fit` sizes, `plan` configures,
`session` hands you a command. That boundary is deliberate for *deployment* —
nothing here moves production traffic — but it made the local case absurd. You
were handed a command for your own laptop and asked to paste it back in.

This closes that one gap: your machine, your weights, one process, loopback by
default. It is still the only thing in the repo that starts anything.

## What this module refuses to do

**Download weights.** The engine already does. mlx_lm's `_download` is documented
as "ensures the model is available locally. If the path does not exist locally,
it is downloaded from the Hugging Face Hub"; vLLM and SGLang behave the same. A
downloader here would be a second, worse cache next to the one the engine keeps —
and the first symptom of it going wrong is 30 GB fetched twice.

**Construct a repo name from a pattern.** This is the part that looks trivial and
is not. Checked against the live index:

- `mlx-community/Llama-3.1-8B-Instruct-8bit` — **404**, and base name plus
  suffix produces exactly this
- `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit` — exists. Note the `Meta-`,
  which the base repo `meta-llama/Llama-3.1-8B-Instruct` does not carry
- `mlx-community/phi-4-8bit` — exists, but HF *search* does not surface it
- `mlx-community/Qwen3-30B-A3B-8bit` — exists

So neither search nor a naming pattern is trustworthy on its own, and a static
model→repo map would be stale the week it was written. What is trustworthy is a
direct existence check. [`resolve_weights`] therefore *proposes* several plausible
spellings and *verifies* each, in order, and returns the first that is really
there — or a [`Refusal`] naming every candidate it tried. It never returns a name
it has not confirmed, because a fabricated repo id fails minutes later inside
someone else's downloader, with their error message.

Precision is part of the identity, not a flag: `-4bit` and `-8bit` are different
repos on MLX (see [`MlxAdapter`][clickllm.engines.MlxAdapter]), which is why
resolution is keyed on the quant the solver picked.

## The three things that stop a launch

1. **It does not fit this machine.** Refuse with the shortfall in GB. A server
   that OOMs ten minutes in is worse than one that never started.
2. **No verified flag dialect for the chosen engine.** Same rule as `engines`:
   another engine's flags would produce a command that cannot run.
3. **The installed engine rejects a flag we generated.** [`unknown_flags`] reads
   `--help` off the real binary. If anything in the argv is not in it, the plan
   is a refusal — handing back a command we can already see will fail is the
   defect this repo has actually shipped before.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from . import catalog, fit, hardware
from .atomicio import update_json
from .catalog import ModelSpec
from .engines import Setting, adapter_for, unknown_flags
from .fit import Fit
from .hardware import Hardware
from .plan import Engine, Requirements, Workload
from .plan import plan as configure

__all__ = [
    "Endpoint",
    "LaunchPlan",
    "Refusal",
    "Resolved",
    "demo",
    "plan",
    "resolve_weights",
    "serve",
]

GB = 1024**3

#: Hugging Face's model endpoint. 200 means the repo is really there; 401/403/404
#: all mean "not something we can serve", which is the only distinction that
#: matters here.
MODEL_URL = "https://huggingface.co/api/models/{repo}"

#: The precision label MLX bakes into a repo name, per catalogue quant. Not a
#: flag: `-4bit` and `-8bit` are different downloads, so the quant the solver
#: chose is part of which repo we are looking for.
#: `fp8` maps to `8bit` because MLX publishes no fp8 — 8-bit integer is the
#: nearest thing that exists, and pretending otherwise would resolve nothing.
#: GGUF quantisation spellings. llama.cpp's own naming, not ours: Q4_K_M is the
#: community default at 4-bit and Q8_0 at 8-bit.
GGUF_SUFFIX = {"q4": "Q4_K_M", "q8": "Q8_0", "fp16": "F16"}

#: Bumped whenever `candidates()` changes what it proposes, so answers cached by
#: the old logic are ignored rather than trusted. Costs one HTTP call per model
#: after an upgrade; buys never serving a repo the current resolver would reject.
_RESOLVER_VERSION = "r2"

MLX_SUFFIX = {
    "q3": "3bit",
    "q4": "4bit",
    "q5": "5bit",
    "q6": "6bit",
    "q8": "8bit",
    "fp8": "8bit",
    "fp16": "bf16",
    "bf16": "bf16",
}

#: How to get each engine, for the error you see when the binary is missing.
#: A launcher that says "command not found" and stops has wasted the one moment
#: it knew exactly what you needed.
INSTALL_HINT = {
    "mlx": "pip install mlx-lm",
    "vllm": "pip install vllm",
    "vllm-tpu": "pip install vllm  # plus the tpu-inference plugin",
    "sglang": "pip install 'sglang[all]'",
    "ollama": "https://ollama.com/download",
}

#: Concurrency levels re-planned when the chosen engine has no dialect we can
#: drive, to find the smallest one that does. Probed rather than tabulated —
#: the engine choice stays in `plan._pick_engine`, where it belongs.
_CONCURRENCY_PROBES = (2, 4, 8, 16, 32, 64)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why nothing was launched, and what to do instead.

    Carries `tried` so a resolution failure reads as "these six names do not
    exist" rather than "could not find it", which is unactionable and is what
    every tool in this space says.

    The field is the candidate list this resolution CONSIDERED, not a claim that
    each was reached — offline, none of them were, and the list is still the
    useful thing to hand someone. `reason` says how many answered; the render
    calls them candidates for the same reason.
    """

    reason: str
    tried: tuple[str, ...] = ()
    next_step: str = ""

    def render(self) -> str:
        out = [f"  REFUSED   {self.reason}"]
        if self.tried:
            out += ["", "  CANDIDATES"] + [f"    · {t}" for t in self.tried]
        if self.next_step:
            out += ["", f"  NEXT      {self.next_step}"]
        return "\n".join(out)


@dataclass(frozen=True, slots=True)
class Resolved:
    """A repo that was confirmed to exist, and everything checked to find it."""

    repo: str
    tried: tuple[str, ...]
    #: True when the answer came off disk, so no request was made this run.
    from_cache: bool = False


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """A command that will start, on this machine, with these weights."""

    model: ModelSpec
    quant: str
    repo: str
    engine: str
    engine_why: str
    argv: tuple[str, ...]
    host: str
    port: int
    fit: Fit
    #: Intents the engine had no verified flag for — see `Plan.command`.
    gaps: tuple[str, ...] = ()
    tried: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    #: Environment this engine needs beyond argv — see `Plan.environment`.
    #: Ollama's whole configuration lives here; `argv` alone would start it
    #: with its defaults, silently ignoring what was solved for.
    env: tuple[tuple[str, str], ...] = ()

    @property
    def base(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def command(self) -> str:
        """The argv as you would type it, environment prefix included.

        Quoted: some flags carry JSON, and an env value could carry anything.
        """
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env)
        rest = " ".join(shlex.quote(a) for a in self.argv)
        return f"{prefix} {rest}" if prefix else rest

    def render(self) -> str:
        f = self.fit
        out = [
            f"  MODEL     {self.model.name} @ {self.quant}"
            f"   ({f.total_bytes / GB:.1f} GB of {f.usable_bytes / GB:.1f} GB usable)",
            f"  WEIGHTS   {self.repo}"
            + (f"   (confirmed; {len(self.tried)} candidate(s) checked)" if self.tried else ""),
            f"  ENGINE    {self.engine}   {self.engine_why}",
            f"  ENDPOINT  {self.base}/v1",
        ]
        if f.tokens_per_sec:
            out.append(
                f"  SPEED     ~{f.tokens_per_sec:.0f} tok/s single-stream"
                "   (roofline estimate, not measured)"
            )
        out += ["", "  RUN", f"    {self.command}"]
        if self.gaps:
            out += ["", "  NOT EXPRESSED"] + [f"    · {g}" for g in self.gaps]
        if self.notes:
            out += ["", "  NOTES"] + [f"    · {n}" for n in self.notes]
        return "\n".join(out)


@dataclass(slots=True)
class Endpoint:
    """A running engine and the URL it answers on."""

    base: str
    model: str
    pid: int
    ready_seconds: float
    process: subprocess.Popen[bytes] = field(repr=False)

    def wait(self) -> int:
        """Block until the engine exits. Ctrl-C stops it rather than orphaning it.

        The child is left in this process group deliberately, so an interactive
        Ctrl-C reaches it directly; this just waits for it to finish dying and
        terminates it explicitly if it does not.
        """
        try:
            return self.process.wait()
        except KeyboardInterrupt:
            self.stop()
            return self.process.returncode or 0

    def stop(self, grace: float = 10.0) -> tuple[int, ...]:
        """Terminate, then kill — the engine AND anything it spawned.

        Signalling only the immediate child is not stopping the engine. vLLM
        runs an API server that forks engine workers, each holding GPU memory;
        terminate the parent alone and the workers are reparented to init and
        keep the device. `serve()`'s timeout path says the engine "has been
        stopped rather than left running", and for a multi-process engine that
        sentence was false.

        Descendants are collected BEFORE the parent is signalled. Afterwards the
        parent links are gone — the children have been reparented — and there is
        no way left to tell which processes were ours.

        A new session would make this trivial (signal the group) and is
        deliberately not used: `serve()` keeps the child in this process group
        so an interactive Ctrl-C reaches it, and that is worth more than tidy
        teardown, because Ctrl-C is what people actually press.

        Returns:
            PIDs that were still alive after the kill — empty in the normal
            case. Non-empty is the honest answer to "did you stop it", and the
            caller can say so rather than claiming the GPU is free.
        """
        if self.process.poll() is not None:
            return ()
        kin = _descendants(self.pid)
        self.process.terminate()
        try:
            self.process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        return _reap(kin, grace)


def _descendants(pid: int) -> tuple[int, ...]:
    """Every process descended from `pid`, deepest last. Empty if we cannot ask.

    Reads the whole pid/ppid table once and walks it, rather than shelling out
    per level: one `ps` is cheap, N is not, and the table is a consistent
    snapshot where N separate calls are not.

    Returning empty on failure is deliberate. A best-effort teardown that
    reports what it could not do beats one that raises on a machine where `ps`
    is missing and takes the shutdown path with it.
    """
    try:
        proc = subprocess.run(  # noqa: S603,S607 — fixed argv, no user input
            ["ps", "-Ao", "pid=,ppid="], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    children: dict[int, list[int]] = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            child, parent = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children.setdefault(parent, []).append(child)

    out: list[int] = []
    queue = list(children.get(pid, ()))
    seen = {pid}
    while queue:
        cur = queue.pop(0)
        if cur in seen:  # a recycled pid could otherwise loop forever
            continue
        seen.add(cur)
        out.append(cur)
        queue.extend(children.get(cur, ()))
    return tuple(out)


def _reap(pids: tuple[int, ...], grace: float) -> tuple[int, ...]:
    """SIGTERM, wait, SIGKILL. Returns whatever is still alive after that."""
    alive = [p for p in pids if _running(p)]
    for pid in alive:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + max(0.1, min(grace, 5.0))
    while time.monotonic() < deadline:
        alive = [p for p in alive if _running(p)]
        if not alive:
            return ()
        time.sleep(0.05)
    for pid in alive:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    time.sleep(0.1)
    return tuple(p for p in alive if _running(p))


def _running(pid: int) -> bool:
    """Whether the pid exists. Signal 0 checks without delivering anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, not ours to signal
    return True


# --------------------------------------------------------------------------- #
# Weight resolution
# --------------------------------------------------------------------------- #


def state_dir() -> Path:
    """Where resolutions are remembered. Never the catalogue.

    A confirmed repo is a fact about the world, not a fact about the model, so
    it does not belong in `models.json` — writing there would mean a resolution
    on one machine silently edited a file the solver reads on every other.
    """
    base = os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")
    return Path(base) / "clickllm"


def _cache_path(cache: Path | None) -> Path:
    return cache or state_dir() / "weights.json"


def _cache_read(path: Path) -> dict[str, str]:
    """The remembered resolutions. A corrupt cache is empty, never an error."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, str)}


def _cache_write(path: Path, key: str, repo: str) -> None:
    """Remember one resolution.

    Two `clickllm run` invocations resolving different models at once used to
    lose one of the two entries — an unlocked read-modify-write plus a scratch
    filename both processes shared. Forty concurrent resolutions kept seven.

    That fix lived here, and the identical defect stayed in `cache.py` and
    `catalog_update.py` for another release, because fixing one instance of a
    duplicated bug is not fixing the bug. All three now call one implementation.

    A failure is not a failure to launch: the answer is already in hand, and a
    read-only home should not stop a server from starting.
    """
    update_json(path, lambda cur: {**(cur if isinstance(cur, dict) else {}), key: repo}, default={})


def hf_exists(repo: str, timeout: float = 10.0) -> bool:
    """Whether `repo` is a real, readable model on the Hub.

    Raises:
        OSError: if the index could not be reached at all. That is deliberately
            not the same as "does not exist" — reporting an offline laptop as
            "none of these repos exist" would send someone hunting a naming bug
            that is not there.
    """
    import urllib.error
    import urllib.request

    from .catalog_update import USER_AGENT

    req = urllib.request.Request(
        MODEL_URL.format(repo=repo), headers={"User-Agent": USER_AGENT}, method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https only
            return resp.status == 200
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            return False
        raise  # a 5xx is the index having a bad day, not an answer about the repo


def _stems(model: ModelSpec) -> list[str]:
    """Plausible spellings of this model's name, in the order worth trying.

    Several, not one, because the drift is real and undirected:
    `meta-llama/Llama-3.1-8B-Instruct` is published on MLX as
    `Meta-Llama-3.1-8B-Instruct`, gaining a prefix the base repo never had.
    Every stem here is a *guess*; nothing is returned to a caller until it has
    been checked against the index.
    """
    out: list[str] = []
    org, _, tail = (model.repo or "").partition("/")
    if tail:
        out.append(tail)
        # The prefix drift: the org's own name, re-attached. `meta-llama` is why
        # this exists, and it costs one 404 on the orgs where it is wrong.
        first = org.split("-")[0]
        prefix = first[:1].upper() + first[1:]
        if prefix and not tail.lower().startswith(prefix.lower()):
            out.append(f"{prefix}-{tail}")
    else:
        out.append(model.id)

    # Instruct-tuned and base weights live under names that differ by one word,
    # and the catalogue records whichever one it recorded.
    suffix = "-Instruct"
    for stem in list(out):
        out.append(stem[: -len(suffix)] if stem.endswith(suffix) else f"{stem}{suffix}")

    return list(dict.fromkeys(out))


def candidates(model: ModelSpec, quant: str, engine: str) -> tuple[str, ...]:
    """Repo names worth checking for (model, quant, engine), best guess first.

    vLLM and SGLang take the base repo — they quantise on load, or accept a
    pre-quantised repo under its own name — so that goes first and is usually
    the whole list. MLX cannot: the precision is in the weights, so the search
    is over `mlx-community` spellings at the right bit-width.

    llama.cpp cannot either, and for a sharper reason: it does not load a
    transformers repo at all. It loads GGUF. Handing `llama-server --model` a
    repo id like `meta-llama/Llama-3.1-8B-Instruct` produces a command that
    fails on start-up — which is exactly the class of unrunnable output this
    resolver exists to prevent, and is what the first cut of the llama.cpp
    adapter emitted.

    Ollama is different again: it has its own registry and its own names, so
    there is no Hugging Face repo to confirm. `ollama serve` starts a daemon and
    pulls on first use, so the resolution is a tag rather than a repo, and there
    is nothing here to check it against.
    """
    if engine == "ollama":
        # Its registry, its names, its pull. Nothing to resolve against the Hub.
        return ()
    if engine == "llama.cpp":
        bits = GGUF_SUFFIX.get(quant)
        if bits is None:
            return ()
        # The community publishes GGUF under two long-standing conventions, and
        # which one a given model uses is not predictable — so both are proposed
        # and each is existence-checked before it is printed.
        out = [f"{org}/{stem}-GGUF" for stem in _stems(model) for org in ("bartowski", "TheBloke")]
        if model.repo:
            out.insert(0, f"{model.repo}-GGUF")
        return tuple(dict.fromkeys(out))

    if engine != "mlx":
        return (model.repo,) if model.repo else ()

    bits = MLX_SUFFIX.get(quant)
    if bits is None:
        return ()
    out = [f"mlx-community/{stem}-{bits}" for stem in _stems(model)]
    if bits == "bf16" and model.repo:
        # Nothing was quantised, so the original repo is a real answer — mlx-lm
        # loads standard HF weights. Last, because the community conversions
        # start faster.
        out.append(model.repo)
    return tuple(dict.fromkeys(out))


def resolve_weights(
    model: ModelSpec,
    quant: str,
    engine: str,
    *,
    exists: Callable[[str], bool] | None = None,
    cache: Path | None = None,
) -> Resolved | Refusal:
    """The repo to serve, confirmed to exist — or every name that was tried.

    Args:
        model: catalogue entry being served.
        quant: the quantisation the solver picked. Part of the repo identity on
            MLX, so it is not optional.
        engine: engine id, as `adapter_for` spells it.
        exists: existence check, injected. Defaults to [`hf_exists`]. Injected
            rather than mocked so the test suite never opens a socket.
        cache: where confirmed resolutions are remembered. Defaults to
            [`state_dir`]`()/weights.json`.

    Returns:
        A [`Resolved`], or a [`Refusal`] listing every candidate checked. Never a
        name that was not confirmed.
    """
    if engine == "ollama":
        # Its own registry, its own pull — see `candidates`. The model argument
        # is not part of the launch command either (`OllamaAdapter.command`), so
        # there is nothing here to verify against the Hub, and refusing for want
        # of one would be refusing to launch an engine that has no such gap.
        return Resolved(model.id, ())

    check = exists or hf_exists
    path = _cache_path(cache)
    # The resolver's own version is part of the key. A cached repo is a fact
    # about (model, quant, engine) AND about the logic that proposed it — and
    # this was proved the hard way: the first cut of the llama.cpp branch
    # proposed the transformers repo, cached it, and kept serving
    # `meta-llama/Llama-3.1-8B-Instruct` to llama-server after the GGUF fix
    # landed. A wrong answer that outlives the bug that made it is worse than no
    # cache, because upgrading does not clear it and nothing says why.
    key = f"{_RESOLVER_VERSION}:{engine}:{model.id}:{quant}"

    if (hit := _cache_read(path).get(key)) is not None:
        return Resolved(hit, (hit,), from_cache=True)

    tried = candidates(model, quant, engine)
    if not tried:
        return Refusal(
            reason=(
                f"no Hugging Face repo is known for {model.id!r}"
                + (
                    f" and {quant!r} has no MLX equivalent"
                    if engine == "mlx" and quant not in MLX_SUFFIX
                    else ""
                )
                + "; the catalogue leaves `repo` absent rather than guessing it"
            ),
            next_step=f"clickllm catalog-add <repo> --params-b {model.params_b:g} --network",
        )

    # `checked` is what was ACTUALLY asked about, which diverges from `tried` the
    # moment the network fails mid-loop. Reporting the full candidate list there
    # says "these six do not exist" when five of them were never reached — and
    # the whole point of listing candidates is to send someone to the right place.
    # `checked` is what actually ANSWERED, which diverges from `tried` on both
    # exits. On the failure path, reporting the full candidate list says "these
    # six do not exist" when five were never reached. On the success path the
    # same overclaim reached the user's screen: `Resolved` renders "confirmed;
    # N candidate(s) checked", and returning `tried` there printed 6 for a match
    # found on the first try. Appended AFTER `check` returns, so the repo whose
    # lookup raised is not counted as one that answered.
    checked: list[str] = []
    try:
        for repo in tried:
            found = check(repo)
            checked.append(repo)
            if found:
                final = repo
                if engine == "llama.cpp":
                    # `--hf-repo` takes an optional `:quant` suffix that selects
                    # the file inside the repo; the bare repo id lets
                    # llama-server default to Q4_K_M regardless of what the
                    # solver actually sized. `tried` is non-empty only when
                    # `quant` has a GGUF_SUFFIX entry, so this is always a hit.
                    final = f"{repo}:{GGUF_SUFFIX[quant]}"
                _cache_write(path, key, final)
                return Resolved(final, tuple(checked))
    except OSError as e:
        return Refusal(
            reason=(
                f"could not reach the model index to confirm any repo — {e}. "
                f"{len(checked)} of {len(tried)} candidates answered before the "
                f"connection failed; the rest are unknown, not absent"
            ),
            tried=tried,
            next_step="connect, or re-run once a previous run has cached the answer",
        )

    return Refusal(
        reason=(
            f"none of these repos exist for {model.id} at {quant} on {engine}. "
            f"Names are not constructible — mlx-community/Llama-3.1-8B-Instruct-8bit "
            f"is a 404 while the Meta- prefixed spelling is real — so this is a "
            f"report of what was checked, not a guess at what to use"
        ),
        tried=tried,
        next_step=(
            "find the real repo on huggingface.co and set it with "
            "`clickllm catalog-add <repo> --network`"
        ),
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def _launchable_alternative(hw: Hardware, req: Requirements) -> tuple[int, str] | None:
    """Smallest concurrency at which the planner picks an engine we can drive.

    Computed by re-planning rather than by a second engine table: which engine
    suits which workload is `plan._pick_engine`'s decision and must stay there,
    or the two will drift and only one of them will be the one that ran.
    """
    for c in _CONCURRENCY_PROBES:
        if c <= req.concurrency:
            continue
        engine = configure(hw, replace(req, concurrency=c)).engine.value
        if adapter_for(engine) is not None:
            return c, engine
    return None


def _does_not_fit(model: ModelSpec, hw: Hardware, context: int, concurrency: int) -> Refusal:
    """The refusal for a model too big for this machine, with the shortfall."""
    smallest = min(model.quants, key=lambda q: catalog.QUANT_BITS[q])
    f = fit.solve(model, smallest, hw, context, concurrency)
    short = f.total_bytes - f.usable_bytes
    return Refusal(
        reason=(
            f"{model.name} does not fit {hw.name}: even at {smallest} it needs "
            f"{f.total_bytes / GB:.1f} GB against {f.usable_bytes / GB:.1f} GB usable — "
            f"short by {short / GB:.1f} GB at {context:,} context, concurrency {concurrency}"
        ),
        next_step=(
            f"`clickllm host {model.id}` ranks the places that can run it, free "
            f"tiers first, and writes a deploy artifact. `clickllm where "
            f"{model.id}` lists the hardware; `--context`/`--concurrency` lower "
            f"the requirement here."
        ),
    )


def plan(
    model_id: str,
    *,
    quant: str | None = None,
    context: int = 32_768,
    concurrency: int = 4,
    port: int = 8000,
    host: str = "127.0.0.1",
    engine: str | None = None,
    hw: Hardware | None = None,
    exists: Callable[[str], bool] | None = None,
    cache: Path | None = None,
) -> LaunchPlan | Refusal:
    """Everything needed to start this model here, or the reason not to.

    Args:
        model_id: catalogue id, e.g. `qwen3-30b-a3b`.
        quant: force a quantisation; default is the best one that fits.
        context: context length to serve.
        concurrency: simultaneous in-flight requests to tune for. Defaults to 4
            rather than 1 because this starts a *server*: the planner's own
            interactive settings already assume burst traffic, and on Apple
            silicon 1 selects llama.cpp, for which no verified flag dialect
            exists.
        port: port to bind.
        host: bind address. Loopback by default — an inference endpoint has no
            authentication, and binding it wider is a decision, not a default.
        engine: force this engine rather than letting the planner pick one.
            Some engines have no hardware signature to be auto-selected by —
            Ollama runs equally well anywhere — so this is their only way in.
        hw: the machine. Detected when omitted.
        exists: repo existence check, forwarded to [`resolve_weights`].
        cache: resolution cache path, forwarded to [`resolve_weights`].

    Returns:
        A [`LaunchPlan`], or a [`Refusal`] when the model does not fit, no repo
        can be confirmed, the engine has no verified dialect, or the installed
        engine would reject a flag.

    Raises:
        KeyError: the model id is not in the catalogue.
        ValueError: `quant` is not one this model publishes, or `engine` names
            no known engine.
    """
    model = catalog.get(model_id)
    machine = hw or hardware.detect()

    if quant is not None and quant not in model.quants:
        raise ValueError(f"{model.id} does not publish {quant!r}; it has {', '.join(model.quants)}")
    forced = Engine(engine) if engine is not None else None

    if quant is None:
        best = fit.best_quant(model, machine, context, concurrency)
        if best is None:
            return _does_not_fit(model, machine, context, concurrency)
        sized = best
    else:
        sized = fit.solve(model, quant, machine, context, concurrency)
        if not sized.feasible:
            return _does_not_fit(model, machine, context, concurrency)

    req = Requirements(Workload.INTERACTIVE, concurrency=concurrency, context=context)
    deployment = configure(machine, req, model, sized.quant, force_engine=forced)
    engine_name = deployment.engine.value

    if engine_name == "llama.cpp":
        # --ctx-size is TOTAL across --parallel slots (LlamaCppAdapter.command
        # multiplies context by it), and the planner's MAX_CONCURRENT knob is
        # not `concurrency` — interactive workloads round it up for burst
        # headroom (e.g. 4 becomes 32). Sizing the fit at the caller's
        # concurrency while launching at the padded slot count would approve a
        # plan whose own launch command then asks for far more KV than checked.
        slots = deployment.get(Setting.MAX_CONCURRENT)
        if slots is not None and isinstance(slots.value, int) and slots.value > concurrency:
            sized = fit.solve(model, sized.quant, machine, context, slots.value)
            if not sized.feasible:
                return _does_not_fit(model, machine, context, slots.value)

    adapter = adapter_for(engine_name)
    if adapter is None:
        alt = _launchable_alternative(machine, req)
        return Refusal(
            reason=(
                f"the planner chose {engine_name} for this workload, and there is no "
                f"verified flag dialect for it — emitting another engine's flags "
                f"would produce a command that cannot start"
            ),
            next_step=(
                f"--concurrency {alt[0]} selects {alt[1]}, which this can launch"
                if alt
                else f"`clickllm build` prints the reasoning for a {engine_name} deployment "
                f"you run yourself"
            ),
        )

    resolved = resolve_weights(model, sized.quant, engine_name, exists=exists, cache=cache)
    if isinstance(resolved, Refusal):
        return resolved

    argv, gaps = deployment.command(resolved.repo)
    env = list(deployment.environment())
    if engine_name == "ollama":
        # `ollama serve --help` has exactly one flag, `--help`; there is no
        # --host/--port to append, only OLLAMA_HOST, so the bind address goes
        # through the same channel as everything else this engine is told.
        env.append(("OLLAMA_HOST", f"{host}:{port}"))
    else:
        # All the flag-driven dialects spell the bind address the same way,
        # which is why this is appended here rather than becoming a `Setting`:
        # there is no translation to do, and a knob with one spelling earns
        # nothing.
        argv += ["--host", host, "--port", str(port)]

    rejected = unknown_flags(adapter, argv)
    if rejected:
        return Refusal(
            reason=(
                f"the installed {engine_name} does not accept {', '.join(rejected)}. "
                f"Its own --help is the authority, and handing back a command that "
                f"cannot start is worse than not handing one back"
            ),
            tried=(" ".join(argv),),
            next_step=f"upgrade {engine_name}, or file this — the dialect in `engines` has drifted",
        )

    notes: list[str] = list(deployment.warnings) + list(deployment.notes)
    if rejected is None:
        notes.append(
            f"{engine_name} is not installed here, so its flags could not be checked "
            f"against the binary — install it with `{INSTALL_HINT.get(engine_name, engine_name)}`"
        )
    if resolved.from_cache:
        notes.append("weights repo came from the local cache; nothing was fetched to plan this")
    if host not in ("127.0.0.1", "localhost", "::1"):
        notes.append(
            f"binding to {host} exposes an unauthenticated inference endpoint to "
            f"your network — anyone who can reach it can use your GPU"
        )
    notes.append("the engine downloads the weights itself on first start; this does not")

    return LaunchPlan(
        model=model,
        quant=sized.quant,
        repo=resolved.repo,
        engine=engine_name,
        engine_why=deployment.engine_why,
        argv=tuple(argv),
        host=host,
        port=port,
        fit=sized,
        gaps=gaps,
        tried=resolved.tried,
        notes=tuple(notes),
        env=tuple(sorted(set(env))),
    )


# --------------------------------------------------------------------------- #
# Running it
# --------------------------------------------------------------------------- #


def _healthy(base: str, model: str = "clickllm-probe", timeout: float = 10.0) -> bool:
    """Whether the engine can actually answer a request yet.

    Readiness is one real token, not a 200 on `/v1/models`. That distinction was
    measured, not assumed: against `mlx_lm.server` still downloading its weights,
    `/v1/models` returned **200 in 4 ms** while a genuine completion returned
    nothing in 30 s. The list endpoint is served from static config, so it proves
    the process bound its port — nothing more.

    An earlier version of this function polled `/v1/models` and its docstring
    asserted the endpoint "only answers once weights are loaded". It does not,
    and the cost of believing so is the worst kind of wrong answer: `clickllm
    run` printing READY over an endpoint that hangs on first use.

    A timeout is "not ready yet", not a failure — during load that is exactly
    what a real request does.
    """
    import json as _json
    import urllib.error
    import urllib.request

    body = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "1"}],
            "max_tokens": 1,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(  # noqa: S310
        f"{base}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status == 200
    except urllib.error.HTTPError as e:
        # The engine answered and rejected the request — it is loaded. A 4xx
        # over an unknown model name still proves inference is up, which is
        # what is being waited for.
        return 400 <= e.code < 500
    except OSError:
        return False


def serve(
    launch_plan: LaunchPlan,
    *,
    wait: bool = True,
    timeout: float = 1800.0,
    poll: float = 1.0,
    echo: Callable[[str], None] = print,
) -> Endpoint:
    """Start the engine and, by default, wait until it answers.

    The command is printed before it runs, in full. This starts a process on
    your machine; you should be able to see exactly what it is, copy it, and run
    it yourself without this tool.

    Args:
        launch_plan: from [`plan`].
        wait: poll until the endpoint is healthy. False returns as soon as the
            process is spawned.
        timeout: seconds to wait for health. Generous by default: the engine
            fetches the weights on first start, and tens of GB over a domestic
            connection is not a hang.
        poll: seconds between health checks.
        echo: where progress goes.

    Returns:
        An [`Endpoint`]. Call `wait()` to block until it exits, `stop()` to end it.

    Raises:
        FileNotFoundError: the engine binary is not installed.
        ChildProcessError: the engine exited before becoming healthy.
        TimeoutError: it never answered within `timeout`.
    """
    echo(f"  $ {launch_plan.command}")
    try:
        # No new session: an interactive Ctrl-C must reach the child too, or
        # stopping the launcher would leave the engine holding the GPU.
        proc = subprocess.Popen(  # noqa: S603 — argv is generated, not user text
            launch_plan.argv,
            env={**os.environ, **dict(launch_plan.env)},
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{launch_plan.argv[0]} is not installed — "
            f"{INSTALL_HINT.get(launch_plan.engine, 'install the engine first')}"
        ) from e

    started = time.monotonic()
    if not wait:
        return Endpoint(launch_plan.base, launch_plan.repo, proc.pid, 0.0, proc)

    deadline = started + timeout
    while time.monotonic() < deadline:
        if (code := proc.poll()) is not None:
            raise ChildProcessError(
                f"{launch_plan.engine} exited with code {code} before serving "
                f"{launch_plan.repo}; its output above is the reason"
            )
        # The probe's own timeout must fit inside what is LEFT of serve()'s
        # budget. It was a fixed 10s default, so `serve(timeout=2.0)` could block
        # ten seconds in the first probe alone — a caller's deadline overrun by
        # 5x by the thing measuring it.
        #
        # No floor under this. A `max(0.5, ...)` floor keeps urlopen off a
        # non-positive timeout, but it re-opens the same hole at the small end:
        # `serve(timeout=0.1)` would hand the probe 0.5s and overrun by 5x
        # again, in the other direction. Instead the loop stops when the budget
        # is gone — which is the honest reading of "it never answered in time",
        # and leaves urlopen a strictly positive timeout on every call it makes.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if _healthy(launch_plan.base, launch_plan.repo, timeout=min(10.0, remaining)):
            return Endpoint(
                launch_plan.base,
                launch_plan.repo,
                proc.pid,
                time.monotonic() - started,
                proc,
            )
        time.sleep(poll)

    ep = Endpoint(launch_plan.base, launch_plan.repo, proc.pid, timeout, proc)
    # The claim below used to be unconditional, and `stop()` only ever signalled
    # the immediate child — so for a multi-process engine it asserted a GPU had
    # been released while workers still held it. Now it is reported, not assumed.
    survivors = ep.stop()
    if survivors:
        raise TimeoutError(
            f"{launch_plan.engine} did not complete a single token at "
            f"{launch_plan.base} within {timeout:.0f}s. It was stopped, but "
            f"{len(survivors)} process(es) it spawned would not die: "
            f"{', '.join(str(p) for p in survivors)}. They may still hold the "
            f"device — kill them before starting another engine."
        )
    raise TimeoutError(
        f"{launch_plan.engine} did not complete a single token at {launch_plan.base} "
        f"within {timeout:.0f}s; it and everything it spawned have been stopped "
        f"rather than left running"
    )


def demo() -> None:
    """Self-check. Run with `python -m clickllm.launch`. Touches no network."""
    import tempfile

    m4 = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * GB,
        usable_bytes=96 * GB,
        bandwidth_gbps=546.0,
        cores=16,
    )
    llama = catalog.get("llama-3.1-8b")
    qwen = catalog.get("qwen3-30b-a3b")

    # The naming drift, which is the whole reason resolution is a search: the
    # obvious construction is a 404 and the real repo carries a prefix the base
    # repo never had.
    naive = "mlx-community/Llama-3.1-8B-Instruct-8bit"
    real = "mlx-community/Meta-Llama-3.1-8B-Instruct-8bit"
    cands = candidates(llama, "q8", "mlx")
    assert naive in cands and real in cands, cands
    assert cands.index(naive) < cands.index(real), "cheapest guess first, but both are checked"

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "weights.json"
        seen: list[str] = []

        def only_real(repo: str) -> bool:
            seen.append(repo)
            return repo == real

        got = resolve_weights(llama, "q8", "mlx", exists=only_real, cache=cache)
        assert isinstance(got, Resolved) and got.repo == real, got
        assert seen[0] == naive, "the 404 is checked, not assumed"

        # Second time is free and offline: an exists() that would explode proves it.
        def never(repo: str) -> bool:
            raise AssertionError(f"the cache should have answered; asked for {repo}")

        again = resolve_weights(llama, "q8", "mlx", exists=never, cache=cache)
        assert isinstance(again, Resolved) and again.repo == real and again.from_cache

        # Nothing exists: every name tried is reported, and none is returned.
        gone = resolve_weights(qwen, "q4", "mlx", exists=lambda r: False, cache=cache)
        assert isinstance(gone, Refusal) and len(gone.tried) >= 2, gone
        assert "mlx-community/Qwen3-30B-A3B-4bit" in gone.tried

        # vLLM takes the base repo as-is — it is not an MLX-style search.
        assert candidates(qwen, "q4", "vllm") == ("Qwen/Qwen3-30B-A3B",)

        # A whole plan, offline, on a machine this fits. 96 GB takes the 8-bit
        # weights, so that is the repo resolution has to find — precision is
        # part of the identity on MLX, not a flag.
        p = plan(
            "qwen3-30b-a3b",
            hw=m4,
            context=8192,
            concurrency=8,
            exists=lambda r: r == "mlx-community/Qwen3-30B-A3B-8bit",
            cache=cache,
        )
        assert isinstance(p, LaunchPlan), p
        assert p.quant == "q8", p.quant
        assert p.engine == "mlx" and p.repo == "mlx-community/Qwen3-30B-A3B-8bit"
        assert p.argv[:2] == ("mlx_lm.server", "--model")
        assert "--host" in p.argv and "127.0.0.1" in p.argv
        assert p.base == "http://127.0.0.1:8000"
        # MLX cannot express several of the planner's intents, and says so rather
        # than dropping them silently.
        assert p.gaps and any("quantization" in g for g in p.gaps), p.gaps
        assert "downloads the weights itself" in " ".join(p.notes)

        # Too big for the machine is a refusal that names the hosting next step.
        small = Hardware(
            kind="apple",
            name="Air",
            total_bytes=16 * GB,
            usable_bytes=12 * GB,
            bandwidth_gbps=100.0,
            cores=8,
        )
        no = plan("qwen3-235b-a22b", hw=small, cache=cache, exists=lambda r: True)
        assert isinstance(no, Refusal) and "clickllm host" in no.next_step, no
        assert "short by" in no.reason

    print(f"launch: {p.repo} -> {p.base}/v1")
    print("ok")


if __name__ == "__main__":
    demo()
