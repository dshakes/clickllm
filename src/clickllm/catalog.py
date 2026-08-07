"""Model catalog — the specs the fit solver needs, and nothing else."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "models.json"

# Bits per weight by quantization label.
QUANT_BITS = {
    "fp16": 16,
    "bf16": 16,
    "fp8": 8,
    "q8": 8,
    "q6": 6,
    "q5": 5,
    "q4": 4.5,
    "q3": 3.5,
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    name: str
    params_b: float
    active_b: float
    layers: int
    kv_heads: int
    head_dim: int
    kv_scheme: str
    max_context: int
    license: str
    license_ok: bool
    quants: tuple[str, ...]
    verified: bool
    kv_lora_rank: int | None = None
    #: Hugging Face repo. `None` when unknown — never guessed.
    repo: str | None = None

    @property
    def is_moe(self) -> bool:
        return self.active_b < self.params_b * 0.95

    def weight_bytes(self, quant: str) -> int:
        """All experts must be resident, even for MoE — sparsity saves compute, not memory."""
        return int(self.params_b * 1e9 * QUANT_BITS[quant] / 8)

    def kv_bytes_per_token(self, kv_dtype_bytes: int = 2) -> int:
        """KV cache growth per token of context.

        MLA (DeepSeek-family) compresses K and V into one low-rank latent, so it
        stores kv_lora_rank per layer instead of 2 x kv_heads x head_dim. Using the
        GQA formula on an MLA model overestimates KV by ~50x.
        """
        if self.kv_scheme == "mla" and self.kv_lora_rank:
            per_layer = self.kv_lora_rank
        else:
            per_layer = 2 * self.kv_heads * self.head_dim
        return per_layer * self.layers * kv_dtype_bytes


def sources() -> list[Path]:
    """Every catalogue file that will be read, in precedence order.

    Built-ins first, then user files, so a later entry overrides an earlier one
    with the same ``id``. That ordering is what lets someone both *add* a model
    and *correct* one of ours without touching the installed package.

    Two ways in, both chosen so an operator or an agent can extend the catalogue
    by writing a file rather than by shipping a release:

    - ``$CLICKLLM_CATALOG`` — ``os.pathsep``-separated files or directories,
      for a one-off or a CI pin.
    - ``~/.config/clickllm/models.d/*.json`` — a drop-in directory, read in
      sorted order. This is the durable one: adding a fleet is `cp` and nothing
      else, and removing it is `rm`.
    """
    found = [CATALOG_PATH]
    for entry in os.environ.get("CLICKLLM_CATALOG", "").split(os.pathsep):
        if not entry.strip():
            continue
        p = Path(entry).expanduser()
        found.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    drop_in = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "clickllm" / "models.d"
    )
    if drop_in.is_dir():
        found.extend(sorted(drop_in.glob("*.json")))
    return found


def _read(path: Path) -> list[dict]:
    """One catalogue file, with the offending file named on any failure.

    A malformed drop-in is an error, never a silent skip. Skipping it would let
    a typo quietly remove a model from the catalogue, and the first symptom
    would be a sizing answer that omits it with no explanation.
    """
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as e:
        raise FileNotFoundError(f"catalogue file not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON — {e}") from e
    models = raw.get("models") if isinstance(raw, dict) else raw
    if not isinstance(models, list):
        raise ValueError(f"{path}: expected a 'models' list (or a bare JSON list of models)")
    return models


#: Parsed catalogue, keyed by the files that produced it and their mtimes.
#: Not an ``lru_cache``: the key has to include what is on disk, or a long-lived
#: process — the MCP server, the workbench — would keep serving the catalogue it
#: read at import and a newly dropped-in model would need a restart to appear.
#: Dropping a file in and having it work is the whole point of the mechanism.
_CACHE: dict[tuple, tuple[ModelSpec, ...]] = {}


def _fingerprint(paths: list[Path]) -> tuple:
    """Identity of the catalogue on disk right now: paths plus mtimes and sizes."""
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))  # missing: _read raises with the name
    return tuple(out)


#: Quantisation labels the solver knows how to size. A label outside this set
#: reaches `QUANT_BITS` as a KeyError deep in the solver, naming neither the
#: model nor the file it came from.
_KNOWN_QUANTS = frozenset(QUANT_BITS)
_KNOWN_KV_SCHEMES = frozenset({"mha", "gqa", "mla"})


#: field -> (accepted types, must be > 0). The dataclass does not coerce and
#: JSON does not type-check, so without this a row loads as whatever the file
#: happened to contain. `bool` is listed before `int` deliberately: in Python
#: `isinstance(True, int)` is True, so the boolean must be matched first.
_TYPES: dict[str, tuple[tuple[type, ...], bool]] = {
    "id": ((str,), False),
    "name": ((str,), False),
    "params_b": ((int, float), True),
    "active_b": ((int, float), True),
    "layers": ((int,), True),
    "kv_heads": ((int,), True),
    "head_dim": ((int,), True),
    "kv_scheme": ((str,), False),
    "max_context": ((int,), True),
    "license": ((str,), False),
    "license_ok": ((bool,), False),
    "verified": ((bool,), False),
    "kv_lora_rank": ((int,), True),
    "repo": ((str,), False),
}


def _check_types(spec: ModelSpec, path: Path) -> None:
    """Refuse a row whose fields are the wrong type or an impossible size.

    The dataclass accepts whatever JSON contained, and two of these are wrong in
    the flattering direction rather than the loud one:

    - `"license_ok": "false"` — a quoted boolean, which JSON and Python spell
      differently enough to invite — is the string "false", and every non-empty
      string is truthy. A model the author meant to mark restricted reports as
      commercially clean.
    - `"params_b": "70"` loads fine here and fails much later inside
      `weight_bytes()` as a bare TypeError, with none of the file-and-model
      attribution this module otherwise attaches.

    Sizes are checked as well as types: `layers: 0` and `params_b: -5` both
    described a model that cannot exist, and produced a weight figure to match.
    """
    for field, (types, positive) in _TYPES.items():
        value = getattr(spec, field)
        if value is None and field in ("kv_lora_rank", "repo"):
            continue
        if type(value) not in types:  # not isinstance: bool is a subclass of int
            want = " or ".join(t.__name__ for t in types)
            raise ValueError(
                f"{path}: model {spec.id!r} has {field}={value!r} "
                f"({type(value).__name__}), which is not {want}"
            )
        if isinstance(value, float) and not math.isfinite(value):
            # On the VALUE, not on the spelling. `parse_constant` catches the
            # bare NaN/Infinity literals and nothing else — "1e400" is ordinary
            # JSON that parses to inf through the normal float path — and a NaN
            # slips the `<= 0` test below, since every comparison with NaN is
            # False. Neither survives sizing: Infinity raises OverflowError
            # inside weight_bytes() and NaN raises ValueError, both as
            # tracebacks out of a CLI that promises a message.
            raise ValueError(
                f"{path}: model {spec.id!r} has {field}={value!r}, which is not "
                f"a number a model can be sized with"
            )
        if positive and value <= 0:
            raise ValueError(
                f"{path}: model {spec.id!r} has {field}={value!r}; it describes "
                f"a model that cannot exist, and sizes one to match"
            )
    if not spec.quants:
        raise ValueError(f"{path}: model {spec.id!r} lists no quantisations")
    if bad := [q for q in spec.quants if not isinstance(q, str)]:
        raise ValueError(f"{path}: model {spec.id!r} has non-string quants {bad!r}")


def _check_invariants(spec: ModelSpec, path: Path) -> None:
    """Refuse an entry the solver would size wrongly, naming the file.

    CLAUDE.md makes the MLA rank load-bearing: "any new catalog entry with
    `kv_scheme: mla` MUST set `kv_lora_rank`". That was enforced by a test which
    parametrises over `load()` at collection time — so it only ever sees the
    built-in `models.json`, and never an entry supplied through
    `CLICKLLM_CATALOG` or the `models.d` drop-in directory, which `sources()`
    documents as the first-class way to add a model.

    An MLA entry with no rank is sized with the GQA formula and overestimates KV
    by ~50x. That is a wrong number, not a crash, so it surfaces as a model that
    "does not fit" on hardware that would hold it — or worse, the reverse.

    Enforced here because `load()` is the funnel every catalogue passes through,
    built-in or not: ADR-0011's rule, applied to the catalogue. Refusing names
    the file and the model, because "missing kv_lora_rank" is unactionable when
    six drop-ins are in play.
    """
    if spec.kv_scheme not in _KNOWN_KV_SCHEMES:
        raise ValueError(
            f"{path}: model {spec.id!r} has kv_scheme {spec.kv_scheme!r}, which the "
            f"solver cannot size. Known: {', '.join(sorted(_KNOWN_KV_SCHEMES))}"
        )
    if spec.kv_scheme == "mla" and not spec.kv_lora_rank:
        raise ValueError(
            f"{path}: model {spec.id!r} declares kv_scheme 'mla' but no "
            f"kv_lora_rank. MLA stores a compressed latent; sizing it with the "
            f"GQA formula overestimates KV by ~50x, so it is refused rather "
            f"than guessed."
        )
    if unknown := set(spec.quants) - _KNOWN_QUANTS:
        raise ValueError(
            f"{path}: model {spec.id!r} lists quantisation(s) "
            f"{', '.join(sorted(unknown))} the solver has no bit-width for. "
            f"Known: {', '.join(sorted(_KNOWN_QUANTS))}"
        )


def load() -> tuple[ModelSpec, ...]:
    """The catalogue: built-ins plus anything dropped in. See [`sources`].

    Re-reads whenever a catalogue file changes, so `clickllm catalog-add` takes
    effect immediately — including inside an already-running MCP server.
    """
    paths = sources()
    key = _fingerprint(paths)
    if (hit := _CACHE.get(key)) is not None:
        return hit

    fields = ModelSpec.__slots__
    by_id: dict[str, ModelSpec] = {}
    for path in paths:
        for m in _read(path):
            if not isinstance(m, dict) or "id" not in m:
                raise ValueError(f"{path}: every model needs an 'id'; got {m!r}")
            # A key ModelSpec does not have is a typo, not a comment. Dropped
            # silently, `"kv_lora_rank_": 512` looks set and is not — which is
            # the easiest way to reach the MLA defect below through a door the
            # MLA check cannot see, since the field keeps its default.
            if extra := sorted(set(m) - set(fields)):
                raise ValueError(
                    f"{path}: model {m['id']!r} has unknown field(s) "
                    f"{', '.join(extra)}. Known: {', '.join(sorted(fields))}"
                )
            try:
                spec = ModelSpec(
                    **{k: (tuple(v) if k == "quants" else v) for k, v in m.items() if k in fields}
                )
            except TypeError as e:
                # Name the file and the model: "missing kv_heads" is unactionable
                # when six drop-ins are in play.
                raise ValueError(f"{path}: model {m['id']!r} is not a valid spec — {e}") from e
            _check_types(spec, path)
            _check_invariants(spec, path)
            by_id[spec.id] = spec  # later file wins, by design

    out = tuple(by_id.values())
    # Bounded: one entry per distinct on-disk state, and a handful of states is
    # all a process ever sees. Cleared wholesale rather than evicted per-key.
    if len(_CACHE) > 8:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def get(model_id: str) -> ModelSpec:
    for m in load():
        if m.id == model_id:
            return m
    raise KeyError(f"unknown model: {model_id}")


def demo() -> None:
    models = load()
    assert len(models) > 5

    q3 = get("qwen3-32b")
    assert not q3.is_moe
    # 32.8B at q4 (4.5 bits) ~= 18.5 GB
    assert 17e9 < q3.weight_bytes("q4") < 20e9, q3.weight_bytes("q4")
    # fp16 must be ~3.5x the q4 size
    assert abs(q3.weight_bytes("fp16") / q3.weight_bytes("q4") - 16 / 4.5) < 1e-6

    moe = get("qwen3-30b-a3b")
    assert moe.is_moe
    # MoE weights track TOTAL params, not active — the common sizing error.
    assert moe.weight_bytes("q4") > 15e9, "MoE must size on total params"

    # GQA vs MLA: MLA is dramatically smaller per token.
    gqa = get("qwen3-32b").kv_bytes_per_token()
    mla = get("deepseek-v3").kv_bytes_per_token()
    assert mla < gqa, f"MLA {mla} should be far below GQA {gqa}"
    # 64 layers x 2 x 8 heads x 128 dim x 2 bytes = 262144 B/token
    assert gqa == 64 * 2 * 8 * 128 * 2, gqa

    print(
        f"{len(models)} models · qwen3-32b q4 = {q3.weight_bytes('q4') / 1024**3:.1f} GB · "
        f"kv/token gqa={gqa}B mla={mla}B"
    )
    print("ok")


if __name__ == "__main__":
    demo()
