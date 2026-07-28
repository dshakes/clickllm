"""Model catalog — the specs the fit solver needs, and nothing else."""

from __future__ import annotations

import json
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
            try:
                spec = ModelSpec(
                    **{k: (tuple(v) if k == "quants" else v) for k, v in m.items() if k in fields}
                )
            except TypeError as e:
                # Name the file and the model: "missing kv_heads" is unactionable
                # when six drop-ins are in play.
                raise ValueError(f"{path}: model {m['id']!r} is not a valid spec — {e}") from e
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
