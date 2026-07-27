"""Model catalog — the specs the fit solver needs, and nothing else."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=1)
def load() -> tuple[ModelSpec, ...]:
    raw = json.loads(CATALOG_PATH.read_text())
    fields = ModelSpec.__slots__
    out = []
    for m in raw["models"]:
        out.append(
            ModelSpec(
                **{k: (tuple(v) if k == "quants" else v) for k, v in m.items() if k in fields}
            )
        )
    return tuple(out)


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
