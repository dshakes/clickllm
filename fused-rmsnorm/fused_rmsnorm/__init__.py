"""fused-rmsnorm — a vllm.general_plugins plugin."""

from __future__ import annotations


def register():
    """Register the op. Called once per process — must be re-entrant.

    vLLM loads plugins in every process it spawns, so under tensor
    parallelism this runs once per worker. A register() that appends
    to a list or raises on a second call breaks the moment TP > 1,
    and the error surfaces far from the cause.
    """
    import torch

    if hasattr(torch.ops, "my_kernels"):
        return  # already registered in this process
    torch.ops.load_library(_library_path())

