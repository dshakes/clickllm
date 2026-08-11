"""Detect the accelerator and memory actually available for inference.

The number that matters is *usable* memory, not installed memory. On Apple
Silicon the GPU shares unified memory with the OS, so the wired limit — not
hw.memsize — is the real ceiling.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from typing import Literal

Kind = Literal["apple", "nvidia", "amd", "tpu", "cpu"]

GB = 1024**3

# Memory bandwidth (GB/s) per Apple chip. Decode is bandwidth-bound, so this
# drives the tok/s roofline. Apple exposes no sysctl for it — hence the table.
# ponytail: lookup table, replace with a measured microbenchmark if estimates drift.
#: Chips Apple ships in more than one memory-bandwidth bin, resolved by memory
#: size. `machdep.cpu.brand_string` returns "M3 Max" for both the 30-core-GPU
#: part at 300 GB/s and the 40-core part at 400, so the brand string alone
#: cannot tell them apart — and the table used to carry only the top bin,
#: overstating the roofline by up to 33% on the base SKU.
#:
#: But the machine is not actually ambiguous: Apple sells each memory capacity
#: on exactly one bin, and `hw.memsize` reports it. So this is a lookup keyed
#: on GB, never a threshold — 96 GB is the trap, being the *low* bin on M3 Max
#: while 48/64/128 GB are the high one. A capacity Apple does not sell with the
#: chip falls through to the low bin and says so, since assuming low understates
#: rather than flatters.
#:
#: Verified against Apple's published tech specs — MacBook Pro (14/16-inch,
#: Nov 2023) for M3 Max and (Oct 2024) for M4 Max — never recalled.
APPLE_MAX_BINS: dict[str, dict[int, int]] = {
    "M3 Max": {36: 300, 48: 400, 64: 400, 96: 300, 128: 400},
    "M4 Max": {36: 410, 48: 546, 64: 546, 128: 546},
}

APPLE_BANDWIDTH = {
    "M1": 68,
    "M1 Pro": 200,
    "M1 Max": 400,
    "M1 Ultra": 800,
    "M2": 100,
    "M2 Pro": 200,
    "M2 Max": 400,
    "M2 Ultra": 800,
    "M3": 100,
    "M3 Pro": 150,
    "M3 Max": 300,  # low bin; memory size picks the real one — see APPLE_MAX_BINS
    "M3 Ultra": 800,
    "M4": 120,
    "M4 Pro": 273,
    "M4 Max": 410,  # low bin; memory size picks the real one — see APPLE_MAX_BINS
    "M5": 153,
    "M5 Pro": 344,
    "M5 Max": 688,
}

# Fraction of unified memory macOS will let the GPU wire down by default.
# Raisable via `sudo sysctl iogpu.wired_limit_mb=<N>`.
APPLE_DEFAULT_WIRED_FRACTION = 0.75
APPLE_MAX_WIRED_FRACTION = 0.92


@dataclass(frozen=True, slots=True)
class Hardware:
    kind: Kind
    name: str
    total_bytes: int
    usable_bytes: int
    bandwidth_gbps: float | None
    cores: int
    devices: int = 1
    note: str = ""

    @property
    def total_gb(self) -> float:
        return self.total_bytes / GB

    @property
    def usable_gb(self) -> float:
        return self.usable_bytes / GB

    def to_dict(self) -> dict:
        return asdict(self) | {
            "total_gb": round(self.total_gb, 1),
            "usable_gb": round(self.usable_gb, 1),
        }


def _sysctl(key: str) -> str | None:
    try:
        return (
            subprocess.run(
                ["sysctl", "-n", key], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _apple_chip(brand: str) -> str:
    """'Apple M4 Max' -> 'M4 Max'."""
    m = re.search(r"(M\d+(?:\s+(?:Pro|Max|Ultra))?)", brand)
    return m.group(1) if m else brand


def _apple_bandwidth(chip: str, total_bytes: int) -> tuple[float | None, str]:
    """Bandwidth for a chip, plus any caveat about how certain that figure is.

    The caveat is empty when the figure is known rather than assumed, because a
    note saying "this could be wrong" on a number that cannot be wrong trains
    people to ignore the ones that can.
    """
    bins = APPLE_MAX_BINS.get(chip)
    if bins is None:
        return APPLE_BANDWIDTH.get(chip), ""
    gb = round(total_bytes / GB)
    if gb in bins:
        return float(bins[gb]), ""
    offered = "/".join(str(v) for v in sorted(set(bins.values())))
    return float(min(bins.values())), (
        f"; {chip} ships in {offered} GB/s bins tied to memory size, and {gb} GB "
        f"is not a capacity Apple pairs with it — the lower is assumed, so a "
        f"roofline here understates rather than flatters"
    )


def _detect_apple() -> Hardware | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    brand = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    total = int(_sysctl("hw.memsize") or 0)
    cores = int(_sysctl("hw.ncpu") or 0)
    if not total:
        return None

    chip = _apple_chip(brand)
    bandwidth, bin_note = _apple_bandwidth(chip, total)
    # Respect an explicitly raised wired limit if the user set one.
    wired_mb = _sysctl("iogpu.wired_limit_mb")
    if wired_mb and wired_mb.isdigit() and int(wired_mb) > 0:
        usable = min(int(wired_mb) * 1024 * 1024, int(total * APPLE_MAX_WIRED_FRACTION))
        note = f"iogpu.wired_limit_mb={wired_mb} is set"
    else:
        usable = int(total * APPLE_DEFAULT_WIRED_FRACTION)
        headroom = int(total * APPLE_MAX_WIRED_FRACTION) / GB
        note = (
            f"raise to ~{headroom:.0f} GB with: "
            f"sudo sysctl iogpu.wired_limit_mb={int(headroom * 1024)}"
        )

    return Hardware(
        kind="apple",
        name=chip,
        total_bytes=total,
        usable_bytes=usable,
        bandwidth_gbps=bandwidth,
        cores=cores,
        note=note + bin_note,
    )


def _detect_nvidia() -> Hardware | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None

    gpus = [ln.split(",") for ln in out.splitlines() if ln.strip()]
    if not gpus:
        return None
    # Every card, not GPU 0 multiplied by the count. `nvidia-smi` prints one
    # line per physical device and they need not match: a rig with one A100
    # 80 GB in slot 0 and three 3090s at 24 GB reported 4 x 80 = 320 GB, when it
    # holds 152 GB — a 2x overstatement in the direction that says "it fits".
    #
    # Tensor parallelism shards evenly, so the SMALLEST card is what actually
    # bounds a sharded model; the aggregate is what bounds the sum of shards.
    # Both are reported, and the note says so, because a heterogeneous rig sized
    # against its largest card is how someone OOMs on the third GPU.
    names = [g[0].strip() for g in gpus]
    sizes = [int(float(g[1])) * 1024 * 1024 for g in gpus]  # MiB -> bytes
    total = sum(sizes)
    mixed = len(set(names)) > 1
    name = names[0] if not mixed else f"{names[0]} + {len(names) - 1} other(s)"
    smallest = min(sizes)
    return Hardware(
        kind="nvidia",
        name=name,
        total_bytes=total,
        # Engines reserve headroom; vLLM's default gpu-memory-utilization is 0.90.
        #
        # On a MIXED rig the usable figure is the smallest card times the count,
        # not the sum. Tensor parallelism shards evenly, so a model is bounded
        # by the smallest device — the note said exactly that while the
        # arithmetic summed anyway, and downstream planning divides usable by
        # `devices` to size a shard. A 192+64 GiB pair reported ~115 GiB per
        # device and picked shards the 64 GiB card cannot hold.
        usable_bytes=int((smallest * len(gpus) if mixed else total) * 0.90),
        bandwidth_gbps=None,
        # Host CPU cores, not GPU cores — this is the denominator `measure.py`
        # uses for load-per-core, and load average is always a host-CPU metric.
        # Hardcoded 0 here used to mean "not tracked" and instead read as "0
        # cores", which crashed `Load.render()` and silently disabled the load
        # gate on every NVIDIA box.
        cores=os.cpu_count() or 0,
        devices=len(gpus),
        note="assumes gpu-memory-utilization=0.90"
        + (f"; {len(gpus)}× {name} — tensor parallelism required" if len(gpus) > 1 else "")
        + (
            f"; MIXED cards ({', '.join(sorted(set(names)))}) — usable is the smallest "
            f"at {smallest / 1024**3:.0f} GiB × {len(gpus)}, not the "
            f"{total / 1024**3:.0f} GiB aggregate — tensor parallelism shards evenly"
            if mixed
            else ""
        ),
    )


def _detect_amd() -> Hardware | None:
    """AMD GPUs via `rocm-smi`, or None.

    "amd" is a declared `Kind`, `hardware_catalog` carries an MI300X profile,
    and `box.py` emits `amd.com/gpu` and a `rocm/vllm` image — every downstream
    piece expects these to be detectable, and nothing detected them. A ROCm box
    reported as CPU-only, which sizes a 192 GB accelerator against host RAM.

    `--showmeminfo vram --json` is the stable machine-readable surface; the
    human table has been reformatted repeatedly across ROCm releases. Anything
    unexpected returns None rather than a guess: this file's contract is that a
    number it reports is one it read.
    """
    # rocm-smi only. `amd-smi` was here as a fallback and it takes a different
    # CLI entirely — invoking it with these flags fails, so the fallback did
    # nothing but look like support. A dead branch that reads as coverage is
    # worse than an honest gap, and an amd-smi-only host is better served by
    # nothing here than by something that silently declines.
    smi = shutil.which("rocm-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        cards = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(cards, dict):
        return None

    sizes: list[int] = []
    names: list[str] = []
    for key, card in cards.items():
        if not isinstance(card, dict):
            continue
        # `rocm-smi --json` keys every physical device `card0`, `card1`, … and
        # everything else is metadata (a `system` block, timestamps). That is
        # the only reliable way to tell "this entry is not a card" from "this
        # entry IS a card whose VRAM key I no longer recognise" — and the skip
        # below could not distinguish them, so a ROCm release renaming the key
        # for one card on a 4-card box reported a 3-card machine. It understates
        # capacity, which is the safe direction, and it is silent, which is not.
        is_card = key.lower().startswith("card")
        # Key spelling has moved between ROCm versions; match on shape.
        # "VRAM Total Used Memory (B)" also contains "vram" and "total", so a
        # broad match could read used VRAM as installed VRAM depending on key
        # order — reporting a busy card as a small one, or an idle one as
        # right-sized. Capacity only.
        total = next(
            (
                v
                for k, v in card.items()
                if "vram" in k.lower() and "total" in k.lower() and "used" not in k.lower()
            ),
            None,
        )
        if total is None:
            if is_card:
                # Fail closed. Declining detection sends the caller to the CPU
                # path with nothing invented; reporting three of four cards
                # hands them a machine that does not exist.
                return None
            continue
        try:
            sizes.append(int(str(total).strip()))
        except ValueError:
            return None  # a VRAM figure we cannot read is not a VRAM figure
        # rocm-smi says "Card Series", amd-smi says "market_name", older
        # builds said "Card model". Matched on shape rather than on the one
        # spelling I happened to look at — the first version of this checked
        # only "product name" and named every card "AMD GPU".
        names.append(
            str(
                next(
                    (
                        v
                        for k, v in card.items()
                        if any(t in k.lower() for t in ("series", "market", "product", "model"))
                    ),
                    "AMD GPU",
                )
            ).strip()
        )
    if not sizes:
        return None

    total_bytes = sum(sizes)
    mixed = len(set(names)) > 1
    name = names[0] if not mixed else f"{names[0]} + {len(names) - 1} other(s)"
    return Hardware(
        kind="amd",
        name=name,
        total_bytes=total_bytes,
        # Same reservation as the NVIDIA path, and the same mixed-rig rule:
        # what a sharded model can use is the smallest card times the count.
        usable_bytes=int((min(sizes) * len(sizes) if mixed else total_bytes) * 0.90),
        bandwidth_gbps=None,
        # See the matching comment in `_detect_nvidia`: host CPU cores, not GPU
        # cores.
        cores=os.cpu_count() or 0,
        devices=len(sizes),
        note="assumes gpu-memory-utilization=0.90"
        + (f"; {len(sizes)}× {name} — tensor parallelism required" if len(sizes) > 1 else "")
        + (
            f"; MIXED cards ({', '.join(sorted(set(names)))}) — usable is the smallest "
            f"at {min(sizes) / 1024**3:.0f} GiB × {len(sizes)}, not the aggregate"
            if mixed
            else ""
        ),
    )


def _detect_cpu() -> Hardware:
    total = 0
    if platform.system() == "Darwin":
        total = int(_sysctl("hw.memsize") or 0)
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                        break
        except OSError:
            pass

    return Hardware(
        kind="cpu",
        name=platform.processor() or platform.machine(),
        total_bytes=total,
        usable_bytes=int(total * 0.70),
        bandwidth_gbps=None,
        cores=os.cpu_count() or 0,
        note="no accelerator detected — expect single-digit tok/s",
    )


def detect() -> Hardware:
    """First accelerator found wins: Apple > NVIDIA > AMD > CPU."""
    return _detect_apple() or _detect_nvidia() or _detect_amd() or _detect_cpu()


def demo() -> None:
    assert _apple_chip("Apple M4 Max") == "M4 Max"
    assert _apple_chip("Apple M1") == "M1"
    assert _apple_chip("Apple M3 Ultra") == "M3 Ultra"
    assert _apple_chip("AMD Ryzen 9") == "AMD Ryzen 9"  # falls through unchanged

    hw = detect()
    assert hw.usable_bytes <= hw.total_bytes, "usable must not exceed total"
    assert hw.kind in ("apple", "nvidia", "amd", "cpu")
    print(
        f"{hw.kind}: {hw.name} · {hw.total_gb:.0f} GB total · "
        f"{hw.usable_gb:.0f} GB usable · {hw.bandwidth_gbps or '?'} GB/s"
    )
    print("ok")


if __name__ == "__main__":
    demo()
