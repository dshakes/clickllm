"""Detect the accelerator and memory actually available for inference.

The number that matters is *usable* memory, not installed memory. On Apple
Silicon the GPU shares unified memory with the OS, so the wired limit — not
hw.memsize — is the real ceiling.
"""

from __future__ import annotations

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
    "M3 Max": 400,
    "M3 Ultra": 800,
    "M4": 120,
    "M4 Pro": 273,
    "M4 Max": 546,
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


def _detect_apple() -> Hardware | None:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    brand = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    total = int(_sysctl("hw.memsize") or 0)
    cores = int(_sysctl("hw.ncpu") or 0)
    if not total:
        return None

    chip = _apple_chip(brand)
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
        bandwidth_gbps=APPLE_BANDWIDTH.get(chip),
        cores=cores,
        note=note,
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
    name = gpus[0][0].strip()
    per_gpu = int(float(gpus[0][1])) * 1024 * 1024  # MiB -> bytes
    total = per_gpu * len(gpus)
    return Hardware(
        kind="nvidia",
        name=name,
        total_bytes=total,
        # Engines reserve headroom; vLLM's default gpu-memory-utilization is 0.90.
        usable_bytes=int(total * 0.90),
        bandwidth_gbps=None,
        cores=0,
        devices=len(gpus),
        note="assumes gpu-memory-utilization=0.90"
        + (f"; {len(gpus)}× {name} — tensor parallelism required" if len(gpus) > 1 else ""),
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
    import os

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
    """First accelerator found wins: Apple > NVIDIA > CPU."""
    return _detect_apple() or _detect_nvidia() or _detect_cpu()


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
