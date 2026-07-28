"""Named hardware classes, so a model can be matched to machines you don't own.

`fit` answers "what runs on *this* box". This answers the inverse — "what box do
I need for *this* model" — which is the question asked when specifying a build,
sizing a cloud instance, or deciding whether a model is reachable at all.

Memory figures are **usable**, not nameplate. A device advertises 80 GB; the
engine reserves some, so planning against 80 gets you an OOM on the first long
request. Apple's unified memory is worse for this: the OS shares it, so the
default wired limit is ~75%.

Bandwidth drives the throughput roofline, and is the number that actually
separates these devices — an H100 is not "faster at maths" than a 4090, it feeds
its arithmetic units three times quicker.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import Hardware

GB = 1024**3

#: Per-additional-device bandwidth scaling under tensor parallelism. Below 1.0
#: because each layer ends in an all-reduce across the interconnect.
#: ponytail: single constant; split by interconnect (NVLink vs PCIe) if the
#: estimate drifts from measurement.
TP_SCALING_EFFICIENCY = 0.85


@dataclass(frozen=True, slots=True)
class Profile:
    """A hardware class you could deploy on."""

    id: str
    name: str
    kind: str
    #: Nameplate memory per device, in GB.
    memory_gb: float
    #: Peak memory bandwidth per device, GB/s.
    bandwidth_gbps: float
    devices: int = 1
    #: Rough hourly rate to rent, USD. `None` where it isn't rented by the hour
    #: (a Mac you buy) or where prices move too fast to state usefully.
    hourly_usd: float | None = None
    note: str = ""

    @property
    def total_memory_gb(self) -> float:
        return self.memory_gb * self.devices

    @property
    def effective_bandwidth_gbps(self) -> float:
        """Aggregate decode bandwidth across the devices.

        Under tensor parallelism each device holds 1/N of the weights and reads
        its shard in parallel, so bandwidth adds — this is the entire reason
        multi-GPU is how large models are served quickly. It does not add
        perfectly: every layer ends in an all-reduce, which costs the scaling
        efficiency below.

        Treating a 4-GPU node as one device's bandwidth understates it ~3.5x and
        makes multi-GPU look pointless, which is wrong in a way that would send
        someone to the wrong hardware.
        """
        if self.devices <= 1:
            return self.bandwidth_gbps
        return self.bandwidth_gbps * (1 + (self.devices - 1) * TP_SCALING_EFFICIENCY)

    def to_hardware(self) -> Hardware:
        """As a :class:`~clickllm.hardware.Hardware` the fit solver can use.

        Usable memory follows the same reservations the live detector applies, so
        a planned answer and a measured one agree on the same machine.
        """
        frac = 0.75 if self.kind == "apple" else 0.90 if self.kind != "cpu" else 0.70
        total = int(self.total_memory_gb * GB)
        return Hardware(
            kind=self.kind,  # type: ignore[arg-type]
            name=self.name,
            total_bytes=total,
            usable_bytes=int(total * frac),
            bandwidth_gbps=self.effective_bandwidth_gbps,
            cores=0,
            devices=self.devices,
            note=self.note,
        )


#: Hardware people actually deploy on, smallest first. Bandwidth and capacity are
#: vendor-published; hourly rates are indicative and move constantly.
PROFILES: tuple[Profile, ...] = (
    Profile(
        "cpu-64",
        "CPU only (64 GB DDR5)",
        "cpu",
        64,
        60,
        note="no accelerator — expect single-digit tok/s on anything large",
    ),
    # --- TPU ------------------------------------------------------------------
    # Specs verified against Google Cloud's per-generation pages (2026-07-27):
    #   v5e   16 GB HBM · 800 GiB/s ·  8 chips/host   docs.cloud.google.com/tpu/docs/v5e
    #   v6e   32 GB HBM · 1638 GB/s ·  8 chips/host   docs.cloud.google.com/tpu/docs/v6e
    #   v5p   95 GiB HBM · 2765 GB/s · 4 chips/host   docs.cloud.google.com/tpu/docs/v5p
    #
    # Modelled per *host*, not per chip, because a TPU is not rented by the chip
    # and a single v6e chip's 32 GB would misrepresent what you can actually
    # deploy on. `devices` carries the chip count so the tensor-parallel
    # bandwidth model applies — ICI is fast but still sub-linear.
    Profile(
        "tpu-v5e-8",
        "TPU v5e (8 chips)",
        "tpu",
        16,
        859,  # 800 GiB/s expressed in GB/s, to match every other row here
        devices=8,
        hourly_usd=1.20,
        note="cheapest TPU per unit memory; 16 GB a chip means sharding early",
    ),
    Profile(
        "tpu-v6e-8",
        "TPU v6e Trillium (8 chips)",
        "tpu",
        32,
        1638,
        devices=8,
        hourly_usd=2.70,
        note="the volume inference TPU; 256 GB a host without leaving one machine",
    ),
    Profile(
        "tpu-v5p-4",
        "TPU v5p (4 chips)",
        "tpu",
        95,
        2765,
        devices=4,
        hourly_usd=4.20,
        note="highest bandwidth per chip, but vLLM lists v5p as experimental",
    ),
    Profile("m4-pro-48", "Apple M4 Pro 48 GB", "apple", 48, 273),
    Profile("rtx-4090", "NVIDIA RTX 4090", "nvidia", 24, 1008, hourly_usd=0.35),
    Profile(
        "l4",
        "NVIDIA L4",
        "nvidia",
        24,
        300,
        hourly_usd=0.28,
        note="low bandwidth for its capacity — capable but slow to decode",
    ),
    Profile("rtx-5090", "NVIDIA RTX 5090", "nvidia", 32, 1792, hourly_usd=0.55),
    Profile("a100-40", "NVIDIA A100 40 GB", "nvidia", 40, 1555, hourly_usd=1.10),
    Profile("l40s", "NVIDIA L40S", "nvidia", 48, 864, hourly_usd=0.80),
    Profile("a100-80", "NVIDIA A100 80 GB", "nvidia", 80, 2039, hourly_usd=1.80),
    Profile("h100", "NVIDIA H100 80 GB SXM", "nvidia", 80, 3350, hourly_usd=2.90),
    Profile("m4-max-128", "Apple M4 Max 128 GB", "apple", 128, 546),
    Profile("h200", "NVIDIA H200 141 GB", "nvidia", 141, 4800, hourly_usd=3.60),
    Profile("mi300x", "AMD MI300X 192 GB", "amd", 192, 5300, hourly_usd=2.50),
    Profile("b200", "NVIDIA B200 192 GB", "nvidia", 192, 8000, hourly_usd=5.50),
    Profile("h100-x2", "2× NVIDIA H100", "nvidia", 80, 3350, devices=2, hourly_usd=5.80),
    Profile(
        "m3-ultra-512",
        "Apple M3 Ultra 512 GB",
        "apple",
        512,
        819,
        note="the largest unified-memory box you can buy; slower per byte than an H100",
    ),
    Profile("h100-x4", "4× NVIDIA H100", "nvidia", 80, 3350, devices=4, hourly_usd=11.60),
    Profile("h200-x8", "8× NVIDIA H200", "nvidia", 141, 4800, devices=8, hourly_usd=28.80),
)


def get(profile_id: str) -> Profile:
    for p in PROFILES:
        if p.id == profile_id:
            return p
    raise KeyError(f"unknown hardware profile: {profile_id}")


def demo() -> None:
    assert len({p.id for p in PROFILES}) == len(PROFILES), "profile ids must be unique"

    for p in PROFILES:
        hw = p.to_hardware()
        assert hw.usable_bytes < hw.total_bytes, f"{p.id} must reserve headroom"
        assert hw.bandwidth_gbps and hw.bandwidth_gbps > 0
        assert hw.devices >= 1

    # Multi-device profiles aggregate capacity AND bandwidth. Treating a 4-GPU
    # node as one device's bandwidth understates decode ~3.5x.
    one, four = get("h100"), get("h100-x4")
    assert four.total_memory_gb == 320
    assert four.to_hardware().devices == 4
    assert four.effective_bandwidth_gbps > one.effective_bandwidth_gbps * 3
    assert four.effective_bandwidth_gbps < one.effective_bandwidth_gbps * 4, (
        "all-reduce overhead means scaling is sub-linear"
    )
    assert one.effective_bandwidth_gbps == one.bandwidth_gbps

    # Apple reserves more than a discrete GPU, because the OS shares the pool.
    apple = get("m4-max-128").to_hardware()
    nvidia = get("h100").to_hardware()
    assert apple.usable_bytes / apple.total_bytes < nvidia.usable_bytes / nvidia.total_bytes

    # Capacity and speed are different axes: the L4 has 4090 capacity at a third
    # of the bandwidth, which is exactly the trap this catalogue exists to expose.
    assert get("l4").memory_gb == get("rtx-4090").memory_gb
    assert get("l4").bandwidth_gbps < get("rtx-4090").bandwidth_gbps / 3

    try:
        get("nope")
    except KeyError:
        pass
    else:  # pragma: no cover
        raise AssertionError("unknown profile must raise")

    print(f"{len(PROFILES)} hardware profiles, {get('h200-x8').total_memory_gb:.0f} GB at the top")
    print("ok")


if __name__ == "__main__":
    demo()
