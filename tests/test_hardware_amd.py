"""AMD/ROCm detection, and the Apple bins the brand string cannot tell apart.

`detect()` is the one function here that reads the real machine, so these
drive it through a fake `rocm-smi` on PATH rather than mocking the parser —
the defect was that nothing called a parser at all.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from clickllm import hardware

CARDS = {
    "card0": {"GPU ID": "0x74a1", "Card Series": "MI300X", "VRAM Total Memory (B)": "206158430208"},
    "card1": {"GPU ID": "0x74a1", "Card Series": "MI300X", "VRAM Total Memory (B)": "206158430208"},
}


def fake_smi(tmp_path, monkeypatch, payload: str, name: str = "rocm-smi"):
    exe = tmp_path / name
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{payload}\nEOF\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", str(tmp_path), prepend=os.pathsep)
    return exe


def test_a_rocm_box_is_not_reported_as_cpu_only(tmp_path, monkeypatch):
    # "amd" is a declared Kind, hardware_catalog carries an MI300X profile and
    # box.py emits amd.com/gpu — everything downstream expected these to be
    # detectable, and nothing detected them.
    fake_smi(tmp_path, monkeypatch, json.dumps(CARDS))
    hw = hardware._detect_amd()
    assert hw is not None
    assert hw.kind == "amd" and hw.devices == 2
    assert hw.total_bytes == 2 * 206158430208
    assert hw.usable_bytes == int(hw.total_bytes * 0.90)


def test_mixed_cards_are_summed_and_the_smallest_is_named(tmp_path, monkeypatch):
    mixed = {
        "card0": {"Card Series": "MI300X", "VRAM Total Memory (B)": "206158430208"},
        "card1": {"Card Series": "MI210", "VRAM Total Memory (B)": "68719476736"},
    }
    fake_smi(tmp_path, monkeypatch, json.dumps(mixed))
    hw = hardware._detect_amd()
    assert hw.total_bytes == 206158430208 + 68719476736
    assert "MIXED" in hw.note
    # The arithmetic has to agree with the note. Summed, usable/devices was
    # ~115 GiB per card and the planner picked shards the 64 GiB card cannot
    # hold — tensor parallelism shards evenly, so the smallest binds.
    assert hw.usable_bytes == int(68719476736 * 2 * 0.90)
    assert hw.usable_bytes // hw.devices <= 68719476736


@pytest.mark.parametrize(
    "payload",
    ["not json at all", "[]", "{}", json.dumps({"card0": {"Card Series": "MI300X"}})],
)
def test_output_it_cannot_read_is_none_rather_than_a_guess(tmp_path, monkeypatch, payload):
    # This file's contract is that a number it reports is one it read. ROCm has
    # reformatted its human table repeatedly, so anything unexpected declines.
    fake_smi(tmp_path, monkeypatch, payload)
    assert hardware._detect_amd() is None


def test_an_unreadable_vram_figure_is_not_a_vram_figure(tmp_path, monkeypatch):
    bad = {"card0": {"Card Series": "MI300X", "VRAM Total Memory (B)": "lots"}}
    fake_smi(tmp_path, monkeypatch, json.dumps(bad))
    assert hardware._detect_amd() is None


def test_used_vram_is_not_read_as_installed_vram(tmp_path, monkeypatch):
    # "VRAM Total Used Memory (B)" contains both "vram" and "total", so a broad
    # match reads a busy card as a small one depending on key order.
    used_first = {
        "card0": {
            "Card Series": "MI300X",
            "VRAM Total Used Memory (B)": "1073741824",
            "VRAM Total Memory (B)": "206158430208",
        }
    }
    fake_smi(tmp_path, monkeypatch, json.dumps(used_first))
    assert hardware._detect_amd().total_bytes == 206158430208


def test_amd_smi_is_not_claimed_as_supported(tmp_path, monkeypatch):
    # It takes a different CLI, so invoking it with rocm-smi's flags fails and
    # the fallback did nothing but look like support.
    fake_smi(tmp_path, monkeypatch, json.dumps(CARDS), name="amd-smi")
    assert hardware._detect_amd() is None


def test_no_rocm_smi_is_simply_not_an_amd_box(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert hardware._detect_amd() is None


def test_detect_reaches_the_amd_probe_at_all(tmp_path, monkeypatch):
    # Through detect(), not _detect_amd(): the defect was that nothing CALLED
    # an AMD probe, so a test that calls one directly passes with the probe
    # left out of the chain entirely. It did, until this was added.
    fake_smi(tmp_path, monkeypatch, json.dumps(CARDS))
    monkeypatch.setattr(hardware, "_detect_apple", lambda: None)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: None)
    assert hardware.detect().kind == "amd"


def test_an_apple_or_nvidia_box_still_wins_over_the_amd_probe(monkeypatch):
    sentinel = hardware.Hardware(
        kind="nvidia", name="x", total_bytes=1, usable_bytes=1, bandwidth_gbps=None, cores=1
    )
    monkeypatch.setattr(hardware, "_detect_apple", lambda: None)
    monkeypatch.setattr(hardware, "_detect_nvidia", lambda: sentinel)
    monkeypatch.setattr(hardware, "_detect_amd", lambda: pytest.fail("probed AMD too early"))
    assert hardware.detect() is sentinel


@pytest.mark.parametrize("chip", sorted(hardware.APPLE_BINNED))
def test_a_binned_apple_chip_carries_the_lower_bandwidth(chip):
    # The brand string returns "M3 Max" for both the 30-core part at 300 GB/s
    # and the 40-core at 400. The table carried only the top bin, overstating
    # the roofline by up to 33% in the flattering direction.
    low, high = hardware.APPLE_BINNED[chip]
    assert low < high
    assert hardware.APPLE_BANDWIDTH[chip] == low
