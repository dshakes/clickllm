"""Find and launch the capture gateway, and turn its log into an eval set.

Two halves of one chain. `observe` puts the gateway in front of a provider and
records what really goes through it; `distill` reads that log back, clusters it
by task shape, and writes the eval set `onpar prove` consumes.

Neither half is new machinery — the gateway is a Rust binary with its own tests,
the clustering is `distill/`. This module is the wiring, which is what was
missing: both halves were built, tested, and unreachable from the command line.

The datapath boundary is ADR-0015 — in the request path only while a migration
is in flight, with leaving as the terminal state. `observe` is the entering, and
it says so when it starts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DistillReport",
    "distill",
    "find_gateway",
    "gateway_argv",
    "run_gateway",
    "state_dir",
    "write_eval_set",
]


def state_dir() -> Path:
    """Where captures and the key live.

    `ONPAR_HOME` first, so a user can put a log full of production prompts on
    a volume they chose deliberately rather than in their home directory.
    """
    return Path(os.environ.get("ONPAR_HOME") or Path.home() / ".onpar")


def find_gateway() -> Path | None:
    """The gateway binary, if this machine has one.

    Searched in the order a user would expect it to win: an explicit override,
    then `PATH`, then this checkout's own build outputs — release before debug,
    because a developer with both means the one they shipped.
    """
    override = os.environ.get("ONPAR_GATEWAY_BIN")
    if override:
        p = Path(override)
        # An override that does not exist is an error, not a reason to fall
        # through and launch a different binary than the one named. Falling
        # through would put an unintended process in the request path.
        if not p.is_file():
            raise FileNotFoundError(f"ONPAR_GATEWAY_BIN={override} is not a file")
        return p

    # Beside this interpreter, before `PATH`. `pip install onpar-gateway`
    # puts the binary in the same `bin/` as the `onpar` script that is
    # running, and that is the one the user meant — not whichever other
    # environment happens to be earlier on `PATH`.
    #
    # It also has to come before `PATH` for a reason that is easy to miss: a
    # venv's `bin/` is only on `PATH` once activated, so invoking the console
    # script by absolute path (or through `uvx`) found nothing at all, while
    # the binary sat next to it. That is not a corner case — it is what
    # happened the first time this was installed from a wheel.
    beside = Path(sys.executable).parent / "onpar-gateway"
    if beside.is_file():
        return beside

    found = shutil.which("onpar-gateway")
    if found:
        return Path(found)

    root = Path(__file__).resolve().parents[2]
    for profile in ("release", "debug"):
        p = root / "target" / profile / "onpar-gateway"
        if p.is_file():
            return p
    return None


def gateway_argv(
    binary: Path,
    upstream: str,
    *,
    port: int = 8787,
    candidate: str | None = None,
    capture: Path | None = None,
    key: Path | None = None,
    no_capture: bool = False,
) -> list[str]:
    """The exact command line, built once so the runner and its test agree.

    Note what cannot be here: nothing that moves traffic. The gateway refuses a
    startup `--percent` outright, and escalation goes through the control
    surface, which records a reason and refuses an unconfirmed increase.
    """
    argv = [str(binary), "--upstream", upstream, "--port", str(port)]
    if candidate:
        argv += ["--candidate", candidate]
    if no_capture:
        argv.append("--no-capture")
    else:
        if capture is not None:
            argv += ["--capture", str(capture)]
        if key is not None:
            argv += ["--key", str(key)]
    return argv


def run_gateway(argv: list[str]) -> int:
    """Run the gateway in the foreground, returning its exit code.

    Foreground on purpose. A background daemon is a thing a user forgets is in
    their request path, and ADR-0015's whole claim is that this leaves.
    """
    try:
        return subprocess.call(argv)
    except KeyboardInterrupt:
        # Ctrl-C is how this is meant to end. Reporting it as a crash would make
        # the normal exit look like a failure.
        return 0


# --- distill ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DistillReport:
    """What the eval set was built from, and what it left out."""

    captures: int
    clusters: int
    items: int
    #: Clusters that received no sample. Named, not counted — "3 uncovered" is
    #: not something a reader can act on.
    uncovered: tuple[str, ...]
    floor_applied: int
    labels: tuple[tuple[str, int, float], ...]

    def render(self) -> str:
        lines = [
            f"  {self.captures} captures  →  {self.clusters} task shapes"
            f"  →  {self.items} eval items",
            "",
        ]
        for label, n, share in self.labels:
            lines.append(f"  {label:<40}{n:>4} sampled{share * 100:7.1f}% of traffic")
        if self.uncovered:
            lines += [
                "",
                f"  {len(self.uncovered)} clusters got no sample at this budget:",
                *(f"    {u}" for u in self.uncovered),
                "",
                "  Raise --budget to cover them, or accept that the proof says",
                "  nothing about that traffic. It must not read as passing.",
            ]
        return "\n".join(lines)


def distill(
    rows: list[dict[str, Any]],
    *,
    budget: int = 200,
    min_per_cluster: int = 3,
    name_with: Callable[[str], str] | None = None,
) -> tuple[dict[str, Any], DistillReport]:
    """Cluster captures by task shape and sample an eval set out of them.

    Returns the eval-set document `onpar prove` reads, and a report of what
    it is made of. Every item's `baseline` is the **incumbent's** answer, which
    is not ground truth — it is what a candidate has to match, a weaker and more
    honest claim, and the provenance block says so in the file itself.
    """
    from .distill.cluster import cluster, sample
    from .distill.shape import from_capture_row

    caps = [from_capture_row(r) for r in rows]
    clusters = cluster(caps)
    if name_with is not None:
        # Between clustering and sampling, so the names reach the report. Never
        # raises and never blocks: a cluster that cannot be named keeps its
        # structural description, which is correct and merely ugly.
        from .distill.name import name_clusters

        name_clusters(clusters, name_with)
    report = sample(clusters, budget=budget, min_per_cluster=min_per_cluster)
    total = sum(c.size for c in clusters) or 1

    by_key = {c.key: c for c in clusters}
    items: list[dict[str, Any]] = []
    shares: dict[str, float] = {}
    names: dict[str, str] = {}
    labels: list[tuple[str, int, float]] = []

    for key, chosen in report.sampled.items():
        c = by_key.get(key)
        label = c.label if c else key
        # Share of *traffic*, not share of the sample. A small cluster sampled
        # up to the floor must not thereby claim a larger slice of the verdict
        # than it holds of the workload — that is how an eval set that looks
        # representative stops being one.
        share = c.share_of(total) if c else 0.0
        names[key] = label
        shares[key] = share
        labels.append((label, len(chosen), share))
        for cap in chosen:
            items.append(
                {
                    "item_id": cap.request_id or f"{key}-{len(items)}",
                    "cluster": key,
                    "prompt": cap.user_text,
                    "baseline": cap.response,
                    "candidate": "",
                    # Dicts, the shape `prove`'s ToolChoice grader reads. A
                    # list of bare names looked equivalent and crashed the
                    # grader four frames down — the next seam along from the
                    # one this pipeline was built to close.
                    "baseline_tool_calls": [
                        {"name": t.get("name", "")} for t in cap.tool_calls if isinstance(t, dict)
                    ],
                    "response_format": cap.response_format,
                }
            )

    labels.sort(key=lambda r: -r[2])
    doc = {
        "items": items,
        "shares": shares,
        "names": names,
        "provenance": {
            "captures": len(caps),
            "clusters": len(clusters),
            "budget": budget,
            "min_per_cluster": min_per_cluster,
            "floor_applied": report.floor_applied,
            "uncovered": list(report.uncovered),
            "note": (
                "Baselines are the incumbent's replies, not ground truth. "
                "Clusters listed in `uncovered` contributed no items, and the "
                "verdict says nothing about that traffic."
            ),
        },
    }
    return doc, DistillReport(
        captures=len(caps),
        clusters=len(clusters),
        items=len(items),
        uncovered=tuple(names.get(u, u) for u in report.uncovered),
        floor_applied=report.floor_applied,
        labels=tuple(labels),
    )


def write_eval_set(doc: dict[str, Any], path: Path) -> Path:
    """Write the eval set, durably.

    Through `atomicio` rather than `write_text`: this is the artifact the whole
    proof is computed over, and a half-written one that still parses is the
    worst available outcome.
    """
    from .atomicio import atomic_write_json

    atomic_write_json(path, doc, sort_keys=True)
    return path


def demo() -> None:
    """Self-check: the wiring, without a gateway or a built extension."""
    b = Path("/usr/bin/true")
    argv = gateway_argv(b, "https://api.openai.com/v1", capture=Path("/tmp/c"))
    assert "--upstream" in argv and "--capture" in argv, argv
    assert "--percent" not in argv, "nothing here may move traffic"
    assert "--capture" not in gateway_argv(b, "u", no_capture=True)

    # Every flag emitted must be one the binary declares. The first version of
    # this function passed `--listen`, which the gateway has never had; the
    # argv looked entirely reasonable and the process refused to start. Reading
    # the binary's own usage text is what closes that, and it works without a
    # built binary because the text is in the source.
    usage = (
        Path(__file__).resolve().parents[2] / "onpar-gateway" / "src" / "main.rs"
    ).read_text()
    for flag in {a for a in argv if a.startswith("--")} | {"--key", "--no-capture", "--candidate"}:
        assert f'"{flag}"' in usage, f"{flag} is not a flag the gateway accepts"

    rows = [
        {
            "request_id": f"r{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"summarise doc {i}"}],
            "response": "a summary",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [],
            "tool_calls": [],
            "response_format": None,
        }
        for i in range(6)
    ]
    rows.append(
        {
            "request_id": "t1",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "refund order 9"}],
            "response": "",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [{"function": {"name": "refund"}}],
            "tool_calls": ["refund"],
            "response_format": "json_object",
        }
    )
    doc, rep = distill(rows, budget=20, min_per_cluster=2)
    assert rep.captures == 7, rep
    assert rep.clusters == 2, "a tool-calling exchange is not the shape of a summary"
    assert doc["items"], "an eval set with no items proves nothing"
    assert abs(sum(doc["shares"].values()) - 1.0) < 1e-9, doc["shares"]
    tool_items = [i for i in doc["items"] if i["baseline_tool_calls"]]
    assert tool_items and tool_items[0]["baseline_tool_calls"] == [{"name": "refund"}]
    assert "not ground truth" in doc["provenance"]["note"]

    # The share a cluster claims is its share of traffic, not of the sample.
    # Six summaries and one tool call, sampled two and two, must not report
    # 50/50 — that is how an eval set stops being representative while looking
    # more balanced than the traffic it came from.
    tool_key = next(k for k, v in doc["names"].items() if "tool" in v.lower() or "refund" in v)
    assert doc["shares"][tool_key] < 0.2, doc["shares"]

    # A budget too small to cover both clusters names what it dropped rather
    # than quietly producing a narrower proof that reads as a complete one.
    _, thin = distill(rows, budget=1, min_per_cluster=1)
    assert thin.uncovered or thin.items <= 1, thin

    # Exercised, not merely defined. This function imported a name `atomicio`
    # does not export, and every assertion above still passed — a green check
    # over a call that was never made.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        out = write_eval_set(doc, Path(d) / "evalset.json")
        assert json.loads(out.read_text())["items"] == doc["items"]
    print("observe: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
    sys.exit(0)
