"""Clustering captured traffic, and sampling each cluster representatively.

Two properties matter more than cleverness here:

1. **Deterministic.** The same traffic must produce the same eval set every run,
   or an equivalence verdict cannot be reproduced, re-run, or audited.
2. **Representative, not first-N.** Taking the first items of a cluster samples
   whatever your traffic happened to do this morning. We stratify across the
   response-length distribution and always keep the extremes, because the short
   and long tails are where a candidate model actually diverges.

Small clusters are surfaced, never silently dropped — a 12-request cluster may be
the one that blocks the migration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .shape import Capture, Cluster, extract_shape

# Below this, a cluster's scores are too noisy to act on. Kept and flagged rather
# than discarded: silent truncation reads as "we covered everything".
MIN_CLUSTER_SIZE = 5


def cluster(captures: list[Capture]) -> list[Cluster]:
    """Group captures by task shape, largest first, with unambiguous names."""
    groups: dict[str, Cluster] = {}
    for cap in captures:
        shape = extract_shape(cap)
        groups.setdefault(shape.signature, Cluster(shape=shape)).captures.append(cap)
    # Ties break on the signature so ordering is stable across runs.
    ordered = sorted(groups.values(), key=lambda c: (-c.size, c.key))
    _disambiguate(ordered)
    return ordered


def _disambiguate(clusters: list[Cluster]) -> None:
    """Ensure no two clusters share a display name.

    Two different system prompts can produce the same structural description, and
    a report with two rows both labelled "free text" is a report nobody can act
    on — which defeats the point of naming clusters at all. Collisions get a
    stable ordinal, assigned in the deterministic order above.
    """
    seen: dict[str, int] = {}
    for c in clusters:
        base = c.shape.describe()
        n = seen.get(base, 0)
        seen[base] = n + 1
        c.label = base if n == 0 else f"{base} #{n + 1}"
    # A name used once needs no ordinal; a name used twice needs it on both.
    firsts = {}
    for c in clusters:
        base = c.shape.describe()
        if seen[base] > 1 and c.label == base:
            firsts[base] = c
    for base, c in firsts.items():
        c.label = f"{base} #1"


@dataclass(frozen=True, slots=True)
class SampleReport:
    """What sampling chose, and what it left out."""

    sampled: dict[str, list[Capture]]
    total_captures: int
    small_clusters: tuple[str, ...]

    @property
    def total_sampled(self) -> int:
        return sum(len(v) for v in self.sampled.values())


def _stable_rank(cap: Capture) -> str:
    """A deterministic per-capture ordering key.

    Hashed rather than positional so the order does not depend on how traffic
    happened to arrive, and is identical on any machine.
    """
    return hashlib.sha256(cap.request_id.encode()).hexdigest()


def sample_cluster(captures: list[Capture], n: int) -> list[Capture]:
    """Pick ``n`` captures spanning the cluster's response-length range.

    Sorted by response length, then taken at even intervals, so the sample covers
    terse and verbose answers rather than clumping. The first and last are always
    included: the extremes are where candidates diverge most.
    """
    if n <= 0 or not captures:
        return []
    if len(captures) <= n:
        return sorted(captures, key=_stable_rank)

    ordered = sorted(captures, key=lambda c: (len(c.response), _stable_rank(c)))
    if n == 1:
        return [ordered[len(ordered) // 2]]

    last = len(ordered) - 1
    picked_idx = {round(i * last / (n - 1)) for i in range(n)}
    # Rounding can collide; backfill deterministically so we always return n.
    for i in range(len(ordered)):
        if len(picked_idx) >= n:
            break
        picked_idx.add(i)
    return [ordered[i] for i in sorted(picked_idx)]


def sample(
    clusters: list[Cluster],
    budget: int = 200,
    min_per_cluster: int = 3,
) -> SampleReport:
    """Allocate a sampling budget across clusters, weighted by traffic share.

    Every cluster gets at least ``min_per_cluster`` so a small-but-critical
    cluster is never sampled to nothing; the rest is distributed by share.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    total = sum(c.size for c in clusters)
    if not clusters or total == 0:
        return SampleReport({}, 0, ())

    floor = min(min_per_cluster, budget // max(len(clusters), 1))
    remaining = max(budget - floor * len(clusters), 0)

    sampled: dict[str, list[Capture]] = {}
    for c in clusters:
        want = floor + int(remaining * c.share_of(total))
        sampled[c.key] = sample_cluster(c.captures, min(want, c.size))

    return SampleReport(
        sampled=sampled,
        total_captures=total,
        small_clusters=tuple(c.key for c in clusters if c.size < MIN_CLUSTER_SIZE),
    )


def demo() -> None:
    def cap(i: int, system: str, resp: str) -> Capture:
        return Capture(
            request_id=f"req-{i}",
            model="gpt-5",
            messages=({"role": "system", "content": system}, {"role": "user", "content": f"q{i}"}),
            response=resp,
            prompt_tokens=500,
        )

    caps = (
        [cap(i, "reviewer", "```py\n" + "x" * i + "\n```") for i in range(60)]
        + [cap(100 + i, "translator", "hola " * (i + 1)) for i in range(30)]
        + [cap(200 + i, "classifier", "spam") for i in range(3)]
    )

    cs = cluster(caps)
    assert len(cs) == 3, [c.name for c in cs]
    assert cs[0].size == 60 and cs[-1].size == 3, "must sort by size"
    assert abs(cs[0].share_of(len(caps)) - 60 / 93) < 1e-9

    # Deterministic: identical input, identical output.
    assert [c.key for c in cluster(caps)] == [c.key for c in cs]
    assert [c.key for c in cluster(list(reversed(caps)))] == [c.key for c in cs], (
        "arrival order must not change the clustering"
    )

    names = [c.name for c in cs]
    assert len(set(names)) == len(names), f"cluster names must be unique, got {names}"
    assert any("#" in n for n in names), "colliding descriptions must be disambiguated"

    rep = sample(cs, budget=30)
    assert rep.total_sampled <= 30
    assert all(len(v) >= 1 for v in rep.sampled.values()), "no cluster sampled to nothing"
    assert cs[-1].key in rep.small_clusters, "a 3-capture cluster must be flagged"

    # Sampling spans the length range rather than clumping.
    picked = rep.sampled[cs[0].key]
    lengths = [len(c.response) for c in picked]
    assert min(lengths) < max(lengths), "sample must cover the response-length range"

    # ...and is stable across runs.
    again = sample(cluster(caps), budget=30)
    assert [c.request_id for c in again.sampled[cs[0].key]] == [c.request_id for c in picked]

    print(f"clusters: {[(c.name[:28], c.size) for c in cs]}")
    print(f"sampled {rep.total_sampled} of {rep.total_captures}")
    print("ok")


if __name__ == "__main__":
    demo()
