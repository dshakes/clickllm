"""Clustering captured traffic, and sampling each cluster representatively.

Two properties matter more than cleverness here:

1. **Deterministic.** The same traffic must produce the same eval set every run,
   or an equivalence verdict cannot be reproduced, re-run, or audited.
2. **Representative, not first-N.** Taking the first items of a cluster samples
   whatever your traffic happened to do this morning. We stratify across the
   response-length distribution and always keep the extremes, because the short
   and long tails are where a candidate model actually diverges.

Small clusters are surfaced, never silently dropped — a 12-request cluster may be
the one that blocks the migration. "Never silently" is the operative word: a
budget too small to cover every cluster is a real situation and sampling says so
in the report rather than quietly returning an eval set with holes in it.
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
    #: The per-cluster minimum sampling actually achieved. Lower than the
    #: requested ``min_per_cluster`` when the budget could not cover every
    #: cluster at that rate, which is a fact about the eval set a caller has to
    #: be able to see.
    floor_applied: int = 0
    #: Clusters that received no samples at all. A cluster sampled to nothing
    #: still has a key in ``sampled``, so ``key in report.sampled`` reads as
    #: covered — this is the field that says otherwise.
    uncovered: tuple[str, ...] = ()

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

    Every cluster gets at least ``min_per_cluster`` where the budget allows it.
    It does not always allow it — ``min_per_cluster`` across every cluster can
    simply exceed ``budget`` — and the guarantee is then arithmetically
    impossible rather than merely unmet. What is not acceptable is breaking it
    silently: ``floor_applied`` reports the minimum actually achieved and
    ``uncovered`` names every cluster that got nothing, because a cluster
    sampled to zero still has a key in ``sampled`` and reads as covered.
    """
    if budget < 0:
        raise ValueError(f"budget must be >= 0, got {budget}")
    total = sum(c.size for c in clusters)
    if not clusters or total == 0:
        return SampleReport({}, 0, (), 0, ())

    n = len(clusters)
    floor = min(min_per_cluster, budget // n)
    remaining = budget - floor * n

    # Largest-remainder apportionment. Truncating each cluster's share
    # independently throws every fraction away, and with many clusters that is
    # not rounding noise but the entire budget: 250 clusters each holding
    # 1/250th of traffic truncate to zero apiece and spend none of it. Handing
    # the leftover units to the largest remainders spends the budget exactly.
    exact = [remaining * c.share_of(total) for c in clusters]
    extra = [int(e) for e in exact]
    left = remaining - sum(extra)
    if left > 0:
        # Ties break on the cluster key, so the allocation is reproducible for
        # the same reason the clustering itself is.
        by_remainder = sorted(range(n), key=lambda i: (-(exact[i] - extra[i]), clusters[i].key))
        for i in by_remainder[:left]:
            extra[i] += 1

    # A cluster cannot give more captures than it holds, and the units its cap
    # frees have to land somewhere. Capping at sampling time and walking away
    # loses them: sizes [1, 1000] with budget 100 spent 98, and no field said
    # so — the same silent under-spend this function was fixed to stop, one
    # layer further down. Each pass either spends the surplus or saturates at
    # least one more cluster, so it runs at most `n` times.
    want = [min(floor + e, c.size) for c, e in zip(clusters, extra, strict=True)]
    for _ in range(n):
        spare = budget - sum(want)
        room = [i for i in range(n) if want[i] < clusters[i].size]
        if spare <= 0 or not room:
            break
        # Largest cluster first, ties on the key, for the same reproducibility
        # reason as the apportionment above.
        room.sort(key=lambda i: (-clusters[i].size, clusters[i].key))
        share, odd = divmod(spare, len(room))
        for rank, i in enumerate(room):
            want[i] = min(want[i] + share + (1 if rank < odd else 0), clusters[i].size)

    sampled: dict[str, list[Capture]] = {}
    for c, w in zip(clusters, want, strict=True):
        sampled[c.key] = sample_cluster(c.captures, w)

    return SampleReport(
        sampled=sampled,
        total_captures=total,
        small_clusters=tuple(c.key for c in clusters if c.size < MIN_CLUSTER_SIZE),
        floor_applied=floor,
        uncovered=tuple(c.key for c in clusters if not sampled[c.key]),
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
    assert rep.floor_applied == 3 and rep.uncovered == (), "this budget covers everything"

    # The regime the parameters above cannot reach: more clusters than the
    # budget can seat, where the floor collapses. `budget=30` over 3 clusters
    # can never get there, so the "no cluster sampled to nothing" assertion
    # above is real but structurally unable to fail.
    many = cluster([cap(1000 + i, f"task-{i // 5}", "y" * (i % 7 + 1)) for i in range(60)])
    assert len(many) == 12, len(many)
    tight = sample(many, budget=8, min_per_cluster=3)
    assert tight.total_sampled == 8, tight.total_sampled
    assert tight.floor_applied == 0, "the floor was not affordable and must say so"
    assert set(tight.uncovered) == {k for k, v in tight.sampled.items() if not v}, (
        "every cluster sampled to nothing must be named in `uncovered`"
    )
    assert tight.uncovered, "a budget of 8 across 12 clusters cannot cover them all"

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
