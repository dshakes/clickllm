"""The migration loop: keep the evidence fresh, and stop at the gate.

A migration is not a command, it is a period of time. Traffic arrives, the eval
set built from last month's traffic slowly stops describing it, the candidate
gets a new build, and somebody has to notice. Doing that by hand means doing it
rarely, and a proof nobody refreshes is a proof that quietly expires.

So this walks the chain on a schedule — distill the captures, prove the
candidate, ask the gate what the evidence permits — and writes what it found to
disk so the next run continues rather than restarts.

**It stops at the gate.** `gate.decide` returns `ADVANCE` as a proposal for a
human, and this module prints it and halts. Nothing here touches the control
surface, moves traffic, or calls the gateway. That is invariant 8, and it is why
the loop is worth having: an agent that could promote itself would need
supervising, and one that cannot can be left running.

Rollback is the asymmetry — it is the safe direction, and the gateway applies it
automatically from its own health signals without asking this module. The loop
reports a `ROLL_BACK` decision so a human sees it; it does not race the gateway
to apply one.

Modelled on `watch.py`, which is the existing precedent here for a resumable,
scheduled job that stages work and never publishes it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

__all__ = [
    "Run",
    "State",
    "install_schedule",
    "load_state",
    "save_state",
    "state_path",
    "step",
]

#: How many runs of history to keep. Enough to see a trend and answer "was it
#: ever green?", not so many that the file becomes a log nobody reads.
HISTORY = 20


def state_path(explicit: str | None = None) -> Path:
    """Where the migration's memory lives.

    On disk rather than in the process, because the loop's whole purpose is to
    survive between runs. A migration that forgets where it was every time it
    starts is a migration that never advances past the first rung.
    """
    if explicit:
        return Path(explicit)
    from .observe import state_dir

    return state_dir() / "migration.json"


@dataclass(frozen=True, slots=True)
class Run:
    """One pass of the loop, as recorded."""

    when: str
    captures: int
    clusters: int
    items: int
    action: str
    reason: str
    stage: str
    proposed: str | None = None
    receipt_digest: str | None = None
    #: Clusters that contributed no eval items at this budget. Carried in the
    #: history because "the proof got greener" and "the proof got narrower" look
    #: identical in a single number.
    uncovered: tuple[str, ...] = ()


@dataclass
class State:
    """Where a migration is, and how it got there."""

    candidate: str
    incumbent: str = "incumbent"
    #: The rung the *operator* says production is on. This module never changes
    #: it — it is written when a human applies a proposal and tells the loop.
    stage_phase: str = "shadow"
    stage_percent: int = 0
    started: str = ""
    runs: list[Run] = field(default_factory=list)

    def stage(self) -> Any:
        from .prove.gate import Stage

        return Stage(self.stage_phase, self.stage_percent)

    def to_json(self) -> str:
        d = asdict(self)
        d["runs"] = [asdict(r) if not isinstance(r, dict) else r for r in self.runs]
        return json.dumps(d, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> State:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("migration state must be a JSON object")
        runs = [Run(**r) if isinstance(r, dict) else r for r in raw.pop("runs", [])]
        # Tuples do not survive JSON. Restore the ones the dataclass declares,
        # so a loaded state and a fresh one behave identically — a round-trip
        # that changes a type is a bug waiting for the first comparison.
        runs = [Run(**{**asdict(r), "uncovered": tuple(r.uncovered)}) for r in runs]
        known = {"candidate", "incumbent", "stage_phase", "stage_percent", "started"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown fields in migration state: {sorted(unknown)}")
        return cls(runs=runs, **raw)


def load_state(path: Path, candidate: str = "") -> State:
    """Read the state, or start one.

    A missing file is a new migration, not an error — the first run has nothing
    to resume from and that is the normal case.
    """
    if path.exists():
        s = State.from_json(path.read_text())
        if candidate and s.candidate != candidate:
            raise ValueError(
                f"{path} tracks a migration to {s.candidate!r}, not {candidate!r}. "
                "Point --state somewhere else, or finish that one first: sharing "
                "a state file between two candidates would interleave their "
                "histories and compare a rung reached by one against evidence "
                "gathered by the other."
            )
        return s
    return State(candidate=candidate, started=date.today().isoformat())


def save_state(state: State, path: Path) -> None:
    """Persist, durably. The history is trimmed here rather than at write sites."""
    from .atomicio import atomic_write

    state.runs = state.runs[-HISTORY:]
    atomic_write(path, state.to_json())


def step(
    state: State,
    *,
    rows: list[dict[str, Any]],
    prove: Any,
    today: date | None = None,
    budget: int = 200,
    min_per_cluster: int = 3,
) -> tuple[State, Run, Any]:
    """One pass: captures in, a decision out. Applies nothing.

    `prove` is passed in rather than called directly so the loop can be tested
    without a live candidate endpoint, and so the caller owns every network
    decision this makes. It takes the eval-set document and returns the suite
    result.

    Returns the updated state, the run just recorded, and the gate's decision —
    which is a proposal. Acting on it is a human's job.
    """
    from .observe import distill
    from .prove.gate import Reading, decide

    doc, report = distill(rows, budget=budget, min_per_cluster=min_per_cluster)
    if not doc["items"]:
        raise ValueError(
            "the captures produced no eval items, so there is nothing to decide. "
            "Capture more traffic, or lower --min-per-cluster."
        )

    result = prove(doc)

    # `Reading` carries `judge_only` because a cluster whose evidence is only a
    # language model's opinion may *block* a migration and must never *advance*
    # one. A `SuiteResult` does not keep per-item tiers — `ClusterScore.outcomes`
    # is `(item_id, passed)` — so that flag cannot be recovered from it.
    #
    # When no judge ran, `judge_only=False` is not a guess: there is no judge
    # tier for a score to have come from. When one did, this loop cannot tell
    # which clusters rest on it, and defaulting to False would let exactly the
    # cluster the flag exists to stop be the one that advances. So it refuses.
    if result.receipt.judge_model:
        raise ValueError(
            "this loop cannot yet drive a judged suite: a SuiteResult does not "
            "carry per-item grader tiers, so it cannot tell which clusters rest "
            "on judge evidence alone. Such a cluster may block a migration and "
            "must never advance one, and assuming otherwise would let precisely "
            "the wrong cluster through. Run the loop with deterministic graders, "
            "and use `clickllm prove` directly when you want a judge."
        )

    scored = next(iter(result.matrix.candidates), None)
    if scored is None:
        raise ValueError("the suite scored no candidate; there is nothing to decide")
    readings = [Reading(score=c, judge_only=False) for c in scored.clusters]

    # The bar comes from the receipt rather than from a parameter of our own.
    # `--bar` reached `suite()` and not `decide()`, so an operator who asked for
    # 0.99 got a receipt recording 0.99 and a decision computed at the 0.90
    # default — which then proposed ADVANCE with the reason "proven at or above
    # the 90% bar". A stricter setting produced a looser answer, and the line
    # explaining the answer named a number the operator never chose.
    #
    # Reading it off the receipt means the two cannot disagree: whatever the
    # proof was measured against is what the decision is made against.
    decision = decide(readings, state.stage(), bar=result.receipt.bar)

    run = Run(
        when=(today or date.today()).isoformat(),
        captures=report.captures,
        clusters=report.clusters,
        items=report.items,
        action=decision.action.value,
        reason=decision.reason,
        stage=state.stage().render(),
        proposed=decision.to.render() if decision.to else None,
        receipt_digest=result.receipt.digest(),
        uncovered=tuple(report.uncovered),
    )
    state.runs.append(run)
    return state, run, decision


def render(state: State, run: Run, decision: Any) -> str:
    """What a human needs to decide, and nothing else."""
    lines = [
        "",
        f"  migration to {state.candidate}  ·  currently {run.stage}",
        f"  {run.captures} captures → {run.clusters} shapes → {run.items} items"
        f"  ·  receipt {(run.receipt_digest or '')[:12]}",
        "",
        f"  {decision.action.value.upper()}: {decision.reason}",
    ]
    if run.proposed:
        lines += [
            "",
            f"  The evidence permits {run.proposed}. It has not been applied.",
            "  Escalation goes through the gateway's control surface, which records",
            "  a reason and refuses an unconfirmed increase — nothing here can do it",
            "  for you, on purpose (invariant 8).",
        ]
    if run.uncovered:
        lines += [
            "",
            f"  {len(run.uncovered)} clusters contributed no items at this budget:",
            *(f"    {u}" for u in run.uncovered),
            "  The decision above says nothing about that traffic.",
        ]
    if len(state.runs) > 1:
        prev = state.runs[-2]
        if prev.items > run.items:
            lines += [
                "",
                f"  Note: fewer items than last run ({prev.items} → {run.items}). A",
                "  narrower proof can read as a steadier one; check the coverage",
                "  before treating this as an improvement.",
            ]
    lines.append("")
    return "\n".join(lines)


def install_schedule(*, interval_hours: int = 24, state: str = "") -> tuple[str, str]:
    """A crontab fragment, and where to put it. Not installed for you.

    Same rule as `watch.install_schedule`: a recurring job that touches your
    traffic is your decision, and a tool that scheduled itself would be doing
    the opposite of what this project promises.
    """
    where = "your crontab (`crontab -e`)"
    arg = f" --state {state}" if state else ""
    every = f"0 */{interval_hours} * * *" if interval_hours < 24 else "0 3 * * *"
    fragment = f"{every} clickllm migrate --step{arg} >> ~/.clickllm/migrate.log 2>&1\n"
    return fragment, where


def demo() -> None:
    """Self-check: a loop that records, resumes, and refuses to advance itself."""
    import tempfile

    from .prove import EvalItem, suite

    rows = [
        {
            "request_id": f"r{i}",
            "model": "gpt-5",
            "messages": [{"role": "user", "content": f"summarise {i}"}],
            "response": "a summary",
            "prompt_tokens": 40,
            "latency_ms": 10,
            "tools": [],
            "tool_calls": [],
            "response_format": None,
        }
        for i in range(60)
    ]

    def fake_prove(doc: dict[str, Any]) -> Any:
        items = [
            EvalItem(
                item_id=str(i["item_id"]),
                cluster=str(i["cluster"]),
                prompt=str(i["prompt"]),
                baseline=str(i["baseline"]),
                candidate=str(i["baseline"]),
            )
            for i in doc["items"]
        ]
        return suite(
            items,
            shares=doc["shares"],
            names=doc["names"],
            candidate="cand",
            incumbent="inc",
            issued="2026-08-11",
        )

    st = State(candidate="cand", started="2026-08-11")
    st, run, decision = step(st, rows=rows, prove=fake_prove, today=date(2026, 8, 11))
    assert run.items > 0 and run.clusters == 1, run
    assert run.stage == "shadow", "a new migration starts in shadow"

    # The whole point: whatever the evidence says, the stage does not move.
    assert st.stage_percent == 0, "the loop moved traffic"
    assert st.stage_phase == "shadow"
    text = render(st, run, decision)
    if run.proposed:
        assert "has not been applied" in text
        assert "invariant 8" in text

    # Resumable: a saved state reloads as itself, tuples and all.
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        save_state(st, p)
        back = load_state(p, "cand")
        assert back.candidate == "cand" and len(back.runs) == len(st.runs)
        assert isinstance(back.runs[0].uncovered, tuple)
        assert back.stage().render() == st.stage().render()

        # A state file belongs to one candidate. Sharing it would compare a rung
        # reached by one against evidence gathered by the other.
        try:
            load_state(p, "a-different-model")
        except ValueError as e:
            assert "not 'a-different-model'" in str(e)
        else:  # pragma: no cover
            raise AssertionError("a mismatched candidate must refuse")

        # The history is a ring, not a log. Trimming happens in `save_state`,
        # so it is only real if a save actually drops the oldest run.
        made = 30
        st.runs = [
            Run(
                when="2026-08-11",
                captures=1,
                clusters=1,
                items=i,
                action="hold",
                reason="x",
                stage="shadow",
            )
            for i in range(made)
        ]
        save_state(st, p)
        kept = load_state(p, "cand").runs
        # `made` is a literal, not `HISTORY + n`. Written in terms of the
        # constant, these assertions hold for *any* value of it — the same
        # tautology this repo has shipped before, a check that cannot fail
        # because it restates its input. Fixed, so raising HISTORY above it
        # stops the trim and this notices.
        assert len(kept) < made, f"nothing was trimmed: kept {len(kept)} of {made}"
        assert len(kept) == HISTORY, f"kept {len(kept)}, cap is {HISTORY}"
        # The *newest* survive. Trimming the wrong end would keep a history that
        # gets staler every run while looking exactly as full.
        assert kept[-1].items == made - 1, kept[-1]

    fragment, _ = install_schedule(interval_hours=6)
    assert "clickllm migrate --step" in fragment and "*/6" in fragment
    print("migrate: ok")


if __name__ == "__main__":  # pragma: no cover
    demo()
