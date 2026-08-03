"""The agentic session — a conversation that ends in a deployment you can trust.

Every other module here answers one question well. `intent` reads a sentence,
`fit` sizes, `plan` configures, `advise` critiques, `prove` scores. A human can
sequence those. Nothing sequenced them *for* you, and that gap is the difference
between a toolbox and a product.

This is the thing that holds the thread: you say what you are building, it works
out what it can, asks only what it genuinely cannot infer, and carries the answer
forward until there is a command to run and an eval suite to run against it.

## Why it is a state machine and not a wizard

A wizard asks twelve questions before it tells you anything, and every one of
them is mandatory because the author could not be bothered to work out which
ones mattered. The failure is not the questions — it is that you learn nothing
until the end, so a wrong answer on question two costs you the whole flow.

This never blocks. At every point there is a usable answer built from what is
known plus stated assumptions, and [`Session.answer`] will give it to you
mid-conversation. Questions *refine*; they do not gate.

## The three rules

**Ask only what changes the answer.** A question is worth asking when a
different answer would produce a different deployment — and that is computable,
not a matter of taste. [`Session.ask`] re-plans under each candidate answer and
stays silent when they agree. A form that asks about concurrency when every
candidate model fits at every concurrency is wasting the one thing the user is
actually spending: attention.

**Every inference shows its evidence.** Inherited from `intent`: what was read
out of your sentence is quoted back with the words that produced it, so a
misreading is obvious now rather than discovered in production.

**It stops at the door.** The session produces a command; it does not run one.
Deployment is a human action and cutover is a human decision, and nothing here
can be persuaded otherwise — the same boundary the MCP surface holds. An agent
may drive this entire flow and still cannot move traffic.

## Resumable, because a conversation outlives a process

The whole state serialises to JSON and back. A session survives a restart, and
an agent can carry it between turns without holding anything in memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import StrEnum

from . import catalog, fit, hardware, hardware_catalog, intent
from .advise import Suggestion, suggest
from .hardware import Hardware
from .plan import Plan, Requirements, Workload, plan

__all__ = [
    "Session",
    "Stage",
    "Turn",
]


class Stage(StrEnum):
    """Where the conversation is. Advances only when the stage below is settled."""

    #: What are you building? Nothing else can be decided before this.
    UNDERSTAND = "understand"
    #: What are you running it on? Detected, or named.
    HARDWARE = "hardware"
    #: Which model. The first stage with a real answer.
    CHOOSE = "choose"
    #: Engine and flags.
    CONFIGURE = "configure"
    #: What to change that you did not ask about.
    OPTIMIZE = "optimize"
    #: The command. Handed over, never run.
    DEPLOY = "deploy"
    #: How you will know it is good enough.
    PROVE = "prove"


@dataclass(frozen=True, slots=True)
class Turn:
    """One exchange: what changed, what is still open, and the answer so far.

    Carries the answer on *every* turn, not just the last one. A session you can
    interrupt at any point and still walk away with something is the whole
    difference from a wizard.
    """

    stage: Stage
    said: str
    #: The single most valuable question, or None when nothing would change the
    #: answer. Never a list — a list is a form.
    question: str | None = None
    #: What was read from what you said, with the words that justified it.
    evidence: tuple[str, ...] = ()
    #: Assumptions in force. Always visible, never silent.
    assuming: tuple[str, ...] = ()
    done: bool = False

    def render(self) -> str:
        out = [self.said]
        if self.evidence:
            out += ["", "  understood:"] + [f"    · {e}" for e in self.evidence]
        if self.assuming:
            out += ["", "  assuming:"] + [f"    · {a}" for a in self.assuming]
        if self.question:
            out += ["", f"  ? {self.question}"]
        return "\n".join(out)


def _fmt_gb(b: float) -> str:
    return f"{b / 1024**3:.1f} GB"


@dataclass
class Session:
    """A conversation with state. Feed it sentences; read answers off it."""

    text: str = ""
    stage: Stage = Stage.UNDERSTAND
    requirements: Requirements = field(
        default_factory=lambda: Requirements(workload=Workload.INTERACTIVE)
    )
    #: Fields the user set explicitly. Everything else is inferred or defaulted,
    #: and the distinction is what stops the session re-asking something answered.
    stated: set[str] = field(default_factory=set)
    hw: Hardware | None = None
    hw_source: str = ""
    model_id: str = ""
    #: Questions already put to the user. Asked once, then assumed — a session
    #: that repeats itself is one nobody finishes.
    asked: set[str] = field(default_factory=set)
    evidence: tuple[str, ...] = ()

    # --- input ---------------------------------------------------------------

    def tell(self, text: str) -> Turn:
        """Feed plain language. Re-reads requirements, keeping anything you stated."""
        self._apply_text(text)
        return self.step()

    def _apply_text(self, text: str) -> None:
        """State mutation only — no `step()`, so no `_worth_asking` side effect.

        Split out so a caller that chains several inputs together before
        deciding what to show the user (`cmd_build`, `mcp._build`) can mutate
        state repeatedly and call `step()` exactly once at the end. See
        `_apply_hardware` and `_apply_fields` for why this matters.
        """
        self.text = f"{self.text} {text}".strip()
        read = intent.read(self.text)
        # An explicit answer outranks a re-reading of the prose: the user said
        # "8 users" once and must not be talked out of it by a later sentence.
        kept = {f: getattr(self.requirements, f) for f in self.stated}
        self.requirements = replace(read.requirements, **kept)
        self.evidence = tuple(i.render() for i in read.inferred)
        if self.stage is Stage.UNDERSTAND:
            self.stage = Stage.HARDWARE

    def set(self, **fields: object) -> Turn:
        """Answer directly. `set(concurrency=32)` is worth ten sentences."""
        self._apply_fields(**fields)
        return self.step()

    def _apply_fields(self, **fields: object) -> None:
        """State mutation only. See `_apply_text`."""
        unknown = set(fields) - set(Requirements.__slots__)
        if unknown:
            raise ValueError(
                f"not a requirement: {', '.join(sorted(unknown))}. "
                f"Known: {', '.join(sorted(Requirements.__slots__))}"
            )
        self.requirements = replace(self.requirements, **fields)
        self.stated |= set(fields)

    def on(self, machine: str | Hardware | None = None) -> Turn:
        """Pick the hardware. A profile id, a `Hardware`, or None to detect local."""
        self._apply_hardware(machine)
        return self.step()

    def _apply_hardware(self, machine: str | Hardware | None = None) -> None:
        """State mutation only. See `_apply_text`."""
        if isinstance(machine, Hardware):
            self.hw, self.hw_source = machine, "supplied"
        elif machine:
            p = hardware_catalog.get(machine)
            self.hw, self.hw_source = p.to_hardware(), f"profile {p.id}"
        else:
            self.hw, self.hw_source = hardware.detect(), "detected locally"
        if self.stage is Stage.HARDWARE:
            self.stage = Stage.CHOOSE

    # --- the part that makes it not a wizard ---------------------------------

    def _candidates(self) -> list[fit.Fit]:
        if self.hw is None:
            return []
        feasible, _ = fit.rank(self.hw, self.requirements.context, self.requirements.concurrency)
        return feasible

    @staticmethod
    def _default_pick(cands: list[fit.Fit]) -> fit.Fit:
        """The biggest model that is not painful to use.

        `fit.rank` orders by capability, so its head is the largest thing that
        technically fits — which on a laptop is routinely a 70B at 96% of memory
        running at 6 tok/s. That is a correct answer to "what fits" and a bad
        answer to "what should I deploy", and the session is being asked the
        second question.

        Falls back to the head when everything is slow: "slow but running" beats
        "nothing", as long as the flag is visible. Same rule as `sdk.best`.
        """
        return next((c for c in cands if not c.slow), cands[0])

    def _plan(self) -> Plan | None:
        if self.hw is None:
            return None
        m = catalog.get(self.model_id) if self.model_id else None
        quant = None
        if m is not None:
            f = fit.best_quant(m, self.hw, self.requirements.context, self.requirements.concurrency)
            quant = f.quant if f else m.quants[0]
        return plan(self.hw, self.requirements, m, quant)

    def _worth_asking(self) -> str | None:
        """The one question whose answer would change the deployment.

        Computed, not curated: each candidate answer is re-planned, and the
        question is only raised when the outcomes actually differ. This is the
        difference between a product that respects attention and a form.

        Mutates `self.asked` — this is the "commit" for a question that is about
        to be shown to someone. It must be called at most once per *external*
        turn, which is exactly why `tell()`, `on()` and `set()` no longer call
        `step()` internally: a caller that chains `_apply_text` -> `_apply_hardware`
        -> `_apply_fields` and reads only the final `step()` gets one commit: the
        true highest-priority question. The old design called `step()` — and
        therefore this — on every intermediate mutator too, so the best question
        was marked asked and discarded by whichever call happened to run first,
        and the Turn a user actually saw showed a lower-priority one instead.
        """
        if self.hw is None:
            return None
        probes: list[tuple[str, str, dict[str, object]]] = [
            (
                "concurrency",
                "How many requests will be in flight at once?",
                {"concurrency": 1},
            ),
            (
                "context",
                "How long are the prompts — a few thousand tokens, or tens of thousands?",
                {"context": 8_192},
            ),
            (
                "prefix_sharing",
                "Do your requests share a long system prompt? (an agent fleet usually does)",
                {"prefix_sharing": 0.8},
            ),
        ]
        for field_name, question, alternative in probes:
            if field_name in self.stated or field_name in self.asked:
                continue
            if self._outcome_differs(alternative):
                self.asked.add(field_name)
                return question
        return None

    def _outcome_differs(self, alternative: dict[str, object]) -> bool:
        """Whether a different answer would produce a different deployment."""
        here = self._signature(self.requirements)
        there = self._signature(replace(self.requirements, **alternative))
        return here != there

    def _signature(self, req: Requirements) -> tuple:
        """What a deployment *is*, for comparison: engine, knobs, and what fits."""
        if self.hw is None:
            return ()
        p = plan(self.hw, req)
        feasible, _ = fit.rank(self.hw, req.context, req.concurrency)
        return (
            p.engine.value,
            tuple(sorted((k.name.value, str(k.value)) for k in p.knobs)),
            tuple(f.model.id for f in feasible[:3]),
            bool(p.warnings),
        )

    # --- output ---------------------------------------------------------------

    def step(self) -> Turn:
        """Do the next thing that can be done, and report it."""
        req = self.requirements
        assuming = tuple(
            f"{f} = {getattr(req, f)}" for f in ("concurrency", "context") if f not in self.stated
        )

        if self.hw is None:
            return Turn(
                stage=Stage.HARDWARE,
                said="Tell me the machine — `on()` to detect this one, or a profile id.",
                question="What are you deploying on?",
                evidence=self.evidence,
            )

        cands = self._candidates()
        if not cands:
            return Turn(
                stage=Stage.CHOOSE,
                said=(
                    f"Nothing in the catalogue fits {self.hw.name} at "
                    f"{req.context:,} context and concurrency {req.concurrency}. "
                    f"That is an answer, not a failure — reduce one of them, or "
                    f"use a bigger machine."
                ),
                evidence=self.evidence,
                assuming=assuming,
            )

        if not self.model_id:
            self.model_id = self._default_pick(cands).model.id
        if self.stage in (Stage.CHOOSE, Stage.HARDWARE):
            self.stage = Stage.CONFIGURE

        p = self._plan()
        question = self._worth_asking()
        best = next((c for c in cands if c.model.id == self.model_id), self._default_pick(cands))
        said = (
            f"{best.model.name} at {best.quant} on {self.hw.name} — "
            f"{_fmt_gb(best.total_bytes)} of {_fmt_gb(self.hw.usable_bytes)}, "
            f"{len(cands)} candidates fit. Engine: {p.engine.value}."
        )
        if p.warnings:
            said += f" Cannot meet: {p.warnings[0]}"
        return Turn(
            stage=self.stage,
            said=said,
            question=question,
            evidence=self.evidence,
            assuming=assuming,
            done=question is None,
        )

    def answer(self) -> str:
        """The best deployment available right now, whatever is still unknown.

        Callable at any point. This is the promise that makes the questions
        optional rather than a gate.
        """
        if self.hw is None:
            return "No machine chosen yet — call on() to detect this one or name a profile."
        cands = self._candidates()
        if not cands:
            return self._what_would_fit()
        best = next((c for c in cands if c.model.id == self.model_id), self._default_pick(cands))
        p = self._plan()
        assert p is not None
        argv, gaps = p.command(catalog.get(best.model.id).repo or best.model.id)
        env = p.environment()
        # Ollama's whole configuration is environment, not argv — printing argv
        # alone here would claim a command that starts with its defaults,
        # silently ignoring what was just planned for it.
        prefix = " ".join(f"{k}={v}" for k, v in env)
        run_line = f"{prefix} {' '.join(argv)}".strip()

        out = [
            f"MODEL     {best.model.name} @ {best.quant}   ({_fmt_gb(best.total_bytes)})",
            f"MACHINE   {self.hw.name}   ({self.hw_source})",
            f"ENGINE    {p.engine.value}   {p.engine_why}",
            "",
            "RUN",
            "  " + run_line if run_line else "  (no verified flag dialect for this engine)",
        ]
        if gaps:
            out += ["", "NOT EXPRESSED"] + [f"  · {g}" for g in gaps]
        if p.warnings:
            out += ["", "CANNOT MEET"] + [f"  · {w}" for w in p.warnings]

        tips = self.optimizations()
        if tips:
            out += ["", "WORTH CHANGING"]
            out += [f"  [{s.impact.value}] {s.action}" for s in tips[:3]]

        out += [
            "",
            "THEN PROVE IT",
            "  clickllm prove evalset.json --candidate "
            f"{best.model.id} --incumbent <your-current-model>",
            "",
            "Nothing above has been run. Deployment is yours to trigger, and no",
            "eval result moves production traffic without a human.",
        ]
        return "\n".join(out)

    def _what_would_fit(self) -> str:
        """ "Nothing fits" is a dead end. Say what to change instead.

        Searches the two axes the user actually controls — how much context and
        how many concurrent requests — and reports the largest value of each
        that works, holding the other fixed. A refusal with a route out of it is
        a different product from a refusal.
        """
        assert self.hw is not None
        req = self.requirements
        head = (
            f"Nothing fits {self.hw.name} at {req.context:,} context and "
            f"concurrency {req.concurrency}. Here is what would:"
        )
        out = [head, ""]

        def fits(**over: object) -> bool:
            r = replace(req, **over)
            feasible, _ = fit.rank(self.hw, r.context, r.concurrency)
            return bool(feasible)

        # Halve rather than scan: the memory is linear in each, so the useful
        # answer is an order of magnitude, and a linear scan would be slower for
        # no extra information.
        ctx = req.context
        while ctx > 2048 and not fits(context=ctx):
            ctx //= 2
        if fits(context=ctx) and ctx < req.context:
            out.append(f"  · same {req.concurrency} users at {ctx:,} context instead")

        conc = req.concurrency
        while conc > 1 and not fits(concurrency=conc):
            conc //= 2
        if fits(concurrency=conc) and conc < req.concurrency:
            out.append(f"  · same {req.context:,} context at {conc} concurrent instead")

        # Usable against usable, and it must actually solve the problem.
        # Suggesting an A100 80GB to someone on an H100 80GB is nameplate
        # arithmetic: the two are the same size, and the advice wastes a purchase.
        here = self.hw.usable_bytes
        for profile in sorted(hardware_catalog.PROFILES, key=lambda x: x.total_memory_gb):
            cand = profile.to_hardware()
            if cand.usable_bytes <= here * 1.25:
                continue
            feasible, _ = fit.rank(cand, req.context, req.concurrency)
            if feasible:
                out.append(
                    f"  · a bigger machine — {profile.name}, "
                    f"{cand.usable_bytes / 1024**3:.0f} GB usable against your "
                    f"{here / 1024**3:.0f} GB, and this workload fits on it"
                )
                break

        if len(out) == 2:
            out.append("  · nothing in reach — this workload needs a different class of machine.")
        out += ["", "Run `clickllm where <model>` to see which hardware serves a model you want."]
        return "\n".join(out)

    def optimizations(self) -> list[Suggestion]:
        """What to change that you did not ask about."""
        p = self._plan()
        return suggest(self.requirements, p) if p else []

    # --- resumable ------------------------------------------------------------

    def to_json(self) -> str:
        req = self.requirements
        return json.dumps(
            {
                "text": self.text,
                "stage": self.stage.value,
                "requirements": {
                    f: (getattr(req, f).value if f == "workload" else getattr(req, f))
                    for f in Requirements.__slots__
                    if f != "lora"
                },
                "stated": sorted(self.stated),
                "asked": sorted(self.asked),
                "model_id": self.model_id,
                "hw": self.hw.to_dict() if self.hw else None,
                "hw_source": self.hw_source,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> Session:
        d = json.loads(text)
        r = dict(d["requirements"])
        r["workload"] = Workload(r["workload"])
        hw = None
        if d.get("hw"):
            h = d["hw"]
            hw = Hardware(
                kind=h["kind"],
                name=h["name"],
                total_bytes=h["total_bytes"],
                usable_bytes=h["usable_bytes"],
                bandwidth_gbps=h.get("bandwidth_gbps"),
                cores=h.get("cores", 0),
                devices=h.get("devices", 1),
                note=h.get("note", ""),
            )
        return cls(
            text=d["text"],
            stage=Stage(d["stage"]),
            requirements=Requirements(**r),
            stated=set(d["stated"]),
            asked=set(d["asked"]),
            model_id=d["model_id"],
            hw=hw,
            hw_source=d.get("hw_source", ""),
        )


def demo() -> None:
    m4 = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * 1024**3,
        usable_bytes=96 * 1024**3,
        bandwidth_gbps=546.0,
        cores=16,
    )

    s = Session()
    t = s.tell("coding assistant for about 20 engineers, needs to feel snappy")
    # Before a machine is named there is nothing to size, and it says so rather
    # than inventing one.
    assert t.stage is Stage.HARDWARE and t.question

    t = s.on(m4)
    assert s.model_id, "a machine plus requirements is enough to choose"
    assert "M4 Max" in t.said
    # It read the sentence and can show its working.
    assert any("workload" in e for e in t.evidence), t.evidence

    # An answer exists mid-conversation, with the assumptions visible.
    early = s.answer()
    assert "RUN" in early and "THEN PROVE IT" in early
    assert "vllm" in early or "llama.cpp" in early or "mlx" in early

    # It asks only what would change the outcome, and never twice.
    first = s.step().question
    if first:
        assert s.step().question != first, "a repeated question is a session nobody finishes"

    # An explicit answer outranks a later re-reading of the prose.
    s.set(concurrency=32)
    assert s.requirements.concurrency == 32
    s.tell("also it should handle long files")
    assert s.requirements.concurrency == 32, "stated fields must survive a re-read"

    # Unknown fields are refused with the list of real ones, not ignored.
    try:
        s.set(nonsense=1)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "not a requirement" in str(e)

    # Resumable across processes.
    again = Session.from_json(s.to_json())
    assert again.requirements.concurrency == 32
    assert again.model_id == s.model_id
    assert again.hw is not None and again.hw.name == "M4 Max"
    assert again.answer() == s.answer(), "a resumed session must answer identically"

    # 32 users at 128k context genuinely does not fit in 96 GB. A refusal with
    # a route out of it is a different product from a refusal.
    stuck = s.answer()
    assert "Here is what would" in stuck, stuck
    assert "context instead" in stuck or "concurrent instead" in stuck, stuck

    # The boundary: it hands over a command and never claims to have run one.
    s.set(context=8192, concurrency=8)
    final = s.answer()
    assert "Nothing above has been run" in final

    # A machine that cannot serve the workload says so instead of pretending.
    tiny = Session()
    tiny.on(replace(m4, name="tiny", usable_bytes=2 * 1024**3))
    tiny.set(context=131_072, concurrency=64)
    assert "Nothing" in tiny.answer()

    print(f"session: stage={s.stage.value} model={s.model_id}")
    print(final)


if __name__ == "__main__":
    demo()
