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


#: Requirement field -> the type its value must be. Coercion happens once, in
#: `Session._apply_fields`, because that is where every entry point converges.
#: `bool` is excluded deliberately: `bool("false")` is True, so a string there
#: must be a refusal rather than a silent yes. Excluding it from this map is
#: only half of that, and for a while it was the only half: a field with no
#: entry here fell through the loop below untouched, so `structured_output`
#: arrived and was *stored* as the string `"false"` — which is truthy, so
#: `plan.py:287` and `plan.py:618` both read it as yes. Saying "false" turned
#: the feature on. `_STRICT_TYPES` is the other half.
#: Fields whose value must ALREADY be the right type, because coercing them is
#: the defect rather than the fix. Checked before the coercion loop, so a
#: field listed here can never take the untouched fall-through path.
_STRICT_TYPES: dict[str, type] = {"structured_output": bool}

_REQUIREMENT_TYPES: dict[str, type] = {
    # `workload` was the one field this map was missing, and the one the error
    # for an unknown key advertises by name: "Known: concurrency, context,
    # itl_ms, lora, prefix_sharing, structured_output, ttft_ms, workload". So
    # an agent read that list, sent the documented value `workload="batch"`,
    # and got `AttributeError: 'str' object has no attribute 'value'` — the
    # message inviting the call that then broke. A StrEnum, so the coercion is
    # `Workload("batch")` and a bad value refuses by name rather than surviving
    # as a plain string until something reaches for `.value`.
    "workload": Workload,
    "concurrency": int,
    "context": int,
    "ttft_ms": int,
    "itl_ms": int,
    "prefix_sharing": float,
}


def _prove_path(model_id: str) -> list[str]:
    """The path from a running model to a receipt, in one place.

    It was in two: `answer()` printed a `clickllm prove` line and the PROVE
    stage printed the same command with the cost and window flags that make the
    saving computable at all. They had already drifted the day the flags landed
    — the deployment block was telling people to run the one form that reports
    "Saving: unknown".
    """
    return [
        "  clickllm observe                       record real requests, redacted before storage",
        "  clickllm distill                       turn them into an eval set",
        f"  clickllm prove evalset.json --candidate {model_id} \\",
        "      --incumbent <your-current-model> --incumbent-cost <$/mo> \\",
        "      --candidate-cost <$/mo> --traffic-window '14 days'",
        "  clickllm brief receipt.json --out brief.html",
    ]


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

    def turn(
        self,
        description: str = "",
        machine: str | Hardware | None = None,
        *,
        detect_hardware: bool = False,
        **fields: object,
    ) -> Turn:
        """One external turn: apply everything the caller has, then step once.

        The public form of the chain every caller already writes by hand.
        `cli.cmd_build` and `mcp._build` each do `_apply_text` →
        `_apply_hardware` → `_apply_fields` → exactly one `step()`, using the
        private mutators, because the public `tell()`/`on()`/`set()` each end in
        their own `step()`.

        That matters more than it looks. `step()` calls `_worth_asking()`, which
        *commits* a question by adding it to `asked` — so two `step()`s in one
        external turn burn the best question on an intermediate state and show
        the user a lower-priority one. A third caller doing the chain slightly
        differently is a bug nobody would see: the answer stays correct and the
        question quietly gets worse.

        So the rule stops being something callers must know and becomes
        something this method does.

        Args:
            description: plain language, appended to what has been said.
            machine: a profile id or a `Hardware`. Passed only when the caller
                has one — `None` here means "not supplied", which is why
                detecting the local machine needs the explicit flag below rather
                than being what `None` happens to mean.
            detect_hardware: detect the local machine when `machine` is None.
            **fields: direct answers, e.g. `concurrency=32`.

        Returns:
            The `Turn` — one question at most, and the best one.
        """
        if description:
            self._apply_text(description)
        if machine is not None or detect_hardware:
            self._apply_hardware(machine)
        if fields:
            self._apply_fields(**fields)
        return self.step()

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
        # Coerce HERE, not in the callers. `cmd_build` coerced with an explicit
        # allowlist before calling this; `mcp._build` forwarded raw JSON values
        # straight through, so `{"concurrency": "8"}` from an agent became a
        # string where every downstream comparison expects a number — silently
        # breaking workload-conditional planning rather than failing.
        #
        # Same lesson as ADR-0011: the constraint belongs to the thing it
        # protects, not to whichever surface happened to reach it. This is the
        # one funnel both paths pass through.
        coerced: dict[str, object] = {}
        for key, value in fields.items():
            strict = _STRICT_TYPES.get(key)
            # `None` is refused here too, unlike the optional numeric fields.
            # `structured_output` is not optional, so there is no "unset" for
            # `None` to mean — storing it would put a `null` in a field the
            # plan and the receipt both declare as a bool.
            if strict is not None and not isinstance(value, strict):
                raise ValueError(
                    f"{key} must be a {strict.__name__}, got {value!r}. "
                    f"Coercing it would be the bug: bool({value!r}) is "
                    f"{bool(value)}, so a value meaning no would read as yes."
                )
            want = _REQUIREMENT_TYPES.get(key)
            if want is None or value is None or isinstance(value, want):
                coerced[key] = value
                continue
            try:
                coerced[key] = want(value)  # type: ignore[operator]
            except (TypeError, ValueError) as e:
                raise ValueError(f"{key} must be {want.__name__}, got {value!r}") from e
        self.requirements = replace(self.requirements, **coerced)
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

    def _better_at(self, **override: int) -> tuple[str, str] | None:
        """(what to change, what it buys) if a neighbouring setting changes the answer.

        Computed, not curated — the same rule `_worth_asking` follows. The plan
        is re-solved at the neighbouring setting and the suggestion is raised
        only when the *outcome* differs: a different model, or the same model
        stopping being slow. A tip that does not change what you would deploy is
        noise dressed as advice.
        """
        req = self.requirements
        now = self._candidates()
        # `now` being empty is not a reason to skip — it is the most useful case
        # there is. On a 16 GB machine at concurrency 4 nothing fits at all, and
        # halving it makes a model fit; excluding that was excluding the one
        # suggestion a stuck user actually needs. Over a 240-configuration sweep
        # this fires 27 times and the pick-changing case fires 8, so both earn
        # their place.
        mine = self._default_pick(now) if now else None

        from . import fit as _fit

        assert self.hw is not None
        context = int(override.get("context", req.context))
        concurrency = int(override.get("concurrency", req.concurrency))
        if (context, concurrency) == (req.context, req.concurrency):
            return None
        # `rank` returns (feasible, rejected). Unpacked, not indexed — the first
        # version bound the whole tuple and `_default_pick` then asked a list for
        # `.slow`, which is the kind of mistake that only shows up at the one
        # input where the branch is reached.
        theirs, _rejected = _fit.rank(self.hw, context, concurrency)
        if not theirs:
            return None
        best = self._default_pick(theirs)

        what = f"context {context:,}" if "context" in override else f"concurrency {concurrency}"
        if mine is None:
            return (what, f"{best.model.name} fits — nothing does at your current settings")
        if best.model.id != mine.model.id:
            return (what, f"{best.model.name} becomes the pick instead of {mine.model.name}")
        if mine.slow and not best.slow:
            return (what, f"{best.model.name} stops being throughput-limited")
        return None

    def _optimizations(self) -> tuple[str, ...]:
        """What to change that you did not ask about, and only if it matters."""
        req = self.requirements
        out = []
        for override in (
            {"concurrency": max(1, req.concurrency // 2)},
            {"context": max(1024, req.context // 2)},
        ):
            found = self._better_at(**override)
            if found:
                out.append(f"at {found[0]}, {found[1]}")
        return tuple(out)

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
                    f"That is an answer, not a failure."
                    # Which change, and what it buys — computed, not gestured at.
                    # "Reduce one of them" was true and useless: it is the moment
                    # a user is most stuck and the message named neither the knob
                    # nor the amount.
                    + (
                        " " + "; ".join(self._optimizations()).capitalize() + "."
                        if self._optimizations()
                        else " Reduce one of them, or use a bigger machine."
                    )
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
        if question is not None or self.stage == Stage.CONFIGURE:
            # Still configuring. `done` stays False once the later stages exist:
            # a plan is not the end of the job, and saying so here is what let
            # `OPTIMIZE`, `DEPLOY` and `PROVE` be declared and never assigned.
            if question is None:
                self.stage = Stage.OPTIMIZE
            return Turn(
                stage=Stage.CONFIGURE,
                said=said,
                question=question,
                evidence=self.evidence,
                assuming=assuming,
                done=False,
            )

        return self._after_configure(said, assuming)

    def _after_configure(self, said: str, assuming: tuple[str, ...]) -> Turn:
        """The three stages that were declared and never reached.

        `Stage.OPTIMIZE`, `DEPLOY` and `PROVE` existed in the enum from the
        start and nothing ever assigned them, so every conversation stopped at a
        deployment plan — including the browser one, which is the surface where
        stopping there is least defensible. One session should carry someone from
        "what am I trying to do" to a receipt.

        Each is one turn, and each ends in something the *human* does. Nothing
        here runs a command, and nothing here claims a migration is justified.
        """
        if self.stage == Stage.OPTIMIZE:
            self.stage = Stage.DEPLOY
            tips = self._optimizations()
            return Turn(
                stage=Stage.OPTIMIZE,
                said=(
                    said
                    + "\n\nBefore you run it: "
                    + (
                        "; ".join(tips) + "."
                        if tips
                        else (
                            "nothing you did not ask about would change this — the "
                            "settings you gave are the ones that matter here."
                        )
                    )
                ),
                evidence=self.evidence,
                assuming=assuming,
                done=False,
            )

        if self.stage == Stage.DEPLOY:
            self.stage = Stage.PROVE
            return Turn(
                stage=Stage.DEPLOY,
                # `answer()` is the command, and it is *handed over*. This tool
                # does not run it: a session that deployed would be a session
                # that can be talked into deploying, and the thing steering it
                # may be an agent reading a customer's request log (invariant 7).
                # No extra sentence: `answer()` already closes with "Nothing
                # above has been run. Deployment is yours to trigger, and no
                # eval result moves production traffic without a human." Saying
                # it twice in one turn is the duplicated fact this repo keeps
                # finding, in the output rather than in the code.
                said=self.answer(),
                evidence=self.evidence,
                assuming=assuming,
                done=False,
            )

        self.stage = Stage.PROVE
        return Turn(
            stage=Stage.PROVE,
            said=(
                "How you will know it is good enough — on your traffic, not a benchmark:\n"
                + "\n".join(_prove_path(self.model_id or "<model>"))
                + "\n\n"
                "The receipt tells you which kinds of request are safe to move and which "
                "must stay. Nothing in it authorises a cutover on its own — shadow mode does."
            ),
            evidence=self.evidence,
            assuming=assuming,
            done=True,
        )

    def answer(self) -> str:
        """The best deployment available right now, whatever is still unknown.

        Callable at any point. This is the promise that makes the questions
        optional rather than a gate.
        """
        if self.hw is None:
            # Surface-neutral. `on()` is a Python API call, and this string is
            # what the CLI and the MCP server hand a user who has no machine
            # chosen — neither of whom can call it.
            return "No machine chosen yet — detect this one, or name a hardware profile."
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
            *_prove_path(best.model.id),
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
                # `lora` was excluded here with no comment and no fallback, and
                # `from_json` never put it back — so a resumed session silently
                # served without the multi-LoRA fleet it had been configured
                # with. It looks identical on resume, which is why nothing
                # caught it: the deployment is different and nothing says so.
                "requirements": {
                    f: (getattr(req, f).value if f == "workload" else getattr(req, f))
                    for f in Requirements.__slots__
                    if f != "lora"
                },
                "lora": (
                    {
                        "adapters": [list(a) for a in req.lora.adapters],
                        "max_rank": req.lora.max_rank,
                        "max_concurrent": req.lora.max_concurrent,
                    }
                    if req.lora
                    else None
                ),
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
        if lora := d.get("lora"):
            from clickllm.engines import LoraFleet

            r["lora"] = LoraFleet(
                adapters=tuple(tuple(a) for a in lora["adapters"]),
                max_rank=lora["max_rank"],
                max_concurrent=lora["max_concurrent"],
            )
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
        out = cls(
            text=d["text"],
            stage=Stage(d["stage"]),
            requirements=Requirements(**r),
            stated=set(d["stated"]),
            asked=set(d["asked"]),
            model_id=d["model_id"],
            hw=hw,
            hw_source=d.get("hw_source", ""),
        )
        # Recomputed, not serialised. `evidence` is "what was read out of your
        # sentence, quoted back with the words that produced it" — one of this
        # module's three stated design rules — and a resumed session had none of
        # it, because `to_json` never wrote it and `from_json` never re-derived
        # it. The prose that produced it *is* restored, so the trail was
        # recoverable and simply was not recovered.
        #
        # Re-reading beats storing for the same reason the receipt derives its
        # intervals: a serialised copy can drift from the text it claims to
        # explain, and then the quotes cite a sentence nobody wrote. Only the
        # evidence is recomputed — `requirements` are restored as saved, because
        # a field the user set explicitly must not be re-read out of the prose
        # and overridden.
        if out.text:
            out.evidence = tuple(i.render() for i in intent.read(out.text).inferred)
        return out


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
