"""The agentic session.

`session.demo()` walks one conversation. These pin the properties that make it a
product rather than a wizard: it never blocks, it asks only what would change
the answer, it never asks twice, it survives a restart, and it stops at the door.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from clickllm import mcp
from clickllm.hardware import Hardware
from clickllm.session import Session, Stage

M4 = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * 1024**3,
    usable_bytes=96 * 1024**3,
    bandwidth_gbps=546.0,
    cores=16,
)
H100 = Hardware(
    kind="nvidia",
    name="H100 80GB",
    total_bytes=80 * 1024**3,
    usable_bytes=72 * 1024**3,
    bandwidth_gbps=3350.0,
    cores=132,
)


def started(text: str = "coding assistant for about 20 engineers", hw: Hardware = M4) -> Session:
    s = Session()
    s.tell(text)
    s.on(hw)
    return s


# --- it never blocks -----------------------------------------------------------


def test_an_answer_exists_before_any_question_is_answered():
    """The promise that makes questions optional rather than a gate."""
    s = started()
    out = s.answer()
    assert "MODEL" in out and "RUN" in out and "THEN PROVE IT" in out


def test_assumptions_are_visible_rather_than_silent():
    s = started()
    turn = s.step()
    assert turn.assuming, "a defaulted field must be shown as an assumption"
    assert any("context" in a for a in turn.assuming)


def test_what_was_read_is_quoted_back_with_its_evidence():
    """A misreading must be obvious now, not discovered in production."""
    s = started("voice agent that has to reply in under a second")
    assert any("realtime" in e for e in s.step().evidence), s.step().evidence
    assert any('from "' in e for e in s.evidence)


def test_without_a_machine_it_says_so_instead_of_inventing_one():
    s = Session()
    turn = s.tell("coding assistant")
    assert turn.stage is Stage.HARDWARE
    assert turn.question
    assert "No machine chosen" in s.answer()


# --- it asks only what would change the answer ---------------------------------


def test_a_question_is_only_raised_when_the_outcome_would_differ():
    """Computable, not curated. If every answer produces the same deployment,
    asking is spending the user's attention for nothing."""
    s = started()
    q = s.step().question
    if q is not None:
        # The claim: some alternative answer really does change the plan.
        field = next(f for f in ("concurrency", "context", "prefix_sharing") if f in s.asked)
        alt = {"concurrency": 1, "context": 8192, "prefix_sharing": 0.8}[field]
        assert s._outcome_differs({field: alt})


def test_the_same_question_is_never_asked_twice():
    """A session that repeats itself is one nobody finishes."""
    s = started()
    seen = []
    for _ in range(5):
        q = s.step().question
        if q is None:
            break
        assert q not in seen, f"asked twice: {q}"
        seen.append(q)


def test_answering_everything_that_matters_ends_the_questions():
    """The questions end. The session does not.

    This asserted that the turn after the last question was `done`, which was
    true while `OPTIMIZE`, `DEPLOY` and `PROVE` were declared and never
    assigned. A plan is not the end of the job — the session now carries on to
    the command and to how you would prove it, and the thing that must stop is
    the *asking*.
    """
    from clickllm.session import Stage

    s = started()
    # `set()` ends in its own `step()`, so the conversation has already moved on
    # by the time the first explicit `step()` is called — which is why this
    # collects from that first call rather than assuming it starts at OPTIMIZE.
    s.set(context=8192, concurrency=8, prefix_sharing=0.8)
    first = s.step()
    assert first.question is None

    stages = [first.stage]
    for _ in range(6):
        turn = s.step()
        stages.append(turn.stage)
        assert turn.question is None, f"asked another question at {turn.stage}: {turn.question}"
        if turn.done:
            break
    else:  # pragma: no cover
        raise AssertionError(f"never finished; reached {stages}")

    assert stages[-1] == Stage.PROVE, stages
    assert Stage.DEPLOY in stages, stages
    # In declared order, never backwards: a conversation that revisits a stage
    # is one the user cannot tell they are making progress through.
    order = list(Stage)
    positions = [order.index(x) for x in stages]
    assert positions == sorted(positions), stages


def test_a_stated_field_is_never_re_asked():
    s = started()
    s.set(concurrency=32)
    for _ in range(4):
        q = s.step().question
        if q is None:
            break
        assert "in flight at once" not in q, "concurrency was stated and must not be re-asked"


# --- the user outranks the parser ----------------------------------------------


def test_an_explicit_answer_survives_a_later_re_reading_of_the_prose():
    """The user said 32 once; a later sentence must not talk them out of it."""
    s = started()
    s.set(concurrency=32)
    s.tell("also it should handle long files")
    assert s.requirements.concurrency == 32


def test_an_unknown_field_is_refused_with_the_real_ones():
    s = started()
    with pytest.raises(ValueError, match="not a requirement"):
        s.set(nonsense=1)


# --- a refusal comes with a route out ------------------------------------------


def test_nothing_fitting_produces_what_would_fit_instead():
    """ "Nothing fits" is a dead end. This product says what to change."""
    s = Session()
    s.on(H100)
    s.set(context=131_072, concurrency=64)
    out = s.answer()
    assert "Here is what would" in out
    assert "context instead" in out or "concurrent instead" in out


def test_a_suggested_machine_is_bigger_on_the_same_measure_and_actually_works():
    """Nameplate arithmetic would offer an A100 80GB to someone on an H100 80GB."""
    s = Session()
    s.on(H100)
    s.set(context=131_072, concurrency=64)
    line = next((ln for ln in s.answer().splitlines() if "bigger machine" in ln), None)
    if line is not None:
        assert "usable against your" in line
        assert "fits on it" in line


def test_a_machine_too_small_for_anything_says_so_plainly():
    s = Session()
    s.on(replace(M4, name="tiny", usable_bytes=2 * 1024**3))
    assert "Nothing" in s.answer()


# --- the default pick is a deployment, not a stunt ------------------------------


def test_the_default_model_is_not_the_biggest_thing_that_technically_fits():
    """fit.rank's head on a laptop is routinely a 70B at 96% of memory and 6
    tok/s. Correct for "what fits", wrong for "what should I deploy"."""
    s = started()
    chosen = next(c for c in s._candidates() if c.model.id == s.model_id)
    if any(not c.slow for c in s._candidates()):
        assert not chosen.slow, f"picked a slow model: {chosen.model.id}"


# --- resumable ------------------------------------------------------------------


def test_a_session_survives_a_restart_and_answers_identically():
    s = started()
    s.set(concurrency=16, context=16_384)
    again = Session.from_json(s.to_json())
    assert again.answer() == s.answer()
    assert again.stated == s.stated
    assert again.explicit == s.explicit
    assert again.asked == s.asked
    assert again.hw is not None and again.hw.name == s.hw.name


def test_a_direct_answer_survives_a_resume_and_a_later_correction():
    """`explicit` protects a direct answer from a later re-read of the prose —
    but only if it round-trips through `to_json`/`from_json`. It did not: a
    resumed session had an empty `explicit`, so `tell()` on the next process
    silently overwrote a value the user set outright with whatever the fresh
    parse of the accumulated text produced.
    """
    s = started()
    s.set(concurrency=32)
    resumed = Session.from_json(s.to_json())
    assert resumed.requirements.concurrency == 32
    resumed.tell("also for 10 engineers")
    assert resumed.requirements.concurrency == 32, "a resumed direct answer must survive a re-read"


# --- the boundary ----------------------------------------------------------------


def test_the_session_hands_over_a_command_and_never_claims_to_have_run_one():
    s = started(hw=H100)
    out = s.answer()
    assert "Nothing above has been run" in out
    assert "Deployment is yours to trigger" in out


def test_it_always_points_at_the_proof_step():
    """A deployment without a way to check it is the thing this product exists
    to stop."""
    assert "clickllm prove" in started(hw=H100).answer()


def test_no_agent_facing_tool_can_move_traffic():
    forbidden = ("cutover", "apply", "promote", "advance", "rollout", "deploy", "serve", "route")
    assert not [n for n in mcp.TOOLS if any(w in n for w in forbidden)]


# --- resuming must not silently discard a chosen machine ------------------------


def test_a_bare_resume_does_not_re_detect_hardware_over_a_saved_profile(tmp_path):
    """Caught by the automated reviewer on PR #16: cmd_build called `s.on(args.on)`
    unconditionally, so `--resume saved.json` with no `--on` silently re-detected
    the local machine and threw away a previously saved remote profile like h100.
    mcp._build already guarded this correctly; cli.cmd_build did not."""
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    saved = tmp_path / "sess.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "clickllm.cli",
            "build",
            "batch scoring overnight",
            "--on",
            "h100",
            "--save",
            str(saved),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    assert "H100" in saved.read_text()

    resumed = tmp_path / "sess2.json"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "clickllm.cli",
            "build",
            "--resume",
            str(saved),
            "--save",
            str(resumed),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    assert "H100" in resumed.read_text(), r.stdout + r.stderr


def test_the_mcp_surface_already_had_the_correct_guard():
    """The reference behaviour cmd_build was fixed to match."""
    t1 = mcp._build(description="batch scoring overnight", machine="h100")
    t2 = mcp._build(state=t1["state"])  # no machine passed on the resumed turn
    assert t2["state"]["hw"]["name"] == t1["state"]["hw"]["name"]


# --- the multi-turn agent surface -------------------------------------------------


def test_an_agent_can_carry_the_conversation_across_calls():
    t1 = mcp._build(description="coding assistant for about 20 engineers", machine="h100")
    assert t1["answer"] and t1["state"]
    t2 = mcp._build(state=t1["state"], context=8192)
    assert t2["state"]["requirements"]["context"] == 8192
    assert "context" in t2["state"]["stated"]
    t3 = mcp._build(state=t2["state"], concurrency=16)
    # Both survive: state is carried, not rebuilt.
    assert t3["state"]["requirements"]["concurrency"] == 16
    assert t3["state"]["requirements"]["context"] == 8192


def test_the_agent_surface_returns_one_question_not_a_form():
    out = mcp._build(description="coding assistant", machine="h100")
    assert out["question"] is None or isinstance(out["question"], str)


def test_the_agent_surface_always_carries_an_answer():
    """So an agent never has to make the user answer something first."""
    out = mcp._build(description="anything at all", machine="h100")
    assert out["answer"]
    assert "not an action" in out["advisory"]


# --- chaining several inputs into one turn must not swallow the best question ---


def test_a_single_build_call_surfaces_the_highest_priority_question():
    """Caught by automated review on PR #16: cmd_build and mcp._build each call
    tell() then on() (and set() for any flags given) before looking at a Turn.
    tell()/on()/set() each end in step(), and step() commits the winning
    candidate to `self.asked` as a side effect of computing it — so whichever of
    those internal calls ran first silently claimed and discarded the single
    most valuable question, and the Turn actually shown to a user carried a
    lower-priority one instead, with no error and no way to tell.

    `_apply_text`/`_apply_hardware`/`_apply_fields` mutate state without calling
    step(), so a caller that applies several inputs and then calls step() once
    gets exactly one commit — the true first candidate in priority order.
    """
    s = Session()
    s._apply_text("coding assistant for about 20 engineers, needs to feel snappy")
    s._apply_hardware(M4)
    assert s.asked == set(), "state mutation alone must never commit a question"

    turn = s.step()
    # concurrency is the first probe in _worth_asking and nothing here stated
    # it, so it must be the question that survives — not context, which is
    # what the old chain-and-discard bug produced instead.
    assert turn.question == "How many requests will be in flight at once?", turn.question
    assert s.asked == {"concurrency"}


def test_cmd_build_asks_the_same_question_the_session_would():
    """End to end through the real CLI, not just the Session class.

    Passes --on explicitly: hardware.detect() under a subprocess with a
    restricted PATH cannot find the system tools it needs and falls back to a
    0-byte "no accelerator" profile, at which nothing fits and no question is
    asked — a real difference in *this test's* environment, not a session bug.
    Naming a profile removes that variable, matching the direct verification.
    """
    import subprocess
    import sys

    src = str(Path(__file__).resolve().parents[1] / "src")
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "clickllm.cli",
            "build",
            "coding assistant for about 20 engineers, needs to feel snappy",
            "--on",
            "h100",
            "--json",
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"},
        check=True,
    )
    payload = json.loads(r.stdout)
    assert payload["question"] == "How many requests will be in flight at once?"


def test_mcp_build_asks_the_same_question_in_one_call():
    """The multi-turn agent surface must not have the same discard bug."""
    out = mcp._build(
        description="coding assistant for about 20 engineers, needs to feel snappy",
        machine="h100",
    )
    assert out["question"] == "How many requests will be in flight at once?"


def test_a_configured_lora_fleet_survives_a_resume():
    """`to_json` excluded `lora` — no comment, no fallback — and `from_json`
    never put it back.

    A session resumed from disk therefore served without the multi-LoRA fleet it
    had been configured with, and looked identical doing it. That is why nothing
    caught it: the deployment is different and nothing says so. Multi-LoRA is
    the cheapest personalisation there is, so the config most worth keeping is
    the one that was dropped.
    """
    from clickllm.engines import LoraFleet
    from clickllm.plan import Requirements, Workload
    from clickllm.session import Session

    fleet = LoraFleet((("support", "org/support"), ("sql", "org/sql")), 48, 2)
    resumed = Session.from_json(
        Session(requirements=Requirements(Workload.INTERACTIVE, 8, lora=fleet)).to_json()
    )
    assert resumed.requirements.lora == fleet

    # Control: a session with no fleet resumes with none, not with an empty one.
    assert Session.from_json(Session().to_json()).requirements.lora is None


def test_turn_does_not_spend_questions_on_intermediate_states():
    """`step()` *commits* a question by adding it to `asked`, and a question
    once asked is never asked again.

    So two `step()`s in one external turn spend one on a half-built state. The
    first draft of this test asserted the visible question would differ — it
    does not, for this input, because `concurrency` outranks `context` and is
    what gets shown either way. The damage is in the commit set: the chained
    session has silently marked `context` as asked, so it can never raise it
    later, while the single-turn session still can.

    That is the quiet failure — the answer stays correct, and a question the
    user should have been asked has been thrown away.
    """
    from clickllm.hardware import Hardware
    from clickllm.session import Session

    # A pinned machine, not the host. `detect_hardware=True` read whatever CI
    # was running on, and on a runner where nothing in the catalogue fits,
    # `step()` returns a refusal and never asks anything — so `asked` came back
    # empty and the comparison was between two empty sets. The vacuity guard at
    # the end is what caught it; without that this would have passed by
    # comparing nothing to nothing.
    #
    # Third time this repo has written a test that was partly about the runner.
    machine = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * 1024**3,
        usable_bytes=96 * 1024**3,
        bandwidth_gbps=546.0,
        cores=16,
    )

    chained = Session()
    chained.on(machine)  # step #1 — has hardware, so it really does commit
    chained.tell("a support chatbot for 20 agents")  # step #2

    single = Session()
    single.turn("a support chatbot for 20 agents", machine=machine)

    spent = chained.asked - single.asked
    assert spent == {"context"}, (
        f"expected chaining to burn 'context' on an intermediate state, spent {spent}"
    )
    assert single.asked, "the single-turn form committed nothing; the check is vacuous"


def test_turn_matches_the_chain_its_callers_hand_write():
    """`cmd_build` and `mcp._build` both do the private chain. `turn()` must be
    the same thing, or it is a third behaviour rather than a shared one."""
    from clickllm.hardware import Hardware
    from clickllm.session import Session

    machine = Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * 1024**3,
        usable_bytes=96 * 1024**3,
        bandwidth_gbps=546.0,
        cores=16,
    )

    by_hand = Session()
    by_hand._apply_text("a support chatbot for 20 agents")
    by_hand._apply_hardware(machine)
    by_hand._apply_fields(concurrency=32)
    expected = by_hand.step()

    via_turn = Session().turn("a support chatbot for 20 agents", machine=machine, concurrency=32)
    assert via_turn.stage == expected.stage
    assert via_turn.question == expected.question
    assert via_turn.said == expected.said


def test_turn_does_not_touch_hardware_it_was_not_given():
    """`machine=None` means "not supplied", not "detect the local machine".

    Conflating them would make every caller that passes only a description
    silently detect hardware, which is a filesystem and subprocess touch a
    caller did not ask for.
    """
    from clickllm.session import Session

    s = Session()
    s.turn("a support chatbot")
    assert s.hw is None, "hardware was detected without being asked for"


# --- the three stages that were declared and never assigned -----------------------


def _walk_to(session, turn, target, limit: int = 10):
    """Step until `target`, with a bound.

    Bounded deliberately. The first version of these tests used
    `while turn.stage != target and not turn.done`, and the control that stops
    the stages advancing turned every one of them into an infinite loop rather
    than a failure — a hang gives CI no signal at all, which is strictly worse
    than a red tick.
    """
    for _ in range(limit):
        if turn.stage == target or turn.done:
            return turn
        turn = session.turn("")
    raise AssertionError(f"never reached {target} in {limit} turns; stuck at {turn.stage}")


def _machine():
    from clickllm.hardware import Hardware

    gb = 1024**3
    return Hardware(
        kind="apple",
        name="M4 Max",
        total_bytes=128 * gb,
        usable_bytes=96 * gb,
        bandwidth_gbps=546.0,
        cores=16,
    )


def test_one_session_carries_you_from_a_description_to_how_you_would_prove_it():
    """`Stage.OPTIMIZE`, `DEPLOY` and `PROVE` were in the enum from the start
    and nothing ever assigned them, so every conversation stopped at a
    deployment plan — including the browser one, which is the surface where
    stopping there is least defensible."""
    from clickllm.session import Session, Stage

    s = Session()
    turn = s.turn("a support chatbot for 20 agents", machine=_machine())
    seen = []
    for _ in range(10):
        seen.append(turn.stage)
        if turn.done:
            break
        turn = s.turn("")
    else:  # pragma: no cover
        raise AssertionError(f"never finished: {seen}")

    for stage in (Stage.CONFIGURE, Stage.OPTIMIZE, Stage.DEPLOY, Stage.PROVE):
        assert stage in seen, f"{stage} was never reached: {seen}"
    assert seen[-1] == Stage.PROVE and turn.done


def test_the_deploy_stage_hands_over_a_command_and_says_it_will_not_run_it():
    """A session that deployed would be a session that can be talked into
    deploying, and the thing steering it may be an agent reading a customer's
    request log (invariant 7)."""
    from clickllm.session import Session, Stage

    s = Session()
    turn = s.turn("a chatbot for 20 agents", machine=_machine())
    turn = _walk_to(s, turn, Stage.DEPLOY)
    assert turn.stage == Stage.DEPLOY
    assert "MODEL" in turn.said and "MACHINE" in turn.said
    # `answer()`'s own closing, not a second sentence saying the same thing.
    assert "Nothing above has been run" in turn.said
    assert "without a human" in turn.said


def test_the_prove_stage_names_the_path_and_refuses_to_authorise_a_cutover():
    """Invariant 8, at the moment a user is most likely to think they are done."""
    from clickllm.session import Session, Stage

    s = Session()
    turn = s.turn("a chatbot for 20 agents", machine=_machine())
    turn = _walk_to(s, turn, Stage.PROVE)
    assert turn.stage == Stage.PROVE and turn.done
    for command in ("clickllm observe", "clickllm distill", "clickllm prove", "clickllm brief"):
        assert command in turn.said, f"{command} is missing from the path"
    assert "shadow mode" in turn.said.lower()
    assert "authorises a cutover" in turn.said


def test_no_stage_after_configure_asks_a_question():
    """One question at a time is the rule, and "no more questions" has to mean
    it — a stage that starts asking again after the plan is a form in disguise."""
    from clickllm.session import Session, Stage

    s = Session()
    turn = s.turn("a chatbot for 20 agents", machine=_machine())
    for _ in range(8):
        if turn.stage != Stage.CONFIGURE or turn.done:
            break
        turn = s.turn("")
    for _ in range(5):
        assert turn.question is None, f"{turn.stage} asked {turn.question!r}"
        if turn.done:
            break
        turn = s.turn("")


def test_a_finished_session_stays_finished():
    """Stepping past the end must not wrap around to the beginning."""
    from clickllm.session import Session, Stage

    s = Session()
    turn = s.turn("a chatbot for 20 agents", machine=_machine())
    turn = _walk_to(s, turn, Stage.PROVE)
    for _ in range(3):
        turn = s.turn("")
        assert turn.stage == Stage.PROVE and turn.done


# --- the optimizer: computed, not curated ----------------------------------------


def _small(usable_gb: int):
    from clickllm.hardware import Hardware

    gb = 1024**3
    return Hardware(
        kind="apple",
        name=f"{usable_gb}GB",
        total_bytes=(usable_gb + 5) * gb,
        usable_bytes=usable_gb * gb,
        bandwidth_gbps=400.0,
        cores=12,
    )


def test_it_says_which_setting_to_change_and_what_that_buys():
    """ "Reduce one of them" was true and useless: it is the moment a user is
    most stuck, and the message named neither the knob nor the amount."""
    from clickllm.session import Session

    s = Session()
    s.tell("a chatbot")
    s.on(_small(8))
    s.set(concurrency=2, context=8192)
    tips = s._optimizations()
    assert tips, "nothing suggested on a machine where nothing fits"
    assert any("concurrency 1" in t or "context" in t for t in tips), tips
    assert any("fits" in t for t in tips), tips


def test_a_suggestion_is_raised_only_when_the_outcome_actually_differs():
    """Computed, not curated — the same rule `_worth_asking` follows. A tip that
    does not change what you would deploy is noise dressed as advice."""
    from clickllm.session import Session

    s = Session()
    s.tell("a chatbot")
    s.on(_machine())
    s.set(concurrency=1, context=8192)
    # A 96 GB machine at concurrency 1: halving anything changes nothing.
    assert s._optimizations() == ()


def test_the_nothing_fits_turn_names_the_change_rather_than_gesturing_at_it():
    from clickllm.session import Session, Stage

    s = Session()
    s.tell("a chatbot")
    s.on(_small(8))
    s.set(concurrency=2, context=8192)
    turn = s.step()
    assert turn.stage == Stage.CHOOSE
    assert "Nothing in the catalogue fits" in turn.said
    assert "fits" in turn.said.split("not a failure.")[1], turn.said


def test_every_question_has_an_answer_that_stops_it_being_asked():
    """A question the tool asks must be answerable in the words it suggests.

    `? How long are the prompts — a few thousand tokens, or tens of thousands?`
    invited exactly the phrasings `intent._context` could not read: it required
    a `k` suffix, so "a few thousand tokens" and "about 2000 tokens" left the
    field at its default with no provenance and the question was asked again.
    Found by recording a real session, not by the suite — every existing test
    used the `k` form the parser handles.

    This is the general control rather than a fix for that one phrasing: for
    each question the session can raise, at least one natural answer must both
    set the field and stop the question. A new probe with no readable answer
    fails here.
    """
    answers = {
        "How many requests will be in flight at once?": ("about 12 at once", "concurrency"),
        "How long are the prompts — a few thousand tokens, or tens of thousands?": (
            "a few thousand tokens",
            "context",
        ),
        "Do your requests share a long system prompt? (an agent fleet usually does)": (
            "yes, they share a long system prompt",
            "prefix_sharing",
        ),
    }

    s = Session()
    turn = s.turn("a support chatbot for 20 agents", detect_hardware=False, machine=_machine())
    asked_any = False
    for _ in range(len(answers)):
        if turn.question is None:
            break
        asked_any = True
        assert turn.question in answers, (
            f"the session asks {turn.question!r} and this test has no answer for it — "
            "add one, or the question has no phrasing a reader would produce"
        )
        reply, field = answers[turn.question]
        previous = turn.question
        turn = s.turn(reply)
        assert field in s.stated or field in s.answered, (
            f"answering {previous!r} with {reply!r} recorded {field} in neither "
            "`stated` nor `answered`, so the question will be asked again"
        )
        assert turn.question != previous, (
            f"{previous!r} was asked again in the turn that consumed its answer"
        )
    assert asked_any, "no question was asked at all — this test would be vacuous"
    assert turn.question is None, f"left with unanswered question: {turn.question}"
