"""The agentic session.

`session.demo()` walks one conversation. These pin the properties that make it a
product rather than a wizard: it never blocks, it asks only what would change
the answer, it never asks twice, it survives a restart, and it stops at the door.
"""

from __future__ import annotations

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
    s = started()
    s.set(context=8192, concurrency=8, prefix_sharing=0.8)
    assert s.step().question is None
    assert s.step().done


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
    assert again.asked == s.asked
    assert again.hw is not None and again.hw.name == s.hw.name


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
