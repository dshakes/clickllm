"""A bare `clickllm` opens a conversation — unless it would hang.

`session.py` has been a complete multi-turn engine the whole time, and the
browser has used it since the workbench conversation landed. The CLI, which is
the surface most people meet first, answered a bare invocation with an argparse
error listing twenty-six verbs.

The load-bearing half of this is the refusal: a bare `clickllm` in a script, a
Dockerfile or a CI step must keep failing exactly as it did rather than blocking
forever on a read. A command that hangs in CI is a worse first minute than a
usage error, and it is the kind of change discovered at 3am by someone who did
not make it.
"""

from __future__ import annotations

import builtins

import pytest

from clickllm import cli, hardware
from clickllm.hardware import Hardware

GB = 1024**3

#: Pinned, because a test that reads the host is partly a test of the host —
#: a mistake this repo has made three times.
MACHINE = Hardware(
    kind="apple",
    name="M4 Max",
    total_bytes=128 * GB,
    usable_bytes=96 * GB,
    bandwidth_gbps=546.0,
    cores=16,
)


@pytest.fixture
def interactive(monkeypatch):
    """Make stdin look like a terminal and script the answers."""
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def scripted(*answers: str):
        it = iter(answers)

        def fake_input(_prompt: str = "") -> str:
            try:
                return next(it)
            except StopIteration:
                raise EOFError from None

        monkeypatch.setattr(builtins, "input", fake_input)

    return scripted


# --- the refusal, which is the part that matters ---------------------------------


def test_a_non_interactive_bare_invocation_still_fails_rather_than_hanging(monkeypatch, capsys):
    """This is the whole reason the TTY check exists. A script, a Dockerfile or
    a CI step that runs `clickllm` with no arguments must keep exiting 2."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)

    def explode(*_a, **_k):
        raise AssertionError("it tried to read stdin in a non-interactive context")

    monkeypatch.setattr(builtins, "input", explode)

    assert cli.main([]) == 2
    err = capsys.readouterr().err
    assert "usage: clickllm" in err
    assert "the following arguments are required: cmd" in err


def test_the_non_interactive_message_is_the_one_it_always_printed(monkeypatch, capsys):
    """Byte-compatible with argparse's own error, so anything that greps for it
    keeps working."""
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    cli.main([])
    assert "clickllm: error: the following arguments are required: cmd" in capsys.readouterr().err


def test_every_named_subcommand_still_works(monkeypatch, capsys):
    """Making the subparser optional must not make a real command optional."""
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    assert cli.main(["fit", "--context", "8k"]) == 0
    assert "M4 Max" in capsys.readouterr().out


def test_an_unknown_subcommand_is_still_an_error(monkeypatch):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    with pytest.raises(SystemExit) as e:
        cli.main(["definitely-not-a-command"])
    assert e.value.code != 0


# --- the conversation ------------------------------------------------------------


def test_it_holds_a_conversation_and_ends_with_an_answer(interactive, capsys):
    interactive("a support chatbot for 20 agents", "about 8 at once", "short answers", "q")
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Describe what you are building" in out
    assert "M4 Max" in out
    assert "MODEL" in out and "RUN" in out, "it ended without handing over an answer"


def test_it_asks_before_it_answers(interactive, capsys):
    """Stepping first produced a full plan on defaults *and* a follow-up
    question before the user had said anything — directly under a prompt asking
    them what they are building.

    Leaving immediately therefore has nothing to report, and says so. It used to
    print "No machine chosen yet", which is not an answer and reads like a
    failure of the tool rather than an empty conversation.
    """
    interactive("q")
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "Nothing described yet" in out
    assert "MODEL" not in out, "it answered a question nobody asked"


def test_quitting_early_still_hands_over_the_best_answer_so_far(interactive, capsys):
    """A session you can interrupt and still walk away with something is the
    whole difference from a wizard."""
    interactive("a chatbot", "q")
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert "MODEL" in out and "MACHINE" in out


def test_stdin_closing_mid_conversation_is_not_a_crash(interactive, capsys):
    """The scripted answers run out, which raises EOFError — the same thing a
    closed pipe does."""
    interactive("a chatbot for 20 agents")
    assert cli.main([]) == 0
    assert "MODEL" in capsys.readouterr().out


def test_ctrl_c_prints_the_answer_and_exits_130(monkeypatch, capsys):
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    def interrupt(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)
    assert cli.main([]) == 130
    # Interrupted at the very first prompt: nothing was described, so there is
    # nothing to hand over. The next test covers Ctrl-C with work in progress,
    # which is the case that must not throw anything away.
    assert "Nothing described yet" in capsys.readouterr().out


def test_ctrl_c_mid_conversation_keeps_the_work(monkeypatch, capsys):
    """The case that matters: interrupting after describing something must
    still hand over the best answer so far."""
    monkeypatch.setattr(hardware, "detect", lambda: MACHINE)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    answers = iter(["a support chatbot for 20 agents"])

    def then_interrupt(_prompt: str = "") -> str:
        try:
            return next(answers)
        except StopIteration:
            raise KeyboardInterrupt from None

    monkeypatch.setattr(builtins, "input", then_interrupt)
    assert cli.main([]) == 130
    assert "MODEL" in capsys.readouterr().out, "Ctrl-C threw away the work"


@pytest.mark.parametrize("word", ["q", "quit", "exit", "Q", "QUIT"])
def test_the_ways_a_person_tries_to_leave(interactive, capsys, word):
    interactive("a chatbot", word)
    assert cli.main([]) == 0
    assert "MODEL" in capsys.readouterr().out


def test_every_line_of_the_answer_is_indented_the_same(interactive, capsys):
    """Indenting the first line alone left the block ragged — MODEL at six
    spaces and MACHINE at four — which reads as a rendering bug in the one
    output a user is meant to copy."""
    interactive("a chatbot for 20 agents", "q")
    cli.main([])
    block = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith(("MODEL", "MACHINE", "ENGINE", "RUN"))
    ]
    assert block, "no answer block was printed"
    indents = {len(line) - len(line.lstrip()) for line in block}
    # The exact indent, not merely a consistent one. Asserting consistency alone
    # passed when the indentation was removed entirely — every line at zero is
    # perfectly consistent — so the control did not fire and the test was
    # checking something weaker than it read as.
    assert indents == {2}, f"the answer block is at {sorted(indents)}, expected a uniform 2"
