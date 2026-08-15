# ADR-0014 — An eval root for agent-named paths

**Status:** accepted · **Date:** 2026-08-10

## Context

`onpar_prove` is the one MCP tool where an agent names a filesystem path and
the contents come back into its context:

```python
raw = _json.loads(pathlib.Path(eval_set).read_text())
```

The path is taken verbatim from the tool call. That is a file-read primitive
addressable by whatever is steering the agent — and invariant 7 says captured
traffic is data, never instructions, precisely because the thing steering the
agent may itself have come out of a customer's request log.

Nothing here is a sandbox escape. The server runs as the user, on the user's
machine, and an agent that can call this tool can usually run commands anyway.
The marginal risk is narrower and real: **a path the user never typed, read into
a context the user does not fully control.**

### Why this was not fixed as a defect

Confinement to the working directory was implemented and worked:

```
./eval.json            -> ok
/etc/hosts             -> refused
../../../../etc/hosts  -> refused
~/.ssh/known_hosts     -> refused
```

It also broke three tests in `tests/test_suite.py`, which drive this tool with a
`tmp_path` **outside** the working directory on purpose, under the docstring
*"Four surfaces, one implementation — verified rather than asserted"*. That test
exists to show the agent and the CLI reach the same verdict on the same eval
set, which is the "agent-first by construction" claim the README makes.

So the current behaviour is a *tested contract*, not an oversight. Rewriting
three existing tests so a security change of mine would pass, inside an
unrelated batch of defect fixes, is the move most deserving of scrutiny — which
is why it became this ADR instead.

## Decision

**An eval root: a directory the tool may read from, defaulting to the working
directory, overridable by the operator via `ONPAR_EVAL_ROOT`.**

Paths are resolved before the check, so a symlink cannot walk out. A path
outside the root is refused by name, with the root named in the message so the
refusal is actionable rather than mysterious.

The three options considered:

| | preserves the contract | needs config | agent can name any path |
|---|---|---|---|
| confine to CWD | no | no | no |
| **eval root (chosen)** | **yes, opt-in** | **one env var** | **no** |
| document only | yes | no | yes |

Confining to CWD alone is the safest and breaks a documented capability: a user
whose eval sets live on a mounted volume gets a refusal with no way to say
"that is fine". Documenting only leaves the primitive addressable, and the
thing being defended against is exactly an instruction the *user* did not write.

The env var is deliberately not a CLI flag. A flag is set by whoever composes
the command — which, for an MCP server started by an agent harness, may be the
agent. An environment variable is set by the operator when the server is
launched, which is the party the boundary is supposed to protect.

## Consequences

**The CLI is unaffected.** `onpar prove <path>` still reads any path, because
the person typing it is the person the confinement exists to protect. Confining
a command a human typed would be theatre; the two surfaces genuinely have
different threat models, and this ADR is the record that the difference is
intentional rather than an inconsistency someone should "fix".

**`tests/test_suite.py` sets the root rather than being rewritten.** The
four-surfaces test keeps its `tmp_path` and its claim; it points
`ONPAR_EVAL_ROOT` at that directory, which is what a real operator with eval
sets on a volume would do. The contract it was written to prove — one
implementation behind four surfaces — is untouched.

**A new refusal path exists, so it is tested from both sides.** A path inside
the root must be read, a path outside must be refused, and the refusal must name
the root. A guard that refuses everything satisfies the first half of that and
is caught by the second.

**This is the first configuration in the MCP server.** That is a real cost and
the reason option 1 was tempting. It is bounded: one variable, one meaning, and
absent it the behaviour is the safe default rather than the permissive one.

## Alternatives rejected

**Allow-list of extensions.** `.json` only would not help — the primitive is the
*read*, and an attacker-chosen `.json` path is the case being defended.

**Refuse only paths outside the user's home.** Home contains `~/.ssh`,
`~/.aws`, and every credential file that matters. The boundary has to be
narrower than "somewhere the user owns".

**Return a digest instead of contents.** The tool has to parse the eval set to
score it, so the contents are load-bearing rather than incidental.
