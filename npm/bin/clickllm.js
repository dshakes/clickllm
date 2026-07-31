#!/usr/bin/env node
// A shim, not a port. `clickllm` is a Python distribution — `clickllm-cli` on
// PyPI, because PyPI refused the bare name as too close to `click-llm`. This
// exists so `npx clickllm` reaches it without a second implementation to keep
// in step.
//
// The version below is pinned to this package's own version, so `npx
// clickllm@0.1.1` runs exactly clickllm-cli 0.1.1. A floating pin would let an
// npm install silently change which Python is executed, and `npm/version.test`
// fails the build if the two ever disagree.
"use strict";

const { spawnSync } = require("node:child_process");
const { version } = require("../package.json");

const SPEC = `clickllm-cli==${version}`;
const args = process.argv.slice(2);

// Ordered by how little they leave behind. `uvx` runs it without installing
// anything permanent, which is the right default for a tool people are trying.
const RUNNERS = [
  { cmd: "uvx", args: ["--from", SPEC, "clickllm", ...args] },
  { cmd: "uv", args: ["tool", "run", "--from", SPEC, "clickllm", ...args] },
  { cmd: "pipx", args: ["run", "--spec", SPEC, "clickllm", ...args] },
];

function have(cmd) {
  const probe = spawnSync(cmd, ["--version"], { stdio: "ignore" });
  return !probe.error && probe.status === 0;
}

for (const runner of RUNNERS) {
  if (!have(runner.cmd)) continue;
  // `inherit` so the terminal is the child's: progress bars, prompts and
  // Ctrl-C all behave as if it had been invoked directly.
  const run = spawnSync(runner.cmd, runner.args, { stdio: "inherit" });
  if (run.error) {
    console.error(`clickllm: ${runner.cmd} failed to start: ${run.error.message}`);
    process.exit(70);
  }
  // A signalled child must not look like a clean exit. 128+n is the shell's
  // convention and what a CI runner upstream will expect.
  if (run.signal) process.exit(128 + (require("node:os").constants.signals[run.signal] || 0));
  process.exit(run.status === null ? 70 : run.status);
}

console.error(
  [
    "clickllm needs a Python runner and none was found.",
    "",
    "  Install uv (recommended, installs nothing permanent):",
    "    curl -LsSf https://astral.sh/uv/install.sh | sh",
    "",
    "  Or use pipx:",
    "    pipx run --spec " + SPEC + " clickllm --help",
    "",
    "  Or install the Python package directly:",
    "    pip install " + SPEC.replace("==", "==") + "",
    "",
    "This package is a shim: the tool itself is Python, published to PyPI as",
    "clickllm-cli. Nothing here reimplements it.",
  ].join("\n"),
);
process.exit(127);
