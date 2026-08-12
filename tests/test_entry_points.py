"""The names a user types, resolved the way packaging resolves them.

Twice today a surface was broken in the one layer no test touched. The MCP
server spoke a transport no conforming client could talk to, and every test
passed because they all called `handle()` directly. The workbench's route was
checked by dict membership rather than by a request.

`[project.scripts]` is the same shape and the last one left. Every test in this
suite calls `cli.main()` in-process, so a typo in the entry-point target —
a renamed module, a renamed function, a package that stops being included —
would leave the suite green and `clickllm` failing at the shell with
`ModuleNotFoundError` for everyone who installed it.

These resolve the targets the way a console script does, from the packaging
metadata rather than from a list written here: a third entry point added
tomorrow is covered without anyone remembering to add it.
"""

from __future__ import annotations

import importlib
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).parent.parent


def _scripts() -> dict[str, str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return data.get("project", {}).get("scripts", {})


def test_there_are_entry_points_to_check():
    """The control for every test below: if `[project.scripts]` is ever empty or
    moves, they would all pass by having nothing to iterate over."""
    scripts = _scripts()
    assert scripts, "no [project.scripts] found — these tests would be vacuous"
    assert "clickllm" in scripts, scripts


@pytest.mark.parametrize("name", sorted(_scripts()))
def test_every_console_script_target_resolves(name):
    """`module:function`, imported and called for real by the generated shim.

    Resolved here the same way, so a rename that misses the metadata fails in
    CI rather than at a stranger's shell prompt.
    """
    target = _scripts()[name]
    assert ":" in target, f"{name} = {target!r} is not module:function"
    module_name, _, attr = target.partition(":")

    module = importlib.import_module(module_name)
    func = getattr(module, attr, None)
    assert func is not None, f"{name} points at {target}, but {attr} is not in {module_name}"
    assert callable(func), f"{name} points at {target}, which is not callable"


@pytest.mark.parametrize("name", sorted(_scripts()))
def test_every_console_script_module_ships_in_the_wheel(name):
    """A target that resolves from the source tree and is not packaged is the
    same defect one step later. `packages` decides what a user actually gets."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    packaged = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    module_name = _scripts()[name].partition(":")[0]
    top = module_name.split(".")[0]
    assert any(p.rstrip("/").endswith(f"/{top}") or p == top for p in packaged), (
        f"{name} points into {top}, which is not in the packaged set {packaged}"
    )
    # And the file backing it exists where the packaging says to look.
    rel = pathlib.Path(module_name.replace(".", "/"))
    assert any((ROOT / p).joinpath(*rel.parts[1:]).with_suffix(".py").exists() for p in packaged), (
        f"{module_name} has no source file under {packaged}"
    )


def test_the_two_shipped_names_are_the_ones_documented():
    """A renamed command is a broken README, a broken Homebrew formula and a
    broken MCP client config all at once — and none of those live in this repo's
    test suite."""
    scripts = _scripts()
    assert set(scripts) == {"clickllm", "clickllm-mcp"}, (
        f"the shipped command names changed to {sorted(scripts)}; the README, the "
        "Homebrew formula and every MCP client config name the old ones"
    )
