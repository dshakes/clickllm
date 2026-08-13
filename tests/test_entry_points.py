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


# --- the shopfront ---------------------------------------------------------------


def test_the_package_ships_a_description():
    """1.1.0 published to PyPI with a completely empty project page.

    `pyproject.toml` had no `readme` field, so the metadata carried a one-line
    summary and nothing else — on the most-linked page this project has. The
    registry confirmed it: `description length: 0`.

    Nothing caught it because every test built the wheel and checked the *code*
    inside it. The metadata is what a human sees, and no test looked at it —
    the same shape as the transport nobody exercised.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    readme = data["project"].get("readme")
    assert readme, "no readme in [project] — the PyPI page will be blank"
    body = (ROOT / readme).read_text()
    assert len(body) > 1500, f"{readme} is {len(body)} bytes; that is not a shopfront"
    assert "clickllm-cli" in body, "it must name the install target, not just the command"


def test_the_pypi_page_has_no_links_that_only_work_on_github():
    """The reason this is a separate file from the README.

    The README carries thirteen relative image sources that resolve against the
    repository. On PyPI they resolve against nothing, so a copy of it would show
    thirteen broken images on the page a launch points at.
    """
    import re

    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    body = (ROOT / data["project"]["readme"]).read_text()

    relative = [
        m
        for m in re.findall(r'(?:src|href)="([^"]+)"', body) + re.findall(r"]\(([^)]+)\)", body)
        if not m.startswith(("http://", "https://", "#", "mailto:"))
    ]
    assert not relative, f"these resolve only inside the repo: {relative}"


def test_the_pypi_page_does_not_republish_a_number_the_solver_disowns():
    """The README published 119 tok/s where the solver produces 60. A shopfront
    that repeats a retired figure is worse than one that is blank."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    body = (ROOT / data["project"]["readme"]).read_text()
    assert "119" not in body, "the retired throughput figure reappeared on the PyPI page"
    if "tok/s" in body:
        assert "roofline" in body or "not a measurement" in body, (
            "throughput is published without saying it is a projection"
        )
