"""A scaffold that does not match the contract its own module documents.

`ENTRY_POINT_GROUPS` states what each of the five plugin groups must hand back,
and `scaffold()` branched two ways — so three of the five received the GENERAL
boilerplate: a side-effecting `register()` loading a torch op library. Those
plugins install, load cleanly, and do nothing, which is the failure this module
exists to prevent.
"""

from __future__ import annotations

import ast

import pytest

from clickllm.kernels import ENTRY_POINT_GROUPS, Plugin, PluginKind, scaffold


def _init(kind: PluginKind) -> str:
    files = scaffold(Plugin(name="demo", kind=kind, target="x"))
    return next(v for k, v in files.items() if k.endswith("__init__.py"))


@pytest.mark.parametrize("kind", list(PluginKind))
def test_every_scaffold_is_valid_python(kind):
    ast.parse(_init(kind))


@pytest.mark.parametrize("kind", list(PluginKind))
def test_every_group_gets_its_own_shape(kind):
    """The load-bearing assertion: no two groups may share a template unless
    their contracts are the same. Three did, and their contracts are not."""
    shapes = {k: _init(k) for k in PluginKind}
    same = [k.name for k in PluginKind if k is not kind and shapes[k] == shapes[kind]]
    assert not same, f"{kind.name} shares its scaffold with {same}"


def test_a_stat_logger_entry_point_is_a_class_not_a_function():
    """`ENTRY_POINT_GROUPS` says "the entry point is the class itself,
    subclassing StatLoggerBase". A `register()` function resolves to a callable
    where vLLM expects a type, and the plugin is skipped."""
    src = _init(PluginKind.STAT_LOGGER)
    tree = ast.parse(src)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    assert classes, "no class emitted"
    assert any(
        isinstance(b, ast.Name) and b.id == "StatLoggerBase" for c in classes for b in c.bases
    ), src
    assert not [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register"]


def test_an_io_processor_returns_a_name_rather_than_loading_a_library():
    """ "the IOProcessor class's fully-qualified name" — a string, not a side
    effect. It used to emit `torch.ops.load_library(...)` and return None."""
    src = _init(PluginKind.IO_PROCESSOR)
    assert "torch.ops.load_library" not in src
    assert "return " in src and "MyIOProcessor" in src


def test_an_endpoint_plugin_returns_routes():
    src = _init(PluginKind.ENDPOINT)
    assert "torch.ops.load_library" not in src
    assert "APIRouter" in src and "return router" in src
    # The group is not loaded by default, and a scaffold that appears to do
    # nothing is usually that rather than the author's code.
    assert "NOT loaded by default" in src


def test_a_platform_plugin_can_still_decline():
    """Returning None is how a plugin says its hardware is absent, and it must
    not stop vLLM starting on a machine that does not have it."""
    src = _init(PluginKind.PLATFORM)
    assert "return None" in src and "MyPlatform" in src


def test_a_general_plugin_still_registers_re_entrantly():
    """The one group the old `else` was written for, unchanged: vLLM loads
    plugins in every process it spawns, so this runs once per TP worker."""
    src = _init(PluginKind.GENERAL)
    assert "torch.ops.load_library" in src
    assert "already registered in this process" in src


@pytest.mark.parametrize("kind", list(PluginKind))
def test_the_scaffold_names_the_contract_it_implements(kind):
    """Every group is documented in `ENTRY_POINT_GROUPS`; the emitted package
    should carry enough of that wording for the author to check it against the
    table without leaving the file."""
    assert kind.value in _init(kind), ENTRY_POINT_GROUPS[kind]
