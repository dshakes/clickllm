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

from onpar.kernels import ENTRY_POINT_GROUPS, Plugin, PluginKind, scaffold


def _init_for(kind: PluginKind, target: str) -> str:
    files = scaffold(Plugin(name="demo", kind=kind, target=target))
    return next(v for k, v in files.items() if k.endswith("__init__.py"))


def _init(kind: PluginKind) -> str:
    return _init_for(kind, "x")


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


def test_an_io_processor_names_a_class_that_actually_exists():
    """`register()` returns `"{pkg}.processor.MyIOProcessor"` — vLLM imports
    that name. Returning a plausible-looking FQN with no `processor.py` behind
    it means selecting the plugin fails on import, not on load."""
    files = scaffold(Plugin(name="demo", kind=PluginKind.IO_PROCESSOR, target="x"))
    processor = next((v for k, v in files.items() if k.endswith("processor.py")), None)
    assert processor is not None, f"no processor.py among {list(files)}"
    tree = ast.parse(processor)
    assert any(isinstance(n, ast.ClassDef) and n.name == "MyIOProcessor" for n in tree.body), (
        processor
    )


def test_a_stat_logger_implements_the_full_abstract_contract():
    """`StatLoggerBase` declares `__init__`, `record`, `log` and
    `log_engine_initialized` as abstract, and `record` takes `mm_cache_stats`.
    A scaffold missing any of these instantiates as an abstract class, or
    raises `TypeError` the first time vLLM calls it with the real signature."""
    src = _init(PluginKind.STAT_LOGGER)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    methods = {m.name: m for m in cls.body if isinstance(m, ast.FunctionDef)}
    assert {"__init__", "record", "log", "log_engine_initialized"} <= set(methods), methods
    record_args = {a.arg for a in methods["record"].args.args + methods["record"].args.kwonlyargs}
    assert "mm_cache_stats" in record_args, record_args


def test_an_endpoint_plugin_implements_the_endpoint_plugin_protocol():
    """A bare `APIRouter` is not what the loader wants: it instantiates a
    zero-argument factory and reads `required_tasks` off what comes back,
    then calls `attach_router(app)`. A function returning a router has
    neither attribute and is skipped."""
    src = _init(PluginKind.ENDPOINT)
    tree = ast.parse(src)
    assert "torch.ops.load_library" not in src
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert classes, "no class emitted"
    methods = {
        m.name: m
        for c in classes.values()
        for m in c.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"attach_router", "init_state"} <= set(methods), methods
    assert "required_tasks" in src


def test_an_endpoint_plugins_init_state_is_awaited_not_called():
    """The lifecycle awaits `init_state(engine_client, state, args)` — a plain
    `def` satisfies "a method named init_state exists" but not the protocol:
    called with no coroutine to await, it does nothing at startup."""
    src = _init(PluginKind.ENDPOINT)
    tree = ast.parse(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    init_state = next(m for m in cls.body if getattr(m, "name", None) == "init_state")
    assert isinstance(init_state, ast.AsyncFunctionDef), "init_state must be an async def"
    params = [a.arg for a in init_state.args.args if a.arg != "self"]
    assert params == ["engine_client", "state", "args"], params
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "register" in funcs, "the entry point must stay a zero-argument factory"
    register = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "register")
    assert not register.args.args, "register() must take no arguments"
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


@pytest.mark.parametrize("kind", list(PluginKind))
def test_the_entry_point_names_something_the_module_defines(kind):
    """The gap in the first version of this fix: the module shape changed and
    the entry-point target did not.

    `STAT_LOGGER` emitted a `MyStatLogger` class while the CLI still wrote
    `pkg:register`, so the generated package installed with an entry point
    resolving to a name its own module no longer contained. It scaffolds, it
    builds, and vLLM finds nothing — the exact failure this module exists to
    prevent, reintroduced by the fix for it.
    """
    from onpar.kernels import entry_point_target

    target = entry_point_target("demo", kind)
    module, _, attr = target.partition(":")
    tree = ast.parse(_init_for(kind, target))
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    assert attr in defined, f"{kind.name}: entry point names {attr}, module defines {defined}"


def test_the_cli_and_the_scaffold_read_the_same_table():
    """One home for "what the entry point must point at". They were two places
    encoding one fact and they disagreed."""
    import inspect

    from onpar import cli
    from onpar.kernels import ENTRY_POINT_ATTR

    tree = ast.parse(inspect.getsource(cli.cmd_kernel).lstrip())
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "entry_point_target" in calls, calls

    # Over the AST, not the text: the first version of this assertion grepped
    # the source for ":register" and matched the *comment* explaining why the
    # hardcoded attribute was removed. A test that reads prose as code is the
    # same defect it is checking for.
    literals = {
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    assert not [x for x in literals if x.endswith(":register")], literals
    assert set(ENTRY_POINT_ATTR) == set(PluginKind), "every group needs an answer"
