def _detect_version() -> str:
    """The installed version, or the source tree's if this is not installed.

    Metadata first: that is what the user actually has. The pyproject fallback
    exists because the repo is normally run with `PYTHONPATH=src`, where no
    distribution is installed and `PackageNotFoundError` would otherwise make
    `--version` fail for every developer.

    The distribution is `clickllm-cli`; PyPI refused `clickllm` as too similar
    to the existing `click-llm`. The command stays `clickllm`.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("clickllm-cli")
    except PackageNotFoundError:
        pass
    import re
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
    except OSError:
        return "unknown"
    return m.group(1) if m else "unknown"


__version__ = _detect_version()
