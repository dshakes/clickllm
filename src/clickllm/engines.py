"""The engine adapter — one intent, many dialects.

The planner decides *what should be true* of a deployment: reuse shared
prefixes, cap concurrency at 32, chunk the prefill. This module turns that into
the argv a specific engine actually accepts.

## Why this layer has to exist

It would be tidier if serving engines agreed on flag names. They do not, and the
disagreements are not cosmetic — verified against the current docs:

| Intent | vLLM | SGLang |
|---|---|---|
| reuse shared prefixes | `--enable-prefix-caching` | `--disable-radix-cache` |
| cap concurrent sequences | `--max-num-seqs` | `--max-running-requests` |
| context length | `--max-model-len` | `--context-length` |
| memory for weights + KV | `--gpu-memory-utilization` | `--mem-fraction-static` |
| chunk the prefill | `--enable-chunked-prefill` (bool) | `--chunked-prefill-size` (int) |

Three different *kinds* of mismatch live in that table, and only the first is
the easy one:

1. **Different names for the same thing.** A lookup table handles it.
2. **Opposite polarity.** vLLM's prefix caching is opt-in; SGLang's radix cache
   is on by default and opted *out*. A name map that did not know this would
   emit `--disable-radix-cache` when asked to *enable* prefix reuse — turning an
   optimisation into its exact inverse, silently, with a config that starts
   cleanly and runs slower.
3. **Different types.** One is a boolean, the other is a size where `-1` means
   off. There is no string substitution that gets this right.

So translation happens per engine, from intent, with the polarity and the type
encoded — never by rewriting a flag name.

## The rule about unverified flags

**A flag this module cannot cite is a flag it does not emit.**

Every entry below carries the source it was checked against. Where the docs were
truncated or ambiguous — SGLang's grammar backend and its speculative-decoding
family, at the time of writing — the adapter returns [`Unsupported`] with the
reason, and the caller surfaces a gap. It does not emit a plausible guess.

This is not caution for its own sake. The output of this module is a command a
human runs against their own hardware. A wrong flag name fails loudly and wastes
an afternoon; a *right-looking* flag with inverted meaning succeeds and quietly
costs them half their throughput for months. The second is the one worth
designing against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "SOURCES",
    "Adapter",
    "Setting",
    "Translated",
    "Unsupported",
    "adapter_for",
    "demo",
]

#: Where each dialect was verified, and when. Kept in the source rather than a
#: changelog because the flags drift and the next person needs to know what to
#: re-check rather than what to trust.
SOURCES = {
    "vllm": "https://docs.vllm.ai/en/stable/configuration/engine_args.html (checked 2026-07-27)",
    "sglang": "https://docs.sglang.io/advanced_features/server_arguments.html (checked 2026-07-27)",
}


class Setting(StrEnum):
    """What the planner wants to be true, independent of any engine's spelling."""

    CONTEXT_LENGTH = "context_length"
    MAX_CONCURRENT = "max_concurrent"
    PREFILL_CHUNK = "prefill_chunk"
    PREFIX_REUSE = "prefix_reuse"
    SPECULATIVE = "speculative"
    MEMORY_FRACTION = "memory_fraction"
    TENSOR_PARALLEL = "tensor_parallel"
    STRUCTURED_OUTPUT = "structured_output"
    QUANTIZATION = "quantization"
    KV_CACHE_DTYPE = "kv_cache_dtype"


@dataclass(frozen=True, slots=True)
class Translated:
    """Argv fragments for one setting. Empty is a valid, meaningful result.

    An empty list means *the engine already does this by default* — SGLang's
    radix cache when prefix reuse is wanted. That is a successful translation,
    not a missing one, and conflating the two is how a caller ends up emitting
    a redundant or inverted flag.
    """

    argv: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True, slots=True)
class Unsupported:
    """The engine has no verified flag for this intent.

    Carries the reason so the caller can tell "this engine cannot do it" apart
    from "we could not confirm the flag name", which are very different problems
    for whoever reads the generated config.
    """

    setting: Setting
    reason: str


class Adapter:
    """Base: an engine's dialect."""

    name: str = ""

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        """Argv for one setting, or why there is none."""
        raise NotImplementedError

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        """Full launch command, plus everything that could not be expressed.

        Returns both halves deliberately. A caller that only wanted the command
        would have no way to learn that a requested optimisation silently did
        not make it in.
        """
        raise NotImplementedError


class VllmAdapter(Adapter):
    """vLLM. Flags verified against the source in [`SOURCES`]."""

    name = "vllm"

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        match setting:
            case Setting.CONTEXT_LENGTH:
                return Translated(("--max-model-len", str(value)))
            case Setting.MAX_CONCURRENT:
                return Translated(("--max-num-seqs", str(value)))
            case Setting.PREFILL_CHUNK:
                # Two flags, not one: the boolean turns the behaviour on, the
                # size tunes it. Emitting only the size leaves it disabled.
                if not value:
                    return Translated(
                        ("--no-enable-chunked-prefill",),
                        "prefill runs whole; a long prompt can monopolise a step",
                    )
                return Translated(
                    ("--enable-chunked-prefill", "--max-num-batched-tokens", str(value))
                )
            case Setting.PREFIX_REUSE:
                # Opt-in here. Contrast SGLang below.
                return (
                    Translated(("--enable-prefix-caching",))
                    if value
                    else Translated((), "prefix caching is off by default in vLLM")
                )
            case Setting.SPECULATIVE:
                if not value or value == "off":
                    return Translated((), "speculative decoding is off by default")
                return Translated(("--speculative-config", str(value)))
            case Setting.MEMORY_FRACTION:
                return Translated(("--gpu-memory-utilization", str(value)))
            case Setting.TENSOR_PARALLEL:
                return Translated(("--tensor-parallel-size", str(value)))
            case Setting.QUANTIZATION:
                return Translated(("--quantization", str(value)))
            case Setting.KV_CACHE_DTYPE:
                return Translated(("--kv-cache-dtype", str(value)))
            case Setting.STRUCTURED_OUTPUT:
                if not value:
                    return Translated(())
                # `--guided-decoding-backend` is GONE. It is now a nested field
                # on `--structured-outputs-config`, which takes JSON. Anything
                # still emitting the old flag produces a config that will not
                # start — this repo did, until the docs were actually read.
                return Translated(
                    ("--structured-outputs-config", f'{{"backend": "{value}"}}'),
                    "nested config; the old --guided-decoding-backend no longer exists",
                )
        return Unsupported(setting, f"no mapping defined for {setting}")

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        argv, gaps = ["vllm", "serve", model], []
        for setting, value in settings.items():
            out = self.translate(setting, value)
            if isinstance(out, Unsupported):
                gaps.append(out)
            else:
                argv.extend(out.argv)
        return argv, gaps


class SglangAdapter(Adapter):
    """SGLang. Flags verified against the source in [`SOURCES`].

    Two entries are deliberately absent. The published argument page was
    truncated before the speculative-decoding family and the grammar backend, so
    neither is emitted. `--grammar-backend` is the *likely* name — it is
    referenced indirectly elsewhere in those docs — and likely is not a standard
    this module is willing to meet.
    """

    name = "sglang"

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        match setting:
            case Setting.CONTEXT_LENGTH:
                return Translated(("--context-length", str(value)))
            case Setting.MAX_CONCURRENT:
                return Translated(("--max-running-requests", str(value)))
            case Setting.PREFILL_CHUNK:
                # A size, not a boolean, and -1 is how it is turned off.
                return Translated(("--chunked-prefill-size", str(value if value else -1)))
            case Setting.PREFIX_REUSE:
                # THE INVERSION. Radix caching is on by default, so wanting it
                # means emitting nothing, and not wanting it means an explicit
                # disable. A name-mapping layer gets this exactly backwards.
                return (
                    Translated((), "radix cache is on by default in SGLang")
                    if value
                    else Translated(("--disable-radix-cache",))
                )
            case Setting.MEMORY_FRACTION:
                return Translated(
                    ("--mem-fraction-static", str(value)),
                    "covers weights and the KV pool together, unlike vLLM's split",
                )
            case Setting.TENSOR_PARALLEL:
                return Translated(("--tp-size", str(value)))
            case Setting.QUANTIZATION:
                return Translated(("--quantization", str(value)))
            case Setting.KV_CACHE_DTYPE:
                # Same flag name as vLLM, different accepted values. Passing
                # vLLM's `fp8_inc` here would be rejected at startup.
                return Translated(("--kv-cache-dtype", str(value)))
            case Setting.SPECULATIVE:
                if not value or value == "off":
                    return Translated((), "speculative decoding is off by default")
                return Unsupported(
                    setting,
                    "SGLang's speculative flags were not on the verified page; "
                    "run `python3 -m sglang.launch_server --help` and add them "
                    "rather than guessing at the family name",
                )
            case Setting.STRUCTURED_OUTPUT:
                if not value:
                    return Translated(())
                return Unsupported(
                    setting,
                    "the grammar-backend flag was not on the verified page. "
                    f"{value!r} is a valid backend value, but the flag that "
                    "carries it is unconfirmed",
                )
        return Unsupported(setting, f"no mapping defined for {setting}")

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        argv = ["python3", "-m", "sglang.launch_server", "--model-path", model]
        gaps = []
        for setting, value in settings.items():
            out = self.translate(setting, value)
            if isinstance(out, Unsupported):
                gaps.append(out)
            else:
                argv.extend(out.argv)
        return argv, gaps


_ADAPTERS: dict[str, Adapter] = {"vllm": VllmAdapter(), "sglang": SglangAdapter()}


def adapter_for(engine: str) -> Adapter | None:
    """The adapter for `engine`, or None if there is no verified dialect for it.

    `None` rather than a permissive fallback: emitting vLLM flags at llama.cpp
    because we had nothing better would produce a command that cannot run.
    """
    return _ADAPTERS.get(engine)


def demo() -> None:
    """Self-check. Run with `python -m clickllm.engines`."""
    v, s = VllmAdapter(), SglangAdapter()

    # The inversion — the thing this module exists to get right.
    assert v.translate(Setting.PREFIX_REUSE, True).argv == ("--enable-prefix-caching",)
    assert s.translate(Setting.PREFIX_REUSE, True).argv == ()
    assert v.translate(Setting.PREFIX_REUSE, False).argv == ()
    assert s.translate(Setting.PREFIX_REUSE, False).argv == ("--disable-radix-cache",)
    # Stated plainly: no engine is ever told to disable what it was asked to enable.
    for eng in (v, s):
        assert "--disable-radix-cache" not in eng.translate(Setting.PREFIX_REUSE, True).argv

    # Different types for the same intent.
    assert v.translate(Setting.PREFILL_CHUNK, 2048).argv == (
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        "2048",
    )
    assert s.translate(Setting.PREFILL_CHUNK, 2048).argv == ("--chunked-prefill-size", "2048")
    assert s.translate(Setting.PREFILL_CHUNK, 0).argv == ("--chunked-prefill-size", "-1")

    # Different names.
    assert v.translate(Setting.MAX_CONCURRENT, 32).argv == ("--max-num-seqs", "32")
    assert s.translate(Setting.MAX_CONCURRENT, 32).argv == ("--max-running-requests", "32")
    assert v.translate(Setting.CONTEXT_LENGTH, 8192).argv[0] == "--max-model-len"
    assert s.translate(Setting.CONTEXT_LENGTH, 8192).argv[0] == "--context-length"

    # The flag that no longer exists is not emitted.
    out = v.translate(Setting.STRUCTURED_OUTPUT, "xgrammar")
    assert "--guided-decoding-backend" not in out.argv, out
    assert out.argv[0] == "--structured-outputs-config"

    # Unverified means refused, with a reason — never a plausible guess.
    for setting in (Setting.STRUCTURED_OUTPUT, Setting.SPECULATIVE):
        gap = s.translate(setting, "xgrammar" if setting is Setting.STRUCTURED_OUTPUT else "eagle3")
        assert isinstance(gap, Unsupported), gap
        assert "unconfirmed" in gap.reason or "not on the verified page" in gap.reason
    assert "--grammar-backend" not in str(s.translate(Setting.STRUCTURED_OUTPUT, "xgrammar"))

    # A full command, and the gaps it could not express.
    want = {
        Setting.CONTEXT_LENGTH: 32768,
        Setting.MAX_CONCURRENT: 64,
        Setting.PREFIX_REUSE: True,
        Setting.STRUCTURED_OUTPUT: "xgrammar",
    }
    argv, gaps = v.command("Qwen/Qwen3-32B", want)
    assert argv[:3] == ["vllm", "serve", "Qwen/Qwen3-32B"]
    assert not gaps

    argv, gaps = s.command("Qwen/Qwen3-32B", want)
    assert argv[:4] == ["python3", "-m", "sglang.launch_server", "--model-path"]
    assert len(gaps) == 1 and gaps[0].setting is Setting.STRUCTURED_OUTPUT
    # The gap is returned, not swallowed: a caller cannot miss it.
    assert "--enable-prefix-caching" not in argv

    # No adapter is better than the wrong adapter.
    assert adapter_for("vllm") is not None
    assert adapter_for("llama.cpp") is None

    print("engines: ok")


if __name__ == "__main__":
    demo()
