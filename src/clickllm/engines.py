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

import json
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

#: Model families that carry their own multi-token-prediction head. For these,
#: `mtp` reuses weights already in the checkpoint, so it needs no draft model —
#: proposing `eagle3` instead would ask for a draft that does not exist.
#: Verified against docs.vllm.ai/en/stable/features/speculative_decoding/mtp,
#: 2026-07-28. Matched on substrings because catalogue ids carry version suffixes.
MTP_NATIVE = ("deepseek-v3", "deepseek-v4", "mimo", "longcat", "glm-4.5", "glm-5")


#: Speculative methods that load a SEPARATE draft checkpoint, so a bare method
#: name is not a runnable config. `mtp` is absent on purpose: those families
#: carry the head in the checkpoint already, which is the whole point of
#: `MTP_NATIVE`. Verified against vLLM's speculative-decoding docs, 2026-07-28.
DRAFT_REQUIRED = frozenset({"eagle", "eagle3", "draft_model", "medusa"})


#: Ranks vLLM's `--max-lora-rank` actually accepts. It is a choice list, not a
#: free integer, so an adapter trained at rank 48 is rejected at startup unless
#: it is rounded UP to the next accepted value. Rounding down would silently
#: truncate the adapter.
#: Verified: docs.vllm.ai/en/stable/configuration/engine_args (2026-07-28).
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


#: Catalogue precision label -> the quantisation *method* vLLM and SGLang accept
#: under `--quantization`. Deliberately almost empty: `fp8` is the only label
#: this repo uses that is also a method name. `q4`/`q8` describe how big the
#: weights are, not how they were quantised — awq, gptq, bitsandbytes and
#: compressed-tensors are all "4-bit" and are not interchangeable. The method
#: is a property of the checkpoint, which both engines read from its config, so
#: the honest answer for everything else is to emit no flag at all.
QUANT_METHODS = {"fp8": "fp8"}


@dataclass(frozen=True, slots=True)
class LoraFleet:
    """Adapters to serve over one set of base weights.

    Multi-LoRA is the cheapest personalisation there is — one copy of the base
    model in memory, N adapters swapped per request — and it is routinely left
    on the table because the flags are per-engine and easy to get subtly wrong.

    Attributes:
        adapters: display name -> path or Hugging Face id. The name is what
            callers put in the request's ``model`` field.
        max_rank: the largest rank across the adapters.
        max_concurrent: how many distinct adapters may appear in one batch.
    """

    adapters: tuple[tuple[str, str], ...] = ()
    max_rank: int = 16
    max_concurrent: int = 1

    def vllm_rank(self) -> int:
        """The rank rounded UP to a value vLLM accepts.

        Up, never down: a rank below what the adapter was trained at does not
        degrade it, it refuses to load it.

        Raises:
            ValueError: if the rank exceeds the largest accepted value, which is
                a real refusal rather than something to clamp silently.
        """
        for r in VLLM_LORA_RANKS:
            if r >= self.max_rank:
                return r
        raise ValueError(
            f"LoRA rank {self.max_rank} exceeds the largest rank vLLM accepts "
            f"({VLLM_LORA_RANKS[-1]}); this adapter cannot be served as-is"
        )


def _speculative_json(value: object) -> str:
    """Render the speculative knob as the JSON object vLLM actually parses.

    Accepts either a bare method name (`"eagle3"`, `"mtp"`) or an already-formed
    mapping. `num_speculative_tokens` defaults to 1 because vLLM's own docs call
    that "a reasonable starting point", and because for MTP a value above 1 runs
    the same head repeatedly and lowers the acceptance rate.
    """
    spec = dict(value) if isinstance(value, dict) else {"method": str(value)}
    spec.setdefault("num_speculative_tokens", 1)
    # Compact separators: this ends up on a command line a human has to read.
    return json.dumps(spec, separators=(",", ":"), sort_keys=True)


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
    # Docs only. `vllm serve --help` initialises hardware before printing, so
    # it cannot be interrogated on a GPU-less CI runner — the flags below are
    # NOT verified against the binary the way MLX's are. Run
    # `CLICKLLM_REQUIRE_ENGINES=1 pytest tests/test_engine_flags.py -k vllm`
    # on a GPU box to close that gap.
    "vllm": "https://docs.vllm.ai/en/stable/configuration/engine_args.html (checked 2026-07-27)",
    "sglang": "https://docs.sglang.io/advanced_features/server_arguments.html (checked 2026-07-27)",
    # Not a docs page. `mlx_lm.server --help` on the installed wheel, which is the
    # only source that cannot drift from the binary you are about to run — and the
    # reason every Unsupported below names a flag that genuinely is not in it.
    "mlx": "mlx_lm.server --help, mlx-lm 0.31.3 (checked 2026-07-28)",
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
    #: Serve N fine-tuned adapters over one set of base weights. The value is a
    #: `LoraFleet`; anything else is a programming error rather than a setting.
    LORA_FLEET = "lora_fleet"


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
    #: Environment this setting requires, as (name, value) pairs.
    #:
    #: Added for Ollama, which has exactly one flag — `--help`. Everything it
    #: can be told is told through OLLAMA_NUM_PARALLEL, OLLAMA_CONTEXT_LENGTH
    #: and their siblings. Without this field every Ollama setting would have to
    #: translate to `Unsupported`, which would be a lie: it supports
    #: concurrency and context perfectly well, just not through argv.
    #:
    #: Rendered as a shell prefix (`OLLAMA_NUM_PARALLEL=4 ollama serve`), which
    #: is native, standalone, and pasteable — not a wrapper script (NFR-4).
    env: tuple[tuple[str, str], ...] = ()


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

    #: How to ask this engine to declare its own flags. Not the flag list — the
    #: question. The answer is read off the installed binary at check time by
    #: [`installed_flags`], so a table in this file can never drift from the
    #: engine you are about to run.
    help_argv: tuple[str, ...] = ()

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        """Argv for one setting, or why there is none."""
        raise NotImplementedError

    def environment(self, settings: dict[Setting, object]) -> tuple[tuple[str, str], ...]:
        """Environment this configuration needs. Empty for flag-driven engines."""
        return ()

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
    help_argv = ("vllm", "serve", "--help")

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
                # vLLM takes a JSON object here, not a bare method name. Emitting
                # `--speculative-config eagle3` produces a command that fails at
                # startup — the same class of defect as the renamed
                # --guided-decoding-backend, and invisible to any test that only
                # asserts the flag is present.
                # Verified: docs.vllm.ai/en/stable/features/speculative_decoding/mtp
                # (checked 2026-07-28).
                spec = json.loads(_speculative_json(value))
                if spec.get("method") in DRAFT_REQUIRED and not spec.get("model"):
                    # One layer below the bug above, and worse. EAGLE3 is a draft
                    # *model*, not a mode: every example in vLLM's docs names a
                    # trained head ("nvidia/Llama-3.3-70B-Instruct-Eagle3"), and
                    # none can be derived from the base model's name.
                    #
                    # `{"method":"eagle3","num_speculative_tokens":1}` is valid
                    # JSON made of real flags, and vLLM cannot start from it —
                    # it has no way to know which head to load. No argv check
                    # catches this, because nothing about the argv is wrong.
                    return Unsupported(
                        setting,
                        f"{spec['method']} needs a trained draft checkpoint and "
                        "none was named; a draft model cannot be inferred from "
                        "the base model. Pass a speculative config carrying "
                        '"model": "<org>/<draft-repo>" to enable it',
                    )
                return Translated(("--speculative-config", _speculative_json(value)))
            case Setting.LORA_FLEET:
                # Verified: docs.vllm.ai/en/stable/features/lora and
                # .../configuration/engine_args (checked 2026-07-28).
                if not isinstance(value, LoraFleet) or not value.adapters:
                    return Translated((), "no LoRA adapters to serve")
                argv = ["--enable-lora"]
                argv += ["--lora-modules", *(f"{n}={p}" for n, p in value.adapters)]
                argv += ["--max-lora-rank", str(value.vllm_rank())]
                # --max-loras is adapters *per batch* and defaults to 1, so
                # loading eight and leaving it unset serialises seven of them.
                argv += ["--max-loras", str(value.max_concurrent)]
                return Translated(tuple(argv))
            case Setting.MEMORY_FRACTION:
                return Translated(("--gpu-memory-utilization", str(value)))
            case Setting.TENSOR_PARALLEL:
                return Translated(("--tensor-parallel-size", str(value)))
            case Setting.QUANTIZATION:
                # `--quantization` names a METHOD (awq, gptq, fp8,
                # compressed-tensors...), not a precision. This repo's q4/q8 are
                # *sizing* labels, so passing one emits `--quantization q4`,
                # which the server rejects at startup against its choice list.
                #
                # Same shape as the `--speculative-config eagle3` defect, and
                # invisible to `unknown_flags` for the same reason: the flag is
                # real, only the value is nonsense. Nothing that inspects argv
                # can catch it, so it has to be refused here.
                method = QUANT_METHODS.get(str(value))
                if method is None:
                    return Unsupported(
                        setting,
                        f"{value} is a precision, not a quantisation method; "
                        "serve a checkpoint already in that format and the "
                        "engine reads the method from its config",
                    )
                return Translated(("--quantization", method))
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
    help_argv = ("python3", "-m", "sglang.launch_server", "--help")

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
                # `--quantization` names a METHOD (awq, gptq, fp8,
                # compressed-tensors...), not a precision. This repo's q4/q8 are
                # *sizing* labels, so passing one emits `--quantization q4`,
                # which the server rejects at startup against its choice list.
                #
                # Same shape as the `--speculative-config eagle3` defect, and
                # invisible to `unknown_flags` for the same reason: the flag is
                # real, only the value is nonsense. Nothing that inspects argv
                # can catch it, so it has to be refused here.
                method = QUANT_METHODS.get(str(value))
                if method is None:
                    return Unsupported(
                        setting,
                        f"{value} is a precision, not a quantisation method; "
                        "serve a checkpoint already in that format and the "
                        "engine reads the method from its config",
                    )
                return Translated(("--quantization", method))
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
            case Setting.LORA_FLEET:
                # Same intent, a genuinely different dialect: `--lora-paths`
                # rather than `--lora-modules`, and the batch cap is
                # `--max-loras-per-batch`, which must leave room for the base
                # model as well as the adapters.
                # Verified: docs.sglang.io/docs/advanced_features/lora (2026-07-28).
                if not isinstance(value, LoraFleet) or not value.adapters:
                    return Translated((), "no LoRA adapters to serve")
                argv = ["--enable-lora"]
                argv += ["--lora-paths", *(f"{n}={p}" for n, p in value.adapters)]
                argv += ["--max-lora-rank", str(value.max_rank)]
                argv += ["--max-loras-per-batch", str(value.max_concurrent + 1)]
                return Translated(tuple(argv))
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


class MlxAdapter(Adapter):
    """MLX on Apple silicon. Flags verified against the binary in [`SOURCES`].

    This adapter says "no" far more than the CUDA ones, and that is the accurate
    answer rather than a gap to be filled in later. MLX is a different machine:
    unified memory means there is no fraction of VRAM to reserve, and the
    published server takes no context-length flag at all.

    The one that trips people up is quantisation. On vLLM you serve the base
    repo and pass `--quantization`. On MLX the precision is *baked into the
    weights you download* — `mlx-community/Qwen3-30B-A3B-8bit` is a different
    repo from the 4-bit one, and there is no serve-time flag that converts
    between them. So `QUANTIZATION` here is not a missing feature; it is a
    setting that has already been spent by the time the server starts, which is
    why choosing the model repo is a resolution step and not string formatting.
    """

    name = "mlx"
    help_argv = ("mlx_lm.server", "--help")

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        match setting:
            case Setting.MAX_CONCURRENT:
                # Two flags, like vLLM's chunked prefill: one governs how many
                # requests decode together, the other how many prompts prefill
                # together. Setting only the first leaves prefill serialised,
                # which is the half that dominates time-to-first-token.
                return Translated(
                    (
                        "--decode-concurrency",
                        str(value),
                        "--prompt-concurrency",
                        str(value),
                    )
                )
            case Setting.PREFILL_CHUNK:
                if not value:
                    return Unsupported(
                        setting,
                        "mlx always chunks prefill; there is no flag to turn it off",
                    )
                return Translated(("--prefill-step-size", str(value)))
            case Setting.PREFIX_REUSE:
                # Empty argv is a successful translation — see `Translated`. MLX
                # keeps a prompt cache without being asked, exactly as SGLang's
                # radix cache does; emitting a flag here would imply we had
                # turned something on that was already on.
                return Translated(
                    (),
                    "mlx keeps a prompt cache by default; "
                    "--prompt-cache-size and --prompt-cache-bytes bound it",
                )
            case Setting.QUANTIZATION:
                return Unsupported(
                    setting,
                    f"mlx bakes precision into the weights: serve an -{value} repo "
                    "rather than converting at start-up",
                )
            case Setting.CONTEXT_LENGTH:
                return Unsupported(
                    setting,
                    "mlx_lm.server takes no context-length flag; context is bounded "
                    "by the model config and by unified memory",
                )
            case Setting.MEMORY_FRACTION:
                return Unsupported(
                    setting,
                    "unified memory: there is no separate VRAM pool to reserve a fraction of",
                )
            case Setting.TENSOR_PARALLEL:
                return Unsupported(
                    setting,
                    "no tensor-parallel size flag; --pipeline selects pipelining "
                    "instead, which is a different topology and not a degree",
                )
            case Setting.KV_CACHE_DTYPE:
                return Unsupported(setting, "no kv-cache dtype flag")
            case Setting.STRUCTURED_OUTPUT:
                return Unsupported(setting, "no grammar or structured-output flag")
            case Setting.SPECULATIVE:
                # --draft-model wants a model to load, not a method name. vLLM's
                # JSON speculative config has no meaning here, and passing one
                # would produce a server that cannot start.
                draft = value.get("model") if isinstance(value, dict) else value
                # A loadable draft is a repo id or a path. `eagle3` and `mtp` are
                # neither — they are the planner's method names, and the planner
                # emits them for every engine because vLLM accepts them. Passing
                # one through here produces `--draft-model eagle3`, which is an
                # accepted *flag* with an unresolvable value: it starts, fails to
                # fetch, and exits. `unknown_flags` cannot catch that, because
                # the flag itself is real.
                if not isinstance(draft, str) or "/" not in draft:
                    return Unsupported(
                        setting,
                        f"--draft-model needs a draft model repo (org/name), not "
                        f"{draft!r}; mlx has no method-name speculative flag",
                    )
                return Translated(("--draft-model", draft))
            case Setting.LORA_FLEET:
                n = len(value.adapters) if isinstance(value, LoraFleet) else 0
                if n > 1:
                    return Unsupported(
                        setting,
                        f"--adapter-path loads one adapter; {n} were asked for, and "
                        "mlx_lm.server has no multi-adapter flag",
                    )
                if isinstance(value, LoraFleet) and value.adapters:
                    # adapters are (display name, path) pairs; the server takes
                    # the path and infers nothing from the name.
                    return Translated(("--adapter-path", value.adapters[0][1]))
                return Unsupported(setting, "no adapters given")
        return Unsupported(setting, f"no mapping defined for {setting}")

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        # The console script, not `python -m mlx_lm.server` — that form prints a
        # deprecation warning on 0.31, and a generated artefact should not hand
        # someone a command the tool itself complains about.
        argv, gaps = ["mlx_lm.server", "--model", model], []
        for setting, value in settings.items():
            out = self.translate(setting, value)
            if isinstance(out, Unsupported):
                gaps.append(out)
            else:
                argv.extend(out.argv)
        return argv, gaps


#: The planner speaks vLLM's KV-cache vocabulary; llama.cpp has its own, and the
#: allowed set is read off `llama-server --help`. 8-bit maps to q8_0 because
#: that is the nearest thing llama.cpp actually implements — there is no fp8
#: KV type at all, so an unmapped request is refused rather than approximated
#: into something the binary rejects on start-up.
LLAMACPP_KV_DTYPE: dict[str, str] = {
    "fp8": "q8_0",
    "fp8_e4m3": "q8_0",
    "fp8_e5m2": "q8_0",
    "int8": "q8_0",
    "q8_0": "q8_0",
    "fp16": "f16",
    "f16": "f16",
    "bf16": "bf16",
    "fp32": "f32",
    "f32": "f32",
    "auto": "f16",
}


class LlamaCppAdapter(Adapter):
    """llama.cpp's `llama-server`. Flags verified against the binary in [`SOURCES`].

    The most widely installed way to run open weights, and the one this repo
    recommended for eighteen releases without being able to launch it: the
    planner picks llama.cpp for single-stream work on Apple silicon — correctly,
    it is the most predictable Metal option — and `clickllm run` then refused for
    want of a dialect. `clickllm fit` had to apologise for its own advice.

    Two things make this dialect unlike the CUDA servers.

    **Quantisation is in the file.** llama.cpp serves GGUF, and a GGUF is already
    Q4_K_M or Q8_0 by the time it reaches disk. There is no serve-time
    conversion, so `QUANTIZATION` is a setting that has been spent — the same
    shape as MLX, for the same reason, and the fix is the same: choose the right
    file, do not pass a flag.

    **Offload is the memory knob.** There is no VRAM fraction to reserve. What
    exists is `--n-gpu-layers`: how much of the model lives on the accelerator.
    A memory *fraction* does not translate into a layer *count* without knowing
    the layer sizes, and inventing that mapping would be exactly the kind of
    plausible arithmetic this repo refuses elsewhere.
    """

    name = "llama.cpp"
    help_argv = ("llama-server", "--help")

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        match setting:
            case Setting.CONTEXT_LENGTH:
                # Total across all slots, not per slot: llama.cpp divides
                # --ctx-size by --parallel to size each sequence. The planner
                # means per-sequence, so the multiplication happens in command()
                # where both numbers are in scope. Alone, this is honest.
                return Translated(("--ctx-size", str(value)))
            case Setting.MAX_CONCURRENT:
                # --cont-batching is on by default in current builds; passing it
                # would imply we turned something on. --parallel is the slot count.
                return Translated(("--parallel", str(value)))
            case Setting.PREFILL_CHUNK:
                if not value:
                    return Unsupported(
                        setting, "llama-server always batches prefill; there is no off switch"
                    )
                # The physical batch is the prefill chunk; --batch-size is the
                # logical one and is a different quantity.
                return Translated(("--ubatch-size", str(value)))
            case Setting.PREFIX_REUSE:
                if not value:
                    return Unsupported(setting, "no flag disables the prompt cache")
                # --cache-reuse takes a minimum chunk length rather than a
                # boolean; 256 is llama.cpp's own documented starting point.
                return Translated(
                    ("--cache-prompt", "--cache-reuse", "256"),
                    "reuses a cached prefix in chunks of >=256 tokens",
                )
            case Setting.SPECULATIVE:
                spec = value if isinstance(value, dict) else {}
                draft = spec.get("model") or spec.get("draft_model")
                if not draft:
                    return Unsupported(
                        setting,
                        f"llama.cpp speculation needs a draft GGUF: --model-draft takes a "
                        f"file, not a method name like {spec.get('method', value)!r}",
                    )
                return Translated(("--model-draft", str(draft), "--draft-max", "16"))
            case Setting.STRUCTURED_OUTPUT:
                return Translated(
                    (),
                    "llama-server accepts a JSON schema per request "
                    "(response_format, or --json-schema to fix one for the server)",
                )
            case Setting.KV_CACHE_DTYPE:
                # llama.cpp's own vocabulary, read off `--help`:
                #   f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
                # The planner speaks vLLM's, and passing `fp8_e4m3` straight
                # through is rejected by the binary at start-up — exactly the
                # cross-dialect leak this layer exists to prevent, committed
                # inside the layer itself.
                mapped = LLAMACPP_KV_DTYPE.get(str(value))
                if mapped is None:
                    return Unsupported(
                        setting,
                        f"llama.cpp has no {value!r} KV type; it takes one of "
                        f"{', '.join(sorted(set(LLAMACPP_KV_DTYPE.values())))}",
                    )
                return Translated(("--cache-type-k", mapped, "--cache-type-v", mapped))
            case Setting.QUANTIZATION:
                return Unsupported(
                    setting,
                    f"llama.cpp serves GGUF, which is already quantised on disk: fetch a "
                    f"{value}-flavoured .gguf rather than converting at start-up",
                )
            case Setting.MEMORY_FRACTION:
                return Unsupported(
                    setting,
                    "no VRAM fraction; --n-gpu-layers decides how much of the model is "
                    "offloaded, and a fraction does not become a layer count without "
                    "knowing the layer sizes",
                )
            case Setting.TENSOR_PARALLEL:
                return Unsupported(
                    setting,
                    "no tensor-parallel degree; --split-mode and --tensor-split distribute "
                    "across devices by a different scheme",
                )
            case Setting.LORA_FLEET:
                if isinstance(value, LoraFleet) and value.adapters:
                    if len(value.adapters) > 1:
                        return Unsupported(
                            setting,
                            f"--lora loads adapters into one server but they are not "
                            f"selectable per request; {len(value.adapters)} were asked for",
                        )
                    return Translated(("--lora", value.adapters[0][1]))
                return Unsupported(setting, "no adapters given")
        return Unsupported(setting, f"no mapping defined for {setting}")

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        # `--model` is a LOCAL file; `--hf-repo` fetches from the Hub. Passing a
        # repo id to --model emits a command that fails on start-up, which is
        # what the first cut of this adapter did.
        locator = (
            ["--hf-repo", model]
            if "/" in model and not model.startswith((".", "/"))
            else [
                "--model",
                model,
            ]
        )
        argv: list[str] = ["llama-server", *locator]
        gaps: list[Unsupported] = []

        # --ctx-size is the TOTAL context divided among --parallel slots, so a
        # per-sequence intent has to be multiplied or every slot silently gets a
        # fraction of what was asked for. This is the one place the two settings
        # are both in scope, which is why it happens here rather than in
        # translate() where each is seen alone.
        ctx = settings.get(Setting.CONTEXT_LENGTH)
        slots = settings.get(Setting.MAX_CONCURRENT)
        rest = dict(settings)
        if isinstance(ctx, int) and isinstance(slots, int) and slots > 1:
            rest[Setting.CONTEXT_LENGTH] = ctx * slots

        for setting, value in rest.items():
            out = self.translate(setting, value)
            if isinstance(out, Unsupported):
                gaps.append(out)
            else:
                argv.extend(out.argv)
        return argv, gaps


class OllamaAdapter(Adapter):
    """Ollama. Configured by environment, verified against `ollama serve --help`.

    The most installed way to run open weights on a laptop, and the one engine
    here that cannot be configured by flags at all:

        $ ollama serve --help
        Flags:
          -h, --help   help for serve

        Environment Variables:
              OLLAMA_HOST                IP Address for the ollama server
              OLLAMA_CONTEXT_LENGTH      Context length to use unless otherwise specified
              OLLAMA_NUM_PARALLEL        Maximum number of parallel requests
              OLLAMA_MAX_LOADED_MODELS   Maximum number of loaded models per GPU
              OLLAMA_MAX_QUEUE           Maximum number of queued requests
              OLLAMA_KEEP_ALIVE          The duration that models stay loaded in memory

    That is why `Translated` carries `env`. Mapping these to `Unsupported` would
    have been the easy read of the existing interface and a false one — Ollama
    honours concurrency and context as well as any engine here, through a
    different channel. An adapter layer that can only describe one channel does
    not abstract engines; it abstracts engines that resemble vLLM.

    Ollama is also the only engine here that fetches and names models itself.
    `ollama serve` starts a daemon and the model is pulled on first use, so the
    model argument is not part of the launch command at all — the generated
    command starts the server and the caller asks for a model by name over the
    API, which is the shape Ollama's own docs use.
    """

    name = "ollama"
    help_argv = ("ollama", "serve", "--help")

    def environment(self, settings: dict[Setting, object]) -> tuple[tuple[str, str], ...]:
        """The environment this configuration needs, as (name, value) pairs.

        Separate from `command()` on purpose. The first cut returned
        `["OLLAMA_CONTEXT_LENGTH=8192", "ollama", "serve"]` as argv, which reads
        correctly and is not runnable either way: `subprocess.Popen` would try to
        exec a binary named `OLLAMA_CONTEXT_LENGTH=8192`, and the printed form
        `shlex.quote`s each element so a paste into bash gives `command not
        found: OLLAMA_CONTEXT_LENGTH=8192`. The docstring claimed "native,
        standalone and pasteable" and the code backed none of it.
        """
        out: list[tuple[str, str]] = []
        for setting, value in settings.items():
            got = self.translate(setting, value)
            if isinstance(got, Translated):
                out.extend(got.env)
        return tuple(sorted(set(out)))

    def translate(self, setting: Setting, value: object) -> Translated | Unsupported:
        match setting:
            case Setting.CONTEXT_LENGTH:
                return Translated((), env=(("OLLAMA_CONTEXT_LENGTH", str(value)),))
            case Setting.MAX_CONCURRENT:
                return Translated((), env=(("OLLAMA_NUM_PARALLEL", str(value)),))
            case Setting.PREFIX_REUSE:
                if not value:
                    return Unsupported(setting, "no switch disables Ollama's prompt cache")
                return Translated(
                    (), "ollama reuses a cached prefix by default; there is nothing to enable"
                )
            case Setting.STRUCTURED_OUTPUT:
                return Translated(
                    (),
                    "ollama takes a JSON schema per request in the `format` field; "
                    "there is no server-wide setting",
                )
            case Setting.QUANTIZATION:
                return Unsupported(
                    setting,
                    f"ollama bakes precision into the tag: pull `:{value}` "
                    f"(e.g. llama3.1:8b-instruct-q8_0) rather than converting at start-up",
                )
            case Setting.PREFILL_CHUNK:
                return Unsupported(setting, "no prefill-batch control is exposed")
            case Setting.MEMORY_FRACTION:
                return Unsupported(
                    setting,
                    "no VRAM fraction; OLLAMA_MAX_LOADED_MODELS bounds how many models "
                    "share a GPU, which is a different quantity",
                )
            case Setting.TENSOR_PARALLEL:
                return Unsupported(setting, "no tensor-parallel degree is exposed")
            case Setting.KV_CACHE_DTYPE:
                return Unsupported(
                    setting,
                    "OLLAMA_KV_CACHE_TYPE exists in some builds and is not in this one's "
                    "documented set, so it is not emitted",
                )
            case Setting.SPECULATIVE:
                return Unsupported(setting, "no draft-model or speculative setting is exposed")
            case Setting.LORA_FLEET:
                return Unsupported(
                    setting,
                    "adapters are baked in with a Modelfile ahead of time, not selected "
                    "at serve time",
                )
        return Unsupported(setting, f"no mapping defined for {setting}")

    def command(
        self, model: str, settings: dict[Setting, object]
    ) -> tuple[list[str], list[Unsupported]]:
        """`ollama serve`, with the environment it needs.

        The model is deliberately absent: `serve` starts a daemon and a model is
        requested by name over the API, so putting it on the command line would
        emit something that does not run.
        """
        argv: list[str] = ["ollama", "serve"]
        gaps: list[Unsupported] = []
        for setting, value in settings.items():
            out = self.translate(setting, value)
            if isinstance(out, Unsupported):
                gaps.append(out)
            else:
                argv.extend(out.argv)
        return argv, gaps


_ADAPTERS: dict[str, Adapter] = {
    "vllm": VllmAdapter(),
    "mlx": MlxAdapter(),
    "llama.cpp": LlamaCppAdapter(),
    "ollama": OllamaAdapter(),
    # vLLM's TPU backend is a different engine (JAX lowering via tpu-inference)
    # wearing the same CLI, so it shares the dialect. Registered explicitly
    # rather than by prefix match: if the CLIs ever diverge this is the line
    # that has to change, and it should be obvious.
    "vllm-tpu": VllmAdapter(),
    "sglang": SglangAdapter(),
}


def installed_flags(adapter: Adapter, *, timeout: float = 120.0) -> frozenset[str] | None:
    """Every long flag the *installed* engine declares, or None if absent.

    Reads `--help` off the real binary rather than trusting anything written
    here. That direction matters: this module's job is to emit flags an engine
    accepts, so the engine has to be the authority on which those are.

    None means "could not ask" — the engine is not installed, or its help did
    not run. That is deliberately distinct from an empty set, which would mean
    "asked, and it accepts nothing", and would fail every command.
    """
    return _installed_flags_detail(adapter, timeout=timeout)[0]


def _installed_flags_detail(
    adapter: Adapter, *, timeout: float = 120.0
) -> tuple[frozenset[str] | None, str]:
    """As [`installed_flags`], plus WHY when the answer is None.

    The reason was being discarded, and three separate causes then looked
    identical from the outside: a missing binary, a console script that is not
    on PATH, and an engine that imports but whose CLI needs hardware. Diagnosing
    the third by guessing cost two CI rounds. The caller can now print it.
    """
    if not adapter.help_argv:
        return None, "no help_argv is declared for this adapter"
    try:
        r = subprocess.run(  # noqa: S603 - argv from this module, never user input
            adapter.help_argv,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return None, f"{adapter.help_argv[0]!r} is not on PATH"
    except subprocess.TimeoutExpired:
        return None, f"`{' '.join(adapter.help_argv)}` timed out after {timeout:.0f}s"
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {e}"
    # argparse writes usage to stdout for --help; take stderr too so an engine
    # that routes it differently is not read as declaring nothing.
    flags = frozenset(re.findall(r"--[a-z0-9][a-z0-9-]*", f"{r.stdout}\n{r.stderr}"))
    if flags:
        return flags, ""
    tail = (r.stderr or r.stdout or "").strip().splitlines()
    return None, (
        f"exit {r.returncode}, and no flags in its output"
        + (f" — last line: {tail[-1][:200]!r}" if tail else " (it printed nothing)")
    )


def unknown_flags(adapter: Adapter, argv: Sequence[str]) -> list[str] | None:
    """Flags in `argv` the installed engine does not accept. None if unaskable.

    An empty list is the good case. A non-empty one means the generated command
    cannot start, which is a defect this repo has actually shipped: a
    `--speculative-config eagle3` that every test asserted was present and no
    test ever tried to run.
    """
    accepted = installed_flags(adapter)
    if accepted is None:
        return None
    return sorted({a for a in argv if a.startswith("--")} - accepted)


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

    # A method name is not a draft model. `--draft-model eagle3` is a real flag
    # with an unresolvable value: it starts, fails to fetch, and exits — which
    # no flag-name check can catch.
    mlx = MlxAdapter()
    assert isinstance(mlx.translate(Setting.SPECULATIVE, "eagle3"), Unsupported)
    assert mlx.translate(Setting.SPECULATIVE, "org/draft-1b").argv == (
        "--draft-model",
        "org/draft-1b",
    )

    # No adapter is better than the wrong adapter.
    assert adapter_for("vllm") is not None
    assert adapter_for("llama.cpp") is not None

    print("engines: ok")


if __name__ == "__main__":
    demo()
