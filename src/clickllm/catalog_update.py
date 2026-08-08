"""Keeping the catalogue current — by proposal, never by silent mutation.

Two jobs:

**Verification.** Most catalogue entries start with parameter counts from public
reporting and *estimated* geometry, which is why they carry ``verified: false``
and why their KV figures come with error bars. A model's own ``config.json``
carries the real numbers. Parsing it turns an estimate into a fact.

**Currency.** Capability rankings move weekly. A catalogue frozen at the day it
was written is wrong within a month.

Both are done as a **proposal**: a reviewable diff the user applies, not a
background process that rewrites your model database while you sleep. An agent
may analyse and recommend; a human presses the button. The same boundary the MCP
server enforces applies here — silently changing the numbers a deployment
decision was made from is exactly the kind of "help" nobody asked for.

Network access is **opt-in and injected**. Parsing is pure, so the interesting
logic is testable offline and the whole module works air-gapped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .atomicio import atomic_write_json
from .catalog import ModelSpec

#: Where a model's real architecture lives.
CONFIG_URL = "https://huggingface.co/{repo}/resolve/main/config.json"

#: A fetcher takes a URL and returns the body, or raises. Injected so the module
#: never reaches the network unless a caller hands it something that can.
Fetcher = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class Architecture:
    """Ground truth parsed from a model's own config."""

    layers: int
    kv_heads: int
    head_dim: int
    kv_scheme: str
    max_context: int
    kv_lora_rank: int | None = None
    experts: int | None = None
    experts_per_token: int | None = None
    architecture: str = ""

    @property
    def is_moe(self) -> bool:
        return bool(self.experts and self.experts > 1)


class Unreachable(RuntimeError):
    """The index could not be read. Distinct from "it had nothing new for us"."""


class ConfigError(ValueError):
    """A config could not be parsed into an architecture we can size."""


def parse_config(cfg: dict[str, Any]) -> Architecture:
    """Derive sizing geometry from a Hugging Face ``config.json``.

    Raises:
        ConfigError: when a field we cannot guess is missing. Guessing here would
            produce a confident, wrong memory figure — the failure mode this whole
            module exists to remove.
    """
    # `json.loads` succeeds on `null`, `[]`, `3` and `"text"` as readily as on an
    # object, and every field read below starts with `cfg.get(...)`. Without this
    # the first one raises `AttributeError`, which is not a `ConfigError` — and
    # `propose()` catches only `ConfigError`, so a single such response escaped
    # the list comprehension in `cmd_catalog`, skipped every model after it, and
    # reached the user as a traceback. The convention is a sentence and exit 2.
    #
    # Here rather than at any one caller: `cli.py` and `watch.py` also call this
    # with a freshly-parsed body, and all three already handle `ConfigError`
    # specifically, so this reaches them as the sentence they know how to print.
    if not isinstance(cfg, dict):
        raise ConfigError(
            f"config is not a JSON object, got {type(cfg).__name__} — "
            "the URL returned something other than a model config"
        )

    layers = _first_int(cfg, "num_hidden_layers", "n_layer", "num_layers")
    if layers is None:
        raise ConfigError("no layer count in config (num_hidden_layers)")

    attn_heads = _first_int(cfg, "num_attention_heads", "n_head")
    # Absent num_key_value_heads means MHA — every attention head has its own KV.
    kv_heads = _first_int(cfg, "num_key_value_heads", "num_kv_heads") or attn_heads
    if kv_heads is None:
        raise ConfigError("no attention or KV head count in config")

    hidden = _first_int(cfg, "hidden_size", "n_embd", "d_model")
    head_dim = _first_int(cfg, "head_dim", "attention_head_dim")
    if head_dim is None:
        if hidden is None or not attn_heads:
            raise ConfigError("cannot derive head_dim: need head_dim, or hidden_size and heads")
        head_dim = hidden // attn_heads

    max_ctx = _first_int(cfg, "max_position_embeddings", "n_positions", "seq_length")
    if max_ctx is None:
        raise ConfigError("no context length in config (max_position_embeddings)")

    # MLA (DeepSeek family) is identified by its compressed KV rank. It must be
    # detected explicitly: sizing an MLA model with the GQA formula overestimates
    # KV by ~50x, which is the difference between "fits" and "buy another node".
    # Aliases, like every other field here — and then a guard, because this one
    # is not just a field. The next line makes the rank the SOLE detector of
    # MLA, so a rank that fails to parse does not merely go missing: the model
    # stops being MLA and is sized with the GQA formula, which overestimates KV
    # by ~50x. CLAUDE.md's invariant ("any entry with kv_scheme: mla must set
    # kv_lora_rank") is enforced by a test over entries that ARE mla — and an
    # entry that never becomes mla walks around it.
    #
    # So: detect the family independently of whether the rank parsed, and refuse
    # rather than silently reclassify. A refusal is a catalogue entry someone
    # fixes; a silent reclassification is a sizing answer that is wrong by a
    # factor of fifty and looks ordinary.
    kv_lora_rank = _first_int(cfg, "kv_lora_rank", "kv_lora_dim", "kv_rank")
    mla_signals = [k for k in ("q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim") if k in cfg]
    # `or []` is not a type check: `7 or []` is `7`, and iterating it raises
    # `TypeError`, which `propose` does not catch. The read three lines below
    # already tested `isinstance(..., list)` — the same field, guarded on one of
    # its two reads. One read now, used by both.
    families = cfg.get("architectures")
    families = families if isinstance(families, list) else []
    # `model_type` as well as `architectures`, because the family name has two
    # homes and this guard read one. Sanitising a non-list `architectures` to
    # `[]` turned `{"architectures": 7, "model_type": "deepseek_v3"}` from a
    # `TypeError` into a silent pass, sized as MHA — and a config carrying only
    # `model_type` was never caught at all, before or after. The whole point of
    # this refusal is that missing it overestimates KV by ~50x.
    arch = " ".join([*(str(a) for a in families), str(cfg.get("model_type", ""))]).lower()
    if not kv_lora_rank and (mla_signals or "deepseek" in arch):
        raise ConfigError(
            f"this config looks like an MLA model "
            f"({', '.join(mla_signals) or arch}) but no kv_lora_rank could be "
            f"read from it. Sizing it as GQA would overestimate KV by ~50x, so "
            f"it is refused rather than guessed — add the rank by hand."
        )
    scheme = "mla" if kv_lora_rank else ("gqa" if kv_heads < (attn_heads or kv_heads) else "mha")

    experts = _first_int(cfg, "num_experts", "num_local_experts", "n_routed_experts")
    per_tok = _first_int(cfg, "num_experts_per_tok", "moe_topk", "num_experts_per_token")

    arch = str(families[0]) if families else ""

    return Architecture(
        layers=layers,
        kv_heads=kv_heads,
        head_dim=head_dim,
        kv_scheme=scheme,
        max_context=max_ctx,
        kv_lora_rank=kv_lora_rank,
        experts=experts,
        experts_per_token=per_tok,
        architecture=arch or str(cfg.get("model_type", "")),
    )


def _number(v: Any, default: float) -> float:
    """A count from JSON we do not control, or the default.

    `bool` is excluded deliberately — it is an `int` in Python, and `true`
    arriving as a download count of 1 is a fabricated number, not a parsed one.

    Non-finite is excluded for two reasons, and the first version of this
    checked only the type. `json.loads` accepts the literals `NaN`, `Infinity`
    and `-Infinity`, and `isinstance(float("nan"), float)` is `True`, so both
    walked straight through: `int(nan)` raises `ValueError` and `int(inf)`
    `OverflowError`, neither of which is `Unreachable`, so discovery aborted on
    one row after all. And a `NaN` that did *not* crash was worse — `trending`
    reaches `out.sort(key=lambda d: (-d.trending, ...))`, where NaN compares
    false against everything and quietly makes the order the comment two lines
    below calls "deterministic" depend on input order.
    """
    if isinstance(v, bool) or not isinstance(v, int | float):
        return default
    # Not `math.isfinite(v)` on its own: a JSON document can carry an `int` too
    # large to convert to a float, and `isfinite` raises `OverflowError` on it —
    # not `Unreachable` either, so the abort path this guard closed stayed open
    # one type down. Nor is `isinstance(v, int)` a licence to pass it through:
    # `trending=float(_number(...))` overflows on the very next line.
    #
    # The question is "usable as a float", which is what every caller does with
    # it, so the conversion is the test.
    try:
        return v if math.isfinite(float(v)) else default
    except OverflowError:
        return default


def _first_int(cfg: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, int) and v > 0:
            return v
        if isinstance(v, float) and v > 0 and v.is_integer():
            return int(v)
    return None


#: The fields `propose` compares against a published config, and the same list
#: `FieldChange.significant` reads. It was two hand-written lists, and the field
#: that appeared in only one of them was `max_context`: a change to it rendered
#: without the `!` marker and was excluded from `UpdateReport.significant`, so
#: `clickllm catalog update --apply` never printed "re-run fit before
#: deploying" — while `fit.max_context()` caps every context figure it prints
#: at `min(model.max_context, ...)`. A vendor advertising 128k against a config
#: publishing 32k is an ordinary discrepancy, and it read as cosmetic.
#:
#: Every field here moves a solver answer, which is why they share one list. A
#: label the comparison starts carrying — a licence, a display name — would be
#: the first entry that does not, and would need `significant` to become a
#: lookup rather than a membership test.
SOLVER_FIELDS: tuple[str, ...] = (
    "layers",
    "kv_heads",
    "head_dim",
    "kv_scheme",
    "max_context",
    "kv_lora_rank",
)


@dataclass(frozen=True, slots=True)
class FieldChange:
    """One catalogue field that the model's own config contradicts."""

    field: str
    current: Any
    proposed: Any

    @property
    def significant(self) -> bool:
        """Whether this moves a number the solver computes with."""
        return self.field in SOLVER_FIELDS

    def render(self) -> str:
        mark = "!" if self.significant else " "
        return f"  {mark} {self.field:<16} {self.current!r} → {self.proposed!r}"


@dataclass(slots=True)
class Proposal:
    """A reviewable set of catalogue changes. Nothing is written until applied."""

    model_id: str
    changes: list[FieldChange] = field(default_factory=list)
    now_verified: bool = False
    error: str | None = None

    @property
    def has_changes(self) -> bool:
        return bool(self.changes) or self.now_verified

    @property
    def significant(self) -> bool:
        """True when a proposed change would move a memory calculation."""
        return any(c.significant for c in self.changes)

    def render(self) -> str:
        if self.error:
            return f"{self.model_id}: could not verify — {self.error}"
        if not self.has_changes:
            return f"{self.model_id}: confirmed, no changes"
        head = f"{self.model_id}:"
        if self.now_verified and not self.changes:
            return f"{head} architecture confirmed — now verified"
        lines = [head, *(c.render() for c in self.changes)]
        if self.significant:
            lines.append("    ! marks fields the solver computes with — re-run fit after applying")
        if self.now_verified:
            lines.append("    architecture confirmed — will be marked verified")
        return "\n".join(lines)


def propose(spec: ModelSpec, repo: str, fetch: Fetcher) -> Proposal:
    """Compare a catalogue entry against the model's published config.

    ``fetch`` is injected. Passing a fetcher is the act of consenting to network
    access; without one this module cannot reach anything.
    """
    p = Proposal(model_id=spec.id)
    try:
        cfg = json.loads(fetch(CONFIG_URL.format(repo=repo)))
    except Exception as e:  # noqa: BLE001 — a fetch failure is data, not a crash
        p.error = str(e)
        return p
    try:
        arch = parse_config(cfg)
    except ConfigError as e:
        p.error = str(e)
        return p

    for name in SOLVER_FIELDS:
        current, proposed = getattr(spec, name), getattr(arch, name)
        if current != proposed:
            p.changes.append(FieldChange(name, current, proposed))

    # A config we could parse is a config we can trust.
    p.now_verified = not spec.verified
    return p


@dataclass(slots=True)
class UpdateReport:
    """The outcome of checking a set of models."""

    proposals: list[Proposal] = field(default_factory=list)

    @property
    def changed(self) -> list[Proposal]:
        return [p for p in self.proposals if p.has_changes and not p.error]

    @property
    def significant(self) -> list[Proposal]:
        return [p for p in self.changed if p.significant]

    @property
    def failed(self) -> list[Proposal]:
        return [p for p in self.proposals if p.error]

    def render(self) -> str:
        if not self.proposals:
            return "Nothing to check."
        out = [p.render() for p in self.proposals if p.has_changes or p.error]
        if not out:
            return f"All {len(self.proposals)} entries confirmed against their published configs."
        summary = (
            f"{len(self.changed)} of {len(self.proposals)} entries would change"
            f"{f', {len(self.significant)} affecting sizing' if self.significant else ''}"
            f"{f', {len(self.failed)} could not be checked' if self.failed else ''}."
        )
        out.append("")
        out.append(summary)
        out.append("Nothing has been written. Apply with: clickllm catalog update --apply")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
# Network — opt-in, stdlib only
# --------------------------------------------------------------------------- #

#: Hugging Face model index, used to discover models we do not yet carry.
INDEX_URL = (
    "https://huggingface.co/api/models"
    "?filter=text-generation&sort=trendingScore&direction=-1&limit={limit}"
)

USER_AGENT = "clickllm-catalog/0.1 (+https://github.com/dshakes/clickllm)"


def http_fetch(url: str, timeout: float = 15.0) -> str:
    """Fetch a URL with the standard library.

    Exists as a separate, explicitly-passed function so that importing this
    module never opens a socket. Air-gapped installs simply never call it.
    """
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https only
        return resp.read().decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class Discovery:
    """A model in the wild that our catalogue does not carry."""

    repo: str
    downloads: int
    likes: int
    trending: float
    #: Licence as the index reported it, verbatim, or "" when it reported none.
    #: `watch` read this off `Discovery` with `getattr(d, "license", "")` and the
    #: field did not exist, so every staged discovery was written "unknown" —
    #: against a docstring promising "the licence string as the index reported
    #: it". Never guessed: an absent licence stays absent, which is the
    #: distinction the module's own invariant turns on.
    license: str = ""

    def render(self) -> str:
        return f"  {self.repo:<44} {self.downloads:>10,} downloads  {self.likes:>6,} likes"


def discover(
    known_repos: set[str],
    fetch: Fetcher,
    limit: int = 40,
) -> list[Discovery]:
    """Trending text-generation models we do not already carry.

    Discovery is a *shortlist*, never a decision. A model trending on a public
    index says nothing about whether it fits your hardware or your workload —
    that is what `fit` and `prove` are for. This only answers "is there something
    here we have not looked at".
    """
    # Raises `Unreachable` rather than returning []. An empty list meant three
    # different things — the fetch failed, the index was malformed, or we
    # already know everything it listed — and `watch.run()` reported ALL of them
    # as `offline=True`. So a corrupt index read as "nothing new upstream",
    # which is the failure this whole module exists to notice.
    try:
        rows = json.loads(fetch(INDEX_URL.format(limit=limit)))
    except Exception as e:  # noqa: BLE001 — no network is a normal state
        raise Unreachable(f"the index could not be fetched: {e}") from e
    if not isinstance(rows, list):
        raise Unreachable(
            f"the index returned {type(rows).__name__}, not a list of models — "
            f"malformed, which is not the same as nothing new"
        )

    seen = {r.casefold() for r in known_repos}
    out: list[Discovery] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        repo = str(r.get("modelId") or r.get("id") or "")
        if not repo or repo.casefold() in seen:
            continue
        # Every field below was coerced without checking what it was, and the
        # index is a third party's JSON. `int("many")` raises `ValueError`,
        # `int([1])` and `float("hot")` raise their own, and `(cardData or
        # {}).get(...)` raises `AttributeError` for a string — none of which is
        # `Unreachable`, so one odd row aborted the whole discovery run with a
        # traceback. The row-level `isinstance(r, dict)` guard above shows the
        # shape was already understood; it just stopped at the row.
        #
        # A bad field reads as absent rather than dropping the row: the repo is
        # the part that matters, and a missing download count is a worse reason
        # to never see a model than it is a number.
        card = r.get("cardData")
        out.append(
            Discovery(
                repo=repo,
                downloads=int(_number(r.get("downloads"), 0)),
                likes=int(_number(r.get("likes"), 0)),
                trending=float(_number(r.get("trendingScore"), 0.0)),
                license=str(
                    r.get("license")
                    or (card.get("license") if isinstance(card, dict) else None)
                    or ""
                ),
            )
        )
    # Deterministic: trending, then downloads, then name.
    out.sort(key=lambda d: (-d.trending, -d.downloads, d.repo))
    return out


def apply_proposals(proposals: list[Proposal], path: Path | None = None) -> int:
    """Write accepted changes to the catalogue file. Returns entries updated.

    Atomic: the new catalogue is written beside the old one and renamed, so an
    interrupted update cannot leave a half-written model database behind.
    """
    from .catalog import CATALOG_PATH

    target = path or CATALOG_PATH
    raw = json.loads(target.read_text())
    by_id = {m["id"]: m for m in raw["models"]}

    updated = 0
    for p in proposals:
        if p.error or not p.has_changes:
            continue
        entry = by_id.get(p.model_id)
        if entry is None:
            continue
        for c in p.changes:
            entry[c.field] = c.proposed
        if p.now_verified:
            entry["verified"] = True
        updated += 1

    if updated:
        atomic_write_json(target, raw)
    return updated


def demo() -> None:
    # A real GQA config, in the shape Hugging Face publishes.
    qwen = {
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 64,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "hidden_size": 5120,
        "max_position_embeddings": 131072,
    }
    a = parse_config(qwen)
    assert (a.layers, a.kv_heads, a.kv_scheme) == (64, 8, "gqa")
    assert a.head_dim == 5120 // 64, "head_dim derived from hidden_size when absent"

    # MHA: no num_key_value_heads means every attention head has its own KV.
    mha = parse_config({**qwen, "num_key_value_heads": None})
    assert mha.kv_heads == 64 and mha.kv_scheme == "mha"

    # MLA must be detected from its rank, not guessed — the GQA formula would
    # overestimate KV by ~50x.
    deepseek = parse_config({**qwen, "kv_lora_rank": 512, "num_hidden_layers": 61})
    assert deepseek.kv_scheme == "mla" and deepseek.kv_lora_rank == 512

    # MoE is recognised from either spelling.
    moe = parse_config({**qwen, "num_local_experts": 128, "num_experts_per_tok": 8})
    assert moe.is_moe and moe.experts_per_token == 8

    # An explicit head_dim wins over derivation.
    assert parse_config({**qwen, "head_dim": 128}).head_dim == 128

    # Missing fields raise rather than guess.
    for broken in [
        {},
        {"num_hidden_layers": 4},
        {"num_hidden_layers": 4, "num_attention_heads": 8},
    ]:
        try:
            parse_config(broken)
        except ConfigError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{broken} should not have parsed")

    # A proposal against a deliberately wrong entry.
    from .catalog import get

    spec = get("qwen3-32b")
    wrong = {**qwen, "num_key_value_heads": 4}
    p = propose(spec, "Qwen/Qwen3-32B", lambda _: json.dumps(wrong))
    assert p.has_changes and p.significant
    assert any(c.field == "kv_heads" and c.significant for c in p.changes)
    assert "re-run fit" in p.render()

    # A matching entry proposes nothing.
    match = {
        **qwen,
        "head_dim": spec.head_dim,
        "num_hidden_layers": spec.layers,
        "num_key_value_heads": spec.kv_heads,
        "max_position_embeddings": spec.max_context,
    }
    clean = propose(spec, "Qwen/Qwen3-32B", lambda _: json.dumps(match))
    assert not clean.changes

    # A fetch failure is reported, never raised into the caller's loop.
    def dead(_: str) -> str:
        raise OSError("network unreachable")

    failed = propose(spec, "x/y", dead)
    assert failed.error and "unreachable" in failed.error
    assert not failed.has_changes

    # An unverified entry that parses becomes verifiable.
    unverified = get("glm-5.2")
    assert not unverified.verified
    up = propose(unverified, "z/glm", lambda _: json.dumps(qwen))
    assert up.now_verified

    rep = UpdateReport([p, clean, failed])
    text = rep.render()
    assert "Nothing has been written" in text
    assert len(rep.changed) == 1 and len(rep.failed) == 1

    # Discovery filters what we already carry and is deterministic.
    index = json.dumps(
        [
            {"modelId": "org/known", "downloads": 10, "likes": 1, "trendingScore": 9},
            {"modelId": "org/fresh", "downloads": 500, "likes": 40, "trendingScore": 7},
            {"modelId": "org/also-new", "downloads": 900, "likes": 5, "trendingScore": 7},
            "not-a-dict",
        ]
    )
    found = discover({"org/known"}, lambda _: index)
    assert [d.repo for d in found] == ["org/also-new", "org/fresh"], [d.repo for d in found]
    # No network is still a normal state — it is now a NAMED one. Returning []
    # made it indistinguishable from a malformed index and from "nothing new",
    # and `watch` reported all three as offline.
    try:
        discover(set(), lambda _: (_ for _ in ()).throw(OSError("offline")))
        raise AssertionError("a failed fetch must be distinguishable from an empty result")
    except Unreachable:
        pass
    try:
        discover(set(), lambda _: '{"error": "nope"}')
        raise AssertionError("a malformed index must not read as 'nothing new'")
    except Unreachable:
        pass
    assert discover({"org/known", "org/also-new", "org/fresh"}, lambda _: index) == [], (
        "reached the index and it had nothing new: empty, and not a failure"
    )

    # Applying writes atomically to a copy, never the real catalogue in a test.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "models.json"
        from .catalog import CATALOG_PATH

        tmp.write_text(CATALOG_PATH.read_text())
        n = apply_proposals([p], path=tmp)
        assert n == 1
        after = {m["id"]: m for m in json.loads(tmp.read_text())["models"]}
        assert after["qwen3-32b"]["kv_heads"] == 4, "the change must land"
        assert apply_proposals([failed], path=tmp) == 0, "failed proposals write nothing"

    print(rep.render())
    print("ok")


if __name__ == "__main__":
    demo()
