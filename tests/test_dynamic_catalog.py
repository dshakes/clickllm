"""A catalogue that grows without a release.

Models arrive constantly. If adding one needs a new version of this tool, the
catalogue is stale the week it ships. These pin the drop-in mechanism: what
overrides what, what is refused, and that a dropped-in model reaches the solver
rather than merely appearing in a list.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from clickllm import catalog

SRC = str(Path(__file__).resolve().parents[1] / "src")

SPEC = {
    "id": "acme-70b",
    "name": "Acme 70B",
    "params_b": 70.0,
    "active_b": 70.0,
    "layers": 80,
    "kv_heads": 8,
    "head_dim": 128,
    "kv_scheme": "gqa",
    "max_context": 131072,
    "license": "Apache-2.0",
    "license_ok": True,
    "quants": ["q4", "q8"],
    "verified": False,
}


def write(d: Path, name: str, models: list[dict]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(json.dumps({"models": models}))
    return p


def test_a_dropped_in_model_joins_the_catalogue(tmp_path, monkeypatch):
    write(tmp_path / "models.d", "acme.json", [SPEC])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(tmp_path / "models.d"))

    assert catalog.get("acme-70b").params_b == 70.0
    # And it is sized by the real solver, not just listed.
    assert catalog.get("acme-70b").kv_bytes_per_token() == 80 * 2 * 8 * 128 * 2


def test_a_drop_in_can_correct_a_built_in_without_editing_the_package():
    """Later files win. This is what makes a wrong built-in fixable in the field
    rather than something you wait a release for."""
    assert catalog.get("qwen3-32b").license == "Apache-2.0"


def test_later_files_override_earlier_ones_by_id(tmp_path, monkeypatch):
    d = tmp_path / "models.d"
    write(d, "01-first.json", [SPEC | {"name": "First"}])
    write(d, "02-second.json", [SPEC | {"name": "Second"}])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(d))
    assert catalog.get("acme-70b").name == "Second"


def test_a_single_file_works_as_well_as_a_directory(tmp_path, monkeypatch):
    p = write(tmp_path, "one.json", [SPEC])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(p))
    assert catalog.get("acme-70b").name == "Acme 70B"


def test_several_paths_are_read_in_order(tmp_path, monkeypatch):
    a = write(tmp_path / "a", "m.json", [SPEC | {"name": "A"}])
    b = write(tmp_path / "b", "m.json", [SPEC | {"name": "B"}])
    monkeypatch.setenv("CLICKLLM_CATALOG", os.pathsep.join([str(a), str(b)]))
    assert catalog.get("acme-70b").name == "B"


# --- refusals: a bad drop-in must be loud ------------------------------------


def test_malformed_json_names_the_file(tmp_path, monkeypatch):
    """Skipping it silently would let a typo remove a model, and the first
    symptom would be a sizing answer that omits it with no explanation."""
    d = tmp_path / "models.d"
    d.mkdir()
    (d / "broken.json").write_text("{not json")
    monkeypatch.setenv("CLICKLLM_CATALOG", str(d))
    with pytest.raises(ValueError, match="broken.json"):
        catalog.load()


def test_a_model_without_an_id_is_refused(tmp_path, monkeypatch):
    d = write(tmp_path / "models.d", "x.json", [{"name": "no id"}])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(d.parent))
    with pytest.raises(ValueError, match="needs an 'id'"):
        catalog.load()


def test_an_incomplete_spec_names_the_file_and_the_model(tmp_path, monkeypatch):
    """ "missing kv_heads" is unactionable when six drop-ins are in play."""
    partial = {"id": "half-baked", "name": "Half", "params_b": 7.0}
    write(tmp_path / "models.d", "partial.json", [partial])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(tmp_path / "models.d"))
    with pytest.raises(ValueError) as e:
        catalog.load()
    assert "partial.json" in str(e.value) and "half-baked" in str(e.value)


def test_a_missing_configured_path_is_an_error_not_a_shrug(tmp_path, monkeypatch):
    monkeypatch.setenv("CLICKLLM_CATALOG", str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="nope.json"):
        catalog.load()


def test_an_unset_variable_leaves_the_built_ins_alone(monkeypatch):
    monkeypatch.delenv("CLICKLLM_CATALOG", raising=False)
    assert catalog.sources() and catalog.sources()[0] == catalog.CATALOG_PATH
    assert len(catalog.load()) > 5


# --- MLA invariant survives the dynamic path ---------------------------------


def test_a_dropped_in_mla_model_still_needs_kv_lora_rank(tmp_path, monkeypatch):
    """The repo's load-bearing invariant: MLA without kv_lora_rank overestimates
    KV by ~50x. A drop-in must not be a way around it."""
    mla = SPEC | {"id": "acme-mla", "kv_scheme": "mla", "kv_lora_rank": 512}
    write(tmp_path / "models.d", "mla.json", [mla])
    monkeypatch.setenv("CLICKLLM_CATALOG", str(tmp_path / "models.d"))

    m = catalog.get("acme-mla")
    assert m.kv_lora_rank == 512
    # MLA stores one compressed latent, so per-token KV is far below the GQA form
    # the same geometry would produce.
    assert m.kv_bytes_per_token() < 80 * 2 * 8 * 128 * 2


# --- the CLI surface ---------------------------------------------------------


def run(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", **env_extra}
    return subprocess.run(
        [sys.executable, "-m", "clickllm.cli", *args], capture_output=True, text=True, env=env
    )


def test_catalog_sources_reports_where_models_came_from(tmp_path):
    d = tmp_path / "models.d"
    write(d, "acme.json", [SPEC])
    r = run(["catalog-sources"], {"CLICKLLM_CATALOG": str(d)})
    assert r.returncode == 0, r.stderr
    assert "acme.json" in r.stdout
    assert "no reinstall" in r.stdout


def test_a_dropped_in_model_reaches_fit_through_the_cli(tmp_path):
    d = tmp_path / "models.d"
    write(d, "acme.json", [SPEC])
    r = run(["fit", "--explain", "acme-70b"], {"CLICKLLM_CATALOG": str(d)})
    assert r.returncode in (0, 1), r.stderr
    assert "weights" in r.stdout and "kv cache" in r.stdout
    assert "Traceback" not in r.stderr


def test_a_broken_drop_in_is_a_clean_error_not_a_traceback(tmp_path):
    d = tmp_path / "models.d"
    d.mkdir()
    (d / "broken.json").write_text("{not json")
    r = run(["models"], {"CLICKLLM_CATALOG": str(d)})
    assert r.returncode != 0
    assert "Traceback" not in r.stderr, r.stderr
    assert "broken.json" in (r.stderr + r.stdout)


def test_catalog_add_requires_network_to_be_opted_into(tmp_path):
    r = run(["catalog-add", "Qwen/Qwen3-14B", "--params-b", "14.8"], {})
    assert r.returncode == 0
    assert "opt-in" in r.stdout


_MLA_BASE = dict(
    num_hidden_layers=61,
    num_attention_heads=128,
    num_key_value_heads=128,
    hidden_size=7168,
    head_dim=56,
    max_position_embeddings=163840,
    vocab_size=129280,
)


def test_an_mla_config_whose_rank_will_not_parse_is_refused_not_reclassified():
    """`kv_lora_rank` was the SOLE detector of MLA, and had no aliases.

        kv_lora_rank = _first_int(cfg, "kv_lora_rank")
        scheme = "mla" if kv_lora_rank else ("gqa" if ... else "mha")

    So a rank that failed to parse did not merely go missing — the model stopped
    being MLA and was sized with the GQA formula, which overestimates KV by
    ~50x. CLAUDE.md makes the rank a load-bearing invariant and a test enforces
    it over entries that ARE `mla`; an entry that never becomes `mla` walks
    around that test entirely.

    Refusing is the right failure: a refusal is a catalogue entry someone fixes,
    a silent reclassification is a sizing answer wrong by a factor of fifty that
    looks completely ordinary.
    """
    from clickllm.catalog_update import ConfigError, parse_config

    cfg = {**_MLA_BASE, "architectures": ["DeepseekV3ForCausalLM"], "q_lora_rank": 1536}
    with pytest.raises(ConfigError, match="kv_lora_rank"):
        parse_config(cfg)


def test_the_mla_rank_is_read_through_aliases_like_every_other_field():
    """Every other geometry field takes 2-3 aliases to absorb naming drift; this
    one took exactly one key. The MLA architecture has already been forked and
    renamed by several open releases."""
    from clickllm.catalog_update import parse_config

    cfg = {
        **_MLA_BASE,
        "architectures": ["DeepseekV3ForCausalLM"],
        "kv_lora_dim": 512,
        "q_lora_rank": 1536,
    }
    arch = parse_config(cfg)
    assert arch.kv_scheme == "mla" and arch.kv_lora_rank == 512


def test_an_ordinary_gqa_config_is_untouched_by_the_mla_guard():
    """The control: refusing MLA-without-a-rank must not refuse everything else."""
    from clickllm.catalog_update import parse_config

    cfg = {**_MLA_BASE, "architectures": ["LlamaForCausalLM"], "num_key_value_heads": 8}
    arch = parse_config(cfg)
    assert arch.kv_scheme == "gqa" and arch.kv_lora_rank is None
