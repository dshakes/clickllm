"""The hosting fallback.

`host.demo()` walks a whole survey. These pin the four properties that make the
answer trustworthy rather than merely present:

1. **Every price is sourced and dated.** A figure without provenance is how
   someone budgets against a number nobody can check (invariant 6).
2. **Cheapest-first is really cheapest-first**, and an unpriced shape sorts
   last rather than sorting as zero.
3. **A free tier that cannot fit is excluded with the reason**, not silently
   dropped. "96 GB, and you need 412" is the answer people came for.
4. **Nothing generated depends on clickllm, and nothing here touches a
   credential.** The artifact deploys with this tool uninstalled (NFR-4), and
   no code path reads a token from anywhere.

Every test is offline. Planning must not contact a provider (NFR-2), and this
suite is what proves it: the network is stubbed out entirely for the survey.
"""

from __future__ import annotations

import ast
import json
import re
from datetime import date
from pathlib import Path

import pytest

from clickllm import catalog, host

GB = 1024**3

#: A 30B MoE: fits a free 96 GB card and most rented ones. The interesting case
#: because the answer is not "everything" or "nothing".
QWEN = "qwen3-30b-a3b"
#: 671B. Fits almost nothing, which is what makes the exclusions readable.
DEEPSEEK = "deepseek-v3"
#: 2.8T. Fits nothing at all.
KIMI = "kimi-k3"


def survey(model_id: str = QWEN, **kw):
    return host.survey(catalog.get(model_id), **kw)


# --------------------------------------------------------------------------- #
# Provenance — the invariant that makes a price usable
# --------------------------------------------------------------------------- #


def test_every_registry_entry_is_sourced_and_dated():
    assert host.REGISTRY, "an empty registry answers nothing"
    for p in host.REGISTRY:
        assert p.source.startswith("https://"), f"{p.id}: price source must be a URL"
        # Parses, and is not in the future — a "checked" date nobody could have
        # checked is worse than none.
        assert date.fromisoformat(p.checked) <= date.today(), f"{p.id}: checked in the future"
        assert p.cold_start, f"{p.id}: cold-start behaviour is part of the cost"


def test_provider_ids_and_offers_are_coherent():
    assert len({p.id for p in host.REGISTRY}) == len(host.REGISTRY), "ids must be unique"
    for p in host.REGISTRY:
        for o in p.offers:
            # Raises if the hardware behind an offer is not described anywhere,
            # which would mean pricing a shape whose memory we cannot state.
            profile = host._profile(o.profile_id)
            assert profile.total_memory_gb > 0
            assert o.label, f"{p.id}: every offer needs the provider's own name for it"
            assert o.usd_per_hour is None or o.usd_per_hour >= 0.0, (p.id, o.label)


def test_a_free_tier_states_its_limits():
    """`free` with no limits is marketing, not information."""
    for p in host.REGISTRY:
        if p.free is not None:
            assert p.free.what, p.id
            assert p.free.limits, f"{p.id}: a free tier with no stated limit is a trap"


def test_every_price_shown_carries_its_provenance():
    for o in survey().options:
        assert o.provider.source in o.price_note
        if o.usd_per_hour is not None:
            assert o.provider.checked in o.price_note
    # And the machine-readable form carries it too — a JSON consumer must not
    # get a bare number.
    for row in host.to_dict(survey())["options"]:
        assert row["price_source"] and row["price_checked"]
        assert row["estimate"] == "roofline, not measured"


def test_an_unpublished_price_renders_as_a_question_mark():
    """Never a plausible-looking guess. Vast.ai is market-priced, so it has none."""
    vast = next(p for p in host.REGISTRY if p.id == "vast")
    assert all(o.usd_per_hour is None for o in vast.offers)

    rendered = survey().render()
    assert "?" in rendered
    unpriced = [o for o in survey().options if o.usd_per_hour is None]
    assert unpriced, "the fixture depends on at least one unpriced shape existing"
    assert all(o.usd_per_mtok is None for o in unpriced), "no price means no $/Mtok"


def test_the_render_dates_its_prices_and_labels_its_estimates():
    text = survey().render()
    assert host.CHECKED in text
    assert "roofline estimates, not measurements" in text
    assert "Re-read the source" in text, "staleness must be stated, not implied"


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #


def test_ranking_is_genuinely_cheapest_first():
    opts = host.options(catalog.get(QWEN), context=32_768, concurrency=4)
    assert len(opts) > 5, "the fixture needs a real spread to test ordering"
    priced = [o.usd_per_hour for o in opts if o.usd_per_hour is not None]
    assert priced == sorted(priced), priced


def test_unpriced_shapes_sort_last_not_as_zero():
    """`?` cannot be compared to a number. Sorting it as 0 puts the one option
    we cannot cost at the very top, above a genuinely free tier."""
    opts = host.options(catalog.get(QWEN), context=32_768, concurrency=4)
    first_unpriced = next(i for i, o in enumerate(opts) if o.usd_per_hour is None)
    assert all(o.usd_per_hour is None for o in opts[first_unpriced:])
    assert first_unpriced > 0


def test_a_fitting_free_tier_is_the_first_row():
    s = survey(QWEN, context=32_768, concurrency=4)
    assert s.cheapest is not None and s.cheapest.free
    assert s.cheapest.provider.id == "hf-zerogpu"
    assert s.cheapest.usd_per_mtok == 0.0


def test_free_only_returns_only_free_shapes():
    opts = host.options(catalog.get(QWEN), context=8192, free_only=True)
    assert opts, "a 30B MoE fits a free 96 GB card"
    assert all(o.free and o.usd_per_hour == 0.0 for o in opts)


def test_cost_per_mtok_rewards_bandwidth_not_just_capacity():
    """A cheap slow card can cost more per token than an expensive fast one —
    the trap this ranking exists to expose."""
    s = survey(QWEN, context=8192, concurrency=8)
    paid = [o for o in s.options if o.usd_per_mtok and not o.free]
    assert paid
    by_hour = min(paid, key=lambda o: o.usd_per_hour)
    by_token = min(paid, key=lambda o: o.usd_per_mtok)
    assert by_hour.usd_per_mtok > 0 and by_token.usd_per_mtok > 0
    assert by_token.usd_per_mtok <= by_hour.usd_per_mtok


def test_sizing_agrees_with_the_fit_solver():
    """No second implementation. An option's memory answer must be the one
    `clickllm fit` would give for the same hardware."""
    from clickllm import fit

    o = survey(QWEN, context=8192, concurrency=1).options[0]
    hw = host._profile(o.offer.profile_id).to_hardware()
    expected = fit.best_quant(catalog.get(QWEN), hw, 8192, 1)
    assert expected is not None
    assert o.fit.quant == expected.quant
    assert o.fit.total_bytes == expected.total_bytes


def test_more_context_never_opens_up_more_options():
    narrow = survey(QWEN, context=8192, concurrency=1)
    wide = survey(QWEN, context=131_072, concurrency=8)
    assert len(wide.options) <= len(narrow.options)


def test_a_forced_quant_is_honoured():
    s = survey(QWEN, quant="q4", context=8192)
    assert s.options and all(o.fit.quant == "q4" for o in s.options)


# --------------------------------------------------------------------------- #
# Exclusions — the reason is the product
# --------------------------------------------------------------------------- #


def test_a_free_tier_that_cannot_fit_is_excluded_with_the_shortfall():
    s = survey(DEEPSEEK, context=131_072, concurrency=4)
    assert not any(o.provider.id == "hf-zerogpu" for o in s.options), (
        "a free 96 GB card must never be offered for a 671B model"
    )
    zero = next(e for e in s.excluded if e.provider_id == "hf-zerogpu")
    assert zero.was_free
    assert "Short by" in zero.reason
    assert "96 GB" in zero.reason and "GB" in zero.reason


def test_every_exclusion_states_a_reason():
    for model_id in (QWEN, DEEPSEEK, KIMI):
        s = survey(model_id, context=32_768, concurrency=1)
        assert all(e.reason for e in s.excluded), model_id
        assert all(e.provider_name for e in s.excluded), model_id


def test_a_shortfall_never_renders_as_zero():
    """Borrowed from `fit`: a deficit under half a gigabyte printed as
    "short by 0 GB", which reads as a solver bug rather than as the answer."""
    for model_id in (DEEPSEEK, KIMI):
        for e in survey(model_id, context=32_768).excluded:
            assert not re.search(r"Short by 0 (GB|MB)\b", e.reason), e.reason


def test_nothing_fits_a_28_trillion_parameter_model_and_it_says_so():
    s = survey(KIMI, context=8192, concurrency=1)
    assert not s.options
    assert s.cheapest is None
    assert s.excluded
    assert "Nothing in the provider registry" in s.render()


def test_colab_is_excluded_by_its_terms_not_by_its_memory():
    """The most useful thing this module can say about the free notebook tiers:
    the reason is policy, so no smaller model makes it available."""
    for model_id, ctx in ((QWEN, 8192), (KIMI, 8192)):
        colab = next(e for e in survey(model_id, context=ctx).excluded if e.provider_id == "colab")
        assert "terms" in colab.reason
        assert "interactive compute" in colab.reason
        assert "Short by" not in colab.reason, "this is not a capacity answer"
    assert not any(o.provider.id == "colab" for o in survey(QWEN).options)


def test_free_credit_is_not_reported_as_a_free_tier_exclusion():
    """Modal gives credit against paid hardware. Losing that to a model needing
    an 8-GPU node is not "the free tier does not fit"."""
    s = survey(DEEPSEEK, context=131_072, concurrency=4)
    modal = next(e for e in s.excluded if e.provider_id == "modal")
    assert not modal.was_free
    assert "FREE TIER" not in s.render().split("NOT AVAILABLE")[1].split("Modal")[1][:40]


# --------------------------------------------------------------------------- #
# Artifacts — standalone, stamped, credential-free
# --------------------------------------------------------------------------- #

ARTIFACT_PROVIDERS = ("hf-zerogpu", "runpod", "modal", "hf-endpoints", "together", "fireworks")


@pytest.fixture
def picked():
    return survey(QWEN, context=32_768, concurrency=4)


@pytest.mark.parametrize("provider_id", ARTIFACT_PROVIDERS)
def test_artifact_carries_a_provenance_header(picked, provider_id):
    option = host.find(picked, provider_id)
    assert option is not None, provider_id
    art = host.artifact(option, today="2026-07-28")
    assert art.files and art.how
    for name, content in art.files:
        assert "clickllm host — generated 2026-07-28" in content, name
        assert option.provider.name in content, name
        # What was chosen, why, and where the price came from.
        assert "OPTION" in content and "WHY" in content, name
        assert option.provider.source in content, name


@pytest.mark.parametrize("provider_id", ARTIFACT_PROVIDERS)
def test_artifact_never_imports_clickllm(picked, provider_id):
    """NFR-4: generated config is native and standalone. Delete this tool and
    the files still deploy."""
    art = host.artifact(host.find(picked, provider_id), today="2026-07-28")
    for name, content in art.files:
        assert not re.search(r"\b(import|install)\s+clickllm\b", content), name
        assert "from clickllm import" not in content, name
        if name.endswith(".py"):
            tree = ast.parse(content)
            imported = {
                alias.name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            } | {
                (node.module or "").split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            assert "clickllm" not in imported, (name, imported)


@pytest.mark.parametrize("provider_id", ARTIFACT_PROVIDERS)
def test_artifact_states_that_prices_go_stale(picked, provider_id):
    art = host.artifact(host.find(picked, provider_id), today="2026-07-28")
    for name, content in art.files:
        assert "Re-read that page" in content, name


def test_docker_artifact_is_a_real_engine_invocation(picked):
    art = host.artifact(host.find(picked, "runpod"), today="2026-07-28")
    files = dict(art.files)
    assert set(files) == {"docker-compose.yml", "run.sh"}
    compose = files["docker-compose.yml"]

    # The engine's own image and the model repo, not a clickllm wrapper.
    assert "vllm/vllm-openai" in compose or "sglang" in compose
    assert catalog.get(QWEN).repo in compose
    # Shared memory: the default 64 MB fails as a cryptic hang, not an error.
    assert "ipc: host" in compose
    # Every command arg is a quoted YAML string. A bare `32768` parses as an
    # integer and compose refuses a non-string in `command`.
    block = compose.split("    command:\n")[1].split("    ports:")[0]
    args = re.findall(r"^      - (.+)$", block, flags=re.M)
    assert args and all(a.startswith('"') and a.endswith('"') for a in args), args
    parsed = [json.loads(a) for a in args]
    assert all(isinstance(a, str) for a in parsed), parsed
    assert "32768" in parsed, "the context length must survive as a string, not an int"
    assert catalog.get(QWEN).repo in parsed, "the model is an argument, not a substring"


def test_docker_artifact_passes_a_token_through_rather_than_embedding_one(picked):
    """A gated repo needs a token. It comes from the operator's shell, and no
    generated file ever contains one."""
    art = host.artifact(host.find(picked, "runpod"), today="2026-07-28")
    compose = dict(art.files)["docker-compose.yml"]
    # Named for passthrough — with no value attached to it.
    assert "- HUGGING_FACE_HUB_TOKEN" in compose
    assert not re.search(r"TOKEN\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{8,}", compose)


def test_modal_artifact_is_importable_python_naming_a_real_accelerator(picked):
    art = host.artifact(host.find(picked, "modal"), today="2026-07-28")
    app = dict(art.files)["modal_app.py"]
    ast.parse(app)  # must be syntactically valid Python
    option = host.find(picked, "modal")
    assert f'GPU = "{option.offer.sku}"' in app
    assert option.offer.sku, "an artifact is only emitted for a confirmed SKU"
    assert "modal.web_server" in app and "scaledown_window" in app


def test_modal_refuses_a_shape_whose_accelerator_name_was_never_confirmed(picked):
    """Guessing a provider's own identifier produces an app that fails at deploy
    with their error message, minutes later. Refuse instead."""
    option = host.find(picked, "modal")
    from dataclasses import replace

    blind = replace(option, offer=replace(option.offer, sku=""))
    with pytest.raises(ValueError, match="not recorded"):
        host.artifact(blind, today="2026-07-28")


def test_zerogpu_artifact_is_a_space_not_a_server(picked):
    """ZeroGPU takes the card back between calls, so a persistent vLLM process
    would be killed. The artifact must be the shape the platform actually is."""
    art = host.artifact(host.find(picked, "hf-zerogpu"), today="2026-07-28")
    files = dict(art.files)
    assert set(files) == {"app.py", "requirements.txt", "README.md"}
    ast.parse(files["app.py"])
    assert "@spaces.GPU" in files["app.py"]
    assert "vllm" not in files["app.py"]
    assert "sdk: gradio" in files["README.md"]
    # The free tier's limits travel with the artifact, not just the terminal.
    assert "daily GPU-seconds quota" in files["README.md"]
    # And the roofline is not passed off as what transformers will do.
    assert "not as what this code will do" in files["README.md"]


def test_manual_artifact_says_why_there_is_no_generated_config(picked):
    """Emitting a config against unconfirmed API field names looks like an
    answer and fails inside someone else's deploy."""
    art = host.artifact(host.find(picked, "hf-endpoints"), today="2026-07-28")
    body = dict(art.files)["README.md"]
    assert "No generated config" in body
    assert "not verified" in body
    assert art.how, "a manual artifact without steps is just a refusal"


def test_artifact_refuses_a_model_with_no_known_repo(picked):
    from dataclasses import replace

    option = host.find(picked, "runpod")
    nameless = replace(
        option,
        placement=replace(
            option.placement, fit=replace(option.fit, model=replace(option.fit.model, repo=None))
        ),
    )
    with pytest.raises(ValueError, match="no Hugging Face repo"):
        host.artifact(nameless, today="2026-07-28")


def test_artifact_write_puts_every_file_on_disk(tmp_path: Path):
    s = survey(QWEN, context=8192, concurrency=4)
    art = host.artifact(host.find(s, "runpod"), today="2026-07-28")
    written = art.write(tmp_path / "deploy")
    assert len(written) == len(art.files)
    for path in written:
        assert path.exists() and path.read_text()


# --------------------------------------------------------------------------- #
# The two promises the module makes about itself
# --------------------------------------------------------------------------- #

SOURCE = Path(host.__file__).read_text()


def test_no_code_path_reads_anything_that_looks_like_a_credential():
    """clickllm handles no credentials: never prompted for, never read, never
    stored, never transmitted. Enforced against the source, not by convention."""
    tree = ast.parse(SOURCE)
    reads = []
    for node in ast.walk(tree):
        # os.environ[...] / os.environ.get(...) / os.getenv(...)
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            reads.append(ast.dump(node))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"input", "getpass"}
        ):
            reads.append(node.func.id)
    assert not reads, f"host.py must not read the environment or prompt: {reads}"
    for word in ("os.environ", "getenv", "getpass", "input("):
        assert word not in SOURCE, word


def test_planning_makes_no_network_call():
    """Zero egress by default (NFR-2). Prices come off the dated registry; a
    background sync to a vendor would be exactly what this repo promises not
    to do."""
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(ast.parse(SOURCE))
        if isinstance(node, ast.ImportFrom)
    }
    banned = {"urllib", "http", "requests", "socket", "httpx", "ssl", "ftplib"}
    assert not (imported & banned), f"host.py must not reach the network: {imported & banned}"


def test_module_self_check_passes():
    host.demo()
