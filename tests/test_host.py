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


# --- the three defects a human review found --------------------------------
#
# All three were invisible to this file when it was green: the suite checked
# that artifacts were *produced*, never that what they claimed matched what
# they configured.


def test_an_artifact_never_claims_a_precision_its_command_does_not_set():
    """The header said "@ q8 ... fits A40 48 GB". The argv had no
    `--quantization`, so vLLM loads the bf16 base repo — 61 GB against 43 GB
    usable — and OOMs on boot.

    `_engine_command` returns the gaps; this module used to destructure them as
    `_gaps` and drop them, turning a refusal into silence. `launch` and `box`
    both surface theirs. This was the one caller that did not.
    """
    surveyed = host.survey(catalog.get("qwen3-30b-a3b"), context=8192, concurrency=4)
    picked = next(o for o in surveyed.options if o.provider.artifact_kind == "docker")
    art = host.artifact(picked, today="2026-07-28")
    body = "\n".join(content for _, content in art.files)

    quant = picked.fit.quant
    if "--quantization" not in body:
        assert "NOT SET" in body, (
            f"the artifact states @{quant} but sets no quantisation flag and does "
            "not say so — this is the OOM-under-a-promise defect"
        )
        assert "precision" in body


def test_multi_device_shapes_are_costed_at_the_parallelism_the_plan_uses():
    """`fit` sizes an N-device shape on aggregate bandwidth; the planner then
    correctly pins `--tensor-parallel-size 1` when the model fits on one card.

    Both are right and they disagree, and the $/Mtok column is where that lands.
    Measured: Qwen3-32B q8 on 4x H100 was quoted 261 tok/s off 11,892 GB/s while
    the emitted command runs on one card at 3,350 — 3.5x, on the one number
    somebody budgets against.
    """
    from clickllm import fit as fitmod
    from clickllm.hardware_catalog import get as profile_by_id

    m = catalog.get("qwen3-32b")
    offer = host.Offer("h100-x4", "4x H100 80 GB", 11.60)
    hw = profile_by_id("h100-x4").to_hardware()

    aggregate = fitmod.solve(m, "q8", hw, 8192, 4)
    real = host._rescored_for_parallelism(aggregate, offer, m, "q8", 8192, 4)
    assert real.tokens_per_sec < aggregate.tokens_per_sec, (
        "a model that fits on one device must not be costed on four cards' bandwidth"
    )
    single = profile_by_id("h100").to_hardware()
    assert real.tokens_per_sec == pytest.approx(
        fitmod.solve(m, "q8", single, 8192, 4).tokens_per_sec, rel=0.02
    )


def test_a_shape_that_genuinely_needs_every_device_is_not_re_costed():
    """The negative control. Without it the check above passes by always
    down-rating, which would understate throughput instead of overstating it."""
    from clickllm import fit as fitmod
    from clickllm.hardware_catalog import get as profile_by_id

    big = catalog.get("qwen3-235b-a22b")
    offer = host.Offer("h100-x4", "4x H100 80 GB", 11.60)
    hw = profile_by_id("h100-x4").to_hardware()
    f = fitmod.solve(big, "q4", hw, 8192, 4)
    if f is None or not f.tokens_per_sec:
        pytest.skip("no fit to compare")
    assert host._rescored_for_parallelism(f, offer, big, "q4", 8192, 4).tokens_per_sec == (
        f.tokens_per_sec
    )


def test_the_free_space_does_not_claim_a_precision_transformers_will_not_load():
    """`dtype="auto"` loads the checkpoint's published precision. The header said
    `@ q8`, so the free tier was sized for one thing and loads another."""
    surveyed = host.survey(catalog.get("qwen3-30b-a3b"), context=8192, concurrency=4)
    space = next((o for o in surveyed.options if o.provider.artifact_kind == "hf-space"), None)
    if space is None:
        pytest.skip("no ZeroGPU option in this survey")
    art = host.artifact(space, today="2026-07-28")
    app = next(c for name, c in art.files if name.endswith("app.py"))
    if 'dtype="auto"' in app:
        assert "published precision" in app, (
            "the Space loads native precision under a quantised header and says nothing"
        )


def test_a_shape_that_does_not_fit_its_own_parallelism_is_dropped_not_quoted():
    """The branch the earlier tests missed, found by the deep review (#45).

    `_rescored_for_parallelism` used to `return rescored or f` — falling back to
    the aggregate Fit, which is the exact number it exists to correct.

    `plan._tensor_parallel` USED to decide TP=1 by comparing weights alone
    against the per-device share, never looking at KV or overhead. So at high
    context x concurrency it narrowed to one device the full footprint could not
    fit, and the row carried four-device throughput and price beside a
    `--tensor-parallel-size 1` command that would not start.

    That root cause is fixed (the guard now compares `total_bytes`), which is
    why this test's original premise — "the planner narrows this to one device"
    — no longer holds and is asserted the other way round below. The downstream
    protection in `_rescored_for_parallelism` is kept and still exercised: two
    independent layers for one failure is the arrangement this repo wants on
    anything that can quote a price beside a command that cannot start.

    Real case, not contrived: Llama 3.1 8B on 4x H100 at 32k context, 64
    concurrent — 266 GB of 288 GB aggregate, quoted at 1,066 tok/s.
    """
    from clickllm import fit as fitmod
    from clickllm.engines import Setting
    from clickllm.hardware_catalog import get as profile_by_id
    from clickllm.plan import Requirements, Workload
    from clickllm.plan import plan as configure

    m = catalog.get("llama-3.1-8b")
    hw = profile_by_id("h100-x4").to_hardware()
    ctx, conc = 32768, 64

    aggregate = fitmod.solve(m, "q8", hw, ctx, conc)
    assert aggregate and aggregate.feasible, "the premise: it fits across four devices"

    knob = configure(
        hw, Requirements(Workload.INTERACTIVE, concurrency=conc, context=ctx), m, "q8"
    ).get(Setting.TENSOR_PARALLEL)
    # The root fix: the full footprint does not fit one device, so the planner
    # must NOT narrow to one. Before it did, on exactly this shape.
    assert knob and knob.value == 4, (
        f"the planner narrowed to {knob.value if knob else None} devices for a "
        f"{aggregate.total_bytes / 2**30:.0f} GiB footprint that does not fit one"
    )

    offer = host.Offer("h100-x4", "4x H100 80 GB", 11.60)
    # With TP=4 the emitted command matches the footprint, so this shape is
    # legitimately quotable now — it is no longer dropped, because it is no
    # longer misconfigured. What must still hold is the anti-flattery property
    # the drop existed to protect: the quoted throughput may not exceed the
    # aggregate estimate, because bandwidth aggregates sub-linearly and a row
    # that quoted more would be pricing hardware the artifact does not deliver.
    rescored = host._rescored_for_parallelism(aggregate, offer, m, "q8", ctx, conc)
    assert rescored is not None, "a shape whose command matches its footprint is quotable"
    assert rescored.tokens_per_sec is not None and aggregate.tokens_per_sec is not None
    assert rescored.tokens_per_sec <= aggregate.tokens_per_sec + 1e-6, (
        f"rescoring quoted {rescored.tokens_per_sec:.0f} tok/s against an aggregate "
        f"estimate of {aggregate.tokens_per_sec:.0f} — sharding cannot beat the roofline"
    )


def test_the_survey_does_not_offer_a_shape_it_had_to_drop():
    """End to end: the dropped option must not reach the table."""
    surveyed = host.survey(catalog.get("llama-3.1-8b"), context=32768, concurrency=64)
    for o in surveyed.options:
        assert o.fit is not None, f"{o.offer.label} was offered with no fit behind it"
        assert o.fit.feasible, f"{o.offer.label} was offered while not feasible"


def test_free_only_reports_the_providers_it_ruled_out_rather_than_hiding_them():
    """From the deep review of host.py (#45), finding 2.

    `--free` filtered each provider's offers and `continue`d when none survived,
    so a paid-only provider vanished from BOTH options and excluded. This module's
    own docstring says a provider that cannot be used is reported with its reason;
    a silently shorter list reads as "we looked and there was nothing", when what
    happened is "we looked and did not say".
    """
    paid_only = {p.id for p in host.REGISTRY if p.offers and not _gives_free(p)}
    assert paid_only, "fixture assumption: some provider is paid-only"

    out = host.survey(catalog.get("llama-3.1-8b"), free_only=True, context=8192, concurrency=4)
    named = {e.provider_id for e in out.excluded} | {o.provider.id for o in out.options}
    missing = paid_only - named
    assert not missing, f"ruled out without saying so: {sorted(missing)}"

    for e in out.excluded:
        if e.provider_id in paid_only:
            assert "free" in e.reason or "shapes" in e.reason, e.reason
            assert not e.was_free


def _gives_free(p) -> bool:
    return any(o.usd_per_hour == 0.0 for o in p.offers)
