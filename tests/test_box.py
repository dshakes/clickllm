"""The box — ADR-0005's artifact.

`box.demo()` walks a whole build. These pin the promises a box makes about
itself, which are all promises about what it will *not* say:

1. the tree is the shape ADR-0005 committed to, and what is missing from it is
   declared rather than quietly absent
2. `weights.lock` never claims a revision it does not have
3. no eval score exists without a receipt behind it
4. nothing generated needs clickllm installed (NFR-4)
5. the argv in `targets/` is the engine's own, from the same adapter
   `clickllm run` uses — a box that drifts from the launcher is two answers
6. nothing reads anything credential-shaped, and nothing is pushed

Every test is offline: `exists` and `revision` are injected, and the resolution
cache is written under `tmp_path`.
"""

from __future__ import annotations

import ast
import json
import re
from itertools import takewhile
from pathlib import Path

import pytest

from clickllm import box, catalog
from clickllm.engines import adapter_for
from clickllm.hardware_catalog import get as profile_by_id
from clickllm.k8s.reconcile import VLLM_FAMILY
from clickllm.plan import Requirements, Workload
from clickllm.plan import plan as configure
from clickllm.prove.equivalence import CandidateReport, ClusterScore
from clickllm.prove.receipt import Receipt, issue
from clickllm.prove.stats import wilson

MODEL = "qwen3-30b-a3b"
TODAY = "2026-07-28"
SHA = "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"


def build(model: str = MODEL, *, tmp_path: Path, revision=None, **kw) -> box.Box:
    """A box built without touching the network."""
    return box.build(
        model,
        today=TODAY,
        exists=lambda repo: True,
        revision=revision or (lambda repo: SHA),
        cache=tmp_path / "weights.json",
        **kw,
    )


@pytest.fixture
def built(tmp_path):
    return build(tmp_path=tmp_path)


def receipt_for(candidate: str) -> Receipt:
    return issue(
        CandidateReport(
            candidate,
            (
                ClusterScore("extract", "extraction", 0.6, wilson(118, 120), 0),
                ClusterScore("long", "long context", 0.4, wilson(41, 80), 0),
            ),
        ),
        incumbent="gpt-5",
        issued=TODAY,
        eval_set="a" * 64,
    )


# --- the tree ------------------------------------------------------------------


def test_the_tree_is_the_shape_adr_0005_committed_to(built):
    names = {n for n, _ in built.files}
    assert {"manifest.yaml", "weights.lock", "bench.json", "README.md"} <= names

    for target in ("linux-cuda", "linux-rocm"):
        assert {
            f"targets/{target}/Dockerfile",
            f"targets/{target}/docker-compose.yml",
            f"targets/{target}/kubernetes.yaml",
        } <= names, target
    # macOS gets a native launch spec, because no container can reach Metal.
    assert {"targets/darwin-metal/launch.json", "targets/darwin-metal/serve.sh"} <= names
    assert not any(n.startswith("targets/darwin-metal/Docker") for n in names)


def test_what_is_not_in_the_box_is_declared_rather_than_missing(built):
    """`targets/cpu/` and a measured bench are both absent for stated reasons.

    An artifact that silently omits two entries from the tree its own ADR
    specifies is indistinguishable from one that failed to build them.
    """
    manifest = dict(built.files)["manifest.yaml"]
    assert "targets/cpu/" in manifest
    assert "llama.cpp" in manifest and "no verified flag dialect" in manifest
    assert "targets/cpu/" in dict(built.files)["README.md"]


def test_every_file_is_written_including_the_nested_ones(built, tmp_path):
    written = built.write(tmp_path / "out")
    assert len(written) == len(built.files)
    assert all(p.exists() and p.read_text() for p in written)
    assert (tmp_path / "out" / "targets" / "linux-cuda" / "Dockerfile").exists()


# --- weights.lock --------------------------------------------------------------


def test_the_lock_records_the_repo_and_the_revision_it_was_given(built):
    lock = dict(built.files)["weights.lock"]
    for weight in built.weights:
        assert weight.revision == SHA and weight.pinned
        assert weight.repo in lock
        assert f"https://huggingface.co/{weight.repo}" in lock
    assert SHA in lock and "pinned: true" in lock


def test_the_lock_never_claims_a_revision_it_does_not_have(tmp_path):
    """No revision means `pinned: false` and a reason — never `main`.

    A lock that pinned a moving ref because the index would not give it a SHA is
    a lock in name only, and the first person to trust it is the one who cannot
    reproduce a regression.
    """
    b = build(tmp_path=tmp_path, revision=lambda repo: None)
    lock = dict(b.files)["weights.lock"]

    assert b.weights and all(not w.pinned for w in b.weights)
    assert all(w.unpinned_reason for w in b.weights)
    assert "revision: null" in lock and "pinned: false" in lock
    assert "unpinned_reason" in lock and "not reproducible" in lock
    assert not re.search(r"revision:\s*\"?(main|master|HEAD)\"?", lock)
    # ...and the fetch command does not pin one either.
    assert "--revision" not in lock


def test_the_lock_says_weights_are_fetched_not_vendored(built):
    lock = dict(built.files)["weights.lock"]
    assert "vendored: false" in lock
    assert "Nothing in this box contains model weights" in lock


def test_each_repo_is_locked_once_and_names_the_targets_that_use_it(built):
    """MLX serves a different repo from vLLM — precision is baked into the
    weights — so the lock is per repo, not per target."""
    repos = [w.repo for w in built.weights]
    assert len(repos) == len(set(repos))
    used = {t for w in built.weights for t in w.targets}
    assert used == {t.target.id for t in built.targets}
    mlx = next(w for w in built.weights if "mlx-community" in w.repo)
    assert mlx.targets == ("darwin-metal",)


# --- evidence ------------------------------------------------------------------


def test_no_receipt_means_no_score_anywhere_in_the_box(built):
    """Invariant 6: a fabricated number is worse than an absent one."""
    manifest = dict(built.files)["manifest.yaml"]
    assert "proven: false" in manifest
    assert "scores: null" in manifest
    assert "no proof was run" in manifest
    assert not built.proven
    assert "no quality claim" in built.render()
    # Nothing in the tree quotes a pass rate.
    for name, content in built.files:
        assert not re.search(r"\b(accuracy|pass[_ ]rate|score)\s*[:=]\s*[0-9.]+", content), name


def test_a_receipt_is_carried_with_its_intervals_and_its_regret_set(tmp_path):
    b = build(tmp_path=tmp_path, receipt=receipt_for(MODEL))
    manifest = dict(b.files)["manifest.yaml"]
    assert b.proven and "proven: true" in manifest
    # The point estimate never appears without the interval and the denominator.
    assert "98% [94%–100%] over 120 items" in manifest
    # The cluster that failed the bar is in the document, not just the one that passed.
    assert "regret" in manifest and "long context" in manifest
    assert "must stay on" in manifest.lower()


def test_a_receipt_for_a_different_model_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not 'qwen3-30b-a3b'"):
        build(tmp_path=tmp_path, receipt=receipt_for("some-other-model"))


def test_bench_json_says_it_measured_nothing(built):
    bench = json.loads(dict(built.files)["bench.json"])
    assert bench["measured"] is False
    assert "roofline estimate" in bench["note"]
    assert bench["estimates"]
    for row in bench["estimates"]:
        assert row["measured"] is False
        assert "not measured" in row["confidence"] or "?" in row["confidence"]
        assert row["method"]


# --- standalone ----------------------------------------------------------------


def test_nothing_generated_imports_or_installs_clickllm(built):
    """NFR-4: delete this tool and the box still deploys."""
    for name, content in built.files:
        assert not re.search(r"\b(import|install)\s+clickllm\b", content), name
        assert "from clickllm import" not in content, name
        # The word may appear in provenance, but never as something to run.
        assert not re.search(r"^\s*(RUN|CMD|exec|\$)\s.*clickllm", content, re.M), name


def test_the_container_files_are_the_engines_own_image_and_flags(built):
    files = dict(built.files)
    compose = files["targets/linux-cuda/docker-compose.yml"]
    assert "vllm/vllm-openai" in compose
    assert catalog.get(MODEL).repo in compose
    # Shared memory: the default 64 MB fails as a cryptic hang, not an error.
    assert "ipc: host" in compose
    # Every command arg is a quoted YAML string — a bare 32768 is an int, and
    # one flag carries JSON that would otherwise parse as a mapping.
    lines = compose.splitlines()
    start = lines.index("    command:")
    args = list(takewhile(lambda ln: ln.startswith("      - "), lines[start + 1 :]))
    assert args and all(ln.startswith('      - "') for ln in args), args
    assert files["targets/linux-cuda/Dockerfile"].rstrip().splitlines()[-1].startswith("CMD [")

    # This used to also assert a JSON-carrying arg was present, since that is
    # *why* everything is quoted. No single-node target emits one any more:
    # eagle3 now requires a named draft checkpoint, and the only families that
    # speculate without one (MTP-native: glm-5.2, deepseek-v3, deepseek-v4-pro)
    # are 355B+ and fit no single node. The blanket quoting rule above is the
    # invariant that matters and still covers the JSON case if it returns.


def test_the_rocm_target_uses_amds_image_and_amds_device_plugin(built):
    """`nvidia.com/gpu` on a ROCm cluster schedules nowhere, forever, silently."""
    files = dict(built.files)
    compose = files["targets/linux-rocm/docker-compose.yml"]
    assert "rocm/vllm" in compose
    assert "/dev/kfd" in compose and "/dev/dri" in compose
    assert "driver: nvidia" not in compose

    manifests = files["targets/linux-rocm/kubernetes.yaml"]
    assert "amd.com/gpu" in manifests and "nvidia.com/gpu" not in manifests
    assert "rocm/vllm" in manifests


def test_the_kubernetes_target_carries_a_deployment_and_a_service(built):
    manifests = dict(built.files)["targets/linux-cuda/kubernetes.yaml"]
    assert 'kind: "Deployment"' in manifests and 'kind: "Service"' in manifests
    assert manifests.count("---") == 1
    # The operator's own provenance annotation, so a cluster and a box agree.
    assert "clickllm.dev/standalone" in manifests


def test_the_native_target_is_a_launch_spec_not_a_container(built):
    spec = json.loads(dict(built.files)["targets/darwin-metal/launch.json"])
    assert spec["binding"] == "native" and spec["engine"] == "mlx"
    assert "Metal is a macOS-only API" in spec["why"]
    assert spec["repo"].startswith("mlx-community/")
    assert spec["install"] == "pip install mlx-lm"
    # The endpoint the box advertises is the one the argv actually binds:
    # mlx_lm.server's own default is port 8080.
    assert spec["endpoint"] == "http://127.0.0.1:8000/v1"
    assert spec["argv"][-4:] == ["--host", "127.0.0.1", "--port", "8000"]
    # What MLX cannot express is stated, not dropped.
    assert any("context-length" in gap for gap in spec["not_expressed"])


# --- drift ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target_id", "path"),
    [
        ("linux-cuda", "targets/linux-cuda/docker-compose.yml"),
        ("linux-rocm", "targets/linux-rocm/docker-compose.yml"),
        ("darwin-metal", "targets/darwin-metal/launch.json"),
    ],
)
def test_the_argv_is_what_the_engine_adapter_produces_for_these_settings(built, target_id, path):
    """The drift guard. A box whose flags disagree with `clickllm run`'s flags
    is two answers to one question, and only one of them ever ran."""
    t = next(t for t in built.targets if t.target.id == target_id)
    hw = profile_by_id(t.target.profile_id).to_hardware()
    p = configure(
        hw,
        Requirements(Workload.INTERACTIVE, concurrency=8, context=32_768),
        catalog.get(MODEL),
        t.quant,
    )
    expected, _gaps = adapter_for(t.engine).command(t.repo, p.settings())

    # The recorded argv is the adapter's, plus only the bind address.
    assert list(t.argv)[: len(expected)] == expected
    assert list(t.argv)[len(expected) :][:1] == ["--host"]

    content = dict(built.files)[path]
    if t.target.binding == "container":
        # The image's entrypoint supplies the binary, so the file drops it.
        dropped = 1 if t.engine in VLLM_FAMILY else 3
        for arg in t.argv[dropped:]:
            assert json.dumps(arg) in content, arg
        assert json.dumps(t.argv[0]) not in content
    else:
        assert json.loads(content)["argv"] == list(t.argv)


def test_the_manifest_records_the_planners_knobs_with_their_reasons(built):
    """The knobs come from `plan`, and each carries the reasoning that produced
    it — a manifest of bare numbers is a config file, which is what this repo
    refuses to hand anyone."""
    manifest = dict(built.files)["manifest.yaml"]
    for t in built.targets:
        assert t.knobs
        for knob in t.knobs:
            assert len(knob.why) > 30
            assert knob.why.split("\n")[0][:40] in manifest


def test_each_target_is_tuned_for_its_own_host_class(built):
    """Tuning is not portable. An H100 fraction on a 192 GB MI300X is wrong."""
    by_id = {t.target.id: t for t in built.targets}
    cuda, rocm = by_id["linux-cuda"], by_id["linux-rocm"]
    assert cuda.fit.usable_bytes != rocm.fit.usable_bytes
    assert cuda.argv != rocm.argv, "same flags on both means one of them was not solved"
    manifest = dict(built.files)["manifest.yaml"]
    assert "re-solve" in manifest and "re-benchmark" in manifest


# --- refusals ------------------------------------------------------------------


def test_a_target_the_model_does_not_fit_is_reported_not_dropped(tmp_path):
    small = box.Target("tiny-gpu", "l4", "container", "24 GB card")
    b = build(tmp_path=tmp_path, targets=(box.TARGETS[0], small))
    assert [t.target.id for t in b.targets] == ["linux-cuda"]
    assert [s.target.id for s in b.skipped] == ["tiny-gpu"]
    assert "does not fit" in b.skipped[0].reason and "GB usable" in b.skipped[0].reason
    assert "tiny-gpu" in dict(b.files)["manifest.yaml"]
    assert "Not built" in dict(b.files)["README.md"]


def test_a_box_with_no_buildable_target_raises_rather_than_shipping_empty(tmp_path):
    tiny = box.Target("tiny-gpu", "l4", "container", "24 GB card")
    with pytest.raises(ValueError, match="no target could be built"):
        build("deepseek-v3", tmp_path=tmp_path, targets=(tiny,))


def test_an_unresolvable_repo_skips_its_target_and_says_what_was_tried(tmp_path):
    b = box.build(
        MODEL,
        today=TODAY,
        exists=lambda repo: "mlx-community" not in repo,
        revision=lambda repo: SHA,
        cache=tmp_path / "weights.json",
    )
    skipped = {s.target.id: s for s in b.skipped}
    assert "darwin-metal" in skipped
    assert skipped["darwin-metal"].tried, "a resolution failure lists what it checked"
    assert "mlx-community/Qwen3-30B-A3B-8bit" in skipped["darwin-metal"].tried


def test_a_quant_the_model_does_not_publish_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not publish"):
        build(tmp_path=tmp_path, quant="q2")


# --- the boundary --------------------------------------------------------------

SOURCE = Path(box.__file__).read_text()


def test_no_code_path_reads_anything_that_looks_like_a_credential():
    """clickllm holds no registry credential: never prompted for, never read,
    never stored. Enforced against the source, not by convention."""
    reads = []
    for node in ast.walk(ast.parse(SOURCE)):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            reads.append(ast.dump(node))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"input", "getpass"}
        ):
            reads.append(node.func.id)
    assert not reads, f"box.py must not read the environment or prompt: {reads}"
    for word in ("os.environ", "getenv", "getpass", "input(", "netrc", "keyring"):
        assert word not in SOURCE, word


def test_the_only_token_in_the_tree_is_one_the_user_passes_through(built):
    """A gated repo needs a token. Compose *names* the variable and reads it from
    the user's own shell — no value is ever written into the box."""
    # A credential *given a value*. Prose about tokens is not a credential; a
    # line reading `HF_TOKEN=hf_...` is, and must never be written into a box.
    assigned = re.compile(
        r"\b(?:[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|KEY)|password|secret|api[-_]?key)\b"
        r"\s*[:=]\s*\S"
    )
    for name, content in built.files:
        for line in content.splitlines():
            assert not assigned.search(line), (name, line)
    compose = dict(built.files)["targets/linux-cuda/docker-compose.yml"]
    # Named, never valued: compose reads it from the environment of whoever runs it.
    assert "      - HUGGING_FACE_HUB_TOKEN\n" in compose


def test_publishing_is_a_command_you_run_never_something_the_build_did(built, tmp_path):
    commands = built.push_commands(tmp_path)
    assert any(c.startswith("oras push") for c in commands)
    assert any("docker build" in c and "docker push" in c for c in commands)
    assert any("cosign sign" in c for c in commands)
    assert all("--password" not in c and "login" not in c for c in commands)
    assert box.ARTIFACT_TYPE in " ".join(commands)
    # The README says who runs them.
    assert "clickllm does not push" in dict(built.files)["README.md"]


def test_a_custom_ref_is_the_one_that_gets_printed(tmp_path):
    b = build(tmp_path=tmp_path, ref="ghcr.io/acme/triage:v3")
    assert all("ghcr.io/acme/triage:v3" in c for c in b.push_commands(tmp_path)[1:])


def test_building_twice_produces_identical_bytes(tmp_path):
    """A box is content-addressed once it is pushed. Two builds of the same
    inputs that differ in a byte are two artifacts in the registry."""
    assert build(tmp_path=tmp_path).files == build(tmp_path=tmp_path).files


def test_demo_runs():
    box.demo()
