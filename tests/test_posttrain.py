"""Post-training recommendations.

`posttrain.demo()` walks one worked example. These pin the refusals, which are
the load-bearing half: a module that recommends training for every gap is a
module that will happily spend a GPU budget on a problem training cannot solve.
"""

from __future__ import annotations

import pytest

from clickllm.posttrain import (
    MIN_EXAMPLES_PER_CLUSTER,
    WIDE_GAP,
    Method,
    Recipe,
    recommend,
)
from clickllm.prove.equivalence import CandidateReport, ClusterScore
from clickllm.prove.judge import Agreement
from clickllm.prove.receipt import Receipt, issue
from clickllm.prove.stats import wilson


def cs(name: str, passed: int, total: int, share: float = 1.0) -> ClusterScore:
    return ClusterScore(name, name, share, wilson(passed, total), 0)


def receipt(*clusters: ClusterScore, **kw) -> Receipt:
    kw.setdefault("incumbent", "gpt-5")
    kw.setdefault("issued", "2026-07-27")
    kw.setdefault("eval_set", "a" * 64)
    return issue(CandidateReport("glm-5.2", tuple(clusters)), **kw)


def only(rec, cluster: str) -> Recipe | None:
    return next((r for r in rec.recipes if r.cluster == cluster), None)


# --- the refusals --------------------------------------------------------------


def test_a_wide_gap_is_refused_because_training_will_not_bridge_it():
    # 33% against a 90% bar is a different capability level. Selling a training
    # run here would take money for something that does not work.
    r = receipt(cs("reason", 40, 120))
    rec = recommend(r, {"reason": 50_000})
    assert not rec.recipes
    assert any("different capability level" in why for _, why in rec.stay)


def test_the_wide_gap_boundary_is_where_the_constant_says():
    # Just inside, training is offered; just outside, refused. Otherwise the
    # constant is decoration.
    inside = receipt(cs("c", 700, 1000))  # 70%, gap 0.20
    outside = receipt(cs("c", 500, 1000))  # 50%, gap 0.40
    assert pytest.approx(0.35) == WIDE_GAP
    assert only(recommend(inside, {"c": 5000}), "c").method is Method.LORA
    assert not recommend(outside, {"c": 5000}).recipes


def test_too_few_examples_is_refused_with_the_number():
    r = receipt(cs("rare", 100, 140))
    rec = recommend(r, {"rare": 40})
    assert not rec.recipes
    why = dict(rec.stay)["rare"]
    assert "40 captured examples" in why and str(MIN_EXAMPLES_PER_CLUSTER) in why


def test_a_cluster_with_no_capture_data_at_all_is_refused_not_assumed():
    # Absent from the counts means "we have no data", which must not read as
    # "plenty of data".
    rec = recommend(receipt(cs("ghost", 90, 120)), {})
    assert not rec.recipes and rec.stay


def test_the_example_floor_is_where_the_constant_says():
    r = receipt(cs("c", 90, 120))
    assert not recommend(r, {"c": MIN_EXAMPLES_PER_CLUSTER - 1}).recipes
    assert recommend(r, {"c": MIN_EXAMPLES_PER_CLUSTER}).recipes


# --- what it does recommend ----------------------------------------------------


def test_a_small_proven_gap_gets_the_free_fix_before_a_gpu():
    # Reaching for training before trying a prompt change is how a team spends a
    # week on a missing instruction.
    rec = recommend(receipt(cs("format", 860, 1000)), {"format": 4000})
    fix = only(rec, "format")
    assert fix.method is Method.PROMPT
    assert "reversible" in " ".join(fix.risks)
    assert not any("LoRA" in s for s in fix.steps)


def test_a_real_gap_with_real_data_distils_from_captured_incumbent_output():
    rec = recommend(receipt(cs("summarise", 90, 120)), {"summarise": 3000})
    lora = only(rec, "summarise")
    assert lora.method is Method.LORA
    # The whole thesis: the teacher is the incumbent, on your own traffic.
    assert "gpt-5 gave" in lora.source
    assert "already redacted" in lora.source.lower()


def test_the_recipe_insists_on_the_same_eval_set():
    # Re-proving against a fresh eval set is the commonest way a fine-tune
    # appears to work when it did not.
    lora = only(recommend(receipt(cs("c", 90, 120)), {"c": 3000}), "c")
    assert any("SAME eval set" in s for s in lora.steps)


def test_the_recipe_says_captured_output_is_not_ground_truth():
    lora = only(recommend(receipt(cs("c", 90, 120)), {"c": 3000}), "c")
    joined = " ".join(lora.risks)
    assert "not ground truth" in joined
    assert "incumbent's failures" in joined


def test_training_is_scoped_to_the_failing_cluster_only():
    r = receipt(cs("good", 118, 120, 0.5), cs("bad", 90, 120, 0.5))
    rec = recommend(r, {"good": 9000, "bad": 3000})
    assert {x.cluster for x in rec.recipes} == {"bad"}
    assert any("only" in s for s in only(rec, "bad").steps)


# --- premises ------------------------------------------------------------------


def test_unproven_clusters_are_excluded_and_the_exclusion_is_stated():
    r = receipt(cs("real", 90, 120, 0.5), cs("thin", 8, 10, 0.5))
    rec = recommend(r, {"real": 3000, "thin": 3000})
    assert "thin" not in {x.cluster for x in rec.recipes}
    assert any("never proven either way" in n for n in rec.notes)


def test_an_untrustworthy_judge_undermines_the_whole_premise():
    r = receipt(cs("c", 90, 120), agreement=Agreement(agreed=10, total=20, model="j"))
    assert any("Fix the judge" in n for n in recommend(r, {"c": 3000}).notes)


def test_a_trustworthy_judge_raises_no_such_note():
    r = receipt(cs("c", 90, 120), agreement=Agreement(agreed=19, total=20, model="j"))
    assert not any("Fix the judge" in n for n in recommend(r, {"c": 3000}).notes)


def test_a_clean_receipt_recommends_nothing():
    rec = recommend(receipt(cs("extract", 118, 120)), {"extract": 9000})
    assert not rec.recipes and not rec.stay and not rec.worth_training
    assert "Nothing to do" in rec.render()


# --- the module never acts -----------------------------------------------------


def test_nothing_here_can_start_a_training_run():
    # Fine-tuning is expensive and easy to get wrong. A tool that kicked off a
    # run because a score dipped would be the most dangerous thing in the repo.
    rec = recommend(receipt(cs("c", 90, 120)), {"c": 3000})
    for obj in (rec, *rec.recipes):
        for verb in ("run", "execute", "train", "start", "apply", "submit"):
            assert not hasattr(obj, verb), f"{type(obj).__name__}.{verb} exists"


def test_what_to_leave_alone_renders_before_what_to_train():
    r = receipt(cs("hopeless", 40, 120, 0.5), cs("fixable", 90, 120, 0.5))
    text = recommend(r, {"hopeless": 5000, "fixable": 3000}).render()
    assert text.index("LEAVE ON THE INCUMBENT") < text.index("WORTH TRAINING")


def test_every_recipe_carries_steps_and_a_stated_risk():
    r = receipt(cs("small", 860, 1000, 0.5), cs("big", 90, 120, 0.5))
    rec = recommend(r, {"small": 4000, "big": 3000})
    assert rec.recipes
    for x in rec.recipes:
        assert x.steps and all(len(s) > 20 for s in x.steps)
        assert x.risks, f"{x.cluster} has no stated risk"
        assert x.render()
