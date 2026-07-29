"""M10 — the guard.

`guard.demo()` covers each drift kind in isolation. These cover the thing that
actually makes the module worth having: that the four kinds stay *separated*
under pressure, and that the one nobody else can detect is detected.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from clickllm.guard import (
    MAX_RECEIPT_AGE_DAYS,
    TRAFFIC_DRIFT_LIMIT,
    UNCOVERED_SHARE_LIMIT,
    Drift,
    check,
    traffic_distance,
)
from clickllm.prove.equivalence import CandidateReport, ClusterScore
from clickllm.prove.receipt import Receipt, issue
from clickllm.prove.stats import wilson

TODAY = date(2026, 7, 27)
FRESH = "2026-07-01"
PRINTS = {"gpt-5": "f" * 64, "glm-5.2": "e" * 64}
MIX = {"extract": 0.6, "classify": 0.4}


def cs(name: str, passed: int, total: int, share: float) -> ClusterScore:
    return ClusterScore(name, name, share, wilson(passed, total), 0)


def receipt(issued: str = FRESH, **kw) -> Receipt:
    kw.setdefault("fingerprints", dict(PRINTS))
    return issue(
        CandidateReport("glm-5.2", (cs("extract", 118, 120, 0.6), cs("classify", 96, 100, 0.4))),
        incumbent="gpt-5",
        issued=issued,
        eval_set="a" * 64,
        **kw,
    )


# --- the separation of concerns that is the whole point ------------------------


@pytest.mark.parametrize(
    ("kind", "kwargs", "voids"),
    [
        (Drift.MODEL_CHANGED, dict(fingerprints={"gpt-5": "9" * 64}), True),
        (Drift.TRAFFIC_UNCOVERED, dict(traffic={"extract": 0.5, "new": 0.5}), True),
        (Drift.TRAFFIC_MOVED, dict(traffic={"extract": 0.1, "classify": 0.9}), True),
        (Drift.AGED, dict(), False),
        (Drift.NEW_CANDIDATE, dict(available=frozenset({"qwen-4"})), False),
    ],
)
def test_each_drift_kind_voids_or_does_not_as_designed(kind, kwargs, voids):
    """Only *not knowing whether production is adequate* voids a receipt.

    An old receipt about an unchanged model on unchanged traffic is still a true
    receipt. Flagging it as void would train people to ignore the flag, and then
    they would ignore it for the model swap too.
    """
    old = "2026-01-01" if kind is Drift.AGED else FRESH
    kwargs.setdefault("traffic", dict(MIX))
    p = check(receipt(old), today=TODAY, **kwargs)
    assert kind in {f.kind for f in p.findings}, p.render()
    assert p.valid is not voids


def test_a_new_release_never_voids_a_receipt_even_alongside_other_noise():
    p = check(
        receipt("2026-01-01"),
        today=TODAY,
        traffic=dict(MIX),
        available=frozenset({"qwen-4", "llama-5"}),
    )
    assert p.valid, p.render()
    assert {f.kind for f in p.findings} == {Drift.AGED, Drift.NEW_CANDIDATE}


def test_urgent_findings_are_ordered_by_urgency_not_discovery():
    p = check(
        receipt("2026-01-01"),
        today=TODAY,
        fingerprints={"gpt-5": "9" * 64},
        traffic={"extract": 0.3, "classify": 0.2, "brand-new": 0.5},
        available=frozenset({"qwen-4"}),
    )
    kinds = [f.kind for f in p.urgent]
    assert kinds[0] is Drift.MODEL_CHANGED
    assert kinds[1] is Drift.TRAFFIC_UNCOVERED
    assert Drift.NEW_CANDIDATE not in kinds, "an opportunity is not urgent"
    assert "re-prove now" in p.action, "the most urgent finding picks the action"


def test_voiding_findings_render_above_the_merely_interesting():
    p = check(
        receipt(),
        today=TODAY,
        fingerprints={"gpt-5": "9" * 64},
        traffic=dict(MIX),
        available=frozenset({"qwen-4"}),
    )
    text = p.render()
    assert text.index("[void]") < text.index("[note]")
    assert text.startswith("RECEIPT VOID")


# --- model drift ---------------------------------------------------------------


def test_a_model_we_did_not_look_at_is_not_reported_as_changed():
    # "We did not check" and "it changed" are different claims, and conflating
    # them would make every partial check look like an incident.
    assert check(receipt(), today=TODAY, fingerprints={}, traffic=dict(MIX)).valid
    assert check(receipt(), today=TODAY, traffic=dict(MIX)).valid


def test_an_unchanged_fingerprint_is_not_a_finding():
    p = check(receipt(), today=TODAY, fingerprints=dict(PRINTS), traffic=dict(MIX))
    assert not p.findings and p.valid


def test_a_receipt_with_no_fingerprints_cannot_detect_model_drift():
    # Stated rather than silently passing: a receipt issued without fingerprints
    # simply has no basis to notice this, and the guard must not pretend it does.
    r = receipt(fingerprints={})
    p = check(r, today=TODAY, fingerprints={"gpt-5": "9" * 64}, traffic=dict(MIX))
    assert Drift.MODEL_CHANGED not in {f.kind for f in p.findings}


# --- traffic drift: the one nobody else can see --------------------------------


def test_uncovered_traffic_names_the_shapes_the_eval_set_never_scored():
    p = check(
        receipt(),
        today=TODAY,
        traffic={"extract": 0.4, "classify": 0.2, "agent-tool-loop": 0.4},
    )
    f = next(x for x in p.findings if x.kind is Drift.TRAFFIC_UNCOVERED)
    assert "agent-tool-loop" in f.detail
    assert "40%" in f.detail
    assert "re-sample" in p.action


def test_uncovered_is_more_urgent_than_merely_moved():
    # Entirely unmeasured traffic is worse than measured traffic in changed
    # proportions, and the action line must reflect which one to fix.
    p = check(receipt(), today=TODAY, traffic={"extract": 0.2, "classify": 0.3, "new": 0.5})
    assert {Drift.TRAFFIC_UNCOVERED, Drift.TRAFFIC_MOVED} <= {f.kind for f in p.findings}
    assert p.urgent[0].kind is Drift.TRAFFIC_UNCOVERED


def test_traffic_within_the_limits_is_not_a_finding():
    # A workload that wobbles slightly must not page anyone.
    p = check(receipt(), today=TODAY, traffic={"extract": 0.55, "classify": 0.45})
    assert not p.findings, p.render()


def test_absent_traffic_data_skips_the_check_rather_than_assuming_no_drift():
    p = check(receipt(), today=TODAY, traffic=None)
    assert not any("traffic" in f.kind for f in p.findings)
    p2 = check(receipt(), today=TODAY, traffic={})
    assert not any("traffic" in f.kind for f in p2.findings)


def test_the_drift_limit_is_the_actual_boundary():
    # Just inside and just outside, so the constant is load-bearing rather than
    # decorative.
    # For a two-cluster shift the total-variation distance equals the shift:
    # 0.5 * (|d| + |d|) = d. Halving it here was wrong and this test caught it.
    inside = TRAFFIC_DRIFT_LIMIT - 0.02
    outside = TRAFFIC_DRIFT_LIMIT + 0.05
    ok = check(receipt(), today=TODAY, traffic={"extract": 0.6 - inside, "classify": 0.4 + inside})
    bad = check(
        receipt(), today=TODAY, traffic={"extract": 0.6 - outside, "classify": 0.4 + outside}
    )
    assert Drift.TRAFFIC_MOVED not in {f.kind for f in ok.findings}
    assert Drift.TRAFFIC_MOVED in {f.kind for f in bad.findings}


def test_the_uncovered_limit_is_the_actual_boundary():
    under = UNCOVERED_SHARE_LIMIT - 0.02
    over = UNCOVERED_SHARE_LIMIT + 0.05
    kinds = lambda s: {  # noqa: E731
        f.kind
        for f in check(
            receipt(),
            today=TODAY,
            traffic={"extract": 0.6 * (1 - s), "classify": 0.4 * (1 - s), "new": s},
        ).findings
    }
    assert Drift.TRAFFIC_UNCOVERED not in kinds(under)
    assert Drift.TRAFFIC_UNCOVERED in kinds(over)


# --- distance ------------------------------------------------------------------


def test_distance_reads_as_the_fraction_of_volume_that_moved():
    assert traffic_distance({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0
    assert traffic_distance({"a": 1.0}, {"b": 1.0}) == 1.0
    # A tenth of volume shifting from a to b reads as 0.10.
    assert traffic_distance({"a": 0.5, "b": 0.5}, {"a": 0.4, "b": 0.6}) == pytest.approx(0.1)


def test_distance_normalises_so_unequal_totals_are_still_comparable():
    # Raw counts vs shares must not register as drift; that would be a sampling
    # artefact reported as a workload change.
    assert traffic_distance({"a": 600, "b": 400}, {"a": 0.6, "b": 0.4}) == pytest.approx(0.0)


def test_distance_with_an_empty_side_is_zero_not_one():
    # No data is not total drift. Returning 1.0 would fire the alarm on a first
    # run, before any traffic had been observed at all.
    assert traffic_distance({}, {"a": 1.0}) == 0.0
    assert traffic_distance({"a": 1.0}, {}) == 0.0


# --- age -----------------------------------------------------------------------


def test_age_is_measured_from_the_receipt_not_from_the_clock():
    r = receipt("2026-01-01")
    assert not check(r, today=date(2026, 1, 2), traffic=dict(MIX)).findings
    aged = check(r, today=date(2026, 12, 1), traffic=dict(MIX))
    assert aged.findings[0].kind is Drift.AGED


def test_the_age_limit_is_the_actual_boundary():
    from datetime import timedelta

    issued = date(2026, 1, 1)
    r = receipt(issued.isoformat())
    at = lambda d: {  # noqa: E731
        f.kind for f in check(r, today=issued + timedelta(days=d), traffic=dict(MIX)).findings
    }
    assert Drift.AGED not in at(MAX_RECEIPT_AGE_DAYS)
    assert Drift.AGED in at(MAX_RECEIPT_AGE_DAYS + 1)


def test_an_unparseable_date_produces_no_finding_rather_than_a_guess():
    for bad in ("whenever", "", "2026-13-45", "27/07/2026"):
        p = check(receipt(bad), today=TODAY, traffic=dict(MIX))
        assert Drift.AGED not in {f.kind for f in p.findings}, bad


# --- the guard never acts ------------------------------------------------------


def test_the_guard_has_no_way_to_change_anything():
    # It runs unattended on a timer. The strongest verb it owns is "re-prove".
    p = check(receipt(), today=TODAY, fingerprints={"gpt-5": "9" * 64})
    assert not hasattr(p, "apply")
    assert all(v.startswith(("re-prove", "re-sample", "evaluate", "nothing")) for v in [p.action])


def test_every_proposal_carries_the_receipt_it_judged():
    r = receipt()
    assert check(r, today=TODAY).receipt_digest == r.digest()


def test_a_clean_check_says_so_plainly():
    p = check(receipt(), today=TODAY, fingerprints=dict(PRINTS), traffic=dict(MIX))
    assert p.valid and not p.findings
    assert p.action.startswith("nothing")
    assert "still holds" in p.render()


# --- the CLI contract ----------------------------------------------------------
# The guard's whole delivery mechanism is a cron job or a CI step, so the exit
# code is the interface. A wrong one means a silent watchdog.


def _write(tmp_path, name, obj):
    import json

    p = tmp_path / name
    p.write_text(obj if isinstance(obj, str) else json.dumps(obj))
    return str(p)


def test_the_cli_exits_zero_while_the_receipt_holds(tmp_path, capsys):
    from clickllm.cli import main

    rp = _write(tmp_path, "r.json", receipt().to_json())
    fp = _write(tmp_path, "fp.json", PRINTS)
    assert main(["guard", rp, "--fingerprints", fp, "--today", "2026-07-27"]) == 0
    assert "still holds" in capsys.readouterr().out


def test_the_cli_exits_nonzero_when_the_receipt_is_void(tmp_path, capsys):
    from clickllm.cli import main

    rp = _write(tmp_path, "r.json", receipt().to_json())
    fp = _write(tmp_path, "fp.json", {"gpt-5": "9" * 64})
    assert main(["guard", rp, "--fingerprints", fp, "--today", "2026-07-27"]) == 1
    assert "RECEIPT VOID" in capsys.readouterr().out


def test_a_new_release_alone_does_not_fail_the_cron_job(tmp_path):
    # Waking someone at 3am for a Hugging Face release is how a watchdog gets
    # muted, and a muted watchdog does not report the model swap either.
    from clickllm.cli import main

    rp = _write(tmp_path, "r.json", receipt().to_json())
    assert main(["guard", rp, "--available", "qwen-4", "--today", "2026-07-27"]) == 0


def test_an_altered_receipt_is_a_sentence_not_a_traceback(tmp_path, capsys):
    import json

    from clickllm.cli import main

    blob = json.loads(receipt().to_json())
    blob["receipt"]["proven"][0]["passed"] = 999
    rp = _write(tmp_path, "bad.json", json.dumps(blob))
    assert main(["receipt", rp]) == 2
    assert "altered" in capsys.readouterr().err


def test_a_missing_file_is_a_sentence_not_a_traceback(tmp_path, capsys):
    from clickllm.cli import main

    assert main(["guard", str(tmp_path / "nope.json")]) == 2
    assert "error:" in capsys.readouterr().err


# --- what a real receipt has to survive ------------------------------------------
# Everything above builds its receipt in-process. These two run `clickllm prove`
# for real and then guard the file it wrote, because the seam that breaks is the
# JSON round trip: a field the guard needs is only carried if it is serialised.


def _proved(tmp_path) -> Path:
    """A receipt written by `clickllm prove`, over two clusters of real traffic."""
    import json

    from clickllm.cli import main

    rows = [
        {
            "item_id": f"c{i}",
            "cluster": "codegen",
            "prompt": f"p{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"a": 1}',
        }
        for i in range(45)
    ] + [
        {
            "item_id": f"r{i}",
            "cluster": "rare-json",
            "prompt": f"q{i}",
            "baseline": '{"a": 1}',
            "candidate": '{"b": 1}',
        }
        for i in range(15)
    ]
    evalset = _write(
        tmp_path,
        "e.json",
        json.dumps({"items": rows, "shares": {"codegen": 0.75, "rare-json": 0.25}}),
    )
    out = tmp_path / "receipt.json"
    main(
        [
            "prove",
            evalset,
            "--candidate",
            "glm-5.2",
            "--incumbent",
            "gpt-5",
            "--issued",
            FRESH,
            "--out",
            str(out),
        ]
    )
    return out


def test_a_receipt_from_prove_carries_what_a_later_check_needs(tmp_path):
    """The doubt-everything list: every number, its interval, its denominator,
    the bar, the questions asked, and who judged. All of it survives the file."""
    import json

    body = json.loads(_proved(tmp_path).read_text())["receipt"]
    assert set(body) >= {
        "bar",
        "candidate",
        "eval_set",
        "fingerprints",
        "incumbent",
        "issued",
        "judge_agreement",
        "judge_model",
        "judge_trustworthy",
        "proven",
        "regret",
        "traffic_captures",
        "unproven",
    }, sorted(body)
    assert len(body["eval_set"]) == 64, "the questions are identified by digest"
    assert body["bar"] == 0.90
    assert body["traffic_captures"] == 60, "asked, not what survived"
    assert body["judge_model"] is None, "no judge ran; the receipt must not imply one"

    claim = body["proven"][0]
    assert set(claim) >= {"cluster", "high", "low", "name", "passed", "share", "total", "ungraded"}
    assert claim["total"] == 45 and claim["low"] < claim["high"]

    # And the deliberate absence: no field can hold a bare verdict, because a
    # receipt whose unknowns can be skipped is one whose unknowns get skipped.
    assert not {"verdict", "status", "result", "passed_overall"} & set(body), sorted(body)


def test_the_guard_can_void_a_receipt_prove_wrote(tmp_path):
    from clickllm.prove.receipt import Receipt

    r = Receipt.from_json(_proved(tmp_path).read_text())

    # The mix it was issued against still holds.
    assert check(
        r, today=date.fromisoformat(FRESH), traffic={"codegen": 0.75, "rare-json": 0.25}
    ).valid

    # The workload moved between the two shapes it *did* score.
    moved = check(r, today=date.fromisoformat(FRESH), traffic={"codegen": 0.1, "rare-json": 0.9})
    assert not moved.valid and moved.urgent[0].kind is Drift.TRAFFIC_MOVED

    # ...and a shape it never scored at all, which needs the per-cluster keys to
    # have survived serialisation.
    uncovered = check(
        r,
        today=date.fromisoformat(FRESH),
        traffic={"codegen": 0.55, "rare-json": 0.2, "agent-loop": 0.25},
    )
    assert uncovered.urgent[0].kind is Drift.TRAFFIC_UNCOVERED
    assert "agent-loop" in uncovered.render()


def test_verifying_a_receipt_against_itself_succeeds(tmp_path, capsys):
    from clickllm.cli import main

    rp = _write(tmp_path, "r.json", receipt().to_json())
    assert main(["receipt", rp, "--against", rp]) == 0
    assert "verified" in capsys.readouterr().out


def test_verifying_against_a_different_run_explains_itself(tmp_path, capsys):
    from clickllm.cli import main

    a = _write(tmp_path, "a.json", receipt().to_json())
    b = _write(tmp_path, "b.json", receipt(fingerprints={"gpt-5": "9" * 64}).to_json())
    assert main(["receipt", a, "--against", b]) == 1
    out = capsys.readouterr().out
    assert "DOES NOT VERIFY" in out and "changed behind its name" in out
