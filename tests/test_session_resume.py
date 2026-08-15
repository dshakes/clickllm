"""What must survive `to_json` / `from_json`.

A resumed session is the same session or it is a different one that looks alike.
"""

from __future__ import annotations

import json
from dataclasses import replace

from onpar.plan import LoraFleet
from onpar.session import Session

_PROSE = "serve 8 concurrent users at 32k context for a nightly batch job"


def _resumed(s: Session) -> Session:
    return Session.from_json(s.to_json())


def test_the_evidence_trail_survives_a_resume():
    """`evidence` is "what was read out of your sentence, quoted back with the
    words that produced it" — one of this module's three stated design rules.

    A resumed session had none of it: `to_json` never wrote it and `from_json`
    never re-derived it, even though the prose that produced it *is* restored.
    The trail was recoverable and simply was not recovered.
    """
    s = Session()
    s._apply_text(_PROSE)
    assert s.evidence, "fixture must produce evidence"

    assert _resumed(s).evidence == s.evidence


def test_the_evidence_is_recomputed_rather_than_stored():
    """Re-reading beats storing for the same reason the receipt derives its
    intervals: a serialised copy can drift from the text it claims to explain,
    and then the quotes cite a sentence nobody wrote."""
    s = Session()
    s._apply_text(_PROSE)
    assert "evidence" not in json.loads(s.to_json())


def test_a_session_with_no_prose_has_nothing_to_recompute():
    s = Session()
    s._apply_fields(concurrency=4)
    assert _resumed(s).evidence == ()


def test_an_explicitly_set_field_is_not_re_read_out_of_the_prose():
    """The control that makes recomputation safe. Only `evidence` is recomputed;
    `requirements` are restored as saved, because a field the user answered
    directly must not be talked out of it by a later re-reading — the same rule
    `_apply_text` enforces with `stated`."""
    s = Session()
    s._apply_text("8 concurrent users")
    s._apply_fields(concurrency=32)

    back = _resumed(s)
    assert back.requirements.concurrency == 32
    assert back.evidence, "and the trail is still there"


def test_a_configured_lora_fleet_survives_a_resume():
    """Already fixed, kept as a guard: `lora` is excluded from the
    `requirements` dict and carried in its own top-level block, so the exclusion
    reads like a drop unless something asserts the round trip."""
    fleet = LoraFleet(adapters=(("a", "repo/a"), ("b", "repo/b")), max_rank=32, max_concurrent=4)
    s = Session()
    s.requirements = replace(s.requirements, lora=fleet)

    assert _resumed(s).requirements.lora == fleet


def test_a_session_without_a_fleet_resumes_without_one():
    assert _resumed(Session()).requirements.lora is None


def test_everything_else_round_trips_too():
    """The broad guard: a field added to `Requirements` and forgotten by
    `to_json` is exactly the shape of both findings here."""
    s = Session()
    s._apply_text(_PROSE)
    s._apply_fields(concurrency=12, context=8192)

    back = _resumed(s)
    assert back.requirements == s.requirements
    assert back.text == s.text
    assert back.stage is s.stage
    assert back.stated == s.stated
