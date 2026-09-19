from __future__ import annotations

from answer_service.validation import Verdict, parse_verdict
from inagecas_shared.schemas import Intent
from inagecas_shared.structured import parse_field, parse_json
from policy_service.classifier import _match
from vision_service.vlm import parse_reading


def test_json_reply_yields_the_field():
    assert parse_field('{"verdict": "PARTIAL"}', "verdict") == "PARTIAL"
    assert parse_field('```json\n{"category": "how_to"}\n```', "category") == "how_to"


def test_prose_reply_is_returned_whole_for_keyword_matching():
    assert parse_field("I think it is SUPPORTED.", "verdict") == "I think it is SUPPORTED."
    assert parse_json("[1, 2]") is None


def test_missing_field_is_empty_not_the_whole_reply():
    assert parse_field('{"other": 1}', "verdict") == ""


def test_verdict_and_intent_parse_both_shapes():
    assert parse_verdict('{"verdict": "CONTRADICTED"}') == Verdict.contradicted
    assert parse_verdict("UNSUPPORTED") == Verdict.unsupported
    assert _match('{"category": "off_topic"}') == Intent.off_topic
    assert _match("troubleshooting") == Intent.troubleshooting
    assert _match('{"category": "weather"}') == Intent.unknown


def test_reading_still_parses_a_fenced_reply():
    reading = parse_reading('```json\n{"readable": true, "text": ["a"]}\n```')
    assert reading is not None and reading.readable
