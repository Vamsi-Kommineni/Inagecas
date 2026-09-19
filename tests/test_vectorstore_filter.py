from __future__ import annotations

from inagecas_shared.vectorstore import _build_filter


def test_filter_always_requires_active_status():
    result = _build_filter(None)
    status = next(condition for condition in result.must if condition.key == "status")
    assert status.match.value == "active"


def test_filter_adds_custom_conditions():
    result = _build_filter({"document_id": "abc"})
    keys = {condition.key for condition in result.must}
    assert keys == {"status", "document_id"}
