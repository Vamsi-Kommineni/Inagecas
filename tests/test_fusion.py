from __future__ import annotations

import uuid

from retrieval_service.fusion import is_stale, reciprocal_rank_fusion


def test_rrf_ranks_items_strong_in_both_lists_first():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    fused = reciprocal_rank_fusion([[a, b, c], [b, a, c]])
    assert set(fused[:2]) == {a, b}
    assert fused[-1] == c


def test_rrf_recovers_item_present_in_only_one_list():
    a, b = uuid.uuid4(), uuid.uuid4()
    # `a` is missed by the first list but recovered from the second.
    fused = reciprocal_rank_fusion([[b], [a, b]])
    assert set(fused) == {a, b}


def test_is_stale_disabled_or_missing_timestamp():
    assert is_stale(None, 30, 1_000_000.0) is False
    assert is_stale(500.0, 0, 1_000_000.0) is False


def test_is_stale_compares_against_threshold():
    now = 1_000_000.0
    assert is_stale(now - 40 * 86_400, 30, now) is True
    assert is_stale(now - 10 * 86_400, 30, now) is False
