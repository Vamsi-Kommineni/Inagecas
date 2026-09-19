from __future__ import annotations

from ingestion_worker.pipeline import _utcnow


def test_utcnow_is_naive_utc():
    # The timestamp columns are TIMESTAMP WITHOUT TIME ZONE, so writes must be
    # naive or asyncpg rejects them.
    now = _utcnow()
    assert now.tzinfo is None
