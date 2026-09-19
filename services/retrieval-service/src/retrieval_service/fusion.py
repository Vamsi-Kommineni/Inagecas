"""Pure helpers for combining rankings and judging staleness."""

from __future__ import annotations

import uuid


def reciprocal_rank_fusion(rankings: list[list[uuid.UUID]], *, k: int = 60) -> list[uuid.UUID]:
    """Merge several ranked id lists into one using Reciprocal Rank Fusion.

    Each list contributes ``1 / (k + rank)`` to an id's score, so an id that
    ranks well in either dense or sparse search rises to the top.
    """
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)


def is_stale(indexed_at: float | None, staleness_days: int, now: float) -> bool:
    if staleness_days <= 0 or not indexed_at:
        return False
    return (now - indexed_at) > staleness_days * 86_400
