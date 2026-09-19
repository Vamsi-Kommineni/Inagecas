from __future__ import annotations

import uuid

from inagecas_shared.config import get_settings
from inagecas_shared.vectorstore import ScoredChunk
from retrieval_service.rerank import Reranker


class FakeEncoder:
    """Scores passages containing 'match' highly, others low."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [5.0 if "match" in text else -5.0 for text in texts]


async def test_cross_encoder_reorders_without_touching_scores():
    settings = get_settings().model_copy(update={"rerank_strategy": "cross_encoder"})
    reranker = Reranker(settings, gateway=None)  # gateway unused for this strategy
    reranker._encoder = FakeEncoder()  # inject to avoid downloading a model

    chunks = [
        ScoredChunk(uuid.uuid4(), 0.9, {"text": "nothing useful here"}),
        ScoredChunk(uuid.uuid4(), 0.1, {"text": "the match passage"}),
    ]
    ranked = await reranker.rerank("q", chunks, top_k=2)

    assert ranked[0].payload["text"] == "the match passage"
    # The cross-encoder decides the order, but the dense cosine stays the score:
    # the retrieval floor and the confidence composition are calibrated to it.
    assert [c.score for c in ranked] == [0.1, 0.9]


async def test_cross_encoder_reads_the_heading_with_the_passage():
    """ "How do I change my plan?" matches the heading "Changing your plan", not
    the body, which only says upgrade and downgrade."""
    settings = get_settings().model_copy(update={"rerank_strategy": "cross_encoder"})
    reranker = Reranker(settings, gateway=None)
    reranker._encoder = FakeEncoder()

    chunks = [
        ScoredChunk(uuid.uuid4(), 0.5, {"text": "Wait and retry.", "heading": "Rate limits"}),
        ScoredChunk(uuid.uuid4(), 0.5, {"text": "Upgrade or downgrade.", "heading": "A match"}),
    ]
    ranked = await reranker.rerank("q", chunks, top_k=1)

    assert ranked[0].payload["heading"] == "A match"


async def test_cross_encoder_falls_back_when_model_unavailable():
    settings = get_settings().model_copy(update={"rerank_strategy": "cross_encoder"})
    reranker = Reranker(settings, gateway=None)
    reranker._encoder_failed = True  # simulate a load failure

    chunks = [ScoredChunk(uuid.uuid4(), 0.7, {"text": "a"})]
    ranked = await reranker.rerank("q", chunks, top_k=5)

    assert [c.chunk_id for c in ranked] == [chunks[0].chunk_id]
