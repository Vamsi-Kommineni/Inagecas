from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from inagecas_shared.config import get_settings
from inagecas_shared.schemas import RetrieveRequest
from inagecas_shared.sparse import SparseVector
from inagecas_shared.vectorstore import ScoredChunk
from retrieval_service.rerank import Reranker
from retrieval_service.retriever import retrieve


def _payload(document_id: uuid.UUID, text: str) -> dict:
    return {
        "document_id": str(document_id),
        "text": text,
        "ordinal": 0,
        "source_uri": "doc.md",
        "title": "Doc",
        "status": "active",
    }


class FakeStore:
    def __init__(self, dense: list[ScoredChunk], sparse: list[ScoredChunk]) -> None:
        self._dense = dense
        self._sparse = sparse

    async def dense_search(self, vector, *, limit, min_score=None, filters=None):
        return self._dense

    async def sparse_search(self, sparse, *, limit, filters=None):
        return self._sparse


class FakeGateway:
    async def embed_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeSparse:
    def embed_query(self, text: str) -> SparseVector:
        return SparseVector([1, 2], [0.5, 0.5])


async def test_hybrid_retrieve_fuses_and_reports_source_diversity():
    settings = get_settings().model_copy(
        update={"rerank_strategy": "none", "hybrid_enabled": True, "retrieval_min_score": 0.0}
    )
    doc_a, doc_b = uuid.uuid4(), uuid.uuid4()
    chunk_a, chunk_b = uuid.uuid4(), uuid.uuid4()
    dense = [
        ScoredChunk(chunk_a, 0.8, _payload(doc_a, "alpha")),
        ScoredChunk(chunk_b, 0.4, _payload(doc_b, "beta")),
    ]
    sparse = [ScoredChunk(chunk_b, 3.0, _payload(doc_b, "beta"))]

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    reranker = Reranker(settings, gateway=None)

    response = await retrieve(
        session,
        FakeGateway(),
        FakeStore(dense, sparse),
        FakeSparse(),
        reranker,
        settings,
        RetrieveRequest(query="q"),
    )

    assert response.source_count == 2
    assert {e.document_id for e in response.evidence} == {doc_a, doc_b}
    scores = {e.chunk_id: e.score for e in response.evidence}
    assert scores[chunk_a] == 0.8  # dense cosine is preserved as the score
    session.commit.assert_awaited_once()


async def test_retrieve_filters_below_min_score():
    settings = get_settings().model_copy(
        update={"rerank_strategy": "none", "hybrid_enabled": False, "retrieval_min_score": 0.5}
    )
    doc = uuid.uuid4()
    weak = ScoredChunk(uuid.uuid4(), 0.2, _payload(doc, "weak match"))

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    reranker = Reranker(settings, gateway=None)

    response = await retrieve(
        session,
        FakeGateway(),
        FakeStore([weak], []),
        FakeSparse(),
        reranker,
        settings,
        RetrieveRequest(query="q"),
    )

    assert response.evidence == []
    assert response.top_score is None


class _UnenthusiasticEncoder:
    """A cross-encoder that scores everything low, as real ones often do."""

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        return [-3.0] * len(texts)


async def test_cross_encoder_rerank_keeps_evidence_above_the_floor():
    """Regression: the cross-encoder score used to replace the dense cosine, so a
    correct passage it happened to score low was cut by retrieval_min_score."""
    settings = get_settings().model_copy(
        update={
            "rerank_strategy": "cross_encoder",
            "hybrid_enabled": False,
            "retrieval_min_score": 0.45,
        }
    )
    doc, chunk = uuid.uuid4(), uuid.uuid4()
    dense = [ScoredChunk(chunk, 0.64, _payload(doc, "upgrade or downgrade from Settings"))]

    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    reranker = Reranker(settings, gateway=None)
    reranker._encoder = _UnenthusiasticEncoder()

    response = await retrieve(
        session,
        FakeGateway(),
        FakeStore(dense, []),
        FakeSparse(),
        reranker,
        settings,
        RetrieveRequest(query="q"),
    )

    assert [e.chunk_id for e in response.evidence] == [chunk]
    assert response.top_score == 0.64
