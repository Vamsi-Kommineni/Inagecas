"""Hybrid retrieval: dense + sparse recall, fusion, rerank, evidence assembly."""

from __future__ import annotations

import asyncio
import time
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared import metrics
from inagecas_shared.config import Settings
from inagecas_shared.ids import new_id
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import RetrievalRun
from inagecas_shared.schemas import Evidence, RetrieveRequest, RetrieveResponse
from inagecas_shared.sparse import SparseEmbedder
from inagecas_shared.vectorstore import ScoredChunk, VectorStore

from .fusion import is_stale, reciprocal_rank_fusion
from .rerank import Reranker


def _to_evidence(chunk: ScoredChunk, stale: bool) -> Evidence:
    payload = chunk.payload
    title = payload.get("title")
    ordinal = payload.get("ordinal", 0)
    return Evidence(
        chunk_id=chunk.chunk_id,
        document_id=uuid.UUID(str(payload["document_id"])),
        source_uri=str(payload.get("source_uri", "")),
        title=title if isinstance(title, str) else None,
        ordinal=ordinal if isinstance(ordinal, int) else 0,
        score=chunk.score,
        text=str(payload.get("text", "")),
        stale=stale,
    )


async def _recall(
    store: VectorStore,
    sparse_embedder: SparseEmbedder,
    settings: Settings,
    request: RetrieveRequest,
    dense_vector: list[float],
    candidate_k: int,
    min_score: float,
) -> list[ScoredChunk]:
    """Dense (and optionally sparse) recall, fused into one candidate list."""
    dense_hits = await store.dense_search(dense_vector, limit=candidate_k, filters=request.filters)
    payloads = {hit.chunk_id: hit.payload for hit in dense_hits}
    dense_scores = {hit.chunk_id: hit.score for hit in dense_hits}
    rankings = [[hit.chunk_id for hit in dense_hits]]

    if settings.hybrid_enabled:
        sparse = await asyncio.to_thread(sparse_embedder.embed_query, request.query)
        if sparse is not None:
            sparse_hits = await store.sparse_search(
                sparse, limit=candidate_k, filters=request.filters
            )
            for hit in sparse_hits:
                payloads.setdefault(hit.chunk_id, hit.payload)
            rankings.append([hit.chunk_id for hit in sparse_hits])

    fused = reciprocal_rank_fusion(rankings)
    # Keep dense cosine as the score; a lexical-only hit sits at the min-score
    # floor so it can still contribute without inflating confidence.
    return [
        ScoredChunk(chunk_id=cid, score=dense_scores.get(cid, min_score), payload=payloads[cid])
        for cid in fused
    ]


async def retrieve(
    session: AsyncSession,
    gateway: ModelGateway,
    store: VectorStore,
    sparse_embedder: SparseEmbedder,
    reranker: Reranker,
    settings: Settings,
    request: RetrieveRequest,
) -> RetrieveResponse:
    top_k = request.top_k or settings.retrieval_top_k
    min_score = settings.retrieval_min_score if request.min_score is None else request.min_score
    candidate_k = max(settings.retrieval_candidate_k, top_k)

    started = time.perf_counter()
    dense_vector = await gateway.embed_one(request.query)
    candidates = await _recall(
        store, sparse_embedder, settings, request, dense_vector, candidate_k, min_score
    )
    ranked = await reranker.rerank(request.query, candidates, top_k=top_k)
    latency_ms = int((time.perf_counter() - started) * 1000)

    now = time.time()
    evidence: list[Evidence] = []
    for chunk in ranked:
        if chunk.score < min_score:
            continue
        indexed_at = chunk.payload.get("indexed_at")
        stale = is_stale(
            indexed_at if isinstance(indexed_at, int | float) else None,
            settings.staleness_days,
            now,
        )
        evidence.append(_to_evidence(chunk, stale))

    top_score = evidence[0].score if evidence else None
    source_count = len({item.document_id for item in evidence})
    metrics.record_retrieval(top_score, len(evidence))

    run = RetrievalRun(
        id=new_id(),
        query=request.query,
        top_k=top_k,
        min_score=min_score,
        result_count=len(evidence),
        top_score=top_score,
        latency_ms=latency_ms,
    )
    session.add(run)
    await session.commit()

    return RetrieveResponse(
        retrieval_run_id=run.id,
        query=request.query,
        evidence=evidence,
        top_score=top_score,
        source_count=source_count,
    )
