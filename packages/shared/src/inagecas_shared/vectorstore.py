"""Qdrant vector store wrapper for hybrid (dense + sparse) retrieval.

Each point carries a dense vector (semantic) and an optional BM25 sparse vector
(lexical). Dense and sparse searches are exposed separately so the retrieval
service can fuse their rankings itself and keep dense cosine as the confidence
signal. The chunk id is the Qdrant point id, and the payload holds everything
retrieval needs to build evidence without a second lookup.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from .config import Settings, get_settings
from .sparse import SparseVector

DENSE = "dense"
SPARSE = "sparse"


@dataclass(slots=True)
class ChunkPoint:
    chunk_id: uuid.UUID
    dense: list[float]
    sparse: SparseVector | None
    payload: dict[str, object]


@dataclass(slots=True)
class ScoredChunk:
    chunk_id: uuid.UUID
    score: float
    payload: dict[str, object]


def _build_filter(filters: dict[str, str] | None) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(key="status", match=models.MatchValue(value="active"))
    ]
    for key, value in (filters or {}).items():
        must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    return models.Filter(must=must)


def _to_scored(point: models.ScoredPoint) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid.UUID(str(point.id)),
        score=point.score,
        payload=point.payload or {},
    )


class VectorStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.collection = self.settings.qdrant_collection
        self._client = AsyncQdrantClient(
            url=self.settings.qdrant_url,
            api_key=self.settings.qdrant_api_key or None,
        )

    async def health(self) -> None:
        await self._client.get_collections()

    async def ensure_collection(self) -> None:
        if await self._client.collection_exists(self.collection):
            return
        await self._client.create_collection(
            self.collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=self.settings.embed_dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        for field in ("document_id", "status"):
            await self._client.create_payload_index(
                self.collection, field, models.PayloadSchemaType.KEYWORD
            )

    async def upsert(self, points: list[ChunkPoint]) -> None:
        structs = []
        for point in points:
            vector: dict[str, object] = {DENSE: point.dense}
            if point.sparse is not None:
                vector[SPARSE] = models.SparseVector(
                    indices=point.sparse.indices, values=point.sparse.values
                )
            structs.append(
                models.PointStruct(id=str(point.chunk_id), vector=vector, payload=point.payload)
            )
        await self._client.upsert(self.collection, points=structs)

    async def dense_search(
        self,
        vector: list[float],
        *,
        limit: int,
        min_score: float | None = None,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        response = await self._client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            limit=limit,
            score_threshold=min_score,
            query_filter=_build_filter(filters),
            with_payload=True,
        )
        return [_to_scored(point) for point in response.points]

    async def sparse_search(
        self,
        sparse: SparseVector,
        *,
        limit: int,
        filters: dict[str, str] | None = None,
    ) -> list[ScoredChunk]:
        response = await self._client.query_points(
            self.collection,
            query=models.SparseVector(indices=sparse.indices, values=sparse.values),
            using=SPARSE,
            limit=limit,
            query_filter=_build_filter(filters),
            with_payload=True,
        )
        return [_to_scored(point) for point in response.points]

    async def document_ids(self) -> set[uuid.UUID]:
        """Every document id that currently has vectors in the collection."""
        found: set[uuid.UUID] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = await self._client.scroll(
                self.collection,
                limit=256,
                offset=offset,
                with_payload=["document_id"],
                with_vectors=False,
            )
            for point in points:
                document_id = (point.payload or {}).get("document_id")
                if document_id:
                    found.add(uuid.UUID(str(document_id)))
            if offset is None:
                return found

    async def delete_document(self, document_id: uuid.UUID) -> None:
        await self._client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        )
                    ]
                )
            ),
        )

    async def aclose(self) -> None:
        await self._client.close()
