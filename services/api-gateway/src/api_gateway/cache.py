"""Reuse retrieval results for repeated questions.

Support traffic repeats itself: a few questions make up most of the volume.
Their evidence is cached in Redis for a short while, keyed by the exact query
and top_k, so the embedding call and the vector search are skipped. Ingesting a
document clears the cache, and the TTL covers whatever lands between.
"""

from __future__ import annotations

from arq import ArqRedis

from inagecas_shared.ids import content_hash
from inagecas_shared.schemas import RetrieveRequest, RetrieveResponse

_PREFIX = "inagecas:retrieval:"


class RetrievalCache:
    def __init__(self, redis: ArqRedis, ttl_seconds: int) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    @staticmethod
    def _key(request: RetrieveRequest) -> str:
        return _PREFIX + content_hash(f"{request.top_k}\n{request.query}")

    async def get(self, request: RetrieveRequest) -> RetrieveResponse | None:
        if self._ttl <= 0 or request.filters:
            return None
        raw = await self._redis.get(self._key(request))
        return RetrieveResponse.model_validate_json(raw) if raw else None

    async def put(self, request: RetrieveRequest, response: RetrieveResponse) -> None:
        if self._ttl <= 0 or request.filters:
            return
        await self._redis.set(self._key(request), response.model_dump_json(), ex=self._ttl)

    async def clear(self) -> None:
        keys = [key async for key in self._redis.scan_iter(match=_PREFIX + "*")]
        if keys:
            await self._redis.delete(*keys)
