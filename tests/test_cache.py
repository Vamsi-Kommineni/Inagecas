from __future__ import annotations

import uuid

from api_gateway.cache import RetrievalCache
from factories import make_evidence
from inagecas_shared.schemas import RetrieveRequest, RetrieveResponse


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def scan_iter(self, match):
        for key in list(self.store):
            yield key

    async def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)


def _response() -> RetrieveResponse:
    return RetrieveResponse(
        retrieval_run_id=uuid.uuid4(), query="q", evidence=[make_evidence()], top_score=0.8
    )


async def test_a_repeated_query_is_served_from_the_cache():
    cache = RetrievalCache(FakeRedis(), ttl_seconds=60)  # type: ignore[arg-type]
    request = RetrieveRequest(query="How do I upgrade?")
    assert await cache.get(request) is None
    response = _response()
    await cache.put(request, response)
    assert (await cache.get(request)) == response
    assert await cache.get(RetrieveRequest(query="How do I upgrade?", top_k=3)) is None


async def test_clear_forgets_everything_and_zero_ttl_disables():
    redis = FakeRedis()
    cache = RetrievalCache(redis, ttl_seconds=60)  # type: ignore[arg-type]
    await cache.put(RetrieveRequest(query="a"), _response())
    await cache.clear()
    assert redis.store == {}

    off = RetrievalCache(redis, ttl_seconds=0)  # type: ignore[arg-type]
    await off.put(RetrieveRequest(query="a"), _response())
    assert redis.store == {} and await off.get(RetrieveRequest(query="a")) is None


async def test_filtered_queries_bypass_the_cache():
    redis = FakeRedis()
    cache = RetrievalCache(redis, ttl_seconds=60)  # type: ignore[arg-type]
    await cache.put(RetrieveRequest(query="a", filters={"source": "x"}), _response())
    assert redis.store == {}
