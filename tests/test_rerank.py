from __future__ import annotations

import uuid

from inagecas_shared.config import get_settings
from inagecas_shared.vectorstore import ScoredChunk
from retrieval_service.rerank import Reranker


def _chunk(text: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk_id=uuid.uuid4(), score=score, payload={"text": text})


class _ScoreGateway:
    """Scores the chunk that mentions 'relevant' highest, regardless of order."""

    async def chat(self, messages, **kwargs) -> str:
        content = messages[-1]["content"]
        return "9" if "relevant" in content else "1"

    async def aclose(self) -> None:
        return None


async def test_none_strategy_is_passthrough():
    settings = get_settings().model_copy(update={"rerank_strategy": "none"})
    reranker = Reranker(settings, _ScoreGateway())
    chunks = [_chunk("a", 0.9), _chunk("b", 0.5), _chunk("c", 0.4)]
    assert await reranker.rerank("q", chunks, top_k=2) == chunks[:2]


async def test_llm_strategy_promotes_relevant_chunk():
    settings = get_settings().model_copy(update={"rerank_strategy": "llm"})
    reranker = Reranker(settings, _ScoreGateway())
    chunks = [_chunk("nothing here", 0.9), _chunk("the relevant passage", 0.1)]
    ranked = await reranker.rerank("q", chunks, top_k=2)
    assert ranked[0].payload["text"] == "the relevant passage"
