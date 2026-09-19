"""Reranking of retrieved chunks.

Three strategies behind one interface:
- ``none``          : keep the fused order.
- ``llm``           : pointwise relevance scoring by the judge model.
- ``cross_encoder`` : a local ONNX cross-encoder (fastembed) scores each
                      query/passage pair.

Every strategy only ever *reorders* candidates -- the dense cosine a chunk
arrived with stays its score. Reranker outputs are good at ranking but are not
comparable across queries, so they must not become the number the retrieval
floor and the confidence composition are measured against.

Model-load failures degrade to the fused order instead of failing the request.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from inagecas_shared.config import Settings
from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.text import with_context
from inagecas_shared.vectorstore import ScoredChunk

log = get_logger("rerank")

_INT = re.compile(r"-?\d+")

_RERANK_SYSTEM = (
    "You score how well a passage answers a question. "
    "Reply with a single integer from 0 (irrelevant) to 10 (fully answers it). "
    "Output only the number."
)


def _passage(chunk: ScoredChunk) -> str:
    """What a reranker scores: the passage under its title and heading, the same
    text the embedder saw. "Changing your plan" is in the heading, not the body,
    so a bare passage loses to a worse one that happens to say "plan".
    """
    title, heading = chunk.payload.get("title"), chunk.payload.get("heading")
    return with_context(
        title if isinstance(title, str) else None,
        heading if isinstance(heading, str) else None,
        str(chunk.payload.get("text", "")),
    )


def _first_int(text: str, default: int = 0) -> int:
    match = _INT.search(text)
    return int(match.group()) if match else default


class Reranker:
    def __init__(self, settings: Settings, gateway: ModelGateway) -> None:
        self.strategy = settings.rerank_strategy
        self.gateway = gateway
        self._cross_encoder_model = settings.cross_encoder_model
        self._judge_model = settings.judge_model
        self._encoder: Any | None = None
        self._encoder_failed = False

    async def rerank(
        self, query: str, chunks: list[ScoredChunk], *, top_k: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        if self.strategy == "cross_encoder":
            return await self._cross_encoder_rerank(query, chunks, top_k)
        if self.strategy == "llm":
            return await self._llm_rerank(query, chunks, top_k)
        return chunks[:top_k]

    # --- cross-encoder -----------------------------------------------------

    def _load_encoder(self) -> Any | None:
        if self._encoder is None and not self._encoder_failed:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._encoder = TextCrossEncoder(self._cross_encoder_model)
            except Exception as exc:
                self._encoder_failed = True
                log.warning("cross_encoder_load_failed", error=str(exc))
        return self._encoder

    async def _cross_encoder_rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        encoder = self._load_encoder()
        if encoder is None:
            return chunks[:top_k]
        texts = [_passage(c) for c in chunks]
        scores = await asyncio.to_thread(lambda: list(encoder.rerank(query, texts)))
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]

    # --- llm ---------------------------------------------------------------

    async def _llm_rerank(
        self, query: str, chunks: list[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        # Bound the number of model calls; the pool is already ranked by fusion.
        pool = chunks[: top_k * 3]
        scores = await asyncio.gather(*(self._llm_score(query, c) for c in pool))
        order = sorted(range(len(pool)), key=lambda i: scores[i], reverse=True)
        return [pool[i] for i in order[:top_k]]

    async def _llm_score(self, query: str, chunk: ScoredChunk) -> int:
        passage = _passage(chunk)[:2000]
        output = await self.gateway.chat(
            [
                {"role": "system", "content": _RERANK_SYSTEM},
                {"role": "user", "content": f"Question: {query}\n\nPassage:\n{passage}\n\nScore:"},
            ],
            temperature=0.0,
            max_tokens=4,
            model=self._judge_model,
        )
        return _first_int(output)
