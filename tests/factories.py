"""Small helpers for building test objects."""

from __future__ import annotations

import uuid

from inagecas_shared.schemas import Evidence


def make_evidence(
    *,
    score: float = 0.8,
    text: str = "Paris is the capital of France.",
    ordinal: int = 0,
    title: str | None = "Doc",
) -> Evidence:
    return Evidence(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source_uri="doc.md",
        title=title,
        ordinal=ordinal,
        score=score,
        text=text,
    )


class FakeGateway:
    """Stand-in for ModelGateway that returns canned chat replies."""

    def __init__(self, reply: str = "", replies: list[str] | None = None) -> None:
        self._reply = reply
        self._replies = list(replies) if replies else None
        self.chat_calls: list[list[dict]] = []
        self.models_used: list[str | None] = []

    async def chat(
        self, messages, *, temperature=None, max_tokens=None, model=None, json_schema=None
    ) -> str:
        self.chat_calls.append(messages)
        self.models_used.append(model)
        if self._replies is not None:
            return self._replies.pop(0)
        return self._reply

    async def describe_image(self, instructions, image_base64, *, max_tokens=None, model=None):
        self.models_used.append(model)
        if self._replies is not None:
            return self._replies.pop(0)
        return self._reply

    async def embed_one(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    async def aclose(self) -> None:
        return None
