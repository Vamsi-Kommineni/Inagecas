"""Local BM25 sparse embeddings for hybrid retrieval.

fastembed's BM25 runs on CPU with no neural model. The import is lazy so
services that never retrieve don't pull the dependency, and a load failure
degrades to dense-only search instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .logging import get_logger

log = get_logger("sparse")


@dataclass(slots=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._failed = False

    def _load(self) -> Any | None:
        if self._model is None and not self._failed:
            try:
                from fastembed import SparseTextEmbedding

                self._model = SparseTextEmbedding(self._model_name)
            except Exception as exc:
                self._failed = True
                log.warning("sparse_model_load_failed", error=str(exc))
        return self._model

    def embed_documents(self, texts: list[str]) -> list[SparseVector | None]:
        model = self._load()
        if model is None:
            return [None] * len(texts)
        return [_to_sparse(embedding) for embedding in model.embed(texts)]

    def embed_query(self, text: str) -> SparseVector | None:
        model = self._load()
        if model is None:
            return None
        return _to_sparse(next(iter(model.query_embed([text]))))


def _to_sparse(embedding: Any) -> SparseVector:
    return SparseVector(
        indices=[int(i) for i in embedding.indices],
        values=[float(v) for v in embedding.values],
    )
