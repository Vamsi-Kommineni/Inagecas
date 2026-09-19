"""The ingestion pipeline: parse, chunk, embed, and index one document."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import suppress
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.ids import content_hash, new_id
from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import Chunk, Document, DocumentVersion, IngestionRun
from inagecas_shared.sparse import SparseEmbedder
from inagecas_shared.text import with_context
from inagecas_shared.vectorstore import ChunkPoint, VectorStore

from .chunking import chunk_document
from .parsers import parse

log = get_logger("ingestion-worker")

_EMBED_BATCH = 64


def _utcnow() -> datetime:
    # Store naive UTC to match the timezone-naive timestamp columns.
    return datetime.now(UTC).replace(tzinfo=None)


async def _embed_all(gateway: ModelGateway, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        vectors.extend(await gateway.embed(texts[start : start + _EMBED_BATCH]))
    return vectors


async def run_ingestion(
    session: AsyncSession,
    gateway: ModelGateway,
    store: VectorStore,
    sparse_embedder: SparseEmbedder,
    *,
    ingestion_run_id: uuid.UUID,
    title: str | None,
    source_type: str,
    source_uri: str,
    content: str | bytes,
) -> None:
    run = await session.get(IngestionRun, ingestion_run_id)
    if run is None:
        log.error("ingestion_run_missing", ingestion_run_id=str(ingestion_run_id))
        return

    run.status = "running"
    await session.commit()

    cleanup_document_id: uuid.UUID | None = None
    try:
        text = parse(source_type, content)
        if not text:
            await _fail(session, run, "Document produced no text")
            return

        digest = content_hash(text)
        existing = await session.scalar(
            select(DocumentVersion).where(DocumentVersion.content_hash == digest)
        )
        if existing is not None:
            run.document_id = existing.document_id
            run.status = "succeeded"
            run.chunks_indexed = 0
            run.finished_at = _utcnow()
            await session.commit()
            log.info("ingestion_skipped_duplicate", ingestion_run_id=str(ingestion_run_id))
            return

        document = Document(
            id=new_id(),
            source_type=source_type,
            source_uri=source_uri,
            title=title,
            status="active",
        )
        version = DocumentVersion(
            id=new_id(),
            document_id=document.id,
            version=1,
            content_hash=digest,
            byte_size=len(text.encode("utf-8")),
        )
        session.add_all([document, version])
        # Flush so the document/version rows exist before their chunks reference them.
        await session.flush()
        cleanup_document_id = document.id

        chunks = chunk_document(text, source_type)
        if not chunks:
            await _fail(session, run, "Document produced no chunks")
            return

        located = [with_context(title, c.heading, c.text) for c in chunks]
        vectors = await _embed_all(gateway, located)
        sparse_vectors = await asyncio.to_thread(sparse_embedder.embed_documents, located)
        indexed_at = time.time()

        points: list[ChunkPoint] = []
        for chunk, vector, sparse in zip(chunks, vectors, sparse_vectors, strict=True):
            chunk_id = new_id()
            session.add(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    document_version_id=version.id,
                    ordinal=chunk.ordinal,
                    heading=chunk.heading,
                    text=chunk.text,
                    token_estimate=chunk.token_estimate,
                )
            )
            points.append(
                ChunkPoint(
                    chunk_id=chunk_id,
                    dense=vector,
                    sparse=sparse,
                    payload={
                        "document_id": str(document.id),
                        "document_version_id": str(version.id),
                        "ordinal": chunk.ordinal,
                        "heading": chunk.heading,
                        "text": chunk.text,
                        "source_uri": source_uri,
                        "title": title,
                        "status": "active",
                        "indexed_at": indexed_at,
                    },
                )
            )

        await store.ensure_collection()
        await store.upsert(points)

        run.document_id = document.id
        run.status = "succeeded"
        run.chunks_indexed = len(points)
        run.finished_at = _utcnow()
        await session.commit()
        log.info(
            "ingestion_succeeded",
            ingestion_run_id=str(ingestion_run_id),
            document_id=str(document.id),
            chunks=len(points),
        )
    except Exception as exc:
        await session.rollback()
        if cleanup_document_id is not None:
            # Remove any vectors upserted before the failure so none are orphaned.
            with suppress(Exception):
                await store.delete_document(cleanup_document_id)
        await _fail(session, run, str(exc))
        log.error("ingestion_failed", ingestion_run_id=str(ingestion_run_id), error=str(exc))


async def _fail(session: AsyncSession, run: IngestionRun, message: str) -> None:
    run.status = "failed"
    run.error = message[:2000]
    run.finished_at = _utcnow()
    await session.commit()
