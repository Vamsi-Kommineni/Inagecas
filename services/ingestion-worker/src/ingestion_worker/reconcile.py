"""Delete vectors whose document no longer exists.

Usage:
    python -m ingestion_worker.reconcile [--dry-run]

Postgres owns the document records and Qdrant owns their vectors, so an ingest
that dies between the two, or a document removed out of band, can leave vectors
behind. They still carry ``status: active``, so retrieval returns them: they
produce citations to documents nobody can open, and because confidence counts
distinct documents, duplicates of one page read as several independent sources.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_session_factory
from inagecas_shared.logging import configure_logging, get_logger
from inagecas_shared.models import Document
from inagecas_shared.vectorstore import VectorStore

log = get_logger("reconcile")


async def find_orphans(session: AsyncSession, store: VectorStore) -> list[uuid.UUID]:
    """Indexed document ids that Postgres no longer knows about."""
    indexed = await store.document_ids()
    known = set((await session.execute(select(Document.id))).scalars())
    return sorted(indexed - known, key=str)


async def reconcile(*, dry_run: bool = False) -> list[uuid.UUID]:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    store = VectorStore(settings)
    try:
        async with get_session_factory()() as session:
            orphans = await find_orphans(session, store)
            for document_id in orphans:
                log.info("orphaned_vectors", document_id=str(document_id), deleted=not dry_run)
                if not dry_run:
                    await store.delete_document(document_id)
        log.info("reconciled", orphans=len(orphans), dry_run=dry_run)
        return orphans
    finally:
        await store.aclose()
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove vectors with no document record.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be deleted and stop"
    )
    args = parser.parse_args()
    asyncio.run(reconcile(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
