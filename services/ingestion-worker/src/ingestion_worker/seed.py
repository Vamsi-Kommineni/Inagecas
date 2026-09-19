"""Bulk-ingest a local directory of documents.

Usage:
    python -m ingestion_worker.seed path/to/docs

Runs the ingestion pipeline directly (no queue) so you get immediate feedback,
which is handy for seeding a corpus or loading the bundled sample docs.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_session_factory
from inagecas_shared.ids import new_id
from inagecas_shared.logging import configure_logging, get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import IngestionRun
from inagecas_shared.sparse import SparseEmbedder
from inagecas_shared.vectorstore import VectorStore

from .pipeline import run_ingestion

log = get_logger("seed")

_EXTENSIONS = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
    ".txt": "text",
}


def _iter_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTENSIONS and p.is_file())


async def seed_directory(directory: Path) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=False)
    gateway = ModelGateway(settings)
    store = VectorStore(settings)
    sparse_embedder = SparseEmbedder(settings.sparse_model)
    try:
        async with get_session_factory()() as session:
            for path in _iter_files(directory):
                source_type = _EXTENSIONS[path.suffix.lower()]
                content: str | bytes = (
                    path.read_bytes() if source_type == "pdf" else path.read_text("utf-8", "ignore")
                )
                run = IngestionRun(
                    id=new_id(),
                    status="queued",
                    source_type=source_type,
                    source_uri=str(path),
                )
                session.add(run)
                await session.commit()

                log.info("seeding", path=str(path), source_type=source_type)
                await run_ingestion(
                    session,
                    gateway,
                    store,
                    sparse_embedder,
                    ingestion_run_id=run.id,
                    title=path.stem,
                    source_type=source_type,
                    source_uri=str(path),
                    content=content,
                )
    finally:
        await gateway.aclose()
        await store.aclose()
        await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Inagecas from a directory of documents.")
    parser.add_argument("directory", type=Path, help="Directory to ingest recursively")
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error(f"Not a directory: {args.directory}")
    asyncio.run(seed_directory(args.directory))


if __name__ == "__main__":
    main()
