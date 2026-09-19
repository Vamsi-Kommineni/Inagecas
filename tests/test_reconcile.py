from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from ingestion_worker.reconcile import find_orphans


class FakeStore:
    def __init__(self, indexed: set[uuid.UUID]) -> None:
        self._indexed = indexed
        self.deleted: list[uuid.UUID] = []

    async def document_ids(self) -> set[uuid.UUID]:
        return self._indexed

    async def delete_document(self, document_id: uuid.UUID) -> None:
        self.deleted.append(document_id)


def _session(known: list[uuid.UUID]) -> MagicMock:
    result = MagicMock()
    result.scalars.return_value = known
    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    return session


async def test_find_orphans_reports_vectors_without_a_document():
    live, orphan = uuid.uuid4(), uuid.uuid4()
    orphans = await find_orphans(_session([live]), FakeStore({live, orphan}))
    assert orphans == [orphan]


async def test_find_orphans_ignores_documents_that_are_not_indexed():
    """A document with no vectors yet is mid-ingest, not an orphan."""
    live, not_yet_indexed = uuid.uuid4(), uuid.uuid4()
    orphans = await find_orphans(_session([live, not_yet_indexed]), FakeStore({live}))
    assert orphans == []
