"""
memory_engine/forgetting.py

Applies the cold-start decay model to archive memories that have
dropped below the survival threshold.

Called periodically (e.g. every N turns) from the API layer.
Replaced by the learned survival classifier in Phase 7.
"""

from __future__ import annotations

from .decay import should_prune
from .mongo import MemoryDB


class ForgettingService:
    def __init__(self, db: MemoryDB):
        self._db = db

    async def run(self, user_id: str) -> list[str]:
        """
        Check all active memories for a user and archive those
        that have decayed past the prune threshold.
        Returns list of archived memory ids.
        """
        records  = await self._db.get_active(user_id)
        archived = []

        for record in records:
            if should_prune(record):
                await self._db.archive(record.id)
                archived.append(record.id)

        return archived