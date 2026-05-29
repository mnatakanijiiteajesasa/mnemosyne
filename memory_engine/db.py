"""
memory_engine/mongo.py

MongoDB client and memory write/read operations.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING

from .models import MemoryRecord, MemoryStatus


class MemoryDB:
    def __init__(self, mongo_url: str, db_name: str = "memories"):
        self._client = AsyncIOMotorClient(mongo_url)
        self._db = self._client[db_name]
        self._col = self._db["memory_records"]

    async def setup_indexes(self):
        """Call once on startup."""
        await self._col.create_index([("user_id", ASCENDING)])
        await self._col.create_index([("user_id", ASCENDING), ("status", ASCENDING)])
        await self._col.create_index([("user_id", ASCENDING), ("memory_type", ASCENDING)])
        await self._col.create_index([("id", ASCENDING)], unique=True)

    # Write

    async def write(self, record: MemoryRecord) -> MemoryRecord:
        await self._col.insert_one(record.dict())
        return record

    # Read

    async def get(self, memory_id: str) -> Optional[MemoryRecord]:
        doc = await self._col.find_one({"id": memory_id})
        return MemoryRecord(**doc) if doc else None

    async def get_active(self, user_id: str) -> list[MemoryRecord]:
        cursor = self._col.find({"user_id": user_id, "status": MemoryStatus.ACTIVE})
        return [MemoryRecord(**doc) async for doc in cursor]

    # Update

    async def update_access(self, memory_id: str):
        """Bump access count and reset age on retrieval."""
        await self._col.update_one(
            {"id": memory_id},
            {"$inc": {"access_count": 1},
             "$set": {"last_accessed_at": datetime.utcnow(),
                      "turns_since_access": 0}}
        )

    async def tick_turns(self, user_id: str):
        """Age all active memories by one turn. Call once per agent turn."""
        await self._col.update_many(
            {"user_id": user_id, "status": MemoryStatus.ACTIVE},
            {"$inc": {"turns_since_access": 1}}
        )

    async def archive(self, memory_id: str):
        await self._col.update_one(
            {"id": memory_id},
            {"$set": {"status": MemoryStatus.ARCHIVED}}
        )

    async def set_embedding_id(self, memory_id: str, embedding_id: str):
        await self._col.update_one(
            {"id": memory_id},
            {"$set": {"embedding_id": embedding_id}}
        )