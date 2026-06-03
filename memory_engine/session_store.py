"""
memory_engine/session_store.py

MongoDB persistence for user sessions.
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

from .session import Session


class SessionStore:
    def __init__(self, mongo_url: str, db_name: str = "memories"):
        self._client = AsyncIOMotorClient(mongo_url)
        self._col    = self._client[db_name]["sessions"]

    async def setup_indexes(self):
        await self._col.create_index([("id", ASCENDING)], unique=True)
        await self._col.create_index([("user_id", ASCENDING)])

    # ------------------------------------------------------------------

    async def create(self, user_id: str) -> Session:
        session = Session(user_id=user_id)
        await self._col.insert_one(session.dict())
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        doc = await self._col.find_one({"id": session_id})
        return Session(**doc) if doc else None

    async def get_or_create(self, user_id: str, session_id: Optional[str]) -> Session:
        if session_id:
            session = await self.get(session_id)
            if session:
                return session
        return await self.create(user_id)

    async def tick(self, session_id: str) -> int:
        """Increment turn count, return new count."""
        result = await self._col.find_one_and_update(
            {"id": session_id},
            {"$inc": {"turn_count": 1},
             "$set": {"last_active": datetime.utcnow()}},
            return_document=True,
        )
        return result["turn_count"]

    async def list_sessions(self, user_id: str) -> list[Session]:
        cursor = self._col.find({"user_id": user_id})
        return [Session(**doc) async for doc in cursor]