"""
memory_engine/interaction_logger.py

Logging system for Phase 8: Autonomous Learning.
Logs every retrieval and user interaction to MongoDB for offline analysis and retraining.
"""

from __future__ import annotations
import time
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

from memory_engine.models import MemoryType


class InteractionLogger:
    """
    Logs user interactions and retrievals to MongoDB.
    """

    def __init__(self, mongo_url: str, db_name: str = "memories"):
        """
        Args:
            mongo_url: MongoDB connection string
            db_name: Database name
        """
        self._client = AsyncIOMotorClient(mongo_url)
        self._db = self._client[db_name]
        self._collection = self._db["interaction_logs"]

    async def setup_indexes(self):
        """Create indexes for efficient querying."""
        await self._collection.create_index([("user_id", ASCENDING), ("timestamp", ASCENDING)])
        await self._collection.create_index([("interaction_type", ASCENDING)])

    async def log_turn(
        self,
        user_id: str,
        session_id: Optional[str],
        query: str,
        top_k: int,
        memories_written: list[str],
        memories_retrieved: list[dict],
        reply: str,
        archived_count: int,
    ) -> None:
        """
        Log a /turn interaction.
        """
        log_entry = {
            "user_id": user_id,
            "session_id": session_id,
            "interaction_type": "turn",
            "timestamp": time.time(),
            "details": {
                "query": query,
                "top_k": top_k,
                "memories_written_count": len(memories_written),
                "memories_written": memories_written,  # Store IDs
                "memories_retrieved_count": len(memories_retrieved),
                "memories_retrieved": [
                    {
                        "memory_id": r.get("memory_id"),
                        "score": r.get("score"),
                        "memory_type": r.get("payload", {}).get("memory_type"),
                        "content_preview": r.get("payload", {}).get("content", "")[:100]  # Truncate for storage
                    }
                    for r in memories_retrieved
                ],
                "reply_preview": reply[:200],  # Truncate reply
                "archived_count": archived_count,
            }
        }
        await self._collection.insert_one(log_entry)

    async def log_retrieval(
        self,
        user_id: str,
        query: str,
        top_k: int,
        results: list[dict],
        search_method: str = "hybrid",  # e.g., 'hybrid', 'vector_only'
    ) -> None:
        """
        Log a /memory/retrieve interaction.
        """
        log_entry = {
            "user_id": user_id,
            "session_id": None,  # Not always available in retrieval endpoint
            "interaction_type": "retrieval",
            "timestamp": time.time(),
            "details": {
                "query": query,
                "top_k": top_k,
                "results_count": len(results),
                "search_method": search_method,
                "results": [
                    {
                        "memory_id": r.get("memory_id"),
                        "score": r.get("score"),
                        "memory_type": r.get("payload", {}).get("memory_type"),
                        "content_preview": r.get("payload", {}).get("content", "")[:100]
                    }
                    for r in results
                ],
            }
        }
        await self._collection.insert_one(log_entry)

    async def log_memory_write(
        self,
        user_id: str,
        session_id: Optional[str],
        content: str,
        memory_type: MemoryType,
        tags: list[str],
        memory_id: str,
    ) -> None:
        """
        Log a /memory/write interaction (optional, but useful for completeness).
        """
        log_entry = {
            "user_id": user_id,
            "session_id": session_id,
            "interaction_type": "memory_write",
            "timestamp": time.time(),
            "details": {
                "content_preview": content[:200],
                "memory_type": memory_type,
                "tags": tags,
                "memory_id": memory_id,
            }
        }
        await self._collection.insert_one(log_entry)

    async def get_user_interactions(
        self,
        user_id: str,
        limit: int = 100,
        interaction_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve interaction logs for a user.
        """
        query = {"user_id": user_id}
        if interaction_type:
            query["interaction_type"] = interaction_type
        cursor = self._collection.find(query).sort("timestamp", -1).limit(limit)
        return [doc async for doc in cursor]

    async def get_recent_interactions(
        self,
        limit: int = 1000,
        interaction_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Retrieve recent interactions across all users (for offline retraining).
        """
        query = {}
        if interaction_type:
            query["interaction_type"] = interaction_type
        cursor = self._collection.find(query).sort("timestamp", -1).limit(limit)
        return [doc async for doc in cursor]