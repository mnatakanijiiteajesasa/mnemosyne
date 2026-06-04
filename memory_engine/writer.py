"""
memory_engine/writer.py

Write pipeline: encode -> store in Qdrant -> persist in MongoDB.
"""

from __future__ import annotations

from .models import MemoryRecord, MemoryType
from .db import MemoryDB
from .embeddings.encoder import EmbeddingEngine
from .gnn_engine.graph import GraphBuilder

def estimate_importance(content: str, memory_type: MemoryType) -> float:
    base = {
        MemoryType.RULE:       0.85,
        MemoryType.PREFERENCE: 0.70,
        MemoryType.FACT:       0.60,
        MemoryType.EPISODE:    0.40,
    }[memory_type]

    boosts = ["always", "never", "important", "prefer", "hate", "love",
              "must", "require", "every time", "remember"]
    boost = sum(0.03 for kw in boosts if kw in content.lower())
    return min(base + boost, 1.0)


class MemoryWriter:
    def __init__(self, db: MemoryDB, encoder: EmbeddingEngine, graph: GraphBuilder):
        self._db      = db
        self._encoder = encoder
        self._graph   = graph

    async def write(
        self,
        user_id:     str,
        session_id:  str,
        content:     str,
        memory_type: MemoryType,
        tags:        list[str] = [],
        source_turn: int = 0,
    ) -> MemoryRecord:

        # 1. Build record
        record = MemoryRecord(
            user_id          = user_id,
            session_id       = session_id,
            content          = content,
            memory_type      = memory_type,
            importance_score = estimate_importance(content, memory_type),
            tags             = tags,
            source_turn      = source_turn,
        )

        # 2. Persist to MongoDB first to get the id
        await self._db.write(record)

        # 3. Encode + store in Qdrant
        await self._encoder.store(
            memory_id = record.id,
            text      = content,
            payload   = {
                "user_id":      user_id,
                "session_id":   session_id,
                "memory_type":  memory_type,
                "importance":   record.importance_score,
            },
        )

        # 4. Link embedding back in MongoDB
        await self._db.set_embedding_id(record.id, record.id)
       
        #5. Build graph edges to similar memories
        await self._graph.build_edges(record.id, user_id)

        return record