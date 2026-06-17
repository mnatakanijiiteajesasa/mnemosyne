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
        self._seed_decay_lambda = 0.15  # Decay rate for seed memories

    async def _decay_seed_memories(self, user_id: str):
        """
        Apply decay to seed memory confidences after new memory is written.
        seed_confidence = 0.3 * e^(-λ * real_memory_count)
        """
        try:
            # Get all active memories for the user
            all_memories = await self.db.get_active(user_id)

            # Count real (non-seed) memories
            real_memory_count = 0
            seed_memories = []

            for memory in all_memories:
                memory_dict = memory.dict() if hasattr(memory, 'dict') else dict(memory)
                seed_confidence = self._get_seed_confidence_from_dict(memory_dict)

                if seed_confidence >= 1.0:
                    # This is a real (non-seed) memory
                    real_memory_count += 1
                else:
                    # This is a seed memory - collect for decay processing
                    seed_memories.append((memory, seed_confidence))

            # If no real memories, no decay needed yet
            if real_memory_count == 0:
                return

            # Calculate decay factor: e^(-λ * real_memory_count)
            import math
            decay_factor = math.exp(-self._seed_decay_lambda * real_memory_count)

            # Apply decay to each seed memory
            for memory, original_confidence in seed_memories:
                # Original seed confidence should be 0.3 when first seeded
                # Apply decay: new_confidence = 0.3 * e^(-λ * real_memory_count)
                new_confidence = 0.3 * decay_factor

                # Only update if confidence has changed significantly
                if abs(new_confidence - original_confidence) > 0.01:
                    # Update the seed confidence in tags
                    await self._update_seed_confidence_in_tags(
                        memory.id,
                        original_confidence,
                        new_confidence
                    )

        except Exception as e:
            print(f"Error in seed decay for user {user_id}: {e}")

    def _get_seed_confidence_from_dict(self, memory_dict: dict) -> float:
        """
        Extract seed confidence from memory dict tags.
        Returns 1.0 for non-seed memories, actual confidence for seed memories.
        """
        tags = memory_dict.get("tags", [])
        seed_confidence = 1.0  # Default for non-seed memories

        for tag in tags:
            if tag.startswith("seed_confidence:"):
                try:
                    confidence_str = tag.split(":", 1)[1]
                    seed_confidence = float(confidence_str)
                    # Ensure it's in valid range
                    seed_confidence = max(0.0, min(1.0, seed_confidence))
                    break
                except (ValueError, IndexError):
                    # If parsing fails, treat as non-seed
                    seed_confidence = 1.0
                    break

        return seed_confidence

    async def _update_seed_confidence_in_tags(self, memory_id: str, old_confidence: float, new_confidence: float):
        """
        Update the seed confidence tag for a memory.
        """
        try:
            # Get the current memory record
            memory = await self.db.get(memory_id)
            if not memory:
                return

            # Convert to dict for manipulation
            memory_dict = memory.dict() if hasattr(memory, 'dict') else dict(memory)
            tags = memory_dict.get("tags", [])

            # Remove old seed_confidence tag and add new one
            new_tags = []
            for tag in tags:
                if not tag.startswith("seed_confidence:"):
                    new_tags.append(tag)

            # Add new seed_confidence tag
            new_tags.append(f"seed_confidence:{new_confidence}")

            # Update the memory record
            memory_dict["tags"] = new_tags

            # Create updated memory record
            updated_memory = MemoryRecord(**memory_dict)

            # Save the updated memory
            await self.db._col.update_one(
                {"id": memory_id},
                {"$set": {"tags": new_tags}}
            )

        except Exception as e:
            print(f"Error updating seed confidence for memory {memory_id}: {e}")

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
                "content":      content,
            },
        )

        # 4. Link embedding back in MongoDB
        await self._db.set_embedding_id(record.id, record.id)
       
        #5. Build graph edges to similar memories
        await self._graph.build_edges(record.id, user_id)

        # 6. Apply seed decay after new memory is written
        # This implements: seed_confidence = 0.3 * e^(-λ * real_memory_count)
        await self._decay_seed_memories(user_id)

        return record