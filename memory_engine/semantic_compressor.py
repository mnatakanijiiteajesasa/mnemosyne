"""
memory_engine/semantic_compressor.py

Component for compressing semantic memories by identifying patterns and extracting key information.
Implements requirement #3 of Phase 9: Create semantic memory compression.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta

from memory_engine.models import MemoryRecord, MemoryType
from memory_engine.db import MemoryDB
from memory_engine.embeddings.encoder import EmbeddingEngine


class SemanticMemoryCompressor:
    """
    Compresses semantic memories by identifying patterns and extracting key information.
    """

    def __init__(
        self,
        db: MemoryDB,
        encoder: EmbeddingEngine,
        similarity_threshold: float = 0.85,
        compression_threshold: int = 5,  # Minimum similar memories to trigger compression
    ):
        """
        Args:
            db: MemoryDB instance for accessing memories
            encoder: EmbeddingEngine for computing similarities
            similarity_threshold: Cosine similarity threshold for considering memories similar
            compression_threshold: Minimum number of similar memories to create a compressed memory
        """
        self.db = db
        self.encoder = encoder
        self.similarity_threshold = similarity_threshold
        self.compression_threshold = compression_threshold

    async def compress_similar_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        days_back: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Identify and compress similar memories for a user.

        Args:
            user_id: User ID
            memory_type: Optional memory type to filter by (None for all types)
            days_back: How many days back to look for memories

        Returns:
            List of compressed memory data dictionaries
        """
        # Fetch memories from the specified time period
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        # Note: In a real implementation, we would add date filtering to the DB query
        # For now, we'll get all active memories and filter in memory

        memories = await self.db.get_active(user_id)

        # Filter by memory type if specified
        if memory_type:
            memories = [m for m in memories if m.memory_type == memory_type]

        # Filter by date (assuming we have created_at field)
        memories = [
            m for m in memories
            if m.created_at >= cutoff_date
        ]

        if len(memories) < self.compression_threshold:
            return []  # Not enough memories to compress

        # Get embeddings for all memories
        memory_ids = [m.id for m in memories]
        embeddings = await self._get_memory_embeddings(memory_ids)

        # Group similar memories
        similarity_groups = self._group_similar_memories(memories, embeddings)

        # Create compressed memories for groups that meet the threshold
        compressed_memories = []
        for group in similarity_groups:
            if len(group) >= self.compression_threshold:
                compressed_memory = await self._create_compressed_memory(group)
                if compressed_memory:
                    compressed_memories.append(compressed_memory)

        return compressed_memories

    async def _get_memory_embeddings(self, memory_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get embeddings for a list of memory IDs."""
        embeddings = {}
        for memory_id in memory_ids:
            # Try to get embedding from Qdrant via the encoder
            # This is a simplified version - in practice we'd use the encoder's search capabilities
            try:
                # We'll use a dummy query to get the vector by ID
                # In a real implementation, we'd have a direct method to fetch vectors by ID
                results = await self.encoder._qdrant.scroll(
                    collection_name=self.encoder.COLLECTION_NAME,
                    scroll_filter=None,
                    limit=1000,  # Get enough to cover our memories
                    with_vectors=True,
                    with_payload=True
                )

                # Find the vectors for our memory IDs
                for point in results[0]:  # scroll returns (points, next_page_offset)
                    if point.payload.get("memory_id") in memory_ids:
                        embeddings[point.payload["memory_id"]] = np.array(point.vector)

            except Exception as e:
                # If we can't get the embedding, we'll skip this memory
                # In a production system, we'd want better error handling
                pass

        return embeddings

    def _group_similar_memories(
        self,
        memories: List[MemoryRecord],
        embeddings: Dict[str, np.ndarray]
    ) -> List[List[MemoryRecord]]:
        """Group memories by similarity using a simple greedy approach."""
        if not memories or not embeddings:
            return []

        # Initialize each memory as its own group
        groups = [[m] for m in memories if m.id in embeddings]

        # For each memory, try to merge with similar groups
        merged = True
        while merged:
            merged = False
            new_groups = []
            used_indices = set()

            for i, group in enumerate(groups):
                if i in used_indices:
                    continue

                current_group = group[:]
                used_indices.add(i)

                # Try to add similar memories to this group
                for j, other_group in enumerate(groups):
                    if j in used_indices or i == j:
                        continue

                    # Check if any memory in current group is similar to any in other_group
                    is_similar = False
                    for mem1 in current_group:
                        for mem2 in other_group:
                            if mem1.id in embeddings and mem2.id in embeddings:
                                similarity = self._cosine_similarity(
                                    embeddings[mem1.id],
                                    embeddings[mem2.id]
                                )
                                if similarity >= self.similarity_threshold:
                                    is_similar = True
                                    break
                        if is_similar:
                            break

                    if is_similar:
                        # Merge the groups
                        current_group.extend(other_group)
                        used_indices.add(j)
                        merged = True

                new_groups.append(current_group)

            groups = new_groups

        return groups

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    async def _create_compressed_memory(
        self,
        memories: List[MemoryRecord]
    ) -> Optional[Dict[str, Any]]:
        """
        Create a compressed memory representation from a group of similar memories.

        Args:
            memories: List of similar MemoryRecord objects

        Returns:
            Dictionary containing compressed memory data, or None if compression failed
        """
        if not memories:
            return None

        # Use the first memory as a base for metadata
        base_memory = memories[0]

        # Extract common content themes
        # In a real implementation, we might use topic modeling or key phrase extraction
        # For now, we'll create a simple summary based on the memories

        # Collect all content
        all_content = [m.content for m in memories]

        # Create a summary - in practice, we'd use an LLM or extractive summarization
        # For this implementation, we'll create a simple concatenation with deduplication
        summary_content = self._create_content_summary(all_content)

        # Calculate average importance
        avg_importance = np.mean([m.importance_score for m in memories])

        # Union of tags
        all_tags = set()
        for m in memories:
            all_tags.update(m.tags)

        # Create compressed memory
        compressed_memory = {
            "content": f"[Compressed {len(memories)} similar memories]: {summary_content}",
            "memory_type": base_memory.memory_type,
            "user_id": base_memory.user_id,
            "session_id": base_memory.session_id,  # Using the first one's session
            "importance_score": min(avg_importance * 1.2, 1.0),  # Boost importance slightly
            "tags": list(all_tags) + ["compressed"],
            "source_turn": 0,  # Will be set by caller
            # Add metadata about the compression
            "_compression_meta": {
                "original_count": len(memories),
                "original_ids": [m.id for m in memories],
                "compression_timestamp": datetime.utcnow().isoformat(),
            }
        }

        return compressed_memory

    def _create_content_summary(self, content_list: List[str]) -> str:
        """
        Create a summary of multiple content strings.
        In a real implementation, this would use more sophisticated NLP techniques.
        """
        if not content_list:
            return ""

        if len(content_list) == 1:
            return content_list[0]

        # Simple approach: find common phrases or just return a summary statement
        # For demonstration, we'll return a indication of compression
        # In practice, we might:
        # 1. Use extractive summarization to pick key sentences
        # 2. Use clustering to find common themes
        # 3. Use an LLM to generate a summary

        # For now, let's just show that we're compressing multiple similar items
        sample_content = content_list[0][:100]  # First 100 chars of first item
        if len(content_list) > 1:
            return f"Similar to: {sample_content}... and {len(content_list)-1} similar memories"
        else:
            return sample_content


# Factory function for easy instantiation
def create_semantic_memory_compressor(
    db: MemoryDB,
    encoder: EmbeddingEngine,
    similarity_threshold: float = 0.85,
    compression_threshold: int = 5,
) -> SemanticMemoryCompressor:
    """
    Create a SemanticMemoryCompressor instance.

    Args:
        db: MemoryDB instance
        encoder: EmbeddingEngine instance
        similarity_threshold: Similarity threshold for grouping memories
        compression_threshold: Minimum memories to trigger compression

    Returns:
        SemanticMemoryCompressor instance
    """
    return SemanticMemoryCompressor(
        db=db,
        encoder=encoder,
        similarity_threshold=similarity_threshold,
        compression_threshold=compression_threshold,
    )