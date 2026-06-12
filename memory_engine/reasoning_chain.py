"""
memory_engine/reasoning_chain.py

Component for managing memory reasoning chains to support logical inference and reasoning.
Implements requirement #5 of Phase 9: Introduce memory reasoning chains.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import time
from collections import defaultdict

from memory_engine.models import MemoryRecord, MemoryType
from memory_engine.db import MemoryDB
from memory_engine.embeddings.encoder import EmbeddingEngine


class ChainType(Enum):
    """Types of reasoning chains."""
    CAUSAL = "causal"         # A leads to B leads to C
    TEMPORAL = "temporal"     # Ordered by time
    SEMANTIC = "semantic"     # Related by meaning
    INFERENTIAL = "inferential" # Each step implies the next
    ASSOCIATIVE = "associative" # Associated by context


class MemoryChainLink:
    """Represents a link between two memories in a reasoning chain."""

    def __init__(
        self,
        source_memory_id: str,
        target_memory_id: str,
        chain_type: ChainType,
        strength: float = 1.0,
        context: Optional[str] = None,
    ):
        """
        Args:
            source_memory_id: ID of the source memory
            target_memory_id: ID of the target memory
            chain_type: Type of relationship between the memories
            strength: Strength of the link (0.0 to 1.0)
            context: Optional context explaining the relationship
        """
        self.source_memory_id = source_memory_id
        self.target_memory_id = target_memory_id
        self.chain_type = chain_type
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.context = context or ""
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "source_memory_id": self.source_memory_id,
            "target_memory_id": self.target_memory_id,
            "chain_type": self.chain_type.value,
            "strength": self.strength,
            "context": self.context,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryChainLink':
        """Create from dictionary."""
        return cls(
            source_memory_id=data["source_memory_id"],
            target_memory_id=data["target_memory_id"],
            chain_type=ChainType(data.get("chain_type", "associative")),
            strength=data.get("strength", 1.0),
            context=data.get("context"),
        )


class MemoryReasoningChain:
    """
    Represents a chain of memories connected by reasoning relationships.
    """

    def __init__(
        self,
        chain_id: str,
        user_id: str,
        chain_type: ChainType,
        name: str = "",
        description: str = "",
    ):
        """
        Args:
            chain_id: Unique identifier for the chain
            user_id: User ID this chain belongs to
            chain_type: Type of reasoning chain
            name: Human-readable name for the chain
            description: Description of the chain's purpose
        """
        self.chain_id = chain_id
        self.user_id = user_id
        self.chain_type = chain_type
        self.name = name
        self.description = description
        self.created_at = time.time()
        self.last_updated = time.time()
        self.memory_ids: List[str] = []  # Ordered list of memory IDs in the chain
        self.links: Dict[Tuple[str, str], MemoryChainLink] = {}  # (source, target) -> link
        self.metadata: Dict[str, Any] = {}

    def add_memory(self, memory_id: str, position: Optional[int] = None) -> bool:
        """
        Add a memory to the chain.

        Args:
            memory_id: ID of the memory to add
            position: Position to insert at (None for append)

        Returns:
            True if memory was added, False if already in chain
        """
        if memory_id in self.memory_ids:
            return False

        if position is None:
            self.memory_ids.append(memory_id)
        else:
            self.memory_ids.insert(position, memory_id)

        self.last_updated = time.time()
        return True

    def remove_memory(self, memory_id: str) -> bool:
        """
        Remove a memory from the chain.

        Args:
            memory_id: ID of the memory to remove

        Returns:
            True if memory was removed, False if not in chain
        """
        if memory_id not in self.memory_ids:
            return False

        self.memory_ids.remove(memory_id)

        # Remove any links involving this memory
        to_remove = []
        for source_id, target_id in self.links:
            if source_id == memory_id or target_id == memory_id:
                to_remove.append((source_id, target_id))

        for link_id in to_remove:
            del self.links[link_id]

        self.last_updated = time.time()
        return True

    def add_link(
        self,
        source_memory_id: str,
        target_memory_id: str,
        chain_type: ChainType,
        strength: float = 1.0,
        context: Optional[str] = None,
    ) -> bool:
        """
        Add a link between two memories in the chain.

        Args:
            source_memory_id: Source memory ID
            target_memory_id: Target memory ID
            chain_type: Type of relationship
            strength: Strength of the link
            context: Optional context for the relationship

        Returns:
            True if link was added, False if memories not in chain
        """
        if source_memory_id not in self.memory_ids or target_memory_id not in self.memory_ids:
            return False

        link = MemoryChainLink(
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            chain_type=chain_type,
            strength=strength,
            context=context,
        )

        self.links[(source_memory_id, target_memory_id)] = link
        self.last_updated = time.time()
        return True

    def get_chain_sequence(self) -> List[str]:
        """
        Get the ordered sequence of memory IDs in the chain.

        Returns:
            List of memory IDs in chain order
        """
        return self.memory_ids.copy()

    def get_links_for_memory(self, memory_id: str) -> List[MemoryChainLink]:
        """
        Get all links involving a specific memory.

        Args:
            memory_id: Memory ID to get links for

        Returns:
            List of MemoryChainLink objects
        """
        links = []
        for (source_id, target_id), link in self.links.items():
            if source_id == memory_id or target_id == memory_id:
                links.append(link)
        return links

    def compute_chain_coherence(self) -> float:
        """
        Compute the overall coherence/strength of the chain based on link strengths.

        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not self.links:
            return 1.0 if len(self.memory_ids) <= 1 else 0.0

        total_strength = sum(link.strength for link in self.links.values())
        max_possible_strength = len(self.links)  # If all links had strength 1.0

        if max_possible_strength == 0:
            return 0.0

        return total_strength / max_possible_strength

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chain_id": self.chain_id,
            "user_id": self.user_id,
            "chain_type": self.chain_type.value,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "memory_ids": self.memory_ids.copy(),
            "links": {
                f"{source}_{target}": link.to_dict()
                for (source, target), link in self.links.items()
            },
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryReasoningChain':
        """Create from dictionary."""
        chain = cls(
            chain_id=data["chain_id"],
            user_id=data["user_id"],
            chain_type=ChainType(data.get("chain_type", "associative")),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )
        chain.created_at = data.get("created_at", time.time())
        chain.last_updated = data.get("last_updated", time.time())
        chain.memory_ids = data.get("memory_ids", []).copy()
        chain.metadata = data.get("metadata", {}).copy()

        # Reconstruct links
        links_data = data.get("links", {})
        for link_key, link_data in links_data.items():
            # Parse the key back to source_id, target_id
            # Assuming format "source_target" where source and target are memory IDs
            parts = link_key.split("_", 1)  # Split on first underscore only
            if len(parts) == 2:
                source_id, target_id = parts
                link = MemoryChainLink.from_dict(link_data)
                chain.links[(source_id, target_id)] = link

        return chain


class MemoryChainManager:
    """
    Manages memory reasoning chains for users.
    """

    def __init__(self, db: MemoryDB):
        """
        Args:
            db: MemoryDB instance for storing and retrieving chains and memories
        """
        self.db = db
        # In a full implementation, we'd have dedicated storage for chains
        # For now, we'll use a simplified approach

    async def create_chain(
        self,
        chain_id: str,
        user_id: str,
        chain_type: ChainType,
        name: str = "",
        description: str = "",
    ) -> MemoryReasoningChain:
        """
        Create a new memory reasoning chain.

        Args:
            chain_id: Unique identifier for the chain
            user_id: User ID this chain belongs to
            chain_type: Type of reasoning chain
            name: Human-readable name
            description: Description of the chain's purpose

        Returns:
            The created MemoryReasoningChain object
        """
        chain = MemoryReasoningChain(
            chain_id=chain_id,
            user_id=user_id,
            chain_type=chain_type,
            name=name,
            description=description,
        )

        # In a full implementation, we would store this in a dedicated collection
        # For now, we'll just return the chain object
        return chain

    async def get_chain(self, chain_id: str, user_id: str) -> Optional[MemoryReasoningChain]:
        """
        Get a memory reasoning chain by ID.

        Args:
            chain_id: Chain ID to retrieve
            user_id: User ID to verify ownership

        Returns:
            MemoryReasoningChain object if found and accessible, None otherwise
        """
        # In a full implementation, we would query the chains collection
        # and verify user ownership
        return None  # Placeholder

    async def add_memory_to_chain(
        self,
        chain_id: str,
        user_id: str,
        memory_id: str,
        position: Optional[int] = None,
    ) -> bool:
        """
        Add a memory to an existing chain.

        Args:
            chain_id: Chain ID
            user_id: User ID (for ownership verification)
            memory_id: Memory ID to add
            position: Position to insert at (None for append)

        Returns:
            True if successful, False otherwise
        """
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return False

        return chain.add_memory(memory_id, position)

    async def create_chain_link(
        self,
        chain_id: str,
        user_id: str,
        source_memory_id: str,
        target_memory_id: str,
        chain_type: ChainType,
        strength: float = 1.0,
        context: Optional[str] = None,
    ) -> bool:
        """
        Create a link between two memories in a chain.

        Args:
            chain_id: Chain ID
            user_id: User ID (for ownership verification)
            source_memory_id: Source memory ID
            target_memory_id: Target memory ID
            chain_type: Type of relationship
            strength: Strength of the link
            context: Optional context for the relationship

        Returns:
            True if successful, False otherwise
        """
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return False

        return chain.add_link(
            source_memory_id=source_memory_id,
            target_memory_id=target_memory_id,
            chain_type=chain_type,
            strength=strength,
            context=context,
        )

    async def get_chain_reasoning_path(
        self,
        chain_id: str,
        user_id: str,
        start_memory_id: Optional[str] = None,
        end_memory_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get a reasoning path through the chain, optionally from start to end memory.

        Args:
            chain_id: Chain ID
            user_id: User ID (for ownership verification)
            start_memory_id: Optional starting memory ID
            end_memory_id: Optional ending memory ID

        Returns:
            List of dictionaries representing the reasoning path
        """
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return []

        # Get the full chain sequence
        memory_ids = chain.get_chain_sequence()

        # If start and end are specified, find the subsequence
        if start_memory_id is not None or end_memory_id is not None:
            start_idx = 0
            end_idx = len(memory_ids)

            if start_memory_id is not None:
                try:
                    start_idx = memory_ids.index(start_memory_id)
                except ValueError:
                    pass  # Keep start_idx as 0

            if end_memory_id is not None:
                try:
                    end_idx = memory_ids.index(end_memory_id) + 1
                except ValueError:
                    pass  # Keep end_idx as len(memory_ids)

            # Ensure valid range
            start_idx = max(0, min(start_idx, len(memory_ids)))
            end_idx = max(start_idx, min(end_idx, len(memory_ids)))
            memory_ids = memory_ids[start_idx:end_idx]

        # Get memories from database
        memories = []
        for memory_id in memory_ids:
            memory = await self.db.get(memory_id)
            if memory:
                memories.append(memory)

        # Build reasoning path
        reasoning_path = []
        for i, memory in enumerate(memories):
            step_info = {
                "step": i + 1,
                "memory_id": memory.id,
                "content": memory.content,
                "memory_type": memory.memory_type.value,
                "importance": memory.importance_score,
                "links": [],
            }

            # Add links from/to this memory
            for link in chain.get_links_for_memory(memory.id):
                step_info["links"].append({
                    "type": link.chain_type.value,
                    "strength": link.strength,
                    "context": link.context,
                    "direction": "outgoing" if link.source_memory_id == memory.id else "incoming",
                    "target_id": link.target_memory_id if link.source_memory_id == memory.id else link.source_memory_id,
                })

            reasoning_path.append(step_info)

        return reasoning_path

    async def evaluate_chain_relevance(
        self,
        chain_id: str,
        user_id: str,
        query_memory_id: str,
    ) -> float:
        """
        Evaluate how relevant a chain is to a given query memory.

        Args:
            chain_id: Chain ID
            user_id: User ID (for ownership verification)
            query_memory_id: Memory ID representing the query/context

        Returns:
            Relevance score (0.0 to 1.0)
        """
        chain = await self.get_chain(chain_id, user_id)
        if not chain:
            return 0.0

        # Get the query memory
        query_memory = await self.db.get(query_memory_id)
        if not query_memory:
            return 0.0

        # Simple relevance: check if query memory is in the chain
        if query_memory_id in chain.memory_ids:
            return 1.0

        # More sophisticated approach would involve semantic similarity
        # between the query memory and memories in the chain
        # For now, we'll return a basic score based on chain coherence
        # if the query is related to any memory in the chain

        # For this implementation, we'll return the chain coherence as a rough estimate
        return chain.compute_chain_coherence()


# Factory function for easy instantiation
def create_memory_chain_manager(db: MemoryDB) -> MemoryChainManager:
    """
    Create a MemoryChainManager instance.

    Args:
        db: MemoryDB instance

    Returns:
        MemoryChainManager instance
    """
    return MemoryChainManager(db=db)