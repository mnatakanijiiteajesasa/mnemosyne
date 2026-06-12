"""
memory_engine/shared_memory.py

Component for managing shared memory spaces accessible by multiple agents.
Implements requirement #4 of Phase 9: Add multi-agent shared memory support.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime

from memory_engine.models import MemoryRecord, MemoryType, MemoryStatus
from memory_engine.db import MemoryDB


class AccessLevel(Enum):
    """Access levels for shared memory spaces."""
    PRIVATE = "private"      # Only the owning agent can read/write
    SHARED = "shared"        # All agents in the group can read/write
    READ_ONLY = "read_only"  # Agents can read but not write
    ADMIN = "admin"          # Full control including managing access


class SharedMemorySpace:
    """
    Represents a shared memory space that multiple agents can access.
    """

    def __init__(
        self,
        space_id: str,
        name: str,
        description: str = "",
        created_by: str = "",
        access_level: AccessLevel = AccessLevel.PRIVATE,
        allowed_agents: Optional[Set[str]] = None,
    ):
        """
        Args:
            space_id: Unique identifier for the memory space
            name: Human-readable name
            description: Description of the space's purpose
            created_by: Agent ID that created this space
            access_level: Default access level for the space
            allowed_agents: Set of agent IDs allowed to access this space (None for all)
        """
        self.space_id = space_id
        self.name = name
        self.description = description
        self.created_by = created_by
        self.created_at = datetime.utcnow()
        self.access_level = access_level
        self.allowed_agents = allowed_agents or set()
        self.metadata: Dict[str, Any] = {}

    def can_access(self, agent_id: str) -> bool:
        """Check if an agent can access this memory space."""
        if self.access_level == AccessLevel.PRIVATE:
            return agent_id == self.created_by
        elif self.access_level == AccessLevel.SHARED:
            return len(self.allowed_agents) == 0 or agent_id in self.allowed_agents
        elif self.access_level == AccessLevel.READ_ONLY:
            return len(self.allowed_agents) == 0 or agent_id in self.allowed_agents
        elif self.access_level == AccessLevel.ADMIN:
            return len(self.allowed_agents) == 0 or agent_id in self.allowed_agents
        return False

    def can_write(self, agent_id: str) -> bool:
        """Check if an agent can write to this memory space."""
        if not self.can_access(agent_id):
            return False

        if self.access_level == AccessLevel.READ_ONLY:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "space_id": self.space_id,
            "name": self.name,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "access_level": self.access_level.value,
            "allowed_agents": list(self.allowed_agents),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SharedMemorySpace':
        """Create from dictionary."""
        space = cls(
            space_id=data["space_id"],
            name=data["name"],
            description=data.get("description", ""),
            created_by=data.get("created_by", ""),
            access_level=AccessLevel(data.get("access_level", "private")),
            allowed_agents=set(data.get("allowed_agents", [])),
        )
        space.metadata = data.get("metadata", {})
        if "created_at" in data:
            space.created_at = datetime.fromisoformat(data["created_at"])
        return space


class SharedMemoryManager:
    """
    Manages multiple shared memory spaces and controls access to them.
    """

    def __init__(self, db: MemoryDB):
        """
        Args:
            db: MemoryDB instance (we'll extend it to handle shared memory spaces)
        """
        self.db = db
        # In a full implementation, we'd have tables/collections for shared spaces
        # For now, we'll use the existing db and add shared memory capabilities

    async def create_shared_space(
        self,
        space_id: str,
        name: str,
        description: str = "",
        created_by: str = "",
        access_level: AccessLevel = AccessLevel.PRIVATE,
        allowed_agents: Optional[List[str]] = None,
    ) -> SharedMemorySpace:
        """
        Create a new shared memory space.

        Args:
            space_id: Unique identifier for the space
            name: Human-readable name
            description: Description of the space
            created_by: Agent creating the space
            access_level: Access level for the space
            allowed_agents: List of agent IDs allowed to access the space

        Returns:
            The created SharedMemorySpace object
        """
        space = SharedMemorySpace(
            space_id=space_id,
            name=name,
            description=description,
            created_by=created_by,
            access_level=access_level,
            allowed_agents=set(allowed_agents or []),
        )

        # In a full implementation, we would store this in a dedicated collection
        # For now, we'll store it as a special memory record with a reserved memory_type
        # We'll handle this by extending the MemoryType enum or using tags

        # For this implementation, we'll just return the space object
        # The actual implementation would depend on how we want to store shared space metadata
        return space

    async def get_shared_space(self, space_id: str) -> Optional[SharedMemorySpace]:
        """
        Get a shared memory space by ID.

        Args:
            space_id: ID of the space to retrieve

        Returns:
            SharedMemorySpace object if found, None otherwise
        """
        # In a full implementation, we would query a dedicated collection
        # For now, returning None as placeholder
        return None

    async def list_shared_spaces(
        self,
        agent_id: str,
        access_level: Optional[AccessLevel] = None,
    ) -> List[SharedMemorySpace]:
        """
        List shared memory spaces accessible by an agent.

        Args:
            agent_id: Agent ID to check access for
            access_level: Optional filter by access level

        Returns:
            List of accessible SharedMemorySpace objects
        """
        # In a full implementation, we would query the shared spaces collection
        # and filter by access permissions
        return []  # Placeholder

    async def write_to_shared_space(
        self,
        space_id: str,
        agent_id: str,
        content: str,
        memory_type: MemoryType,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Write a memory to a shared space if the agent has permission.

        Args:
            space_id: ID of the shared memory space
            agent_id: Agent attempting to write
            content: Memory content
            memory_type: Type of memory
            tags: Optional tags for the memory

        Returns:
            Memory ID if successful, None if failed or denied
        """
        # Check if the space exists and agent has write permission
        space = await self.get_shared_space(space_id)
        if not space or not space.can_write(agent_id):
            return None

        # In a full implementation, we would:
        # 1. Associate the memory with the shared space (via tags or a separate relationship)
        # 2. Store the memory in the database with appropriate metadata
        # 3. Return the memory ID

        # For now, we'll return None as a placeholder
        # The actual implementation would depend on our storage strategy for shared memories
        return None

    async def read_from_shared_space(
        self,
        space_id: str,
        agent_id: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryRecord]:
        """
        Read memories from a shared space if the agent has permission.

        Args:
            space_id: ID of the shared memory space
            agent_id: Agent attempting to read
            limit: Maximum number of memories to return
            memory_type: Optional filter by memory type

        Returns:
            List of MemoryRecord objects
        """
        # Check if the space exists and agent has access permission
        space = await self.get_shared_space(space_id)
        if not space or not space.can_access(agent_id):
            return []

        # In a full implementation, we would:
        # 1. Query memories associated with this shared space
        # 2. Apply any filters (memory_type, limit, etc.)
        # 3. Return the results

        # For now, we'll return an empty list as a placeholder
        return []


# Factory function for easy instantiation
def create_shared_memory_manager(db: MemoryDB) -> SharedMemoryManager:
    """
    Create a SharedMemoryManager instance.

    Args:
        db: MemoryDB instance

    Returns:
        SharedMemoryManager instance
    """
    return SharedMemoryManager(db=db)