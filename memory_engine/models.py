"""
memory_engine/models.py

Memory schema for MongoDB documents.
Defines the four memory types and their metadata fields.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MemoryType(str, Enum):
    PREFERENCE = "preference"   # user communication / UX preferences
    FACT       = "fact"         # stable factual info about the user
    EPISODE    = "episode"      # a specific event in a conversation
    RULE       = "rule"         # a behavioural directive for the agent
    PLANNING   = "planning"     # strategic plans and intentions
    TOOL_USAGE = "tool_usage"   # history of tool usage and outcomes


class MemoryStatus(str, Enum):
    ACTIVE   = "active"
    ARCHIVED = "archived"


class MemoryRecord(BaseModel):
    id:               str         = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id:          str
    session_id:       str
    content:          str
    memory_type:      MemoryType
    status:           MemoryStatus = MemoryStatus.ACTIVE

    # Scoring
    importance_score: float = 0.5   # heuristic, overwritten by GNN later
    access_count:     int   = 0

    # Temporal
    created_at:       datetime = Field(default_factory=datetime.utcnow)
    last_accessed_at: datetime = Field(default_factory=datetime.utcnow)
    turns_since_access: int = 0

    # Graph (populated in Phase 4)
    degree:           int = 0
    cluster_id:       int = -1

    # Embedding stored separately in Qdrant; we keep the id reference here
    embedding_id:     Optional[str] = None

    tags:             list[str] = []
    source_turn:      int = 0

    class Config:
        use_enum_values = True