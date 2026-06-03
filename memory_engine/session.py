"""
memory_engine/session.py

Session tracking — persists conversation sessions in MongoDB.
"""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class Session(BaseModel):
    id:          str      = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id:     str
    turn_count:  int      = 0
    started_at:  datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True