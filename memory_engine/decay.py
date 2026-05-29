"""
memory_engine/decay.py

Cold-start exponential decay system.

Scores each memory's survival probability based on type, importance,
and how many turns have passed since it was last accessed.

Used in Phase 2 before the GNN survival classifier is available (Phase 7).

Formula:  survival(t) = importance * exp(-lambda_type * turns_since_access)
"""

from __future__ import annotations
import math
from .models import MemoryRecord, MemoryType


# Decay rates per memory type (from paper Table 1)
LAMBDA = {
    MemoryType.RULE:       0.0005,
    MemoryType.FACT:       0.001,
    MemoryType.PREFERENCE: 0.003,
    MemoryType.EPISODE:    0.008,
}

# Memories below this threshold get archived
PRUNE_THRESHOLD = 0.08


def survival_score(record: MemoryRecord) -> float:
    """Compute current survival probability for a memory."""
    lam = LAMBDA.get(MemoryType(record.memory_type), 0.003)
    return record.importance_score * math.exp(-lam * record.turns_since_access)


def should_prune(record: MemoryRecord) -> bool:
    return survival_score(record) < PRUNE_THRESHOLD