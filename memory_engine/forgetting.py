"""
memory_engine/forgetting.py

Learned forgetting system for Phase 7.
Uses a survival classifier to predict which memories to archive.
Implements context-pressure eviction strategies.
"""

from __future__ import annotations
from typing import List, Optional
import os

from .survival_classifier import SurvivalClassifierTrainer, create_survival_classifier
from .db import MemoryDB
from .models import MemoryRecord


class ForgettingService:
    """
    Service responsible for identifying and archiving memories that are
    predicted to be forgotten (low survival probability).
    """

    def __init__(
        self,
        db: MemoryDB,
        model_path: Optional[str] = None,
        device: str = "cpu",
        survival_threshold: float = 0.5,
        # Context-pressure eviction parameters
        high_watermark: int = 1000,   # If active memories > high_watermark, increase pruning pressure
        low_watermark: int = 100,     # If active memories < low_watermark, decrease pruning pressure
        pressure_survival_threshold_shift: float = 0.1,  # How much to adjust threshold under pressure
    ):
        """
        Args:
            db: MemoryDB instance
            model_path: Path to survival classifier checkpoint. If None, use heuristic.
            device: torch device for model inference
            survival_threshold: Base threshold for survival probability (0-1).
                               Below this, memory is considered for archiving.
            high_watermark: Number of active memories that triggers high-pressure mode.
            low_watermark: Number of active memories that triggers low-pressure mode.
            pressure_survival_threshold_shift: Amount to adjust survival_threshold under pressure.
                                               In high pressure, threshold increases (more pruning).
                                               In low pressure, threshold decreases (less pruning).
        """
        self._db = db
        self._survival_threshold = survival_threshold
        self._high_watermark = high_watermark
        self._low_watermark = low_watermark
        self._pressure_shift = pressure_survival_threshold_shift

        # Try to load learned model; fall back to heuristic if fails or no path
        self._use_learned = False
        self._trainer: Optional[SurvivalClassifierTrainer] = None
        if model_path:
            try:
                self._trainer = create_survival_classifier(
                    model_path=model_path,
                    device=device,
                )
                self._use_learned = True
                print(f"ForgettingService: Loaded learned survival classifier from {model_path}")
            except Exception as e:
                print(f"ForgettingService: Failed to load learned model ({e}); falling back to heuristic")
        else:
            print("ForgettingService: No model path provided; using heuristic decay model")

    async def run(self, user_id: str) -> List[str]:
        """
        Check all active memories for a user and archive those predicted to be forgotten.
        Implements context-pressure eviction: adjusts survival threshold based on
        current number of active memories.
        Returns list of archived memory IDs.
        """
        records = await self._db.get_active(user_id)
        if not records:
            return []

        # Context-pressure eviction: adjust threshold based on number of active memories
        active_count = len(records)
        threshold = self._survival_threshold
        if active_count > self._high_watermark:
            # High pressure: increase threshold (more aggressive pruning)
            threshold = min(1.0, threshold + self._pressure_shift)
        elif active_count < self._low_watermark:
            # Low pressure: decrease threshold (less aggressive pruning)
            threshold = max(0.0, threshold - self._pressure_shift)

        archived: List[str] = []
        for record in records:
            should_archive = False
            if self._use_learned and self._trainer is not None:
                # Use learned model
                should_archive = self._trainer.should_prune(record, threshold=threshold)
            else:
                # Fall back to heuristic
                should_archive = self._heuristic_should_prune(record)

            if should_archive:
                await self._db.archive(record.id)
                archived.append(record.id)

        return archived

    def _heuristic_should_prune(self, record: MemoryRecord) -> bool:
        """
        Heuristic survival score (same as original decay.py).
        Used as fallback when learned model is not available.
        """
        # Parameters from decay.py
        LAMBDA = {
            MemoryType.PREFERENCE: 0.003,
            MemoryType.FACT: 0.001,
            MemoryType.EPISODE: 0.008,
            MemoryType.RULE: 0.0005,
        }
        PRUNE_THRESHOLD = 0.08
        lam = LAMBDA.get(MemoryType(record.memory_type), 0.003)
        import math
        survival_score = record.importance_score * math.exp(-lam * record.turns_since_access)
        return survival_score < PRUNE_THRESHOLD