"""
memory_engine/survival_classifier.py

Learned survival classifier for Phase 7.
Replaces the heuristic decay model with a logistic regression model trained
on historical recall behavior.
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path

from memory_engine.models import MemoryRecord, MemoryType
from memory_engine.db import MemoryDB


class SurvivalClassifier(nn.Module):
    """
    Logistic regression model for predicting memory survival probability.
    """

    def __init__(self, input_dim: int = 7):
        """
        Args:
            input_dim: Number of input features.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        Returns:
            Survival probability tensor of shape [batch_size, 1]
        """
        return self.sigmoid(self.linear(x))


class SurvivalClassifierTrainer:
    """
    Trainer for the survival classifier.
    """

    def __init__(
        self,
        model: SurvivalClassifier,
        learning_rate: float = 0.01,
        device: str = "cpu",
        model_path: Optional[str] = None,
    ):
        """
        Args:
            model: SurvivalClassifier instance
            learning_rate: Learning rate for optimizer
            device: torch device
            model_path: Path to save/load model checkpoint
        """
        self.model = model.to(device)
        self.device = device
        self.model_path = model_path
        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        # Load model if checkpoint exists
        if model_path and Path(model_path).exists():
            self.load_checkpoint()

    def prepare_features(self, record: MemoryRecord) -> np.ndarray:
        """
        Convert a MemoryRecord to a feature vector.
        Features:
          [0] importance_score
          [1-4] memory_type one-hot (PREFERENCE, FACT, EPISODE, RULE)
          [5] log(1 + turns_since_access)
          [6] log(1 + access_count)
        """
        # Importance score
        importance = record.importance_score

        # Memory type one-hot
        type_map = {
            MemoryType.PREFERENCE: 0,
            MemoryType.FACT: 1,
            MemoryType.EPISODE: 2,
            MemoryType.RULE: 3,
        }
        type_onehot = [0.0] * 4
        if record.memory_type in type_map:
            type_onehot[type_map[record.memory_type]] = 1.0

        # Log-scaled turns since access
        log_turns = np.log1p(record.turns_since_access)

        # Log-scaled access count
        log_access = np.log1p(record.access_count)

        features = np.array(
            [importance] + type_onehot + [log_turns, log_access],
            dtype=np.float32,
        )
        return features

    def prepare_batch(self, records: List[MemoryRecord]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare a batch of records for training.
        Returns:
            (features_tensor, labels_tensor)
        """
        features_list = []
        labels_list = []
        for record in records:
            features_list.append(self.prepare_features(record))
            # Label: 1 if we consider the memory should survive (not pruned), 0 otherwise
            # For now, we use the heuristic as a placeholder label.
            # In practice, this should be based on whether the memory was accessed again in the future.
            label = 1.0 if not self._heuristic_should_prune(record) else 0.0
            labels_list.append(label)

        features_tensor = torch.tensor(np.stack(features_list), dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1).to(self.device)
        return features_tensor, labels_tensor

    def _heuristic_should_prune(self, record: MemoryRecord) -> bool:
        """
        Placeholder heuristic for labeling (matches current decay.py).
        This should be replaced with real labels from historical data.
        """
        # Same parameters as in decay.py
        LAMBDA = {
            MemoryType.PREFERENCE: 0.003,
            MemoryType.FACT: 0.001,
            MemoryType.EPISODE: 0.008,
            MemoryType.RULE: 0.0005,
        }
        PRUNE_THRESHOLD = 0.08
        lam = LAMBDA.get(MemoryType(record.memory_type), 0.003)
        survival_score = record.importance_score * np.exp(-lam * record.turns_since_access)
        return survival_score < PRUNE_THRESHOLD

    def train_step(self, records: List[MemoryRecord]) -> float:
        """
        Perform a single training step on a batch of records.
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        features, labels = self.prepare_batch(records)
        outputs = self.model(features)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict_survival(self, record: MemoryRecord) -> float:
        """
        Predict survival probability for a single record.
        Returns:
            Probability in [0, 1] that the memory should survive (not be pruned).
        """
        self.model.eval()
        with torch.no_grad():
            features = self.prepare_features(record)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
            prob = self.model(features_tensor).item()
        return prob

    def should_prune(self, record: MemoryRecord, threshold: float = 0.5) -> bool:
        """
        Decide whether to prune a memory based on predicted survival probability.
        Args:
            record: MemoryRecord to check
            threshold: If survival probability < threshold, then prune.
        Returns:
            True if the memory should be pruned (archived).
        """
        survival_prob = self.predict_survival(record)
        return survival_prob < threshold

    def save_checkpoint(self, metrics: Optional[dict] = None):
        """
        Save model checkpoint.
        """
        if self.model_path is None:
            return
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
        }
        torch.save(checkpoint, self.model_path)
        print(f"Saved survival classifier checkpoint to {self.model_path}")

    def load_checkpoint(self):
        """
        Load model checkpoint.
        """
        if self.model_path is None or not Path(self.model_path).exists():
            print(f"No checkpoint found at {self.model_path}")
            return
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded survival classifier checkpoint from {self.model_path}")


def create_survival_classifier(
    model_path: Optional[str] = None,
    device: str = "cpu",
    learning_rate: float = 0.01,
) -> SurvivalClassifierTrainer:
    """
    Factory function to create a SurvivalClassifierTrainer.
    """
    model = SurvivalClassifier()
    trainer = SurvivalClassifierTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device,
        model_path=model_path,
    )
    return trainer