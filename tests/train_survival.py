"""
tests/train_survival.py

Training script for the survival classifier (Phase 7).
This script can be used to train a logistic regression model on historical
memory access data. For now, it uses synthetic data for demonstration.
"""

from __future__ import annotations
import asyncio
import os
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from memory_engine.survival_classifier import SurvivalClassifierTrainer, create_survival_classifier
from memory_engine.models import MemoryRecord, MemoryType
from memory_engine.db import MemoryDB


def generate_synthetic_data(num_samples: int = 1000) -> List[MemoryRecord]:
    """
    Generate synthetic memory records for training.
    In practice, replace this with loading real data from the database.
    """
    records = []
    for _ in range(num_samples):
        # Random importance score
        importance = np.random.uniform(0.0, 1.0)
        # Random memory type (returns numpy string, convert to enum)
        mtype_str = np.random.choice([mt.value for mt in MemoryType])
        mtype = MemoryType(mtype_str)
        # Random turns since access (0-100)
        turns = np.random.randint(0, 101)
        # Random access count (0-50)
        access = np.random.randint(0, 51)
        # Random content (not used in features)
        content = f"synthetic memory {_}"

        record = MemoryRecord(
            user_id="synthetic_user",
            session_id="synthetic_session",
            content=content,
            memory_type=mtype,
            importance_score=importance,
            turns_since_access=turns,
            access_count=access,
        )
        records.append(record)
    return records


async def train_survival_classifier(
    mongo_url: str,
    model_save_path: str,
    num_epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    device: str = "cpu",
):
    """
    Train the survival classifier.
    Args:
        mongo_url: MongoDB connection string (unused in synthetic mode, but kept for API consistency)
        model_save_path: Path to save the trained model
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        device: torch device
    """
    print(f"Training survival classifier on device: {device}")

    # Initialize trainer
    trainer = create_survival_classifier(
        model_path=model_save_path,  # We'll save to this path
        device=device,
        learning_rate=learning_rate,
    )

    # Generate or load training data
    # In a real scenario, you would query the database for historical memory records
    # and label them based on whether they were accessed again in the future.
    print("Generating synthetic training data...")
    all_records = generate_synthetic_data(num_samples=5000)

    # Simple training loop
    for epoch in range(num_epochs):
        # Shuffle data
        np.random.shuffle(all_records)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(all_records), batch_size):
            batch_records = all_records[i:i+batch_size]
            loss = trainer.train_step(batch_records)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Optionally save checkpoint every few epochs
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(metrics={"epoch": epoch+1, "loss": avg_loss})

    # Final save
    trainer.save_checkpoint(metrics={"epoch": num_epochs, "loss": avg_loss})
    print(f"Training complete. Model saved to {model_save_path}")


async def main():
    # These would typically come from environment or config
    mongo_url = os.getenv(
        "MONGO_URL", "mongodb://agent:agent@mongo:27017/memories?authSource=admin"
    )
    model_save_path = os.getenv("SURVIVAL_MODEL_PATH", "survival_classifier.pt")
    await train_survival_classifier(
        mongo_url=mongo_url,
        model_save_path=model_save_path,
        num_epochs=20,
        batch_size=256,
        learning_rate=0.01,
        device="cpu",
    )


if __name__ == "__main__":
    asyncio.run(main())