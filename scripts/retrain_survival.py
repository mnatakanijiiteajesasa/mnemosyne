#!/usr/bin/env python3
"""
scripts/retrain_survival.py

Retrain survival classifier using real data from MongoDB.
Labels are based on whether a memory was accessed more than once in the lookback period
(indication of recurrence).
"""

import argparse
import os
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict
import torch
from torch import nn

# Add the project root to the path so we can import memory_engine
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from memory_engine.interaction_logger import InteractionLogger
from memory_engine.db import MemoryDB
from memory_engine.models import MemoryRecord, MemoryType
from memory_engine.survival_classifier import SurvivalClassifier, SurvivalClassifierTrainer


def memory_type_onehot(memory_type: MemoryType) -> np.ndarray:
    """Convert MemoryType to one-hot vector of length 5."""
    mapping = {
        MemoryType.PREFERENCE: 0,
        MemoryType.FACT: 1,
        MemoryType.EPISODE: 2,
        MemoryType.RULE: 3,
        MemoryType.PLANNING: 4,
    }
    vec = np.zeros(5, dtype=np.float32)
    vec[mapping[memory_type]] = 1.0
    return vec


async def load_memories_and_interactions(
    mongo_url: str, lookback_days: int = 60
) -> tuple[List[dict], List[dict]]:
    """
    Load memory records and interaction logs from MongoDB for the given lookback period.
    Returns:
        memories: list of memory documents (as dicts)
        interactions: list of interaction documents (as dicts)
    """
    db = MemoryDB(mongo_url)
    logger = InteractionLogger(mongo_url)

    # Calculate cutoff time
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    cutoff_timestamp = cutoff.timestamp()

    # Fetch memories that have been accessed within the lookback window
    print(f"Loading memories from the last {lookback_days} days...")
    cursor = db._col.find({"last_accessed_at": {"$gte": cutoff}})
    memories = [doc async for doc in cursor]
    print(f"Loaded {len(memories)} memories.")

    # Fetch interactions within the same window
    print(f"Loading interactions from the last {lookback_days} days...")
    interactions = await logger.get_recent_interactions(limit=200000)  # generous limit
    # Filter by timestamp
    recent_interactions = [
        it for it in interactions if it.get("timestamp", 0) >= cutoff.timestamp()
    ]
    print(f"Loaded {len(recent_interactions)} interactions.")

    await db.close()
    return memories, recent_interactions


def build_memory_access_times(
    memories: List[dict], interactions: List[dict]
) -> Dict[str, List[float]]:
    """
    Build a dictionary mapping memory_id to list of timestamps (float) when the memory was accessed.
    Accessed via turns, retrievals, or writes.
    """
    # Initialize with empty lists
    access_times = {mem["id"]: [] for mem in memories}

    for interaction in interactions:
        it_type = interaction.get("interaction_type")
        timestamp = interaction.get("timestamp", 0)
        if it_type == "turn":
            details = interaction.get("details", {})
            retrieved = details.get("memories_retrieved", [])
            for mem_ref in retrieved:
                mid = mem_ref.get("memory_id")
                if mid in access_times:
                    access_times[mid].append(timestamp)
        elif it_type == "retrieval":
            details = interaction.get("details", {})
            results = details.get("results", [])
            for mem_ref in results:
                mid = mem_ref.get("memory_id")
                if mid in access_times:
                    access_times[mid].append(timestamp)
        elif it_type == "memory_write":
            details = interaction.get("details", {})
            mem_id = details.get("memory_id")
            if mem_id in access_times:
                access_times[mem_id].append(timestamp)

    return access_times


def label_memories_by_recurrence(
    memories: List[dict], access_times: Dict[str, List[float]]
) -> List[float]:
    """
    Label each memory as 1.0 if it has been accessed more than once (indicating recurrence),
    else 0.0.
    """
    labels = []
    for mem in memories:
        mid = mem["id"]
        times = access_times.get(mid, [])
        if len(times) > 1:
            labels.append(1.0)
        else:
            labels.append(0.0)
    return labels


def main():
    parser = argparse.ArgumentParser(description="Retrain survival classifier using real MongoDB data.")
    parser.add_argument("--mongo-url", default=os.getenv("MONGO_URL", "mongodb://agent:agent@mongo:27017/memories?authSource=admin"),
                        help="MongoDB connection string")
    parser.add_argument("--model-output", default=os.getenv("SURVIVAL_MODEL_PATH", "survival_classifier.pt"),
                        help="Path to save the retrained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--lookback-days", type=int, default=60, help="How many days of data to look back for training")
    args = parser.parse_args()

    print(f"Starting survival classifier retraining...")
    print(f"Mongo URL: {args.mongo_url}")
    print(f"Model output: {args.model_output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Lookback days: {args.lookback_days}")

    # Load data
    memories, interactions = asyncio.run(
        load_memories_and_interactions(args.mongo_url, args.lookback_days)
    )
    if not memories:
        print("No memories found. Exiting.")
        return

    # Build access times
    print("Building memory access times...")
    access_times = build_memory_access_times(memories, interactions)

    # Label memories
    print("Labeling memories based on recurrence (more than one access)...")
    labels = label_memories_by_recurrence(memories, access_times)
    print(f"Positive labels (recurrent): {sum(labels)} / {len(labels)}")

    # Prepare features and labels for training
    print("Preparing features...")
    # We'll create a temporary trainer to use its prepare_features method
    dummy_trainer = SurvivalClassifierTrainer(device=args.device)
    features_list = []
    for mem in memories:
        record = MemoryRecord(**mem)
        features_list.append(dummy_trainer.prepare_features(record))
    features = np.stack(features_list)  # (N, 8)
    labels_arr = np.array(labels, dtype=np.float32).reshape(-1, 1)  # (N, 1)

    # Convert to tensors
    features_tensor = torch.from_numpy(features).float().to(args.device)
    labels_tensor = torch.from_numpy(labels_arr).float().to(args.device)

    # Initialize model and optimizer/criterion
    print("Initializing model...")
    model = SurvivalClassifier().to(args.device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("Starting training...")
    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(features_tensor)
        loss = criterion(outputs, labels_tensor)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), args.model_output)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss_val:.6f} -> **best**")
        else:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss_val:.6f}")

    print(f"Training completed. Best loss: {best_loss:.6f}")
    print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    asyncio.run(main())