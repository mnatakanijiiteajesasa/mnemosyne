#!/usr/bin/env python3
"""
scripts/retrain_gnn.py

Offline GNN retraining script (Phase 8).
Loads interaction logs and memory records from MongoDB, computes features,
builds temporal graphs per user, and trains a MemoryGNN model.
"""

import argparse
import os
import sys
import asyncio
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer

# Add the project root to the path so we can import memory_engine
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from memory_engine.interaction_logger import InteractionLogger
from memory_engine.gnn_engine.processor import GraphProcessor
from memory_engine.gnn_engine.model import MemoryGNN
from memory_engine.gnn_engine.trainer import GNNTrainer
from memory_engine.db import MemoryDB
from memory_engine.models import MemoryType, MemoryStatus
import torch
from torch_geometric.data import Data


def compute_embedding(text: str, embedder: SentenceTransformer) -> np.ndarray:
    """Compute sentence embedding for text."""
    return embedder.encode([text], normalize_embeddings=True)[0]


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


def build_user_graph(
    user_id: str,
    memories: List[dict],
    interactions: List[dict],
    embedder: SentenceTransformer,
    similarity_threshold: float = 0.8,
) -> Optional[Data]:
    """
    Build a PyTorch Geometric Data object for a user's memory graph.

    Args:
        user_id: User identifier.
        memories: List of memory dictionaries (from MongoDB) for this user.
        interactions: List of interaction dictionaries (from InteractionLogger) for this user.
        embedder: SentenceTransformer model for computing embeddings.
        similarity_threshold: Cosine threshold for adding similarity edges.

    Returns:
        torch_geometric.Data object or None if insufficient data.
    """
    if not memories:
        return None

    # We'll need to map memory_id to index
    memory_id_to_idx = {mem["id"]: i for i, mem in enumerate(memories)}
    n_nodes = len(memories)

    # Prepare node features: [embedding(384), type_onehot(5), importance(1), log1p(turns_since_access)(1), log1p(access_count)(1)]
    feats = []
    for mem in memories:
        # Embedding
        emb = compute_embedding(mem["content"], embedder)  # shape (384,)
        # Type one-hot
        mtype = MemoryType(mem["memory_type"])
        type_onehot = memory_type_onehot(mtype)  # shape (5,)
        # Importance
        importance = np.array([mem["importance_score"]], dtype=np.float32)
        # Normalized age (log1p of turns_since_access)
        turns_since = mem.get("turns_since_access", 0)
        norm_age = np.array([np.log1p(turns_since)], dtype=np.float32)
        # Log access count
        access = mem.get("access_count", 0)
        log_access = np.array([np.log1p(access)], dtype=np.float32)

        feat = np.concatenate([emb, type_onehot, importance, norm_age, log_access])  # (384+5+1+1+1=392)
        feats.append(feat)
    x = np.stack(feats)  # (n_nodes, 392)
    x_tensor = torch.from_numpy(x).float()

    # Initialize edge lists
    edge_list = []

    # Temporal edges: based on interaction timeline.
    # We'll create a directed edge from earlier memory to later memory if they appear in the same interaction sequence.
    # For simplicity, we can use the interaction logs to order memories by access time.
    # We'll gather all timestamps when each memory was accessed (from interactions).
    memory_access_times = {mem["id"]: [] for mem in memories}
    for interaction in interactions:
        if interaction.get("interaction_type") == "turn":
            details = interaction.get("details", {})
            retrieved = details.get("memories_retrieved", [])
            for mem_ref in retrieved:
                mid = mem_ref.get("memory_id")
                if mid in memory_id_to_idx:
                    memory_access_times[mid].append(interaction["timestamp"])
        elif interaction.get("interaction_type") == "retrieval":
            details = interaction.get("details", {})
            results = details.get("results", [])
            for mem_ref in results:
                mid = mem_ref.get("memory_id")
                if mid in memory_id_to_idx:
                    memory_access_times[mid].append(interaction["timestamp"])

    # For each memory, compute the earliest access time (or use a default)
    # Then sort memories by earliest access time and add edges from earlier to later.
    mem_times = []
    for mem in memories:
        mid = mem["id"]
        times = memory_access_times.get(mid, [])
        if times:
            earliest = min(times)
        else:
            # If no interaction times, use a fallback: maybe the memory's creation time? Not stored.
            # We'll use a very old timestamp so they appear first.
            earliest = 0
        mem_times.append((mid, earliest))

    # Sort by earliest time
    mem_times.sort(key=lambda x: x[1])
    # Add edges from each memory to the next in chronological order
    for i in range(len(mem_times) - 1):
        idx_curr = memory_id_to_idx[mem_times[i][0]]
        idx_next = memory_id_to_idx[mem_times[i + 1][0]]
        edge_list.append([idx_curr, idx_next])  # directed edge earlier -> later

    # Optional: similarity edges based on embedding cosine similarity
    if similarity_threshold > 0:
        # Compute cosine similarity between all pairs (could be heavy for large n)
        # We'll do a simple O(n^2) for now; in production we might approximate.
        emb_matrix = x[:, :384]  # first 384 dims are embedding
        # Normalize rows to unit length
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        normed = emb_matrix / (norms + 1e-8)
        sim = np.dot(normed, normed.T)  # (n, n)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if sim[i, j] >= similarity_threshold:
                    # Add undirected edge (both directions)
                    edge_list.append([i, j])
                    edge_list.append([j, i])

    if not edge_list:
        # If no edges, create a self-loop to avoid empty edge_index (though GNN may complain)
        # Better to skip this user? We'll create a single self-loop on the first node.
        edge_list = [[0, 0]]

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # (2, E)

    # Prepare labels
    # Relevance label: 1 if memory was retrieved in the next N turns after its creation/access.
    # We'll approximate: for each memory, look at its access times and see if there was a retrieval
    # interaction within a time window (e.g., next 5 interactions?).
    # Since we don't have turn numbers, we'll use timestamp windows.
    # We'll define a horizon of, say, 1 hour (3600 seconds) after the memory's last access.
    # If there is a retrieval interaction (where the memory appears in results) within that horizon, label 1.
    # This is a simplification.
    relevance_labels = []
    horizon_seconds = 3600  # 1 hour
    for mem in memories:
        mid = mem["id"]
        times = memory_access_times.get(mid, [])
        if not times:
            relevance_labels.append(0)
            continue
        last_access = max(times)
        # Check if there is a retrieval interaction after last_access and before last_access + horizon
        found = False
        for interaction in interactions:
            if interaction.get("interaction_type") == "retrieval":
                details = interaction.get("details", {})
                results = details.get("results", [])
                for mem_ref in results:
                    if mem_ref.get("memory_id") == mid:
                        if (last_access < interaction["timestamp"] <= last_access + horizon_seconds):
                            found = True
                            break
                if found:
                    break
        relevance_labels.append(1 if found else 0)
    relevance_tensor = torch.tensor(relevance_labels, dtype=torch.float).unsqueeze(1)  # (N,1)

    # Cluster label: memory type (already known)
    cluster_labels = []
    for mem in memories:
        mtype = MemoryType(mem["memory_type"])
        cluster_labels.append(
            {
                MemoryType.PREFERENCE: 0,
                MemoryType.FACT: 1,
                MemoryType.EPISODE: 2,
                MemoryType.RULE: 3,
                MemoryType.PLANNING: 4,
            }[mtype]
        )
    cluster_tensor = torch.tensor(cluster_labels, dtype=torch.long)  # (N,)

    # Create Data object
    data = Data(
        x=x_tensor,
        edge_index=edge_index,
        y_relevance=relevance_tensor,
        y_cluster=cluster_tensor,
    )
    return data


async def train_gnn_model(
    mongo_url: str,
    qdrant_url: str,
    model_output: str,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = "cpu",
    lookahead_days: int = 30,
    similarity_threshold: float = 0.8,
):
    """
    Train GNN model offline using interaction logs.
    This function can be called programmatically.
    """
    # Import inside function to avoid top-level import issues if not needed
    import asyncio
    import numpy as np
    import random
    from datetime import datetime, timedelta, timezone
    from pathlib import Path
    import sys
    import torch
    from sentence_transformers import SentenceTransformer
    from torch_geometric.data import Data
    from typing import List, Dict

    # Add the project root to the path so we can import memory_engine
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    from memory_engine.interaction_logger import InteractionLogger
    from memory_engine.gnn_engine.processor import GraphProcessor
    from memory_engine.gnn_engine.model import MemoryGNN
    from memory_engine.gnn_engine.trainer import GNNTrainer
    from memory_engine.db import MemoryDB
    from memory_engine.models import MemoryType, MemoryStatus

    print(f"Starting offline GNN retraining...")
    print(f"Mongo URL: {mongo_url}")
    print(f"Qdrant URL: {qdrant_url}")
    print(f"Model output: {model_output}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    print(f"Lookback days: {lookahead_days}")
    print(f"Similarity threshold: {similarity_threshold}")

    # Initialize components
    interaction_logger = InteractionLogger(mongo_url)
    graph_processor = GraphProcessor(mongo_url, qdrant_url)
    db = MemoryDB(mongo_url)

    # Load embedding model (same as used in the engine)
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Determine cutoff time for logs
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookahead_days)
    cutoff_timestamp = cutoff.timestamp()

    # Fetch interactions (we'll get a lot and filter)
    print("Fetching interaction logs...")
    all_interactions = await interaction_logger.get_recent_interactions(limit=50000)  # adjust as needed
    # Filter by timestamp
    recent_interactions = [
        it for it in all_interactions if it.get("timestamp", 0) >= cutoff_timestamp
    ]
    print(f"Found {len(recent_interactions)} interactions in the last {lookahead_days} days.")

    # Group interactions by user_id
    user_interactions: Dict[str, List[dict]] = {}
    for it in recent_interactions:
        uid = it.get("user_id")
        if uid:
            user_interactions.setdefault(uid, []).append(it)

    # For each user, get their memories and build graph
    print("Building user graphs...")
    graph_data_list: List[Data] = []
    for user_id, interactions in user_interactions.items():
        # Get memories for this user (all statuses)
        # We'll use the MongoDB collection directly
        cursor = db._col.find({"user_id": user_id})
        memories = [doc async for doc in cursor]
        if not memories:
            continue
        print(f"  User {user_id}: {len(memories)} memories, {len(interactions)} interactions")
        data = build_user_graph(
            user_id=user_id,
            memories=memories,
            interactions=interactions,
            embedder=embedder,
            similarity_threshold=similarity_threshold,
        )
        if data is not None:
            graph_data_list.append(data)

    if not graph_data_list:
        print("No graph data generated. Exiting.")
        return

    print(f"Generated {len(graph_data_list)} user graphs.")

    # Split each user's data into train and validation (we'll do a simple split across graphs)
    # For simplicity, we'll split at graph level: 80% train, 20% val.
    # This avoids leakage across users.
    random.shuffle(graph_data_list)
    split_idx = int(0.8 * len(graph_data_list))
    train_data = graph_data_list[:split_idx]
    val_data = graph_data_list[split_idx:]
    print(f"Training graphs: {len(train_data)}, Validation graphs: {len(val_data)}")

    # Initialize model and trainer
    print("Initializing model and trainer...")
    model = MemoryGNN().to(device)
    trainer = GNNTrainer(model, device=device, learning_rate=learning_rate)

    # Training loop
    print("Starting training...")
    best_val_loss = float("inf")
    for epoch in range(epochs):
        # Train
        train_loss, train_rel_loss, train_cluster_loss = trainer.train_epoch(
            train_data, alpha=0.5
        )
        # Evaluate
        with torch.no_grad():
            val_loss, val_rel_loss, val_cluster_loss, val_rel_acc, val_cluster_acc = trainer.evaluate(
                val_data, alpha=0.5
            )
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} (rel: {train_rel_loss:.4f}, category: {train_cluster_loss:.4f}) | "
            f"Val Loss: {val_loss:.4f} (rel: {val_rel_loss:.4f}, cluster: {val_cluster_loss:.4f}) | "
            f"Rel Acc: {val_rel_acc:.4f}, Cluster Acc: {val_cluster_acc:.4f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_output)
            print(f"  -> New best model saved to {model_output}")

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {model_output}")


async def main():
    parser = argparse.ArgumentParser(description="Retrain GNN model offline using interaction logs.")
    parser.add_argument("--mongo-url", default=os.getenv("MONGO_URL", "mongodb://agent:agent@mongo:27017/memories?authSource=admin"),
                        help="MongoDB connection string")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://qdrant:6333"),
                        help="Qdrant connection string")
    parser.add_argument("--model-output", default=os.getenv("GNN_MODEL_PATH", "gnn_model.pt"),
                        help="Path to save the retrained model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--lookahead-days", type=int, default=30, help="How many days of logs to look back")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Cosine similarity threshold for adding edges")
    args = parser.parse_args()

    await train_gnn_model(
        mongo_url=args.mongo_url,
        qdrant_url=args.qdrant_url,
        model_output=args.model_output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        lookahead_days=args.lookahead_days,
        similarity_threshold=args.similarity_threshold,
    )


if __name__ == "__main__":
    asyncio.run(main())