"""
gnn_engine/processor.py

Convert MongoDB memory records + Qdrant embeddings + edges
into PyTorch Geometric Data objects for GNN training/inference.

Key responsibility:
  1. Fetch all memories for a user from MongoDB
  2. Get embeddings from Qdrant
  3. Build node feature matrix (391-dim per memory)
  4. Get adjacency from memory_edges collection
  5. Construct torch_geometric.Data object
"""

from __future__ import annotations
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from memory_engine.models import MemoryType


# Memory type to one-hot index
MEMORY_TYPE_MAP = {
    MemoryType.PREFERENCE: 0,
    MemoryType.FACT:       1,
    MemoryType.EPISODE:    2,
    MemoryType.RULE:       3,
}

COLLECTION_NAME = "memories"


class GraphProcessor:
    """
    Build PyTorch Geometric graphs from memory database + embeddings.
    """

    def __init__(self, mongo_url: str, qdrant_url: str, db_name: str = "memories"):
        self._mongo_client = AsyncIOMotorClient(mongo_url)
        self._db = self._mongo_client[db_name]
        self._memories_col = self._db["memories"]
        self._edges_col = self._db["memory_edges"]
        self._qdrant = AsyncQdrantClient(url=qdrant_url)

    async def build_graph(self, user_id: str, device: str = "cpu") -> Data:
        """
        Build a complete graph for a user.

        Args:
            user_id: User ID
            device: torch device ('cpu' or 'cuda')

        Returns:
            torch_geometric.Data object with:
              - x: node features [N, 391]
              - edge_index: edge list [2, E]
              - memory_ids: list of memory IDs for reference
        """
        # 1. Fetch all active memories for this user
        memories = await self._fetch_user_memories(user_id)
        if not memories:
            raise ValueError(f"No memories found for user {user_id}")

        memory_ids = [m["_id"] for m in memories]
        memory_id_to_idx = {mid: i for i, mid in enumerate(memory_ids)}

        # 2. Build node feature matrix
        x = await self._build_node_features(memories)
        x = torch.tensor(x, dtype=torch.float32, device=device)

        # 3. Build edge_index from adjacency
        edge_index = await self._build_edge_index(memory_ids, memory_id_to_idx, device)

        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
        )
        data.memory_ids = memory_ids  # Attach for tracing back

        return data

    async def _fetch_user_memories(self, user_id: str) -> list[dict]:
        """Fetch all active (not archived) memories for a user."""
        cursor = self._memories_col.find({
            "user_id": user_id,
            "status": "active",
        })
        return [doc async for doc in cursor]

    async def _build_node_features(self, memories: list[dict]) -> np.ndarray:
        """
        Build 391-dim feature vector for each memory.

        Dim breakdown:
          [0:384]   - embedding vector
          [384:388] - one-hot memory type (4-dim)
          [388]     - importance (normalized 0-1)
          [389]     - age_turns (normalized 0-1 with log scale)
          [390]     - access_count (normalized 0-1 with log scale)
        """
        N = len(memories)
        features = np.zeros((N, 391), dtype=np.float32)

        # Collect memory_ids to batch fetch embeddings
        memory_ids = [m["_id"] for m in memories]

        # Fetch embeddings from Qdrant
        embeddings = await self._fetch_embeddings(memory_ids)
        embedding_dict = {mid: emb for mid, emb in zip(memory_ids, embeddings)}

        # Normalize for log-scale features
        max_age = max([m.get("turns_since_access", 1) for m in memories] + [1])
        max_access = max([m.get("access_count", 1) for m in memories] + [1])

        for i, mem in enumerate(memories):
            mid = mem["_id"]

            # Embedding [0:384]
            emb = embedding_dict.get(mid)
            if emb is not None:
                features[i, 0:384] = emb
            else:
                # Fallback: zero vector if embedding missing
                features[i, 0:384] = 0.0

            # Memory type one-hot [384:388]
            mtype = MemoryType(mem.get("memory_type", "fact"))
            type_idx = MEMORY_TYPE_MAP[mtype]
            features[i, 384 + type_idx] = 1.0

            # Importance [388] - already in 0-1 range
            importance = mem.get("importance_score", 0.5)
            features[i, 388] = min(1.0, max(0.0, importance))

            # Age turns [389] - normalize with log scale
            age = mem.get("turns_since_access", 0)
            normalized_age = np.log1p(age) / np.log1p(max_age + 1)
            features[i, 389] = normalized_age

            # Access count [390] - normalize with log scale
            access_count = mem.get("access_count", 0)
            normalized_access = np.log1p(access_count) / np.log1p(max_access + 1)
            features[i, 390] = normalized_access

        return features

    async def _fetch_embeddings(self, memory_ids: list[str]) -> list[np.ndarray]:
        """
        Fetch embedding vectors from Qdrant for a list of memory_ids.

        Qdrant stores points with:
          - id: hash of memory_id
          - payload: {memory_id, user_id, ...}
          - vector: 384-dim embedding
        """
        embeddings = []

        for mid in memory_ids:
            try:
                # Search by memory_id in payload
                results = await self._qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_filter=Filter(
                        must=[FieldCondition(
                            key="memory_id",
                            match=MatchValue(value=mid)
                        )]
                    ),
                    query_vector=[0.0] * 384,  # dummy query
                    limit=1,
                    with_vectors=True,
                )

                if results:
                    emb = np.array(results[0].vector, dtype=np.float32)
                    embeddings.append(emb)
                else:
                    # Missing embedding
                    embeddings.append(np.zeros(384, dtype=np.float32))

            except Exception as e:
                print(f"Error fetching embedding for {mid}: {e}")
                embeddings.append(np.zeros(384, dtype=np.float32))

        return embeddings

    async def _build_edge_index(self, memory_ids: list[str],
                                 memory_id_to_idx: dict[str, int],
                                 device: str = "cpu") -> torch.Tensor:
        """
        Build edge_index tensor from memory_edges collection.

        Format: [2, E] where each column is [source_idx, target_idx]
        """
        edges = []

        cursor = self._edges_col.find({
            "$or": [
                {"source_id": {"$in": memory_ids}},
                {"target_id": {"$in": memory_ids}},
            ]
        })

        async for doc in cursor:
            src_id = doc.get("source_id")
            tgt_id = doc.get("target_id")

            if src_id in memory_id_to_idx and tgt_id in memory_id_to_idx:
                src_idx = memory_id_to_idx[src_id]
                tgt_idx = memory_id_to_idx[tgt_id]

                # Add edge in both directions (undirected graph)
                edges.append([src_idx, tgt_idx])
                edges.append([tgt_idx, src_idx])

        if not edges:
            # If no edges, create self-loops for isolated nodes
            N = len(memory_ids)
            edges = [[i, i] for i in range(N)]

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    async def build_training_batch(self, user_id: str, window_size: int = 5,
                                    device: str = "cpu") -> Optional[Data]:
        """
        Build a training batch with labels.

        Labels:
          - y_relevance: Did this memory get accessed in the next `window_size` turns?
          - y_cluster: Memory type (0-3)

        Note: Requires access_count history (Phase 8 feature).
        Currently uses proxy: accessed in last turn → label=1
        """
        data = await self.build_graph(user_id, device=device)

        N = len(data.memory_ids)

        # TODO: Fetch actual access history from revision logs
        # For now, use simple heuristic: high access_count → high relevance
        relevance_labels = torch.zeros(N, dtype=torch.long, device=device)

        cluster_labels = torch.zeros(N, dtype=torch.long, device=device)
        memories = await self._fetch_user_memories(user_id)
        memory_id_to_mem = {m["_id"]: m for m in memories}

        for i, mid in enumerate(data.memory_ids):
            mem = memory_id_to_mem[mid]

            # Relevance heuristic: access_count > 2 or importance_score > 0.7
            access_count = mem.get("access_count", 0)
            importance = mem.get("importance_score", 0.5)
            if access_count > 2 or importance > 0.7:
                relevance_labels[i] = 1

            # Cluster: memory type
            mtype = MemoryType(mem.get("memory_type", "fact"))
            cluster_labels[i] = MEMORY_TYPE_MAP[mtype]

        data.y_relevance = relevance_labels
        data.y_cluster = cluster_labels

        return data