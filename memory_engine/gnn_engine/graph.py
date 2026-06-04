"""
gnn_engine/graph.py

Memory graph construction.

For each new memory written, find existing memories with cosine
similarity >= threshold and create edges between them.
Edges are stored in MongoDB. The adjacency structure is used by
the GNN in Phase 5.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

EDGE_THRESHOLD  = 0.75   # minimum cosine similarity to draw an edge
COLLECTION_NAME = "memories"


@dataclass
class MemoryEdge:
    source_id:  str
    target_id:  str
    weight:     float
    created_at: datetime = field(default_factory=datetime.utcnow)

    def dict(self):
        return {
            "source_id":  self.source_id,
            "target_id":  self.target_id,
            "weight":     self.weight,
            "created_at": self.created_at.isoformat(),
        }


class GraphBuilder:
    def __init__(self, mongo_url: str, qdrant_url: str, db_name: str = "memories"):
        self._client  = AsyncIOMotorClient(mongo_url)
        self._edges   = self._client[db_name]["memory_edges"]
        self._qdrant  = AsyncQdrantClient(url=qdrant_url)

    async def setup_indexes(self):
        await self._edges.create_index([("source_id", ASCENDING)])
        await self._edges.create_index([("target_id", ASCENDING)])
        await self._edges.create_index(
            [("source_id", ASCENDING), ("target_id", ASCENDING)], unique=True
        )

    async def build_edges(self, memory_id: str, user_id: str, top_k: int = 20):
        """
        Find the top_k most similar memories for a given memory_id
        within the same user and create edges where similarity >= threshold.
        """
        # Get the vector for this memory from Qdrant
        results = await self._qdrant.search(
            collection_name = COLLECTION_NAME,
            query_filter    = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            query_vector    = await self._get_vector(memory_id),
            limit           = top_k + 1,  # +1 because it will match itself
            with_payload    = True,
        )

        edges_created = []
        for result in results:
            candidate_id = result.payload.get("memory_id")
            if candidate_id == memory_id:
                continue
            if result.score < EDGE_THRESHOLD:
                continue

            edge = MemoryEdge(
                source_id = memory_id,
                target_id = candidate_id,
                weight    = result.score,
            )

            # Upsert — ignore if edge already exists
            await self._edges.update_one(
                {"source_id": memory_id, "target_id": candidate_id},
                {"$setOnInsert": edge.dict()},
                upsert=True,
            )
            edges_created.append(candidate_id)

        return edges_created

    async def get_neighbours(self, memory_id: str) -> list[dict]:
        """Return all edges where memory_id is source or target."""
        cursor = self._edges.find({
            "$or": [{"source_id": memory_id}, {"target_id": memory_id}]
        })
        return [doc async for doc in cursor]

    async def get_adjacency(self, user_memory_ids: list[str]) -> dict[str, list[tuple[str, float]]]:
        """
        Build adjacency dict for a set of memory ids.
        Returns {memory_id: [(neighbour_id, weight), ...]}
        """
        id_set = set(user_memory_ids)
        cursor = self._edges.find({
            "$or": [
                {"source_id": {"$in": user_memory_ids}},
                {"target_id": {"$in": user_memory_ids}},
            ]
        })
        adj: dict[str, list[tuple[str, float]]] = {mid: [] for mid in user_memory_ids}
        async for doc in cursor:
            s, t, w = doc["source_id"], doc["target_id"], doc["weight"]
            if s in id_set:
                adj[s].append((t, w))
            if t in id_set:
                adj[t].append((s, w))
        return adj

    async def _get_vector(self, memory_id: str) -> list[float]:
        """Retrieve the vector for a memory_id from Qdrant."""
        point_id = abs(hash(memory_id)) % (2**63)
        results  = await self._qdrant.retrieve(
            collection_name = COLLECTION_NAME,
            ids             = [point_id],
            with_vectors    = True,
        )
        if not results:
            raise ValueError(f"No vector found for memory_id {memory_id}")
        return results[0].vector