"""
embeddings/encoder.py

Embedding generation and Qdrant vector storage.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


COLLECTION_NAME = "memories"
EMBEDDING_DIM   = 384   # all-MiniLM-L6-v2 output dim


class EmbeddingEngine:
    def __init__(self, qdrant_url: str, model_name: str = "all-MiniLM-L6-v2"):
        self._model  = SentenceTransformer(model_name)
        self._qdrant = AsyncQdrantClient(url=qdrant_url)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def setup_collection(self):
        """Create Qdrant collection if it doesn't exist. Call once on startup."""
        existing = await self._qdrant.get_collections()
        names = [c.name for c in existing.collections]
        if COLLECTION_NAME not in names:
            await self._qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def store(self, memory_id: str, text: str, payload: dict) -> str:
        """
        Encode text and upsert into Qdrant.
        Returns the memory_id used as the Qdrant point id (via payload lookup).
        Qdrant point id is the memory_id string stored in payload.
        """
        vector = self.encode(text)
        point  = PointStruct(
            id      = abs(hash(memory_id)) % (2**63),  # Qdrant needs uint64
            vector  = vector,
            payload = {"memory_id": memory_id, **payload},
        )
        await self._qdrant.upsert(collection_name=COLLECTION_NAME, points=[point])
        return memory_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Returns list of {memory_id, score, payload} dicts.
        """
        vector  = self.encode(query)
        results = await self._qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {"memory_id": r.payload["memory_id"], "score": r.score, "payload": r.payload}
            for r in results
        ]