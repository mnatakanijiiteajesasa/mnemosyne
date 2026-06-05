"""
gnn_engine/inference.py

GNN inference pipeline for scoring memories during retrieval.

Steps:
  1. Build graph for user's memory
  2. Forward pass through GNN to get relevance scores + enriched embeddings
  3. Rank memories by hybrid score (relevance + similarity + recency)
  4. Return top-K with enriched embeddings for improved downstream retrieval
"""

from __future__ import annotations
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from memory_engine.models import MemoryType
from .model import MemoryGNN, OUTPUT_DIM
from .processor import GraphProcessor, MEMORY_TYPE_MAP


class GNNInferenceEngine:
    """
    Inference wrapper for memory ranking using GNN.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 processor: Optional[GraphProcessor] = None):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: torch device
            processor: GraphProcessor instance for building graphs
        """
        self.device = device
        self.model = MemoryGNN().to(device)

        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model from {model_path}")
        else:
            print("No model checkpoint found; using random initialization")

        self.model.eval()
        self.processor = processor

    @torch.no_grad()
    async def score_memories(self, user_id: str, query_embedding: np.ndarray,
                              top_k: int = 5) -> list[dict]:
        """
        Score and rank memories for a user using GNN + hybrid scoring.

        Args:
            user_id: User ID
            query_embedding: Query embedding [384-dim]
            top_k: Number of memories to return

        Returns:
            List of dicts: {
              memory_id,
              relevance_score,
              similarity_score,
              hybrid_score,
              enriched_embedding,
              ...
            }
        """
        if not self.processor:
            raise RuntimeError("GraphProcessor not set in InferenceEngine")

        # 1. Build graph for user
        try:
            data = await self.processor.build_graph(user_id, device=self.device)
        except ValueError:
            # No memories yet
            return []

        # 2. Forward pass
        h, r, c = self.model(data.x, data.edge_index)

        # h: enriched embeddings [N, 128]
        # r: relevance scores [N, 1]
        # c: cluster logits [N, 4]

        # 3. Hybrid scoring
        scores = []
        query_emb_tensor = torch.tensor(query_embedding, dtype=torch.float32, device=self.device)

        for i, memory_id in enumerate(data.memory_ids):
            # Relevance score from GNN
            relevance = r[i].item()

            # Similarity: cosine between enriched embedding and query
            enriched_emb = h[i].detach().cpu().numpy()
            similarity = np.dot(enriched_emb, query_embedding) / (
                np.linalg.norm(enriched_emb) * np.linalg.norm(query_embedding) + 1e-8
            )
            similarity = max(0.0, (similarity + 1) / 2)  # Normalize to [0, 1]

            # Cluster confidence (max softmax prob)
            cluster_probs = torch.softmax(c[i], dim=0)
            cluster_conf = cluster_probs.max().item()

            # Hybrid score: weighted combination
            # Relevance is the primary signal (trained on data)
            # Similarity and confidence are secondary
            hybrid = 0.5 * relevance + 0.3 * similarity + 0.2 * cluster_conf

            scores.append({
                "memory_id": memory_id,
                "relevance_score": float(relevance),
                "similarity_score": float(similarity),
                "cluster_confidence": float(cluster_conf),
                "hybrid_score": float(hybrid),
                "enriched_embedding": enriched_emb,
            })

        # 4. Rank by hybrid score
        scores.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return scores[:top_k]

    @torch.no_grad()
    async def get_enriched_embeddings(self, user_id: str) -> dict[str, np.ndarray]:
        """
        Get enriched 128-dim embeddings for all memories of a user.
        Useful for caching after model update.

        Returns:
            {memory_id: enriched_embedding [128-dim]}
        """
        if not self.processor:
            raise RuntimeError("GraphProcessor not set in InferenceEngine")

        try:
            data = await self.processor.build_graph(user_id, device=self.device)
        except ValueError:
            return {}

        h, _, _ = self.model(data.x, data.edge_index)

        embeddings = {}
        for i, memory_id in enumerate(data.memory_ids):
            embeddings[memory_id] = h[i].detach().cpu().numpy()

        return embeddings

    async def batch_score(self, user_ids: list[str],
                           query_embedding: np.ndarray,
                           top_k: int = 5) -> dict[str, list[dict]]:
        """
        Score memories for multiple users in parallel.

        Returns:
            {user_id: [(scored_memories)]}
        """
        results = {}
        for user_id in user_ids:
            try:
                results[user_id] = await self.score_memories(
                    user_id, query_embedding, top_k
                )
            except Exception as e:
                print(f"Error scoring for user {user_id}: {e}")
                results[user_id] = []

        return results


class GNNRetrievalScorer:
    """
    Standalone scorer that integrates GNN scores with existing
    vector retrieval results for hybrid ranking.

    Usage:
      1. Get cosine similarity results from Qdrant
      2. Pass through GNN to get relevance + enriched embeddings
      3. Combine scores and re-rank
    """

    def __init__(self, inference_engine: GNNInferenceEngine):
        self.engine = inference_engine

    async def rerank_with_gnn(self, user_id: str, retrieval_results: list[dict],
                               alpha: float = 0.6) -> list[dict]:
        """
        Re-rank retrieval results using GNN scores.

        Args:
            user_id: User ID
            retrieval_results: Results from Qdrant with similarity scores
            alpha: Weight for GNN relevance (vs. retrieval similarity)

        Returns:
            Re-ranked results with combined scores
        """
        if not retrieval_results:
            return []

        # Extract memory IDs
        memory_ids = [r["memory_id"] for r in retrieval_results]

        # Get GNN scores
        try:
            gnn_scores = await self._get_gnn_scores_for_ids(user_id, memory_ids)
        except Exception as e:
            print(f"GNN scoring failed: {e}; falling back to retrieval scores")
            return retrieval_results

        # Combine scores
        for result in retrieval_results:
            mid = result["memory_id"]
            retrieval_score = result["payload"].get("score", 0.5)
            gnn_relevance = gnn_scores.get(mid, 0.5)

            # Weighted combination
            combined = alpha * gnn_relevance + (1 - alpha) * retrieval_score
            result["combined_score"] = combined

        # Re-rank
        retrieval_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)

        return retrieval_results

    async def _get_gnn_scores_for_ids(self, user_id: str,
                                       memory_ids: list[str]) -> dict[str, float]:
        """
        Get GNN relevance scores for a specific set of memory IDs.
        """
        if not self.engine.processor:
            return {mid: 0.5 for mid in memory_ids}

        try:
            data = await self.engine.processor.build_graph(user_id, device=self.engine.device)
        except ValueError:
            return {mid: 0.5 for mid in memory_ids}

        h, r, c = self.engine.model(data.x, data.edge_index)

        scores = {}
        for i, memory_id in enumerate(data.memory_ids):
            if memory_id in memory_ids:
                scores[memory_id] = r[i].item()

        # Default for missing IDs
        for mid in memory_ids:
            if mid not in scores:
                scores[mid] = 0.5

        return scores