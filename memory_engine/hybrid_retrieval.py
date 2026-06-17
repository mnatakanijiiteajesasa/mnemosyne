"""
memory_engine/hybrid_retrieval.py

Hybrid retrieval system for Phase 6.
Combines cosine similarity, GNN relevance, recency, and cluster boosts.
"""

from __future__ import annotations
import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch

from .embeddings.encoder import EmbeddingEngine
from .gnn_engine.inference import GNNInferenceEngine, GNNRetrievalScorer
from .gnn_engine.processor import GraphProcessor
from .models import MemoryType
from .db import MemoryDB


@dataclass
class RetrievalCacheEntry:
    """Cache entry for retrieval results."""
    results: List[Dict[str, Any]]
    timestamp: float
    ttl: float = 300.0  # 5 minutes default TTL

    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class LRUCache:
    """Simple LRU cache for retrieval results."""

    def __init__(self, capacity: int = 1000):
        self._cache: OrderedDict[str, RetrievalCacheEntry] = OrderedDict()
        self._capacity = capacity

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        if key in self._cache:
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.results
        return None

    def put(self, key: str, value: List[Dict[str, Any]], ttl: float = 300.0):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._capacity:
                self._cache.popitem(last=False)  # Remove least recently used

        self._cache[key] = RetrievalCacheEntry(
            results=value,
            timestamp=time.time(),
            ttl=ttl
        )

    def clear(self):
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class HybridRetrievalEngine:
    """
    Hybrid retrieval engine that combines vector similarity with GNN scoring.

    Features:
    1. Two-stage retrieval: vector search → GNN re-ranking
    2. Hybrid scoring: relevance + similarity + recency + cluster boosts
    3. Caching for frequent queries
    4. Graceful fallback to similarity-only when GNN unavailable
    5. Async processing support
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        graph_processor: GraphProcessor,
        memory_db: MemoryDB,
        model_path: Optional[str] = None,
        device: str = "cpu",
        cache_size: int = 1000,
        enable_cache: bool = True,
        vector_candidates_multiplier: int = 4,  # Get 4x more candidates for re-ranking
        alpha: float = 0.6,  # Weight for GNN relevance vs vector similarity
    ):
        self.embedding_engine = embedding_engine
        self.graph_processor = graph_processor
        self.memory_db = memory_db
        self.device = device
        self.enable_cache = enable_cache
        self.vector_candidates_multiplier = vector_candidates_multiplier
        self.alpha = alpha  # Weight for combining GNN and vector scores

        # Initialize cache
        self.cache = LRUCache(capacity=cache_size) if enable_cache else None

        # Initialize GNN inference engine (may be None if no model)
        self.gnn_engine: Optional[GNNInferenceEngine] = None
        self.gnn_scorer: Optional[GNNRetrievalScorer] = None

        try:
            self.gnn_engine = GNNInferenceEngine(
                model_path=model_path,
                device=device,
                processor=graph_processor
            )
            self.gnn_scorer = GNNRetrievalScorer(self.gnn_engine)
            print(f"HybridRetrievalEngine: GNN model loaded from {model_path or 'random init'}")
        except Exception as e:
            print(f"HybridRetrievalEngine: Failed to load GNN model: {e}")
            print("Falling back to similarity-only retrieval")

    def _compute_cache_key(
        self,
        query: str,
        user_id: str,
        top_k: int,
        **kwargs
    ) -> str:
        """Compute cache key for retrieval request."""
        key_data = f"{query}:{user_id}:{top_k}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for memories using hybrid retrieval.
        Implements dynamic confidence gating based on real memory count.

        Args:
            query: Search query text
            user_id: User ID to filter memories
            top_k: Number of results to return
            use_hybrid: Whether to use hybrid scoring (if GNN available)
            **kwargs: Additional arguments passed to embedding engine

        Returns:
            List of memory dicts with scores and metadata
        """
        # Check cache first
        if self.enable_cache and self.cache:
            cache_key = self._compute_cache_key(query, user_id, top_k, use_hybrid=use_hybrid, **kwargs)
            cached_results = self.cache.get(cache_key)
            if cached_results is not None:
                return cached_results[:top_k]  # Return cached top-k

        # Determine how many candidates to retrieve for re-ranking
        candidate_k = min(
            top_k * self.vector_candidates_multiplier,
            100  # Reasonable upper bound
        ) if use_hybrid and self.gnn_engine else top_k

        # Stage 1: Vector similarity search
        vector_results = await self.embedding_engine.search(
            query=query,
            top_k=candidate_k,
            user_id=user_id,
            **kwargs
        )

        # If no results or hybrid not requested/available, return vector results
        if not vector_results or not use_hybrid or not self.gnn_engine:
            final_results = vector_results[:top_k]
            # Cache results
            if self.enable_cache and self.cache:
                self.cache.put(cache_key, vector_results)
            return final_results

        # Calculate dynamic alpha based on real memory count
        # α = min(1.0, real_memory_count / threshold)
        # Where threshold is tunable (default 15)
        real_memory_count = await self._count_real_memories(user_id)
        threshold = 15  # Tunable constant - memories needed for full GNN trust
        dynamic_alpha = min(1.0, real_memory_count / threshold)

        # Stage 2: GNN re-ranking
        try:
            # Build user graph for GNN scoring
            graph_data = await self.graph_processor.build_graph(user_id, device=self.device)

            # Extract memory IDs from vector results
            memory_ids = [r["memory_id"] for r in vector_results]

            # Get GNN scores for these memories
            gnn_scores = await self._get_gnn_scores_for_memory_ids(
                user_id, memory_ids, graph_data
            )

            # Combine scores and re-rank with dynamic alpha
            hybrid_results = self._compute_hybrid_scores(
                vector_results, gnn_scores, dynamic_alpha
            )

            # Take top-k
            final_results = hybrid_results[:top_k]

        except Exception as e:
            print(f"Hybrid retrieval failed: {e}; falling back to vector similarity")
            final_results = vector_results[:top_k]

        # Cache results
        if self.enable_cache and self.cache:
            self.cache.put(cache_key, vector_results)  # Cache the broader vector results

        return final_results

    async def _count_real_memories(self, user_id: str) -> int:
        """
        Count non-seed (real) memories for a user.
        Seed memories are identified by having 'is_seed:true' in tags.
        """
        try:
            # Get all active memories for the user
            all_memories = await self.db.get_active(user_id)

            # Count memories that are NOT seed memories
            real_count = 0
            for memory in all_memories:
                # Convert MemoryRecord to dict for tag checking
                memory_dict = memory.dict() if hasattr(memory, 'dict') else dict(memory)
                seed_confidence = self._get_seed_confidence(memory_dict)
                # If seed confidence is 1.0 (default), it's not a seed memory
                if seed_confidence >= 1.0:
                    real_count += 1

            return real_count
        except Exception as e:
            print(f"Error counting real memories for user {user_id}: {e}")
            return 0  # Fail safe - assume no real memories

    async def _get_gnn_scores_for_memory_ids(
        self,
        user_id: str,
        memory_ids: List[str],
        graph_data
    ) -> Dict[str, float]:
        """
        Get GNN relevance scores for a specific set of memory IDs.

        Returns:
            Dict mapping memory_id to relevance score (0-1)
        """
        if not self.gnn_engine or not memory_ids:
            return {mid: 0.5 for mid in memory_ids}

        try:
            # Run GNN forward pass
            with torch.no_grad():
                h, r, c = self.gnn_engine.model(graph_data.x, graph_data.edge_index)

                # Extract scores
                relevance_scores = r.squeeze(-1).cpu().numpy()  # [N]
                cluster_logits = c  # [N, num_clusters]

                # Create mapping from memory_id to index
                memory_id_to_idx = {
                    mid: i for i, mid in enumerate(graph_data.memory_ids)
                }

                # Get scores for requested memory IDs
                scores = {}
                for mid in memory_ids:
                    if mid in memory_id_to_idx:
                        idx = memory_id_to_idx[mid]
                        relevance = float(relevance_scores[idx])
                        # Cluster confidence (max softmax probability)
                        cluster_probs = torch.softmax(cluster_logits[idx], dim=0)
                        cluster_conf = float(cluster_probs.max())
                        # Combined GNN score: relevance weighted by cluster confidence
                        gnn_score = 0.7 * relevance + 0.3 * cluster_conf
                        scores[mid] = max(0.0, min(1.0, gnn_score))
                    else:
                        # Memory not in graph (shouldn't happen, but be safe)
                        scores[mid] = 0.5

                return scores

        except Exception as e:
            print(f"Error computing GNN scores: {e}")
            return {mid: 0.5 for mid in memory_ids}

    def _compute_hybrid_scores(
        self,
        vector_results: List[Dict[str, Any]],
        gnn_scores: Dict[str, float],
        alpha: float
    ) -> List[Dict[str, Any]]:
        """
        Compute hybrid scores combining vector similarity and GNN relevance.
        Implements confidence gating for seed memories.

        Args:
            vector_results: Results from vector search with 'score' field
            gnn_scores: Dict mapping memory_id to GNN relevance score
            alpha: Base weight for GNN score (will be adjusted for seed memories)

        Returns:
            Results sorted by hybrid score (descending)
        """
        for result in vector_results:
            mid = result["memory_id"]

            # Vector similarity score (from Qdrant, already 0-1)
            vector_score = result.get("score", 0.5)

            # GNN relevance score
            gnn_score = gnn_scores.get(mid, 0.5)

            # Check if this is a seed memory and get seed confidence
            seed_confidence = self._get_seed_confidence(result)
            is_seed_memory = seed_confidence > 0

            # Apply seed confidence to GNN score (seeds have lower confidence)
            adjusted_gnn_score = gnn_score * seed_confidence if is_seed_memory else gnn_score

            # Hybrid score: weighted combination
            hybrid_score = alpha * adjusted_gnn_score + (1 - alpha) * vector_score

            # Store all scores for transparency/debugging
            result["vector_score"] = vector_score
            result["gnn_score"] = gnn_score
            result["adjusted_gnn_score"] = adjusted_gnn_score
            result["seed_confidence"] = seed_confidence
            result["is_seed_memory"] = is_seed_memory
            result["hybrid_score"] = hybrid_score

            # Use hybrid score as the main score for compatibility
            result["score"] = hybrid_score

        # Sort by hybrid score (descending)
        vector_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return vector_results

    def _get_seed_confidence(self, memory_result: Dict[str, Any]) -> float:
        """
        Extract seed confidence from memory result tags.
        Returns 1.0 for non-seed memories, 0.0-1.0 for seed memories.
        """
        tags = memory_result.get("tags", [])
        seed_confidence = 1.0  # Default for non-seed memories

        for tag in tags:
            if tag.startswith("seed_confidence:"):
                try:
                    confidence_str = tag.split(":", 1)[1]
                    seed_confidence = float(confidence_str)
                    # Ensure it's in valid range
                    seed_confidence = max(0.0, min(1.0, seed_confidence))
                    break
                except (ValueError, IndexError):
                    # If parsing fails, treat as non-seed
                    seed_confidence = 1.0
                    break

        return seed_confidence

    async def search_with_recency_boost(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
        recency_weight: float = 0.1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search with additional recency boosting based on turns_since_access.

        Args:
            query: Search query text
            user_id: User ID
            top_k: Number of results
            recency_weight: Weight for recency boost (0-1)
            **kwargs: Additional arguments

        Returns:
            Recency-boosted search results
        """
        # Get hybrid results first
        results = await self.search(
            query=query,
            user_id=user_id,
            top_k=min(top_k * 2, 20),  # Get more candidates for re-ranking
            **kwargs
        )

        if not results or recency_weight <= 0:
            return results[:top_k]

        # Fetch memory details to get turns_since_access
        memory_ids = [r["memory_id"] for r in results]
        # Get memories from database
        memories = []
        for mid in memory_ids:
            memory = await self.memory_db.get(mid)
            if memory:
                memories.append(memory)

        # Create mapping for quick lookup
        memory_details = {m.id: m for m in memories} if memories else {}

        # Apply recency boost
        max_turns = max(
            [getattr(memory_details.get(mid), 'turns_since_access', 0) for mid in memory_ids] + [1]
        )

        for result in results:
            mid = result["memory_id"]
            memory = memory_details.get(mid)

            if memory and hasattr(memory, 'turns_since_access'):
                turns_since_access = getattr(memory, 'turns_since_access', 0)
                # Normalize recency: lower turns_since_access = higher score
                recency_score = 1.0 - (turns_since_access / (max_turns + 1))
                recency_score = max(0.0, min(1.0, recency_score))

                # Combine with existing hybrid score
                original_score = result["score"]
                boosted_score = (
                    (1 - recency_weight) * original_score +
                    recency_weight * recency_score
                )
                result["score"] = boosted_score
                result["recency_score"] = recency_score
                result["turns_since_access"] = turns_since_access

        # Re-sort by boosted score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.enable_cache or not self.cache:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": self.cache.size(),
            "capacity": self.cache._capacity,
        }

    def clear_cache(self):
        """Clear the retrieval cache."""
        if self.enable_cache and self.cache:
            self.cache.clear()


# Helper function to create HybridRetrievalEngine from API components
def create_hybrid_retrieval_engine(
    mongo_url: str,
    qdrant_url: str,
    model_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> HybridRetrievalEngine:
    """
    Factory function to create HybridRetrievalEngine with standard components.

    Args:
        mongo_url: MongoDB connection string
        qdrant_url: Qdrant connection string
        model_path: Path to GNN model checkpoint
        device: torch device ('cpu' or 'cuda')
        **kwargs: Additional arguments for HybridRetrievalEngine

    Returns:
        Configured HybridRetrievalEngine instance
    """
    # Initialize components
    embedding_engine = EmbeddingEngine(qdrant_url=qdrant_url)
    graph_processor = GraphProcessor(
        mongo_url=mongo_url,
        qdrant_url=qdrant_url
    )
    memory_db = MemoryDB(mongo_url=mongo_url)

    # Create hybrid engine
    return HybridRetrievalEngine(
        embedding_engine=embedding_engine,
        graph_processor=graph_processor,
        memory_db=memory_db,
        model_path=model_path,
        device=device,
        **kwargs
    )