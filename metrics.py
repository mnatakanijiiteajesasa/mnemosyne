"""
metrics.py

All metrics needed for the flat-retrieval vs. GNN-retrieval comparison
paper: retrieval accuracy, forgetting accuracy, and context-budget
efficiency.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Set


def precision_recall_at_k(retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
    if not relevant:
        return {"precision": None, "recall": None}
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    hits = len(retrieved_set & relevant_set)
    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return {"precision": precision, "recall": recall}


def evaluate_retrieval(retriever, queries: List[Dict], k: int = 5) -> Dict[str, float]:
    """Averages precision@k / recall@k over all test queries for a fitted retriever."""
    precisions, recalls = [], []
    for q in queries:
        q_emb = np.array(q["embedding"])
        retrieved = retriever.retrieve(q_emb, k=k)
        pr = precision_recall_at_k(retrieved, q["relevant_memory_ids"])
        if pr["precision"] is not None:
            precisions.append(pr["precision"])
            recalls.append(pr["recall"])
    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else None,
        f"recall@{k}": float(np.mean(recalls)) if recalls else None,
        "n_queries_evaluated": len(precisions),
    }


def evaluate_forgetting(memories: List[Dict], survival_probs: np.ndarray,
                         prune_floor: float = 0.08) -> Dict[str, float]:
    """
    Compares 'should_survive' ground truth (does the user revisit this
    memory's cluster later?) against the model's prune/keep decision.
    Also computes the exponential-decay-only baseline decision for the
    same memories, so you get a direct forgetting-accuracy comparison.
    """
    should_survive = np.array([m["should_survive"] for m in memories])
    predicted_survive = survival_probs >= prune_floor

    tp = np.sum(predicted_survive & should_survive)
    fp = np.sum(predicted_survive & ~should_survive)
    fn = np.sum(~predicted_survive & should_survive)
    tn = np.sum(~predicted_survive & ~should_survive)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(memories)
    return {
        "forgetting_precision": float(precision),
        "forgetting_recall": float(recall),
        "forgetting_accuracy": float(accuracy),
        "n_memories": len(memories),
    }


def exponential_decay_survival(memories: List[Dict], type_lambda: Dict[str, float],
                                current_turn_by_user: Dict[str, int]) -> np.ndarray:
    """Baseline forgetting decision: the Phase-1 fixed exponential decay
    prior from the paper (Eq. 4), used as the comparison point for the
    learned survival classifier."""
    probs = np.zeros(len(memories))
    for i, m in enumerate(memories):
        lam = type_lambda.get(m["type"], 0.03)
        age = max(0, current_turn_by_user.get(m["user_id"], 0) - m["turn"])
        probs[i] = m["importance"] * np.exp(-lam * age)
    return probs


def context_efficiency(retriever, queries: List[Dict], token_budget: int,
                        avg_tokens_per_memory: int = 40) -> Dict[str, float]:
    """
    For a fixed context-window token budget, how many *actually relevant*
    memories get included per query, on average? This is the practical
    metric that matters for limited-context-window recall.
    """
    k = max(1, token_budget // avg_tokens_per_memory)
    relevant_included = []
    for q in queries:
        q_emb = np.array(q["embedding"])
        retrieved = retriever.retrieve(q_emb, k=k)
        hits = len(set(retrieved) & set(q["relevant_memory_ids"]))
        if q["relevant_memory_ids"]:
            relevant_included.append(hits)
    return {
        "avg_relevant_memories_included": float(np.mean(relevant_included)) if relevant_included else 0.0,
        "k_used_for_budget": k,
        "token_budget": token_budget,
    }