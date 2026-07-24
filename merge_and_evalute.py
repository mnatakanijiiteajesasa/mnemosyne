"""
merge_and_evaluate.py

Merges gold_queries.json (from annotate_sample.py) into dataset_raw.json's
queries by id, then runs the existing harness's retrievers + metrics
against real production-derived data instead of synthetic data.

Requires the memory_eval_harness/ files (retrievers.py, metrics.py) to be
importable -- either copy them into this project directory, or add that
folder to PYTHONPATH.

Usage:
    python merge_and_evaluate.py --k 5
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from retrievers import FlatCosineRetriever, GraphSAGERetriever
from metrics import evaluate_retrieval, evaluate_forgetting, context_efficiency
from dataset_generator import TYPE_DECAY_LAMBDA
from metrics import exponential_decay_survival


def merge(dataset_path="dataset_raw.json", gold_path="gold_queries.json"):
    with open(dataset_path) as f:
        dataset = json.load(f)
    with open(gold_path) as f:
        gold = {q["id"]: q for q in json.load(f)}

    matched = 0
    merged_queries = []
    for q in dataset["queries"]:
        if q["id"] in gold:
            q = {**q, "relevant_memory_ids": gold[q["id"]]["relevant_memory_ids"]}
            merged_queries.append(q)
            matched += 1
    dataset["queries"] = merged_queries  # ONLY gold-labeled queries go into eval

    print(f"Matched {matched} gold-labeled queries out of {len(gold)} annotated.")
    if matched == 0:
        raise SystemExit("No matching queries found -- did you run extract_dataset.py "
                          "and annotate_sample.py against the same dataset_raw.json?")
    return dataset


def run(dataset, k=5, theta=0.75, token_budget=400):
    memories = dataset["memories"]
    queries = dataset["queries"]

    flat = FlatCosineRetriever().fit(memories)
    gnn = GraphSAGERetriever(theta=theta, use_message_passing=True).fit(memories)

    print("\n=== REAL-DATA RETRIEVAL ACCURACY (flat vs. GNN), gold-labeled queries only ===")
    for name, retr in [("flat_cosine", flat), ("gnn_graphsage", gnn)]:
        r = evaluate_retrieval(retr, queries, k=k)
        ctx = context_efficiency(retr, queries, token_budget=token_budget)
        print(f"  {name:15s} precision@{k}={r.get(f'precision@{k}'):.3f}  "
              f"recall@{k}={r.get(f'recall@{k}'):.3f}  "
              f"context_hits/query={ctx['avg_relevant_memories_included']:.2f}  "
              f"(n={r['n_queries_evaluated']})")

    print("\n=== FORGETTING ACCURACY (exponential decay vs. learned survival), all memories ===")
    current_turn_by_user = {}
    for m in memories:
        current_turn_by_user[m["user_id"]] = max(current_turn_by_user.get(m["user_id"], 0), m.get("turn", 0))
    exp_probs = exponential_decay_survival(memories, TYPE_DECAY_LAMBDA, current_turn_by_user)
    learned_probs = gnn.survival_probability()

    for name, probs in [("exponential_decay_baseline", exp_probs), ("gnn_learned_survival", learned_probs)]:
        r = evaluate_forgetting(memories, probs)
        print(f"  {name:25s} accuracy={r['forgetting_accuracy']:.3f}  "
              f"precision={r['forgetting_precision']:.3f}  recall={r['forgetting_recall']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.75)
    args = parser.parse_args()

    dataset = merge()
    run(dataset, k=args.k, theta=args.theta)