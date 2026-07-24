"""
extract_dataset.py

Pulls real memory records and interaction logs from the (test-stack)
MongoDB that simulate.py just populated, and reshapes them into the
same {memories, queries} schema dataset_generator.py produces -- so
retrievers.py / metrics.py / run_harness.py work unchanged on real data.

READ THIS BEFORE TRUSTING ANY PRECISION/RECALL NUMBER FROM THIS FILE'S
OUTPUT:
------------------------------------------------------------------------
`interaction_logs.details.memories_retrieved` reflects what the GNN
hybrid retriever ALREADY chose to serve in production. Using it as
ground truth "relevant_memory_ids" for evaluating flat-vs-GNN retrieval
accuracy is circular -- you'd be scoring the GNN against its own past
opinions. This script therefore outputs two separate things:

  1. dataset_raw.json -- memories + queries with re-computed embeddings,
     and 'reused'/'should_survive' labels for the FORGETTING evaluation
     only. That eval is legitimate because it asks "does this memory get
     touched again later", which is about future recurrence, not about
     which retriever chose it this time. (There's still a mild bootstrap
     bias here, same as the paper's own Sec 4.3 approach -- flag this in
     your methods section, don't hide it.)

  2. candidate_pools.json -- for a sample of queries, the UNION of
     top-k-by-flat-cosine and top-k-as-actually-retrieved-by-GNN, with
     neither retriever's ranking preserved (shuffled) and no labels.
     Feed this into annotate_sample.py to get real, non-circular
     relevant_memory_ids via manual labeling. THAT is what you use for
     the retrieval-accuracy comparison in the paper.
------------------------------------------------------------------------

Usage:
    python extract_dataset.py --mongo_url mongodb://... --pool_queries 40
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import numpy as np

from memory_engine.embeddings.encoder import EmbeddingEngine


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-9)
    B = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return B @ a


async def extract(mongo_url: str, qdrant_url: str, db_name: str = "memories"):
    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

    memory_docs = [doc async for doc in db["memory_records"].find({})]
    log_docs = [doc async for doc in db["interaction_logs"].find({"interaction_type": "turn"})]

    print(f"Pulled {len(memory_docs)} memory records, {len(log_docs)} turn logs.")

    encoder = EmbeddingEngine(qdrant_url)

    # --- memories: re-encode content with the same model (deterministic,
    # so this matches what's in Qdrant without needing a Qdrant round trip) ---
    memories = []
    for doc in memory_docs:
        emb = encoder.encode(doc["content"])
        memories.append({
            "id": doc["id"], "user_id": doc["user_id"], "session_id": doc["session_id"],
            "content": doc["content"],
            "turn": doc.get("source_turn", 0), "type": doc["memory_type"],
            "cluster": doc.get("cluster_id", -1), "embedding": emb,
            "importance": doc.get("importance_score", 0.5),
            "status": doc.get("status", "active"),
            "access_count": doc.get("access_count", 0),
            "reused": False, "should_survive": False,  # filled in below
        })
    mem_by_id = {m["id"]: m for m in memories}

    # --- reuse labels: a memory is "reused" if it shows up in
    # memories_retrieved for MORE THAN ONE distinct query (paper's Sec 4.3
    # bootstrap: recalled-later = positive). should_survive = same signal,
    # used for the forgetting eval. ---
    retrieval_counts = defaultdict(int)
    for log in log_docs:
        for r in log["details"].get("memories_retrieved", []):
            mid = r.get("memory_id")
            if mid:
                retrieval_counts[mid] += 1
    for mid, count in retrieval_counts.items():
        if mid in mem_by_id:
            mem_by_id[mid]["reused"] = count >= 2
            mem_by_id[mid]["should_survive"] = count >= 2 or mem_by_id[mid]["status"] == "active"

    # --- queries: re-encode query text, keep what was ACTUALLY retrieved
    # (for the candidate pool step) but do NOT treat it as ground truth ---
    queries = []
    for i, log in enumerate(log_docs):
        details = log["details"]
        q_text = details.get("query", "")
        if not q_text:
            continue
        emb = encoder.encode(q_text)
        retrieved_ids = [r.get("memory_id") for r in details.get("memories_retrieved", []) if r.get("memory_id")]
        queries.append({
            "id": f"log_{i}", "user_id": log["user_id"], "session_id": log.get("session_id"),
            "turn": None, "cluster": -1, "embedding": emb,
            "query_text": q_text,  # kept for annotate_sample.py's human-readable prompt
            "gnn_retrieved_ids": retrieved_ids,  # NOT ground truth -- see module docstring
            "relevant_memory_ids": [],  # left empty; fill via annotate_sample.py
        })

    dataset = {"memories": memories, "queries": queries,
               "meta": {"n_users": len({m['user_id'] for m in memories}),
                         "source": "production_test_stack", "extracted_real_data": True}}

    with open("dataset_raw.json", "w") as f:
        json.dump(dataset, f)
    print(f"Wrote dataset_raw.json: {len(memories)} memories, {len(queries)} queries.")
    print(f"  {sum(m['reused'] for m in memories)} memories marked 'reused' "
          f"(retrieved 2+ times) -- used for forgetting eval only.")

    return dataset, encoder


def build_candidate_pools(dataset: dict, k: int = 10, n_sample_queries: int = 40, seed: int = 0):
    """TREC-style pooling: for each sampled query, union top-k-by-flat-cosine
    with whatever GNN actually retrieved. Shuffled, unlabeled -- hands this
    to annotate_sample.py for human judgment, so neither retriever's
    ranking or presence biases the label."""
    rng = random.Random(seed)
    memories = dataset["memories"]
    mem_ids = [m["id"] for m in memories]
    mem_embs = np.array([m["embedding"] for m in memories])
    mem_by_id = {m["id"]: m for m in memories}

    queries = [q for q in dataset["queries"] if q["query_text"].strip()]
    sample = rng.sample(queries, min(n_sample_queries, len(queries)))

    pools = []
    for q in sample:
        q_emb = np.array(q["embedding"])
        sims = cosine_sim(q_emb, mem_embs)
        top_flat = [mem_ids[i] for i in np.argsort(-sims)[:k]]
        top_gnn = q["gnn_retrieved_ids"][:k]

        union_ids = list(dict.fromkeys(top_flat + top_gnn))  # dedupe, preserve some order
        rng.shuffle(union_ids)  # remove ranking signal before showing to human

        pools.append({
            "query_id": q["id"], "user_id": q["user_id"], "query_text": q["query_text"],
            "candidates": [
                {"id": mid, "content": mem_by_id[mid]["content"]}
                for mid in union_ids if mid in mem_by_id
            ],
        })

    with open("candidate_pools.json", "w") as f:
        json.dump(pools, f, indent=2)
    print(f"Wrote candidate_pools.json: {len(pools)} queries sampled for annotation "
          f"(pool size up to {2*k} candidates each, deduped).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo_url", type=str,
                         default="mongodb://agent:agent@localhost:27018/memories?authSource=admin")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6334")
    parser.add_argument("--pool_k", type=int, default=10)
    parser.add_argument("--pool_queries", type=int, default=40)
    args = parser.parse_args()

    dataset, encoder = asyncio.run(extract(args.mongo_url, args.qdrant_url))
    build_candidate_pools(dataset, k=args.pool_k, n_sample_queries=args.pool_queries)