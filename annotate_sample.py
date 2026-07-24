"""
annotate_sample.py

Interactive terminal labeling tool. Reads candidate_pools.json (built by
extract_dataset.py's TREC-style pooling: union of flat-cosine top-k and
GNN-retrieved memories, shuffled, unlabeled) and asks YOU to judge which
candidates are actually relevant to each query.

This is the step that gets you real, non-circular ground truth for the
paper's central retrieval-accuracy comparison. It's manual and a bit
tedious by design -- there's no shortcut that doesn't reintroduce
circularity (see extract_dataset.py's docstring).

Usage:
    python annotate_sample.py
    # answer y/n for each candidate, per query
    # progress is saved after every query, so you can quit and resume

Output:
    gold_queries.json -- same shape as dataset_generator.py's query
    objects, with real `relevant_memory_ids`, ready to feed into
    metrics.evaluate_retrieval() alongside dataset_raw.json's memories.
"""
from __future__ import annotations
import json
from pathlib import Path

POOLS_PATH = Path("candidate_pools.json")
PROGRESS_PATH = Path("gold_queries.json")


def load_pools():
    with open(POOLS_PATH) as f:
        return json.load(f)


def load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH) as f:
            return {q["id"]: q for q in json.load(f)}
    return {}


def save_progress(done: dict):
    with open(PROGRESS_PATH, "w") as f:
        json.dump(list(done.values()), f, indent=2)


def annotate():
    pools = load_pools()
    done = load_progress()

    remaining = [p for p in pools if p["query_id"] not in done]
    print(f"{len(done)}/{len(pools)} queries already labeled. "
          f"{len(remaining)} remaining.\n")

    for i, pool in enumerate(remaining):
        print("=" * 70)
        print(f"Query {i+1}/{len(remaining)}  (user: {pool['user_id']})")
        print(f'  "{pool["query_text"]}"')
        print("-" * 70)

        relevant_ids = []
        for j, cand in enumerate(pool["candidates"]):
            content = cand["content"]
            print(f"  [{j+1}/{len(pool['candidates'])}] {content}")
            while True:
                ans = input("      relevant to the query above? (y/n/skip-query/quit): ").strip().lower()
                if ans in ("y", "n", "skip-query", "quit"):
                    break
                print("      please answer y, n, skip-query, or quit")

            if ans == "quit":
                save_progress(done)
                print(f"\nSaved progress. {len(done)}/{len(pools)} done. Resume any time.")
                return
            if ans == "skip-query":
                break
            if ans == "y":
                relevant_ids.append(cand["id"])

        else:
            # only save if we didn't break out via skip-query
            done[pool["query_id"]] = {
                "id": pool["query_id"], "user_id": pool["user_id"],
                "query_text": pool["query_text"],
                "relevant_memory_ids": relevant_ids,
            }
            save_progress(done)
            print(f"  -> {len(relevant_ids)} marked relevant. Saved.\n")

    print(f"\nAll done: {len(done)}/{len(pools)} queries labeled -> {PROGRESS_PATH}")
    print("Next: merge gold_queries.json's relevant_memory_ids into dataset_raw.json's "
          "queries (matching on id), then run the harness's evaluate_retrieval() "
          "against real embeddings + real gold labels.")


if __name__ == "__main__":
    annotate()