# Mnemosyne Eval Project

Separate project that installs `memory_engine` as a package and drives it
against 10 scripted personas, on an isolated test database stack, to
produce real (not synthetic) data for the flat-vs-GNN retrieval research
paper.

## Setup

```bash
pip install -e /path/to/mnemosyne          # from the feature/package-memory-engine branch
pip install -r requirements.txt             # this project's own deps (see below)
cp /path/to/mnemosyne/memory_eval_harness/{retrievers.py,metrics.py,dataset_generator.py} .
docker compose -f /path/to/mnemosyne/docker-compose.test.yml up -d
```

requirements.txt for this project: `python-dotenv`, plus whatever
`memory_engine`'s pyproject.toml already pulls in (motor, qdrant-client,
sentence-transformers, openai, etc. come along with the `-e` install).

## Pipeline (run in this order)

```bash
# 1. Simulate 10 personas across many sessions against the TEST stack.
#    Start small to sanity check before committing to the full run.
python simulate.py --sessions 3 --turns 3 --concurrency 3     # pilot
python simulate.py --sessions 20 --turns 3 --concurrency 3    # full run

# 2. Pull memories + interaction logs from Mongo, re-embed, build
#    unbiased candidate pools for annotation.
python extract_dataset.py --pool_queries 40

# 3. Label the candidate pools by hand (this is the step that makes the
#    retrieval-accuracy numbers real instead of circular -- see the
#    docstring in extract_dataset.py for why this can't be automated).
python annotate_sample.py
#    (resumable -- ctrl-C or type "quit" any time, progress is saved)

# 4. Merge gold labels back in and run flat-vs-GNN comparison on real data.
python merge_and_evaluate.py --k 5
```

## What's genuinely valid vs. what needs care in the paper

**Solid, non-circular:**
- Retrieval precision/recall from step 4 -- gold labels came from you,
  judging an unbiased candidate pool, blind to which retriever surfaced
  what.

**Valid with a stated caveat:**
- Forgetting accuracy -- "reused"/"should_survive" labels come from
  memories being retrieved 2+ times in production logs, which is the
  same bootstrap approach the source paper describes (Sec 4.3), but note
  in your methods section that "retrieved" was decided by the GNN system
  itself, so there's a mild self-reinforcing bias. A stronger version
  would independently verify reuse (e.g., did the memory's topic
  genuinely recur in the conversation, judged by you or an LLM judge
  blind to which system retrieved it) -- worth doing if you have time.

**Not yet addressed:**
- Statistical significance (paired bootstrap across the gold-labeled
  queries) before claiming the real-data gap is real and not noise --
  40 queries is a reasonable pilot sample size but likely too small to
  publish on alone. Scale up personas/sessions and annotation once the
  pipeline is validated at small scale.
- The `GraphSAGERetriever` in `retrievers.py` is still the dependency-
  light numpy reimplementation from the synthetic-data phase, not your
  actual trained Phase 5 PyTorch checkpoint. Real paper results should
  come from your production model's actual scores -- swap
  `GraphSAGERetriever.retrieve()` for a call into your Phase 5 inference
  engine (`memory_engine.gnn_engine.inference`) before finalizing numbers.