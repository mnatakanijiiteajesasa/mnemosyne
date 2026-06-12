"""
agent_api/app.py
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memory_engine.models import MemoryType
from memory_engine.db import MemoryDB
from memory_engine.writer import MemoryWriter
from memory_engine.forgetting import ForgettingService
from memory_engine.session_store import SessionStore
from memory_engine.embeddings.encoder import EmbeddingEngine
from memory_engine.gnn_engine.graph import GraphBuilder
from memory_engine.llm_client import QwenClient
from memory_engine.hybrid_retrieval import HybridRetrievalEngine, create_hybrid_retrieval_engine
from memory_engine.interaction_logger import InteractionLogger
from memory_engine.episodic_summarizer import EpisodicSummarizer, create_episodic_summarizer


db:            MemoryDB          = None
encoder:       EmbeddingEngine   = None
graph:         GraphBuilder      = None
hybrid_retrieval: HybridRetrievalEngine = None
writer:        MemoryWriter      = None
forgetting:    ForgettingService = None
session_store: SessionStore      = None
llm:           QwenClient        = None
interaction_logger: InteractionLogger = None
episodic_summarizer: EpisodicSummarizer = None

FORGET_EVERY_N_TURNS = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, encoder, graph, hybrid_retrieval, writer, forgetting, session_store, llm, interaction_logger, episodic_summarizer

    mongo_url  = os.getenv("MONGO_URL",  "mongodb://agent:agent@mongo:27017/memories?authSource=admin")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    gnn_model_path = os.getenv("GNN_MODEL_PATH", None)  # Optional GNN model path
    survival_model_path = os.getenv("SURVIVAL_MODEL_PATH", None)  # Optional survival classifier model path
    survival_threshold = float(os.getenv("SURVIVAL_THRESHOLD", "0.5"))
    # Context-pressure watermarks from env (optional)
    high_watermark = int(os.getenv("FORGETTING_HIGH_WATERMARK", "1000"))
    low_watermark = int(os.getenv("FORGETTING_LOW_WATERMARK", "100"))
    pressure_shift = float(os.getenv("FORGETTING_PRESSURE_SHIFT", "0.1"))

    db            = MemoryDB(mongo_url)
    encoder       = EmbeddingEngine(qdrant_url)
    graph         = GraphBuilder(mongo_url, qdrant_url)
    # Initialize hybrid retrieval system
    hybrid_retrieval = create_hybrid_retrieval_engine(
        mongo_url=mongo_url,
        qdrant_url=qdrant_url,
        model_path=gnn_model_path,
        device="cpu",  # Could make this configurable
        cache_size=1000,
        enable_cache=True
    )
    writer        = MemoryWriter(db, encoder, graph)
    # Initialize forgetting service with learned model option
    forgetting    = ForgettingService(
        db=db,
        model_path=survival_model_path,
        device="cpu",
        survival_threshold=survival_threshold,
        high_watermark=high_watermark,
        low_watermark=low_watermark,
        pressure_survival_threshold_shift=pressure_shift,
    )
    session_store = SessionStore(mongo_url)
    llm           = QwenClient()
    # Initialize interaction logger
    interaction_logger = InteractionLogger(mongo_url)
    # Initialize episodic summarizer
    episodic_summarizer = create_episodic_summarizer(llm)

    await db.setup_indexes()
    await encoder.setup_collection()
    await graph.setup_indexes()
    await session_store.setup_indexes()
    await interaction_logger.setup_indexes()

    yield


app = FastAPI(title="Mnemosyne", version="0.6.0", lifespan=lifespan)

# CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models

class WriteRequest(BaseModel):
    user_id:     str
    session_id:  Optional[str] = None
    content:     str
    memory_type: MemoryType
    tags:        list[str] = []


class RetrieveRequest(BaseModel):
    user_id:    str
    session_id: Optional[str] = None
    query:      str
    top_k:      int = 5


class TurnRequest(BaseModel):
    """
    Main entry point per conversation turn.
    Ticks the session, ages memories, runs forgetting every N turns,
    retrieves context, and calls Qwen with memory-injected prompt.
    """
    user_id:    str
    session_id: Optional[str] = None
    memories:   list[dict] = []   # [{content, memory_type, tags}]
    query:      str = ""
    top_k:      int = 5
    history:    list[dict] = []   # NEW — [{"role": "user"|"assistant", "content": "..."}]


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.6.0"}


@app.post("/turn")
async def process_turn(req: TurnRequest):
    # 1. Get or create session
    session = await session_store.get_or_create(req.user_id, req.session_id)
    turn    = await session_store.tick(session.id)

    # 2. Age all memories
    await db.tick_turns(req.user_id)

    # 3. Write any new memories from this turn
    written = []
    for m in req.memories:
        record = await writer.write(
            user_id     = req.user_id,
            session_id  = session.id,
            content     = m["content"],
            memory_type = MemoryType(m["memory_type"]),
            tags        = m.get("tags", []),
            source_turn = turn,
        )
        written.append(record.id)

    # 4. Optionally create episodic summary from conversation history
    summary_written = []
    if req.history and len(req.history) > 0 and episodic_summarizer:
        try:
            summary_data = await episodic_summarizer.summarize_conversation(
                conversation_history=req.history,
                user_id=req.user_id,
                session_id=session.id,
                max_length=200,
            )
            if summary_data:
                # Create the summary memory
                summary_record = await writer.write(
                    user_id     = req.user_id,
                    session_id  = session.id,
                    content     = summary_data["content"],
                    memory_type = MemoryType(summary_data["memory_type"]),
                    tags        = summary_data.get("tags", []),
                    source_turn = turn,
                    importance_score = summary_data.get("importance_score", 0.6),
                )
                summary_written.append(summary_record.id)
        except Exception as e:
            # Log error but don't fail the turn
            await interaction_logger.log_turn(
                user_id=req.user_id,
                session_id=session.id,
                query=req.query,
                top_k=req.top_k,
                memories_written=written,
                memories_retrieved=[],
                reply=f"Error creating episodic summary: {str(e)}",
                archived_count=0,
            )

    # 5. Retrieve relevant memories using hybrid retrieval
    retrieved = []
    if req.query:
        retrieved = await hybrid_retrieval.search(
            query=req.query,
            user_id=req.user_id,
            top_k=req.top_k,
            use_hybrid=True  # Use hybrid scoring when available
        )
        for r in retrieved:
            await db.update_access(r["memory_id"])

    # 6. Call Qwen with memory-injected prompt
    reply = ""
    if req.query:
        reply = await llm.chat(
            user_message       = req.query,
            retrieved_memories = retrieved,
            conversation_history = req.history,
        )

    # 7. Run forgetting every N turns
    archived = []
    if turn % FORGET_EVERY_N_TURNS == 0:
        archived = await forgetting.run(req.user_id)

    # 8. Log the interaction
    await interaction_logger.log_turn(
        user_id=req.user_id,
        session_id=session.id,
        query=req.query,
        top_k=req.top_k,
        memories_written=written + summary_written,
        memories_retrieved=retrieved,
        reply=reply,
        archived_count=len(archived),
    )

    return {
        "session_id": session.id,
        "turn":       turn,
        "written":    written + summary_written,
        "retrieved":  retrieved,
        "archived":   archived,
        "reply":      reply,             # NEW — Qwen's response
    }


@app.post("/memory/write")
async def write_memory(req: WriteRequest):
    session = await session_store.get_or_create(req.user_id, req.session_id)
    record  = await writer.write(
        user_id     = req.user_id,
        session_id  = session.id,
        content     = req.content,
        memory_type = req.memory_type,
        tags        = req.tags,
    )
    # Log the write operation
    await interaction_logger.log_memory_write(
        user_id=req.user_id,
        session_id=session.id,
        content=req.content,
        memory_type=req.memory_type,
        tags=req.tags,
        memory_id=record.id,
    )
    return {"status": "written", "memory_id": record.id}


@app.post("/memory/retrieve")
async def retrieve_memory(req: RetrieveRequest):
    # Use hybrid retrieval instead of direct encoder search
    user_results = await hybrid_retrieval.search(
        query=req.query,
        user_id=req.user_id,
        top_k=req.top_k,
        use_hybrid=True
    )
    for r in user_results:
        await db.update_access(r["memory_id"])
    # Log the retrieval
    await interaction_logger.log_retrieval(
        user_id=req.user_id,
        query=req.query,
        top_k=req.top_k,
        results=user_results,
        search_method="hybrid"
    )
    return {"query": req.query, "results": user_results}


@app.get("/memory/list/{user_id}")
async def list_memories(user_id: str):
    records = await db.get_active(user_id)
    return {"user_id": user_id, "count": len(records), "memories": [r.dict() for r in records]}


@app.get("/memory/graph/{user_id}")
async def get_graph(user_id: str):
    records = await db.get_active(user_id)
    ids     = [r.id for r in records]
    adj     = await graph.get_adjacency(ids)
    return {"user_id": user_id, "nodes": ids, "edges": adj}


@app.get("/sessions/{user_id}")
async def list_sessions(user_id: str):
    sessions = await session_store.list_sessions(user_id)
    return {"user_id": user_id, "sessions": [s.dict() for s in sessions]}