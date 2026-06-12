"""
agent_api/app.py
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from memory_engine.models import MemoryType
from memory_engine.db import MemoryDB
from memory_engine.writer import MemoryWriter
from memory_engine.forgetting import ForgettingService
from memory_engine.session_store import SessionStore
from memory_engine.embeddings.encoder import EmbeddingEngine
from memory_engine.gnn_engine.graph import GraphBuilder
from memory_engine.llm_client import QwenClient      


db:            MemoryDB          = None
encoder:       EmbeddingEngine   = None
graph:         GraphBuilder      = None
writer:        MemoryWriter      = None
forgetting:    ForgettingService = None
session_store: SessionStore      = None
llm:           QwenClient        = None           

FORGET_EVERY_N_TURNS = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, encoder, graph, writer, forgetting, session_store, llm

    mongo_url  = os.getenv("MONGO_URL",  "mongodb://agent:agent@mongo:27017/memories?authSource=admin")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

    db            = MemoryDB(mongo_url)
    encoder       = EmbeddingEngine(qdrant_url)
    graph         = GraphBuilder(mongo_url, qdrant_url)
    writer        = MemoryWriter(db, encoder, graph)
    forgetting    = ForgettingService(db)
    session_store = SessionStore(mongo_url)
    llm           = QwenClient()                      

    await db.setup_indexes()
    await encoder.setup_collection()
    await graph.setup_indexes()
    await session_store.setup_indexes()

    yield


app = FastAPI(title="Mnemosyne", version="0.4.0", lifespan=lifespan)

#Request models

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
    return {"status": "ok", "version": "0.4.0"}


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

    # 4. Retrieve relevant memories
    retrieved = []
    if req.query:
       # results   = await encoder.search(req.query, top_k=req.top_k)
        retrieved = await encoder.search(req.query, top_k=req.top_k, user_id=req.user_id)
        for r in retrieved:
            await db.update_access(r["memory_id"])

    # 5. Call Qwen with memory-injected prompt 
    reply = ""
    if req.query:
        reply = await llm.chat(
            user_message       = req.query,
            retrieved_memories = retrieved,
            conversation_history = req.history,
        )

    # 6. Run forgetting every N turns
    archived = []
    if turn % FORGET_EVERY_N_TURNS == 0:
        archived = await forgetting.run(req.user_id)

    return {
        "session_id": session.id,
        "turn":       turn,
        "written":    written,
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
    return {"status": "written", "memory_id": record.id}


@app.post("/memory/retrieve")
async def retrieve_memory(req: RetrieveRequest):
    # results      = await encoder.search(req.query, top_k=req.top_k)
    user_results = await encoder.search(req.query, top_k=req.top_k, user_id=req.user_id)     # [r for r in results if r["payload"].get("user_id") == req.user_id]
    for r in user_results:
        await db.update_access(r["memory_id"])
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