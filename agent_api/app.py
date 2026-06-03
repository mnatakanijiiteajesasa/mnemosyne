"""
agent_api/main.py
"""

from __future__ import annotations
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from memory_engine.models import MemoryType
from memory_engine.db import MemoryDB
from memory_engine.writer import MemoryWriter
from memory_engine.forgetting import ForgettingService
from memory_engine.embeddings.encoder import EmbeddingEngine


db:         MemoryDB         = None
encoder:    EmbeddingEngine  = None
writer:     MemoryWriter     = None
forgetting: ForgettingService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, encoder, writer, forgetting

    mongo_url  = os.getenv("MONGO_URL", "mongodb://agent:agent@mongo:27017/memories?authSource=admin")
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")

    db         = MemoryDB(mongo_url)
    encoder    = EmbeddingEngine(qdrant_url)
    writer     = MemoryWriter(db, encoder)
    forgetting = ForgettingService(db)

    await db.setup_indexes()
    await encoder.setup_collection()

    yield


app = FastAPI(title="Mnemosyne", version="0.2.0", lifespan=lifespan)



# Request models


class WriteRequest(BaseModel):
    user_id:     str
    session_id:  str
    content:     str
    memory_type: MemoryType
    tags:        list[str] = []
    source_turn: int = 0

class RetrieveRequest(BaseModel):
    user_id: str
    query:   str
    top_k:   int = 5

class TickRequest(BaseModel):
    user_id: str


# Endpoints

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.2.0"}


@app.post("/memory/write")
async def write_memory(req: WriteRequest):
    record = await writer.write(
        user_id     = req.user_id,
        session_id  = req.session_id,
        content     = req.content,
        memory_type = req.memory_type,
        tags        = req.tags,
        source_turn = req.source_turn,
    )
    return {"status": "written", "memory_id": record.id}


@app.post("/memory/retrieve")
async def retrieve_memory(req: RetrieveRequest):
    results = await encoder.search(req.query, top_k=req.top_k)
    user_results = [r for r in results if r["payload"].get("user_id") == req.user_id]
    for r in user_results:
        await db.update_access(r["memory_id"])
    return {"query": req.query, "results": user_results}


@app.get("/memory/list/{user_id}")
async def list_memories(user_id: str):
    records = await db.get_active(user_id)
    return {"user_id": user_id, "count": len(records), "memories": [r.dict() for r in records]}


@app.post("/memory/tick")
async def tick(req: TickRequest):
    """Age all memories by one turn and run the forgetting pass."""
    await db.tick_turns(req.user_id)
    archived = await forgetting.run(req.user_id)
    return {"ticked": True, "archived": archived}