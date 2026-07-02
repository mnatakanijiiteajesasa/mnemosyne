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
from memory_engine.memory_extractor import AutonomousMemoryExtractor, create_memory_extractor


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
episodic_summarizer: EpisodicSummarizer = None
memory_extractor:    AutonomousMemoryExtractor = None   # NEW


FORGET_EVERY_N_TURNS = 10


class ArchetypeSeeder:
    """
    Seeds new user graphs with archetype memories to resolve GNN cold start problem.
    Combines archetype bootstrapping with confidence-gated GNN retrieval.
    """

    def __init__(self, db: MemoryDB, encoder: EmbeddingEngine, graph: GraphBuilder):
        self.db = db
        self.encoder = encoder
        self.graph = graph
        # In a full implementation, this would load pre-computed archetypes
        # For now, we'll use a simple default archetype
        self.default_archetype = self._create_default_archetype()

    def _create_default_archetype(self) -> list[dict]:
        """
        Create a default archetype subgraph representing a general user persona.
        In production, this would be loaded from a pre-trained archetype library.
        """
        # Simple archetype: 4 nodes representing basic user interaction patterns
        return [
            {
                "content": "User prefers clear and concise responses",
                "memory_type": "preference",
                "importance_score": 0.7,
                "tags": ["communication", "style"]
            },
            {
                "content": "User is interested in learning new concepts",
                "memory_type": "preference",
                "importance_score": 0.6,
                "tags": ["learning", "curiosity"]
            },
            {
                "content": "User asks follow-up questions to deepen understanding",
                "memory_type": "episode",
                "importance_score": 0.5,
                "tags": ["interaction", "engagement"]
            },
            {
                "content": "User appreciates practical examples and use cases",
                "memory_type": "preference",
                "importance_score": 0.65,
                "tags": ["practical", "application"]
            }
        ]

    async def seed_if_new_user(self, user_id: str):
        """
        Check if user is new (no existing memories) and seed with archetype if needed.
        Implements archetype bootstrapping with initial seed confidence.
        """
        # Check if user has any existing active memories
        existing_memories = await self.db.get_active(user_id)

        # If user already has memories, no seeding needed
        if len(existing_memories) > 0:
            return

        # This is a new user - seed with archetype memories
        print(f"ArchetypeSeeder: Seeding new user {user_id} with default archetype")

        # Seed each archetype memory with is_seed=true and seed_confidence=0.3
        # We'll store this information in tags for now (in production, might extend schema)
        seeded_count = 0
        for i, archetype_mem in enumerate(self.default_archetype):
            try:
                # Add seed metadata to tags
                seed_tags = archetype_mem.get("tags", []) + [
                    f"is_seed:true",
                    f"seed_confidence:0.3",
                    f"seed_index:{i}"
                ]

                # Create memory record
                from memory_engine.models import MemoryRecord, MemoryType
                record = MemoryRecord(
                    user_id=user_id,
                    session_id="archetype_seed",  # Special session ID for archetype seeds
                    content=archetype_mem["content"],
                    memory_type=MemoryType(archetype_mem["memory_type"]),
                    importance_score=archetype_mem["importance_score"],
                    tags=seed_tags,
                    source_turn=0
                )

                # Persist to MongoDB
                await self.db.write(record)

                # Encode and store in Qdrant
                await self.encoder.store(
                    memory_id=record.id,
                    text=record.content,
                    payload={
                        "user_id": user_id,
                        "session_id": record.session_id,
                        "memory_type": record.memory_type.value,
                        "importance": record.importance_score,
                        "content": record.content,
                        "is_seed": True,
                        "seed_confidence": 0.3
                    },
                )

                # Link embedding back in MongoDB
                await self.db.set_embedding_id(record.id, record.id)

                seeded_count += 1

            except Exception as e:
                print(f"ArchetypeSeeder: Error seeding memory {i}: {e}")
                continue

        # TODO: Build edges between seeded memories to create archetype subgraph structure
        # For now, we'll rely on the graph builder to create edges based on similarity

        print(f"ArchetypeSeeder: Seeded {seeded_count} archetype memories for user {user_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, encoder, graph, hybrid_retrieval, writer, forgetting, session_store, llm, interaction_logger, episodic_summarizer, memory_extractor

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
    memory_extractor    = create_memory_extractor(llm)       # NEW

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

    # Check if this is a new user with no memories and seed archetype if needed
    archetype_seeder = ArchetypeSeeder(db, encoder, graph)
    await archetype_seeder.seed_if_new_user(req.user_id)

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
    
    # 6. Call Qwen with memory-injected prompt
    reply = ""
    if req.query:
        reply = await llm.chat(
            user_message       = req.query,
            retrieved_memories = retrieved,
            conversation_history = req.history,
        )

    #NEW — autonomously extract memories from this turn regardless of
    # what the caller sent in req.memories. This is what actually makes
    # memory accumulate — previously nothing ran unless the frontend
    # explicitly populated `memories`, which it never did.
    auto_written = []
    if req.query and reply and memory_extractor:
        try:
            print("="*60)
            print("Memory extractor:", memory_extractor)
            print("Query:", req.query)
            print("Reply:", reply)
            auto_memories = await memory_extractor.extract_memories(
                user_message=req.query,
                assistant_reply=reply,
                user_id=req.user_id,
                session_id=session.id,
                turn_number=turn,
            )
            print("Extracted memories:")
            print(auto_memories)
            print(type(auto_memories))
            print("Count:", len(auto_memories))
            
            for m in auto_memories:
                record = await writer.write(
                    user_id          = req.user_id,
                    session_id       = session.id,
                    content          = m["content"],
                    memory_type      = m["memory_type"],
                    tags             = m.get("tags", []),
                    source_turn      = m.get("source_turn", turn),
                    importance_score = m.get("importance_score", 0.5),
                )
                auto_written.append(record.id)
        except Exception as e:
            print(f"/turn: autonomous extraction failed: {e}")

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
        memories_written=written + summary_written + auto_written,   # CHANGED
        memories_retrieved=retrieved,
        reply=reply,
        archived_count=len(archived),
    )

    return {
        "session_id": session.id,
        "turn":       turn,
        "written":    written + summary_written + auto_written,      # CHANGED
        "retrieved":  retrieved,
        "archived":   archived,
        "reply":      reply,
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