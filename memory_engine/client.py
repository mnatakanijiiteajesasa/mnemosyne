"""
memory_engine/client.py

MnemosyneClient -- a reusable, importable async client that wraps the
memory_engine components the same way agent_api/app.py's `lifespan()` and
`/turn` endpoint do, WITHOUT going through FastAPI/HTTP.

Why this exists
----------------
agent_api/app.py currently owns all the wiring logic (component
construction, ArchetypeSeeder, the full /turn sequence) inline in module-
level globals and a route handler. That's fine for serving production
traffic, but it means the only way to drive the system programmatically
(e.g. for the eval harness, or for 10 scripted personas) is over HTTP
against a running server.

This file duplicates NOTHING structurally -- it re-implements the same
sequence as a plain class so it can be:
  - installed as a package and imported from a separate eval project
  - pointed at an isolated test database stack (docker-compose.test.yml)
  - driven directly in a script/notebook, no server required

It does not modify agent_api/app.py, memory_engine/*, or anything else in
the existing repo -- it is a pure addition that imports and composes the
existing, unmodified memory_engine modules.

Usage
-----
    import asyncio
    from memory_engine.client import MnemosyneClient, MnemosyneConfig

    async def main():
        client = await MnemosyneClient.create(
            MnemosyneConfig(
                mongo_url="mongodb://agent:agent@localhost:27018/memories?authSource=admin",
                qdrant_url="http://localhost:6334",
            )
        )
        result = await client.turn(
            user_id="persona_01",
            query="What did I say my favorite programming language was?",
        )
        print(result["reply"])

    asyncio.run(main())
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import MemoryType
from .db import MemoryDB
from .writer import MemoryWriter
from .forgetting import ForgettingService
from .session_store import SessionStore
from .embeddings.encoder import EmbeddingEngine
from .gnn_engine.graph import GraphBuilder
from .llm_client import QwenClient
from .hybrid_retrieval import create_hybrid_retrieval_engine
from .interaction_logger import InteractionLogger
from .episodic_summarizer import create_episodic_summarizer
from .memory_extractor import create_memory_extractor


FORGET_EVERY_N_TURNS = 10


@dataclass
class MnemosyneConfig:
    """All connection/config values agent_api/app.py currently reads from
    os.getenv() inside lifespan(). Pass these explicitly here instead of
    relying on environment variables -- this is what makes it safe to point
    an eval project at a completely separate database stack."""
    mongo_url: str = "mongodb://agent:agent@localhost:27017/memories?authSource=admin"
    qdrant_url: str = "http://localhost:6333"
    gnn_model_path: Optional[str] = None
    survival_model_path: Optional[str] = None
    survival_threshold: float = 0.5
    forgetting_high_watermark: int = 1000
    forgetting_low_watermark: int = 100
    forgetting_pressure_shift: float = 0.1
    device: str = "cpu"


class _ArchetypeSeeder:
    """Verbatim port of agent_api/app.py's ArchetypeSeeder. Kept private
    to this module so agent_api/app.py never needs to change or export it."""

    def __init__(self, db: MemoryDB, encoder: EmbeddingEngine, graph: GraphBuilder):
        self.db = db
        self.encoder = encoder
        self.graph = graph
        self.default_archetype = self._create_default_archetype()

    def _create_default_archetype(self) -> list[dict]:
        return [
            {"content": "User prefers clear and concise responses",
             "memory_type": "preference", "importance_score": 0.7,
             "tags": ["communication", "style"]},
            {"content": "User is interested in learning new concepts",
             "memory_type": "preference", "importance_score": 0.6,
             "tags": ["learning", "curiosity"]},
            {"content": "User asks follow-up questions to deepen understanding",
             "memory_type": "episode", "importance_score": 0.5,
             "tags": ["interaction", "engagement"]},
            {"content": "User appreciates practical examples and use cases",
             "memory_type": "preference", "importance_score": 0.65,
             "tags": ["practical", "application"]},
        ]

    async def seed_if_new_user(self, user_id: str):
        existing_memories = await self.db.get_active(user_id)
        if existing_memories:
            return

        from .models import MemoryRecord, MemoryType

        seeded_count = 0
        for i, archetype_mem in enumerate(self.default_archetype):
            try:
                seed_tags = archetype_mem.get("tags", []) + [
                    "is_seed:true", "seed_confidence:0.3", f"seed_index:{i}",
                ]
                record = MemoryRecord(
                    user_id=user_id,
                    session_id="archetype_seed",
                    content=archetype_mem["content"],
                    memory_type=MemoryType(archetype_mem["memory_type"]),
                    importance_score=archetype_mem["importance_score"],
                    tags=seed_tags,
                    source_turn=0,
                )
                await self.db.write(record)
                await self.encoder.store(
                    memory_id=record.id,
                    text=record.content,
                    payload={
                        "user_id": user_id, "session_id": record.session_id,
                        "memory_type": record.memory_type.value,
                        "importance": record.importance_score,
                        "content": record.content,
                        "is_seed": True, "seed_confidence": 0.3,
                    },
                )
                await self.db.set_embedding_id(record.id, record.id)
                seeded_count += 1
            except Exception:
                continue



class MnemosyneClient:
    """Async facade over the full memory_engine pipeline, mirroring
    agent_api/app.py's /turn endpoint exactly. Construct with
    `await MnemosyneClient.create(config)`, not `__init__` directly, since
    setup requires async index creation."""

    def __init__(self, config: MnemosyneConfig):
        self.config = config
        self.db: MemoryDB
        self.encoder: EmbeddingEngine
        self.graph: GraphBuilder
        self.hybrid_retrieval = None
        self.writer: MemoryWriter
        self.forgetting: ForgettingService
        self.session_store: SessionStore
        self.llm: QwenClient
        self.interaction_logger: InteractionLogger
        self.episodic_summarizer = None
        self.memory_extractor = None

    @classmethod
    async def create(cls, config: Optional[MnemosyneConfig] = None) -> "MnemosyneClient":
        config = config or MnemosyneConfig()
        self = cls(config)

        self.db = MemoryDB(config.mongo_url)
        self.encoder = EmbeddingEngine(config.qdrant_url)
        self.graph = GraphBuilder(config.mongo_url, config.qdrant_url)
        self.hybrid_retrieval = create_hybrid_retrieval_engine(
            mongo_url=config.mongo_url,
            qdrant_url=config.qdrant_url,
            model_path=config.gnn_model_path,
            device=config.device,
            cache_size=1000,
            enable_cache=True,
        )
        self.writer = MemoryWriter(self.db, self.encoder, self.graph)
        self.forgetting = ForgettingService(
            db=self.db,
            model_path=config.survival_model_path,
            device=config.device,
            survival_threshold=config.survival_threshold,
            high_watermark=config.forgetting_high_watermark,
            low_watermark=config.forgetting_low_watermark,
            pressure_survival_threshold_shift=config.forgetting_pressure_shift,
        )
        self.session_store = SessionStore(config.mongo_url)
        self.llm = QwenClient()
        self.interaction_logger = InteractionLogger(config.mongo_url)
        self.episodic_summarizer = create_episodic_summarizer(self.llm)
        self.memory_extractor = create_memory_extractor(self.llm)

        await self.db.setup_indexes()
        await self.encoder.setup_collection()
        await self.graph.setup_indexes()
        await self.session_store.setup_indexes()
        await self.interaction_logger.setup_indexes()

        return self

    async def turn(
        self,
        user_id: str,
        query: str = "",
        session_id: Optional[str] = None,
        memories: Optional[list[dict]] = None,
        top_k: int = 5,
        history: Optional[list[dict]] = None,
    ) -> dict:
        """Exact port of agent_api/app.py's process_turn(). Same return
        shape: {session_id, turn, written, retrieved, archived, reply}."""
        memories = memories or []
        history = history or []

        session = await self.session_store.get_or_create(user_id, session_id)

        seeder = _ArchetypeSeeder(self.db, self.encoder, self.graph)
        await seeder.seed_if_new_user(user_id)

        turn = await self.session_store.tick(session.id)
        await self.db.tick_turns(user_id)

        written = []
        for m in memories:
            record = await self.writer.write(
                user_id=user_id, session_id=session.id, content=m["content"],
                memory_type=MemoryType(m["memory_type"]), tags=m.get("tags", []),
                source_turn=turn,
            )
            written.append(record.id)

        summary_written = []
        if history and self.episodic_summarizer:
            try:
                summary_data = await self.episodic_summarizer.summarize_conversation(
                    conversation_history=history, user_id=user_id,
                    session_id=session.id, max_length=200,
                )
                if summary_data:
                    summary_record = await self.writer.write(
                        user_id=user_id, session_id=session.id,
                        content=summary_data["content"],
                        memory_type=MemoryType(summary_data["memory_type"]),
                        tags=summary_data.get("tags", []), source_turn=turn,
                        importance_score=summary_data.get("importance_score", 0.6),
                    )
                    summary_written.append(summary_record.id)
            except Exception:
                pass  # matches app.py: don't fail the turn on summary errors

        retrieved = []
        if query:
            retrieved = await self.hybrid_retrieval.search(
                query=query, user_id=user_id, top_k=top_k, use_hybrid=True,
            )
            for r in retrieved:
                await self.db.update_access(r["memory_id"])

        reply = ""
        if query:
            reply = await self.llm.chat(
                user_message=query, retrieved_memories=retrieved,
                conversation_history=history,
            )

        auto_written = []
        if query and reply and self.memory_extractor:
            try:
                auto_memories = await self.memory_extractor.extract_memories(
                    user_message=query, assistant_reply=reply, user_id=user_id,
                    session_id=session.id, turn_number=turn,
                )
                for m in auto_memories:
                    record = await self.writer.write(
                        user_id=user_id, session_id=session.id, content=m["content"],
                        memory_type=m["memory_type"], tags=m.get("tags", []),
                        source_turn=m.get("source_turn", turn),
                        importance_score=m.get("importance_score", 0.5),
                    )
                    auto_written.append(record.id)
            except Exception:
                pass

        archived = []
        if turn % FORGET_EVERY_N_TURNS == 0:
            archived = await self.forgetting.run(user_id)

        await self.interaction_logger.log_turn(
            user_id=user_id, session_id=session.id, query=query, top_k=top_k,
            memories_written=written + summary_written + auto_written,
            memories_retrieved=retrieved, reply=reply, archived_count=len(archived),
        )

        return {
            "session_id": session.id, "turn": turn,
            "written": written + summary_written + auto_written,
            "retrieved": retrieved, "archived": archived, "reply": reply,
        }

    async def get_memories(self, user_id: str) -> list[dict]:
        """Convenience accessor for the eval harness's dataset extraction
        step -- returns active memory records for a user as plain dicts."""
        records = await self.db.get_active(user_id)
        return [r.dict() for r in records]