# MNEMOSYNE

MNEMOSYNE is a persistent memory AI agent system designed to maintain long-term context about users across sessions. It combines traditional memory storage with vector embeddings, graph relationships, and LLM integration to provide personalized, context-aware responses.

## Overview

The system implements a sophisticated memory architecture with a web-based frontend for interacting with the AI agent:

- **Backend API** (FastAPI): handles memory storage, retrieval, processing, and LLM communication
- **Frontend** (Streamlit): provides an interactive chat interface for users to converse with the agent
- **Memory Engine**: core memory system with vector storage, graph relationships, and intelligent retrieval/forgetting

## Key Features

- **Multi-type Memory Storage**: PREFERENCE, FACT, EPISODE, RULE, PLANNING, TOOL_USAGE
- **Vector Embeddings**: SentenceTransformers + Qdrant for semantic similarity search
- **Memory Graph**: Relationships between similar memories for enhanced retrieval
- **Intelligent Retrieval**: Hybrid scoring combining vector similarity, GNN relevance, recency, and cluster boosts
- **Learned Forgetting**: Trainable survival classifier replacing heuristic decay model
- **Context-Pressure Eviction**: Dynamic forgetting thresholds based on system load
- **Autonomous Learning Foundation**: Comprehensive interaction logging for model improvement
- **Advanced Intelligence**: Episodic summarization, semantic compression, shared memory, reasoning chains
- **Web Interface**: Streamlit-based chat application for easy interaction with the AI agent

## Architecture

```
• Frontend Layer (Streamlit)
  └─ Chat interface for user interaction

• API Layer (FastAPI)
  ├─ Memory Write Endpoints
  ├─ Memory Retrieval Endpoints (Hybrid)
  ├─ Conversation Processing (/turn)
  └─ Health & Status Endpoints

• Memory Engine
  ├─ Storage Layer (MongoDB)
  ├─ Vector Layer (Qdrant)
  ├─ Embedding Engine (SentenceTransformers)
  ├─ GNN Engine (GraphSAGE for relevance scoring)
  ├─ Forgetting Service (Learned survival classifier)
  ├─ Retrieval System (Hybrid vector + GNN scoring)
  ├─ Interaction Logger (Complete interaction tracking)
  ├─ Episodic Summarizer (LLM-based conversation summaries)
  ├─ Semantic Compressor (Memory deduplication & strengthening)
  ├─ Shared Memory Support (Multi-agent access control)
  └─ Reasoning Chains (Logical inference support)
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt` (for backend development)
   - Frontend dependencies are in `frontend/requirements.txt`
3. Set up environment variables in `.env` file
4. Start required services via Docker Compose (recommended for full stack)

## Environment Variables

See `.env.example` for required variables including:
- Database connections (MongoDB, Qdrant, PostgreSQL, Redis)
- LLM API keys (Qwen via Alibaba Cloud)
- Model paths (GNN, survival classifier)
- Forgetting parameters (thresholds, watermarks)
- Retrieval settings (cache sizes, hybrid weights)
- Frontend-backend communication (BACKEND_URL for frontend)

## API Endpoints

- `GET /health` - Health check
- `POST /turn` - Main conversation endpoint (session tick, memory processing, retrieval, LLM call)
- `POST /memory/write` - Direct memory storage
- `POST /memory/retrieve` - Hybrid memory retrieval
- `GET /memory/list/{user_id}` - List active memories
- `GET /memory/graph/{user_id}` - Memory relationship graph
- `GET /sessions/{user_id}` - List user sessions

## Memory Lifecycle

1. **Writing**: Text → Embedding (Qdrant) → Metadata (MongoDB) → Graph Edges
2. **Retrieval**: Query embedding → Qdrant search → Access count update → LLM context injection
3. **Forgetting**: Periodic survival scoring → Archival of low-probability memories
4. **Summarization**: Automatic episodic summary creation from conversation history
5. **Compression**: Similar memory grouping and compression for storage efficiency

## Testing

Run integration tests:
```bash
python tests/test_mnemosyne.py
```

Run GNN unit tests:
```bash
pytest tests/test_gnn.py -v
```

## Docker Deployment

The project includes `docker-compose.yml` for easy deployment of all required services:
- API (FastAPI) on port 8000
- Frontend (Streamlit) on port 8501
- MongoDB (memory storage)
- Qdrant (vector storage)
- PostgreSQL/pgvector (future extensions)
- Redis (caching)

To run the full stack:
```bash
docker compose up --build
```

Then access:
- **Frontend**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## License

MIT License

## Acknowledgments

Built with:
- FastAPI
- Streamlit
- SentenceTransformers
- Qdrant
- PyTorch Geometric (GNN)
- MongoDB
- Qwen LLM (Alibaba Cloud)