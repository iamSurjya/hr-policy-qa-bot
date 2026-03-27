# HR Policy Q&A Bot — Production RAG System

A production-grade HR policy question-answering bot built with RAG (Retrieval Augmented Generation). Employees ask questions in natural language and get accurate, grounded answers with source attribution.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│              HTTP request (POST /query)              │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                     API Layer                        │
│     FastAPI + Pydantic v2 validation (schemas.py)    │
│          routes.py → POST /query, GET /health        │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                  Pipeline Layer                      │
│        pipeline.py — orchestrates full RAG flow      │
│   search() → filter → format_context() → generate() │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  Retrieval  │          │   Generation    │
│  searcher   │          │   providers.py  │
│  ChromaDB   │          │  Groq / Claude  │
│  MiniLM-L6  │          │  (swap via env) │
└─────────────┘          └─────────────────┘
```

---

## Tech Stack

| Layer | Dev | Production |
|---|---|---|
| LLM | Groq — llama-3.3-70b-versatile (free) | Claude Sonnet — switch via `LLM_PROVIDER` |
| LLM (alt) | Gemini 2.0 Flash — preserved, needs billing | — |
| Embeddings | all-MiniLM-L6-v2 (local, free) | same |
| Vector DB | ChromaDB PersistentClient | same |
| API | FastAPI | same |
| Validation | Pydantic v2 + pydantic-settings | same |
| Logging | structlog JSON | same |
| Package mgmt | Poetry 2.3.2 | same |
| Python | 3.11.9 via pyenv | same |

---

## Project Structure

```
hr-policy-qa-bot/
├── src/
│   └── hr_bot/
│       ├── __init__.py
│       ├── config.py                  # pydantic-settings — all config from .env
│       ├── pipeline.py                # RAG orchestrator — search → generate
│       ├── ingestion/
│       │   ├── loader.py              # Load .md policy files with metadata
│       │   ├── chunker.py             # Recursive character splitting, 2048 chars
│       │   └── indexer.py             # Embed and store in ChromaDB
│       ├── retrieval/
│       │   └── searcher.py            # Embed query, cosine search, ranked results
│       ├── generation/
│       │   ├── providers.py           # LLMProvider ABC + Groq, Gemini, Claude
│       │   └── prompt.py              # SYSTEM_PROMPT + PromptBuilder
│       └── api/
│           ├── schemas.py             # QueryRequest, QueryResponse, HealthResponse
│           ├── routes.py              # POST /api/v1/query, GET /api/v1/health
│           └── main.py                # FastAPI app factory + lifespan handler
├── scripts/
│   └── ingest.py                      # CLI — rebuild ChromaDB index
├── data/
│   ├── policy_docs/                   # 16 .md policy files (source of truth)
│   └── vector_store/                  # ChromaDB — 62 chunks, 384-dim embeddings
├── tests/
│   ├── unit/                          # Chunker, prompt, pipeline (mock LLM)
│   └── integration/                   # Full pipeline tests
├── .env                               # Real secrets — gitignored
├── .env.example                       # Template — committed
├── pyproject.toml                     # Poetry config
├── poetry.lock                        # Locked deps — committed
├── pyrightconfig.json                 # Pylance basic mode
└── .vscode/settings.json              # VS Code interpreter config
```

---

## Reproducing From Scratch

### Prerequisites

- macOS with Homebrew
- pyenv (`brew install pyenv`)
- Poetry (`brew install poetry`)
- Python 3.11.9 via pyenv
- VS Code with Python extension
- Groq API key from console.groq.com (free)
- Anthropic API key from console.anthropic.com (production)

### Step 1 — Clone and branch

```bash
git clone <your-repo-url>
cd hr-policy-qa-bot
git checkout -b v2-production
```

### Step 2 — Pin Python version

```bash
pyenv local 3.11.9
```

### Step 3 — Install dependencies

```bash
poetry install
```

### Step 4 — Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-from-console.groq.com
GROQ_MODEL=llama-3.3-70b-versatile
ANTHROPIC_API_KEY=your-key (production)
CLAUDE_MODEL=claude-sonnet-4-6
GEMINI_API_KEY=your-key (needs billing enabled)
GEMINI_MODEL=gemini-2.0-flash
CHROMA_PATH=./data/vector_store
POLICY_DOCS_PATH=./data/policy_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
ENVIRONMENT=development
ANONYMIZED_TELEMETRY=False
```

### Step 5 — Configure VS Code interpreter

```bash
poetry env info --path   # copy this path
```

`Cmd+Shift+P` → `Python: Select Interpreter` → paste path + `/bin/python`

### Step 6 — Build the vector index

```bash
python -m scripts.ingest
```

To force a full rebuild after policy changes:

```bash
python -m scripts.ingest --reset
```

### Step 7 — Start the API

```bash
uvicorn hr_bot.api.main:app --reload --port 8000
```

### Step 8 — Test it

```bash
# Health check
curl http://127.0.0.1:8000/api/v1/health | python -m json.tool

# Query
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many vacation days do employees get?"}' | python -m json.tool
```

---

## API Reference

### `POST /api/v1/query`

**Request:**
```json
{
  "question": "How many vacation days do employees get?",
  "k": 5
}
```

**Response:**
```json
{
  "answer": "According to the Benefits And Perks policy, 37signals offers 20 days of vacation per year.",
  "sources": ["benefits-and-perks.md", "how-we-work.md"],
  "chunks_used": 5,
  "query": "How many vacation days do employees get?"
}
```

### `GET /api/v1/health`

**Response:**
```json
{
  "status": "ok",
  "provider": "groq",
  "environment": "development",
  "index_path": "./data/vector_store"
}
```

---

## Switching LLM Providers

Change one line in `.env`, restart the server. Zero code changes.

```
LLM_PROVIDER=groq    # development (free, fast)
LLM_PROVIDER=gemini  # alternative dev (needs Google billing)
LLM_PROVIDER=claude  # production
```

---

## Running Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=hr_bot

# Specific file
poetry run pytest tests/unit/test_chunker.py -v
```

---

## Design Patterns Applied

| Pattern | Where | Why |
|---|---|---|
| Abstract Base Class | `LLMProvider` in `providers.py` | Enforces `generate()` contract on all providers |
| Factory | `get_llm_provider()` | One function returns correct provider from config |
| Strategy | Provider abstraction | Swap LLM backend via config, zero code changes |
| Singleton | `get_settings()` with `@lru_cache` | Config read once at startup, shared everywhere |
| Dependency Injection | `run_pipeline(llm=None)` | Injectable LLM makes unit testing easy |
| Fail Fast | API key validation in provider `__init__` | Crash at startup with clear message, not mid-request |
| Dataclass | `Document`, `Chunk`, `SearchResult`, `RAGResponse` | Typed containers — no dict soup |
| Separation of Concerns | ingestion / retrieval / generation / api | Each layer independently testable and replaceable |

---

## Key Design Decisions

### Why Poetry over pip
Poetry resolves the full dependency graph and writes `poetry.lock` — identical environments guaranteed across machines. `requirements.txt` has no conflict resolution and no lock.

### Why ChromaDB over FAISS
FAISS requires `allow_dangerous_deserialization=True` (pickle exploit). ChromaDB stores vectors and metadata together, persists to disk safely, and supports metadata filtering.

### Why provider abstraction
Adding a new LLM provider means one new class + one line in the factory. The pipeline never knows which provider it's using — swap Groq for Claude in production with a single env var change.

### Why pydantic-settings
All config validated at startup. Missing API keys fail immediately with clear error messages, not mid-request with cryptic API errors.

### Why structlog
Structured JSON logs are searchable and alertable in production. Every log event carries typed fields — queryable in any log aggregator.

### Chunk size: 2048 chars with 200-char overlap
Previous version used 300 chars — too small, split sentences mid-thought. 2048 chars (~512 tokens) captures full policy paragraphs. Overlap ensures context isn't lost at boundaries.

---

## What Was Deliberately Removed

| Removed | Reason |
|---|---|
| LoRA fine-tuned T5 model | Requires GPU, needs retraining on policy changes. RAG + frontier model is strictly better. |
| Gradio UI | Not a production API. No request validation, no auth, `share=True` exposes data publicly. |
| FAISS | Replaced by ChromaDB — safer, persistent, metadata-aware. |
| torch / transformers / peft | Only needed for local model training. Removed ~2GB of dependencies. |
| Hardcoded paths | All paths in `.env` via pydantic-settings. |

---

## Git History

| Commit | Description |
|---|---|
| `483b023` | docs: update README with completed layers and build progress |
| `65479a7` | docs: add comprehensive README with architecture and reproduction steps |
| `f9d9b7d` | feat: add ingestion layer — loader, chunker, indexer |
| `427b1cf` | feat: add retrieval layer — searcher with ChromaDB |
| `1adbba8` | feat: initialize v2 production structure |

---

*Last updated: API layer complete. Next: unit tests, Dockerfile.*
