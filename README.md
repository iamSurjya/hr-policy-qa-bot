cat > README.md << 'EOF'
# HR Policy Q&A Bot — Production RAG System

A production-grade HR policy question-answering bot built with RAG (Retrieval Augmented Generation). Employees ask questions in natural language and get accurate answers grounded strictly in company policy documents.

---

## Architecture
```
┌─────────────────────────────────────────────────────┐
│                    Client Layer                      │
│         FastAPI REST endpoint (POST /query)          │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                     API Layer                        │
│         Pydantic validation + auth middleware        │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                    Core Layer                        │
│                                                      │
│   Retriever → Reranker → Generator → Validator       │
│   (embed+search) (score)  (Claude/  (ground truth)  │
│                           Gemini)                    │
└──────┬──────────────────────────┬───────────────────┘
       │                          │
┌──────▼──────┐          ┌────────▼────────┐
│  Data Layer │          │   Logs / Evals  │
│  ChromaDB   │          │   structlog     │
│  (vectors + │          │   JSON output   │
│   metadata) │          └─────────────────┘
└─────────────┘
```

## Tech Stack

| Layer | Dev | Production |
|---|---|---|
| LLM | Gemini 2.0 Flash (free) | Claude Sonnet (switch via config) |
| Embeddings | all-MiniLM-L6-v2 (local, free) | same or voyage-3 |
| Vector DB | ChromaDB (local) | same or pgvector |
| API | FastAPI | same |
| Validation | Pydantic v2 | same |
| Logging | structlog JSON | same |
| Package mgmt | Poetry | same |
| Python | 3.11.9 via pyenv | same |

---

## Project Structure
```
hr-policy-qa-bot/
├── src/
│   └── hr_bot/
│       ├── config.py               # All settings via pydantic-settings
│       ├── ingestion/
│       │   ├── loader.py           # Load .md files with metadata
│       │   ├── chunker.py          # Chunk text with overlap
│       │   └── indexer.py          # Embed and store in ChromaDB
│       ├── retrieval/
│       │   ├── embedder.py         # Embed query
│       │   ├── searcher.py         # Vector search
│       │   └── reranker.py         # Cross-encoder reranking
│       ├── generation/
│       │   ├── providers.py        # Gemini + Claude (swap via config)
│       │   └── prompt.py           # System prompt + prompt builder
│       ├── pipeline.py             # Orchestrates full RAG flow
│       └── api/
│           ├── main.py             # FastAPI app factory
│           ├── routes.py           # POST /query, GET /health
│           └── schemas.py          # Pydantic request/response models
├── data/
│   └── policy_docs/                # Raw .md policy files (source of truth)
├── tests/
│   ├── unit/                       # Test individual functions
│   └── integration/                # Test full pipeline
├── .env.example                    # Template — commit this
├── .env                            # Real secrets — never commit
├── pyproject.toml                  # Dependencies + project metadata
├── poetry.lock                     # Exact locked versions — commit this
├── pyrightconfig.json              # VS Code type checking config
└── .vscode/settings.json           # VS Code interpreter config
```

---

## Reproducing This Project From Scratch

### Prerequisites

- macOS
- pyenv installed (`brew install pyenv`)
- Poetry installed (`brew install poetry` or official installer)
- Python 3.11.9 via pyenv
- VS Code with Python extension (ms-python.python)
- Gemini API key from aistudio.google.com (free)
- Anthropic API key from console.anthropic.com (for production)

### Step 1 — Clone and create branch
```bash
git clone <your-repo-url>
cd hr-policy-qa-bot
git checkout -b v2-production
```

### Step 2 — Set Python version
```bash
pyenv local 3.11.9
```

This writes a `.python-version` file ensuring this project always uses 3.11.9.

### Step 3 — Create project structure
```bash
mkdir -p src/hr_bot/ingestion
mkdir -p src/hr_bot/retrieval
mkdir -p src/hr_bot/generation
mkdir -p src/hr_bot/api
mkdir -p tests/unit
mkdir -p tests/integration

touch src/hr_bot/__init__.py
touch src/hr_bot/ingestion/__init__.py
touch src/hr_bot/retrieval/__init__.py
touch src/hr_bot/generation/__init__.py
touch src/hr_bot/api/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py
```

### Step 4 — Install dependencies
```bash
poetry install
```

This creates an isolated virtual environment and installs all packages from `poetry.lock`. Guaranteed identical environment every time.

### Step 5 — Configure environment
```bash
cp .env.example .env
```

Edit `.env` and fill in your real API keys:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-real-key-here
ANTHROPIC_API_KEY=your-real-key-here (for production)
```

### Step 6 — Configure VS Code

1. `Cmd+Shift+P` → `Python: Select Interpreter`
2. Click "Enter interpreter path"
3. Run `poetry env info --path` to get the path
4. Enter: `<path>/bin/python`
5. `Cmd+Shift+P` → `Python: Restart Language Server`

### Step 7 — Verify installation
```bash
python -c "import fastapi; import chromadb; import google.genai; print('all good')"
```

### Step 8 — Ingest policy documents
```bash
# Coming soon — run once to build the vector store
python -m scripts.ingest
```

### Step 9 — Run the API
```bash
# Coming soon
uvicorn hr_bot.api.main:app --reload
```

---

## Key Design Decisions & Why

### Why Poetry over pip/requirements.txt
Poetry resolves the full dependency graph and writes `poetry.lock` — a contract guaranteeing identical package versions on every machine. `requirements.txt` has no conflict resolution and no lock file.

### Why src/ layout
Forces proper package installation. Without it, Python finds your package via the filesystem even if it's not installed correctly — masking packaging bugs until production.

### Why ChromaDB over FAISS
FAISS is a pure similarity search library with no metadata storage, no persistence layer, and a pickle-based security vulnerability (`allow_dangerous_deserialization=True`). ChromaDB stores vectors and metadata together, persists to disk safely, and supports metadata filtering.

### Why provider abstraction for LLM
One config variable (`LLM_PROVIDER=gemini` or `claude`) switches the entire LLM backend. The pipeline never knows which provider it's using. Adding a third provider means one new class and one line in the factory — nothing else changes.

### Why pydantic-settings for config
All configuration validated at startup. Missing API keys fail immediately with clear error messages, not mid-request with cryptic API errors. No scattered `os.getenv()` calls across files.

### Why structlog over print()
Structured JSON logs are searchable, filterable, and alertable in production log aggregators. `print()` is unstructured noise. Every log event carries fields: `{"event": "query_received", "query": "...", "timestamp": "..."}`.

### Chunk size: 512 tokens with 50 token overlap
Your old code used 300 characters (~75 tokens) — too small, splitting sentences mid-thought. 512 tokens captures full policy paragraphs while staying well within embedding model limits. 50 token overlap ensures context isn't lost at chunk boundaries.

---

## Switching to Production (Claude)

Change one line in `.env`:
```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your-real-anthropic-key
```

Restart the server. No code changes required.

---

## Running Tests
```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=hr_bot

# Specific test file
poetry run pytest tests/unit/test_chunker.py -v
```

---

## What Was Deliberately Left Out (and Why)

| Removed | Reason |
|---|---|
| LoRA fine-tuned model | Requires GPU, needs retraining on policy changes, can hallucinate. RAG + frontier model is strictly better for this use case. |
| Gradio UI | Not a production API. No request validation, no auth, `share=True` exposes data publicly. Replaced with FastAPI. |
| FAISS | Replaced by ChromaDB — safer, persistent, metadata-aware. |
| torch / transformers / peft | Only needed for local model training. Removed ~2GB of dependencies. |
| Hardcoded paths | All paths now in `.env` via pydantic-settings. |

---

## Git Commit History

| Commit | Description |
|---|---|
| `1adbba8` | feat: initialize v2 production structure |
| `be48a60` | old codebase: with correct environment |

---

*This README is updated at each stage of the build. Last updated: ingestion layer in progress.*