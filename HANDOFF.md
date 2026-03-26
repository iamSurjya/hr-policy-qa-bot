# HR Policy Q&A Bot — Session Handoff Document

## Purpose
This document summarizes everything built in session 1 so the next session can continue without re-explaining context, decisions, or architecture.

---

## Project Location
```
~/Documents/projects/hr-policy-qa-bot
Branch: v2-production
```

---

## What This Project Is
A production-grade HR Policy Q&A bot rebuilt from scratch. Employees ask questions in natural language, the system retrieves relevant policy chunks from ChromaDB, and an LLM generates a grounded answer with source attribution.

The original project was a script-based RAG bot with hardcoded paths, a fine-tuned T5 model via LoRA, Gradio UI, and FAISS. Everything has been replaced with a proper production architecture.

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| LLM (dev) | Groq — llama-3.3-70b-versatile | Free, fast, no billing required |
| LLM (prod) | Claude Sonnet (claude-sonnet-4-6) | Best reasoning, switch via config |
| LLM (alt dev) | Gemini 2.0 Flash | Code preserved, needs billing enabled |
| Embeddings | all-MiniLM-L6-v2 | Free, local, good quality |
| Vector DB | ChromaDB (PersistentClient) | Metadata-aware, safe, no pickle exploit |
| API | FastAPI | Production standard, async, typed |
| Validation | Pydantic v2 + pydantic-settings | Config + request validation |
| Logging | structlog JSON | Structured, searchable logs |
| Package mgmt | Poetry 2.3.2 | Locked deps, isolated virtualenv |
| Python | 3.11.9 via pyenv | Pinned, isolated |
| Type checking | Pylance basic mode | Catches real bugs, not cosmetic |

---

## Environment Setup

```bash
# Python version
pyenv local 3.11.9

# Virtual environment
poetry install

# VS Code interpreter path
/Users/surajchoudhary/Library/Caches/pypoetry/virtualenvs/hr-bot-OGI5iPGC-py3.11/bin/python

# Run commands
python <script>   # inside VS Code terminal (virtualenv auto-activated)
```

### .env variables (all required)
```
LLM_PROVIDER=groq
GEMINI_API_KEY=<key — needs billing enabled on Google Cloud>
ANTHROPIC_API_KEY=<key — for production>
GEMINI_MODEL=gemini-2.0-flash
CLAUDE_MODEL=claude-sonnet-4-6
GROQ_API_KEY=<working key — get from console.groq.com>
GROQ_MODEL=llama-3.3-70b-versatile
CHROMA_PATH=./data/vector_store
POLICY_DOCS_PATH=./data/policy_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
LOG_LEVEL=INFO
ENVIRONMENT=development
ANONYMIZED_TELEMETRY=False
```

---

## Project Structure (current state)

```
hr-policy-qa-bot/
├── src/
│   └── hr_bot/
│       ├── __init__.py
│       ├── config.py                  ✅ DONE
│       ├── pipeline.py                ✅ DONE
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py              ✅ DONE
│       │   ├── chunker.py             ✅ DONE
│       │   └── indexer.py             ✅ DONE
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── searcher.py            ✅ DONE
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── providers.py           ✅ DONE
│       │   └── prompt.py              ✅ DONE
│       └── api/
│           └── __init__.py            ⬜ NOT STARTED
├── data/
│   ├── policy_docs/                   ✅ 16 .md files (source of truth)
│   └── vector_store/                  ✅ Built — 62 chunks indexed in ChromaDB
├── tests/
│   ├── unit/                          ⬜ NOT STARTED
│   └── integration/                   ⬜ NOT STARTED
├── .env                               ✅ Real secrets (gitignored)
├── .env.example                       ✅ Template (committed)
├── pyproject.toml                     ✅ Poetry config
├── poetry.lock                        ✅ Locked deps
├── pyrightconfig.json                 ✅ Pylance config
├── .vscode/settings.json              ✅ VS Code config
└── README.md                          ✅ Comprehensive, kept updated
```

---

## Files Completed — Key Details

### `config.py`
- Uses `pydantic-settings` `BaseSettings`
- Reads all values from `.env` automatically
- `@lru_cache` on `get_settings()` — reads `.env` once at startup
- `settings` singleton imported everywhere as `from hr_bot.config import settings`
- Contains: `llm_provider`, `gemini_*`, `anthropic_*`, `groq_*`, `chroma_path`, `policy_docs_path`, `embedding_model`, `log_level`, `environment`

### `generation/providers.py`
- `LLMProvider` — abstract base class with `@abstractmethod generate()`
- `GeminiProvider` — uses `google.genai` SDK, accesses response via `candidates[0].content.parts[0].text`
- `ClaudeProvider` — uses `anthropic` SDK, type-narrows with `isinstance(block, TextBlock)`
- `GroqProvider` — uses `groq` SDK, OpenAI-compatible chat completions
- `get_llm_provider()` — factory pattern, reads `LLM_PROVIDER` from config
- Switch provider: change `LLM_PROVIDER` in `.env`, restart

### `generation/prompt.py`
- `SYSTEM_PROMPT` — module-level constant, rules for HR bot behaviour
- `PromptBuilder.build()` — returns `tuple[str, str]` (system, user_prompt)
- Returns them separately because Claude has dedicated `system=` param, Gemini prepends

### `ingestion/loader.py`
- `Document` dataclass: `content`, `source` (filename), `title` (human readable)
- `load_documents()` — loads all `.md` and `.txt` from `policy_docs/`
- Sorted file loading — deterministic, reproducible
- Skips empty files and non-document files gracefully

### `ingestion/chunker.py`
- `Chunk` dataclass: `content`, `source`, `title`, `chunk_index`
- `RecursiveCharacterTextSplitter` — splits on `\n\n` first, then `\n`, then `. `, then words
- `CHUNK_SIZE=2048` chars, `CHUNK_OVERLAP=200` chars
- Skips chunks shorter than 50 chars
- Result: 62 chunks from 16 documents

### `ingestion/indexer.py`
- `build_index(chunks, reset=False)` — embeds and stores in ChromaDB
- `reset=True` — deletes collection first, use when policies change
- Guards against duplicate indexing
- Stores per chunk: id, document text, metadata (source, title, chunk_index)
- `# type: ignore` on ChromaDB imports — library has broken type stubs
- ChromaDB telemetry disabled via `ANONYMIZED_TELEMETRY=False` in `.env`

### `retrieval/searcher.py`
- `SearchResult` dataclass: `content`, `source`, `title`, `chunk_index`, `similarity`
- `search(query, k=5)` — embeds query, queries ChromaDB, returns ranked results
- Similarity = `1 - cosine_distance` (higher = more relevant)
- Defensive unpacking — guards against None fields from ChromaDB
- Sample result: "How many vacation days?" → top similarity 0.5774 from `benefits-and-perks.md`

### `pipeline.py`
- `RAGResponse` dataclass: `answer`, `sources`, `chunks_used`, `query`
- `run_pipeline(question, llm=None, k=5)` — full RAG flow
- Flow: `search()` → filter by `MIN_SIMILARITY=0.15` → `format_context()` → `PromptBuilder.build()` → `llm.generate()`
- `llm` param is injectable — makes testing easy (pass mock LLM)
- Returns sources list for attribution — employees can verify answers

---

## ChromaDB Index
- Location: `./data/vector_store/`
- Collection: `hr_policies`
- 62 chunks from 16 policy documents
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions, cosine similarity)
- To rebuild after policy changes:
```bash
python -c "
from hr_bot.ingestion.loader import load_documents
from hr_bot.ingestion.chunker import chunk_documents
from hr_bot.ingestion.indexer import build_index
docs = load_documents('./data/policy_docs')
chunks = chunk_documents(docs)
build_index(chunks, reset=True)
"
```

---

## Known Issues / Technical Debt

| Issue | Status | Notes |
|---|---|---|
| ChromaDB telemetry warnings | Suppressed via env var | `ANONYMIZED_TELEMETRY=False` |
| ChromaDB broken type stubs | `# type: ignore` applied | Library issue, not our code |
| Gemini free tier quota | Blocked | New Google project needs billing enabled |
| `SentenceTransformerEmbeddingFunction` import path changed | Fixed | Use `from chromadb.utils.embedding_functions import ...` with `# type: ignore` |

---

## Git History (v2-production branch)

```
483b023  docs: update README with completed layers and build progress
65479a7  docs: add comprehensive README with architecture and reproduction steps
f9d9b7d  feat: add ingestion layer - loader, chunker, indexer
427b1cf  feat: add retrieval layer - searcher with ChromaDB
1adbba8  feat: initialize v2 production structure
be48a60  (origin/main) old codebase: with correct environment
```

---

## What Needs to Be Done Next (in order)

### 1. Run and verify pipeline end-to-end (immediate next step)
```bash
python -c "
from hr_bot.pipeline import run_pipeline
response = run_pipeline('How many vacation days do employees get?')
print('ANSWER:', response.answer)
print('SOURCES:', response.sources)
print('CHUNKS USED:', response.chunks_used)
"
```

### 2. API layer — `src/hr_bot/api/`
Three files to create:

**`schemas.py`** — Pydantic request/response models:
```python
class QueryRequest(BaseModel):
    question: str
    k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    query: str
```

**`routes.py`** — FastAPI endpoints:
- `POST /query` — main endpoint, calls `run_pipeline()`
- `GET /health` — returns system status + index stats

**`main.py`** — FastAPI app factory:
- Lifespan handler — loads LLM provider once at startup (not per request)
- CORS middleware
- Mounts routes
- Startup validation — fail fast if API keys missing

### 3. `scripts/ingest.py` — CLI ingestion script
```bash
python -m scripts.ingest           # normal run
python -m scripts.ingest --reset   # full rebuild
```

### 4. Unit tests — `tests/unit/`
- `test_chunker.py` — test chunk sizes, overlap, metadata preservation
- `test_prompt.py` — test prompt builder output
- `test_pipeline.py` — test with mock LLM provider

### 5. `Dockerfile`
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --only main
COPY src/ ./src/
COPY data/policy_docs/ ./data/policy_docs/
CMD ["uvicorn", "hr_bot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6. Update README.md with completed API layer

---

## Design Patterns Used (for resume/interviews)

| Pattern | Where used | Why |
|---|---|---|
| Abstract Base Class | `LLMProvider` | Enforces interface contract on all providers |
| Factory Pattern | `get_llm_provider()` | One function returns correct provider from config |
| Strategy Pattern | Provider abstraction | Swap LLM backend via config, zero code changes |
| Singleton | `settings = get_settings()` | Config read once, shared everywhere |
| Dependency Injection | `run_pipeline(llm=None)` | Injectable LLM makes testing easy |
| Fail Fast | API key validation in `__init__` | Crash at startup with clear message, not mid-request |
| Dataclass | `Document`, `Chunk`, `SearchResult`, `RAGResponse` | Typed containers, no dict soup |
| Separation of Concerns | ingestion / retrieval / generation / api layers | Each layer independently testable and replaceable |

---

## How to Start the Next Session

Tell the assistant:
> "I am continuing the hr-policy-qa-bot project. Please read the HANDOFF.md file I am pasting below and continue from where we left off."

Then paste this entire document.
