# We Made What

Word-mixing web game powered by embedding vector arithmetic. Combine and subtract words to discover new ones (like InfiniteCraft).

## Project Structure

- `backend/` — Python 3.13 + FastAPI + sentence-transformers + FAISS
- `frontend/` — (coming soon) React 19 + TypeScript + Vite
- `scripts/` — Vocabulary build scripts
- `plans/` — Architecture documentation

## Backend Setup

```bash
cd backend
uv sync --extra dev
```

### Build the vocabulary index

1. Fetch pop culture + slang data (slow, rate-limited):
   ```bash
   uv run python scripts/fetch-popculture.py
   ```

2. Build FAISS index (requires GPU for embedding model):
   ```bash
   uv run python scripts/build-vocab.py
   ```

   Environment variables:
   - `WMW_EMBEDDING_MODEL` — Model name (default: `Qwen/Qwen3-Embedding-8B`)
   - `WMW_EMBEDDING_DEVICE` — `cuda` or `cpu` (default: `cuda`)
   - `WMW_EMBEDDING_DIM` — Matryoshka dimension (default: `512`)

3. Evaluate word arithmetic quality:
   ```bash
   uv run python scripts/evaluate-model.py
   ```

### Run Tests

```bash
cd backend
uv run pytest tests/ -v
```

### Run Server

```bash
cd backend
uv run uvicorn wmw.main:app --reload
```

## Key Architecture

- **Embedding provider**: Swappable via `EmbeddingProvider` protocol (`backend/src/wmw/embedding/provider.py`)
- **Vocabulary index**: FAISS `IndexFlatIP` with L2-normalized vectors for cosine similarity
- **Word mixer**: Vector arithmetic (add/subtract) with input exclusion
- **Vocab sources**: NLTK nouns (WordNet-filtered), Wikidata SPARQL (pop culture, memes, subcultures), HuggingFace (Gen-Z slang)

## Conventions

- Backend uses `uv` for package management
- All scripts run from repo root: `uv run python scripts/<script>.py`
- Generated data goes in `backend/src/wmw/data/` (gitignored)
- Tests use fake providers with deterministic random vectors (no GPU needed)
