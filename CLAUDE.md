# We Made What

Word-mixing web game powered by LLM word combination. Combine and subtract words to discover new ones (like InfiniteCraft).

## Project Structure

- `backend/` — Python 3.13 + FastAPI + llama-cpp-python
- `frontend/` — (coming soon) React 19 + TypeScript + Vite
- `models/` — GGUF model files (gitignored)
- `plans/` — Architecture documentation

## Backend Setup

```bash
cd backend
uv sync --extra dev
```

### Download the model

Download Gemma-4-E4B-Uncensored Q4_K_M GGUF (~5GB):

```bash
mkdir -p models
uv pip install huggingface-hub
uv run python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive',
    filename='Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf',
    local_dir='models',
)
"
```

Set the model path:
```bash
export WMW_MODEL_PATH=models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
```

### Lint, Format & Type Check

```bash
cd backend
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/
uv run ty check src/
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

- **LLM mixer**: Prompts a local LLM (Gemma-4-E4B via llama-cpp-python) to creatively combine/subtract words
- **Result caching**: Same word combination always returns the same result
- **LLM provider**: Swappable via `LLMProvider` protocol (`backend/src/wmw/llm/provider.py`)

## Conventions

- Backend uses `uv` for package management
- Linting: `ruff` (lint + format), type checking: `ty` (by Astral)
- Generated data goes in `backend/src/wmw/data/` (gitignored)
- Model files go in `models/` (gitignored)
- Tests use fake LLM providers with deterministic responses (no GPU needed)
- Use `/git-pr` skill before creating pull requests
