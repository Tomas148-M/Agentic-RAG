# Agentic RAG

**Status: In development.**

FastAPI application for document ingestion and question answering with citations. It supports interchangeable LLM providers and basic or LangGraph-based RAG workflows.

![Target Agentic RAG architecture](docs/architecture-agentic-rag.png)

## Quick start

Requires Python 3.10 or later.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.main:app --reload
```

Health check: `http://127.0.0.1:8000/api/v1/health`

Interactive API docs: `http://127.0.0.1:8000/docs`

## Documentation

- [Architecture and LangGraph workflow](docs/architecture.md)
- [Setup, configuration, ingestion, chat, and limitations](docs/usage.md)

## Checks

```powershell
python -m pytest
python -m ruff check .
```
