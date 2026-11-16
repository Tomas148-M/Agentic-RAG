# Agentic RAG

This project contains a FastAPI application skeleton, a health endpoint, and a common interface for LLM providers.

## Architecture

The diagram shows the target architecture; several components and endpoints are planned and are not yet implemented. See the [architecture documentation](docs/architecture.md) for the current scope and planned workflow.

![Target Agentic RAG architecture](docs/architecture-agentic-rag.png)

## Getting started

Python 3.10 or later is required. Create and activate a virtual environment in the project root, then install the application and development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Optionally, copy `.env.example` to `.env` and adjust the configuration. Environment variables take precedence over values in `.env`.

```powershell
uvicorn app.main:app --reload
```

The health endpoint is available at `http://127.0.0.1:8000/api/v1/health` and interactive API documentation at `http://127.0.0.1:8000/docs`. The endpoint confirms that the API is running; it does not currently check external services.

## LLM provider

Select a provider using `AGENTIC_RAG_LLM_PROVIDER` (`ollama` or `codex`); see `.env.example` for an example configuration. Settings are loaded when the application starts, so restart it after changing `.env`.

For Ollama, start the server and download the model specified by `AGENTIC_RAG_OLLAMA_MODEL` locally (default: `llama3.2`). Set `AGENTIC_RAG_OLLAMA_BASE_URL` to change the server address. For Codex, install and authenticate Codex CLI on the machine running the API; optionally configure the executable path and model. Codex is an agent with tools, and its read-only sandbox may not be available on every platform; it is not equivalent to a purely local LLM.

Both adapters expose a synchronous `generate(prompt: str) -> str` method and raise `LLMProviderError` on failure. The call blocks; run it outside the event loop when using it in an asynchronous endpoint. Use `get_llm_provider` for dependency injection. A chat endpoint and RAG integration have not yet been implemented.

## Tests

```powershell
python -m pytest
```

Run static analysis with `python -m ruff check .`. The workflow for implementation tasks is described in `AGENTS.md`.

## External links

- [Portál občana – sign in](https://obcan.portal.gov.cz/prihlaseni)
