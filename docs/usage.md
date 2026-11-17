# Usage and configuration

**Status: In development.** For the system overview and LangGraph flow, see [architecture.md](architecture.md).

## Install and run

Python 3.10 or later is required. In the project root, create a virtual environment and install the application with development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Optionally copy `.env.example` to `.env` and adjust settings. Environment variables take precedence over `.env` values. Run the API with:

```powershell
uvicorn app.main:app --reload
```

The health endpoint is `http://127.0.0.1:8000/api/v1/health`; interactive API documentation is at `http://127.0.0.1:8000/docs`. Health currently confirms that the API is running and does not check external services.

## LLM provider

Choose `ollama` or `codex` with `AGENTIC_RAG_LLM_PROVIDER`. Settings load at startup, so restart the application after changing `.env`.

For Ollama, run the server and download the configured model (default: `llama3.2`). Set `AGENTIC_RAG_OLLAMA_BASE_URL` to change its address. For Codex, install and authenticate Codex CLI on the API host; the executable path and model can also be configured. Codex runs as an agent, and its read-only sandbox may not be available on every platform. It may read local files, so isolate it from sensitive data when handling untrusted questions or documents.

Both adapters expose `generate(prompt: str) -> str` and raise `LLMProviderError` on failure. Calls are synchronous and should run outside the event loop in asynchronous endpoints. Use `get_llm_provider` for dependency injection.

## Document ingestion

Ingest a UTF-8 `.txt`, `.md`, or text-based `.pdf` file, or recursively ingest a directory:

```powershell
python -m scripts.ingest_documents data/documents
```

Sentence Transformers runs on CPU. On first use, model weights may be downloaded from Hugging Face; subsequent runs can use the cache. PyTorch dependencies can require substantial disk space.

Configure `AGENTIC_RAG_EMBEDDING_MODEL`, `AGENTIC_RAG_CHROMA_PATH`, `AGENTIC_RAG_CHROMA_COLLECTION`, `AGENTIC_RAG_CHUNK_SIZE`, and `AGENTIC_RAG_CHUNK_OVERLAP` in `.env` as needed. Chunk sizes are characters, not model tokens, and long chunks may be truncated. Overlap must be smaller than chunk size.

Chroma data defaults to `data/chroma`. Reingesting a source path replaces its chunks, including stale chunks. Removing or renaming a source file does not remove old entries. Source paths are stored in metadata. When changing embedding models, use a new collection and reingest documents. PDF ingestion does not perform OCR or preserve page numbers. Avoid concurrent ingestion of the same source.

## Chat and citations

After ingestion and startup, send a question to `POST /api/v1/chat`:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/chat `
  -ContentType 'application/json' -Body '{"question":"What do the documents say?"}'
```

The response contains an `answer` with citation markers such as `[S1]` and a `citations` list with source path, chunk index, and excerpt. Source paths and excerpts are returned to the caller. In `basic` mode, an empty index returns a message with no citations and does not call the LLM. Retrieval failures return HTTP 503; provider failures and invalid citations return HTTP 502.

Citation validation recognizes source-style markers such as `[S1]`; other formats are not comprehensively validated. Labels such as `[C1]` may be mistaken for citations and cause rejection.

## LangGraph mode

Set `AGENTIC_RAG_RAG_MODE=agentic` to route chat through the LangGraph workflow. It classifies the request, rewrites the search query, retrieves and grades excerpts, generates a cited answer, and evaluates grounding. Retries are bounded by `AGENTIC_RAG_AGENTIC_MAX_RETRIES` (default: 2). Out-of-scope requests receive a fixed response. If no relevant excerpts are found after retries, the endpoint returns an empty citation list. If answer evaluation still fails, it returns HTTP 502. The default `basic` mode keeps the single-retrieval pipeline.

Routing, relevance grading, generation, and evaluation use the configured LLM. These model judgments do not guarantee factual correctness or protection against prompt injection. Basic mode verifies that cited IDs exist among retrieved chunks, but does not verify that every claim is supported. Similarity search has no relevance threshold, so retrieved chunks may be unrelated. Documents are untrusted input; prompt instructions alone cannot prevent prompt injection.

## MCP and web search

MCP and web search are available in `agentic` mode. Enable web search and provide a Tavily API key with `AGENTIC_RAG_WEB_SEARCH_ENABLED=true` and `AGENTIC_RAG_WEB_SEARCH_API_KEY`. The endpoint, result count, request timeout, and maximum excerpt size are configurable in `.env.example`.

To connect one MCP Streamable HTTP server, set `AGENTIC_RAG_MCP_SERVER_URL`, an explicit JSON array of retrieval-only tool names in `AGENTIC_RAG_MCP_ALLOWED_TOOLS`, and, when required, `AGENTIC_RAG_MCP_AUTH_TOKEN`. The allowlist is required: advertised tools outside it cannot be selected or called. The client performs MCP initialization, paginates tool discovery, and calls selected allowlisted tools after checking their input schemas. If the optional server is unavailable during source discovery, the agent excludes MCP and can still use documents or web search.

The source selector chooses local documents, web search, an MCP tool, or a specific combination of these sources. Only the selected sources execute. Results from selected sources enter the same relevance grading, answer generation, and citation checks. Web citations identify the result URL; MCP citations identify the server tool. Failures during an explicitly selected external tool call return HTTP 503. Keep MCP servers and credentials under your control. MCP and web search are not used by the default `basic` mode.

## Development checks

```powershell
python -m pytest
python -m ruff check .
```

See [AGENTS.md](../AGENTS.md) for the project change workflow.
