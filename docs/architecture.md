# Agentic RAG architecture

**Status: In development.**

The diagram describes the target architecture for version 1. It is a design overview, not an inventory of implemented features.

![Target Agentic RAG architecture](architecture-agentic-rag.png)

Clients will communicate with a FastAPI backend. The planned agent layer uses LangGraph to route and refine queries, select sources, generate answers, and evaluate results with retries where appropriate. Potential sources include document retrieval, MCP tools, and web search. Providers are intended to be interchangeable, with local embeddings and Chroma supporting document retrieval. The diagram also outlines planned observability, tests, and deployment components.

The current implementation provides the FastAPI skeleton, health endpoint, a common interface with Codex CLI and Ollama adapters, local embeddings, document ingestion, and persistent Chroma storage. Chat and document endpoints, LangGraph orchestration, answer generation, MCP, web search, and the observability and deployment integrations remain planned.
