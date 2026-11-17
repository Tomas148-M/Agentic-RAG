from typing import TypedDict

from app.rag.retriever import RetrievedChunk
from app.schemas.chat import Citation


class AgentState(TypedDict, total=False):
    """Data passed between the LangGraph RAG nodes."""

    question: str
    query: str
    route: str
    selected_sources: list[str]
    mcp_tool: str
    mcp_arguments: dict[str, object]
    chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    answer: str
    citations: list[Citation]
    feedback: str
    generation_feedback: str
    retry_count: int
    answer_attempts: int
    answer_valid: bool
