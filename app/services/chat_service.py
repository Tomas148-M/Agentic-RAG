import re

from app.agents.prompts.generator import build_rag_prompt
from app.agents.tools.registry import ToolRegistry
from app.core.config import (
    CITATION_MARKER_PATTERN,
    DEFAULT_AGENTIC_MAX_RETRIES,
    RAG_MODE_AGENTIC,
    RAG_MODE_BASIC,
    VALID_CITATION_PATTERN,
)
from app.integrations.llm.base import LLMProvider, LLMProviderError
from app.observability.phoenix import PhoenixObservability
from app.rag.retriever import DocumentRetriever
from app.schemas.chat import ChatResponse, Citation

_CITATION_MARKER = re.compile(CITATION_MARKER_PATTERN)
_VALID_MARKER = re.compile(VALID_CITATION_PATTERN)


class ChatService:
    """Answer questions from retrieved documents using an LLM provider."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        llm: LLMProvider,
        rag_mode: str = RAG_MODE_BASIC,
        max_retries: int = DEFAULT_AGENTIC_MAX_RETRIES,
        tool_registry: ToolRegistry | None = None,
        observability: PhoenixObservability | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.rag_mode = rag_mode
        self.max_retries = max_retries
        self.tool_registry = tool_registry
        self.observability = observability

    def answer(self, question: str) -> ChatResponse:
        """Retrieve context, generate an answer, and validate its citations.

        Args:
            question (str): Non-empty question to answer.

        Returns:
            ChatResponse: Answer and referenced document chunks.

        Raises:
            RetrievalError: If document retrieval fails.
            LLMProviderError: If generation fails or citations are invalid.
        """
        if self.observability is None:
            return self._answer_untraced(question)
        with self.observability.chat_trace(question, self.rag_mode) as span:
            response = self._answer_untraced(question)
            self.observability.record_chat_output(
                span, response.answer, len(response.citations)
            )
            return response

    def _answer_untraced(self, question: str) -> ChatResponse:
        """Run the selected RAG pipeline without creating a chat-level span."""
        if self.rag_mode == RAG_MODE_AGENTIC:
            return self._answer_with_agent(question)
        chunks = self.retriever.retrieve(question)
        if not chunks:
            return ChatResponse(
                answer="No relevant documents were found.", citations=[]
            )

        context = [
            {"id": f"S{i}", "text": chunk.text}
            for i, chunk in enumerate(chunks, start=1)
        ]
        prompt = build_rag_prompt(question, context)
        answer = self.llm.generate(prompt).strip()
        markers = _CITATION_MARKER.findall(answer)
        cited_ids: list[str] = []
        for marker in markers:
            match = _VALID_MARKER.fullmatch(marker)
            if not match or int(match.group(1)) > len(chunks):
                raise LLMProviderError("Generated answer contains an invalid citation")
            identifier = f"S{match.group(1)}"
            if identifier not in cited_ids:
                cited_ids.append(identifier)
        if not cited_ids:
            raise LLMProviderError("Generated answer contains no valid citations")
        citations = []
        for identifier in cited_ids:
            chunk = chunks[int(identifier[1:]) - 1]
            citations.append(Citation(
                id=identifier,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                excerpt=chunk.text,
            ))
        return ChatResponse(answer=answer, citations=citations)

    def _answer_with_agent(self, question: str) -> ChatResponse:
        """Run LangGraph and convert its final state to an API response."""
        from app.agents.graph import build_agentic_rag_graph

        result = build_agentic_rag_graph(
            self.retriever,
            self.llm,
            self.max_retries,
            self.tool_registry,
        )(question)
        return ChatResponse(
            answer=result["answer"],
            citations=result.get("citations", []),
        )
