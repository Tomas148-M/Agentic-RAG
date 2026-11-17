import re

from app.agents.prompts.generator import build_rag_prompt
from app.integrations.llm.base import LLMProvider, LLMProviderError
from app.rag.retriever import DocumentRetriever
from app.schemas.chat import ChatResponse, Citation

_CITATION_MARKER = re.compile(
    r"\[(?:[A-Za-z]\d+|[Ss][\s+-]+\d+)[^\]]*\]"
)
_VALID_MARKER = re.compile(r"\[S([1-9]\d*)\]")


class ChatService:
    """Answer questions from retrieved documents using an LLM provider."""

    def __init__(self, retriever: DocumentRetriever, llm: LLMProvider) -> None:
        self.retriever = retriever
        self.llm = llm

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
