import re
from collections.abc import Callable

from app.agents.prompts.generator import (
    ANSWER_RETRY_FEEDBACK,
    build_rag_prompt,
)
from app.agents.state import AgentState
from app.core.config import (
    CITATION_MARKER_PATTERN,
    VALID_CITATION_PATTERN,
)
from app.integrations.llm.base import LLMProvider
from app.schemas.chat import Citation


def create_generate_node(llm: LLMProvider) -> Callable[[AgentState], AgentState]:
    """Create a node that generates an answer and validates citation IDs.

    Args:
        llm (LLMProvider): Model used to generate a document-grounded answer.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph answer generation node.
    """
    citation_marker = re.compile(CITATION_MARKER_PATTERN)
    valid_marker = re.compile(VALID_CITATION_PATTERN)

    def generate(state: AgentState) -> AgentState:
        chunks = state.get("relevant_chunks", [])
        context = [
            {"id": f"S{index}", "text": chunk.text}
            for index, chunk in enumerate(chunks, start=1)
        ]
        prompt = build_rag_prompt(
            state["question"], context, state.get("generation_feedback", "")
        )
        answer = llm.generate(prompt).strip()
        citations = []
        valid = True
        cited_ids = []
        for marker in citation_marker.findall(answer):
            match = valid_marker.fullmatch(marker)
            if not match or int(match.group(1)) > len(chunks):
                valid = False
                continue
            identifier = f"S{match.group(1)}"
            if identifier not in cited_ids:
                cited_ids.append(identifier)
        if not cited_ids:
            valid = False
        for identifier in cited_ids:
            chunk = chunks[int(identifier[1:]) - 1]
            citations.append(
                Citation(
                    id=identifier,
                    source=chunk.source,
                    chunk_index=chunk.chunk_index,
                    excerpt=chunk.text,
                )
            )
        return {
            "answer": answer,
            "citations": citations,
            "citations_valid": valid,
            "answer_grounded": False,
            "generation_attempts": state.get("generation_attempts", 0) + 1,
            "generation_feedback": ANSWER_RETRY_FEEDBACK,
        }

    return generate
