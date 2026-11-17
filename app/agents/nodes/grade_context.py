from collections.abc import Callable

from app.agents.prompts.grader import build_context_grader_prompt
from app.agents.state import AgentState
from app.core.config import CONTEXT_RELEVANT
from app.integrations.llm.base import LLMProvider


def create_grade_context_node(
    llm: LLMProvider,
) -> Callable[[AgentState], AgentState]:
    """Create a node that filters retrieved chunks for question relevance.

    Args:
        llm (LLMProvider): Model used to grade each retrieved chunk.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph context grading node.
    """

    def grade(state: AgentState) -> AgentState:
        relevant = []
        for chunk in state.get("chunks", []):
            result = llm.generate(
                build_context_grader_prompt(state["question"], chunk.text)
            )
            if result.strip().upper() == CONTEXT_RELEVANT:
                relevant.append(chunk)
        return {"relevant_chunks": relevant}

    return grade
