from collections.abc import Callable

from app.agents.prompts.rewriter import build_query_rewrite_prompt
from app.agents.state import AgentState
from app.core.config import MAX_QUERY_LENGTH
from app.integrations.llm.base import LLMProvider


def create_rewrite_query_node(
    llm: LLMProvider,
) -> Callable[[AgentState], AgentState]:
    """Create a node that refines a search query using retrieval feedback.

    Args:
        llm (LLMProvider): Model used to rewrite the request.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph query rewrite node.
    """

    def rewrite(state: AgentState) -> AgentState:
        prompt = build_query_rewrite_prompt(
            state["question"], state.get("feedback", "")
        )
        query = llm.generate(prompt).strip()
        if not query:
            query = state["question"]
        return {"query": query[:MAX_QUERY_LENGTH]}

    return rewrite
