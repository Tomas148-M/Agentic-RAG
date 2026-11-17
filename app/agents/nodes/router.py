from collections.abc import Callable

from app.agents.prompts.router import build_router_prompt
from app.agents.state import AgentState
from app.core.config import ROUTE_RETRIEVE, ROUTE_UNSUPPORTED
from app.integrations.llm.base import LLMProvider


def create_router_node(llm: LLMProvider) -> Callable[[AgentState], AgentState]:
    """Create a node that routes document questions to retrieval.

    Args:
        llm (LLMProvider): Model used to classify the request.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph routing node.
    """

    def route(state: AgentState) -> AgentState:
        result = llm.generate(build_router_prompt(state["question"]))
        choice = result.strip().upper()
        return {
            "route": (
                ROUTE_UNSUPPORTED
                if choice == ROUTE_UNSUPPORTED.upper()
                else ROUTE_RETRIEVE
            )
        }

    return route
