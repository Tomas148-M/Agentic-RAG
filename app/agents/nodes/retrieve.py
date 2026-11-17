from collections.abc import Callable

from app.agents.state import AgentState
from app.agents.tools.registry import ToolRegistry
from app.rag.retriever import DocumentRetriever


def create_retrieve_node(
    retriever: DocumentRetriever | None = None,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[AgentState], AgentState]:
    """Create a node that retrieves chunks for the current refined query.

    Args:
        retriever (DocumentRetriever | None): Document search dependency.
        tool_registry (ToolRegistry | None): Optional selected-source registry.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph retrieval node.
    """

    def retrieve(state: AgentState) -> AgentState:
        if tool_registry is not None:
            return {
                "chunks": tool_registry.execute(
                    state["query"],
                    state.get("route", ""),
                    state.get("mcp_tool", ""),
                    state.get("mcp_arguments", {}),
                    state.get("selected_sources"),
                )
            }
        if retriever is None:
            raise ValueError("A retriever or tool registry is required")
        return {"chunks": retriever.retrieve(state["query"])}

    return retrieve
