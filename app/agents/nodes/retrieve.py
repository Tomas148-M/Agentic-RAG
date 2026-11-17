from collections.abc import Callable

from app.agents.state import AgentState
from app.rag.retriever import DocumentRetriever


def create_retrieve_node(
    retriever: DocumentRetriever,
) -> Callable[[AgentState], AgentState]:
    """Create a node that retrieves chunks for the current refined query.

    Args:
        retriever (DocumentRetriever): Search dependency for stored documents.

    Returns:
        Callable[[AgentState], AgentState]: LangGraph retrieval node.
    """

    def retrieve(state: AgentState) -> AgentState:
        return {"chunks": retriever.retrieve(state["query"])}

    return retrieve
