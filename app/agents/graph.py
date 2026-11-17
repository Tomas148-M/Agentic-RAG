from collections.abc import Callable

from langgraph.graph import END, START, StateGraph

from app.agents.edges.routing import (
    route_after_classification,
    route_after_evaluation,
    route_after_grading,
)
from app.agents.nodes.evaluate import create_evaluate_node
from app.agents.nodes.generate import create_generate_node
from app.agents.nodes.grade_context import create_grade_context_node
from app.agents.nodes.retrieve import create_retrieve_node
from app.agents.nodes.rewrite_query import create_rewrite_query_node
from app.agents.nodes.router import create_router_node
from app.agents.prompts.rewriter import RETRIEVAL_RETRY_FEEDBACK
from app.agents.state import AgentState
from app.agents.tools.registry import ToolRegistry
from app.core.config import (
    ANSWER_ERROR_MESSAGE,
    END_NODE,
    EVALUATE_NODE,
    FAIL_NODE,
    GENERATE_NODE,
    GRADE_CONTEXT_NODE,
    NO_CONTEXT_NODE,
    NO_DOCUMENTS_ANSWER,
    RETRIEVE_NODE,
    RETRY_QUERY_NODE,
    REWRITE_NODE,
    ROUTE_NODE,
    UNSUPPORTED_NODE,
    UNSUPPORTED_REQUEST_ANSWER,
    get_agentic_recursion_limit,
)
from app.integrations.llm.base import LLMProvider, LLMProviderError
from app.rag.retriever import DocumentRetriever


def build_agentic_rag_graph(
    retriever: DocumentRetriever,
    llm: LLMProvider,
    max_retries: int,
    tool_registry: ToolRegistry | None = None,
) -> Callable[[str], AgentState]:
    """Build a bounded LangGraph workflow for document question answering.

    Args:
        retriever (DocumentRetriever): Document chunk search dependency.
        llm (LLMProvider): Model used for routing and evaluation tasks.
        max_retries (int): Retry limit for retrieval and generation recovery.
        tool_registry (ToolRegistry | None): Optional MCP and web tools.

    Returns:
        Callable[[str], AgentState]: Runner for a single user question.
    """
    workflow = StateGraph(AgentState)
    workflow.add_node(ROUTE_NODE, create_router_node(llm, tool_registry))
    workflow.add_node(REWRITE_NODE, create_rewrite_query_node(llm))
    workflow.add_node(
        RETRIEVE_NODE,
        create_retrieve_node(retriever, tool_registry),
    )
    workflow.add_node(GRADE_CONTEXT_NODE, create_grade_context_node(llm))
    workflow.add_node(GENERATE_NODE, create_generate_node(llm))
    workflow.add_node(EVALUATE_NODE, create_evaluate_node(llm))

    def retry_query(state: AgentState) -> AgentState:
        """Record that another retrieval attempt is needed."""
        return {
            "retry_count": state.get("retry_count", 0) + 1,
            "feedback": RETRIEVAL_RETRY_FEEDBACK,
        }

    def no_context(_: AgentState) -> AgentState:
        """Finish safely when no relevant excerpts were found."""
        return {
            "answer": NO_DOCUMENTS_ANSWER,
            "citations": [],
            "answer_valid": True,
        }

    def unsupported(_: AgentState) -> AgentState:
        """Respond without inventing a document-grounded answer."""
        return {
            "answer": UNSUPPORTED_REQUEST_ANSWER,
            "citations": [],
            "answer_valid": True,
        }

    def fail(_: AgentState) -> AgentState:
        """Raise a stable error when answer evaluation exhausts retries."""
        raise LLMProviderError(ANSWER_ERROR_MESSAGE)

    workflow.add_node(RETRY_QUERY_NODE, retry_query)
    workflow.add_node(NO_CONTEXT_NODE, no_context)
    workflow.add_node(UNSUPPORTED_NODE, unsupported)
    workflow.add_node(FAIL_NODE, fail)
    workflow.add_edge(START, ROUTE_NODE)
    workflow.add_conditional_edges(
        ROUTE_NODE,
        route_after_classification,
        {REWRITE_NODE: REWRITE_NODE, UNSUPPORTED_NODE: UNSUPPORTED_NODE},
    )
    workflow.add_edge(REWRITE_NODE, RETRIEVE_NODE)
    workflow.add_edge(RETRIEVE_NODE, GRADE_CONTEXT_NODE)
    workflow.add_conditional_edges(
        GRADE_CONTEXT_NODE,
        lambda state: route_after_grading(state, max_retries),
        {
            GENERATE_NODE: GENERATE_NODE,
            RETRY_QUERY_NODE: RETRY_QUERY_NODE,
            NO_CONTEXT_NODE: NO_CONTEXT_NODE,
        },
    )
    workflow.add_edge(RETRY_QUERY_NODE, REWRITE_NODE)
    workflow.add_edge(GENERATE_NODE, EVALUATE_NODE)
    workflow.add_conditional_edges(
        EVALUATE_NODE,
        lambda state: route_after_evaluation(state, max_retries),
        {GENERATE_NODE: GENERATE_NODE, FAIL_NODE: FAIL_NODE, END_NODE: END},
    )
    workflow.add_edge(NO_CONTEXT_NODE, END)
    workflow.add_edge(UNSUPPORTED_NODE, END)
    workflow.add_edge(FAIL_NODE, END)
    graph = workflow.compile()

    def invoke(question: str) -> AgentState:
        """Run the compiled graph for one question."""
        return graph.invoke(
            {
                "question": question,
                "query": question,
                "retry_count": 0,
                "answer_attempts": 0,
                "feedback": "",
                "generation_feedback": "",
            },
            config={
                "recursion_limit": get_agentic_recursion_limit(max_retries)
            },
        )

    return invoke
