from app.agents.state import AgentState
from app.core.config import (
    END_NODE,
    FAIL_NODE,
    GENERATE_NODE,
    NO_CONTEXT_NODE,
    RETRY_QUERY_NODE,
    REWRITE_NODE,
    ROUTE_DOCUMENTS,
    ROUTE_HYBRID,
    ROUTE_MCP,
    ROUTE_RETRIEVE,
    ROUTE_WEB,
    UNSUPPORTED_NODE,
)


def route_after_classification(state: AgentState) -> str:
    """Choose retrieval or the safe unsupported-request response."""
    return (
        REWRITE_NODE
        if state.get("route") in {
            ROUTE_RETRIEVE,
            ROUTE_DOCUMENTS,
            ROUTE_WEB,
            ROUTE_MCP,
            ROUTE_HYBRID,
        }
        else UNSUPPORTED_NODE
    )


def route_after_grading(state: AgentState, max_retries: int) -> str:
    """Continue with useful context, retry, or finish without context."""
    if state.get("relevant_chunks"):
        return GENERATE_NODE
    if can_retry_retrieval(state, max_retries):
        return RETRY_QUERY_NODE
    return NO_CONTEXT_NODE


def route_after_evaluation(state: AgentState, max_retries: int) -> str:
    """Finish with a checked answer or retry generation within the limit."""
    if state.get("answer_grounded"):
        return END_NODE
    if can_retry_generation(state, max_retries):
        return GENERATE_NODE
    return FAIL_NODE


def can_retry_retrieval(state: AgentState, max_retries: int) -> bool:
    """Return whether another search is allowed.

    ``retrieval_retries`` counts retries after the initial search. A limit of
    two therefore permits three total search attempts.
    """
    return state.get("retrieval_retries", 0) < max_retries


def can_retry_generation(state: AgentState, max_retries: int) -> bool:
    """Return whether another answer generation is allowed.

    ``generation_attempts`` includes the initial generation. A limit of two
    therefore permits three total generations.
    """
    return state.get("generation_attempts", 0) <= max_retries
