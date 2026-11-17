from app.agents.state import AgentState
from app.core.config import (
    END_NODE,
    FAIL_NODE,
    GENERATE_NODE,
    NO_CONTEXT_NODE,
    RETRY_QUERY_NODE,
    REWRITE_NODE,
    ROUTE_RETRIEVE,
    UNSUPPORTED_NODE,
)


def route_after_classification(state: AgentState) -> str:
    """Choose retrieval or the safe unsupported-request response."""
    return (
        REWRITE_NODE
        if state.get("route") == ROUTE_RETRIEVE
        else UNSUPPORTED_NODE
    )


def route_after_grading(state: AgentState, max_retries: int) -> str:
    """Continue with useful context, retry, or finish without context."""
    if state.get("relevant_chunks"):
        return GENERATE_NODE
    if state.get("retry_count", 0) < max_retries:
        return RETRY_QUERY_NODE
    return NO_CONTEXT_NODE


def route_after_evaluation(state: AgentState, max_retries: int) -> str:
    """Finish with a checked answer or retry generation within the limit."""
    if state.get("answer_valid"):
        return END_NODE
    if state.get("answer_attempts", 0) <= max_retries:
        return GENERATE_NODE
    return FAIL_NODE
