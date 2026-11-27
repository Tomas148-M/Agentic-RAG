import json
from collections.abc import Callable
from typing import Any

from app.agents.prompts.source_selector import build_source_selector_prompt
from app.agents.state import AgentState
from app.agents.tools.registry import ToolRegistry
from app.core.config import (
    HYBRID_SOURCES_KEY,
    MCP_TOOL_NAME_KEY,
    ROUTE_DOCUMENTS,
    ROUTE_HYBRID,
    ROUTE_MCP,
    ROUTE_RETRIEVE,
    ROUTE_SOURCE_OPTIONS,
    ROUTE_UNSUPPORTED,
    SOURCE_SELECTION_MCP_ARGUMENTS_KEY,
    SOURCE_SELECTION_MCP_TOOL_KEY,
    SOURCE_SELECTION_ROUTE_KEY,
)
from app.integrations.llm.base import LLMProvider


def parse_source_selection(response: str) -> dict[str, Any]:
    """Parse the selector response as JSON or a plain route token."""
    try:
        selection = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        selection = {SOURCE_SELECTION_ROUTE_KEY: response.strip().upper()}

    if not isinstance(selection, dict):
        return {SOURCE_SELECTION_ROUTE_KEY: ROUTE_DOCUMENTS}
    return selection


def normalize_hybrid_sources(
    selected_sources: Any,
    available_sources: list[str],
    mcp_tool: str,
) -> tuple[str, list[str]]:
    """Filter and deduplicate hybrid sources, falling back when invalid."""
    available = {
        source
        for source in available_sources
        if source in ROUTE_SOURCE_OPTIONS
    }
    if not isinstance(selected_sources, list):
        selected_sources = []

    normalized_sources = list(
        dict.fromkeys(
            source
            for source in selected_sources
            if isinstance(source, str) and source in available
        )
    )
    if len(normalized_sources) < 2:
        return ROUTE_DOCUMENTS, [ROUTE_DOCUMENTS]

    if ROUTE_MCP in normalized_sources and not mcp_tool:
        normalized_sources.remove(ROUTE_MCP)
        if len(normalized_sources) < 2:
            return ROUTE_DOCUMENTS, [ROUTE_DOCUMENTS]

    return ROUTE_HYBRID, normalized_sources


def normalize_source_selection(
    selection: dict[str, Any],
    available_sources: list[str],
    tools: list[dict[str, Any]],
) -> AgentState:
    """Validate the selected route, tool, and optional hybrid sources."""
    choice = str(selection.get(SOURCE_SELECTION_ROUTE_KEY, "")).strip().upper()
    if choice == ROUTE_UNSUPPORTED.upper():
        return {"route": ROUTE_UNSUPPORTED}

    if choice == ROUTE_RETRIEVE.upper():
        choice = ROUTE_DOCUMENTS

    allowed = {*ROUTE_SOURCE_OPTIONS, ROUTE_HYBRID}
    if choice not in allowed or choice not in available_sources:
        choice = ROUTE_DOCUMENTS

    valid_tool_names = {
        str(tool.get(MCP_TOOL_NAME_KEY)) for tool in tools
    }
    mcp_tool = selection.get(SOURCE_SELECTION_MCP_TOOL_KEY, "")
    if not isinstance(mcp_tool, str) or mcp_tool not in valid_tool_names:
        mcp_tool = ""
    if choice == ROUTE_MCP and not mcp_tool:
        choice = ROUTE_DOCUMENTS

    arguments = selection.get(SOURCE_SELECTION_MCP_ARGUMENTS_KEY, {})
    if not isinstance(arguments, dict):
        arguments = {}

    if choice == ROUTE_HYBRID:
        choice, selected_sources = normalize_hybrid_sources(
            selection.get(HYBRID_SOURCES_KEY, []), available_sources, mcp_tool
        )
    else:
        selected_sources = [choice]

    return {
        "route": choice,
        "selected_sources": selected_sources,
        "mcp_tool": mcp_tool,
        "mcp_arguments": arguments,
    }


def create_router_node(
    llm: LLMProvider, tool_registry: ToolRegistry | None = None
) -> Callable[[AgentState], AgentState]:
    """Create a node that selects a supported information source.

    Args:
        llm: Model used to classify and route the request.
        tool_registry: Optional configured retrieval tools.

    Returns:
        A LangGraph node returning the selected source and MCP arguments.
    """

    def route(state: AgentState) -> AgentState:
        sources = tool_registry.available_sources() if tool_registry else []
        available_sources = sources or [ROUTE_DOCUMENTS]
        tools = tool_registry.mcp_tool_definitions() if tool_registry else []
        response = llm.generate(
            build_source_selector_prompt(
                state["question"], available_sources, tools
            )
        )
        selection = parse_source_selection(response)
        return normalize_source_selection(selection, available_sources, tools)

    return route
