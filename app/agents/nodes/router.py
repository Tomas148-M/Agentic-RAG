import json
from collections.abc import Callable

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
        result = llm.generate(
            build_source_selector_prompt(
                state["question"], available_sources, tools
            )
        )
        try:
            selection = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            selection = {SOURCE_SELECTION_ROUTE_KEY: result.strip().upper()}
        if not isinstance(selection, dict):
            selection = {SOURCE_SELECTION_ROUTE_KEY: ROUTE_DOCUMENTS}

        choice = str(
            selection.get(SOURCE_SELECTION_ROUTE_KEY, "")
        ).strip().upper()
        if choice == ROUTE_UNSUPPORTED.upper():
            return {"route": ROUTE_UNSUPPORTED}
        if choice == ROUTE_RETRIEVE.upper():
            choice = ROUTE_DOCUMENTS
        allowed = {*ROUTE_SOURCE_OPTIONS, ROUTE_HYBRID}
        if choice not in allowed or choice not in available_sources:
            choice = ROUTE_DOCUMENTS

        mcp_tool = selection.get(SOURCE_SELECTION_MCP_TOOL_KEY, "")
        valid_tool_names = {
            str(tool.get(MCP_TOOL_NAME_KEY)) for tool in tools
        }
        if not isinstance(mcp_tool, str) or mcp_tool not in valid_tool_names:
            mcp_tool = ""
        if choice == ROUTE_MCP and not mcp_tool:
            choice = ROUTE_DOCUMENTS
        arguments = selection.get(SOURCE_SELECTION_MCP_ARGUMENTS_KEY, {})
        if not isinstance(arguments, dict):
            arguments = {}
        selected_sources = selection.get(HYBRID_SOURCES_KEY, [])
        if choice == ROUTE_HYBRID:
            available = {
                source
                for source in available_sources
                if source in ROUTE_SOURCE_OPTIONS
            }
            if not isinstance(selected_sources, list):
                selected_sources = []
            selected_sources = list(
                dict.fromkeys(
                    source
                    for source in selected_sources
                    if isinstance(source, str) and source in available
                )
            )
            if len(selected_sources) < 2:
                choice = ROUTE_DOCUMENTS
                selected_sources = [ROUTE_DOCUMENTS]
            elif ROUTE_MCP in selected_sources and not mcp_tool:
                selected_sources.remove(ROUTE_MCP)
                if len(selected_sources) < 2:
                    choice = ROUTE_DOCUMENTS
                    selected_sources = [ROUTE_DOCUMENTS]
        else:
            selected_sources = [choice]
        return {
            "route": choice,
            "selected_sources": selected_sources,
            "mcp_tool": mcp_tool,
            "mcp_arguments": arguments,
        }

    return route
