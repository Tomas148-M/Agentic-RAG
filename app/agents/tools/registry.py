from typing import Any

from app.agents.tools.mcp_tool import MCPTool
from app.agents.tools.rag_tool import RAGTool
from app.agents.tools.web_tool import WebSearchTool
from app.core.config import (
    MCP_UNAVAILABLE_ERROR,
    RETRIEVAL_HYBRID_SELECTION_REQUIRED_ERROR,
    RETRIEVAL_MCP_FAILED_ERROR,
    RETRIEVAL_MCP_TOOL_MISSING_ERROR,
    RETRIEVAL_WEB_UNAVAILABLE_ERROR,
    ROUTE_DOCUMENTS,
    ROUTE_HYBRID,
    ROUTE_MCP,
    ROUTE_WEB,
    Settings,
)
from app.integrations.mcp.client import MCPError
from app.integrations.mcp.manager import MCPToolManager
from app.integrations.web.client import WebSearchError
from app.rag.retriever import DocumentRetriever, RetrievalError, RetrievedChunk


class ToolRegistry:
    """Register enabled retrieval tools and execute selected sources."""

    def __init__(self, retriever: DocumentRetriever, settings: Settings) -> None:
        self.settings = settings
        self.rag = RAGTool(retriever)
        self.web = (
            WebSearchTool(settings)
            if settings.web_search_enabled and settings.web_search_api_key
            else None
        )
        self.mcp_manager = MCPToolManager(settings)
        self.mcp = MCPTool(settings, self.mcp_manager)
        self._mcp_tools: list[dict[str, Any]] = []
        self._mcp_discovered = False

    def available_sources(self) -> list[str]:
        """Return routes that currently have at least one configured source."""
        sources = [ROUTE_DOCUMENTS]
        if self.web is not None:
            sources.append(ROUTE_WEB)
        if self.mcp_manager.client is not None:
            self.mcp_tool_definitions()
            if self._mcp_tools:
                sources.append(ROUTE_MCP)
        if len(sources) > 1:
            sources.append(ROUTE_HYBRID)
        return sources

    def mcp_tool_definitions(self) -> list[dict[str, Any]]:
        """Return allowed MCP schemas, skipping an unavailable optional server."""
        if not self._mcp_discovered:
            try:
                self._mcp_tools = self.mcp.list_tools()
            except MCPError:
                self._mcp_tools = []
            self._mcp_discovered = True
        return self._mcp_tools

    def execute(
        self,
        query: str,
        route: str,
        mcp_tool: str = "",
        mcp_arguments: dict[str, Any] | None = None,
        selected_sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Execute only the sources selected by the routing node."""
        if route == ROUTE_HYBRID and selected_sources is None:
            raise RetrievalError(RETRIEVAL_HYBRID_SELECTION_REQUIRED_ERROR)
        sources = (
            selected_sources
            if selected_sources is not None
            else self._sources_for_route(route)
        )
        available = set(self.available_sources())
        if any(source not in available for source in sources):
            raise RetrievalError(MCP_UNAVAILABLE_ERROR)

        chunks: list[RetrievedChunk] = []
        if ROUTE_DOCUMENTS in sources:
            chunks.extend(self.rag.search(query))
        if ROUTE_WEB in sources and self.web is not None:
            try:
                chunks.extend(self.web.search(query))
            except WebSearchError as exc:
                raise RetrievalError(RETRIEVAL_WEB_UNAVAILABLE_ERROR) from exc
        if ROUTE_MCP in sources:
            if not mcp_tool:
                raise RetrievalError(RETRIEVAL_MCP_TOOL_MISSING_ERROR)
            # Preserve an explicit empty object for valid no-argument tools.
            arguments = dict(mcp_arguments) if mcp_arguments is not None else {}
            try:
                chunks.extend(self.mcp.call(mcp_tool, arguments))
            except (MCPError, TypeError, ValueError) as exc:
                raise RetrievalError(RETRIEVAL_MCP_FAILED_ERROR) from exc
        return chunks

    @staticmethod
    def _sources_for_route(route: str) -> list[str]:
        if route in {ROUTE_DOCUMENTS, ROUTE_WEB, ROUTE_MCP}:
            return [route]
        return []
