from typing import Any

from app.core.config import (
    MCP_INPUT_SCHEMA_KEY,
    MCP_TOOL_NAME_KEY,
    MCP_TOOL_NOT_AVAILABLE_ERROR,
    Settings,
)
from app.integrations.mcp.client import MCPClient, validate_tool_arguments


class MCPToolManager:
    """Discover and invoke tools exposed by one configured MCP server."""

    def __init__(self, settings: Settings) -> None:
        self.client = MCPClient(settings) if settings.mcp_server_url else None
        self.allowed_tools = set(settings.mcp_allowed_tools)
        self._tools: list[dict[str, Any]] | None = None

    def list_tools(self) -> list[dict[str, Any]]:
        """Return the server's advertised tool definitions."""
        if self.client is None:
            return []
        if self._tools is None:
            if not self.allowed_tools:
                self._tools = []
            else:
                self._tools = [
                    tool
                    for tool in self.client.list_tools()
                    if tool.get(MCP_TOOL_NAME_KEY) in self.allowed_tools
                ]
        return self._tools

    def call_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke a tool only after confirming the server advertised it."""
        tool = next(
            (
                tool
                for tool in self.list_tools()
                if tool.get(MCP_TOOL_NAME_KEY) == name
            ),
            None,
        )
        if tool is None or name not in self.allowed_tools:
            raise ValueError(MCP_TOOL_NOT_AVAILABLE_ERROR)
        if self.client is None:
            raise ValueError(MCP_TOOL_NOT_AVAILABLE_ERROR)
        schema = tool.get(MCP_INPUT_SCHEMA_KEY, {})
        if not isinstance(schema, dict):
            schema = {}
        validate_tool_arguments(arguments, schema)
        return self.client.call_tool(name, arguments)
