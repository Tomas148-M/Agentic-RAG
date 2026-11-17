import json
from typing import Any

from app.core.config import (
    MCP_CONTENT_BLOCK_TEXT_KEY,
    MCP_CONTENT_BLOCK_TEXT_TYPE,
    MCP_CONTENT_BLOCK_TYPE_KEY,
    MCP_CONTENT_KEY,
    MCP_SEARCH_SOURCE_PREFIX,
    MCP_STRUCTURED_CONTENT_KEY,
    Settings,
)
from app.integrations.mcp.manager import MCPToolManager
from app.rag.retriever import RetrievedChunk


class MCPTool:
    """Adapt a selected MCP server tool's result to cited RAG context."""

    def __init__(self, settings: Settings, manager: MCPToolManager) -> None:
        self.settings = settings
        self.manager = manager

    def list_tools(self) -> list[dict[str, Any]]:
        """Return advertised tool metadata for source selection."""
        return self.manager.list_tools()

    def call(
        self, name: str, arguments: dict[str, Any]
    ) -> list[RetrievedChunk]:
        """Call an advertised MCP tool and normalize its textual result."""
        payload = self.manager.call_tool(name, arguments)
        texts = []
        structured = payload.get(MCP_STRUCTURED_CONTENT_KEY)
        if structured is not None:
            texts.append(json.dumps(structured, ensure_ascii=False))
        for block in payload.get(MCP_CONTENT_KEY, []):
            if (
                isinstance(block, dict)
                and block.get(MCP_CONTENT_BLOCK_TYPE_KEY)
                == MCP_CONTENT_BLOCK_TEXT_TYPE
            ):
                text = block.get(MCP_CONTENT_BLOCK_TEXT_KEY)
                if isinstance(text, str):
                    texts.append(text)
        output = "\n".join(texts).strip()
        if not output:
            return []
        output = output[: self.settings.tool_result_max_chars]
        return [
            RetrievedChunk(
                text=output,
                source=f"{MCP_SEARCH_SOURCE_PREFIX}{name}",
                chunk_index=0,
                distance=0.0,
            )
        ]
