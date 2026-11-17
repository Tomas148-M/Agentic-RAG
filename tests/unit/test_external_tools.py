from unittest.mock import Mock

import httpx
import pytest

from app.agents.nodes.retrieve import create_retrieve_node
from app.agents.nodes.router import create_router_node
from app.agents.tools.registry import ToolRegistry
from app.core.config import (
    MCP_TOOL_NOT_AVAILABLE_ERROR,
    ROUTE_DOCUMENTS,
    ROUTE_HYBRID,
    ROUTE_MCP,
    ROUTE_WEB,
    Settings,
)
from app.integrations.mcp.client import MCPClient, MCPError
from app.integrations.mcp.manager import MCPToolManager
from app.integrations.web.client import WebSearchClient
from app.rag.retriever import RetrievedChunk


def _response(status_code: int, **kwargs: object) -> httpx.Response:
    """Build an HTTP response with a request attached for raise_for_status."""
    request = httpx.Request("POST", "http://localhost:8001/mcp")
    return httpx.Response(status_code, request=request, **kwargs)


def test_mcp_client_paginates_and_matches_sse_response(monkeypatch) -> None:
    """List all pages and ignore an unrelated SSE notification."""
    settings = Settings(mcp_server_url="http://localhost:8001/mcp")
    responses = [
        _response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"capabilities": {}},
            },
        ),
        _response(202),
        _response(
            200,
            text=(
                'data: {"jsonrpc":"2.0","method":"notifications/progress"}'
                "\n\n"
                'data: {"jsonrpc":"2.0","id":2,"result":'
                '{"tools":[{"name":"search"}],"nextCursor":"page-2"}}'
            ),
            headers={"content-type": "text/event-stream"},
        ),
        _response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"tools": [{"name": "lookup"}]},
            },
        ),
        _response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "result": {"content": []},
            },
        ),
    ]
    post = Mock(side_effect=responses)
    monkeypatch.setattr(httpx, "post", post)
    client = MCPClient(settings)

    tools = client.list_tools()
    result = client.call_tool("search", {"query": "latest"})

    assert [tool["name"] for tool in tools] == ["search", "lookup"]
    assert result == {"content": []}
    assert [call.kwargs["json"]["method"] for call in post.call_args_list] == [
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/list",
        "tools/call",
    ]
    assert post.call_args_list[3].kwargs["json"]["params"] == {
        "cursor": "page-2"
    }


def test_mcp_manager_rejects_advertised_tool_without_allowlist(monkeypatch) -> None:
    """An advertised tool cannot run unless explicitly allowlisted."""
    settings = Settings(
        mcp_server_url="http://localhost:8001/mcp",
        mcp_allowed_tools=["search"],
    )
    manager = MCPToolManager(settings)
    monkeypatch.setattr(
        manager.client,
        "list_tools",
        Mock(
            return_value=[
                {
                    "name": "search",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
                {"name": "delete_everything", "inputSchema": {"type": "object"}},
            ]
        ),
    )
    call = Mock(return_value={"content": []})
    monkeypatch.setattr(manager.client, "call_tool", call)

    assert [tool["name"] for tool in manager.list_tools()] == ["search"]
    with pytest.raises(ValueError, match=MCP_TOOL_NOT_AVAILABLE_ERROR):
        manager.call_tool("delete_everything", {})
    with pytest.raises(ValueError):
        manager.call_tool("search", {"query": "latest", "unexpected": True})
    manager.call_tool("search", {"query": "latest"})
    call.assert_called_once_with("search", {"query": "latest"})


def test_mcp_manager_preserves_valid_empty_arguments(monkeypatch) -> None:
    """A schema for a no-argument tool receives an empty object unchanged."""
    settings = Settings(
        mcp_server_url="http://localhost:8001/mcp",
        mcp_allowed_tools=["status"],
    )
    manager = MCPToolManager(settings)
    monkeypatch.setattr(
        manager.client,
        "list_tools",
        Mock(
            return_value=[
                {
                    "name": "status",
                    "inputSchema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                }
            ]
        ),
    )
    call = Mock(return_value={"content": []})
    monkeypatch.setattr(manager.client, "call_tool", call)

    manager.call_tool("status", {})

    call.assert_called_once_with("status", {})


def test_mcp_manager_accepts_nullable_union_arguments(monkeypatch) -> None:
    """Nullable JSON Schema union values pass validation and reach MCP."""
    settings = Settings(
        mcp_server_url="http://localhost:8001/mcp",
        mcp_allowed_tools=["lookup"],
    )
    manager = MCPToolManager(settings)
    monkeypatch.setattr(
        manager.client,
        "list_tools",
        Mock(
            return_value=[
                {
                    "name": "lookup",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {"type": ["string", "null"]}
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                }
            ]
        ),
    )
    call = Mock(return_value={"content": []})
    monkeypatch.setattr(manager.client, "call_tool", call)

    manager.call_tool("lookup", {"query": None})

    call.assert_called_once_with("lookup", {"query": None})


def test_retrieve_node_forwards_selected_hybrid_sources() -> None:
    """The LangGraph retrieval node passes the exact hybrid selection."""
    registry = Mock()
    registry.execute.return_value = []
    selected_sources = [ROUTE_DOCUMENTS, ROUTE_MCP]
    node = create_retrieve_node(tool_registry=registry)

    node(
        {
            "query": "question",
            "route": ROUTE_HYBRID,
            "mcp_tool": "search",
            "mcp_arguments": {"query": "question"},
            "selected_sources": selected_sources,
        }
    )

    registry.execute.assert_called_once_with(
        "question",
        ROUTE_HYBRID,
        "search",
        {"query": "question"},
        selected_sources,
    )


def test_web_search_normalizes_results(monkeypatch) -> None:
    """Normalize provider result fields and truncate excerpt size."""
    settings = Settings(
        web_search_enabled=True,
        web_search_api_key="test-key",
        tool_result_max_chars=20,
    )
    response = Mock()
    response.json.return_value = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
                "content": "A result with a long excerpt.",
            }
        ]
    }
    monkeypatch.setattr(httpx, "post", Mock(return_value=response))

    results = WebSearchClient(settings).search("example")

    assert results == [
        {
            "title": "Example",
            "url": "https://example.com",
            "content": "A result with a long",
        }
    ]


def test_source_router_selects_only_configured_sources() -> None:
    """Source routing exposes configured choices in its prompt."""
    llm = Mock()
    llm.generate.return_value = (
        '{"route":"WEB","sources":["WEB"],"mcp_tool":"",'
        '"mcp_arguments":{}}'
    )
    registry = Mock()
    registry.available_sources.return_value = [ROUTE_DOCUMENTS, ROUTE_WEB]
    registry.mcp_tool_definitions.return_value = []

    state = create_router_node(llm, registry)({"question": "Current news"})

    assert state["route"] == ROUTE_WEB
    assert state["selected_sources"] == [ROUTE_WEB]
    assert ROUTE_WEB in llm.generate.call_args.args[0]


def test_mcp_unavailable_does_not_disable_document_source(monkeypatch) -> None:
    """An unavailable optional MCP server is omitted from source selection."""
    settings = Settings(
        mcp_server_url="http://localhost:8001/mcp",
        mcp_allowed_tools=["search"],
    )
    registry = ToolRegistry(Mock(), settings)
    monkeypatch.setattr(
        registry.mcp,
        "list_tools",
        Mock(side_effect=MCPError("offline")),
    )

    assert registry.available_sources() == [ROUTE_DOCUMENTS]


def test_hybrid_executes_only_selected_document_and_mcp_sources(
    monkeypatch,
) -> None:
    """The source selector's hybrid subset is respected exactly."""
    settings = Settings(
        web_search_enabled=True,
        web_search_api_key="test-key",
        mcp_server_url="http://localhost:8001/mcp",
        mcp_allowed_tools=["search"],
    )
    retriever = Mock()
    rag_chunk = RetrievedChunk("Local", "file.txt", 0, 0.0)
    mcp_chunk = RetrievedChunk("Remote", "mcp://search", 0, 0.0)
    retriever.retrieve.return_value = [rag_chunk]
    registry = ToolRegistry(retriever, settings)
    registry._mcp_tools = [{"name": "search"}]
    registry._mcp_discovered = True
    monkeypatch.setattr(registry.mcp, "call", Mock(return_value=[mcp_chunk]))
    web_search = Mock()
    monkeypatch.setattr(registry.web, "search", web_search)

    chunks = registry.execute(
        "query",
        ROUTE_HYBRID,
        "search",
        {"query": "query"},
        [ROUTE_DOCUMENTS, ROUTE_MCP],
    )

    assert chunks == [rag_chunk, mcp_chunk]
    web_search.assert_not_called()


def test_registry_adapts_web_results_to_citable_chunks(monkeypatch) -> None:
    """A selected web source returns citation-ready chunks."""
    settings = Settings(
        web_search_enabled=True,
        web_search_api_key="test-key",
    )
    registry = ToolRegistry(Mock(), settings)
    result = RetrievedChunk(
        "Title\nExcerpt", "web://https://example.com", 0, 0.0
    )
    monkeypatch.setattr(registry.web, "search", Mock(return_value=[result]))

    chunks = registry.execute("query", ROUTE_WEB)

    assert chunks == [result]
    registry.rag.retriever.retrieve.assert_not_called()
