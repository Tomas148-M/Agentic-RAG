from unittest.mock import Mock

import httpx
import pytest

from app.agents.tools.registry import ToolRegistry
from app.core.config import RAG_MODE_AGENTIC, Settings
from app.rag.retriever import DocumentRetriever
from app.services.chat_service import ChatService


def test_agentic_chat_routes_web_search_and_returns_url_citation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agentic chat routes to web search and returns its URL as a citation."""
    settings = Settings(
        rag_mode=RAG_MODE_AGENTIC,
        web_search_enabled=True,
        web_search_api_key="test-key",
        mcp_server_url=None,
        mcp_allowed_tools=[],
    )
    retriever = Mock(spec=DocumentRetriever)
    registry = ToolRegistry(retriever, settings)

    response = httpx.Response(
        200,
        json={
            "results": [
                {
                    "title": "Prague population",
                    "url": "https://example.test/prague-population",
                    "content": "The population is 1.3 million.",
                }
            ]
        },
        request=httpx.Request("POST", settings.web_search_endpoint),
    )
    search_request = Mock(return_value=response)
    monkeypatch.setattr(httpx, "post", search_request)

    llm = Mock()
    llm.generate.side_effect = [
        (
            '{"route":"WEB","sources":["WEB"],"mcp_tool":"",'
            '"mcp_arguments":{}}'
        ),
        "Prague population",
        "YES",
        "The population is 1.3 million. [S1]",
        "PASS",
    ]
    service = ChatService(
        retriever,
        llm,
        rag_mode=RAG_MODE_AGENTIC,
        max_retries=0,
        tool_registry=registry,
    )

    result = service.answer("What is Prague's population?")

    assert result.answer == "The population is 1.3 million. [S1]"
    assert len(result.citations) == 1
    assert result.citations[0].source == "https://example.test/prague-population"
    assert result.citations[0].excerpt == (
        "Prague population\nThe population is 1.3 million."
    )
    assert search_request.call_args.kwargs["json"]["query"] == (
        "Prague population"
    )
    retriever.retrieve.assert_not_called()
