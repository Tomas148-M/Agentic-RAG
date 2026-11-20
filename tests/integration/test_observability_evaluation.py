import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from typing_extensions import Self

from app.core.config import Settings
from app.integrations.llm.base import LLMProvider
from app.observability.phoenix import PhoenixObservability
from app.rag.retriever import DocumentRetriever, RetrievedChunk
from app.services.chat_service import ChatService
from scripts.evaluate_chat import run_evaluation


class RecordingSpan:
    """Record attributes written by the Phoenix tracing adapter."""

    def __init__(self) -> None:
        self.attributes: dict[str, str | int] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def set_attribute(self, name: str, value: str | int) -> None:
        self.attributes[name] = value


class RecordingTracer:
    """Provide in-memory spans instead of a live Phoenix collector."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, RecordingSpan]] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        span = RecordingSpan()
        self.spans.append((name, span))
        return span


def test_chat_service_emits_chat_and_llm_spans() -> None:
    """Trace a cited chat request through RAG and its LLM provider."""
    retriever = Mock(spec=DocumentRetriever)
    retriever.retrieve.return_value = [
        RetrievedChunk("Czech capital is Prague.", "cities.txt", 0, 0.1)
    ]
    provider = Mock(spec=LLMProvider)
    provider.generate.return_value = "Prague is the capital. [S1]"
    tracer = RecordingTracer()
    observer = PhoenixObservability(
        Settings(phoenix_enabled=True), tracer=tracer
    )
    service = ChatService(
        retriever,
        observer.instrument(provider),
        observability=observer,
    )

    response = service.answer("What is the capital of Czechia?")

    assert response.answer == "Prague is the capital. [S1]"
    assert response.citations[0].source == "cities.txt"
    assert [name for name, _ in tracer.spans] == [
        "agentic-rag.chat",
        "agentic-rag.llm.generate",
    ]
    assert tracer.spans[0][1].attributes["agentic_rag.citation_count"] == 1


def test_evaluation_runner_scores_api_responses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run the evaluation harness against a mocked chat API response."""
    dataset = [
        {
            "id": "capital",
            "question": "What is the capital?",
            "expected_answer_phrases": ["Prague"],
            "expected_citation_sources": ["cities.txt"],
        }
    ]
    dataset_path = tmp_path / "cases.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    response = Mock()
    response.json.return_value = {
        "answer": "Prague is the capital. [S1]",
        "citations": [{"source": "cities.txt"}],
    }
    response.raise_for_status.return_value = None

    class FakeClient:
        """Minimal HTTP client fake for the evaluation runner."""

        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def post(self, url: str, json: dict[str, str]) -> Mock:
            assert url == "http://testserver/api/v1/chat"
            assert json == {"question": "What is the capital?"}
            return response

    monkeypatch.setattr("scripts.evaluate_chat.httpx.Client", FakeClient)
    report = run_evaluation(
        Settings(
            evaluation_endpoint="http://testserver/api/v1/chat",
            evaluation_dataset_path=str(dataset_path),
        )
    )

    assert report["summary"]["cases_total"] == 1
    assert report["summary"]["cases_passed"] == 1
    assert report["results"][0]["passed"] is True
