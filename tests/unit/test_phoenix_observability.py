import sys
from types import ModuleType
from unittest.mock import Mock

from typing_extensions import Self

import app.observability.phoenix as phoenix_module
from app.core.config import Settings
from app.integrations.llm.base import LLMProvider
from app.observability.phoenix import PhoenixObservability, flush_phoenix


class RecordingSpan:
    """Collect span attributes for deterministic tracing tests."""

    def __init__(self) -> None:
        self.attributes: dict[str, str | int] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def set_attribute(self, name: str, value: str | int) -> None:
        self.attributes[name] = value


class RecordingTracer:
    """Return recording spans and retain their names."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, RecordingSpan]] = []

    def start_as_current_span(self, name: str) -> RecordingSpan:
        span = RecordingSpan()
        self.spans.append((name, span))
        return span


def test_disabled_phoenix_keeps_original_provider() -> None:
    """Leave providers unwrapped when tracing is disabled."""
    provider = Mock(spec=LLMProvider)
    observer = PhoenixObservability(Settings(phoenix_enabled=False))

    assert observer.instrument(provider) is provider


def test_span_creation_failure_does_not_break_chat() -> None:
    """A failed exporter falls back to the untraced request path."""

    class BrokenTracer:
        """Simulate an unavailable OpenTelemetry tracer provider."""

        def start_as_current_span(self, name: str) -> RecordingSpan:
            raise RuntimeError("collector unavailable")

    observer = PhoenixObservability(
        Settings(phoenix_enabled=True), tracer=BrokenTracer()
    )

    with observer.chat_trace("question", "basic") as span:
        assert span is None


def test_chat_and_generation_spans_omit_sensitive_content_by_default() -> None:
    """Capture basic span metrics without prompts or user question text."""
    tracer = RecordingTracer()
    settings = Settings(phoenix_enabled=True, phoenix_capture_content=False)
    observer = PhoenixObservability(settings, tracer=tracer)
    provider = Mock(spec=LLMProvider)
    provider.generate.return_value = "Prague [S1]"
    traced_provider = observer.instrument(provider)

    with observer.chat_trace("Secret question", "basic") as chat_span:
        answer = traced_provider.generate("Secret prompt")
        observer.record_chat_output(chat_span, answer, citation_count=1)

    names = [name for name, _ in tracer.spans]
    assert names == ["agentic-rag.chat", "agentic-rag.llm.generate"]
    chat_attributes = tracer.spans[0][1].attributes
    generation_attributes = tracer.spans[1][1].attributes
    assert "Secret question" not in str(chat_attributes.values())
    assert "Secret prompt" not in str(generation_attributes.values())
    assert chat_attributes["agentic_rag.citation_count"] == 1
    assert generation_attributes["agentic_rag.prompt_length"] == len(
        "Secret prompt"
    )
    provider.generate.assert_called_once_with("Secret prompt")


def test_capture_content_records_opted_in_prompt_and_answer() -> None:
    """Record prompts and answers only after explicit opt-in."""
    tracer = RecordingTracer()
    observer = PhoenixObservability(
        Settings(phoenix_enabled=True, phoenix_capture_content=True),
        tracer=tracer,
    )
    provider = Mock(spec=LLMProvider)
    provider.generate.return_value = "answer"

    with observer.chat_trace("question text", "basic") as chat_span:
        observer.instrument(provider).generate("prompt text")
        observer.record_chat_output(chat_span, "answer text", citation_count=0)

    assert tracer.spans[0][1].attributes["input.value"] == "question text"
    assert tracer.spans[0][1].attributes["output.value"] == "answer text"
    assert tracer.spans[1][1].attributes["input.value"] == "prompt text"
    assert tracer.spans[1][1].attributes["output.value"] == "answer"


def test_phoenix_registration_uses_http_and_can_restart(
    monkeypatch,
) -> None:
    """Register the HTTP collector explicitly and reinitialize after shutdown."""
    tracer = RecordingTracer()
    providers = []

    class RecordingProvider:
        """Fake Phoenix tracer provider with lifecycle tracking."""

        def get_tracer(self, name: str) -> RecordingTracer:
            assert name == "agentic-rag"
            return tracer

        def force_flush(self) -> None:
            return None

        def shutdown(self) -> None:
            return None

    def register(**kwargs: object) -> RecordingProvider:
        providers.append(kwargs)
        return RecordingProvider()

    phoenix_package = ModuleType("phoenix")
    phoenix_package.__path__ = []
    otel_module = ModuleType("phoenix.otel")
    otel_module.register = register
    monkeypatch.setitem(sys.modules, "phoenix", phoenix_package)
    monkeypatch.setitem(sys.modules, "phoenix.otel", otel_module)
    monkeypatch.setattr(phoenix_module, "_setup_attempted", False)
    monkeypatch.setattr(phoenix_module, "_tracer", None)
    monkeypatch.setattr(phoenix_module, "_tracer_provider", None)
    settings = Settings(phoenix_enabled=True)

    first = PhoenixObservability(settings)
    flush_phoenix()
    second = PhoenixObservability(settings)

    assert first.tracer is tracer
    assert second.tracer is tracer
    assert len(providers) == 2
    for options in providers:
        assert options["endpoint"] == settings.phoenix_endpoint
        assert options["protocol"] == "http/protobuf"
        assert options["set_global_tracer_provider"] is False
