"""Optional OpenTelemetry tracing for Arize Phoenix."""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.core.config import (
    PHOENIX_CAPTURE_DISABLED_VALUE,
    PHOENIX_CHAIN_KIND,
    PHOENIX_CITATION_COUNT_ATTRIBUTE,
    PHOENIX_ENVIRONMENT_ATTRIBUTE,
    PHOENIX_GENERATION_NAME,
    PHOENIX_HTTP_PROTOBUF_PROTOCOL,
    PHOENIX_INITIALIZATION_WARNING,
    PHOENIX_INPUT_VALUE_ATTRIBUTE,
    PHOENIX_LLM_KIND,
    PHOENIX_MODE_ATTRIBUTE,
    PHOENIX_OUTPUT_VALUE_ATTRIBUTE,
    PHOENIX_PROMPT_LENGTH_ATTRIBUTE,
    PHOENIX_RESPONSE_LENGTH_ATTRIBUTE,
    PHOENIX_SET_GLOBAL_TRACER_PROVIDER,
    PHOENIX_SPAN_KIND_ATTRIBUTE,
    PHOENIX_SPAN_WARNING,
    PHOENIX_TRACE_NAME,
    PHOENIX_TRACER_NAME,
    Settings,
)
from app.integrations.llm.base import LLMProvider

logger = logging.getLogger(__name__)
_setup_lock = threading.Lock()
_tracer_provider: Any | None = None
_tracer: Any | None = None
_setup_attempted = False


class PhoenixObservability:
    """Record chat and generation spans when Phoenix is configured."""

    def __init__(self, settings: Settings, tracer: Any | None = None) -> None:
        self.settings = settings
        self.tracer = tracer if tracer is not None else _get_tracer(settings)

    @contextmanager
    def chat_trace(self, question: str, mode: str) -> Iterator[Any | None]:
        """Create a root chat span without exposing content by default."""
        with self._span(PHOENIX_TRACE_NAME, PHOENIX_CHAIN_KIND) as span:
            if span is not None:
                self._set_attribute(span, PHOENIX_MODE_ATTRIBUTE, mode)
                self._set_attribute(
                    span,
                    PHOENIX_ENVIRONMENT_ATTRIBUTE,
                    self.settings.phoenix_environment,
                )
                if self.settings.phoenix_capture_content:
                    self._set_attribute(span, PHOENIX_INPUT_VALUE_ATTRIBUTE, question)
            yield span

    @contextmanager
    def generation_trace(self, prompt: str) -> Iterator[Any | None]:
        """Create a child generation span for an LLM provider call."""
        with self._span(PHOENIX_GENERATION_NAME, PHOENIX_LLM_KIND) as span:
            if span is not None:
                self._set_attribute(
                    span,
                    PHOENIX_PROMPT_LENGTH_ATTRIBUTE,
                    len(prompt),
                )
                if self.settings.phoenix_capture_content:
                    self._set_attribute(span, PHOENIX_INPUT_VALUE_ATTRIBUTE, prompt)
                else:
                    self._set_attribute(
                        span,
                        PHOENIX_INPUT_VALUE_ATTRIBUTE,
                        PHOENIX_CAPTURE_DISABLED_VALUE,
                    )
            yield span

    def record_chat_output(
        self, span: Any | None, answer: str, citation_count: int
    ) -> None:
        """Attach safe output metrics and optionally the answer text."""
        if span is None:
            return
        self._set_attribute(span, PHOENIX_CITATION_COUNT_ATTRIBUTE, citation_count)
        self._set_attribute(span, PHOENIX_RESPONSE_LENGTH_ATTRIBUTE, len(answer))
        output = answer if self.settings.phoenix_capture_content else (
            PHOENIX_CAPTURE_DISABLED_VALUE
        )
        self._set_attribute(span, PHOENIX_OUTPUT_VALUE_ATTRIBUTE, output)

    def instrument(self, provider: LLMProvider) -> LLMProvider:
        """Wrap an LLM provider to record generation calls when tracing is on."""
        if self.tracer is None:
            return provider
        return _TracedLLMProvider(provider, self)

    @contextmanager
    def _span(self, name: str, kind: str) -> Iterator[Any | None]:
        if self.tracer is None:
            yield None
            return
        try:
            manager = self.tracer.start_as_current_span(name)
            span = manager.__enter__()
        except Exception:
            logger.warning(PHOENIX_SPAN_WARNING, exc_info=True)
            yield None
            return
        try:
            self._set_attribute(span, PHOENIX_SPAN_KIND_ATTRIBUTE, kind)
            yield span
        except BaseException as error:
            try:
                manager.__exit__(type(error), error, error.__traceback__)
            except Exception:
                logger.warning(PHOENIX_SPAN_WARNING, exc_info=True)
            raise
        else:
            try:
                manager.__exit__(None, None, None)
            except Exception:
                logger.warning(PHOENIX_SPAN_WARNING, exc_info=True)

    @staticmethod
    def _set_attribute(span: Any, name: str, value: str | int) -> None:
        try:
            span.set_attribute(name, value)
        except Exception:
            logger.warning(PHOENIX_SPAN_WARNING, exc_info=True)


class _TracedLLMProvider(LLMProvider):
    """Add a Phoenix generation span around an existing LLM provider."""

    def __init__(self, provider: LLMProvider, observability: PhoenixObservability):
        self.provider = provider
        self.observability = observability

    def generate(self, prompt: str) -> str:
        """Generate a response and capture its length or opted-in content."""
        with self.observability.generation_trace(prompt) as span:
            output = self.provider.generate(prompt)
            if span is not None:
                self.observability._set_attribute(
                    span, PHOENIX_RESPONSE_LENGTH_ATTRIBUTE, len(output)
                )
                if self.observability.settings.phoenix_capture_content:
                    self.observability._set_attribute(
                        span, PHOENIX_OUTPUT_VALUE_ATTRIBUTE, output
                    )
                else:
                    self.observability._set_attribute(
                        span,
                        PHOENIX_OUTPUT_VALUE_ATTRIBUTE,
                        PHOENIX_CAPTURE_DISABLED_VALUE,
                    )
            return output


def _get_tracer(settings: Settings) -> Any | None:
    """Initialize one process-wide Phoenix exporter on first use."""
    global _setup_attempted, _tracer, _tracer_provider
    if not settings.phoenix_enabled:
        return None
    with _setup_lock:
        if _setup_attempted:
            return _tracer
        _setup_attempted = True
        try:
            from phoenix.otel import register

            options: dict[str, Any] = {
                "endpoint": settings.phoenix_endpoint,
                "project_name": settings.phoenix_project_name,
                "protocol": PHOENIX_HTTP_PROTOBUF_PROTOCOL,
                "auto_instrument": False,
                "batch": True,
                "set_global_tracer_provider": (
                    PHOENIX_SET_GLOBAL_TRACER_PROVIDER
                ),
            }
            if settings.phoenix_api_key:
                options["api_key"] = settings.phoenix_api_key.get_secret_value()
            _tracer_provider = register(**options)
            _tracer = _tracer_provider.get_tracer(PHOENIX_TRACER_NAME)
        except Exception:
            logger.warning(PHOENIX_INITIALIZATION_WARNING, exc_info=True)
            _tracer = None
            _tracer_provider = None
        return _tracer


def flush_phoenix() -> None:
    """Flush and shut down the process-wide exporter during app shutdown."""
    global _setup_attempted, _tracer, _tracer_provider
    with _setup_lock:
        try:
            if _tracer_provider is not None:
                _tracer_provider.force_flush()
                _tracer_provider.shutdown()
        except Exception:
            logger.warning(PHOENIX_SPAN_WARNING, exc_info=True)
        finally:
            _setup_attempted = False
            _tracer = None
            _tracer_provider = None
