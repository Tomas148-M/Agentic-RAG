import subprocess

import httpx
import pytest
from pydantic import ValidationError

from app.api.dependencies import get_llm_provider
from app.core.config import Settings
from app.integrations.llm.base import LLMProviderError
from app.integrations.llm.codex import CodexProvider
from app.integrations.llm.factory import create_llm_provider
from app.integrations.llm.ollama import OllamaProvider


def test_factory_selects_configured_provider() -> None:
    """The configured provider selects the corresponding implementation."""
    assert isinstance(create_llm_provider(Settings(llm_provider="ollama", _env_file=None)), OllamaProvider)
    assert isinstance(create_llm_provider(Settings(llm_provider="codex", _env_file=None)), CodexProvider)
    with pytest.raises(ValidationError):
        Settings(llm_provider="unknown", _env_file=None)


def test_dependency_uses_injected_settings() -> None:
    """The dependency constructs a provider using the supplied settings."""
    settings = Settings(llm_provider="codex", codex_model="example", _env_file=None)
    provider = get_llm_provider(settings)
    assert isinstance(provider, CodexProvider)
    assert provider.model == "example"


def test_provider_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider selection and model can be configured through the environment."""
    monkeypatch.setenv("AGENTIC_RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGENTIC_RAG_OLLAMA_MODEL", "local-model")
    provider = create_llm_provider(Settings(_env_file=None))
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "local-model"


def test_ollama_sends_prompt_and_returns_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama receives a non-streaming generation request."""
    def post(url: str, *, json: dict, timeout: float) -> httpx.Response:
        assert url == "http://localhost:11434/api/generate"
        assert json == {"model": "model", "prompt": "hello", "stream": False}
        assert timeout == 5
        return httpx.Response(200, json={"response": "answer"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    assert OllamaProvider("http://localhost:11434/", "model", 5).generate("hello") == "answer"


@pytest.mark.parametrize("payload", [{}, {"response": 42}, {"response": " "}, []])
def test_ollama_rejects_invalid_responses(monkeypatch: pytest.MonkeyPatch, payload: object) -> None:
    """Malformed or empty provider responses fail with the common exception."""
    def post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=payload, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMProviderError):
        OllamaProvider("http://localhost:11434", "model", 5).generate("hello")


def test_ollama_wraps_connection_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection failures are exposed through the provider exception."""
    def post(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(LLMProviderError):
        OllamaProvider("http://localhost:11434", "model", 5).generate("hello")


def test_codex_passes_prompt_via_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex receives the prompt on stdin and runs in read-only mode."""
    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        assert command == ["codex", "exec", "--sandbox", "read-only", "--skip-git-repo-check", "--model", "test-model", "-"]
        assert kwargs["input"] == "hello"
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, stdout="answer\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert CodexProvider("codex", "test-model", 5).generate("hello") == "answer"


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("codex", 5)])
def test_codex_wraps_launch_failures(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Missing executables and timeouts use the common provider exception."""
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(LLMProviderError):
        CodexProvider("codex", None, 5).generate("hello")


def test_codex_rejects_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed CLI process never returns its stdout as a successful answer."""
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess("codex", 1, stdout="partial", stderr="failed")

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(LLMProviderError):
        CodexProvider("codex", None, 5).generate("hello")
