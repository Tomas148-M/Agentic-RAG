import httpx

from app.integrations.llm.base import LLMProvider, LLMProviderError


class OllamaProvider(LLMProvider):
    """Generate text using Ollama's non-streaming HTTP API."""

    def __init__(self, base_url: str, model: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Generate text using the Ollama API.

        Args:
            prompt (str): Input text sent to the configured model.

        Returns:
            str: The model's generated response.

        Raises:
            LLMProviderError: If the request fails or returns invalid text.
        """
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            raise LLMProviderError("Ollama could not complete the request") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise LLMProviderError("Ollama returned an invalid response")
        if not payload["response"].strip():
            raise LLMProviderError("Ollama returned an empty response")
        return payload["response"]
