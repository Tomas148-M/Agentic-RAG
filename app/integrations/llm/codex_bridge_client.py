"""HTTP client for the local Codex bridge."""

import httpx

from app.integrations.llm.base import LLMProvider, LLMProviderError


class CodexBridgeProvider(LLMProvider):
    """Generate text through an authenticated local Codex bridge."""

    def __init__(self, base_url: str, token: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        """Send one prompt to the bridge and return its generated text."""
        if not self.token:
            raise LLMProviderError("Codex bridge token is not configured")
        try:
            response = httpx.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt},
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            raise LLMProviderError("Codex bridge could not complete the request") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise LLMProviderError("Codex bridge returned an invalid response")
        if not payload["response"].strip():
            raise LLMProviderError("Codex bridge returned an empty response")
        return payload["response"].strip()
