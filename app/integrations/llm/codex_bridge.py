"""Small authenticated HTTP bridge that exposes only Codex text generation."""

import hmac
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from app.core.config import Settings
from app.integrations.llm.base import LLMProviderError
from app.integrations.llm.codex import CodexProvider


class RequestBodyLimitMiddleware:
    """Bound request-body reads before FastAPI parses JSON, including chunked bodies."""

    def __init__(self, app: Callable[..., Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Awaitable[Any]], send: Callable[..., Awaitable[None]]) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (value for name, value in scope.get("headers", []) if name.lower() == b"content-length"),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)(scope, receive, send)
                return
            if declared_length < 0 or declared_length > self.max_bytes:
                await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
                return

        chunks: list[bytes] = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > self.max_bytes:
                await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        sent = False

        async def replay_body() -> Any:
            nonlocal sent
            if sent:
                return {"type": "http.request", "body": b"", "more_body": False}
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_body, send)


class GenerateRequest(BaseModel):
    """Prompt accepted by the bridge."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1)


class GenerateResponse(BaseModel):
    """Generated text returned to the caller."""

    response: str


def create_codex_bridge(settings: Settings) -> FastAPI:
    """Create a token-protected bridge exposing a single Codex operation."""
    token = settings.codex_bridge_token.get_secret_value() if settings.codex_bridge_token else ""
    if not token.strip():
        raise ValueError("AGENTIC_RAG_CODEX_BRIDGE_TOKEN must be set to a non-empty value")

    provider = CodexProvider(
        settings.codex_executable, settings.codex_model, settings.llm_timeout_seconds
    )
    application = FastAPI(title="Local Codex Bridge", docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(
        RequestBodyLimitMiddleware, max_bytes=settings.codex_bridge_max_body_bytes
    )

    @application.post("/generate", response_model=GenerateResponse)
    def generate(
        request: GenerateRequest,
        authorization: str | None = Header(default=None),
    ) -> GenerateResponse:
        """Authenticate and pass a prompt to the fixed Codex provider."""
        scheme, separator, supplied = (authorization or "").partition(" ")
        if (
            scheme.lower() != "bearer"
            or not separator
            or not hmac.compare_digest(supplied.encode(), token.encode())
        ):
            raise HTTPException(status_code=401, detail="Unauthorized")
        if len(request.prompt) > settings.codex_bridge_max_prompt_chars:
            raise HTTPException(status_code=413, detail="Prompt too large")
        try:
            return GenerateResponse(response=provider.generate(request.prompt))
        except LLMProviderError as exc:
            # Do not expose CLI output, paths, or authentication details to callers.
            raise HTTPException(status_code=502, detail="Codex generation failed") from exc

    return application
