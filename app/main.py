from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as api_router
from app.core.config import get_settings
from app.observability.phoenix import flush_phoenix


def create_app() -> FastAPI:
    """Create the API application using the configured settings and routes.

    Returns:
        FastAPI: The configured application instance.
    """
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Flush buffered observability events during graceful shutdown."""
        yield
        flush_phoenix()

    application = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(api_router)
    return application


app = create_app()
