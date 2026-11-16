from fastapi import FastAPI

from app.api.v1.router import router as api_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create the API application using the configured settings and routes.

    Returns:
        FastAPI: The configured application instance.
    """
    settings = get_settings()
    application = FastAPI(title=settings.app_name, debug=settings.debug)
    application.include_router(api_router)
    return application


app = create_app()
