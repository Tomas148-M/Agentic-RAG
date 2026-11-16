from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.dependencies import get_settings
from app.core.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Report the API status and configured application name.

    Attributes:
        status (str): Current API status.
        app_name (str): Configured application name.
    """

    status: str
    app_name: str


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the API is responding; external services are not checked.

    Args:
        settings (Settings): Settings supplied by dependency injection.

    Returns:
        HealthResponse: API status and configured application name.
    """
    return HealthResponse(status="ok", app_name=settings.app_name)
