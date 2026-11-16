from fastapi.testclient import TestClient

from app.api.dependencies import get_settings
from app.core.config import Settings
from app.main import create_app


def test_health_uses_injected_settings() -> None:
    """The health endpoint returns the application name from its dependency."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(
        app_name="Test API", _env_file=None
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "app_name": "Test API"}
