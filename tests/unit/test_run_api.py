from unittest.mock import Mock

import run_api


def test_main_starts_api_on_configured_loopback_port(monkeypatch) -> None:
    """Start Uvicorn with the configured local address and port."""
    run_server = Mock()
    monkeypatch.setattr(run_api.uvicorn, "run", run_server)

    run_api.main()

    run_server.assert_called_once_with(
        run_api.app,
        host=run_api.DEFAULT_API_HOST,
        port=run_api.DEFAULT_API_PORT,
    )
