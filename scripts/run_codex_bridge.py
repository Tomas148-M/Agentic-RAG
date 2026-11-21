"""Run the local HTTP bridge that forwards generation to Codex CLI."""

import uvicorn

from app.core.config import get_settings
from app.integrations.llm.codex_bridge import create_codex_bridge


def main() -> None:
    """Start the bridge using the configured host and port."""
    settings = get_settings()
    application = create_codex_bridge(settings)
    uvicorn.run(application, host=settings.codex_bridge_bind_host, port=settings.codex_bridge_port)


if __name__ == "__main__":
    main()
