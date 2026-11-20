"""Run the Agentic RAG API as a desktop executable entry point."""

import uvicorn

from app.core.config import DEFAULT_API_HOST, DEFAULT_API_PORT
from app.main import app


def main() -> None:
    """Start the API server on the local machine."""
    uvicorn.run(app, host=DEFAULT_API_HOST, port=DEFAULT_API_PORT)


if __name__ == "__main__":
    main()
