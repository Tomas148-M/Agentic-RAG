from app.core.config import Settings
from app.integrations.web.client import WebSearchClient
from app.rag.retriever import RetrievedChunk


class WebSearchTool:
    """Adapt web results to the RAG workflow's cited chunk format."""

    def __init__(self, settings: Settings) -> None:
        self.client = WebSearchClient(settings)

    def search(self, query: str) -> list[RetrievedChunk]:
        """Search the web and return results carrying their source URLs."""
        return [
            RetrievedChunk(
                text=f"{result['title']}\n{result['content']}",
                source=result["url"],
                chunk_index=index,
                distance=0.0,
            )
            for index, result in enumerate(self.client.search(query))
        ]
