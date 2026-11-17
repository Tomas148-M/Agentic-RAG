from app.rag.retriever import DocumentRetriever, RetrievedChunk


class RAGTool:
    """Expose local document retrieval through the shared tool interface."""

    def __init__(self, retriever: DocumentRetriever) -> None:
        self.retriever = retriever

    def search(self, query: str) -> list[RetrievedChunk]:
        """Search ingested document chunks for a query."""
        return self.retriever.retrieve(query)
