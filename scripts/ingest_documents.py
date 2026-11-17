import argparse
from pathlib import Path

from app.core.config import get_settings
from app.integrations.embeddings.local import LocalEmbeddingProvider
from app.rag.ingestion import ingest_document
from app.rag.vector_store import ChromaVectorStore


def main() -> None:
    """Ingest one file or supported files in a directory into Chroma."""
    parser = argparse.ArgumentParser(description="Ingest documents into the local vector database.")
    parser.add_argument("path", type=Path, help="Document or directory containing documents")
    args = parser.parse_args()
    if args.path.is_file():
        files = [args.path]
    elif args.path.is_dir():
        files = sorted(p for p in args.path.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md", ".pdf"})
    else:
        parser.error(f"Path does not exist: {args.path}")
    if not files:
        parser.error("No supported documents found")
    settings = get_settings()
    store = ChromaVectorStore(settings.chroma_path, settings.chroma_collection, settings.embedding_model)
    embeddings = LocalEmbeddingProvider(settings.embedding_model)
    for path in files:
        count = ingest_document(path, store, embeddings, settings.chunk_size, settings.chunk_overlap)
        print(f"{path}: stored {count} chunks")


if __name__ == "__main__":
    main()
