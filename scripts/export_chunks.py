"""Export document chunks to JSON sidecar files without using a vector store."""

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.rag.chunking import chunk_text
from app.rag.loaders import load_document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def find_documents(path: Path) -> list[Path]:
    """Return a file or supported files in a directory recursively."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES
        )
    raise FileNotFoundError(f"Path does not exist: {path}")


def export_document(path: Path, chunk_size: int, chunk_overlap: int) -> tuple[Path, int]:
    """Write a deterministic JSON sidecar containing one document's chunks."""
    document = load_document(path)
    chunks = chunk_text(document.text, chunk_size, chunk_overlap)
    sidecar = path.with_name(f"{path.name}.chunks.json")
    payload = {
        "source": document.source,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": [
            {"index": index, "text": text}
            for index, text in enumerate(chunks)
        ],
    }
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar, len(chunks)


def main() -> None:
    """Load documents and write their chunks beside the original files."""
    parser = argparse.ArgumentParser(
        description="Export document chunks to JSON files beside their sources."
    )
    parser.add_argument("path", type=Path, help="Document or directory containing documents")
    args = parser.parse_args()
    try:
        files = find_documents(args.path)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    if not files:
        parser.error("No supported documents found")

    settings = get_settings()
    for path in files:
        sidecar, count = export_document(path, settings.chunk_size, settings.chunk_overlap)
        print(f"{path}: wrote {count} chunks to {sidecar}")


if __name__ == "__main__":
    main()
