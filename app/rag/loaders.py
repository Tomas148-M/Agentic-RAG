from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Document:
    """A loaded document with a stable absolute source path."""

    source: str
    text: str


def load_document(path: Path) -> Document:
    """Load a UTF-8 text, Markdown, or text-based PDF file.

    Args:
        path (Path): Path to a supported document.

    Returns:
        Document: Extracted text and its resolved source path.

    Raises:
        ValueError: If the extension is unsupported or no text was extracted.
    """
    path = path.resolve(strict=True)
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".pdf":
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        raise ValueError(f"Unsupported document type: {suffix}")
    if not text.strip():
        raise ValueError(f"No text extracted from document: {path}")
    return Document(source=str(path), text=text)
