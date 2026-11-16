from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from app.core.config import Settings
from app.rag.chunking import chunk_text
from app.rag.loaders import load_document


def test_load_utf8_document(tmp_path: Path) -> None:
    """Text documents retain Unicode text and a resolved source path."""
    source = tmp_path / "example.md"
    source.write_text("Žluťoučký text", encoding="utf-8")
    document = load_document(source)
    assert document.text == "Žluťoučký text"
    assert document.source == str(source.resolve())


def test_load_rejects_empty_and_unsupported_files(tmp_path: Path) -> None:
    """Empty documents and unsupported formats cannot be ingested."""
    empty = tmp_path / "empty.txt"
    empty.write_text("  ", encoding="utf-8")
    unsupported = tmp_path / "file.csv"
    unsupported.write_text("text", encoding="utf-8")
    for path in (empty, unsupported):
        with pytest.raises(ValueError):
            load_document(path)


def test_blank_pdf_is_rejected(tmp_path: Path) -> None:
    """PDF files without extractable text are not silently ingested."""
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as output:
        writer.write(output)
    with pytest.raises(ValueError, match="No text extracted"):
        load_document(path)


def test_chunk_overlap_and_boundaries() -> None:
    """Adjacent chunks overlap and the final chunk is not duplicated."""
    assert chunk_text("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]
    assert chunk_text("abcd", 4, 1) == ["abcd"]
    assert chunk_text("  ", 4, 0) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (4, -1), (4, 4)])
def test_invalid_chunk_settings(size: int, overlap: int) -> None:
    """Invalid sizes and overlaps are rejected before chunking."""
    with pytest.raises(ValueError):
        chunk_text("text", size, overlap)


def test_settings_reject_overlap_at_least_chunk_size() -> None:
    """Invalid chunk settings are rejected before the application uses them."""
    with pytest.raises(ValidationError):
        Settings(chunk_size=10, chunk_overlap=10, _env_file=None)
