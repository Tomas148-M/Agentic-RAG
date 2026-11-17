def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping character windows, omitting blank windows.

    Args:
        text (str): Source text.
        size (int): Maximum characters per chunk.
        overlap (int): Shared characters between adjacent windows.

    Returns:
        list[str]: Nonblank text windows in source order.

    Raises:
        ValueError: If chunk size or overlap is invalid.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap must be in [0, size)")
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size]
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks
