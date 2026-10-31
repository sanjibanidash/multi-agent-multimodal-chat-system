"""
rag/chunking.py
----------------
Splits a long document string into overlapping chunks.

Why chunk?
  Embedding models (and LLM context windows) have a token limit.
  Splitting into chunks lets us embed and retrieve only the
  most relevant pieces instead of the whole document.

Strategy: Fixed-size character window with overlap.
  chunk_size   = 800 characters  (~200 tokens)
  chunk_overlap = 150 characters  (keeps sentence context across boundaries)
"""
from typing import Iterator


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[str]:
    """
    Split `text` into overlapping chunks.

    Args:
        text:          Full document text.
        chunk_size:    Target size of each chunk in characters.
        chunk_overlap: How many characters from the end of one chunk
                       are repeated at the start of the next.

    Returns:
        List of text chunks (strings).
    """
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Advance by (chunk_size - overlap) to create the sliding window
        start += chunk_size - chunk_overlap

    return chunks
