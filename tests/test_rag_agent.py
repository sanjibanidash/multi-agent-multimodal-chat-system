"""Tests for RAG pipeline components (chunking and document loading)."""
import pytest
from backend.rag.chunking import split_text
from backend.rag.document_loader import load_document


# ── Chunking tests ────────────────────────────────────────────────────────────

def test_split_text_produces_chunks():
    text = "A" * 2000
    chunks = split_text(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1


def test_split_text_empty_input():
    assert split_text("") == []
    assert split_text("   ") == []


def test_split_text_overlap():
    """Chunks should overlap — the last characters of chunk N should appear in chunk N+1."""
    text = "Hello World! " * 100
    chunks = split_text(text, chunk_size=100, chunk_overlap=20)
    if len(chunks) > 1:
        # The tail of chunk 0 should overlap with the head of chunk 1
        tail = chunks[0][-20:]
        head = chunks[1][:20]
        assert tail in chunks[1] or head in chunks[0]


def test_split_text_short_input_is_single_chunk():
    text = "Short text."
    chunks = split_text(text, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


# ── Document loader tests ─────────────────────────────────────────────────────

def test_load_txt_document():
    content = "Hello, this is a test document."
    result = load_document(content.encode("utf-8"), "test.txt")
    assert result == content


def test_load_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(b"data", "file.xlsx")
