"""
rag/vector_store.py
--------------------
Manages ChromaDB collections — one per RAG session or document set.

Workflow:
  1. `ingest_document()` — load, chunk, embed, and store a document.
  2. `get_vector_store()` — return a ChromaDB collection for querying.

ChromaDB uses its own built-in embedding function by default.
We override it with OpenAI embeddings for better quality.
"""
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from backend.config.settings import settings
from backend.rag.document_loader import load_document
from backend.rag.chunking import split_text

# Persistent ChromaDB client — data survives server restarts
_client = chromadb.PersistentClient(path=str(settings.CHROMA_DIR))

# OpenAI embedding function used for all collections
_embedding_fn = OpenAIEmbeddingFunction(
    api_key=settings.OPENAI_API_KEY,
    model_name=settings.EMBEDDING_MODEL,
)


def get_vector_store(collection_name: str = "default") -> chromadb.Collection:
    """Return (or create) a ChromaDB collection by name."""
    return _client.get_or_create_collection(
        name=collection_name,
        embedding_function=_embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_document(
    file_bytes: bytes,
    filename: str,
    collection_name: str = "default",
) -> dict:
    """
    Full RAG ingestion pipeline for one document.

    1. Extract text from the file.
    2. Split into chunks.
    3. Embed and store in ChromaDB.

    Returns:
        {"chunks_stored": int, "collection": str, "filename": str}
    """
    # Step 1 – Extract
    raw_text = load_document(file_bytes, filename)

    # Step 2 – Chunk
    chunks = split_text(raw_text)
    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    # Step 3 – Embed + Store
    collection = get_vector_store(collection_name)

    ids = [f"{filename}::chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"source_file": filename, "chunk_index": i}
        for i in range(len(chunks))
    ]

    # ChromaDB upsert — safe to call multiple times (won't create duplicates)
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return {
        "chunks_stored": len(chunks),
        "collection": collection_name,
        "filename": filename,
    }


def delete_collection(collection_name: str) -> None:
    """Delete a ChromaDB collection (e.g., when a session ends)."""
    try:
        _client.delete_collection(collection_name)
    except Exception:
        pass  # Collection didn't exist — that's fine


def list_collections() -> list[str]:
    """Return names of all existing ChromaDB collections."""
    return [c.name for c in _client.list_collections()]
