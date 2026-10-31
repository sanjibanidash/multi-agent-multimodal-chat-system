"""
agents/rag_agent.py
--------------------
Retrieval-Augmented Generation (RAG) agent.

Flow:
  1. Receive user question
  2. Retrieve the most relevant text chunks from ChromaDB
  3. Build a prompt = question + retrieved context
  4. Call LLM → get an answer grounded in the documents
  5. Return answer + source chunk references
"""
from typing import Optional

from openai import OpenAI

from backend.config.settings import settings
from backend.rag.vector_store import get_vector_store

client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Number of chunks to retrieve from the vector store
TOP_K = 4

# If no relevant content is found, say so instead of hallucinating
NO_CONTEXT_REPLY = (
    "I could not find relevant information in the uploaded documents to answer your question. "
    "Please make sure you have uploaded a document, or rephrase your question."
)


def run(question: str, collection_name: str = "default") -> dict:
    """
    RAG pipeline entry point.

    Args:
        question:         The user's question.
        collection_name:  ChromaDB collection to search (one per session or 'default').

    Returns:
        {
          "answer":   str,
          "sources":  list[dict],  # [{text, source_file, chunk_index}, ...]
          "error":    str | None
        }
    """
    try:
        store = get_vector_store(collection_name)

        # ── 1. Retrieve relevant chunks ──────────────────────────────────────
        results = store.query(
            query_texts=[question],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            return {"answer": NO_CONTEXT_REPLY, "sources": [], "error": None}

        # ── 2. Build context block ───────────────────────────────────────────
        context_parts = []
        sources = []
        for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
            context_parts.append(f"[Chunk {i+1}]:\n{doc}")
            sources.append(
                {
                    "chunk_index": meta.get("chunk_index", i),
                    "source_file": meta.get("source_file", "unknown"),
                    "relevance_score": round(1 - dist, 3),  # cosine similarity
                    "text_preview": doc[:200],
                }
            )

        context_text = "\n\n".join(context_parts)

        # ── 3. LLM call ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions ONLY based on the "
                        "provided document context. If the answer is not in the context, say "
                        "'I don't know based on the provided documents.' "
                        "Do not make up information. Cite which chunk(s) supported your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context from documents:\n{context_text}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content

        return {"answer": answer, "sources": sources, "error": None}

    except Exception as e:
        return {
            "answer": f"RAG agent encountered an error: {e}",
            "sources": [],
            "error": str(e),
        }
