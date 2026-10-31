"""
api/routes.py
--------------
All FastAPI route handlers. Thin layer — each handler:
  1. Validates input (Pydantic does this automatically)
  2. Calls the router to pick an agent
  3. Checks / updates the cache
  4. Calls the agent
  5. Updates session memory
  6. Returns a structured response
"""
import io
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends

from backend.api.schemas import (
    ChatRequest, ChatResponse, UploadResponse,
    SessionResponse, SessionMessage, DeleteSessionResponse,
    SourceChunk,
)
from backend.orchestration.router import route
from backend.memory import session_memory
from backend.cache.cache_manager import cache
from backend.rag.vector_store import ingest_document
from backend import agents  # noqa – submodules imported below

from backend.agents import multimodal_agent, sql_agent, rag_agent

router = APIRouter()


# ── /chat ─────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message to the agent system")
async def chat(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    agent: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Main chat endpoint. Supports optional file uploads alongside the text message.
    """
    # ── Session ──────────────────────────────────────────────────────────────
    sid = session_memory.get_or_create_session(session_id)

    # ── File handling ─────────────────────────────────────────────────────────
    file_bytes: Optional[bytes] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None
    file_is_image = False

    if file and file.filename:
        file_bytes = await file.read()
        filename = file.filename
        content_type = file.content_type or ""
        file_is_image = content_type.startswith("image/")

    # ── Route ─────────────────────────────────────────────────────────────────
    # Check if the RAG collection for this session has documents
    from backend.rag.vector_store import list_collections
    has_rag_docs = sid in list_collections()

    chosen_agent = route(
        question=message,
        has_file=bool(file_bytes),
        file_is_image=file_is_image,
        has_rag_documents=has_rag_docs,
        forced_agent=agent if agent in ("multimodal", "sql", "rag") else None,
    )

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = cache.make_key(chosen_agent, message)
    hit, cached_response = cache.get(cache_key)
    if hit and not file_bytes:  # Never serve cached result when a file is attached
        return ChatResponse(
            answer=cached_response["answer"],
            agent_used=chosen_agent,
            session_id=sid,
            cache_hit=True,
            sql_query=cached_response.get("sql_query"),
            raw_results=cached_response.get("raw_results"),
            sources=cached_response.get("sources"),
            metadata=cached_response.get("metadata"),
        )

    # ── Add user message to memory ────────────────────────────────────────────
    session_memory.add_message(sid, "user", message)
    history = session_memory.get_history(sid)

    # ── Dispatch to agent ─────────────────────────────────────────────────────
    result: dict = {}

    if chosen_agent == "sql":
        result = sql_agent.run(message)

    elif chosen_agent == "rag":
        result = rag_agent.run(message, collection_name=sid)

    else:  # multimodal (default)
        result = multimodal_agent.run(
            user_message=message,
            file_bytes=file_bytes,
            filename=filename,
            content_type=content_type,
        )

    # ── Add assistant response to memory ─────────────────────────────────────
    session_memory.add_message(sid, "assistant", result.get("answer", ""))

    # ── Store in cache (only text-only queries) ───────────────────────────────
    if not file_bytes:
        cache.set(cache_key, result)

    # ── Build response ────────────────────────────────────────────────────────
    raw_sources = result.get("sources")
    sources = (
        [SourceChunk(**s) for s in raw_sources] if raw_sources else None
    )

    return ChatResponse(
        answer=result.get("answer", ""),
        agent_used=chosen_agent,
        session_id=sid,
        cache_hit=False,
        sql_query=result.get("sql_query"),
        raw_results=result.get("raw_results"),
        sources=sources,
        metadata=result.get("metadata"),
    )


# ── /upload/document ──────────────────────────────────────────────────────────

@router.post("/upload/document", response_model=UploadResponse,
             summary="Upload a document for the RAG agent")
async def upload_document(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    """Ingest a PDF, TXT, or DOCX file into the RAG vector store."""
    sid = session_memory.get_or_create_session(session_id)
    file_bytes = await file.read()
    try:
        info = ingest_document(file_bytes, file.filename, collection_name=sid)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    return UploadResponse(
        message=f"Document '{file.filename}' ingested successfully.",
        filename=file.filename,
        chunks_stored=info["chunks_stored"],
        collection=sid,
    )


# ── /upload/image ─────────────────────────────────────────────────────────────

@router.post("/upload/image", response_model=UploadResponse,
             summary="Upload an image for the multimodal agent")
async def upload_image(file: UploadFile = File(...)):
    """Accept an image upload (handled via /chat; this endpoint just validates)."""
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")
    return UploadResponse(
        message=f"Image '{file.filename}' ready. Include it in a /chat request.",
        filename=file.filename,
    )


# ── /session/{session_id} ─────────────────────────────────────────────────────

@router.get("/session/{session_id}", response_model=SessionResponse,
            summary="Get conversation history for a session")
async def get_session(session_id: str):
    history = session_memory.get_full_history(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return SessionResponse(
        session_id=session_id,
        messages=[SessionMessage(**m) for m in history],
    )


@router.delete("/session/{session_id}", response_model=DeleteSessionResponse,
               summary="Delete a session and its conversation history")
async def delete_session(session_id: str):
    deleted = session_memory.delete_session(session_id)
    return DeleteSessionResponse(deleted=deleted, session_id=session_id)
