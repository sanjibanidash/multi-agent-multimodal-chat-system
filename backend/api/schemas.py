"""
api/schemas.py
---------------
Pydantic models for all API request and response bodies.
These act as contracts between the frontend and backend.
"""
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User's text message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    agent: Optional[Literal["multimodal", "sql", "rag"]] = Field(
        None, description="Force a specific agent (optional; auto-routed if None)"
    )


class SourceChunk(BaseModel):
    chunk_index: int
    source_file: str
    relevance_score: float
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    agent_used: str
    session_id: str
    cache_hit: bool
    sql_query: Optional[str] = None          # SQL agent only
    raw_results: Optional[list[dict]] = None  # SQL agent only
    sources: Optional[list[SourceChunk]] = None  # RAG agent only
    metadata: Optional[dict[str, Any]] = None    # Multimodal agent only


# ── Document / Image upload ───────────────────────────────────────────────────

class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_stored: Optional[int] = None     # RAG only
    collection: Optional[str] = None        # RAG only


# ── Session ───────────────────────────────────────────────────────────────────

class SessionMessage(BaseModel):
    role: str
    content: str
    timestamp: str


class SessionResponse(BaseModel):
    session_id: str
    messages: list[SessionMessage]


class DeleteSessionResponse(BaseModel):
    deleted: bool
    session_id: str
