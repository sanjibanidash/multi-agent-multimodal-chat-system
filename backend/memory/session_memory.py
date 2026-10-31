"""
memory/session_memory.py
-------------------------
In-memory, session-scoped conversation history.

Strategy: keep only the last N messages (sliding window).
This prevents the LLM context window from overflowing on long conversations.

Difference from cache:
  - MEMORY  = per-user conversation thread (who said what, in order)
  - CACHE   = global result store keyed by query (avoids re-running the same work)
"""
from collections import deque
from datetime import datetime
from typing import Optional
import uuid

from backend.config.settings import settings

# Format used by OpenAI Chat API
Message = dict  # {"role": "user"|"assistant"|"system", "content": str}

# Global in-memory store: session_id → deque of messages
_sessions: dict[str, deque] = {}


def create_session() -> str:
    """Create a new session and return its ID."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = deque(maxlen=settings.MEMORY_MAX_MESSAGES)
    return session_id


def get_or_create_session(session_id: Optional[str]) -> str:
    """Return the session_id unchanged if it exists, else create a new session."""
    if session_id and session_id in _sessions:
        return session_id
    return create_session()


def add_message(session_id: str, role: str, content: str) -> None:
    """Append a message to the session history."""
    if session_id not in _sessions:
        _sessions[session_id] = deque(maxlen=settings.MEMORY_MAX_MESSAGES)
    _sessions[session_id].append(
        {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
    )


def get_history(session_id: str) -> list[Message]:
    """Return the conversation history as a list of {role, content} dicts (no timestamp)."""
    if session_id not in _sessions:
        return []
    # Strip internal 'timestamp' key before passing to the LLM
    return [{"role": m["role"], "content": m["content"]} for m in _sessions[session_id]]


def get_full_history(session_id: str) -> list[dict]:
    """Return history including timestamps (for the API /session endpoint)."""
    if session_id not in _sessions:
        return []
    return list(_sessions[session_id])


def delete_session(session_id: str) -> bool:
    """Remove a session. Returns True if it existed."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False


def list_sessions() -> list[str]:
    """Return all active session IDs."""
    return list(_sessions.keys())
