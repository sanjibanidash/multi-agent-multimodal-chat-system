"""
orchestration/router.py
------------------------
Decides which agent should handle a given user request.

Routing priority:
  1. If the user explicitly selected an agent in the UI → honour that choice.
  2. If files are attached and they look like RAG documents → RAG agent.
  3. Otherwise ask a lightweight LLM call to classify the intent.

Why LLM-based routing?
  Simple keyword matching breaks on paraphrasing. A small LLM call is more
  robust and only adds ~100 ms of latency.
"""
import re
from typing import Optional, Literal

from openai import OpenAI

from backend.config.settings import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

AgentType = Literal["multimodal", "sql", "rag"]

# Keywords that strongly suggest the SQL agent
SQL_KEYWORDS = re.compile(
    r"\b(sales|revenue|total|profit|customers?|products?|employees?|"
    r"top \d|best|worst|average|database|table|query|sql|records?|rows?|"
    r"how many|count|sum|list all|show me all)\b",
    re.IGNORECASE,
)

# Keywords that strongly suggest the RAG agent
RAG_KEYWORDS = re.compile(
    r"\b(document|pdf|uploaded|file|according to|based on|in the (doc|file|text|report)|"
    r"summarize|summary|what does .* say|extract from)\b",
    re.IGNORECASE,
)


def _llm_classify(question: str) -> AgentType:
    """
    Use a tiny LLM prompt to classify the intent.
    Returns 'sql', 'rag', or 'multimodal'.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # always use the cheap model for routing
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a routing classifier for a multi-agent chat system. "
                    "Given a user question, output EXACTLY one word: "
                    "'sql' if the question is about business data, sales, employees, or database records; "
                    "'rag' if the question is about an uploaded document, PDF, or file content; "
                    "'multimodal' for everything else (general questions, images, audio)."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=5,
    )
    label = response.choices[0].message.content.strip().lower()
    if label in ("sql", "rag", "multimodal"):
        return label  # type: ignore
    return "multimodal"


def route(
    question: str,
    has_file: bool = False,
    file_is_image: bool = False,
    has_rag_documents: bool = False,
    forced_agent: Optional[AgentType] = None,
) -> AgentType:
    """
    Determine which agent should handle the request.

    Args:
        question:          The user's text input.
        has_file:          True if any file is attached.
        file_is_image:     True if the attachment is an image.
        has_rag_documents: True if the RAG vector store has documents loaded.
        forced_agent:      If the user explicitly chose an agent in the UI.

    Returns:
        One of 'multimodal', 'sql', 'rag'.
    """
    # ── 1. Respect explicit user choice ────────────────────────────────────
    if forced_agent:
        return forced_agent

    # ── 2. Image or audio file → always multimodal ──────────────────────────
    if has_file and file_is_image:
        return "multimodal"

    # ── 3. Non-image file upload AND RAG docs loaded → RAG ─────────────────
    if has_file and not file_is_image and has_rag_documents:
        return "rag"

    # ── 4. Fast keyword pre-check to avoid LLM call ──────────────────────
    if SQL_KEYWORDS.search(question):
        return "sql"
    if RAG_KEYWORDS.search(question) and has_rag_documents:
        return "rag"

    # ── 5. LLM-based classification ────────────────────────────────────────
    return _llm_classify(question)
