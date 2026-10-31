"""
agents/multimodal_agent.py
---------------------------
Handles text, image, PDF, DOCX and audio inputs.

Flow:
  1. Detect file type for each uploaded file
  2. Extract text content from the file
  3. Combine user message + extracted content into a prompt
  4. Send to LLM and return the response
"""
import io
import mimetypes
from pathlib import Path
from typing import Optional

from openai import OpenAI

from backend.config.settings import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)


# ─── File helpers ────────────────────────────────────────────────────────────

def _get_media_type(filename: str, content_type: str = "") -> str:
    """Return a simplified category for the file."""
    mime = content_type or mimetypes.guess_type(filename)[0] or ""
    if mime.startswith("image/"):
        return "image"
    if mime == "application/pdf" or filename.endswith(".pdf"):
        return "pdf"
    if mime.startswith("text/") or filename.endswith(".txt"):
        return "text"
    if "wordprocessingml" in mime or filename.endswith(".docx"):
        return "docx"
    if mime.startswith("audio/") or filename.endswith((".mp3", ".wav", ".m4a")):
        return "audio"
    return "unknown"


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def _extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return f"[DOCX extraction failed: {e}]"


def _transcribe_audio(file_bytes: bytes, filename: str) -> str:
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes),
        )
        return transcript.text
    except Exception as e:
        return f"[Audio transcription failed: {e}]"


def _analyze_image_with_vision(file_bytes: bytes, user_question: str) -> str:
    """Use GPT-4o Vision to directly analyze an image."""
    import base64
    b64 = base64.b64encode(file_bytes).decode()
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question or "Describe this image in detail."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Main entry point ────────────────────────────────────────────────────────

def run(
    user_message: str,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> dict:
    """
    Process a multimodal request and return a dict:
      {
        "answer": str,
        "metadata": {
          "media_type": str,      # what kind of file was processed
          "extracted_text": str,  # raw text extracted (if any)
        }
      }
    """
    media_type = "text"
    extracted_text = ""
    answer = ""

    # ── Handle uploaded file ─────────────────────────────────────────────────
    if file_bytes and filename:
        media_type = _get_media_type(filename, content_type or "")

        if media_type == "image":
            # Images go directly to vision API — no text extraction needed
            answer = _analyze_image_with_vision(file_bytes, user_message)
            return {
                "answer": answer,
                "metadata": {"media_type": media_type, "extracted_text": ""},
            }

        elif media_type == "pdf":
            extracted_text = _extract_text_from_pdf(file_bytes)

        elif media_type == "docx":
            extracted_text = _extract_text_from_docx(file_bytes)

        elif media_type == "text":
            extracted_text = file_bytes.decode("utf-8", errors="ignore")

        elif media_type == "audio":
            extracted_text = _transcribe_audio(file_bytes, filename)

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt_parts = []
    if user_message:
        prompt_parts.append(f"User question: {user_message}")
    if extracted_text.strip():
        prompt_parts.append(
            f"\n--- Extracted content from uploaded {media_type.upper()} ---\n"
            f"{extracted_text[:4000]}"  # limit context length
        )

    full_prompt = "\n".join(prompt_parts) or "Hello! How can you help me?"

    # ── LLM call ─────────────────────────────────────────────────────────────
    response = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful multimodal AI assistant. You can understand text, "
                    "extracted PDF/DOCX content, image descriptions, and audio transcriptions. "
                    "Answer clearly and concisely based on the provided information."
                ),
            },
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "metadata": {
            "media_type": media_type,
            "extracted_text": extracted_text[:500] if extracted_text else "",
        },
    }
