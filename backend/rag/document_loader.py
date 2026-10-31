"""
rag/document_loader.py
-----------------------
Accepts raw file bytes + filename, extracts plain text,
and returns it ready for chunking.

Supported formats: PDF, TXT, DOCX
"""
import io
from pathlib import Path


def load_document(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a document.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename:   Original filename (used to detect format).

    Returns:
        Extracted plain text as a single string.

    Raises:
        ValueError: If the file format is unsupported.
    """
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        return _load_pdf(file_bytes)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif ext == ".docx":
        return _load_docx(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Please upload a PDF, TXT, or DOCX file."
        )


def _load_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2

        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF support. Run: pip install PyPDF2")
    except Exception as e:
        raise RuntimeError(f"Failed to extract PDF text: {e}")


def _load_docx(file_bytes: bytes) -> str:
    try:
        import docx

        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Run: pip install python-docx")
    except Exception as e:
        raise RuntimeError(f"Failed to extract DOCX text: {e}")
