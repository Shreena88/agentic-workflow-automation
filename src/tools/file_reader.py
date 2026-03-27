"""File reader tool supporting plain text and PDF files."""
from __future__ import annotations
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Configurable allowed base directory (prevents path traversal)
ALLOWED_DIR = Path(os.getenv("ALLOWED_FILE_DIRS", "./data")).resolve()


def file_reader_tool(file_path: str) -> str:
    """Read a plain text or PDF file. Search in ALLOWED_DIR if not found."""
    path = Path(file_path)
    if not path.exists():
        # Fallback to checking the allowed data directory
        path = ALLOWED_DIR / path.name

    resolved = path.resolve()
    # Security: restrict to allowed directory
    if not str(resolved).startswith(str(ALLOWED_DIR)):
        raise PermissionError(f"Access denied: '{file_path}' is outside the allowed directory.")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(resolved)
    elif suffix in (".txt", ".md", ".csv", ".json"):
        return resolved.read_text(encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _read_pdf(path: Path) -> str:
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF reading: pip install PyPDF2")

    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)
