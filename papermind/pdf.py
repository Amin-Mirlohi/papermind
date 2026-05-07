"""Lightweight PDF → text helpers."""

from pathlib import Path

from pypdf import PdfReader


def read_pdf(path: str | Path, *, max_pages: int | None = 2) -> str:
    """Read a PDF and return its text.

    By default, returns only the first 2 pages — enough for metadata
    (title, authors, abstract). Pass max_pages=None for the full paper.
    """
    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    return "\n\n".join(page.extract_text() or "" for page in pages)
