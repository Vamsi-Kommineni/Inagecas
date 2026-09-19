"""Turn raw documents into normalized plain text for chunking."""

from __future__ import annotations

import io
import re

from bs4 import BeautifulSoup

_WHITESPACE = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _parse_html(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return _normalize(soup.get_text("\n"))


def _parse_pdf(raw: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _normalize("\n\n".join(pages))


def parse(source_type: str, data: str | bytes) -> str:
    """Normalize ``data`` to plain text.

    Markdown is kept mostly intact so headings can drive chunking; HTML and PDF
    are reduced to their visible text.
    """
    if source_type == "pdf":
        if isinstance(data, str):
            data = data.encode("utf-8", "ignore")
        return _parse_pdf(data)

    text = data.decode("utf-8", "ignore") if isinstance(data, bytes) else data
    if source_type == "html":
        return _parse_html(text)
    return _normalize(text)
