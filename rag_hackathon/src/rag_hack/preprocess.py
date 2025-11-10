"""Pre-processing utilities."""
from __future__ import annotations

import html
import logging
import re
from typing import Iterable

from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)
_SPACE_RE = re.compile(r"\s+")
_NORMALIZE_MAP = str.maketrans({
    "ё": "е",
    "Ё": "е",
})


def clean_html(raw_html: str) -> str:
    """Strip HTML tags, scripts and styles."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        element.decompose()
    text = soup.get_text(" ", strip=True)
    text = html.unescape(text)
    return text


def normalize_text(text: str) -> str:
    """Lowercase, normalize spacing and simple Cyrillic normalization."""
    text = text.translate(_NORMALIZE_MAP)
    text = text.lower()
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_documents(records: Iterable[dict]) -> list[dict]:
    """Apply cleaning and create combined doc_text field."""
    processed: list[dict] = []
    for item in records:
        title = normalize_text(clean_html(item.get("title", ""))) if item.get("title") else ""
        body = normalize_text(clean_html(item.get("text", ""))) if item.get("text") else ""
        doc_text = f"{title}\n\n{body}".strip()
        processed.append({
            **item,
            "title": title,
            "text": body,
            "doc_text": doc_text,
        })
    LOGGER.debug("Preprocessed %d documents", len(processed))
    return processed
