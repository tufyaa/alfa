"""Pre-processing utilities for the hackathon corpus."""
from __future__ import annotations

import html
import logging
import re
from typing import Iterable

from bs4 import BeautifulSoup

LOGGER = logging.getLogger(__name__)

_SPACE_RE = re.compile(r"\s+")
_DANGLING_PUNCT_RE = re.compile(r"[«»“”„]")
_NORMALIZE_MAP = str.maketrans(
    {
        "Ё": "Е",
        "ё": "е",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "\u00a0": " ",
    }
)


def clean_html(raw_html: str) -> str:
    """Strip markup, inline scripts/styles and decode HTML entities."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer", "noscript", "svg"]):
        element.decompose()
    text = soup.get_text(" ", strip=True)
    return html.unescape(text)


def normalize_text(text: str) -> str:
    """Lower-case, normalize spacing and harmonise punctuation."""
    if not text:
        return ""
    text = text.translate(_NORMALIZE_MAP)
    text = _DANGLING_PUNCT_RE.sub(" ", text)
    text = text.lower()
    text = _SPACE_RE.sub(" ", text)
    return text.strip()


def preprocess_documents(records: Iterable[dict]) -> list[dict]:
    """Apply deterministic cleaning and build the doc_text field."""
    processed: list[dict] = []
    for item in records:
        title = normalize_text(clean_html(item.get("title", "") or ""))
        body = normalize_text(clean_html(item.get("text", "") or ""))
        doc_text = f"{title}\n\n{body}".strip()
        processed.append(
            {
                **item,
                "title": title,
                "text": body,
                "doc_text": doc_text,
            }
        )
    LOGGER.debug("Preprocessed %d documents", len(processed))
    return processed
