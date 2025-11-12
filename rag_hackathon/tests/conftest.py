"""Pytest-wide fixtures and import ordering guards."""
from __future__ import annotations

# Import the embedder module early so that SentenceTransformer (torch) is
# initialised before pandas/pyarrow are loaded elsewhere. This prevents a
# Windows-specific c10.dll initialisation failure when pandas loads first.
from rag_hack import embedder  # noqa: F401
