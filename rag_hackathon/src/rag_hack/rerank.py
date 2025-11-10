"""Placeholder for future cross-encoder reranker implementations."""
from __future__ import annotations

from typing import Iterable, List


def rerank_candidates(query: str, candidates: Iterable[int]) -> List[int]:
    """Identity reranker placeholder."""
    return list(candidates)
