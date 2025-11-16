"""Cross-encoder reranker utilities.

This module provides an optional second-stage reranker using a cross-encoder
from sentence-transformers. It is imported lazily and supports a lightweight
"dry" mode controlled by env var RAG_DRY_RERANK=1 for environments without
the model available.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Sequence

import numpy as np


class CrossEncoderReranker:
    """Thin wrapper around sentence-transformers CrossEncoder.

    If RAG_DRY_RERANK=1 is set, produces deterministic scores based on
    input lengths to enable testing without model downloads.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.device = device
        self._model = None
        self._dry = os.environ.get("RAG_DRY_RERANK", "0") == "1"

    def _ensure_model(self) -> None:
        if self._dry or self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception as exc:  # pragma: no cover - import-time environment
            raise RuntimeError(
                "CrossEncoder not available. Install sentence-transformers or enable RAG_DRY_RERANK=1"
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Return relevance scores for (query, passage) pairs.

        Higher is better. Output shape: (len(passages),).
        """
        if not passages:
            return np.zeros(0, dtype=np.float32)
        if self._dry:
            # Simple deterministic score: favor length match between query and passage
            q = max(1, len(query or ""))
            scores = []
            for p in passages:
                lp = max(1, len(p or ""))
                # Closer lengths => higher score; add light character diversity bonus
                diversity = len(set((p or "").lower().split()))
                val = 1.0 / (1.0 + abs(lp - q) / (q + 1.0)) + 0.0001 * diversity
                scores.append(val)
            return np.asarray(scores, dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        pairs = [(query, p) for p in passages]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]
        return np.asarray(scores, dtype=np.float32)

    def rerank(self, query: str, doc_ids: Sequence[int], doc_texts: Sequence[str]) -> list[int]:
        scores = self.score(query, doc_texts)
        order = np.argsort(scores)[::-1]
        return [int(doc_ids[i]) for i in order]


def rerank_identity(query: str, candidates: Iterable[int]) -> List[int]:
    """No-op reranker for compatibility."""
    return list(candidates)
