"""Hybrid retrieval combining dense ANN and corpus-level BM25 reranking."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from razdel import tokenize

from .config import PipelineConfig
from .embedder import TextEmbedder
from .indexer import FaissIndexer

LOGGER = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    tokens = [t.text for t in tokenize(text or "")]
    if not tokens:
        tokens = (text or "").split()
    return tokens or [" "]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(scores.min())
    max_score = float(scores.max())
    if np.isclose(max_score, min_score):
        if np.isclose(max_score, 0.0):
            return np.zeros_like(scores)
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


@dataclass
class Retriever:
    indexer: FaissIndexer
    embedder: TextEmbedder
    websites_df: pd.DataFrame
    config: PipelineConfig
    _doc_text: pd.Series = field(init=False)
    _bm25: BM25Okapi | None = field(init=False, default=None)
    _bm25_ids: np.ndarray = field(init=False)
    _bm25_index: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        df = self.websites_df.copy()
        df["web_id"] = df["web_id"].astype(int)
        for column in ("title", "text"):
            if column not in df:
                df[column] = ""
            df[column] = df[column].fillna("")
        if "doc_text" not in df:
            df["doc_text"] = (df["title"].str.strip() + "\n\n" + df["text"].str.strip()).str.strip()
        df["doc_text"] = df["doc_text"].fillna("")
        self.websites_df = df
        self._doc_text = df.set_index("web_id")["doc_text"]
        self._all_web_ids = df["web_id"].tolist()

        tokens: list[list[str]] = []
        order_ids: list[int] = []
        for web_id, text in self._doc_text.items():
            order_ids.append(int(web_id))
            tokens.append(_tokenize(text))
        self._bm25_ids = np.array(order_ids, dtype=np.int64)
        self._bm25_index = {int(web_id): idx for idx, web_id in enumerate(order_ids)}
        self._bm25 = BM25Okapi(tokens) if tokens else None

    # Dense retrieval helpers -------------------------------------------------
    def _ann_candidates(self, query_vec: np.ndarray) -> Dict[int, float]:
        if self.indexer.index.ntotal == 0:
            return {}
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        scores, indices = self.indexer.search(query_vec, self.config.top_k_ann)
        candidate_scores: Dict[int, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.indexer.mapping.get(int(idx))
            if not meta:
                continue
            web_id = int(meta["web_id"])
            candidate_scores[web_id] = max(candidate_scores.get(web_id, float("-inf")), float(score))
        return candidate_scores

    # BM25 helpers ------------------------------------------------------------
    def _lexical_scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        if not self._bm25 or not query_tokens:
            return np.zeros(len(self._bm25_ids), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(list(query_tokens)), dtype=np.float32)

    def _lexical_candidates(self, lexical_scores: np.ndarray) -> list[int]:
        if lexical_scores.size == 0:
            return []
        top_k = min(
            lexical_scores.shape[0],
            max(self.config.bm25_top_k, self.config.top_k_return * 2),
        )
        if top_k == 0:
            return []
        top_idx = np.argpartition(lexical_scores, -top_k)[-top_k:]
        sorted_idx = top_idx[np.argsort(lexical_scores[top_idx])[::-1]]
        return [int(self._bm25_ids[i]) for i in sorted_idx]

    def _subset_lexical_scores(self, lexical_scores: np.ndarray, web_ids: list[int]) -> np.ndarray:
        result = np.zeros(len(web_ids), dtype=np.float32)
        if lexical_scores.size == 0:
            return result
        for i, web_id in enumerate(web_ids):
            idx = self._bm25_index.get(int(web_id))
            if idx is not None:
                result[i] = lexical_scores[idx]
        return result

    # Public API --------------------------------------------------------------
    def retrieve_topk_for_query(self, query: str, top_k: int | None = None) -> list[int]:
        embeddings = self.embedder.encode([query or ""])
        return self._retrieve_from_vector(embeddings[0], query or "", top_k=top_k)

    def retrieve_batch(self, queries: Iterable[str], top_k: int | None = None) -> list[list[int]]:
        query_list = [q or "" for q in queries]
        if not query_list:
            return []
        embeddings = self.embedder.encode(query_list)
        return [
            self._retrieve_from_vector(vec, query, top_k=top_k)
            for vec, query in zip(embeddings, query_list)
        ]

    # Internal ranking --------------------------------------------------------
    def _retrieve_from_vector(self, query_vec: np.ndarray, query: str, top_k: int | None = None) -> list[int]:
        target_k = top_k or self.config.top_k_return
        candidate_scores = self._ann_candidates(query_vec)
        query_tokens = _tokenize(query)
        lexical_scores_full = self._lexical_scores(query_tokens)
        lexical_candidates = self._lexical_candidates(lexical_scores_full)

        ordered_candidates = list(dict.fromkeys(list(candidate_scores.keys()) + lexical_candidates))
        if not ordered_candidates:
            return self._fallback_results(target_k)

        ann_scores = np.array([candidate_scores.get(web_id, 0.0) for web_id in ordered_candidates], dtype=np.float32)
        bm25_scores = self._subset_lexical_scores(lexical_scores_full, ordered_candidates)
        ann_scores = _normalize_scores(ann_scores)
        bm25_scores = _normalize_scores(bm25_scores)

        alpha = self.config.hybrid_alpha
        final_scores = alpha * ann_scores + (1 - alpha) * bm25_scores

        ranking = sorted(zip(ordered_candidates, final_scores), key=lambda x: x[1], reverse=True)
        top_results = self._ensure_topk([web_id for web_id, _ in ranking], lexical_candidates, target_k)

        return top_results

    def _ensure_topk(self, ranked_ids: list[int], lexical_candidates: list[int], target_k: int) -> list[int]:
        result: list[int] = []
        for web_id in ranked_ids:
            if web_id not in result:
                result.append(web_id)
            if len(result) == target_k:
                return result
        fallback_pool = lexical_candidates + self._all_web_ids
        for web_id in fallback_pool:
            if web_id not in result:
                result.append(web_id)
            if len(result) == target_k:
                break
        return result[:target_k]

    def _fallback_results(self, target_k: int) -> list[int]:
        if not self._all_web_ids:
            return []
        unique_ids: list[int] = []
        for web_id in self._all_web_ids:
            if web_id not in unique_ids:
                unique_ids.append(web_id)
            if len(unique_ids) == target_k:
                break
        return unique_ids
