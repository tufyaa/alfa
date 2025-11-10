"""Hybrid retrieval combining dense ANN and BM25 reranking."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from razdel import tokenize

from .config import PipelineConfig
from .embedder import TextEmbedder
from .indexer import FaissIndexer

LOGGER = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return [t.text for t in tokenize(text)]


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = scores.min()
    max_score = scores.max()
    if np.isclose(max_score, min_score):
        return np.ones_like(scores)
    return (scores - min_score) / (max_score - min_score)


@dataclass
class Retriever:
    indexer: FaissIndexer
    embedder: TextEmbedder
    websites_df: pd.DataFrame
    config: PipelineConfig

    def __post_init__(self) -> None:
        self._doc_text = self.websites_df.set_index("web_id")["doc_text"].fillna("")

    def retrieve_topk_for_query(self, query: str) -> list[int]:
        query_vec = self.embedder.encode([query])
        scores, indices = self.indexer.search(query_vec, self.config.top_k_ann)
        scores = scores[0]
        indices = indices[0]

        candidate_scores: Dict[int, float] = {}
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            meta = self.indexer.mapping[int(idx)]
            web_id = int(meta["web_id"])
            candidate_scores[web_id] = max(candidate_scores.get(web_id, float("-inf")), float(score))

        if not candidate_scores:
            return []

        web_ids = list(candidate_scores.keys())
        ann_scores = np.array([candidate_scores[w] for w in web_ids], dtype=np.float32)
        ann_scores = _normalize_scores(ann_scores)

        doc_texts = self._doc_text.loc[web_ids]
        bm25 = BM25Okapi([_tokenize(text) for text in doc_texts])
        bm25_scores = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)
        bm25_scores = _normalize_scores(bm25_scores)

        alpha = self.config.hybrid_alpha
        final_scores = alpha * ann_scores + (1 - alpha) * bm25_scores

        ranking = sorted(zip(web_ids, final_scores), key=lambda x: x[1], reverse=True)
        top_webs = [web_id for web_id, _ in ranking[: self.config.top_k_return]]
        return top_webs

    def retrieve_batch(self, queries: Iterable[str]) -> list[list[int]]:
        results = []
        for query in queries:
            results.append(self.retrieve_topk_for_query(query))
        return results
