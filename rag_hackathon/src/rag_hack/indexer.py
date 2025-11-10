"""FAISS index management."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class FaissIndexer:
    """Thin wrapper around IndexFlatIP with serialization helpers."""

    def __init__(self, dim: int) -> None:
        self.index = faiss.IndexFlatIP(dim)
        self.mapping: dict[int, dict] = {}

    def add(self, embeddings: np.ndarray, metadata: Iterable[dict]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        start_id = self.index.ntotal
        self.index.add(embeddings)
        for i, meta in enumerate(metadata):
            self.mapping[start_id + i] = meta
        LOGGER.info("Added %d vectors to index", embeddings.shape[0])

    def save(self, index_path: Path, mapping_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as fout:
            json.dump(self.mapping, fout, ensure_ascii=False)
        LOGGER.info("Saved index to %s", index_path)

    @classmethod
    def load(cls, index_path: Path, mapping_path: Path) -> "FaissIndexer":
        index = faiss.read_index(str(index_path))
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        instance = cls(index.d)
        instance.index = index
        instance.mapping = {int(k): v for k, v in mapping.items()}
        LOGGER.info("Loaded index with %d vectors", index.ntotal)
        return instance

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        return self.index.search(query_vectors, top_k)


def build_faiss_index(embeddings: np.ndarray, chunks_df: pd.DataFrame) -> FaissIndexer:
    indexer = FaissIndexer(dim=embeddings.shape[1])
    metadata = (
        {
            "web_id": int(row.web_id),
            "chunk_id": row.chunk_id,
            "chunk_order": int(row.chunk_order),
        }
        for row in chunks_df.itertuples()
    )
    indexer.add(embeddings, metadata)
    return indexer
