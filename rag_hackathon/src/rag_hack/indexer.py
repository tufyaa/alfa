"""FAISS index management with NumPy fallback.

On some macOS/Apple Silicon setups FAISS/OpenMP can cause lock contention.
Set environment variable `RAG_DISABLE_FAISS=1` to use a pure-NumPy fallback.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import os
from typing import Iterable

import numpy as np
import pandas as pd
from typing import Optional

try:
    _DISABLE_FAISS = os.environ.get("RAG_DISABLE_FAISS", "0") == "1"
    if not _DISABLE_FAISS:
        import faiss  # type: ignore
        _HAS_FAISS = True
    else:
        _HAS_FAISS = False
except Exception:
    _HAS_FAISS = False

LOGGER = logging.getLogger(__name__)


class FaissIndexer:
    """Thin wrapper around IndexFlatIP with serialization helpers."""

    def __init__(self, dim: int) -> None:
        self._dim = int(dim)
        self.mapping: dict[int, dict] = {}
        if _HAS_FAISS:
            # Limit FAISS OpenMP threads to reduce lock contention with BLAS/tokenizers
            try:
                threads = int(os.environ.get("FAISS_NUM_THREADS", "1"))
                if threads > 0 and hasattr(faiss, "omp_set_num_threads"):  # type: ignore[name-defined]
                    faiss.omp_set_num_threads(threads)  # type: ignore[name-defined]
            except Exception:
                pass
            self.index = faiss.IndexFlatIP(self._dim)  # type: ignore[name-defined]
            self._vectors: Optional[np.ndarray] = None
        else:
            # Fallback: keep vectors in-memory and emulate an index API
            class _DummyIndex:
                def __init__(self) -> None:
                    self.ntotal = 0

            self.index = _DummyIndex()
            self._vectors = np.empty((0, self._dim), dtype=np.float32)
        # mapping initialized in __init__

    def add(self, embeddings: np.ndarray, metadata: Iterable[dict]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        start_id = int(self.index.ntotal)
        if _HAS_FAISS:
            self.index.add(embeddings)  # type: ignore[union-attr]
        else:
            assert self._vectors is not None
            self._vectors = np.vstack([self._vectors, embeddings])
            self.index.ntotal = int(self._vectors.shape[0])
        for i, meta in enumerate(metadata):
            self.mapping[start_id + i] = meta
        LOGGER.info("Added %d vectors to index", embeddings.shape[0])

    def save(self, index_path: Path, mapping_path: Path) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_FAISS:
            faiss.write_index(self.index, str(index_path))  # type: ignore[name-defined]
        else:
            assert self._vectors is not None
            np.savez_compressed(index_path, vectors=self._vectors, dim=self._dim)
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with mapping_path.open("w", encoding="utf-8") as fout:
            json.dump(self.mapping, fout, ensure_ascii=False)
        LOGGER.info("Saved index to %s", index_path)

    @classmethod
    def load(cls, index_path: Path, mapping_path: Path) -> "FaissIndexer":
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if _HAS_FAISS:
            index = faiss.read_index(str(index_path))  # type: ignore[name-defined]
            instance = cls(index.d)  # type: ignore[attr-defined]
            instance.index = index
            instance.mapping = {int(k): v for k, v in mapping.items()}
            LOGGER.info("Loaded index with %d vectors (FAISS)", index.ntotal)
            return instance
        else:
            data = np.load(index_path)
            vectors = data["vectors"].astype(np.float32)
            dim = int(data["dim"]) if "dim" in data.files else vectors.shape[1]
            instance = cls(dim)
            instance._vectors = vectors
            instance.index.ntotal = int(vectors.shape[0])
            instance.mapping = {int(k): v for k, v in mapping.items()}
            LOGGER.info("Loaded index with %d vectors (NumPy)", instance.index.ntotal)
            return instance

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        if _HAS_FAISS:
            return self.index.search(query_vectors, top_k)  # type: ignore[union-attr]
        # Fallback: cosine-similarity like FAISS IP over L2-normalized vectors
        assert self._vectors is not None
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        # Ensure normalization to match behavior in embedder
        q = query_vectors
        scores = q @ self._vectors.T
        k = min(top_k, scores.shape[1]) if scores.size else 0
        if k == 0:
            return np.zeros((scores.shape[0], 0), dtype=np.float32), np.full((scores.shape[0], 0), -1, dtype=np.int64)
        # Argpartition top-k per row
        idx_part = np.argpartition(scores, -k, axis=1)[:, -k:]
        # Sort top-k per row
        row_indices = np.arange(scores.shape[0])[:, None]
        top_scores = np.take_along_axis(scores, idx_part, axis=1)
        order = np.argsort(top_scores, axis=1)[:, ::-1]
        sorted_idx = np.take_along_axis(idx_part, order, axis=1)
        sorted_scores = np.take_along_axis(scores, sorted_idx, axis=1)
        return sorted_scores.astype(np.float32), sorted_idx.astype(np.int64)


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
