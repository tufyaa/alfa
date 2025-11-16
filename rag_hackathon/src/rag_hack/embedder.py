"""Embedding utilities using SentenceTransformer.

This module avoids importing heavy libraries at import time to reduce
the chance of low-level lock contention on some macOS setups.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import os
from typing import Any, Iterable, Optional
import logging
import numpy as np
import pandas as pd

# Lazy import: keep a placeholder and import on demand in __init__
SentenceTransformer: Any | None = None

# Avoid tokenizer parallelism deadlocks on some platforms
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

LOGGER = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper around SentenceTransformer for batch embeddings."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        self.batch_size = batch_size
        self.device = device or "auto"
        self.model_name = model_name

        # Dry mode: skip heavy imports entirely
        if os.environ.get("RAG_DRY_EMBED", "0") == "1":
            self.model = None
            LOGGER.info(
                "Initialized TextEmbedder in DRY mode (no HF/torch import), batch_size=%s",
                self.batch_size,
            )
            return

        global SentenceTransformer
        if SentenceTransformer is None:
            # Import on demand to avoid import-time side effects
            from sentence_transformers import SentenceTransformer as _ST  # type: ignore

            SentenceTransformer = _ST
        try:
            import tokenizers as _tok  # type: ignore

            if hasattr(_tok, "set_parallelism"):
                _tok.set_parallelism(False)  # type: ignore[attr-defined]
        except Exception:
            pass

        self.model = SentenceTransformer(model_name, device=device)  # type: ignore[operator]
        # Prefer fast tokenizer (Rust) with parallelism disabled; fallback to slow if unavailable
        try:
            from transformers import AutoTokenizer  # type: ignore

            force_slow = os.environ.get("RAG_FORCE_SLOW_TOKENIZER", "0") == "1"
            tok = None
            if not force_slow:
                try:
                    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
                    LOGGER.info("Using FAST tokenizer for %s", model_name)
                except Exception:
                    tok = None
            if tok is None:
                tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                LOGGER.info("Using SLOW tokenizer for %s", model_name)
            if hasattr(self.model, "tokenizer"):
                self.model.tokenizer = tok
        except Exception:
            pass
        # Optionally reduce PyTorch thread usage to avoid contention
        try:
            import torch  # type: ignore

            torch_threads = int(os.environ.get("TORCH_NUM_THREADS", "1"))
            if torch_threads > 0:
                torch.set_num_threads(torch_threads)
            interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1"))
            if interop_threads > 0 and hasattr(torch, "set_num_interop_threads"):
                torch.set_num_interop_threads(interop_threads)
        except Exception:
            pass
        LOGGER.info(
            "Initialized TextEmbedder model=%s device=%s batch_size=%s dry=%s",
            model_name,
            self.device,
            self.batch_size,
            os.environ.get("RAG_DRY_EMBED", "0"),
        )

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        LOGGER.info("Encoding %d texts (batch_size=%d)", len(texts_list), self.batch_size)
        # Optional dry-run mode for debugging: generate deterministic vectors
        if os.environ.get("RAG_DRY_EMBED", "0") == "1":
            dim = int(os.environ.get("RAG_DRY_EMBED_DIM", "384"))
            rng = np.random.default_rng(42)
            # Create simple deterministic embeddings based on text length with jitter
            base = np.array([i for i in range(dim)], dtype=np.float32)
            vecs = []
            for t in texts_list:
                scale = float(len(t or "")) + 1.0
                noise = rng.normal(0, 0.01, size=dim).astype(np.float32)
                vecs.append(base * scale + noise)
            embeddings = np.vstack(vecs)
        else:
            kwargs = dict(
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            # Call SentenceTransformer.encode without non-standard kwargs.
            # Newer versions do not accept `num_workers`; passing it raises ValueError.
            embeddings = self.model.encode(  # type: ignore[union-attr]
                texts_list,
                **kwargs,
            )
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings /= norms
        return embeddings


class TFIDFEmbedder:
    """Local TF-IDF + TruncatedSVD embedder (no HF/Torch).

    Fits on the corpus used for index building and persists components.
    Later loads the same components to encode queries consistently.
    """

    def __init__(
        self,
        vectorizer: Optional["TfidfVectorizer"] = None,
        svd: Optional["TruncatedSVD"] = None,
        dim: int = 384,
        max_features: int = 50000,
        ngram_max: int = 2,
    ) -> None:
        self.dim = dim
        self.max_features = max_features
        self.ngram_max = ngram_max
        self.vectorizer = vectorizer
        self.svd = svd

    def fit(self, texts: list[str]) -> "TFIDFEmbedder":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        LOGGER.info(
            "Fitting TFIDF (max_features=%d, ngram_max=%d) and SVD (dim=%d) on %d texts",
            self.max_features,
            self.ngram_max,
            self.dim,
            len(texts),
        )
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, self.ngram_max),
            lowercase=True,
        )
        X = self.vectorizer.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=self.dim, random_state=42)
        Z = self.svd.fit_transform(X)
        Z = Z.astype(np.float32)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        Z /= norms
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.vectorizer is not None and self.svd is not None, "TFIDFEmbedder is not fitted"
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        Z = Z.astype(np.float32)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        Z /= norms
        return Z

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return self.transform(list(texts))


def save_tfidf_components(embedder: TFIDFEmbedder, path: Path) -> None:
    from joblib import dump

    path.parent.mkdir(parents=True, exist_ok=True)
    dump({
        "vectorizer": embedder.vectorizer,
        "svd": embedder.svd,
        "dim": embedder.dim,
        "max_features": embedder.max_features,
        "ngram_max": embedder.ngram_max,
    }, path)
    LOGGER.info("Saved TFIDF components to %s", path)


def load_tfidf_components(path: Path) -> TFIDFEmbedder:
    from joblib import load

    obj = load(path)
    return TFIDFEmbedder(
        vectorizer=obj.get("vectorizer"),
        svd=obj.get("svd"),
        dim=int(obj.get("dim", 384)),
        max_features=int(obj.get("max_features", 50000)),
        ngram_max=int(obj.get("ngram_max", 2)),
    )


def embed_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    output_dir: Path | None = None,
    file_prefix: str = "chunks",
    backend: str = "st",
) -> np.ndarray:
    """Embed text column from dataframe and optionally persist artifacts."""
    texts = df[text_column].tolist()
    if backend == "tfidf":
        tfidf = TFIDFEmbedder()
        tfidf.fit(texts)
        embeddings = tfidf.transform(texts)
        if output_dir:
            save_tfidf_components(tfidf, output_dir / "tfidf_svd.joblib")
    else:
        embedder = TextEmbedder(model_name=model_name, device=device, batch_size=batch_size)
        embeddings = embedder.encode(texts)
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"{file_prefix}_embeddings.npy", embeddings)
        df.to_parquet(output_dir / f"{file_prefix}_meta.parquet", index=False)
        LOGGER.info("Saved embeddings to %s", output_dir)
    return embeddings


def save_query_embeddings(
    queries: pd.DataFrame,
    embeddings: np.ndarray,
    output_dir: Path,
    file_prefix: str = "queries",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{file_prefix}_embeddings.npy", embeddings)
    queries.to_parquet(output_dir / f"{file_prefix}_meta.parquet", index=False)
    LOGGER.info("Saved query embeddings to %s", output_dir)


def save_id_mapping(mapping: dict[int, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(mapping, fout, ensure_ascii=False, indent=2)
    LOGGER.info("Saved mapping to %s", path)
