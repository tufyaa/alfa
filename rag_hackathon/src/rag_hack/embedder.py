"""Embedding utilities using SentenceTransformer."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)


class TextEmbedder:
    """Wrapper around SentenceTransformer for batch embeddings."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 32) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings /= norms
        return embeddings


def embed_dataframe(
    df: pd.DataFrame,
    text_column: str,
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
    output_dir: Path | None = None,
    file_prefix: str = "chunks",
) -> np.ndarray:
    """Embed text column from dataframe and optionally persist artifacts."""
    embedder = TextEmbedder(model_name=model_name, device=device, batch_size=batch_size)
    texts = df[text_column].tolist()
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
