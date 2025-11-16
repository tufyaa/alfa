"""High-level orchestration for the RAG pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .chunker import ChunkParams, chunk_documents
from .config import DataPaths, PipelineConfig
from .data import load_websites
from .embedder import TextEmbedder, TFIDFEmbedder, embed_dataframe, save_tfidf_components, load_tfidf_components
from .indexer import FaissIndexer, build_faiss_index
from .preprocess import preprocess_documents
from .retrieve import Retriever

LOGGER = logging.getLogger(__name__)


def build_all(paths: DataPaths, config: PipelineConfig) -> dict:
    LOGGER.info("Loading websites from %s", paths.websites)
    websites_df = load_websites(Path(paths.websites))
    processed_docs = preprocess_documents(websites_df.to_dict("records"))
    processed_df = pd.DataFrame(processed_docs)
    processed_df["web_id"] = processed_df["web_id"].astype(int)
    processed_df = processed_df.drop_duplicates(subset=["web_id"])
    processed_df["doc_text"] = processed_df["doc_text"].fillna("")
    processed_df = processed_df[processed_df["doc_text"].str.len() > 0].reset_index(drop=True)

    chunk_params = ChunkParams(
        config.chunk_chars,
        config.chunk_overlap,
        config.min_chunk_chars,
        config.context_left_chars,
        config.context_right_chars,
    )
    chunks_df = chunk_documents(processed_df.to_dict("records"), chunk_params)
    if chunks_df.empty:
        raise ValueError("No chunks generated. Check preprocessing parameters.")
    chunks_df = (
        chunks_df.drop_duplicates(subset=["web_id", "chunk_text"])
        .reset_index(drop=True)
    )

    LOGGER.info("Embedding %d chunks", len(chunks_df))
    if config.embed_backend == "tfidf":
        # Fit local TFIDF embedder, persist components, and produce embeddings
        tfidf = TFIDFEmbedder()
        tfidf.fit(chunks_df["chunk_text"].tolist())
        embeddings = tfidf.transform(chunks_df["chunk_text"].tolist())
        save_tfidf_components(tfidf, config.artifacts_dir / "tfidf_svd.joblib")
    else:
        embeddings = embed_dataframe(
            chunks_df,
            text_column="chunk_text",
            model_name=config.model_name,
            batch_size=config.batch_size,
            device=config.device,
            output_dir=config.artifacts_dir,
            file_prefix="chunks",
        )

    indexer = build_faiss_index(embeddings, chunks_df)
    index_path = config.artifacts_dir / "chunks.index"
    mapping_path = config.artifacts_dir / "chunks_mapping.json"
    indexer.save(index_path, mapping_path)
    processed_df.to_parquet(config.artifacts_dir / "websites_processed.parquet", index=False)
    chunks_df.to_parquet(config.artifacts_dir / "chunks.parquet", index=False)

    return {
        "websites": processed_df,
        "chunks": chunks_df,
        "embeddings": embeddings,
        "indexer": indexer,
        "index_path": index_path,
        "mapping_path": mapping_path,
    }


def load_index_and_data(artifacts_dir: Path, config: PipelineConfig | None = None):
    config = config or PipelineConfig()
    index_path = artifacts_dir / "chunks.index.npz"
    mapping_path = artifacts_dir / "chunks_mapping.json"
    websites_path = artifacts_dir / "websites_processed.parquet"
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Index artifacts not found")
    indexer = FaissIndexer.load(index_path, mapping_path)
    websites_df = pd.read_parquet(websites_path)
    tfidf_path = artifacts_dir / "tfidf_svd.joblib"
    if tfidf_path.exists():
        embedder = load_tfidf_components(tfidf_path)
    else:
        embedder = TextEmbedder(config.model_name, device=config.device, batch_size=config.batch_size)
    return indexer, websites_df, embedder


def answer_all_questions(
    questions_df: pd.DataFrame,
    retriever: Retriever,
    top_k: int | None = None,
) -> pd.DataFrame:
    if questions_df.empty:
        return pd.DataFrame(columns=["q_id", "web_list"])
    queries = questions_df["query"].astype(str).fillna("").tolist()
    q_ids = questions_df["q_id"].astype(int).tolist()
    target_k = top_k or retriever.config.top_k_return
    retrieved_lists = retriever.retrieve_batch(queries, top_k=target_k)
    records = [{"q_id": q_id, "web_list": web_ids} for q_id, web_ids in zip(q_ids, retrieved_lists)]
    return pd.DataFrame(records)


def dataframe_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    def _format(row: Iterable[int | None]) -> str:
        values: list[int] = []
        for item in row:
            if item is None:
                continue
            val = int(item)
            if val not in values:
                values.append(val)
            if len(values) == 5:
                break
        if len(values) != 5:
            raise ValueError("Each question must have exactly 5 unique web_id predictions.")
        return str(values)

    return pd.DataFrame({"q_id": df["q_id"].astype(int), "web_list": df["web_list"].apply(_format)})
