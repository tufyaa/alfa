"""High-level orchestration for the RAG pipeline."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .chunker import ChunkParams, chunk_documents
from .config import DataPaths, PipelineConfig
from .data import load_websites
from .embedder import TextEmbedder, embed_dataframe
from .indexer import FaissIndexer, build_faiss_index
from .preprocess import preprocess_documents
from .retrieve import Retriever

LOGGER = logging.getLogger(__name__)


def build_all(paths: DataPaths, config: PipelineConfig) -> dict:
    LOGGER.info("Loading websites from %s", paths.websites)
    websites_df = load_websites(Path(paths.websites))
    processed_docs = preprocess_documents(websites_df.to_dict("records"))
    processed_df = pd.DataFrame(processed_docs)

    chunk_params = ChunkParams(config.chunk_chars, config.chunk_overlap)
    chunks_df = chunk_documents(processed_docs, chunk_params)
    chunks_df = chunks_df.drop_duplicates(subset=["chunk_id"]).reset_index(drop=True)
    chunks_df = chunks_df[chunks_df["chunk_text"].str.len() > 0]

    LOGGER.info("Embedding %d chunks", len(chunks_df))
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


def load_index_and_data(artifacts_dir: Path, config: PipelineConfig | None = None) -> tuple[FaissIndexer, pd.DataFrame, TextEmbedder]:
    config = config or PipelineConfig()
    index_path = artifacts_dir / "chunks.index"
    mapping_path = artifacts_dir / "chunks_mapping.json"
    websites_path = artifacts_dir / "websites_processed.parquet"
    if not index_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Index artifacts not found")
    indexer = FaissIndexer.load(index_path, mapping_path)
    websites_df = pd.read_parquet(websites_path)
    embedder = TextEmbedder(config.model_name, device=config.device, batch_size=config.batch_size)
    return indexer, websites_df, embedder


def answer_all_questions(
    questions_df: pd.DataFrame,
    retriever: Retriever,
    top_k: int | None = None,
) -> pd.DataFrame:
    results = []
    all_web_ids = retriever.websites_df["web_id"].tolist() if hasattr(retriever, "websites_df") else []
    for row in questions_df.itertuples():
        web_ids = retriever.retrieve_topk_for_query(row.query)
        if top_k is not None:
            web_ids = web_ids[:top_k]
        if len(web_ids) < retriever.config.top_k_return:
            for fallback in all_web_ids:
                if fallback not in web_ids:
                    web_ids.append(fallback)
                if len(web_ids) >= retriever.config.top_k_return:
                    break
        results.append({"q_id": int(row.q_id), "web_list": web_ids[:retriever.config.top_k_return]})
    df = pd.DataFrame(results)
    return df


def dataframe_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    def format_list(row: Iterable[int | None]) -> str:
        values: list[int] = []
        for item in row:
            if item is None:
                continue
            val = int(item)
            if val not in values:
                values.append(val)
        candidate = 1
        while len(values) < 5:
            while candidate in values:
                candidate += 1
            values.append(candidate)
            candidate += 1
        return str(values[:5])

    return pd.DataFrame({"q_id": df["q_id"], "web_list": df["web_list"].apply(format_list)})
