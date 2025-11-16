"""Configuration utilities for the RAG pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class DataPaths:
    """Container for dataset locations."""

    questions: Path = Path("data/questions_clean.csv")
    websites: Path = Path("data/websites_updated.csv")
    sample_submission: Path = Path("data/sample_submission.csv")
    qrels: Optional[Path] = None


@dataclass(slots=True)
class PipelineConfig:
    """Hyper-parameters for the pipeline."""

    chunk_chars: int = 800
    chunk_overlap: int = 120
    min_chunk_chars: int = 80
    # Context-enriched chunking windows (in characters). When both are zero,
    # behavior is identical to the previous implementation.
    context_left_chars: int = 0
    context_right_chars: int = 0
    top_k_ann: int = 100
    top_k_return: int = 5
    bm25_top_k: int = 200
    hybrid_alpha: float = 0.7
    # Hybrid fusion control: 'weighted' (alpha*ANN + (1-alpha)*BM25) or 'rrf'
    combine_method: str = "weighted"
    rrf_k: int = 60
    model_name: str = "ai-forever/ru-en-RoSBERTa"
    embed_backend: str = "st"  # "st" (SentenceTransformers) or "tfidf"
    batch_size: int = 32
    device: Optional[str] = None
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))
    # Second-stage reranking with a cross-encoder
    rerank_enable: bool = False
    rerank_model: Optional[str] = None  # e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rerank_candidates: int = 100  # how many hybrid candidates to rerank
    rerank_max_chars: int = 1200   # truncate doc text for reranking pairs


DEFAULT_CONFIG = PipelineConfig()
DEFAULT_PATHS = DataPaths()
