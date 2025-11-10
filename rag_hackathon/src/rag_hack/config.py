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
    top_k_ann: int = 100
    top_k_return: int = 5
    bm25_top_k: int = 200
    hybrid_alpha: float = 0.7
    model_name: str = "ai-forever/ru-en-RoSBERTa"
    batch_size: int = 32
    device: Optional[str] = None
    artifacts_dir: Path = field(default_factory=lambda: Path("artifacts"))


DEFAULT_CONFIG = PipelineConfig()
DEFAULT_PATHS = DataPaths()
