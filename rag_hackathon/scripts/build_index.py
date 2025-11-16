"""CLI to build the retrieval index."""
from __future__ import annotations

from pathlib import Path

import sys
from typing import Optional

import typer

import os
import logging

# Reduce chances of native lib lock contention on some platforms
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR / "src"))

app = typer.Typer()


@app.command()
def main(
    websites: Path = typer.Option(Path("data/websites_updated.csv"), help="Path to websites CSV"),
    outdir: Path = typer.Option(Path("artifacts"), help="Output directory for artifacts"),
    chunk_chars: int = typer.Option(800, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(120, help="Chunk overlap in characters"),
    chunk_min_chars: int = typer.Option(80, help="Discard chunks shorter than this length"),    
    context_left: int = typer.Option(0, help="Left context window size (chars) for context-enriched chunking"),
    context_right: int = typer.Option(0, help="Right context window size (chars) for context-enriched chunking"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
    device: Optional[str] = typer.Option(None, help="Torch device for embeddings (e.g. 'cpu')"),
    model_name: Optional[str] = typer.Option(None, help="SentenceTransformer model name (for embedder=st)"),
    disable_faiss: bool = typer.Option(False, help="Disable FAISS and use NumPy fallback"),
    dry_embed: bool = typer.Option(False, help="Generate fake embeddings for debugging"),
    threads: int = typer.Option(1, help="Limit threads for BLAS/FAISS/tokenizers/torch"),
    slow_tokenizer: bool = typer.Option(False, help="Force slow (pure-Python) tokenizer"),
    embedder: str = typer.Option("st", help="Embedding backend: 'st' (SentenceTransformers) or 'tfidf'"),
):
    typer.echo("Starting build: loading data and preparing pipeline…")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("rag_hack").setLevel(logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    # Defer heavy imports until after environment variables are set and CLI parsed
    from rag_hack.config import DataPaths, PipelineConfig
    from rag_hack.pipeline import build_all
    typer.echo("Imports OK. Building artifacts…")

    # Apply runtime toggles from CLI
    if disable_faiss:
        os.environ["RAG_DISABLE_FAISS"] = "1"
    if dry_embed:
        os.environ["RAG_DRY_EMBED"] = "1"
    if slow_tokenizer:
        os.environ["RAG_FORCE_SLOW_TOKENIZER"] = "1"
    # Thread limits
    if threads and threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
        os.environ["FAISS_NUM_THREADS"] = str(threads)
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        os.environ["TORCH_NUM_THREADS"] = str(threads)

    paths = DataPaths(websites=websites)
    config = PipelineConfig(
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=chunk_min_chars,
        context_left_chars=context_left,
        context_right_chars=context_right,
        batch_size=batch_size,
        device=device,
        artifacts_dir=outdir,
        model_name=model_name or PipelineConfig().model_name,
        embed_backend=embedder,
    )
    build_all(paths, config)
    typer.echo(f"Artifacts saved to {outdir}")


if __name__ == "__main__":
    app()
