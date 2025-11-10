"""CLI to build the retrieval index."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rag_hack.config import DataPaths, PipelineConfig
from rag_hack.pipeline import build_all

app = typer.Typer()


@app.command()
def main(
    websites: Path = typer.Option(Path("data/websites_updated.csv"), help="Path to websites CSV"),
    outdir: Path = typer.Option(Path("artifacts"), help="Output directory for artifacts"),
    chunk_chars: int = typer.Option(800, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(120, help="Chunk overlap in characters"),
    chunk_min_chars: int = typer.Option(80, help="Discard chunks shorter than this length"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
    device: Optional[str] = typer.Option(None, help="Torch device for embeddings"),
):
    paths = DataPaths(websites=websites)
    config = PipelineConfig(
        chunk_chars=chunk_chars,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=chunk_min_chars,
        batch_size=batch_size,
        device=device,
        artifacts_dir=outdir,
    )
    build_all(paths, config)
    typer.echo(f"Artifacts saved to {outdir}")


if __name__ == "__main__":
    app()
