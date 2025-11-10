"""CLI to run retrieval for questions."""
from __future__ import annotations

from pathlib import Path

import typer

from rag_hack.config import PipelineConfig
from rag_hack.data import load_questions
from rag_hack.pipeline import answer_all_questions, load_index_and_data
from rag_hack.retrieve import Retriever

app = typer.Typer()


@app.command()
def main(
    questions: Path = typer.Option(Path("data/questions_clean.csv"), help="Questions CSV"),
    websites: Path = typer.Option(Path("data/websites_updated.csv"), help="Websites CSV"),
    index: Path = typer.Option(Path("artifacts"), help="Directory with FAISS index"),
    out: Path = typer.Option(Path("submit/raw_top5.parquet"), help="Output parquet"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
    device: str | None = typer.Option(None, help="Torch device"),
):
    if not websites.exists():
        raise FileNotFoundError(f"Websites file not found: {websites}")
    config = PipelineConfig(batch_size=batch_size, device=device, artifacts_dir=index)
    questions_df = load_questions(questions)
    indexer, websites_df, embedder = load_index_and_data(index, config)
    retriever = Retriever(indexer=indexer, embedder=embedder, websites_df=websites_df, config=config)
    answers_df = answer_all_questions(questions_df, retriever)
    out.parent.mkdir(parents=True, exist_ok=True)
    answers_df.to_parquet(out, index=False)
    typer.echo(f"Saved retrieval results to {out}")


if __name__ == "__main__":
    app()
