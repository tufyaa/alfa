"""CLI to generate submission CSV."""
from __future__ import annotations

from pathlib import Path

import typer

from rag_hack.config import PipelineConfig
from rag_hack.data import load_questions
from rag_hack.pipeline import answer_all_questions, dataframe_to_submission, load_index_and_data
from rag_hack.retrieve import Retriever

app = typer.Typer()


@app.command()
def main(
    questions: Path = typer.Option(Path("data/questions_clean.csv"), help="Questions CSV"),
    index: Path = typer.Option(Path("artifacts"), help="Directory with FAISS index"),
    out: Path = typer.Option(Path("submit/submit.csv"), help="Output CSV"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
    device: str | None = typer.Option(None, help="Torch device"),
):
    config = PipelineConfig(batch_size=batch_size, device=device, artifacts_dir=index)
    questions_df = load_questions(questions)
    indexer, websites_df, embedder = load_index_and_data(index, config)
    retriever = Retriever(indexer=indexer, embedder=embedder, websites_df=websites_df, config=config)
    answers_df = answer_all_questions(questions_df, retriever)
    submission_df = dataframe_to_submission(answers_df)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(out, index=False)
    typer.echo(f"Saved submission to {out}")


if __name__ == "__main__":
    app()
