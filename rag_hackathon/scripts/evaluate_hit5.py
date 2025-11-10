"""CLI to evaluate Hit@5."""
from __future__ import annotations

from pathlib import Path

import typer

from rag_hack.eval_hitk import hit_at_k, load_predictions, load_qrels

app = typer.Typer()


@app.command()
def main(
    pred: Path = typer.Option(Path("submit/submit.csv"), help="Predictions CSV"),
    qrels: Path = typer.Option(Path("data/qrels.csv"), help="Ground truth"),
    k: int = typer.Option(5, help="Cutoff for Hit@K"),
):
    preds = load_predictions(pred)
    gold = load_qrels(qrels)
    score = hit_at_k(preds, gold, k=k)
    typer.echo(f"Hit@{k}: {score:.4f}")


if __name__ == "__main__":
    app()
