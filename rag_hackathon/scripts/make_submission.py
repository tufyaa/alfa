"""CLI to generate submission CSV."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import typer

CURRENT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(CURRENT_DIR / "src"))

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
    top_k_ann: int = typer.Option(100, help="Number of ANN candidates"),
    bm25_top_k: int = typer.Option(200, help="Number of BM25 candidates"),
    hybrid_alpha: float = typer.Option(0.7, help="Dense/BM25 balance (0..1)"),
    combine_method: str = typer.Option("weighted", help="Fusion: 'weighted' or 'rrf'"),
    rrf_k: int = typer.Option(60, help="RRF k parameter (if combine_method=rrf)"),
    batch_size: int = typer.Option(32, help="Embedding batch size"),
    device: Optional[str] = typer.Option(None, help="Torch device"),
    rerank: bool = typer.Option(False, help="Enable cross-encoder reranking"),
    rerank_model: Optional[str] = typer.Option(None, help="Cross-encoder model name"),
    rerank_candidates: int = typer.Option(100, help="Number of candidates to rerank"),
    rerank_max_chars: int = typer.Option(1200, help="Max chars from doc_text for reranking"),
):
    config = PipelineConfig(
        batch_size=batch_size,
        device=device,
        artifacts_dir=index,
        top_k_ann=top_k_ann,
        bm25_top_k=bm25_top_k,
        hybrid_alpha=hybrid_alpha,
        combine_method=combine_method,
        rrf_k=rrf_k,
        rerank_enable=rerank,
        rerank_model=rerank_model,
        rerank_candidates=rerank_candidates,
        rerank_max_chars=rerank_max_chars,
    )
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
