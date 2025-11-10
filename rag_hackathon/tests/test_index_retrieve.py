from typing import Any

import numpy as np
import pandas as pd
import pytest

from rag_hack import embedder
from rag_hack.chunker import ChunkParams, chunk_documents
from rag_hack.config import PipelineConfig
from rag_hack.embedder import embed_dataframe
from rag_hack.indexer import build_faiss_index
from rag_hack.pipeline import answer_all_questions
from rag_hack.preprocess import preprocess_documents
from rag_hack.retrieve import Retriever


class DummyModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def encode(
        self,
        texts,
        batch_size: int = 32,
        show_progress_bar: bool = True,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ):
        vecs = []
        for text in texts:
            base = float(len(text))
            vecs.append(np.array([base, base + 1, base + 2], dtype=np.float32))
        return np.vstack(vecs)


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder, "SentenceTransformer", DummyModel)


def test_retrieve_returns_five_ids() -> None:
    websites = pd.DataFrame(
        {
            "web_id": range(1, 7),
            "title": [f"Ответ про RAG {i}" for i in range(1, 7)],
            "text": [f"Описание шага {i}. Подробности о пайплайне." for i in range(1, 7)],
        }
    )
    processed = preprocess_documents(websites.to_dict("records"))
    chunks = chunk_documents(processed, ChunkParams(chunk_chars=30, chunk_overlap=5))
    embeddings = embed_dataframe(chunks, "chunk_text", model_name="dummy")
    indexer = build_faiss_index(embeddings, chunks)

    config = PipelineConfig(top_k_ann=10, top_k_return=5, bm25_top_k=20)
    retriever = Retriever(
        indexer=indexer,
        embedder=embedder.TextEmbedder("dummy"),
        websites_df=pd.DataFrame(processed),
        config=config,
    )

    questions = pd.DataFrame(
        {
            "q_id": [1, 2, 3],
            "query": ["Что такое rag?", "Как построить индекс?", "Как получить топ сайты?"],
        }
    )
    answers = answer_all_questions(questions, retriever)

    assert len(answers) == 3
    for row in answers.itertuples():
        assert len(row.web_list) == 5
        assert len(set(row.web_list)) == 5
        for web_id in row.web_list:
            assert web_id in websites["web_id"].values
