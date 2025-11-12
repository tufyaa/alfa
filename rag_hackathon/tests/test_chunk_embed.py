from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rag_hack.chunker import ChunkParams, chunk_documents
from rag_hack.embedder import embed_dataframe
from rag_hack.preprocess import preprocess_documents


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
        arr = np.arange(len(texts) * 3, dtype=np.float32).reshape(len(texts), 3)
        return arr


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_hack import embedder

    monkeypatch.setattr(embedder, "SentenceTransformer", DummyModel)


def test_chunk_and_embed(tmp_path: Path) -> None:
    import pandas as pd

    docs = pd.DataFrame(
        {
            "web_id": [1, 2],
            "title": ["Что такое RAG?", "Настройка индекса"],
            "text": ["<p>Вопрос-ответ о retrieval.</p>", "<div>Загрузка и поиск.</div>"],
        }
    )
    processed = preprocess_documents(docs.to_dict("records"))
    chunks = chunk_documents(processed, ChunkParams(chunk_chars=20, chunk_overlap=5))
    assert not chunks.empty
    embeddings = embed_dataframe(chunks, "chunk_text", model_name="dummy")
    assert embeddings.shape[0] == len(chunks)
    assert embeddings.shape[1] == 3
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)
