from pathlib import Path

import pandas as pd

from rag_hack.data import load_questions, load_websites


def test_load_questions_and_websites(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions_clean.csv"
    websites_path = tmp_path / "websites_updated.csv"

    pd.DataFrame(
        {
            "q_id": [1, 2],
            "query": ["Как запустить rag пайплайн?", "Где найти топ-5 сайтов?"],
        }
    ).to_csv(questions_path, index=False)
    pd.DataFrame(
        {
            "web_id": [10, 20],
            "url": ["https://example.com/1", "https://example.com/2"],
            "kind": ["article", "blog"],
            "title": ["RAG overview", "Building index"],
            "text": ["<p>Some text</p>", "More text"],
        }
    ).to_csv(websites_path, index=False)

    questions = load_questions(questions_path)
    websites = load_websites(websites_path)

    assert set(questions.columns) == {"q_id", "query"}
    assert len(questions) == 2
    assert set(websites.columns) == {"web_id", "url", "kind", "title", "text"}
    assert len(websites) == 2
