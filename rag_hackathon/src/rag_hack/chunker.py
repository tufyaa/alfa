"""Document chunking utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(slots=True)
class ChunkParams:
    chunk_chars: int
    chunk_overlap: int
    min_chunk_chars: int = 0


def _chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    length = len(text)
    step = max(1, chunk_chars - chunk_overlap)
    while start < length:
        end = min(length, start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_documents(docs: Iterable[dict], params: ChunkParams) -> pd.DataFrame:
    records: list[dict] = []
    for doc in docs:
        web_id = doc["web_id"]
        text = doc.get("doc_text", "")
        chunks = _chunk_text(text, params.chunk_chars, params.chunk_overlap)
        last_chunk = ""
        for order, chunk in enumerate(chunks):
            if params.min_chunk_chars and len(chunk) < params.min_chunk_chars:
                continue
            if chunk == last_chunk:
                continue
            records.append(
                {
                    "web_id": web_id,
                    "chunk_id": f"{web_id}_{order}",
                    "chunk_text": chunk,
                    "chunk_order": order,
                    "n_chars": len(chunk),
                }
            )
            last_chunk = chunk
    if not records:
        return pd.DataFrame(columns=["web_id", "chunk_id", "chunk_text", "chunk_order", "n_chars"])
    df = pd.DataFrame.from_records(records)
    return df
