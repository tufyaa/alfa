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
    # Optional context windows (in characters) to enrich each base chunk
    # by adding a small slice from the left/right neighboring regions.
    # Overlapped parts already present due to chunk_overlap are not duplicated.
    context_left_chars: int = 0
    context_right_chars: int = 0


def _chunk_text(text: str, chunk_chars: int, chunk_overlap: int) -> list[tuple[int, int]]:
    """Return list of (start, end) spans for base chunks.

    Using spans preserves exact boundaries, enabling precise context addition
    without duplicating overlap regions.
    """
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(text)
    step = max(1, chunk_chars - chunk_overlap)
    while start < length:
        end = min(length, start + chunk_chars)
        if end > start:
            spans.append((start, end))
        start += step
    return spans


def chunk_documents(docs: Iterable[dict], params: ChunkParams) -> pd.DataFrame:
    records: list[dict] = []
    for doc in docs:
        web_id = doc["web_id"]
        text = doc.get("doc_text", "")
        spans = _chunk_text(text, params.chunk_chars, params.chunk_overlap)

        last_base_text = ""
        for order, (start, end) in enumerate(spans):
            base_text = text[start:end]
            base_text_stripped = base_text.strip()
            if params.min_chunk_chars and len(base_text_stripped) < params.min_chunk_chars:
                continue
            if base_text_stripped == last_base_text:
                continue

            # Compute context-enriched chunk
            left_extra = 0
            if params.context_left_chars > 0 and order > 0:
                prev_start, prev_end = spans[order - 1]
                overlap_prev = max(0, prev_end - start)
                # Add only the part not already included via overlap
                left_extra = max(0, params.context_left_chars - overlap_prev)
                left_extra = min(left_extra, start)  # don't go before 0
            right_extra = 0
            if params.context_right_chars > 0 and order < len(spans) - 1:
                next_start, next_end = spans[order + 1]
                overlap_next = max(0, end - next_start)
                # Add only the part not already included via overlap
                right_extra = max(0, params.context_right_chars - overlap_next)
                right_extra = min(right_extra, len(text) - end)

            enriched_start = start - left_extra
            enriched_end = end + right_extra
            enriched_text = text[enriched_start:enriched_end].strip()

            records.append(
                {
                    "web_id": web_id,
                    "chunk_id": f"{web_id}_{order}",
                    "chunk_text": enriched_text,
                    "chunk_order": order,
                    "n_chars": len(enriched_text),
                }
            )
            last_base_text = base_text_stripped
    if not records:
        return pd.DataFrame(columns=["web_id", "chunk_id", "chunk_text", "chunk_order", "n_chars"])
    df = pd.DataFrame.from_records(records)
    return df
