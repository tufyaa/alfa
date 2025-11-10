"""Data loading helpers with schema validation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_QUESTION_COLUMNS = {"q_id", "query"}
REQUIRED_WEBSITE_COLUMNS = {"web_id", "url", "kind", "title", "text"}


def _validate_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> pd.DataFrame:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


def load_questions(path: Path) -> pd.DataFrame:
    """Load questions CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    df = _validate_columns(df, REQUIRED_QUESTION_COLUMNS, path)
    if df.empty:
        raise ValueError(f"Questions file {path} is empty")
    df = df.loc[:, sorted(REQUIRED_QUESTION_COLUMNS)].astype({"q_id": int, "query": str})
    return df


def load_websites(path: Path) -> pd.DataFrame:
    """Load websites CSV and ensure required columns exist."""
    df = pd.read_csv(path)
    df = _validate_columns(df, REQUIRED_WEBSITE_COLUMNS, path)
    if df.empty:
        raise ValueError(f"Websites file {path} is empty")
    df = df.loc[:, ["web_id", "url", "kind", "title", "text"]]
    df["web_id"] = df["web_id"].astype(int)
    for col in ("url", "kind", "title", "text"):
        df[col] = df[col].fillna("").astype(str)
    return df
