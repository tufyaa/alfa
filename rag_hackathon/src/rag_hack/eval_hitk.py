"""Hit@K evaluation metric."""
from __future__ import annotations

from typing import Dict, Iterable, Sequence, Set


def hit_at_k(preds: Dict[int, Sequence[int]], gold: Dict[int, Set[int]], k: int = 5) -> float:
    if not preds:
        return 0.0
    hits = []
    for q_id, pred_list in preds.items():
        if q_id not in gold or not gold[q_id]:
            continue
        topk = pred_list[:k]
        hits.append(int(bool(set(topk) & gold[q_id])))
    if not hits:
        return 0.0
    return sum(hits) / len(hits)


def load_qrels(path) -> Dict[int, Set[int]]:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"q_id", "web_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in qrels")
    grouped = df.groupby("q_id")["web_id"].apply(lambda x: set(map(int, x)))
    return grouped.to_dict()


def load_predictions(path) -> Dict[int, Sequence[int]]:
    import ast
    import pandas as pd

    df = pd.read_csv(path)
    required = {"q_id", "web_list"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in predictions")
    result = {}
    for row in df.itertuples():
        web_list = ast.literal_eval(row.web_list)
        result[int(row.q_id)] = list(map(int, web_list))
    return result
