"""Shared ranking metrics used by the final full-candidate ranker."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _average_precision_at_k(ranked: list[str], relevant: set[str], *, k: int = 10) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, candidate_id in enumerate(ranked[:k], start=1):
        if candidate_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(k, len(relevant))


def _ndcg_at_k(ranked: list[str], relevant: set[str], *, k: int = 10) -> float:
    gains = np.asarray([1.0 if candidate_id in relevant else 0.0 for candidate_id in ranked[:k]], dtype=float)
    if gains.size == 0:
        return 0.0
    ideal_len = min(k, len(relevant))
    if ideal_len == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    ideal = float(np.sum(1.0 / np.log2(np.arange(2, ideal_len + 2))))
    return float(np.sum(gains * discounts) / ideal)


def _metrics(ranked: dict[str, list[str]], relevant_by_query: dict[str, set[str]]) -> dict[str, float]:
    rows: list[tuple[float, float, float, float]] = []
    for query_id, values in ranked.items():
        relevant = relevant_by_query.get(query_id, set())
        if not relevant:
            continue
        top10 = values[:10]
        top20 = values[:20]
        rows.append(
            (
                len(set(top10).intersection(relevant)) / len(relevant),
                len(set(top20).intersection(relevant)) / len(relevant),
                _ndcg_at_k(values, relevant, k=10),
                _average_precision_at_k(values, relevant, k=10),
            )
        )
    arr = np.asarray(rows, dtype=float) if rows else np.zeros((0, 4), dtype=float)
    return {
        "Recall@10": float(np.nanmean(arr[:, 0])) if len(arr) else float("nan"),
        "Recall@20": float(np.nanmean(arr[:, 1])) if len(arr) else float("nan"),
        "nDCG@10": float(np.nanmean(arr[:, 2])) if len(arr) else float("nan"),
        "mAP@10": float(np.nanmean(arr[:, 3])) if len(arr) else float("nan"),
        "query_count": float(len(rows)),
    }


def _ranked_from_frame(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        str(query_id): group.sort_values("rank")["candidate_id"].astype(str).tolist()
        for query_id, group in frame.groupby("query_id", sort=False)
    }
