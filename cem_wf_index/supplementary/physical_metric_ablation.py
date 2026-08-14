"""Evaluate a task-conditioned physical-state metric ablation."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from .statistics import paired_bootstrap_interval


def _parse_time(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc


def evaluate_physical_metric_ablation(
    payload: dict[str, Any],
    *,
    resamples: int = 10_000,
    seed: int = 20262202,
) -> dict[str, Any]:
    """Aggregate query-level ranking and top-1 track-error comparisons."""

    if payload.get("study_scope") != "development":
        raise ValueError("this evaluator accepts development ablations only")
    folds = payload.get("temporal_folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("temporal_folds must be a non-empty list")
    fold_windows: dict[str, tuple[datetime, datetime]] = {}
    for fold in folds:
        name = str(fold["fold"])
        if name in fold_windows:
            raise ValueError("fold names must be unique")
        train_end = _parse_time(fold["train_end"])
        evaluation_start = _parse_time(fold["evaluation_start"])
        evaluation_end = _parse_time(fold["evaluation_end"])
        if not train_end < evaluation_start <= evaluation_end:
            raise ValueError("each temporal fold must place evaluation strictly after training")
        fold_windows[name] = (evaluation_start, evaluation_end)
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list")

    baseline_ndcg: list[float] = []
    cem_ndcg: list[float] = []
    baseline_error: list[float] = []
    cem_error: list[float] = []
    fold_deltas: dict[str, list[float]] = defaultdict(list)
    seen_queries: set[str] = set()
    for row in queries:
        query_id = str(row["query_id"])
        if query_id in seen_queries:
            raise ValueError("query_id values must be unique")
        seen_queries.add(query_id)
        fold_name = str(row["fold"])
        if fold_name not in fold_windows:
            raise ValueError("each query must reference a declared temporal fold")
        timestamp = _parse_time(row["timestamp"])
        fold_start, fold_end = fold_windows[fold_name]
        if not fold_start <= timestamp <= fold_end:
            raise ValueError("query timestamp lies outside its fold evaluation period")
        values = np.asarray(
            [
                row["baseline_ndcg10"],
                row["cem_ndcg10"],
                row["baseline_top1_track_error_km"],
                row["cem_top1_track_error_km"],
            ],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("query-level metrics must be finite and non-negative")
        if values[0] > 1.0 or values[1] > 1.0:
            raise ValueError("nDCG values must not exceed one")
        baseline_ndcg.append(float(values[0]))
        cem_ndcg.append(float(values[1]))
        baseline_error.append(float(values[2]))
        cem_error.append(float(values[3]))
        fold_deltas[fold_name].append(float(values[1] - values[0]))

    expected_folds = int(payload.get("expected_temporal_folds", 3))
    if len(fold_deltas) != expected_folds or len(fold_windows) != expected_folds:
        raise ValueError(f"expected {expected_folds} non-empty temporal folds")
    fold_mean_deltas = {
        fold: float(np.mean(values)) for fold, values in sorted(fold_deltas.items())
    }
    interval = paired_bootstrap_interval(
        baseline_ndcg,
        cem_ndcg,
        resamples=resamples,
        seed=seed,
    )
    # Each input value is already the query/candidate Track RMSE across its
    # common trajectory grid, so the study reports their query mean.
    baseline_rmse = float(np.mean(baseline_error))
    cem_rmse = float(np.mean(cem_error))
    return {
        "study_scope": "development",
        "ranking_metric": "nDCG@10",
        "query_count": len(queries),
        "temporal_fold_count": len(fold_deltas),
        "all_temporal_folds_positive": all(value > 0.0 for value in fold_mean_deltas.values()),
        "fold_deltas": fold_mean_deltas,
        "baseline_mean": float(np.mean(baseline_ndcg)),
        "cem_mean": float(np.mean(cem_ndcg)),
        "paired_bootstrap": interval.as_dict(),
        "baseline_top1_track_rmse_km": baseline_rmse,
        "cem_top1_track_rmse_km": cem_rmse,
        "top1_track_rmse_reduction_km": baseline_rmse - cem_rmse,
    }
