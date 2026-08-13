"""Evaluate fold-local ClimateNet atmospheric-river portability scores."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from .metrics import continuous_ndcg_at_k
from .statistics import paired_bootstrap_interval


def _require_development_boundary(payload: dict[str, Any]) -> None:
    if payload.get("study_scope") != "development":
        raise ValueError("this evaluator accepts development studies only")
    if payload.get("formal_test_opened", True):
        raise ValueError("formal_test_opened must be false")
    if not payload.get("scores_sealed_before_relevance", False):
        raise ValueError("scores must be sealed before evaluation relevance is read")
    if not payload.get("input_and_evaluation_annotations_disjoint", False):
        raise ValueError("input and evaluation annotations must be disjoint")


def _parse_time(value: object) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO-8601 timestamp: {value}") from exc


def _temporal_fold_contract(payload: dict[str, Any]) -> dict[str, tuple[datetime, datetime]]:
    rows = payload.get("temporal_folds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("temporal_folds must be a non-empty list")
    folds: dict[str, tuple[datetime, datetime]] = {}
    for row in rows:
        name = str(row["fold"])
        if name in folds:
            raise ValueError("fold names must be unique")
        train_end = _parse_time(row["train_end"])
        evaluation_start = _parse_time(row["evaluation_start"])
        evaluation_end = _parse_time(row["evaluation_end"])
        if not train_end < evaluation_start <= evaluation_end:
            raise ValueError("each temporal fold must place evaluation strictly after training")
        folds[name] = (evaluation_start, evaluation_end)
    return folds


def evaluate_climatenet_portability(
    payload: dict[str, Any],
    *,
    resamples: int = 10_000,
    seed: int = 308000,
) -> dict[str, Any]:
    """Aggregate a temporal-fold ClimateNet development comparison.

    Each query supplies two already-sealed score vectors and an independently
    supplied continuous-relevance vector.  Evaluation relevance is never used
    to construct either score vector in this function.
    """

    _require_development_boundary(payload)
    temporal_folds = _temporal_fold_contract(payload)
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list")

    control_values: list[float] = []
    cem_values: list[float] = []
    fold_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seen_queries: set[str] = set()
    for row in queries:
        query_id = str(row["query_id"])
        if query_id in seen_queries:
            raise ValueError("query_id values must be unique")
        seen_queries.add(query_id)
        fold = str(row["fold"])
        if fold not in temporal_folds:
            raise ValueError("each query must reference a declared temporal fold")
        timestamp = _parse_time(row["timestamp"])
        start, end = temporal_folds[fold]
        if not start <= timestamp <= end:
            raise ValueError("query timestamp lies outside its fold evaluation period")
        seed_rows = row.get("seed_scores")
        expected_seeds = int(payload.get("model_seed_count", 5))
        if not isinstance(seed_rows, list) or len(seed_rows) != expected_seeds:
            raise ValueError(f"each query must provide {expected_seeds} sealed seed-score pairs")
        seed_names = [str(seed_row["seed"]) for seed_row in seed_rows]
        if len(set(seed_names)) != len(seed_names):
            raise ValueError("seed values must be unique within each query")
        control_per_seed = [
            continuous_ndcg_at_k(seed_row["control_scores"], row["continuous_relevance"], k=10)
            for seed_row in seed_rows
        ]
        cem_per_seed = [
            continuous_ndcg_at_k(seed_row["cem_scores"], row["continuous_relevance"], k=10)
            for seed_row in seed_rows
        ]
        control = float(np.mean(control_per_seed))
        cem = float(np.mean(cem_per_seed))
        control_values.append(control)
        cem_values.append(cem)
        fold_pairs[fold].append((control, cem))

    expected_folds = int(payload.get("expected_temporal_folds", 3))
    if len(fold_pairs) != expected_folds or len(temporal_folds) != expected_folds:
        raise ValueError(f"expected {expected_folds} non-empty temporal folds")
    fold_deltas = {
        fold: float(np.mean([cem - control for control, cem in pairs]))
        for fold, pairs in sorted(fold_pairs.items())
    }
    interval = paired_bootstrap_interval(
        control_values,
        cem_values,
        resamples=resamples,
        seed=seed,
    )
    return {
        "study_scope": "development",
        "metric": "continuous_nDCG@10",
        "object_count": int(payload["object_count"]),
        "model_seed_count": int(payload.get("model_seed_count", 5)),
        "query_count": len(queries),
        "temporal_fold_count": len(fold_pairs),
        "all_temporal_folds_positive": all(value > 0.0 for value in fold_deltas.values()),
        "fold_deltas": fold_deltas,
        "control_mean": float(np.mean(control_values)),
        "cem_mean": float(np.mean(cem_values)),
        "paired_bootstrap": interval.as_dict(),
    }
