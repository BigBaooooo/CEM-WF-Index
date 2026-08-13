"""Matched downstream-scoring control for candidate-pool comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import ranked_binary_metrics
from .statistics import paired_bootstrap_interval


def _sealed_score_map(rows: object) -> dict[str, float]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("sealed_scores must be a non-empty list")
    pairs = [(str(row["candidate_id"]), float(row["score"])) for row in rows]
    identifiers = [candidate_id for candidate_id, _ in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identifiers must be unique in sealed_scores")
    if not np.isfinite([score for _, score in pairs]).all():
        raise ValueError("candidate scores must be finite")
    return dict(pairs)


def _rankable_pool(identifiers: object, sealed_scores: dict[str, float]) -> list[tuple[str, float]]:
    if not isinstance(identifiers, list) or not identifiers:
        raise ValueError("each candidate pool must be a non-empty identifier list")
    candidate_ids = [str(candidate_id) for candidate_id in identifiers]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate identifiers must be unique within a pool")
    missing = set(candidate_ids).difference(sealed_scores)
    if missing:
        raise ValueError("every candidate must have a score from the shared sealed scorer")
    return [(candidate_id, sealed_scores[candidate_id]) for candidate_id in candidate_ids]


def evaluate_matched_scoring(
    payload: dict[str, Any],
    *,
    backend: str = "acorn",
    full_system: str = "cem",
    resamples: int = 10_000,
    seed: int = 20262202,
) -> dict[str, Any]:
    """Compare candidate pools while holding their downstream score map fixed.

    This is a candidate-support control.  It does not retrain either backend
    and does not identify the full-pool arm with the complete paper pipeline.
    """

    if payload.get("study_scope") != "development":
        raise ValueError("this evaluator accepts development controls only")
    if not payload.get("scorer_sealed_before_labels", False):
        raise ValueError("the shared scorer must be sealed before labels are read")
    if not str(payload.get("scorer_seal", "")).strip():
        raise ValueError("a non-empty scorer_seal is required")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list")

    backend_ndcg: list[float] = []
    full_ndcg: list[float] = []
    backend_relevant: list[int] = []
    full_relevant: list[int] = []
    seen_queries: set[str] = set()
    for query in queries:
        query_id = str(query["query_id"])
        if query_id in seen_queries:
            raise ValueError("query_id values must be unique")
        seen_queries.add(query_id)
        relevant = {str(item) for item in query["ground_truth_ids"]}
        if len(relevant) != 20:
            raise ValueError("each query must contain 20 distinct relevant events")
        sealed_scores = _sealed_score_map(query.get("sealed_scores"))
        systems = query.get("systems", {})
        backend_rows = _rankable_pool(systems.get(backend), sealed_scores)
        full_rows = _rankable_pool(systems.get(full_system), sealed_scores)
        backend_ids = {candidate_id for candidate_id, _ in backend_rows}
        full_ids = {candidate_id for candidate_id, _ in full_rows}
        if not backend_ids.issubset(full_ids):
            raise ValueError("the backend shortlist must be a subset of the full candidate pool")
        if len(full_rows) != 1_000 or int(payload.get("full_pool_size", 0)) != 1_000:
            raise ValueError("the matched control requires the declared full M=1000 candidate pool")
        if set(sealed_scores) != full_ids:
            raise ValueError("sealed_scores must contain exactly the full candidate pool")

        b_ndcg, b_relevant = ranked_binary_metrics(backend_rows, relevant)
        f_ndcg, f_relevant = ranked_binary_metrics(full_rows, relevant)
        backend_ndcg.append(b_ndcg)
        full_ndcg.append(f_ndcg)
        backend_relevant.append(b_relevant)
        full_relevant.append(f_relevant)

    interval = paired_bootstrap_interval(
        backend_ndcg,
        full_ndcg,
        resamples=resamples,
        seed=seed,
    )
    return {
        "study_scope": "development",
        "metric": "nDCG@10",
        "query_count": len(queries),
        "backend": backend,
        "full_system": full_system,
        "backend_mean": float(np.mean(backend_ndcg)),
        "full_system_mean": float(np.mean(full_ndcg)),
        "backend_relevant_at_20": float(np.mean(backend_relevant)),
        "full_system_relevant_at_20": float(np.mean(full_relevant)),
        "paired_bootstrap": interval.as_dict(),
    }
