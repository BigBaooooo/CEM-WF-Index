"""Recompute the compact reviewer-facing evidence tables from anonymous rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistics import paired_bootstrap_interval


def _metadata(directory: Path, name: str) -> dict[str, Any]:
    payload = json.loads((directory / f"{name}_metadata.json").read_text(encoding="utf-8"))
    if int(payload["bootstrap_resamples"]) < 100:
        raise ValueError("bootstrap_resamples must be at least 100")
    return payload


def _study_result(
    baseline: np.ndarray,
    proposed: np.ndarray,
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    interval = paired_bootstrap_interval(
        baseline,
        proposed,
        resamples=int(metadata["bootstrap_resamples"]),
        seed=int(metadata["bootstrap_seed"]),
    )
    return {
        "control_mean": float(np.mean(baseline)),
        "cem_mean": float(np.mean(proposed)),
        "gain": interval.estimate,
        "ci95": [interval.lower, interval.upper],
        "query_count": interval.query_count,
        "bootstrap_seed": interval.seed,
        "bootstrap_resamples": interval.resamples,
    }


def verify_published_evidence(directory: str | Path) -> dict[str, Any]:
    """Recompute all public supplementary summaries from anonymous query rows."""

    root = Path(directory)
    climate_meta = _metadata(root, "climatenet_portability")
    climate = pd.read_csv(root / "climatenet_portability_per_query.csv")
    if len(climate) != int(climate_meta["query_count"]):
        raise ValueError("ClimateNet query count does not match its metadata")
    climate_result = _study_result(
        climate["control_continuous_ndcg10"].to_numpy(float),
        climate["cem_continuous_ndcg10"].to_numpy(float),
        metadata=climate_meta,
    )
    fold_delta = climate.assign(
        delta=climate["cem_continuous_ndcg10"] - climate["control_continuous_ndcg10"]
    ).groupby("fold", sort=True).delta.mean()
    climate_result.update(
        {
            "object_count": int(climate_meta["object_count"]),
            "temporal_fold_count": int(fold_delta.size),
            "all_temporal_folds_positive": bool((fold_delta > 0).all()),
            "fold_deltas": {key: float(value) for key, value in fold_delta.items()},
            "model_seed_count": int(climate_meta["model_seed_count"]),
        }
    )
    seed_summary = json.loads(
        (root / "climatenet_seed_summary.json").read_text(encoding="utf-8")
    )
    seed_values = np.asarray(list(seed_summary["seed_values"].values()), dtype=float)
    if seed_values.size != int(climate_meta["model_seed_count"]):
        raise ValueError("ClimateNet seed count does not match its metadata")
    climate_result.update(
        {
            "seed_mean": float(np.mean(seed_values)),
            "five_seed_sd": float(np.std(seed_values)),
        }
    )

    matched_meta = _metadata(root, "matched_scoring")
    matched = pd.read_csv(root / "matched_scoring_per_query.csv")
    if len(matched) != int(matched_meta["query_count"]):
        raise ValueError("Matched-scoring query count does not match its metadata")
    matched_result = _study_result(
        matched["acorn_ndcg10"].to_numpy(float),
        matched["cem_pool_ndcg10"].to_numpy(float),
        metadata=matched_meta,
    )
    matched_result.update(
        {
            "acorn_relevant_at20": float(matched["acorn_relevant_at20"].mean()),
            "cem_pool_relevant_at20": float(matched["cem_pool_relevant_at20"].mean()),
            "full_pool_size": int(matched_meta["full_pool_size"]),
        }
    )

    physical_meta = _metadata(root, "physical_metric")
    physical = pd.read_csv(root / "physical_metric_per_query.csv")
    if len(physical) != int(physical_meta["query_count"]):
        raise ValueError("Physical-metric query count does not match its metadata")
    physical_result = _study_result(
        physical["control_ndcg10"].to_numpy(float),
        physical["physical_metric_ndcg10"].to_numpy(float),
        metadata=physical_meta,
    )
    physical_fold_delta = physical.assign(
        delta=physical["physical_metric_ndcg10"] - physical["control_ndcg10"]
    ).groupby("fold", sort=True).delta.mean()
    physical_result.update(
        {
            "temporal_fold_count": int(physical_fold_delta.size),
            "all_temporal_folds_positive": bool((physical_fold_delta > 0).all()),
            "track_rmse_reduction_km": float(
                physical["control_track_rmse_km"].mean()
                - physical["physical_metric_track_rmse_km"].mean()
            ),
        }
    )

    diversity = pd.read_csv(root / "candidate_diversity_summary.csv")
    audit = json.loads((root / "audit_export_summary.json").read_text(encoding="utf-8"))
    return {
        "climatenet_portability": climate_result,
        "matched_scoring": matched_result,
        "physical_metric_ablation": physical_result,
        "candidate_diversity": diversity.to_dict("records"),
        "audit_export": audit,
    }
