"""Frozen feature contract and executable inference-input audit."""

from __future__ import annotations

import hashlib
import json
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_LIST_PATH = Path(__file__).with_name("typhoon_frozen_feature_names.csv")
CONTEXT_RANK_FEATURE = "score_context_rank_prior"

_ALLOWED_NUMERIC_NON_FEATURES = {
    "rank",
    "pool_rank",
    "context_rank",
    "source_context_rank",
    "source_metadata_rank",
    "distance",
    "vector_row",
    "event_duration_hours",
    "context_candidate",
    "context_rank_prior_available",
    "context_used_for_candidate_generation",
    "standalone_direct_context_term_selected",
    "is_gt_top20",
    "grade",
    "uses_cma_gt_labels_for_training",
    "uses_cma_numeric_at_inference",
    "uses_test_for_tuning",
    "uses_cma_for_gt_only",
}
_FORBIDDEN_EXACT = {
    "is_gt_top20",
    "grade",
    "grade_copy",
    "gt_distance",
    "ground_truth_distance",
    "cma_pressure",
    "cma_wind",
    "cma_landfall",
    "test_ndcg",
    "test_map",
    "validation_ndcg",
    "validation_map",
}
_FORBIDDEN_PATTERNS = (
    re.compile(r"(^|_)(is_)?gt(_|$)"),
    re.compile(r"ground_?truth"),
    re.compile(r"(^|_)grade(_|$)"),
    re.compile(r"(^|_)cma_(pressure|wind|track|landfall|latitude|longitude|numeric)"),
    re.compile(r"(^|_)(validation|val|test)_(ndcg|map|recall|metric|score|label)"),
)


def load_frozen_feature_names(path: str | Path = FEATURE_LIST_PATH) -> list[str]:
    """Load the public ordered feature contract."""

    frame = pd.read_csv(Path(path))
    if list(frame.columns) != ["feature"]:
        raise ValueError("Frozen feature file must contain exactly one 'feature' column")
    names = frame["feature"].astype(str).tolist()
    if not names or len(names) != len(set(names)):
        raise ValueError("Frozen feature names must be non-empty and unique")
    return names


FROZEN_FEATURE_NAMES = tuple(load_frozen_feature_names())


def feature_list_sha256(feature_names: Sequence[str] = FROZEN_FEATURE_NAMES) -> str:
    """Hash an ordered list using a platform-independent representation."""

    payload = json.dumps(list(feature_names), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_sha256(value: Any) -> str:
    """Hash mappings/sequences deterministically for a compact asset receipt."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def model_sha256(model: Any) -> str:
    """Hash the stable LightGBM model text without writing a model artifact."""

    booster = getattr(model, "booster_", None)
    if booster is None or not hasattr(booster, "model_to_string"):
        raise TypeError("A fitted LightGBM model with booster_.model_to_string() is required")
    return hashlib.sha256(booster.model_to_string().encode("utf-8")).hexdigest()


def forbidden_feature_names(feature_names: Sequence[str]) -> list[str]:
    """Return requested model-input names that match evaluation-only semantics."""

    hits: list[str] = []
    for raw_name in feature_names:
        name = str(raw_name).strip().lower()
        if name in _FORBIDDEN_EXACT or any(pattern.search(name) for pattern in _FORBIDDEN_PATTERNS):
            hits.append(str(raw_name))
    return hits


def unexpected_numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Identify numeric table fields outside the frozen inputs and declared metadata."""

    allowed = set(FROZEN_FEATURE_NAMES) | _ALLOWED_NUMERIC_NON_FEATURES
    return [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column]) and str(column) not in allowed
    ]


def validate_feature_contract(
    frame: pd.DataFrame,
    requested_feature_names: Sequence[str] | None = None,
    *,
    warn_unexpected_numeric: bool = True,
) -> dict[str, Any]:
    """Validate the exact ordered model matrix before training or inference."""

    requested = list(FROZEN_FEATURE_NAMES if requested_feature_names is None else requested_feature_names)
    forbidden = forbidden_feature_names(requested)
    if forbidden:
        raise ValueError(f"Evaluation-only fields requested as model inputs: {', '.join(forbidden)}")
    if requested != list(FROZEN_FEATURE_NAMES):
        missing_from_request = [name for name in FROZEN_FEATURE_NAMES if name not in requested]
        unregistered = [name for name in requested if name not in FROZEN_FEATURE_NAMES]
        raise ValueError(
            "Requested features do not match the frozen ordered contract; "
            f"missing={missing_from_request}, unregistered={unregistered}"
        )
    missing = [name for name in FROZEN_FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Candidate rows are missing frozen model features: {', '.join(missing)}")
    non_numeric = [name for name in FROZEN_FEATURE_NAMES if not pd.api.types.is_numeric_dtype(frame[name])]
    if non_numeric:
        raise TypeError(f"Frozen model features must be numeric: {', '.join(non_numeric)}")
    unexpected = unexpected_numeric_columns(frame)
    suspicious = forbidden_feature_names(unexpected)
    forbidden_table_fields = forbidden_feature_names([str(column) for column in frame.columns])
    if warn_unexpected_numeric and unexpected:
        warnings.warn(
            "Numeric columns outside the frozen model contract were ignored: " + ", ".join(unexpected),
            UserWarning,
            stacklevel=2,
        )
    return {
        "actual_feature_columns": requested,
        "feature_list_sha256": feature_list_sha256(requested),
        "forbidden_feature_hits": forbidden,
        "forbidden_table_fields_excluded_from_model": forbidden_table_fields,
        "unexpected_numeric_columns": unexpected,
        "suspicious_non_input_columns": suspicious,
        "context_rank_prior_consumed_by_lambdarank": CONTEXT_RANK_FEATURE in requested,
        "passed": True,
    }


def graph_asset_sha256(
    seed_to_train_queries: Mapping[str, Sequence[tuple[str, float]]],
    train_query_to_positive_candidates: Mapping[str, Mapping[str, float]],
) -> str:
    """Hash both train-only graph mappings in canonical key order."""

    graph = {
        "seed_to_train_queries": {
            str(seed): sorted((str(query), float(weight)) for query, weight in values)
            for seed, values in sorted(seed_to_train_queries.items(), key=lambda item: str(item[0]))
        },
        "train_query_to_positive_candidates": {
            str(query): {str(candidate): float(weight) for candidate, weight in sorted(values.items())}
            for query, values in sorted(train_query_to_positive_candidates.items(), key=lambda item: str(item[0]))
        },
    }
    return stable_sha256(graph)


def input_audit_receipt(
    frame: pd.DataFrame,
    requested_feature_names: Sequence[str],
    *,
    model: Any | None = None,
    seed_to_train_queries: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    train_query_to_positive_candidates: Mapping[str, Mapping[str, float]] | None = None,
    warn_unexpected_numeric: bool = True,
) -> dict[str, Any]:
    """Build a receipt from the exact fields and assets used for prediction."""

    receipt = validate_feature_contract(
        frame,
        requested_feature_names,
        warn_unexpected_numeric=warn_unexpected_numeric,
    )
    if model is not None and getattr(model, "booster_", None) is not None:
        receipt["model_sha256"] = model_sha256(model)
    elif model is not None and getattr(model, "audit_sha256", None):
        receipt["model_sha256"] = str(model.audit_sha256)
    if seed_to_train_queries is not None and train_query_to_positive_candidates is not None:
        receipt["graph_asset_sha256"] = graph_asset_sha256(
            seed_to_train_queries,
            train_query_to_positive_candidates,
        )
    receipt["uses_validation_or_test_labels_at_inference"] = bool(
        forbidden_feature_names(receipt["actual_feature_columns"])
    )
    receipt["uses_cma_numeric_at_inference"] = any(
        "cma_" in name.lower() for name in receipt["actual_feature_columns"]
    )
    receipt["passed"] = not (
        receipt["forbidden_feature_hits"]
        or receipt["uses_validation_or_test_labels_at_inference"]
        or receipt["uses_cma_numeric_at_inference"]
    )
    if not receipt["passed"]:
        raise ValueError("Model-input leakage audit failed")
    return receipt
