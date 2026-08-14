"""Frozen feature contract and executable inference-input audit."""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_LIST_PATH = Path(__file__).with_name("typhoon_frozen_feature_names.csv")
FROZEN_ASSET_MANIFEST_PATH = Path(__file__).with_name("typhoon_frozen_asset_manifest.json")
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


def load_frozen_asset_manifest(
    source: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and minimally validate the hash-bound frozen asset contract."""

    if source is None:
        payload = json.loads(FROZEN_ASSET_MANIFEST_PATH.read_text(encoding="utf-8"))
    elif isinstance(source, Mapping):
        payload = json.loads(json.dumps(dict(source)))
    else:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
    required = {"schema_version", "feature_contract", "model", "train_label_graph"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Frozen asset manifest is missing sections: {missing}")
    if payload["schema_version"] != "cem_wf_frozen_assets_v1":
        raise ValueError("Unsupported frozen asset manifest schema")
    return payload


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


def _model_hash(model: Any) -> str:
    if getattr(model, "booster_", None) is not None:
        return model_sha256(model)
    value = getattr(model, "audit_sha256", None)
    if not value:
        raise TypeError("A fitted LightGBM model or an explicitly audited test double is required")
    return str(value)


def _model_input_dimension(model: Any) -> int:
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "num_feature"):
        return int(booster.num_feature())
    if getattr(model, "n_features_in_", None) is not None:
        return int(model.n_features_in_)
    if getattr(model, "audit_num_feature", None) is not None:
        return int(model.audit_num_feature)
    raise TypeError("The model does not expose its fitted input dimension")


def context_feature_importance(
    model: Any,
    feature_names: Sequence[str] = FROZEN_FEATURE_NAMES,
) -> dict[str, float | int | bool]:
    """Read split/gain importance for the Context prior from the fitted scorer."""

    names = list(feature_names)
    if CONTEXT_RANK_FEATURE not in names:
        raise ValueError(f"Frozen feature contract does not contain {CONTEXT_RANK_FEATURE}")
    position = names.index(CONTEXT_RANK_FEATURE)
    booster = getattr(model, "booster_", None)
    if booster is not None and hasattr(booster, "feature_importance"):
        split_values = booster.feature_importance(importance_type="split")
        gain_values = booster.feature_importance(importance_type="gain")
        split_count = int(split_values[position])
        gain = float(gain_values[position])
    else:
        split_count = int(getattr(model, "audit_context_split_count", 0))
        gain = float(getattr(model, "audit_context_gain", 0.0))
    return {
        "context_rank_prior_split_count": split_count,
        "context_rank_prior_gain": gain,
        "context_rank_prior_used_by_frozen_model": bool(split_count > 0 and gain > 0.0),
    }


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
        "unexpected_numeric_columns": unexpected,
        "suspicious_non_input_columns": suspicious,
        "context_rank_prior_available_to_lambdarank": CONTEXT_RANK_FEATURE in requested,
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


def build_frozen_asset_manifest(
    model: Any,
    seed_to_train_queries: Mapping[str, Sequence[tuple[str, float]]],
    train_query_to_positive_candidates: Mapping[str, Mapping[str, float]],
    *,
    feature_names: Sequence[str] = FROZEN_FEATURE_NAMES,
    source_split: str = "train",
    frozen_before_test: bool = True,
) -> dict[str, Any]:
    """Build a manifest for an explicitly supplied frozen asset set.

    Production inference uses the published manifest. This constructor is also
    used by synthetic examples so their test doubles never impersonate it.
    """

    importance = context_feature_importance(model, feature_names)
    return {
        "schema_version": "cem_wf_frozen_assets_v1",
        "feature_contract": {
            "feature_count": len(feature_names),
            "feature_list_sha256": feature_list_sha256(feature_names),
            "context_rank_feature": CONTEXT_RANK_FEATURE,
        },
        "model": {
            "sha256": _model_hash(model),
            "input_dimension": _model_input_dimension(model),
            **importance,
        },
        "train_label_graph": {
            "sha256": graph_asset_sha256(seed_to_train_queries, train_query_to_positive_candidates),
            "source_split": source_split,
            "frozen_before_test": bool(frozen_before_test),
            "train_positive_query_count": len(train_query_to_positive_candidates),
            "train_positive_seed_count": len(seed_to_train_queries),
        },
    }


def verify_frozen_asset_bindings(
    model: Any,
    requested_feature_names: Sequence[str],
    seed_to_train_queries: Mapping[str, Sequence[tuple[str, float]]],
    train_query_to_positive_candidates: Mapping[str, Mapping[str, float]],
    *,
    frozen_asset_manifest: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify model, feature order and graph against one frozen manifest."""

    manifest = load_frozen_asset_manifest(frozen_asset_manifest)
    feature_contract = manifest["feature_contract"]
    expected_model = manifest["model"]
    expected_graph = manifest["train_label_graph"]
    requested = list(requested_feature_names)
    actual_feature_hash = feature_list_sha256(requested)
    actual_model_hash = _model_hash(model)
    actual_dimension = _model_input_dimension(model)
    actual_graph_hash = graph_asset_sha256(seed_to_train_queries, train_query_to_positive_candidates)
    importance = context_feature_importance(model, requested)

    failures: list[str] = []
    if len(requested) != int(feature_contract["feature_count"]):
        failures.append("feature count")
    if actual_feature_hash != str(feature_contract["feature_list_sha256"]):
        failures.append("feature order/hash")
    if actual_dimension != int(expected_model["input_dimension"]):
        failures.append("model input dimension")
    if actual_model_hash != str(expected_model["sha256"]):
        failures.append("model hash")
    if actual_graph_hash != str(expected_graph["sha256"]):
        failures.append("train-label graph hash")
    if importance["context_rank_prior_split_count"] != int(
        expected_model["context_rank_prior_split_count"]
    ):
        failures.append("Context split importance")
    if not math.isclose(
        float(importance["context_rank_prior_gain"]),
        float(expected_model["context_rank_prior_gain"]),
        rel_tol=1e-12,
        abs_tol=1e-9,
    ):
        failures.append("Context gain importance")
    if failures:
        raise ValueError("Frozen asset verification failed: " + ", ".join(failures))

    graph_hash_match = actual_graph_hash == str(expected_graph["sha256"])
    model_hash_match = actual_model_hash == str(expected_model["sha256"])
    provenance_verified = bool(
        graph_hash_match
        and str(expected_graph.get("source_split", "")) == "train"
        and expected_graph.get("frozen_before_test") is True
    )
    return {
        "feature_count": len(requested),
        "feature_list_sha256": actual_feature_hash,
        "model_input_dimension": actual_dimension,
        "model_sha256": actual_model_hash,
        "model_hash_match": model_hash_match,
        "graph_asset_sha256": actual_graph_hash,
        "graph_hash_match": graph_hash_match,
        "training_split_only": provenance_verified,
        "frozen_before_test": provenance_verified,
        "provenance_basis": "hash_match_to_frozen_train_graph",
        **importance,
        "asset_binding_passed": True,
    }


def input_audit_receipt(
    frame: pd.DataFrame,
    requested_feature_names: Sequence[str],
    *,
    model: Any | None = None,
    seed_to_train_queries: Mapping[str, Sequence[tuple[str, float]]] | None = None,
    train_query_to_positive_candidates: Mapping[str, Mapping[str, float]] | None = None,
    frozen_asset_manifest: str | Path | Mapping[str, Any] | None = None,
    warn_unexpected_numeric: bool = True,
) -> dict[str, Any]:
    """Build a receipt from the exact fields and assets used for prediction."""

    receipt = validate_feature_contract(
        frame,
        requested_feature_names,
        warn_unexpected_numeric=warn_unexpected_numeric,
    )
    supplied_assets = (
        model is not None,
        seed_to_train_queries is not None,
        train_query_to_positive_candidates is not None,
    )
    if any(supplied_assets) and not all(supplied_assets):
        raise ValueError("Model and both train-label graph mappings must be audited together")
    if all(supplied_assets):
        receipt.update(
            verify_frozen_asset_bindings(
                model,
                requested_feature_names,
                seed_to_train_queries,
                train_query_to_positive_candidates,
                frozen_asset_manifest=frozen_asset_manifest,
            )
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
