from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cem_wf_index.final_release.feature_contract import (
    FROZEN_FEATURE_NAMES,
    build_frozen_asset_manifest,
    feature_list_sha256,
    graph_asset_sha256,
    input_audit_receipt,
    load_frozen_asset_manifest,
    validate_feature_contract,
)


class _AuditedModel:
    audit_sha256 = "synthetic-audited-model"
    audit_num_feature = len(FROZEN_FEATURE_NAMES)
    audit_context_split_count = 7
    audit_context_gain = 3.5

    @staticmethod
    def predict(values):
        return np.zeros(len(values), dtype=np.float32)


def _frame(rows: int = 4) -> pd.DataFrame:
    values = {
        name: np.linspace(index, index + 1, rows, dtype=float)
        for index, name in enumerate(FROZEN_FEATURE_NAMES)
    }
    values.update(
        {
            "query_id": ["synthetic-query"] * rows,
            "candidate_id": [f"synthetic-candidate-{index}" for index in range(rows)],
            "is_gt_top20": [True, False, False, True],
            "grade": [3, 0, 0, 2],
        }
    )
    return pd.DataFrame(values)


def test_published_feature_list_is_the_exact_frozen_order():
    assert len(FROZEN_FEATURE_NAMES) == 31
    assert FROZEN_FEATURE_NAMES[3] == "score_context_rank_prior"
    assert feature_list_sha256() == "73ea3cf18b18d38c368d6d305166f57d0a8ecad09e742b15922cf3b2f16b6bac"


def test_published_manifest_binds_the_verified_model_graph_and_context_receipt():
    manifest = load_frozen_asset_manifest()
    assert manifest["feature_contract"] == {
        "feature_count": 31,
        "feature_list_sha256": "73ea3cf18b18d38c368d6d305166f57d0a8ecad09e742b15922cf3b2f16b6bac",
        "context_rank_feature": "score_context_rank_prior",
    }
    assert manifest["model"]["sha256"] == (
        "7e8485f6e48eb18d02adabf66479d2e4ede97510e01a5963af6b916f0835000b"
    )
    assert manifest["model"]["context_rank_prior_split_count"] == 1389
    assert manifest["model"]["context_rank_prior_gain"] == pytest.approx(1327.8582847118378)
    assert manifest["train_label_graph"]["sha256"] == (
        "e1cb39f788c6e32ab06a5297e8d5732349a3fe3fc8bf4cd3ec230df9342e2581"
    )
    assert manifest["train_label_graph"]["train_positive_query_count"] == 1035
    assert manifest["train_label_graph"]["train_positive_seed_count"] == 1151


def test_previous_selector_resolves_to_the_same_frozen_order():
    frame = _frame()
    forbidden_exact = {
        "query_id",
        "candidate_id",
        "task",
        "split_role",
        "is_gt_top20",
        "rank",
        "grade",
        "method_variant",
        "selected_model",
        "candidate_pool",
        "feature_profile",
        "tiebreaker_profile",
        "tail_profile",
        "method_definition",
        "fingerprint_input_scope",
        "context_candidate_path",
        "context_rank_prior_available",
        "direct_context_score_status",
        "pool_source",
        "v7_4_merge_source",
    }
    previous = [
        column
        for column in frame.columns
        if column not in forbidden_exact
        and not column.lower().startswith("uses_")
        and "cma_numeric" not in column.lower()
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    assert previous == list(FROZEN_FEATURE_NAMES)


def test_labels_and_extra_numeric_columns_are_not_model_inputs():
    frame = _frame()
    frame["harmless_numeric_diagnostic"] = 7.0
    with pytest.warns(UserWarning, match="harmless_numeric_diagnostic"):
        receipt = validate_feature_contract(frame)
    assert receipt["actual_feature_columns"] == list(FROZEN_FEATURE_NAMES)
    assert "is_gt_top20" not in receipt["actual_feature_columns"]
    assert "grade" not in receipt["actual_feature_columns"]
    assert receipt["unexpected_numeric_columns"] == ["harmless_numeric_diagnostic"]


@pytest.mark.parametrize("dangerous", ["grade_copy", "cma_pressure", "gt_distance", "test_ndcg"])
def test_dangerous_fields_cannot_be_requested_as_model_features(dangerous: str):
    frame = _frame()
    frame[dangerous] = 1.0
    requested = list(FROZEN_FEATURE_NAMES)
    requested[-1] = dangerous
    with pytest.raises(ValueError, match="Evaluation-only fields"):
        validate_feature_contract(frame, requested)


def test_missing_or_reordered_frozen_features_fail_before_prediction():
    frame = _frame().drop(columns=["graph_mean_z"])
    with pytest.raises(ValueError, match="missing frozen model features"):
        validate_feature_contract(frame)
    reordered = list(FROZEN_FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="ordered contract"):
        validate_feature_contract(_frame(), reordered)


def test_receipt_is_derived_from_actual_inputs_and_train_graph():
    model = _AuditedModel()
    seeds = {"synthetic-seed": [("synthetic-train-query", 1.0)]}
    positives = {"synthetic-train-query": {"synthetic-candidate": 2.0}}
    receipt = input_audit_receipt(
        _frame(),
        list(FROZEN_FEATURE_NAMES),
        model=model,
        seed_to_train_queries=seeds,
        train_query_to_positive_candidates=positives,
        frozen_asset_manifest=build_frozen_asset_manifest(model, seeds, positives),
    )
    assert receipt["passed"] is True
    assert receipt["forbidden_feature_hits"] == []
    assert receipt["uses_validation_or_test_labels_at_inference"] is False
    assert receipt["uses_cma_numeric_at_inference"] is False
    assert receipt["context_rank_prior_available_to_lambdarank"] is True
    assert receipt["graph_asset_sha256"] == graph_asset_sha256(
        seeds,
        positives,
    )


def test_model_and_graph_are_bound_to_the_supplied_frozen_manifest():
    model = _AuditedModel()
    seeds = {"synthetic-seed": [("synthetic-train-query", 1.0)]}
    positives = {"synthetic-train-query": {"synthetic-candidate": 2.0}}
    manifest = build_frozen_asset_manifest(model, seeds, positives)
    receipt = input_audit_receipt(
        _frame(),
        list(FROZEN_FEATURE_NAMES),
        model=model,
        seed_to_train_queries=seeds,
        train_query_to_positive_candidates=positives,
        frozen_asset_manifest=manifest,
    )
    assert receipt["model_hash_match"] is True
    assert receipt["graph_hash_match"] is True
    assert receipt["context_rank_prior_split_count"] == 7
    assert receipt["context_rank_prior_gain"] == 3.5
    assert receipt["context_rank_prior_used_by_frozen_model"] is True
    assert receipt["provenance_basis"] == "hash_match_to_frozen_train_graph"

    changed_graph = {"synthetic-seed": [("synthetic-train-query", 0.5)]}
    with pytest.raises(ValueError, match="train-label graph hash"):
        input_audit_receipt(
            _frame(),
            list(FROZEN_FEATURE_NAMES),
            model=model,
            seed_to_train_queries=changed_graph,
            train_query_to_positive_candidates=positives,
            frozen_asset_manifest=manifest,
        )

    changed_manifest = json.loads(json.dumps(manifest))
    changed_manifest["model"]["sha256"] = "not-the-frozen-model"
    with pytest.raises(ValueError, match="model hash"):
        input_audit_receipt(
            _frame(),
            list(FROZEN_FEATURE_NAMES),
            model=model,
            seed_to_train_queries=seeds,
            train_query_to_positive_candidates=positives,
            frozen_asset_manifest=changed_manifest,
        )
