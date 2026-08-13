from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cem_wf_index.final_release.feature_contract import (
    FEATURE_LIST_PATH,
    FROZEN_FEATURE_NAMES,
    feature_list_sha256,
    graph_asset_sha256,
    input_audit_receipt,
    validate_feature_contract,
)


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
    assert hashlib.sha256(FEATURE_LIST_PATH.read_bytes()).hexdigest() == (
        "9deaad5c3c8c9e7cae9cf3a9dd577b8e638454c3a2d7736611c173125717b246"
    )
    assert feature_list_sha256() == "73ea3cf18b18d38c368d6d305166f57d0a8ecad09e742b15922cf3b2f16b6bac"


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
    assert set(receipt["forbidden_table_fields_excluded_from_model"]) == {"is_gt_top20", "grade"}
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
    receipt = input_audit_receipt(
        _frame(),
        list(FROZEN_FEATURE_NAMES),
        seed_to_train_queries={"synthetic-seed": [("synthetic-train-query", 1.0)]},
        train_query_to_positive_candidates={"synthetic-train-query": {"synthetic-candidate": 2.0}},
    )
    assert receipt["passed"] is True
    assert receipt["forbidden_feature_hits"] == []
    assert receipt["uses_validation_or_test_labels_at_inference"] is False
    assert receipt["uses_cma_numeric_at_inference"] is False
    assert receipt["context_rank_prior_consumed_by_lambdarank"] is True
    assert receipt["graph_asset_sha256"] == graph_asset_sha256(
        {"synthetic-seed": [("synthetic-train-query", 1.0)]},
        {"synthetic-train-query": {"synthetic-candidate": 2.0}},
    )
