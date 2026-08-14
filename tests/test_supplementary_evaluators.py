from __future__ import annotations

import json
from copy import deepcopy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from cem_wf_index.supplementary.climatenet_portability import (
    evaluate_climatenet_portability,
)
from cem_wf_index.supplementary.matched_scoring import evaluate_matched_scoring
from cem_wf_index.supplementary.physical_metric_ablation import (
    evaluate_physical_metric_ablation,
)
from cem_wf_index.supplementary.run_evaluation import evaluate, main
from cem_wf_index.supplementary.statistics import paired_bootstrap_interval
from cem_wf_index.supplementary.published_evidence import verify_published_evidence


TEMPORAL_FOLDS = [
    {
        "fold": "fold-a",
        "train_end": "2000-12-31",
        "evaluation_start": "2001-01-01",
        "evaluation_end": "2001-12-31",
    },
    {
        "fold": "fold-b",
        "train_end": "2001-12-31",
        "evaluation_start": "2002-01-01",
        "evaluation_end": "2002-12-31",
    },
    {
        "fold": "fold-c",
        "train_end": "2002-12-31",
        "evaluation_start": "2003-01-01",
        "evaluation_end": "2003-12-31",
    },
]


def _climatenet_payload() -> dict[str, object]:
    temporal_folds = deepcopy(TEMPORAL_FOLDS)
    queries = []
    for index, fold in enumerate(temporal_folds, start=1):
        queries.append(
            {
                "query_id": f"synthetic-query-{index}",
                "timestamp": f"{2000 + index}-06-01",
                "fold": fold["fold"],
                "continuous_relevance": [1.0, 0.6, 0.1],
                "seed_scores": [
                    {
                        "seed": seed,
                        "control_scores": [0.1, 0.9, 0.5],
                        "cem_scores": [0.9, 0.6, 0.1],
                    }
                    for seed in range(5)
                ],
            }
        )
    return {
        "study_scope": "development",
        "formal_test_opened": False,
        "scores_sealed_before_relevance": True,
        "input_and_evaluation_annotations_disjoint": True,
        "object_count": 12,
        "model_seed_count": 5,
        "expected_temporal_folds": 3,
        "temporal_folds": temporal_folds,
        "queries": queries,
    }


def _matched_payload() -> dict[str, object]:
    ground_truth = [f"relevant-{index:02d}" for index in range(20)]
    background = [f"background-{index:04d}" for index in range(980)]
    full_pool = ground_truth + background
    scores = {
        candidate_id: (2.0 - index / 100.0 if candidate_id in ground_truth else 0.1 - index / 10_000.0)
        for index, candidate_id in enumerate(full_pool)
    }
    backend_pool = background[:30]
    return {
        "study_scope": "development",
        "scorer_sealed_before_labels": True,
        "scorer_seal": "synthetic-shared-scorer",
        "full_pool_size": 1_000,
        "queries": [
            {
                "query_id": "synthetic-query",
                "ground_truth_ids": ground_truth,
                "sealed_scores": [
                    {"candidate_id": candidate_id, "score": scores[candidate_id]}
                    for candidate_id in full_pool
                ],
                "systems": {
                    "acorn": backend_pool,
                    "cem": full_pool,
                },
            }
        ],
    }


def _physical_payload() -> dict[str, object]:
    temporal_folds = deepcopy(TEMPORAL_FOLDS)
    queries = []
    for index, fold in enumerate(temporal_folds, start=1):
        queries.append(
            {
                "query_id": f"synthetic-query-{index}",
                "timestamp": f"{2000 + index}-07-01",
                "fold": fold["fold"],
                "baseline_ndcg10": 0.4,
                "cem_ndcg10": 0.6,
                "baseline_top1_track_error_km": 800.0,
                "cem_top1_track_error_km": 500.0,
            }
        )
    return {
        "study_scope": "development",
        "expected_temporal_folds": 3,
        "temporal_folds": temporal_folds,
        "queries": queries,
    }


def test_paired_bootstrap_is_deterministic_and_paired():
    baseline = [0.1, 0.2, 0.3, 0.4]
    proposed = [0.2, 0.3, 0.4, 0.5]
    first = paired_bootstrap_interval(baseline, proposed, resamples=500, seed=7)
    second = paired_bootstrap_interval(baseline, proposed, resamples=500, seed=7)
    assert first == second
    assert first.estimate == pytest.approx(0.1)
    assert first.lower == pytest.approx(0.1)
    assert first.upper == pytest.approx(0.1)


def test_climatenet_contract_and_positive_temporal_folds():
    result = evaluate_climatenet_portability(_climatenet_payload(), resamples=500)
    assert result["query_count"] == 3
    assert result["temporal_fold_count"] == 3
    assert result["model_seed_count"] == 5
    assert result["all_temporal_folds_positive"] is True
    assert result["cem_mean"] > result["control_mean"]


def test_climatenet_rejects_evaluation_mask_leakage_and_nonchronological_fold():
    leakage = _climatenet_payload()
    leakage["input_and_evaluation_annotations_disjoint"] = False
    with pytest.raises(ValueError, match="disjoint"):
        evaluate_climatenet_portability(leakage, resamples=500)

    invalid_time = _climatenet_payload()
    invalid_time["temporal_folds"][0]["train_end"] = "2001-06-01"
    with pytest.raises(ValueError, match="strictly after training"):
        evaluate_climatenet_portability(invalid_time, resamples=500)


def test_matched_scoring_holds_shared_candidate_scores_fixed():
    result = evaluate_matched_scoring(_matched_payload(), resamples=500)
    assert result["query_count"] == 1
    assert result["full_system_mean"] > result["backend_mean"]
    assert result["full_system_relevant_at_20"] == 20.0
    assert result["backend_relevant_at_20"] == 0.0

    missing_score = _matched_payload()
    missing_score["queries"][0]["sealed_scores"].pop()
    with pytest.raises(ValueError, match="every candidate"):
        evaluate_matched_scoring(missing_score, resamples=500)


def test_matched_scoring_rejects_unsealed_scorer():
    payload = _matched_payload()
    payload["scorer_sealed_before_labels"] = False
    with pytest.raises(ValueError, match="sealed"):
        evaluate_matched_scoring(payload, resamples=500)


def test_physical_metric_ablation_reports_fold_and_rmse_improvements():
    result = evaluate_physical_metric_ablation(_physical_payload(), resamples=500)
    assert result["all_temporal_folds_positive"] is True
    assert result["paired_bootstrap"]["estimate"] == pytest.approx(0.2)
    assert result["top1_track_rmse_reduction_km"] == pytest.approx(300.0)


def test_cli_writes_only_requested_summary(tmp_path):
    source = tmp_path / "synthetic-input.json"
    output = tmp_path / "synthetic-summary.json"
    source.write_text(json.dumps(_climatenet_payload()), encoding="utf-8")
    exit_code = main(
        [
            "climatenet-portability",
            "--input",
            str(source),
            "--output",
            str(output),
            "--resamples",
            "500",
        ]
    )
    assert exit_code == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["study_scope"] == "development"
    assert result["query_count"] == 3


def test_dispatch_covers_all_three_public_studies():
    common = Namespace(resamples=500, seed=1, backend="acorn", full_system="cem")
    for study, payload in [
        ("climatenet-portability", _climatenet_payload()),
        ("matched-scoring", _matched_payload()),
        ("physical-metric-ablation", _physical_payload()),
    ]:
        common.study = study
        assert evaluate(common, payload)["study_scope"] == "development"


def test_supplementary_readme_limits_public_result_summary():
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "supplementary" / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert text.startswith(
        "# Focused supplementary development results added in response to reviewer questions; "
        "separate from the manuscript's main test tables"
    )
    assert "0.575" in text and "0.663" in text
    assert "0.397" in text and "0.498" in text
    assert "0.148" in text and "469 km" in text
    assert "formal test remains unopened" in text
    assert "manuscript's main held-out test tables" in text


def test_anonymous_public_evidence_reproduces_reported_results():
    evidence = Path(__file__).resolve().parents[1] / "supplementary" / "evidence"
    result = verify_published_evidence(evidence)

    climate = result["climatenet_portability"]
    assert climate["control_mean"] == pytest.approx(0.5746805273)
    assert climate["cem_mean"] == pytest.approx(0.6631771177)
    assert climate["gain"] == pytest.approx(0.0884965905)
    assert climate["ci95"] == pytest.approx([0.0679602931, 0.1094292903])
    assert climate["all_temporal_folds_positive"] is True
    assert climate["seed_mean"] == pytest.approx(0.6631771177)
    assert climate["five_seed_sd"] == pytest.approx(0.0034445708)
    assert climate["model_seed_count"] == 5

    matched = result["matched_scoring"]
    assert matched["control_mean"] == pytest.approx(0.3966762422)
    assert matched["cem_mean"] == pytest.approx(0.4980782698)
    assert matched["gain"] == pytest.approx(0.1014020276)
    assert matched["ci95"] == pytest.approx([0.0541534407, 0.1490239264])
    assert matched["bootstrap_seed"] == 240813
    assert matched["acorn_relevant_at20"] == pytest.approx(4.4193548387)
    assert matched["cem_pool_relevant_at20"] == pytest.approx(7.9677419355)

    physical = result["physical_metric_ablation"]
    assert physical["gain"] == pytest.approx(0.1475719619)
    assert physical["ci95"] == pytest.approx([0.1145335711, 0.1808469085])
    assert physical["all_temporal_folds_positive"] is True
    assert physical["track_rmse_reduction_km"] == pytest.approx(468.5683629)

    diversity = result["candidate_diversity"]
    assert all(row["unique_event_ratio"] == 1.0 for row in diversity)
    assert [row["early_tnms_reduction"] for row in diversity] == pytest.approx(
        [0.0612372881, 0.0935530086]
    )
    audit = result["audit_export"]
    assert audit["field_completeness"] == 1.0
    assert audit["score_consistency"] == 1.0
    assert audit["leakage_check_pass_rate"] == 1.0
    assert audit["p95_overhead_ms_per_query"] == pytest.approx(0.197315)
