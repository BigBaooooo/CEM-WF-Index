from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cem_wf_index.context import (
    ContextIndex,
    FingerprintArchive,
    deduplicate_events,
    early_temporal_nms,
    fuse_candidate_channels,
    synthetic_audit_example,
    unique_event_ratio,
    validate_audit_record,
)
from scripts.run_synthetic_pipeline import run
from scripts.run_synthetic_pipeline import SyntheticFrozenModel
from cem_wf_index.context import run_retrieval_pipeline
from cem_wf_index.final_release.feature_contract import FROZEN_FEATURE_NAMES, build_frozen_asset_manifest


def _synthetic_manifest(model: SyntheticFrozenModel) -> dict:
    return build_frozen_asset_manifest(model, {}, {})


def _archive(tmp_path: Path, *, rows: int = 49, dimension: int = 8) -> FingerprintArchive:
    matrix = np.arange(rows * dimension, dtype=np.float32).reshape(rows, dimension)
    starts = np.datetime64("2020-01-01T00", "h") + np.arange(rows).astype("timedelta64[h]")
    ends = starts + np.timedelta64(24, "h")
    np.save(tmp_path / "all_fp1024.npy", matrix)
    np.save(tmp_path / "start_times.npy", starts)
    np.save(tmp_path / "end_times.npy", ends)
    return FingerprintArchive.from_directory(tmp_path, expected_dim=dimension)


def test_fingerprint_archive_validates_views_and_fixed_normalization(tmp_path: Path):
    archive = _archive(tmp_path)
    receipt = archive.validate()
    assert receipt["row_count"] == 49
    assert receipt["finite_values"] is True
    assert len(archive.view("1h")) == 49
    assert len(archive.view("6h")) == 9
    assert len(archive.view("daily")) == 3

    pooled = archive.normalized_mean_pool(
        "2020-01-01T00",
        "2020-01-02T00",
        cadence="6h",
        feature_mean=np.zeros(8, dtype=np.float32),
        feature_std=np.ones(8, dtype=np.float32),
    )
    assert pooled.shape == (8,)
    assert np.isclose(np.linalg.norm(pooled), 1.0)
    with pytest.raises(ValueError, match="supplied together"):
        archive.normalized_mean_pool(
            "2020-01-01T00", "2020-01-02T00", feature_mean=np.zeros(8, dtype=np.float32)
        )


def test_fingerprint_archive_rejects_misaligned_arrays(tmp_path: Path):
    np.save(tmp_path / "all_fp1024.npy", np.zeros((3, 8), dtype=np.float32))
    np.save(tmp_path / "start_times.npy", np.arange(2).astype("datetime64[h]"))
    np.save(tmp_path / "end_times.npy", np.arange(2).astype("datetime64[h]") + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="identical lengths"):
        FingerprintArchive.from_directory(tmp_path, expected_dim=8)


def test_context_index_exact_l2_exclusions_and_ties():
    vectors = np.asarray([[0, 0], [1, 0], [-1, 0], [4, 0]], dtype=np.float32)
    index = ContextIndex(dimension=2, backend="exact").fit(
        vectors,
        ["self", "a", "b", "c"],
        group_ids=["same", "other-a", "other-b", "other-c"],
    )
    hits = index.search(np.asarray([0, 0], dtype=np.float32), 2, exclude_ids={"self"})
    assert [hit.candidate_id for hit in hits] == ["a", "b"]
    assert [hit.distance for hit in hits] == [1.0, 1.0]
    assert index.configuration() == {
        "backend": "exact",
        "space": "l2",
        "dimension": 2,
        "M": 32,
        "efConstruction": 200,
        "efSearch": 200,
    }
    assert not index.search(np.zeros(2, dtype=np.float32), 2, exclude_group_ids={"same", "other-a", "other-b", "other-c"})


def test_context_index_hnsw_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ContextIndex, "_hnswlib", staticmethod(lambda: None))
    vectors = np.zeros((2, 2), dtype=np.float32)
    assert ContextIndex(dimension=2, backend="auto").fit(vectors, ["a", "b"]).backend_used == "exact"
    with pytest.raises(RuntimeError, match="optional hnswlib"):
        ContextIndex(dimension=2, backend="hnsw").fit(vectors, ["a", "b"])


def test_candidate_fusion_retains_context_evidence_without_direct_score_claim():
    fused = fuse_candidate_channels(
        {
            "event": [
                {"candidate_id": "a", "rank": 1, "distance": 0.1, "provenance": "event-index"},
                {"candidate_id": "b", "rank": 2, "distance": 0.2, "provenance": "event-index"},
            ],
            "context": [
                {"candidate_id": "b", "rank": 1, "distance": 0.3, "provenance": "context-index"},
                {"candidate_id": "c", "rank": 2, "distance": 0.4, "provenance": "context-index"},
            ],
            "metadata": [
                {"candidate_id": "c", "rank": 1, "distance": 0.2, "provenance": "metadata-index"},
            ],
        },
        limit=3,
    )
    by_id = {row["candidate_id"]: row for row in fused}
    assert by_id["b"]["source_membership"] == ["event", "context"]
    assert by_id["b"]["context_rank"] == 1
    assert by_id["b"]["score_context_rank_prior"] == 0.5
    assert by_id["a"]["score_context_rank_prior"] == 1 / 5
    assert by_id["a"]["source_context_rank"] == 4
    assert by_id["b"]["source_event_rank"] == 2
    assert by_id["b"]["standalone_direct_context_term_selected"] is False
    assert by_id["c"]["source_membership"] == ["context", "metadata"]
    assert by_id["c"]["source_metadata_rank"] == 1


def _candidate(candidate_id: str, event_id: str, start: str, end: str, rank: int) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "event_id": event_id,
        "event_type": "storm",
        "time_coverage_start": start,
        "time_coverage_end": end,
        "pool_rank": rank,
    }


def test_temporal_nms_and_hard_event_deduplication():
    rows = [
        _candidate("a-window-1", "a", "2020-01-01", "2020-01-03", 1),
        _candidate("a-window-2", "a", "2020-01-02", "2020-01-04", 2),
        _candidate("b-window", "b", "2020-01-05", "2020-01-07", 3),
    ]
    result = early_temporal_nms(rows, overlap_threshold=0.49, start_time_delta_hours=None)
    assert [row["candidate_id"] for row in result.candidates] == ["a-window-1", "b-window"]
    assert result.reduction_ratio == pytest.approx(1 / 3)
    assert result.unique_event_ratio == 1.0
    assert unique_event_ratio(rows) == pytest.approx(2 / 3)

    distinct, removed = deduplicate_events(rows, top_k=2)
    assert [row["candidate_id"] for row in distinct] == ["a-window-1", "b-window"]
    assert removed == ["a-window-2"]


def test_temporal_nms_uses_inclusive_overlap_and_start_proximity_boundaries():
    overlap_base = _candidate("overlap-base", "event-a", "2020-01-01T00", "2020-01-02T00", 1)
    overlap_equal = _candidate("overlap-equal", "event-b", "2020-01-01T12", "2020-01-02T12", 2)
    overlap_result = early_temporal_nms(
        [overlap_base, overlap_equal],
        overlap_threshold=0.5,
        start_time_delta_hours=None,
    )
    assert overlap_result.overlap_suppressed_candidate_ids == ("overlap-equal",)

    short_base = _candidate("start-base", "event-c", "2020-01-03T00", "2020-01-03T01", 1)
    start_equal = _candidate("start-equal", "event-d", "2020-01-03T12", "2020-01-03T13", 2)
    start_result = early_temporal_nms(
        [short_base, start_equal],
        overlap_threshold=0.5,
        start_time_delta_hours=12.0,
    )
    assert start_result.start_proximity_suppressed_candidate_ids == ("start-equal",)
    assert start_result.overlap_suppressed_candidate_ids == ()


def test_integrated_exclusion_temporal_suppression_and_event_dedup_boundaries(tmp_path: Path):
    archive = _archive(tmp_path, rows=289, dimension=8)
    query = _candidate("query", "query", "2020-01-01T00", "2020-01-02T00", 1)
    query["group_id"] = "query-group"
    same_group = _candidate("same-group", "same-group", "2020-01-04", "2020-01-05", 1)
    same_group["group_id"] = "query-group"
    overlap_a = _candidate("overlap-a", "event-a", "2020-01-05", "2020-01-07", 2)
    overlap_a["group_id"] = "group-a"
    overlap_b = _candidate("overlap-b", "event-b", "2020-01-06", "2020-01-08", 3)
    overlap_b["group_id"] = "group-b"
    duplicate_low_overlap = _candidate("event-a-late", "event-a", "2020-01-10", "2020-01-11", 4)
    duplicate_low_overlap["candidate_id"] = "event-a-late"
    duplicate_low_overlap["event_id"] = "event-a"
    duplicate_low_overlap["group_id"] = "group-a"
    start_base = _candidate("start-base", "event-c", "2020-01-12T00", "2020-01-12T01", 5)
    start_base["group_id"] = "group-c"
    close_start_low_overlap = _candidate("close-start", "event-d", "2020-01-12T12", "2020-01-12T13", 6)
    close_start_low_overlap["group_id"] = "group-d"
    candidates = [
        query,
        same_group,
        overlap_a,
        overlap_b,
        duplicate_low_overlap,
        start_base,
        close_start_low_overlap,
    ]
    try:
        model = SyntheticFrozenModel()
        result = run_retrieval_pipeline(
            archive=archive,
            query_event=query,
            archive_events=[query, same_group, overlap_a, overlap_b, start_base, close_start_low_overlap],
            event_candidates=candidates,
            anchor_top20_rows=[overlap_a, start_base],
            frozen_lambdarank_model=model,
            feature_columns=list(FROZEN_FEATURE_NAMES),
            feature_fill_values=pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
            seed_to_train_queries={},
            train_query_to_positive_candidates={},
            candidate_limit=10,
            top_k=3,
            cadence="1h",
            context_index=ContextIndex(dimension=8, backend="exact"),
            overlap_threshold=0.49,
            start_time_delta_hours=12.0,
            frozen_asset_manifest=_synthetic_manifest(model),
        )
    finally:
        archive.close()
    audit = result.audit
    exclusion = audit["exclusions_and_deduplication"]["query_and_same_group_exclusion"]
    assert set(exclusion["excluded_candidate_ids"]) == {"query", "same-group"}
    assert "overlap-b" in audit["temporal_suppression"]["suppressed_candidate_ids"]
    assert "close-start" in audit["temporal_suppression"]["start_proximity_suppressed_candidate_ids"]
    pre_dedup = audit["exclusions_and_deduplication"]["pre_reranking_same_event_deduplication"]
    assert "event-a-late" in pre_dedup["removed_candidate_ids"]
    standalone = early_temporal_nms(
        [start_base, close_start_low_overlap],
        overlap_threshold=0.49,
        start_time_delta_hours=None,
    )
    assert [row["candidate_id"] for row in standalone.candidates] == ["start-base", "close-start"]
    assert audit["exclusions_and_deduplication"]["post_ranking_event_deduplication"]["applied"] is True


def test_audit_schema_enforces_leakage_boundaries():
    example = synthetic_audit_example()
    validate_audit_record(example)
    assert example["context_stage"]["candidate_path"] == "upstream"
    broken = json.loads(json.dumps(example))
    broken["leakage_check"]["uses_cma_numeric_at_inference"] = True
    with pytest.raises(ValueError, match="CMA numeric"):
        validate_audit_record(broken)


def test_synthetic_pipeline_runs_all_public_stages():
    payload = run(backend="exact")
    assert payload["top_k"]
    assert payload["audit"]["context_stage"]["backend"] == "exact"
    assert payload["audit"]["candidate_generation"]["source_ranks_and_provenance_retained"] is True
    assert payload["audit"]["candidate_generation"]["sources"] == ["event", "context", "metadata"]
    assert any("metadata" in row["source_membership"] for row in payload["top_k"])
    assert payload["audit"]["output"]["event_distinct"] is True
    assert payload["audit"]["leakage_check"]["passed"] is True
    assert payload["audit"]["leakage_check"]["forbidden_feature_hits"] == []
    assert payload["audit"]["leakage_check"]["actual_feature_columns"] == list(FROZEN_FEATURE_NAMES)


def test_pipeline_excludes_query_group_from_every_candidate_channel(tmp_path: Path):
    archive = _archive(tmp_path, rows=73, dimension=8)
    query = _candidate("query", "query", "2020-01-01T00", "2020-01-02T00", 1)
    query["group_id"] = "same-cyclone"
    same_group = _candidate("same", "same", "2020-01-02T00", "2020-01-03T00", 1)
    same_group["group_id"] = "same-cyclone"
    eligible = _candidate("eligible", "eligible", "2020-01-03T00", "2020-01-04T00", 2)
    eligible["group_id"] = "other-cyclone"
    try:
        model = SyntheticFrozenModel()
        result = run_retrieval_pipeline(
            archive=archive,
            query_event=query,
            archive_events=[query, same_group, eligible],
            event_candidates=[query, same_group, eligible],
            anchor_top20_rows=[eligible],
            frozen_lambdarank_model=model,
            feature_columns=list(FROZEN_FEATURE_NAMES),
            feature_fill_values=pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
            seed_to_train_queries={},
            train_query_to_positive_candidates={},
            candidate_limit=3,
            top_k=1,
            cadence="6h",
            context_index=ContextIndex(dimension=8, backend="exact"),
            start_time_delta_hours=12.0,
            frozen_asset_manifest=_synthetic_manifest(model),
        )
    finally:
        archive.close()
    assert result.top_k["candidate_id"].tolist() == ["eligible"]
    exclusion = result.audit["exclusions_and_deduplication"]["query_and_same_group_exclusion"]
    assert exclusion["query_event_excluded"] is True
    assert exclusion["same_group_excluded"] is True
    assert set(exclusion["excluded_candidate_ids"]) == {"query", "same"}
