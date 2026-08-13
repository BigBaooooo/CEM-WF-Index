"""Reference composition of context retrieval and the frozen final ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from cem_wf_index.context.audit import RetrievalAudit
from cem_wf_index.context.candidates import fuse_candidate_channels
from cem_wf_index.context.fingerprints import FingerprintArchive
from cem_wf_index.context.index import ContextIndex
from cem_wf_index.context.temporal import deduplicate_events, early_temporal_nms
from cem_wf_index.final_release.online_inference import SELECTED_PROFILE, rank_v8_22_online


@dataclass(frozen=True)
class PipelineResult:
    top_k: pd.DataFrame
    reranker_input: pd.DataFrame
    audit: dict[str, Any]


def _event_vector(archive: FingerprintArchive, row: Mapping[str, Any], cadence: str | int) -> np.ndarray:
    start = row.get("time_coverage_start", row.get("event_start_time"))
    end = row.get("time_coverage_end", row.get("event_end_time"))
    if start is None or end is None:
        raise ValueError("Event rows require time_coverage_start/end or event_start_time/end")
    return archive.normalized_mean_pool(start, end, cadence=cadence)


def run_retrieval_pipeline(
    *,
    archive: FingerprintArchive,
    query_event: Mapping[str, Any],
    archive_events: Sequence[Mapping[str, Any]],
    event_candidates: Sequence[Mapping[str, Any]],
    anchor_top20_rows: Sequence[Mapping[str, Any]] | pd.DataFrame,
    frozen_lambdarank_model: Any,
    feature_columns: Sequence[str],
    feature_fill_values: pd.Series,
    seed_to_train_queries: dict[str, list[tuple[str, float]]],
    train_query_to_positive_candidates: dict[str, dict[str, float]],
    candidate_limit: int = 1000,
    top_k: int = 20,
    cadence: str | int = "6h",
    context_index: ContextIndex | None = None,
    metadata_candidates: Sequence[Mapping[str, Any]] | None = None,
    overlap_threshold: float = 0.5,
) -> PipelineResult:
    """Run the public stage composition without refitting the final scorer."""

    if candidate_limit <= 0 or top_k <= 0:
        raise ValueError("candidate_limit and top_k must be positive")
    query_id = str(query_event.get("event_id", query_event.get("query_id", "")))
    if not query_id:
        raise ValueError("query_event requires event_id or query_id")
    query_group = query_event.get("group_id")
    rows = [dict(row) for row in archive_events]
    ids = [str(row.get("event_id", row.get("candidate_id", ""))) for row in rows]
    if any(not value for value in ids):
        raise ValueError("archive_events require event_id or candidate_id")
    groups = [None if row.get("group_id") is None else str(row["group_id"]) for row in rows]
    vectors = np.stack([_event_vector(archive, row, cadence) for row in rows]).astype(np.float32)
    index = context_index if context_index is not None else ContextIndex(dimension=archive.dimension)
    index.fit(vectors, ids, group_ids=groups)
    query_vector = _event_vector(archive, query_event, cadence)
    hits = index.search(
        query_vector,
        k=min(candidate_limit, len(rows)),
        exclude_ids={query_id},
        exclude_group_ids=() if query_group is None else {str(query_group)},
    )
    metadata_by_id = {candidate_id: row for candidate_id, row in zip(ids, rows)}
    context_rows: list[dict[str, Any]] = []
    for hit in hits:
        row = hit.to_dict()
        row.update(metadata_by_id[hit.candidate_id])
        row["candidate_id"] = hit.candidate_id
        context_rows.append(row)

    excluded_ids: list[str] = []
    eligible_event_rows: list[dict[str, Any]] = []
    for raw in event_candidates:
        row = dict(raw)
        candidate_id = str(row.get("candidate_id", row.get("event_id", "")))
        if not candidate_id:
            raise ValueError("event_candidates require candidate_id or event_id")
        metadata = metadata_by_id.get(candidate_id, {})
        for key, value in metadata.items():
            row.setdefault(key, value)
        row["candidate_id"] = candidate_id
        same_group = query_group is not None and str(row.get("group_id")) == str(query_group)
        if candidate_id == query_id or same_group:
            excluded_ids.append(candidate_id)
            continue
        eligible_event_rows.append(row)

    eligible_metadata_rows: list[dict[str, Any]] = []
    for raw in metadata_candidates or ():
        row = dict(raw)
        candidate_id = str(row.get("candidate_id", row.get("event_id", "")))
        if not candidate_id:
            raise ValueError("metadata_candidates require candidate_id or event_id")
        metadata = metadata_by_id.get(candidate_id, {})
        for key, value in metadata.items():
            row.setdefault(key, value)
        row["candidate_id"] = candidate_id
        same_group = query_group is not None and str(row.get("group_id")) == str(query_group)
        if candidate_id == query_id or same_group:
            excluded_ids.append(candidate_id)
            continue
        eligible_metadata_rows.append(row)

    candidate_channels: dict[str, Sequence[Mapping[str, Any]]] = {
        "event": eligible_event_rows,
        "context": context_rows,
    }
    if metadata_candidates is not None:
        candidate_channels["metadata"] = eligible_metadata_rows
    fused = fuse_candidate_channels(candidate_channels, limit=candidate_limit)
    nms = early_temporal_nms(
        fused,
        overlap_threshold=overlap_threshold,
        rank_key="pool_rank",
        event_key="event_id",
    )
    event_distinct_candidates, pre_rank_duplicates = deduplicate_events(nms.candidates, event_key="event_id")
    reranker_input = pd.DataFrame(event_distinct_candidates)
    if reranker_input.empty:
        raise ValueError("No candidates remain after exclusions and temporal suppression")
    reranker_input["query_id"] = query_id
    for column, default in (("score_final", 0.0), ("score_calibrated", 0.0)):
        if column not in reranker_input:
            reranker_input[column] = default
    anchors = pd.DataFrame(anchor_top20_rows).copy()
    if anchors.empty or "candidate_id" not in anchors:
        raise ValueError("anchor_top20_rows must contain the frozen anchor candidate IDs")
    eligible_ids = set(reranker_input["candidate_id"].astype(str))
    anchors["candidate_id"] = anchors["candidate_id"].astype(str)
    anchors = anchors[anchors["candidate_id"].isin(eligible_ids)].copy()
    if anchors.empty:
        raise ValueError("No frozen anchor candidate remains in the reranker input")
    if "rank" not in anchors:
        anchors["rank"] = np.arange(1, len(anchors) + 1)
    anchors["query_id"] = query_id
    ranked = rank_v8_22_online(
        reranker_input,
        anchors,
        frozen_lambdarank_model,
        list(feature_columns),
        feature_fill_values,
        seed_to_train_queries,
        train_query_to_positive_candidates,
    )
    feature_audit = dict(ranked.attrs["feature_audit"])
    ranked_rows = ranked.sort_values("rank", kind="mergesort").to_dict("records")
    distinct_rows, removed = deduplicate_events(ranked_rows, event_key="event_id", top_k=top_k)
    top = pd.DataFrame(distinct_rows)
    if not top.empty:
        top["rank"] = np.arange(1, len(top) + 1)

    audit = RetrievalAudit(
        query_id=query_id,
        active_profile=SELECTED_PROFILE,
        context_stage={
            **index.configuration(),
            "candidate_path": "upstream",
            "retrieved_count": len(context_rows),
            "rank_prior_retained": "score_context_rank_prior" in reranker_input.columns,
            "context_used_for_candidate_generation": True,
            "context_rank_prior_consumed_by_lambdarank": feature_audit[
                "context_rank_prior_consumed_by_lambdarank"
            ],
            "standalone_direct_context_term_selected": False,
        },
        candidate_generation={
            "sources": list(candidate_channels),
            "fused_candidate_count": len(fused),
            "reranker_candidate_count": len(reranker_input),
            "source_ranks_and_provenance_retained": True,
        },
        temporal_suppression={
            "stage": "temporal_overlap_suppression_before_reranking",
            **nms.diagnostics(),
        },
        exclusions_and_deduplication={
            "query_and_same_group_exclusion": {
                "query_event_excluded": query_id not in reranker_input["candidate_id"].astype(str).tolist(),
                "same_group_excluded": query_group is None
                or all(str(value) != str(query_group) for value in reranker_input.get("group_id", [])),
                "excluded_candidate_ids": excluded_ids,
            },
            "pre_reranking_same_event_deduplication": {
                "removed_candidate_ids": pre_rank_duplicates,
            },
            "post_ranking_event_deduplication": {
                "applied": True,
                "removed_candidate_ids": removed,
            },
        },
        graph_provenance={"asset_scope": "training_split_only", "frozen_before_test": True},
        leakage_check=feature_audit,
        output={
            "requested_top_k": top_k,
            "returned_count": len(top),
            "event_distinct": not top.get("event_id", pd.Series(dtype=object)).duplicated().any(),
            "candidate_provenance_retained": "source_provenance" in top.columns,
        },
    ).to_dict()
    return PipelineResult(top_k=top, reranker_input=reranker_input, audit=audit)
