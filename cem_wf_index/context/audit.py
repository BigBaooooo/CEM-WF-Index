"""Compact, serializable audit contract for event-level retrieval."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = "cem_wf_retrieval_audit_v2"


@dataclass(frozen=True)
class RetrievalAudit:
    query_id: str
    active_profile: str
    context_stage: Mapping[str, Any]
    candidate_generation: Mapping[str, Any]
    temporal_suppression: Mapping[str, Any]
    exclusions_and_deduplication: Mapping[str, Any]
    graph_provenance: Mapping[str, Any]
    leakage_check: Mapping[str, Any]
    output: Mapping[str, Any]
    schema_version: str = AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        record = asdict(self)
        validate_audit_record(record)
        return record

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def audit_json_schema() -> dict[str, Any]:
    """Return a dependency-free JSON Schema for emitted audit records."""

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CEM-WF-Index retrieval audit",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "query_id",
            "active_profile",
            "context_stage",
            "candidate_generation",
            "temporal_suppression",
            "exclusions_and_deduplication",
            "graph_provenance",
            "leakage_check",
            "output",
        ],
        "properties": {
            "schema_version": {"const": AUDIT_SCHEMA_VERSION},
            "query_id": {"type": "string", "minLength": 1},
            "active_profile": {"type": "string", "minLength": 1},
            "context_stage": {"type": "object"},
            "candidate_generation": {"type": "object"},
            "temporal_suppression": {"type": "object"},
            "exclusions_and_deduplication": {"type": "object"},
            "graph_provenance": {"type": "object"},
            "leakage_check": {
                "type": "object",
                "required": [
                    "actual_feature_columns",
                    "feature_list_sha256",
                    "model_sha256",
                    "graph_asset_sha256",
                    "forbidden_feature_hits",
                    "passed",
                    "uses_validation_or_test_labels_at_inference",
                    "uses_cma_numeric_at_inference",
                ],
                "properties": {
                    "actual_feature_columns": {"type": "array"},
                    "feature_list_sha256": {"type": "string"},
                    "model_sha256": {"type": "string"},
                    "graph_asset_sha256": {"type": "string"},
                    "forbidden_feature_hits": {"type": "array", "maxItems": 0},
                    "passed": {"const": True},
                    "uses_validation_or_test_labels_at_inference": {"const": False},
                    "uses_cma_numeric_at_inference": {"const": False},
                },
            },
            "output": {"type": "object"},
        },
    }


def validate_audit_record(record: Mapping[str, Any]) -> None:
    """Validate required boundaries without adding a JSON Schema dependency."""

    required = set(audit_json_schema()["required"])
    missing = sorted(required.difference(record))
    if missing:
        raise ValueError(f"Audit record is missing required fields: {', '.join(missing)}")
    if record["schema_version"] != AUDIT_SCHEMA_VERSION:
        raise ValueError("Unexpected audit schema_version")
    if not str(record["query_id"]).strip() or not str(record["active_profile"]).strip():
        raise ValueError("query_id and active_profile must be non-empty")
    for key in required.difference({"schema_version", "query_id", "active_profile"}):
        if not isinstance(record[key], Mapping):
            raise TypeError(f"Audit field {key!r} must be an object")
    leakage = record["leakage_check"]
    missing_leakage = sorted(
        set(audit_json_schema()["properties"]["leakage_check"]["required"]).difference(leakage)
    )
    if missing_leakage:
        raise ValueError(f"Leakage audit is missing executable receipt fields: {', '.join(missing_leakage)}")
    if leakage.get("passed") is not True or leakage.get("forbidden_feature_hits"):
        raise ValueError("Model-input leakage audit did not pass")
    if leakage.get("uses_validation_or_test_labels_at_inference") is not False:
        raise ValueError("Audit cannot certify inference that accesses held-out labels")
    if leakage.get("uses_cma_numeric_at_inference") is not False:
        raise ValueError("Audit cannot certify inference that accesses CMA numeric fields")


def synthetic_audit_example() -> dict[str, Any]:
    """Return a data-free example illustrating every audit boundary."""

    from cem_wf_index.final_release.feature_contract import FROZEN_FEATURE_NAMES, feature_list_sha256

    return RetrievalAudit(
        query_id="synthetic-query",
        active_profile="validation-selected-frozen-profile",
        context_stage={
            "candidate_path": "upstream",
            "index_space": "l2",
            "dimension": 1024,
            "rank_prior_retained": True,
            "context_used_for_candidate_generation": True,
            "context_rank_prior_consumed_by_lambdarank": True,
            "standalone_direct_context_term_selected": False,
        },
        candidate_generation={
            "sources": ["event", "context", "metadata"],
            "source_ranks_and_provenance_retained": True,
            "fused_candidate_count": 24,
        },
        temporal_suppression={
            "stage": "before_expensive_reranking",
            "pre_tnms_count": 24,
            "post_tnms_count": 22,
            "tnms_reduction_ratio": 2 / 24,
            "unique_event_ratio": 1.0,
        },
        exclusions_and_deduplication={
            "query_and_same_group_exclusion": {
                "query_event_excluded": True,
                "same_group_excluded": True,
                "excluded_candidate_ids": [],
            },
            "pre_reranking_same_event_deduplication": {"removed_candidate_ids": []},
            "post_ranking_event_deduplication": {"applied": True, "removed_candidate_ids": []},
        },
        graph_provenance={
            "asset_scope": "training_split_only",
            "frozen_before_test": True,
        },
        leakage_check={
            "actual_feature_columns": list(FROZEN_FEATURE_NAMES),
            "feature_list_sha256": feature_list_sha256(),
            "model_sha256": "synthetic-model-sha256",
            "graph_asset_sha256": "synthetic-graph-sha256",
            "forbidden_feature_hits": [],
            "passed": True,
            "uses_validation_or_test_labels_at_inference": False,
            "uses_cma_numeric_at_inference": False,
        },
        output={"top_k": 20, "event_distinct": True, "candidate_provenance_retained": True},
    ).to_dict()
