"""Background-context candidate generation for CEM-WF-Index.

This package is deliberately upstream of the frozen final-score implementation.
It exposes the precomputed fingerprint input, its L2 index, candidate provenance,
temporal suppression, and audit records without changing final-score weights.
"""

from cem_wf_index.context.audit import (
    AUDIT_SCHEMA_VERSION,
    RetrievalAudit,
    audit_json_schema,
    synthetic_audit_example,
    validate_audit_record,
)
from cem_wf_index.context.candidates import fuse_candidate_channels
from cem_wf_index.context.fingerprints import FingerprintArchive, FingerprintView
from cem_wf_index.context.index import ContextIndex, SearchHit
from cem_wf_index.context.pipeline import PipelineResult, run_retrieval_pipeline
from cem_wf_index.context.temporal import (
    deduplicate_events,
    early_temporal_nms,
    temporal_overlap,
    unique_event_ratio,
)

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "ContextIndex",
    "FingerprintArchive",
    "FingerprintView",
    "PipelineResult",
    "RetrievalAudit",
    "SearchHit",
    "audit_json_schema",
    "deduplicate_events",
    "early_temporal_nms",
    "fuse_candidate_channels",
    "run_retrieval_pipeline",
    "synthetic_audit_example",
    "temporal_overlap",
    "unique_event_ratio",
    "validate_audit_record",
]
