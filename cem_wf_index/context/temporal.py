"""Early temporal suppression and final event-level de-duplication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence


def _timestamp(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("Missing candidate timestamp")
    return datetime.fromisoformat(text)


def _time_bounds(candidate: Mapping[str, Any]) -> tuple[datetime, datetime]:
    start_value = (
        candidate.get("time_coverage_start")
        or candidate.get("event_start_time")
        or candidate.get("start_time")
        or candidate.get("start_date")
    )
    end_value = (
        candidate.get("time_coverage_end")
        or candidate.get("event_end_time")
        or candidate.get("end_time")
        or candidate.get("end_date")
    )
    if start_value is None:
        raise ValueError("Candidate requires a temporal start field")
    start = _timestamp(start_value)
    if end_value is not None:
        end = _timestamp(end_value)
    else:
        duration = float(candidate.get("event_duration_hours", 24.0))
        end = start + timedelta(hours=duration)
    if end <= start:
        raise ValueError("Candidate temporal coverage must have positive duration")
    return start, end


def temporal_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    """Compute intersection-over-shorter-window and temporal IoU."""

    a_start, a_end = _time_bounds(a)
    b_start, b_end = _time_bounds(b)
    intersection = max(0.0, (min(a_end, b_end) - max(a_start, b_start)).total_seconds() / 3600.0)
    duration_a = (a_end - a_start).total_seconds() / 3600.0
    duration_b = (b_end - b_start).total_seconds() / 3600.0
    union = max(duration_a + duration_b - intersection, 1e-12)
    return {
        "intersection_hours": intersection,
        "overlap_ratio_min": intersection / min(duration_a, duration_b),
        "overlap_ratio_union": intersection / union,
    }


def _event_identity(candidate: Mapping[str, Any], event_key: str) -> str:
    value = candidate.get(event_key)
    if value is None:
        value = candidate.get("group_id", candidate.get("candidate_id"))
    if value is None:
        raise ValueError("Candidate requires an event identity")
    return str(value)


def unique_event_ratio(candidates: Sequence[Mapping[str, Any]], *, event_key: str = "event_id") -> float:
    """Fraction of rows representing distinct physical events."""

    if not candidates:
        return 0.0
    return len({_event_identity(row, event_key) for row in candidates}) / float(len(candidates))


@dataclass(frozen=True)
class TemporalNMSResult:
    candidates: list[dict[str, Any]]
    pre_count: int
    post_count: int
    reduction_ratio: float
    unique_event_ratio: float
    suppressed_candidate_ids: tuple[str, ...]
    overlap_suppressed_candidate_ids: tuple[str, ...]
    start_proximity_suppressed_candidate_ids: tuple[str, ...]
    overlap_threshold: float
    start_time_delta_hours: float | None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "pre_tnms_count": self.pre_count,
            "post_tnms_count": self.post_count,
            "tnms_reduction_ratio": self.reduction_ratio,
            "unique_event_ratio": self.unique_event_ratio,
            "suppressed_candidate_ids": list(self.suppressed_candidate_ids),
            "overlap_suppressed_candidate_ids": list(self.overlap_suppressed_candidate_ids),
            "start_proximity_suppressed_candidate_ids": list(
                self.start_proximity_suppressed_candidate_ids
            ),
            "overlap_threshold": self.overlap_threshold,
            "start_time_delta_hours": self.start_time_delta_hours,
        }


def early_temporal_nms(
    candidates: Sequence[Mapping[str, Any]],
    *,
    overlap_threshold: float = 0.5,
    start_time_delta_hours: float | None,
    rank_key: str = "pool_rank",
    event_key: str = "event_id",
    scope_key: str | None = "event_type",
) -> TemporalNMSResult:
    """Suppress windows conflicting by overlap or nearby start time.

    Input rank is ascending and stable. Suppression is applied only within the
    same optional scope (normally event type), while event identity itself is
    intentionally not used as a shortcut for temporal overlap.
    """

    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")
    if start_time_delta_hours is not None and start_time_delta_hours < 0.0:
        raise ValueError("start_time_delta_hours must be non-negative or None")
    rows = [dict(row) for row in candidates]
    ordered = sorted(
        enumerate(rows),
        key=lambda pair: (
            int(pair[1].get(rank_key, pair[0] + 1)),
            str(pair[1].get("candidate_id", pair[0])),
        ),
    )
    kept: list[dict[str, Any]] = []
    suppressed: list[str] = []
    overlap_suppressed: list[str] = []
    start_suppressed: list[str] = []
    for _, candidate in ordered:
        candidate_scope = candidate.get(scope_key) if scope_key else None
        conflicts = False
        overlap_conflict = False
        start_conflict = False
        for existing in kept:
            if scope_key and existing.get(scope_key) != candidate_scope:
                continue
            overlap_conflict = (
                temporal_overlap(candidate, existing)["overlap_ratio_min"] >= overlap_threshold
            )
            if start_time_delta_hours is not None:
                candidate_start, _ = _time_bounds(candidate)
                existing_start, _ = _time_bounds(existing)
                start_delta = abs((candidate_start - existing_start).total_seconds()) / 3600.0
                start_conflict = start_delta <= start_time_delta_hours
            if overlap_conflict or start_conflict:
                conflicts = True
                break
        if conflicts:
            candidate_id = str(candidate.get("candidate_id", ""))
            suppressed.append(candidate_id)
            if overlap_conflict:
                overlap_suppressed.append(candidate_id)
            if start_conflict:
                start_suppressed.append(candidate_id)
        else:
            kept.append(candidate)
    pre_count = len(rows)
    post_count = len(kept)
    return TemporalNMSResult(
        candidates=kept,
        pre_count=pre_count,
        post_count=post_count,
        reduction_ratio=0.0 if pre_count == 0 else (pre_count - post_count) / float(pre_count),
        unique_event_ratio=unique_event_ratio(kept, event_key=event_key),
        suppressed_candidate_ids=tuple(suppressed),
        overlap_suppressed_candidate_ids=tuple(overlap_suppressed),
        start_proximity_suppressed_candidate_ids=tuple(start_suppressed),
        overlap_threshold=float(overlap_threshold),
        start_time_delta_hours=None
        if start_time_delta_hours is None
        else float(start_time_delta_hours),
    )


def deduplicate_events(
    candidates: Sequence[Mapping[str, Any]],
    *,
    event_key: str = "event_id",
    top_k: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep the highest-ranked row for each physical event."""

    if top_k is not None and top_k < 0:
        raise ValueError("top_k cannot be negative")
    kept: list[dict[str, Any]] = []
    removed: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        row = dict(raw)
        identity = _event_identity(row, event_key)
        if identity in seen:
            removed.append(str(row.get("candidate_id", identity)))
            continue
        seen.add(identity)
        kept.append(row)
        if top_k is not None and len(kept) >= top_k:
            break
    return kept, removed
