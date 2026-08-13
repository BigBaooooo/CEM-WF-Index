"""Deterministic multi-channel candidate union with explicit provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def _record(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError("Candidate rows must be mappings or dataclass instances")


def _candidate_id(row: Mapping[str, Any]) -> str:
    value = row.get("candidate_id", row.get("event_id"))
    if value is None or str(value) == "":
        raise ValueError("Every candidate row requires candidate_id or event_id")
    return str(value)


def fuse_candidate_channels(
    channels: Mapping[str, Sequence[Any]],
    *,
    limit: int,
    channel_priority: Sequence[str] = ("event", "context", "metadata"),
    context_channel: str = "context",
) -> list[dict[str, Any]]:
    """Return a stable rank union without treating the union as a final score.

    Candidate membership, per-source ranks, distances, and context provenance
    remain visible to the downstream learned ranker and audit output.  The
    reciprocal context-rank value is a rank prior, not a probability,
    similarity, or standalone final score.
    """

    if limit <= 0:
        return []
    ordered_channels = [name for name in channel_priority if name in channels]
    ordered_channels.extend(sorted(name for name in channels if name not in ordered_channels))
    if not ordered_channels:
        return []

    by_id: dict[str, dict[str, Any]] = {}
    for channel in ordered_channels:
        seen_in_channel: set[str] = set()
        for ordinal, raw in enumerate(channels[channel], start=1):
            row = _record(raw)
            candidate_id = _candidate_id(row)
            if candidate_id in seen_in_channel:
                continue
            seen_in_channel.add(candidate_id)
            source_rank = int(row.get("rank", ordinal))
            if source_rank <= 0:
                raise ValueError("Source ranks must be positive")
            entry = by_id.setdefault(
                candidate_id,
                {
                    "candidate_id": candidate_id,
                    "source_membership": [],
                    "source_ranks": {},
                    "source_distances": {},
                    "source_provenance": {},
                },
            )
            entry["source_membership"].append(channel)
            entry["source_ranks"][channel] = source_rank
            if row.get("distance") is not None:
                entry["source_distances"][channel] = float(row["distance"])
            entry["source_provenance"][channel] = str(row.get("provenance", f"{channel}_candidate_path"))
            for key in (
                "event_id",
                "group_id",
                "event_type",
                "event_start_time",
                "event_end_time",
                "time_coverage_start",
                "time_coverage_end",
                "score_final",
                "score_calibrated",
            ):
                if key in row and key not in entry:
                    entry[key] = row[key]

    missing_rank = int(limit) + 1
    channel_order = {name: idx for idx, name in enumerate(ordered_channels)}

    def ordering(entry: Mapping[str, Any]) -> tuple[Any, ...]:
        ranks = entry["source_ranks"]
        min_rank = min(ranks.values())
        membership_order = min(channel_order[name] for name in entry["source_membership"])
        return (
            min_rank,
            membership_order,
            *(int(ranks.get(name, missing_rank)) for name in ordered_channels),
            str(entry["candidate_id"]),
        )

    fused = sorted(by_id.values(), key=ordering)[: int(limit)]
    for pool_rank, entry in enumerate(fused, start=1):
        ranks = entry["source_ranks"]
        context_rank = ranks.get(context_channel)
        entry["pool_rank"] = pool_rank
        entry["source_min_rank"] = min(ranks.values())
        all_source_ranks = [int(ranks.get(name, missing_rank)) for name in ordered_channels]
        entry["source_rank_gap"] = max(all_source_ranks) - min(all_source_ranks)
        for name in ordered_channels:
            entry[f"source_{name}_rank"] = int(ranks.get(name, missing_rank))
        entry["context_candidate"] = context_rank is not None
        entry["context_rank"] = int(context_rank) if context_rank is not None else None
        entry["score_context_rank_prior"] = 1.0 / float(
            (int(context_rank) + 1) if context_rank is not None else (int(limit) + 2)
        )
        entry["context_candidate_path"] = "upstream"
        entry["context_rank_prior_available"] = context_rank is not None
        entry["direct_context_score_status"] = "not_selected_by_validation"
    return fused
