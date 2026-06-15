"""Minimal helper functions for final candidate-row handling and anchor merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


TARGET = {
    "nDCG@10": 0.82,
    "mAP@10": 0.72,
    "Recall@10": 0.40,
    "Recall@20": 0.64,
    "Track RMSE/km": 850.0,
    "Landfall dist/km": 750.0,
}


def _safe_float(value: Any, default: float = float("nan")) -> float:
    out = pd.to_numeric(value, errors="coerce")
    return float(out) if pd.notna(out) else default


def _read_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["query_id"] = frame["query_id"].astype(str)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["is_gt_top20"] = frame["is_gt_top20"].astype(bool)
    for col in ["rank", "source_event_rank", "source_sequence_rank", "source_min_rank"]:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce").fillna(1501).clip(lower=0)
            frame[f"{col}_inv"] = 1.0 / (vals + 1.0)
    return frame


def _relevant(frame: pd.DataFrame) -> dict[str, set[str]]:
    return {
        str(query_id): set(group[group["is_gt_top20"].astype(bool)]["candidate_id"].astype(str))
        for query_id, group in frame.groupby("query_id", sort=False)
    }


def _anchor_merge(anchor: pd.DataFrame, proposal: pd.DataFrame, *, keep_top: int, tail_from: int, profile: str) -> pd.DataFrame:
    if keep_top <= 0 and tail_from > 20:
        out = proposal.copy()
        out["anchor_keep_top"] = 0
        out["anchor_tail_from"] = tail_from
        out["anchor_profile"] = profile
        return out
    anchor_groups = {str(q): g.sort_values("rank").copy() for q, g in anchor.groupby("query_id", sort=False)}
    pieces: list[pd.DataFrame] = []
    for query_id, prop_group in proposal.groupby("query_id", sort=False):
        prop = prop_group.sort_values("rank").copy()
        prop_by_id = {str(row.candidate_id): row._asdict() for row in prop.itertuples(index=False)}
        anchor_group = anchor_groups.get(str(query_id), pd.DataFrame()).copy()
        anchor_by_id = {str(row.candidate_id): row._asdict() for row in anchor_group.itertuples(index=False)} if not anchor_group.empty else {}
        ordered: list[str] = []
        for candidate_id in anchor_group.head(keep_top)["candidate_id"].astype(str).tolist():
            if candidate_id not in ordered:
                ordered.append(candidate_id)
        proposal_head_limit = max(tail_from - 1, keep_top)
        for candidate_id in prop["candidate_id"].astype(str).tolist():
            if len(ordered) >= proposal_head_limit:
                break
            if candidate_id not in ordered:
                ordered.append(candidate_id)
        if tail_from <= 20:
            for candidate_id in anchor_group["candidate_id"].astype(str).tolist():
                if candidate_id not in ordered:
                    ordered.append(candidate_id)
                if len(ordered) >= 20:
                    break
        for candidate_id in prop["candidate_id"].astype(str).tolist():
            if len(ordered) >= 20:
                break
            if candidate_id not in ordered:
                ordered.append(candidate_id)
        rows: list[dict[str, Any]] = []
        for rank, candidate_id in enumerate(ordered[:20], start=1):
            row = dict(prop_by_id.get(candidate_id) or anchor_by_id.get(candidate_id) or {})
            if not row:
                continue
            row["rank"] = rank
            row["anchor_keep_top"] = keep_top
            row["anchor_tail_from"] = tail_from
            row["anchor_profile"] = profile
            row["anchor_source"] = (
                "retained_head"
                if rank <= keep_top and candidate_id in anchor_by_id
                else "retained_tail"
                if tail_from <= rank <= 20 and candidate_id in anchor_by_id
                else "proposal"
            )
            rows.append(row)
        if rows:
            pieces.append(pd.DataFrame(rows))
    if not pieces:
        return proposal.head(0).copy()
    return pd.concat(pieces, ignore_index=True)
