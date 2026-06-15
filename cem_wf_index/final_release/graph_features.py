"""Train-label graph features used by the final label-graph LambdaRank model."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def _query_z(rows: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    out = np.zeros(len(rows), dtype=np.float32)
    for _, idx in rows.groupby("query_id", sort=False).indices.items():
        local = raw[idx]
        std = float(np.std(local))
        out[idx] = ((local - float(np.mean(local))) / (std if std > 1e-9 else 1.0)).astype(np.float32)
    return out


def _col_z(rows: pd.DataFrame, col: str) -> np.ndarray:
    if col not in rows.columns:
        return np.zeros(len(rows), dtype=np.float32)
    return _query_z(rows, pd.to_numeric(rows[col], errors="coerce").fillna(0.0).to_numpy(float))


def _train_graph(train: pd.DataFrame) -> tuple[dict[str, list[tuple[str, float]]], dict[str, dict[str, float]]]:
    pos = train[train["is_gt_top20"].astype(bool)].copy()
    pos["candidate_id"] = pos["candidate_id"].astype(str)
    pos["query_id"] = pos["query_id"].astype(str)
    grade = pd.to_numeric(pos.get("grade", 1.0), errors="coerce").fillna(1.0).astype(float)
    pos["grade_weight"] = 1.0 + grade / max(float(grade.max()), 1.0)
    seed_to_queries: dict[str, list[tuple[str, float]]] = defaultdict(list)
    query_to_pos: dict[str, dict[str, float]] = {}
    for query_id, group in pos.groupby("query_id", sort=False):
        qmap: dict[str, float] = {}
        for row in group.itertuples(index=False):
            candidate_id = str(row.candidate_id)
            weight = float(row.grade_weight)
            qmap[candidate_id] = max(qmap.get(candidate_id, 0.0), weight)
            seed_to_queries[candidate_id].append((str(query_id), weight))
        query_to_pos[str(query_id)] = qmap
    return seed_to_queries, query_to_pos


def _graph_scores(
    rows: pd.DataFrame,
    seed_to_queries: dict[str, list[tuple[str, float]]],
    query_to_pos: dict[str, dict[str, float]],
    *,
    seed_k: int,
    activated_query_cap: int,
) -> np.ndarray:
    out = np.zeros(len(rows), dtype=np.float32)
    index_to_pos = {idx: pos for pos, idx in enumerate(rows.index)}
    final_z = _col_z(rows, "score_final")
    source_z = _col_z(rows, "source_min_rank_inv")
    frame = rows.copy()
    frame["_seed_score"] = 0.70 * final_z + 0.30 * source_z
    for _, group in frame.groupby("query_id", sort=False):
        local = group.sort_values("_seed_score", ascending=False)
        seeds = local["candidate_id"].astype(str).head(seed_k).tolist()
        activated: dict[str, float] = defaultdict(float)
        for rank, seed in enumerate(seeds, start=1):
            seed_weight = 1.0 / rank
            for train_query, weight in seed_to_queries.get(seed, []):
                activated[train_query] += seed_weight * weight
        if not activated:
            continue
        activated_items = sorted(activated.items(), key=lambda item: item[1], reverse=True)[:activated_query_cap]
        cand_scores: dict[str, float] = defaultdict(float)
        norm = 1e-6
        for train_query, q_weight in activated_items:
            norm += q_weight
            for candidate_id, c_weight in query_to_pos.get(train_query, {}).items():
                cand_scores[candidate_id] += q_weight * c_weight
        if not cand_scores:
            continue
        for idx, candidate_id in zip(local.index, local["candidate_id"].astype(str)):
            out[index_to_pos[idx]] = float(cand_scores.get(candidate_id, 0.0) / norm)
    return out
