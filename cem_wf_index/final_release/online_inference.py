"""Final online inference reference.

This file is a paper-package reference implementation of the frozen V8.22
online ranking path. It assumes the offline stage has already built the
train-label graph and trained the LightGBM LambdaRank model.

Policy boundary:
- train-split CMA GT labels are used only by the offline graph/model.
- online/test ranking must not read CMA numeric track/intensity/landfall fields.
- precomputed ERA5 background-context fingerprints enter through the upstream
  candidate rows; this downstream reference does not train that representation.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from cem_wf_index.final_release.feature_contract import input_audit_receipt
from cem_wf_index.final_release.feature_contract import verify_frozen_asset_bindings


METHOD_DEFINITION = "v8_22_label_graph_lambdarank"
METHOD_NAME = "CEM-WF-Index v8.22 Label-Graph LambdaRank"
SELECTED_PROFILE = "lgbm_ranker_grade_leaves63_lr03|blend_f0.10_m0.45_g0.30_c0.15_s0.00|anchor_keep5"
GRAPH_CONFIGS = [
    ("g18_seed20_cap200", 20, 200),
    ("g19_seed16_cap200", 16, 200),
    ("gwide_seed20_cap500", 20, 500),
    ("gdeep_seed40_cap500", 40, 500),
]
BLEND = {"wf": 0.10, "wm": 0.45, "wg": 0.30, "wc": 0.15, "ws": 0.00}
ANCHOR_KEEP = 5


def query_z(rows: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=float)
    out = np.zeros(len(rows), dtype=np.float32)
    for _, idx in rows.groupby("query_id", sort=False).indices.items():
        local = raw[idx]
        std = float(np.std(local))
        out[idx] = ((local - float(np.mean(local))) / (std if std > 1e-9 else 1.0)).astype(np.float32)
    return out


def col_z(rows: pd.DataFrame, col: str) -> np.ndarray:
    if col not in rows.columns:
        return np.zeros(len(rows), dtype=np.float32)
    return query_z(rows, pd.to_numeric(rows[col], errors="coerce").fillna(0.0).to_numpy(float))


def graph_scores(
    candidate_rows: pd.DataFrame,
    seed_to_train_queries: dict[str, list[tuple[str, float]]],
    train_query_to_positive_candidates: dict[str, dict[str, float]],
    *,
    seed_k: int,
    activated_query_cap: int,
) -> np.ndarray:
    scores = np.zeros(len(candidate_rows), dtype=np.float32)
    frame = candidate_rows.copy()
    frame["_seed_score"] = 0.70 * col_z(frame, "score_final") + 0.30 * col_z(frame, "source_min_rank_inv_safe")
    row_pos = {idx: pos for pos, idx in enumerate(frame.index)}
    for _, group in frame.groupby("query_id", sort=False):
        local = group.sort_values("_seed_score", ascending=False)
        seeds = local["candidate_id"].astype(str).head(seed_k).tolist()
        activated: dict[str, float] = defaultdict(float)
        for rank, seed in enumerate(seeds, start=1):
            for train_query, weight in seed_to_train_queries.get(seed, []):
                activated[train_query] += weight / rank
        activated_items = sorted(activated.items(), key=lambda item: item[1], reverse=True)[:activated_query_cap]
        propagated: dict[str, float] = defaultdict(float)
        norm = 1e-6
        for train_query, q_weight in activated_items:
            norm += q_weight
            for candidate_id, c_weight in train_query_to_positive_candidates.get(train_query, {}).items():
                propagated[candidate_id] += q_weight * c_weight
        for idx, candidate_id in zip(local.index, local["candidate_id"].astype(str)):
            scores[row_pos[idx]] = float(propagated.get(candidate_id, 0.0) / norm)
    return scores


def add_v8_22_online_features(
    candidate_rows: pd.DataFrame,
    seed_to_train_queries: dict[str, list[tuple[str, float]]],
    train_query_to_positive_candidates: dict[str, dict[str, float]],
) -> pd.DataFrame:
    frame = candidate_rows.copy()
    for col in ["source_event_rank", "source_sequence_rank", "source_min_rank", "source_rank_gap"]:
        if col in frame.columns:
            vals = pd.to_numeric(frame[col], errors="coerce")
            frame[f"{col}_inv_safe"] = 1.0 / (1.0 + vals.fillna(9999.0))
    if {"score_final", "score_calibrated"}.issubset(frame.columns):
        final = pd.to_numeric(frame["score_final"], errors="coerce").fillna(0.0)
        cal = pd.to_numeric(frame["score_calibrated"], errors="coerce").fillna(0.0)
        frame["score_final_minus_calibrated"] = final - cal
        frame["score_final_plus_calibrated"] = final + cal
    for name, seed_k, cap in GRAPH_CONFIGS:
        raw = graph_scores(
            frame,
            seed_to_train_queries,
            train_query_to_positive_candidates,
            seed_k=seed_k,
            activated_query_cap=cap,
        )
        frame[name] = raw
        frame[f"{name}_z"] = query_z(frame, raw)
    graph_cols = [name for name, _, _ in GRAPH_CONFIGS]
    frame["graph_max"] = frame[graph_cols].max(axis=1)
    frame["graph_mean"] = frame[graph_cols].mean(axis=1)
    frame["graph_max_z"] = query_z(frame, frame["graph_max"].to_numpy(dtype=float))
    frame["graph_mean_z"] = query_z(frame, frame["graph_mean"].to_numpy(dtype=float))
    return frame


def rank_v8_22_online(
    candidate_rows: pd.DataFrame,
    anchor_top20_rows: pd.DataFrame,
    frozen_lambdarank_model: Any,
    feature_columns: list[str],
    feature_fill_values: pd.Series,
    seed_to_train_queries: dict[str, list[tuple[str, float]]],
    train_query_to_positive_candidates: dict[str, dict[str, float]],
    *,
    frozen_asset_manifest: dict[str, Any] | str | None = None,
) -> pd.DataFrame:
    """Return V8.22 top20 analogues with provenance flags."""
    # Asset identity is checked before graph features or model predictions are computed.
    verify_frozen_asset_bindings(
        frozen_lambdarank_model,
        feature_columns,
        seed_to_train_queries,
        train_query_to_positive_candidates,
        frozen_asset_manifest=frozen_asset_manifest,
    )
    frame = add_v8_22_online_features(candidate_rows, seed_to_train_queries, train_query_to_positive_candidates)
    feature_audit = input_audit_receipt(
        frame,
        feature_columns,
        model=frozen_lambdarank_model,
        seed_to_train_queries=seed_to_train_queries,
        train_query_to_positive_candidates=train_query_to_positive_candidates,
        frozen_asset_manifest=frozen_asset_manifest,
    )
    x = frame[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(feature_fill_values).to_numpy(np.float32)
    model_score = query_z(frame, np.asarray(frozen_lambdarank_model.predict(x), dtype=float))
    blend_score = (
        BLEND["wf"] * col_z(frame, "score_final")
        + BLEND["wm"] * model_score
        + BLEND["wg"] * col_z(frame, "graph_max_z")
        + BLEND["wc"] * col_z(frame, "score_calibrated")
        + BLEND["ws"] * col_z(frame, "source_min_rank_inv_safe")
    )
    proposal = frame.copy()
    proposal["_v8_22_score"] = blend_score
    ranked_pieces: list[pd.DataFrame] = []
    anchor_groups = {str(q): g.sort_values("rank").copy() for q, g in anchor_top20_rows.groupby("query_id", sort=False)}
    for query_id, group in proposal.groupby("query_id", sort=False):
        anchor = anchor_groups.get(str(query_id), pd.DataFrame()).copy()
        ordered: list[str] = []
        for candidate_id in anchor.head(ANCHOR_KEEP)["candidate_id"].astype(str).tolist():
            if candidate_id not in ordered:
                ordered.append(candidate_id)
        for candidate_id in group.sort_values("_v8_22_score", ascending=False)["candidate_id"].astype(str).tolist():
            if len(ordered) >= 20:
                break
            if candidate_id not in ordered:
                ordered.append(candidate_id)
        lookup = {str(r.candidate_id): r._asdict() for r in group.itertuples(index=False)}
        rows = []
        for rank, candidate_id in enumerate(ordered[:20], start=1):
            row = dict(lookup[candidate_id])
            row["rank"] = rank
            row["method_variant"] = METHOD_NAME
            row["method_definition"] = METHOD_DEFINITION
            row["selected_profile"] = SELECTED_PROFILE
            row["uses_cma_gt_labels_for_training"] = True
            row["uses_cma_numeric_at_inference"] = False
            row["uses_test_for_tuning"] = False
            row["fingerprint_input_scope"] = "upstream_precomputed_background_context"
            row["context_candidate_path"] = "upstream"
            row["context_rank_prior_available"] = "score_context_rank_prior" in frame.columns
            row["context_used_for_candidate_generation"] = True
            row["context_rank_prior_used_by_frozen_model"] = feature_audit[
                "context_rank_prior_used_by_frozen_model"
            ]
            row["context_rank_prior_split_count"] = feature_audit["context_rank_prior_split_count"]
            row["context_rank_prior_gain"] = feature_audit["context_rank_prior_gain"]
            row["standalone_direct_context_term_selected"] = False
            row["feature_list_sha256"] = feature_audit["feature_list_sha256"]
            row["model_sha256"] = feature_audit.get("model_sha256")
            row["graph_asset_sha256"] = feature_audit["graph_asset_sha256"]
            row["graph_hash_match"] = feature_audit["graph_hash_match"]
            rows.append(row)
        ranked_pieces.append(pd.DataFrame(rows))
    result = pd.concat(ranked_pieces, ignore_index=True)
    result.attrs["feature_audit"] = feature_audit
    return result
