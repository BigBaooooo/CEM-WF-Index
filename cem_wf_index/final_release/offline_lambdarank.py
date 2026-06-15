"""Run the final label-graph augmented LambdaRank ranker for Typhoon Full.

This script keeps a retained strict-safe anchor ranking intact and trains a real
grouped ranker over the full M1500 candidate rows. The ranker receives only
inference-available priors plus train-label graph features. CMA GT is used as
the train label only; validation selects one profile; test is report-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier, LGBMRanker
except Exception as exc:  # pragma: no cover
    raise RuntimeError("V8.22 requires LightGBM in the cemwf environment.") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in [PROJECT_ROOT, SCRIPTS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cem_wf_index.final_release.physical_errors import typhoon_top1_physical_errors_from_ranked
from cem_wf_index.final_release.graph_features import _col_z, _graph_scores, _query_z, _train_graph
from cem_wf_index.final_release.candidate_rows import TARGET, _anchor_merge, _read_rows, _relevant, _safe_float
from cem_wf_index.final_release.ranking_utils import _metrics, _ranked_from_frame


METHOD_NAME = "CEM-WF-Index v8.22 Label-Graph LambdaRank"
METHOD_DEFINITION = "v8_22_label_graph_lambdarank"
VALIDATION_SPLIT = "validation_context_gate"
TEST_SPLIT = "test_report_only"
PRIMARY_ROOT = Path(os.environ.get("CEMWF_ARTIFACT_ROOT", "artifacts/v8_22"))
SECONDARY_ROOT = Path(os.environ.get("CEMWF_SECONDARY_ARTIFACT_ROOT", "artifacts_secondary/v8_22"))
MIN_FREE_BYTES = 10 * 1024**3
# Retained-anchor reference values are used only for reproduction sanity checks,
# not for training, validation profile selection, or inference.
RETAINED_ANCHOR_BASELINE = {
    "nDCG@10": 0.8695522108335824,
    "mAP@10": 0.8023448773448774,
    "Recall@10": 0.4227272727272727,
    "Recall@20": 0.6218181818181817,
    "Track RMSE/km": 770.0902883244523,
    "Landfall dist/km": 671.8737125003018,
}
GRAPH_CONFIGS = [
    ("g18_seed20_cap200", 20, 200),
    ("g19_seed16_cap200", 16, 200),
    ("gwide_seed20_cap500", 20, 500),
    ("gdeep_seed40_cap500", 40, 500),
]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), **payload}, sort_keys=True) + "\n")


def _check_space(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return {"absolute_path": str(path), "free_gb": round(usage.free / 1024**3, 3), "above_10gb_floor": usage.free >= MIN_FREE_BYTES}


def _sort_for_group(frame: pd.DataFrame) -> pd.DataFrame:
    cols = ["query_id", "source_min_rank"]
    existing = [col for col in cols if col in frame.columns]
    return frame.sort_values(existing, kind="mergesort").reset_index(drop=True) if existing else frame.reset_index(drop=True)


def _group_sizes(frame: pd.DataFrame) -> list[int]:
    return [int(size) for size in frame.groupby("query_id", sort=False).size().tolist()]


def _add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["source_event_rank", "source_sequence_rank", "source_min_rank", "source_rank_gap"]:
        if col in out.columns:
            vals = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_inv_safe"] = 1.0 / (1.0 + vals.fillna(9999.0))
    if {"score_final", "score_calibrated"}.issubset(out.columns):
        final = pd.to_numeric(out["score_final"], errors="coerce").fillna(0.0)
        cal = pd.to_numeric(out["score_calibrated"], errors="coerce").fillna(0.0)
        out["score_final_minus_calibrated"] = final - cal
        out["score_final_plus_calibrated"] = final + cal
    return out


def _add_graph_features(frame: pd.DataFrame, seed_to_queries: dict[str, list[tuple[str, float]]], query_to_pos: dict[str, dict[str, float]]) -> pd.DataFrame:
    out = frame.copy()
    for name, seed_k, cap in GRAPH_CONFIGS:
        raw = _graph_scores(out, seed_to_queries, query_to_pos, seed_k=seed_k, activated_query_cap=cap)
        out[name] = raw
        out[f"{name}_z"] = _query_z(out, raw)
    graph_cols = [name for name, _, _ in GRAPH_CONFIGS]
    out["graph_max"] = out[graph_cols].max(axis=1)
    out["graph_mean"] = out[graph_cols].mean(axis=1)
    out["graph_max_z"] = _query_z(out, out["graph_max"].to_numpy(dtype=float))
    out["graph_mean_z"] = _query_z(out, out["graph_mean"].to_numpy(dtype=float))
    return out


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    forbidden_exact = {
        "query_id",
        "candidate_id",
        "task",
        "split_role",
        "is_gt_top20",
        "rank",
        "grade",
        "method_variant",
        "selected_model",
        "candidate_pool",
        "feature_profile",
        "tiebreaker_profile",
        "tail_profile",
        "method_definition",
        "context_only_claim_status",
        "pool_source",
        "v7_4_merge_source",
    }
    cols: list[str] = []
    for col in frame.columns:
        lower = col.lower()
        if col in forbidden_exact or lower.startswith("uses_") or "cma_numeric" in lower:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def _matrix(frame: pd.DataFrame, cols: list[str], fill: pd.Series | None = None) -> tuple[np.ndarray, pd.Series]:
    numeric = frame[cols].apply(pd.to_numeric, errors="coerce")
    if fill is None:
        fill = numeric.median(numeric_only=True).fillna(0.0)
    x = numeric.fillna(fill).to_numpy(dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), fill


def _labels(frame: pd.DataFrame, mode: str) -> np.ndarray:
    is_pos = frame["is_gt_top20"].astype(bool).to_numpy()
    if mode == "graded" and "grade" in frame.columns:
        grade = pd.to_numeric(frame["grade"], errors="coerce").fillna(0.0).clip(lower=0, upper=10).astype(int).to_numpy()
        return np.where(is_pos, np.maximum(grade, 1), 0).astype(int)
    return is_pos.astype(int)


def _rank_with_score(frame: pd.DataFrame, score: np.ndarray, profile: str) -> pd.DataFrame:
    out = frame.copy()
    out["v8_22_score"] = score
    out["v8_22_profile"] = profile
    pieces: list[pd.DataFrame] = []
    for _, group in out.groupby("query_id", sort=False):
        g = group.sort_values("v8_22_score", ascending=False).head(20).copy()
        g["rank"] = np.arange(1, len(g) + 1)
        g["method_variant"] = METHOD_NAME
        g["method_definition"] = METHOD_DEFINITION
        g["uses_cma_gt_labels_for_training"] = True
        g["uses_cma_numeric_at_inference"] = False
        g["uses_test_for_tuning"] = False
        g["trains_new_fingerprint"] = False
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True)


def _blend_scores(rows: pd.DataFrame, model_score: np.ndarray) -> list[tuple[str, np.ndarray]]:
    model = _query_z(rows, model_score)
    final = _col_z(rows, "score_final")
    cal = _col_z(rows, "score_calibrated")
    source = _col_z(rows, "source_min_rank_inv_safe") if "source_min_rank_inv_safe" in rows.columns else _col_z(rows, "source_min_rank_inv")
    graph = _col_z(rows, "graph_max_z")
    profiles: list[tuple[str, np.ndarray]] = [("lambdarank_only", model)]
    for wm in [0.35, 0.45, 0.55, 0.65]:
        for wg in [0.10, 0.20, 0.30]:
            for wc in [0.05, 0.15, 0.25]:
                for ws in [0.0, 0.06]:
                    wf = 1.0 - wm - wg - wc - ws
                    if wf < 0:
                        continue
                    name = f"blend_f{wf:.2f}_m{wm:.2f}_g{wg:.2f}_c{wc:.2f}_s{ws:.2f}"
                    profiles.append((name, wf * final + wm * model + wg * graph + wc * cal + ws * source))
    return profiles


def _models() -> list[tuple[str, str, Any]]:
    return [
        (
            "lgbm_ranker_bin_leaves31_lr04",
            "binary",
            LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=420,
                learning_rate=0.04,
                num_leaves=31,
                min_child_samples=45,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=20262201,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
        (
            "lgbm_ranker_grade_leaves63_lr03",
            "graded",
            LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=520,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=35,
                subsample=0.90,
                colsample_bytree=0.90,
                random_state=20262202,
                n_jobs=-1,
                verbose=-1,
                label_gain=list(range(0, 12)),
            ),
        ),
        (
            "lgbm_binary_graph_balanced",
            "binary_classifier",
            LGBMClassifier(
                objective="binary",
                n_estimators=520,
                learning_rate=0.035,
                num_leaves=63,
                min_child_samples=40,
                subsample=0.90,
                colsample_bytree=0.90,
                class_weight="balanced",
                random_state=20262203,
                n_jobs=-1,
                verbose=-1,
            ),
        ),
    ]


def _score_model(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _composite(metrics: dict[str, float]) -> float:
    return 0.30 * metrics["nDCG@10"] + 0.30 * metrics["mAP@10"] + 0.25 * metrics["Recall@20"] + 0.10 * metrics["Recall@10"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidate-detail-dir", default=None, help="Directory containing train/validation/test candidate row CSV files.")
    parser.add_argument("--retained-validation-ranking", default=None, help="Retained anchor validation ranking detail CSV.")
    parser.add_argument("--retained-test-ranking", default=None, help="Retained anchor test ranking detail CSV.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    run_id = f"v8_22_label_graph_lambdarank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    details_dir = root / "outputs" / "v8_22_migration" / "table_details"
    reports_dir = root / "outputs" / "v8_22_migration" / "reports"
    details_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = PRIMARY_ROOT / "progress"
    progress_path = progress_dir / "progress.json"
    events_path = progress_dir / "events.jsonl"
    storage = {"run_id": run_id, "primary": _check_space(progress_dir), "secondary": _check_space(SECONDARY_ROOT / "split_cache"), "minimum_free_gb": 10}
    _write_json(details_dir / "storage_registry_v8_22.json", storage)
    _write_json(progress_path, {"stage": "start", "run_id": run_id, "storage": storage})
    _append_event(events_path, {"stage": "start", "run_id": run_id})

    candidate_details = Path(args.candidate_detail_dir).resolve() if args.candidate_detail_dir else root / "outputs" / "v8_migration" / "table_details"
    train = _sort_for_group(_read_rows(candidate_details / "v8_train_candidate_rows_sample.csv"))
    validation = _sort_for_group(_read_rows(candidate_details / "v8_validation_candidate_rows.csv"))
    test = _sort_for_group(_read_rows(candidate_details / "v8_test_candidate_rows.csv"))
    seed_to_queries, query_to_pos = _train_graph(train)
    graph_manifest = {
        "train_positive_seed_count": len(seed_to_queries),
        "train_positive_query_count": len(query_to_pos),
        "uses_cma_gt_labels_for_training": True,
        "uses_cma_numeric_at_inference": False,
    }
    _write_json(details_dir / "typhoon_v8_22_train_label_graph_manifest.json", graph_manifest)

    _write_json(progress_path, {"stage": "add_graph_features", "split": "train", "run_id": run_id})
    train_f = _add_graph_features(_add_derived_features(train), seed_to_queries, query_to_pos)
    _write_json(progress_path, {"stage": "add_graph_features", "split": "validation", "run_id": run_id})
    val_f = _add_graph_features(_add_derived_features(validation), seed_to_queries, query_to_pos)
    _write_json(progress_path, {"stage": "add_graph_features", "split": "test", "run_id": run_id})
    test_f = _add_graph_features(_add_derived_features(test), seed_to_queries, query_to_pos)

    feature_cols = _feature_columns(train_f)
    x_train, fill = _matrix(train_f, feature_cols)
    x_val, _ = _matrix(val_f, feature_cols, fill=fill)
    x_test, _ = _matrix(test_f, feature_cols, fill=fill)
    pd.DataFrame({"feature": feature_cols}).to_csv(details_dir / "typhoon_v8_22_feature_names.csv", index=False)
    train_groups = _group_sizes(train_f)
    rel_val = _relevant(val_f)
    rel_test = _relevant(test_f)
    retained_validation_path = Path(args.retained_validation_ranking).resolve() if args.retained_validation_ranking else root / "outputs" / "retained_anchor" / "table_details" / "typhoon_validation_ranking_detail.csv"
    retained_test_path = Path(args.retained_test_ranking).resolve() if args.retained_test_ranking else root / "outputs" / "retained_anchor" / "table_details" / "typhoon_test_ranking_detail.csv"
    base_val = pd.read_csv(retained_validation_path, low_memory=False)
    base_test = pd.read_csv(retained_test_path, low_memory=False)

    rows: list[dict[str, Any]] = []
    selected_frames: dict[str, tuple[pd.DataFrame, np.ndarray, str, int]] = {}
    for model_name, label_mode, model in _models():
        _write_json(progress_path, {"stage": "fit_model", "model": model_name, "label_mode": label_mode, "ledger_rows": len(rows)})
        y_train = _labels(train_f, "graded" if label_mode == "graded" else "binary")
        if isinstance(model, LGBMRanker):
            model.fit(x_train, y_train, group=train_groups)
        else:
            model.fit(x_train, train_f["is_gt_top20"].astype(int).to_numpy())
        val_model_score = _score_model(model, x_val)
        for blend_name, val_score in _blend_scores(val_f, val_model_score):
            proposal = _rank_with_score(val_f, val_score, f"{model_name}|{blend_name}")
            for keep_top in [0, 1, 3, 5]:
                profile = f"{model_name}|{blend_name}|anchor_keep{keep_top}"
                val_ranked = _anchor_merge(base_val, proposal, keep_top=keep_top, tail_from=21, profile=profile)
                metrics = _metrics(_ranked_from_frame(val_ranked), rel_val)
                row = {
                    "profile": profile,
                    "model_name": model_name,
                    "label_mode": label_mode,
                    "blend_name": blend_name,
                    "keep_top": keep_top,
                    "validation_nDCG@10": metrics["nDCG@10"],
                    "validation_mAP@10": metrics["mAP@10"],
                    "validation_Recall@10": metrics["Recall@10"],
                    "validation_Recall@20": metrics["Recall@20"],
                    "validation_composite": _composite(metrics),
                    "uses_cma_gt_labels_for_training": True,
                    "uses_cma_numeric_at_inference": False,
                    "uses_test_for_tuning": False,
                    "trains_new_fingerprint": False,
                    "method_definition": METHOD_DEFINITION,
                }
                rows.append(row)
        pd.DataFrame(rows).to_csv(details_dir / "typhoon_v8_22_lambdarank_search.partial.csv", index=False)
        selected_frames[model_name] = (model, val_model_score, label_mode, len(rows))

    search = pd.DataFrame(rows).sort_values(["validation_composite", "validation_Recall@20"], ascending=False)
    search.to_csv(details_dir / "typhoon_v8_22_lambdarank_search.csv", index=False)
    candidates = search[
        (pd.to_numeric(search["validation_nDCG@10"], errors="coerce") >= 0.86)
        & (pd.to_numeric(search["validation_mAP@10"], errors="coerce") >= 0.80)
        & (pd.to_numeric(search["validation_Recall@10"], errors="coerce") >= 0.41)
        & (pd.to_numeric(search["validation_Recall@20"], errors="coerce") >= 0.735)
        & (pd.to_numeric(search["keep_top"], errors="coerce") >= 1)
    ].copy()
    if candidates.empty:
        candidates = search[pd.to_numeric(search["keep_top"], errors="coerce") >= 1].copy()
    selected = candidates.sort_values(["validation_Recall@20", "validation_composite", "validation_mAP@10"], ascending=False).iloc[0].to_dict()
    _write_json(progress_path, {"stage": "selected_validation_profile", "selected": selected})

    selected_model_name = str(selected["model_name"])
    model, _, _, _ = selected_frames[selected_model_name]
    test_model_score = _score_model(model, x_test)
    val_model_score = _score_model(model, x_val)
    selected_val_score = None
    selected_test_score = None
    for blend_name, score in _blend_scores(val_f, val_model_score):
        if blend_name == selected["blend_name"]:
            selected_val_score = score
            break
    for blend_name, score in _blend_scores(test_f, test_model_score):
        if blend_name == selected["blend_name"]:
            selected_test_score = score
            break
    if selected_val_score is None or selected_test_score is None:
        raise RuntimeError("Selected blend score was not reconstructed.")
    selected_profile = str(selected["profile"])
    keep_top = int(selected["keep_top"])
    val_proposal = _rank_with_score(val_f, selected_val_score, selected_profile)
    test_proposal = _rank_with_score(test_f, selected_test_score, selected_profile)
    selected_val = _anchor_merge(base_val, val_proposal, keep_top=keep_top, tail_from=21, profile=selected_profile)
    selected_test = _anchor_merge(base_test, test_proposal, keep_top=keep_top, tail_from=21, profile=selected_profile)
    selected_val.to_csv(details_dir / "v8_22_typhoon_validation_ranking_detail.csv", index=False)
    selected_test.to_csv(details_dir / "v8_22_typhoon_full_ranking_detail.csv", index=False)

    source_dir = root / "outputs" / "v6_migration"
    source_details = source_dir / "table_details"
    descriptor_eval = pd.read_csv(source_details / "typhoon_proxy_descriptor_v6.csv", low_memory=False)
    test_ranked = _ranked_from_frame(selected_test)
    test_metrics = _metrics(test_ranked, rel_test)
    test_physical = typhoon_top1_physical_errors_from_ranked(root, source_details, descriptor_eval, test_ranked)
    summary = {
        "run_id": run_id,
        "method_name": METHOD_NAME,
        "method_definition": METHOD_DEFINITION,
        "selected_profile": selected_profile,
        "selected_by_validation": True,
        "validation_nDCG@10": selected.get("validation_nDCG@10", ""),
        "validation_mAP@10": selected.get("validation_mAP@10", ""),
        "validation_Recall@10": selected.get("validation_Recall@10", ""),
        "validation_Recall@20": selected.get("validation_Recall@20", ""),
        "validation_composite": selected.get("validation_composite", ""),
        "test_nDCG@10_report_only": test_metrics["nDCG@10"],
        "test_mAP@10_report_only": test_metrics["mAP@10"],
        "test_Recall@10_report_only": test_metrics["Recall@10"],
        "test_Recall@20_report_only": test_metrics["Recall@20"],
        **test_physical,
        "retained_anchor_nDCG@10": RETAINED_ANCHOR_BASELINE["nDCG@10"],
        "retained_anchor_mAP@10": RETAINED_ANCHOR_BASELINE["mAP@10"],
        "retained_anchor_Recall@10": RETAINED_ANCHOR_BASELINE["Recall@10"],
        "retained_anchor_Recall@20": RETAINED_ANCHOR_BASELINE["Recall@20"],
        "strict_promotion_pass": bool(
            test_metrics["nDCG@10"] >= RETAINED_ANCHOR_BASELINE["nDCG@10"]
            and test_metrics["mAP@10"] >= RETAINED_ANCHOR_BASELINE["mAP@10"]
            and test_metrics["Recall@10"] >= RETAINED_ANCHOR_BASELINE["Recall@10"]
            and test_metrics["Recall@20"] >= RETAINED_ANCHOR_BASELINE["Recall@20"]
            and _safe_float(test_physical.get("Track RMSE/km")) <= RETAINED_ANCHOR_BASELINE["Track RMSE/km"]
            and _safe_float(test_physical.get("Landfall dist/km")) <= RETAINED_ANCHOR_BASELINE["Landfall dist/km"]
        ),
        "hard_a_target_pass": bool(
            test_metrics["nDCG@10"] >= TARGET["nDCG@10"]
            and test_metrics["mAP@10"] >= TARGET["mAP@10"]
            and test_metrics["Recall@10"] >= TARGET["Recall@10"]
            and test_metrics["Recall@20"] >= TARGET["Recall@20"]
            and _safe_float(test_physical.get("Track RMSE/km")) < TARGET["Track RMSE/km"]
            and _safe_float(test_physical.get("Landfall dist/km")) < TARGET["Landfall dist/km"]
        ),
        "feature_count": len(feature_cols),
        "graph_manifest": graph_manifest,
        "test_grid_policy": "test ranking computed only after validation-selected model/blend/anchor profile",
        "uses_cma_gt_labels_for_training": True,
        "uses_cma_numeric_at_inference": False,
        "uses_test_for_tuning": False,
        "trains_new_fingerprint": False,
        "context_only_claim_status": "pending_not_used_by_full_context_weight_0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    pd.DataFrame([summary]).to_csv(details_dir / "typhoon_v8_22_lambdarank_selected_summary.csv", index=False)
    _write_json(reports_dir / "v8_22_lambdarank_selected_summary.json", summary)
    _write_json(progress_path, {"stage": "complete", "summary": summary})
    _append_event(events_path, {"stage": "complete", "summary": summary})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
