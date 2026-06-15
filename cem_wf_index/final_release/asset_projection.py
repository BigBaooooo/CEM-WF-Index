"""Project regenerated final ranking outputs into paper-facing assets.

This optional script is intentionally source-only: it expects users to provide
locally regenerated summary/detail files and a strict-table template directory.
It does not ship or require generated result files in this repository.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


FINAL_METHOD_NAME = "CEM-WF-Index Label-Graph LambdaRank"
HEATWAVE_PUBLIC_METHOD = "CEM-WF-Index Heatwave Full-Candidate"
HEATWAVE_SOURCE_LINEAGE = "retained_heatwave_full_candidate_result"
ABLATION_RETAINED_BASELINE = "Retained Full-Candidate baseline"
TARGET = {
    "nDCG@10": 0.82,
    "mAP@10": 0.72,
    "Recall@10": 0.40,
    "Recall@20": 0.64,
    "Track RMSE/km": 850.0,
    "Landfall dist/km": 750.0,
}


def _read_one(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).iloc[0].to_dict()


def _copy_strict_tables(src_strict: Path, dst_strict: Path) -> None:
    if not src_strict.exists():
        raise FileNotFoundError(src_strict)
    dst_strict.mkdir(parents=True, exist_ok=True)
    for src in src_strict.glob("*.csv"):
        shutil.copy2(src, dst_strict / src.name)


def _update_typhoon_main(dst_strict: Path, summary: dict[str, object]) -> None:
    candidates = sorted(dst_strict.glob("*_typhoon_main.csv"))
    if not candidates:
        raise FileNotFoundError("No typhoon main strict table found in template output.")
    path = candidates[0]
    frame = pd.read_csv(path)
    method_col = "Typhoon Method"
    mask = frame[method_col].astype(str).str.contains("CEM-WF-Index", na=False)
    frame.loc[mask, method_col] = FINAL_METHOD_NAME
    mapping = {
        "Recall@10": "test_Recall@10_report_only",
        "Recall@20": "test_Recall@20_report_only",
        "nDCG@10": "test_nDCG@10_report_only",
        "mAP@10": "test_mAP@10_report_only",
        "Track RMSE/km": "Track RMSE/km",
        "PRES MAE/hPa": "PRES MAE/hPa",
        "WND MAE/m/s": "WND MAE/m/s",
        "Landfall dist/km": "Landfall dist/km",
    }
    for column, key in mapping.items():
        if column in frame.columns and key in summary:
            frame.loc[mask, column] = summary[key]
    frame.to_csv(path, index=False)


def _clean_public_method_labels(dst_strict: Path) -> None:
    for heat_path in dst_strict.glob("*_heatwave_main.csv"):
        heat = pd.read_csv(heat_path)
        heat_col = "Heatwave Method"
        if heat_col in heat.columns:
            heat_mask = heat[heat_col].astype(str).str.contains("Full-Candidate", na=False)
            heat.loc[heat_mask, heat_col] = HEATWAVE_PUBLIC_METHOD
            heat.to_csv(heat_path, index=False)
    for ablation_path in dst_strict.glob("*_ablation.csv"):
        ablation = pd.read_csv(ablation_path)
        variant_col = "Variant"
        if variant_col in ablation.columns:
            baseline_mask = ablation[variant_col].astype(str).str.contains("Full-Candidate", na=False)
            ablation.loc[baseline_mask, variant_col] = ABLATION_RETAINED_BASELINE
            ablation.to_csv(ablation_path, index=False)


def _hard_target_status(summary: dict[str, object]) -> dict[str, bool]:
    return {
        "nDCG@10": float(summary["test_nDCG@10_report_only"]) >= TARGET["nDCG@10"],
        "mAP@10": float(summary["test_mAP@10_report_only"]) >= TARGET["mAP@10"],
        "Recall@10": float(summary["test_Recall@10_report_only"]) >= TARGET["Recall@10"],
        "Recall@20": float(summary["test_Recall@20_report_only"]) >= TARGET["Recall@20"],
        "Track RMSE/km": float(summary["Track RMSE/km"]) < TARGET["Track RMSE/km"],
        "Landfall dist/km": float(summary["Landfall dist/km"]) < TARGET["Landfall dist/km"],
    }


def _write_reports(dst_reports: Path, summary: dict[str, object], baseline: dict[str, object] | None) -> None:
    dst_reports.mkdir(parents=True, exist_ok=True)
    status = _hard_target_status(summary)
    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "method_name": FINAL_METHOD_NAME,
        "hard_a_target_pass": all(status.values()),
        "hard_a_target_status": status,
        "a_targets": TARGET,
        "selected_typhoon_full": summary,
        "retained_anchor_baseline": baseline or {},
        "uses_cma_gt_labels_for_training": True,
        "uses_cma_numeric_at_inference": False,
        "uses_test_for_tuning": False,
        "trains_new_fingerprint": False,
        "context_only_claim_status": "pending_not_used_by_full_context_weight_0",
    }
    (dst_reports / "final_s8_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame([audit]).to_csv(dst_reports / "final_s8_audit.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-strict-dir", required=True, type=Path)
    parser.add_argument("--selected-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--retained-anchor-summary", default=None, type=Path)
    args = parser.parse_args()

    dst = args.output_dir.resolve()
    dst_strict = dst / "tables" / "pdf_strict_display"
    dst_details = dst / "table_details"
    dst_reports = dst / "reports"
    dst_details.mkdir(parents=True, exist_ok=True)

    summary = _read_one(args.selected_summary)
    baseline = _read_one(args.retained_anchor_summary) if args.retained_anchor_summary else None
    _copy_strict_tables(args.template_strict_dir, dst_strict)
    _update_typhoon_main(dst_strict, summary)
    _clean_public_method_labels(dst_strict)
    shutil.copy2(args.selected_summary, dst_details / "final_lambdarank_selected_summary.csv")
    if args.retained_anchor_summary:
        shutil.copy2(args.retained_anchor_summary, dst_details / "retained_anchor_selected_summary.csv")
    _write_reports(dst_reports, summary, baseline)
    print(json.dumps({"output": str(dst), "hard_target_status": _hard_target_status(summary)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
