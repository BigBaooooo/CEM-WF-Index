"""Run the complete public retrieval composition on synthetic data only."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cem_wf_index.context import ContextIndex, FingerprintArchive, run_retrieval_pipeline
from cem_wf_index.final_release.feature_contract import FROZEN_FEATURE_NAMES, feature_list_sha256


class SyntheticFrozenModel:
    """Small deterministic stand-in for an already-frozen scorer."""

    audit_sha256 = "synthetic-frozen-model-sha256"

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float32)
        context_position = list(FROZEN_FEATURE_NAMES).index("score_context_rank_prior")
        return values[:, context_position] if values.shape[1] else np.zeros(len(values), dtype=np.float32)


def _event(event_id: str, group_id: str, start_hour: int, vector_row: int) -> dict[str, Any]:
    start = pd.Timestamp("2020-01-01") + pd.Timedelta(hours=start_hour)
    end = start + pd.Timedelta(hours=24)
    return {
        "event_id": event_id,
        "candidate_id": event_id,
        "group_id": group_id,
        "event_type": "synthetic_weather_event",
        "time_coverage_start": start.isoformat(),
        "time_coverage_end": end.isoformat(),
        "vector_row": vector_row,
    }


def _write_archive(directory: Path) -> FingerprintArchive:
    rng = np.random.default_rng(20262202)
    matrix = rng.normal(size=(96, 1024)).astype(np.float32)
    starts = np.datetime64("2020-01-01T00", "h") + np.arange(96).astype("timedelta64[h]")
    ends = starts + np.timedelta64(24, "h")
    np.save(directory / "all_fp1024.npy", matrix)
    np.save(directory / "start_times.npy", starts)
    np.save(directory / "end_times.npy", ends)
    return FingerprintArchive.from_directory(directory)


def run(*, backend: str = "exact") -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cem_wf_synthetic_") as raw_directory:
        archive = _write_archive(Path(raw_directory))
        try:
            query = _event("query", "storm-query", 0, 0)
            events = [
                _event("candidate-a", "storm-a", 24, 24),
                _event("candidate-b", "storm-b", 30, 30),
                _event("candidate-c", "storm-c", 48, 48),
                _event("candidate-d", "storm-d", 72, 72),
            ]
            event_candidates = []
            for rank, row in enumerate(events, start=1):
                event_candidates.append(
                    {
                        **row,
                        "rank": rank,
                        "distance": float(rank),
                        "provenance": "synthetic_event_channel",
                        "score_final": 1.0 / (rank + 1),
                        "score_calibrated": 1.0 / (rank + 2),
                    }
                )
            metadata_candidates = [
                {
                    **events[2],
                    "rank": 1,
                    "distance": 0.25,
                    "provenance": "synthetic_metadata_channel",
                },
                {
                    **events[3],
                    "rank": 2,
                    "distance": 0.50,
                    "provenance": "synthetic_metadata_channel",
                },
            ]
            index = ContextIndex(backend=backend)
            result = run_retrieval_pipeline(
                archive=archive,
                query_event=query,
                archive_events=[query, *events],
                event_candidates=event_candidates,
                metadata_candidates=metadata_candidates,
                anchor_top20_rows=event_candidates,
                frozen_lambdarank_model=SyntheticFrozenModel(),
                feature_columns=list(FROZEN_FEATURE_NAMES),
                feature_fill_values=pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
                seed_to_train_queries={},
                train_query_to_positive_candidates={},
                candidate_limit=5,
                top_k=3,
                cadence="6h",
                context_index=index,
            )
            return {
                "top_k": result.top_k[
                    ["rank", "candidate_id", "source_membership", "score_context_rank_prior"]
                ].to_dict("records"),
                "audit": result.audit,
                "feature_list_sha256": feature_list_sha256(),
            }
        finally:
            archive.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "hnsw", "exact"], default="exact")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()
    payload = run(backend=args.backend)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
