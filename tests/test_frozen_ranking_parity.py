"""Guard the frozen final ranker's numerical and anchor behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cem_wf_index.final_release.online_inference import ANCHOR_KEEP, BLEND, rank_v8_22_online


class _LinearModel:
    @staticmethod
    def predict(values: np.ndarray) -> np.ndarray:
        return 0.55 * values[:, 0] - 0.25 * values[:, 1]


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    anchors = []
    for query_index, query_id in enumerate(("synthetic-query-a", "synthetic-query-b")):
        rng = np.random.default_rng(20262202 + query_index)
        count = 30
        ids = [f"synthetic-candidate-{query_index}-{i:02d}" for i in range(count)]
        pieces.append(
            pd.DataFrame(
                {
                    "query_id": query_id,
                    "candidate_id": ids,
                    "score_final": rng.normal(size=count),
                    "score_calibrated": rng.normal(size=count),
                    "source_min_rank_inv_safe": 1.0 / (1.0 + np.arange(count)),
                    "score_context_rank_prior": 1.0 / (2.0 + np.arange(count)),
                    "feature_a": rng.normal(size=count),
                    "feature_b": rng.normal(size=count),
                }
            )
        )
        anchor_order = rng.permutation(count)
        anchors.append(
            pd.DataFrame(
                {
                    "query_id": query_id,
                    "candidate_id": [ids[index] for index in anchor_order[:20]],
                    "rank": np.arange(1, 21),
                }
            )
        )
    return pd.concat(pieces, ignore_index=True), pd.concat(anchors, ignore_index=True)


def test_frozen_weights_anchor_and_structured_metadata_are_stable():
    candidates, anchor = _fixture()
    ranked = rank_v8_22_online(
        candidates,
        anchor,
        _LinearModel(),
        ["feature_a", "feature_b"],
        pd.Series({"feature_a": 0.0, "feature_b": 0.0}),
        {},
        {},
    )

    assert BLEND == {"wf": 0.10, "wm": 0.45, "wg": 0.30, "wc": 0.15, "ws": 0.00}
    assert ANCHOR_KEEP == 5
    for query_id, group in ranked.groupby("query_id", sort=False):
        expected_anchor = (
            anchor[anchor["query_id"].eq(query_id)].sort_values("rank").head(5)["candidate_id"].tolist()
        )
        assert group.sort_values("rank").head(5)["candidate_id"].tolist() == expected_anchor
        assert len(group) == 20
        assert group["candidate_id"].is_unique

    assert ranked["context_candidate_path"].eq("upstream").all()
    assert ranked["context_rank_prior_available"].all()
    assert ranked["direct_context_score_status"].eq("not_selected_by_validation").all()
    assert "context_only_claim_status" not in ranked.columns
