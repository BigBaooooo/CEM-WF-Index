"""Guard the frozen final ranker's numerical and anchor behavior."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cem_wf_index.final_release.online_inference import ANCHOR_KEEP, BLEND, rank_v8_22_online
from cem_wf_index.final_release.feature_contract import FROZEN_FEATURE_NAMES, build_frozen_asset_manifest


class _LinearModel:
    audit_sha256 = "synthetic-linear-model-sha256"
    audit_num_feature = len(FROZEN_FEATURE_NAMES)
    audit_context_split_count = 3
    audit_context_gain = 2.0

    @staticmethod
    def predict(values: np.ndarray) -> np.ndarray:
        context = list(FROZEN_FEATURE_NAMES).index("score_context_rank_prior")
        final = list(FROZEN_FEATURE_NAMES).index("score_final")
        return 0.55 * values[:, context] - 0.25 * values[:, final]


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    pieces = []
    anchors = []
    for query_index, query_id in enumerate(("synthetic-query-a", "synthetic-query-b")):
        rng = np.random.default_rng(20262202 + query_index)
        count = 30
        ids = [f"synthetic-candidate-{query_index}-{i:02d}" for i in range(count)]
        frame = pd.DataFrame(
                {
                    "query_id": query_id,
                    "candidate_id": ids,
                    "score_final": rng.normal(size=count),
                    "score_calibrated": rng.normal(size=count),
                    "source_min_rank_inv_safe": 1.0 / (1.0 + np.arange(count)),
                    "score_context_rank_prior": 1.0 / (2.0 + np.arange(count)),
                }
            )
        for position, feature in enumerate(FROZEN_FEATURE_NAMES):
            if feature not in frame:
                frame[feature] = rng.normal(size=count) if position < 19 else 0.0
        pieces.append(frame)
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


def _manifest(model: _LinearModel) -> dict:
    return build_frozen_asset_manifest(model, {}, {})


def test_frozen_weights_anchor_and_structured_metadata_are_stable():
    candidates, anchor = _fixture()
    ranked = rank_v8_22_online(
        candidates,
        anchor,
        _LinearModel(),
        list(FROZEN_FEATURE_NAMES),
        pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
        {},
        {},
        frozen_asset_manifest=_manifest(_LinearModel()),
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
    assert ranked["context_rank_prior_used_by_frozen_model"].all()
    assert ~ranked["standalone_direct_context_term_selected"].all()
    assert "context_only_claim_status" not in ranked.columns


def test_context_rank_prior_is_consumed_beyond_the_frozen_anchor_head():
    candidates, anchor = _fixture()
    baseline = rank_v8_22_online(
        candidates,
        anchor,
        _LinearModel(),
        list(FROZEN_FEATURE_NAMES),
        pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
        {},
        {},
        frozen_asset_manifest=_manifest(_LinearModel()),
    )
    changed = candidates.copy()
    changed["score_context_rank_prior"] = changed.groupby("query_id", sort=False)[
        "score_context_rank_prior"
    ].transform(lambda values: values.iloc[::-1].to_numpy())
    treatment = rank_v8_22_online(
        changed,
        anchor,
        _LinearModel(),
        list(FROZEN_FEATURE_NAMES),
        pd.Series(0.0, index=list(FROZEN_FEATURE_NAMES)),
        {},
        {},
        frozen_asset_manifest=_manifest(_LinearModel()),
    )
    for query_id in baseline["query_id"].unique():
        base_ids = baseline[baseline["query_id"].eq(query_id)].sort_values("rank")["candidate_id"].tolist()
        changed_ids = treatment[treatment["query_id"].eq(query_id)].sort_values("rank")["candidate_id"].tolist()
        assert base_ids[:5] == changed_ids[:5]
        assert base_ids[5:] != changed_ids[5:]
