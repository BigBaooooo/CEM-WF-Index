from pathlib import Path
import re


def test_release_contains_no_generated_outputs():
    root = Path(__file__).resolve().parents[1]
    forbidden_dirs = {"outputs", "data", "figures", "tables", "bundle", "cache", "progress"}
    present = {p.name for p in root.iterdir() if p.is_dir()}
    assert not (present & forbidden_dirs)


def test_online_reference_policy_flags():
    import numpy as np
    import pandas as pd

    from online.final_online_inference_reference import METHOD_DEFINITION, SELECTED_PROFILE, rank_v8_22_online

    assert METHOD_DEFINITION == "v8_22_label_graph_lambdarank"
    assert "anchor_keep5" in SELECTED_PROFILE

    class _ZeroModel:
        @staticmethod
        def predict(values):
            return np.zeros(len(values), dtype=np.float32)

    candidates = pd.DataFrame(
        {
            "query_id": ["q"] * 20,
            "candidate_id": [f"c{i:02d}" for i in range(20)],
            "score_final": np.linspace(1.0, 0.0, 20),
            "score_calibrated": np.linspace(0.0, 1.0, 20),
            "source_min_rank_inv_safe": np.linspace(1.0, 0.1, 20),
            "score_context_rank_prior": np.linspace(1.0, 0.1, 20),
        }
    )
    anchor = candidates[["query_id", "candidate_id"]].copy()
    anchor["rank"] = np.arange(1, 21)
    ranked = rank_v8_22_online(
        candidates,
        anchor,
        _ZeroModel(),
        ["score_final"],
        pd.Series({"score_final": 0.0}),
        {},
        {},
    )
    assert ranked["context_candidate_path"].eq("upstream").all()
    assert ranked["context_rank_prior_available"].all()
    assert ranked["direct_context_score_status"].eq("not_selected_by_validation").all()
    assert "context_only_claim_status" not in ranked.columns


def test_no_machine_specific_paths_in_readme():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    drive_tokens = [letter + ":" + "\\" for letter in "DEFG"]
    assert all(token not in readme for token in drive_tokens)


def test_public_text_has_no_internal_or_projection_markers():
    root = Path(__file__).resolve().parents[1]
    blocked = [
        "projected_new_fingerprint",
        "pending_not_used_by_full_context_weight_0",
        "trains_new_fingerprint",
        "combined_formal_",
        "improved_fingerprint",
        "legacy_fingerprint",
        "staged_only",
    ]
    drive_path = re.compile(r"\b[A-Za-z]:[\\/]")
    this_file = Path(__file__).resolve()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".py", ".md", ".json", ".txt", ".csv"}:
            continue
        if path.resolve() == this_file:
            continue
        text = path.read_text(encoding="utf-8")
        assert not drive_path.search(text), path
        for token in blocked:
            assert token not in text, (path, token)


def test_published_evidence_uses_only_anonymous_query_ids():
    import pandas as pd

    root = Path(__file__).resolve().parents[1] / "supplementary" / "evidence"
    expected_prefixes = {
        "climatenet_portability_per_query.csv": "climate_query_",
        "matched_scoring_per_query.csv": "matched_query_",
        "physical_metric_per_query.csv": "physical_query_",
    }
    for filename, prefix in expected_prefixes.items():
        frame = pd.read_csv(root / filename)
        assert frame["query_id"].is_unique
        assert frame["query_id"].str.fullmatch(rf"{prefix}\d{{3}}").all()
