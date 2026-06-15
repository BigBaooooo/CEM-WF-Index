from pathlib import Path


def test_release_contains_no_generated_outputs():
    root = Path(__file__).resolve().parents[1]
    forbidden_dirs = {"outputs", "data", "figures", "tables", "bundle", "cache", "progress"}
    present = {p.name for p in root.iterdir() if p.is_dir()}
    assert not (present & forbidden_dirs)


def test_online_reference_policy_flags():
    from online.final_online_inference_reference import METHOD_DEFINITION, SELECTED_PROFILE

    assert METHOD_DEFINITION == "v8_22_label_graph_lambdarank"
    assert "anchor_keep5" in SELECTED_PROFILE


def test_no_machine_specific_paths_in_readme():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    drive_tokens = [letter + ":" + "\\" for letter in "DEFG"]
    assert all(token not in readme for token in drive_tokens)
