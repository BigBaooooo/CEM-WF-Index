# Source Guide

This guide maps the final CEM-WF-Index paper method to the Python source files in this repository.

## Final Package

The implementation lives in `cem_wf_index/final_release/`.

- `offline_lambdarank.py` is the main offline algorithm. It loads regenerated candidate-detail rows, builds graph-propagation features from train-split labels, trains grouped LightGBM LambdaRank models, performs validation-only model/profile selection, and writes report-only test outputs.
- `online_inference.py` is the final inference reference. It computes inference-available candidate priors, activates the frozen train-label graph, builds graph features, applies the frozen ranker/blend profile, preserves anchor candidates, and returns the top-20 analogue set.
- `graph_features.py` implements query-normalized score transforms and train-label graph propagation.
- `candidate_rows.py` implements candidate-row normalization, relevance-set extraction, target gates, and anchor-preserving merge logic.
- `ranking_utils.py` implements Recall, nDCG, mAP, and ranked-list conversion helpers.
- `physical_errors.py` computes evaluation-only typhoon physical consistency metrics from CMA records. These metrics are not used as online inference features.
- `asset_projection.py` optionally projects regenerated outputs into paper-facing asset directories after local reproduction.

## CLI Wrappers

The `scripts/` files are intentionally thin wrappers:

```bash
python scripts/run_final_label_graph_lambdarank.py --help
python scripts/project_final_lambdarank_assets.py --help
```

They call the corresponding `cem_wf_index.final_release` modules. The algorithm logic is not hidden in the scripts; it is in the package.

## Online Wrapper

`online/final_online_inference_reference.py` re-exports `cem_wf_index.final_release.online_inference` for readers who want a direct online-reference path.

## Reproduction Boundary

The repository contains source code only. It does not include the meteorological data, candidate tables, trained models, final paper tables, figures, or generated bundle. This is deliberate: reviewers can inspect the final algorithm implementation here, and users with prepared data can rerun the pipeline using the provided entrypoints.

## Leakage Boundary

The final method uses train-split CMA ground-truth labels for supervised calibration. Validation is used for selecting the model/profile. Test is report-only. Online/test inference does not read CMA numeric labels, and this source release does not train a new CMA-STI-compatible fingerprint.
