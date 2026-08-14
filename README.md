# CEM-WF-Index Paper-Review Source

This repository is the inspectable source-code portion of the CEM-WF-Index review artifact. It contains the grouped label-graph ranker, its frozen online-ranking reference, the upstream context/candidate interface, focused supplementary evaluators, and lightweight tests.

With the declared external inputs, the code executes the released ranking protocol; the repository also provides a data-free synthetic workflow for immediate inspection. See [ARTIFACT_MANIFEST.md](ARTIFACT_MANIFEST.md) for the component/data contract and [REVIEW_ALIGNMENT.md](REVIEW_ALIGNMENT.md) for reviewer-item correspondence.

The precomputed ERA5 background-context archive is distributed through the
[rebuttal artifact release](https://github.com/BigBaooooo/CEM-WF-Index/releases/tag/rebuttal-artifact).

## Quick Start

Use Python 3 with `numpy`, `pandas`, `scikit-learn`, `lightgbm`, and `pytest`. From the repository root:

```bash
python -m compileall cem_wf_index scripts online tests
python -m pytest -q
python -c "from cem_wf_index.final_release import METHOD_DEFINITION, SELECTED_PROFILE; print(METHOD_DEFINITION, SELECTED_PROFILE)"
python -m scripts.run_final_label_graph_lambdarank --help
python -m cem_wf_index.supplementary.run_evaluation --help
```

The `python -m ...` form is required by the current package layout. These commands check imports, policy flags, and CLI availability without downloading large external inputs.

## Repository Layout

```text
cem_wf_index/final_release/
  offline_lambdarank.py   grouped training, validation selection, report-only test
  online_inference.py     frozen online scoring and anchor-preserving top-20
  typhoon_frozen_asset_manifest.json  verified model/feature/train-graph contract
  graph_features.py       train-label graph construction and propagation
  candidate_rows.py       candidate normalization, relevance sets, anchor merge
  ranking_utils.py        Recall, nDCG, mAP, and ranked-list helpers
  physical_errors.py      evaluation-only typhoon physical errors
cem_wf_index/context/
  fingerprints.py         precomputed background-context archive contract
  index.py                context candidate index
  candidates.py           multi-channel fusion and rank provenance
  temporal.py             early temporal NMS and event deduplication
  audit.py                structured candidate-stage audit records
  pipeline.py             synthetic upstream pipeline composition
cem_wf_index/supplementary/ paired bootstrap and focused diagnostic evaluators
scripts/                  offline wrapper and synthetic context example
online/                   compatibility import for the online reference
tests/                    source-only checks
```

The implementation guide is in [SOURCE_GUIDE.md](SOURCE_GUIDE.md).

## Synthetic Upstream Example

The context package demonstrates the upstream boundary with generated arrays and events; it does not download data, fit the final ranker, or reproduce a paper result:

```bash
python -m scripts.run_synthetic_pipeline --help
python -m scripts.run_synthetic_pipeline
```

The example composes a `FingerprintArchive`, a `ContextIndex`, event/context/metadata candidate fusion, early temporal NMS, event deduplication, and a structured audit. Its 24-hour-window demonstration applies the documented overlap rule together with a 12-hour start-time threshold. Fused rows carry source ranks, channel provenance, and `score_context_rank_prior`; the verified frozen model consumes this prior, while validation did not select an additional standalone context term.

## Released Ranking Protocol

The offline runner operates on three separate candidate-row tables: train, validation, and test.

1. **Train only:** build the positive-candidate graph and fit the candidate models. `is_gt_top20` supplies binary relevance; `grade`, when present, supplies the graded LambdaRank label.
2. **Validation only:** evaluate the model/blend/anchor grid and select one profile. The implemented composite is

   ```text
   0.30*nDCG@10 + 0.30*mAP@10 + 0.25*Recall@20 + 0.10*Recall@10
   ```

   This is the literal, unnormalized implementation. A validation floor is applied first; if no profile meets it, selection falls back to profiles that retain at least one anchor candidate.
3. **Freeze:** fix the selected model, blend, and anchor count.
4. **Test report only:** score the test candidates once with the selected profile, form the top-20 list, and then compute retrieval and physical-consistency metrics. Test labels are not used in fitting or profile selection.

The released online reference exposes the frozen configuration:

| Component | Frozen setting |
| --- | --- |
| Model | graded LightGBM LambdaRank, 520 trees, learning rate 0.03, 63 leaves, minimum child samples 35 |
| Tree feature sampling | `colsample_bytree=0.90` |
| Seed | `20262202` |
| Blend | base score 0.10, model 0.45, graph 0.30, calibrated score 0.15, explicit source-min-rank term 0.00 |
| Graph configurations | `(seed_k, activated_query_cap)` = `(20,200)`, `(16,200)`, `(20,500)`, `(40,500)` |
| Anchor | keep the first 5 retained candidates, then fill from the proposal without duplicates |
| Output | at most 20 candidates per query |

The frozen model constructor records `subsample=0.90` and
`colsample_bytree=0.90`; this completion does not change its training behavior.

The complete experimental protocol is consolidated here together with the
released context input and reviewer-requested development evaluators.

The verified software environment is Python 3.10.20, NumPy 2.2.6, pandas
2.3.3, scikit-learn 1.7.2, LightGBM 4.6.0, and hnswlib 0.8.0.

## Candidate and Feature Contract

### Typhoon evaluation protocol

The typhoon archive covers 1979--2020 using six-hourly ERA5 fields at
0.25° × 0.25° resolution. Chronological train/validation/test periods are
1979--2016, 2017--2018, and 2019--2020, containing 1,035/63/55 query events
over 1,153 mapped archive events. Candidates mapped to the same CMA-STI
cyclone as the query are excluded.

Ground truth is constructed independently of every evaluated retrieval
method. For each query, `G20` is the 20 eligible archive events with the
smallest Euclidean distances in a standardized CMA-STI event-summary space:
mean and span of latitude/longitude, track-point count, minimum/mean pressure,
and maximum/mean wind. `Recall@K = |TopK ∩ G20| / 20`. Landfall consistency is
reported separately as a physical-consistency metric and does not construct
`G20`.

The context index uses 1,024-dimensional L2 vectors with `M=32`,
`efConstruction=200`, and `efSearch=200`.

The runner expects these compatibility basenames under `--candidate-detail-dir`:

```text
v8_train_candidate_rows_sample.csv
v8_validation_candidate_rows.csv
v8_test_candidate_rows.csv
```

Core row fields are:

| Field | Role |
| --- | --- |
| `query_id`, `candidate_id` | group and identify query-candidate pairs |
| `is_gt_top20` | train label; validation/test evaluation relevance only |
| `grade` | optional graded train label |
| `score_final`, `score_calibrated` | inference-available upstream scores |
| `source_event_rank`, `source_sequence_rank`, `source_min_rank`, `source_rank_gap` | upstream rank-level candidate evidence |
| `rank` | retained-ranking order when used as an anchor input |

Training and online inference both enforce the same ordered 31-feature contract in `cem_wf_index/final_release/typhoon_frozen_feature_names.csv`. The adjacent frozen-asset manifest binds that order to the verified 31-input LightGBM model and train-only label graph. Missing, reordered, unregistered, or hash-mismatched inputs fail before prediction. Labels and evaluation fields may remain in prepared offline tables, but they are never selected into the model matrix; other numeric diagnostics are reported and ignored. The source-rank pool name is a configured upper-bound label, not a guarantee that every query has that many rows.

The upstream context/fingerprint stage is outside the final ranker's fit function. Candidate tables carry context-derived membership and rank-prior evidence. The hash-verified frozen Booster uses `score_context_rank_prior` in 1,389 splits with gain importance 1,327.8583; validation did not select an additional standalone direct-context term in the final linear blend. The final ranker does not train the precomputed fingerprint archive. CMA numeric track, intensity, and landfall fields are evaluation-only and are not online ranking inputs.

## Offline Use With Prepared Inputs

Inspect the data-dependent runner first:

```bash
python -m scripts.run_final_label_graph_lambdarank --help
```

Then provide a working root, candidate tables, and retained validation/test ranking tables:

```bash
python -m scripts.run_final_label_graph_lambdarank \
  --root REVIEW_RUN \
  --candidate-detail-dir PREPARED_CANDIDATES \
  --retained-validation-ranking RETAINED_VALIDATION.csv \
  --retained-test-ranking RETAINED_TEST.csv
```

`REVIEW_RUN` must also satisfy the evaluation-file contract used by `physical_errors.py`. The runner writes generated details/reports beneath that working root and progress metadata beneath the artifact roots selected by `CEMWF_ARTIFACT_ROOT` and `CEMWF_SECONDARY_ARTIFACT_ROOT`. Generated files are intentionally ignored by Git.

The runner requires retained anchor rows for the implemented profile. In the online API, every retained top-five candidate must also occur in the supplied candidate pool.

## Online API

The public online function is `rank_v8_22_online`:

```python
from cem_wf_index.final_release.online_inference import rank_v8_22_online
```

It accepts candidate rows, retained top-20 anchor rows, a frozen fitted model, the frozen feature order and fill values, and the two frozen train-label graph mappings. Before graph propagation or prediction, it verifies feature order/dimension, model identity, Context importance, and train-graph identity against the published frozen-asset manifest. It returns at most 20 rows per query and attaches the executable input-audit receipt to the returned frame. The online call does not fit a model, read CMA numeric fields, or select a profile.

## Scope and Claims

- Standard nearest-neighbor, LightGBM, metric, and bootstrap primitives are treated as backends or evaluation tools, not as individually novel algorithms.
- The paper's contribution claim remains at the system level: event/context candidate organization, leakage-controlled calibration, frozen online ranking, and auditable evaluation boundaries.
- Any statistical interval is valid only for the exact paired per-query rows supplied to the corresponding tool; the presence of uncertainty tooling alone is not a main-result confidence-interval claim.
- External datasets remain subject to their own licenses and distribution terms and are not redistributed here.
