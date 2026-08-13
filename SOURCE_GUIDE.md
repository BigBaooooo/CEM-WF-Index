# Source Guide

This guide maps the paper-level method description to the executable source. It describes code correspondence, not numeric-result provenance; generated measurements and datasets are outside this repository.

## Paper-to-Code Map

| Paper concept | Primary implementation | What the code does |
| --- | --- | --- |
| Full-candidate supervised calibration | `cem_wf_index/final_release/offline_lambdarank.py` | Loads split-specific candidate rows, derives features, fits grouped rankers/classifier, evaluates the validation grid, freezes one profile, and only then scores test rows. |
| Frozen online ranking | `cem_wf_index/final_release/online_inference.py` | Recreates inference-time derived/graph features, applies the fitted ranker and frozen blend, keeps five anchor candidates, and fills a duplicate-free top-20 proposal. |
| Train-label graph | `cem_wf_index/final_release/graph_features.py` | Builds candidate-to-train-query and train-query-to-positive-candidate mappings from train positives; activates and propagates them for each query. |
| Candidate and anchor handling | `cem_wf_index/final_release/candidate_rows.py` | Normalizes identifiers/relevance, derives reciprocal source ranks, extracts relevance sets, and performs anchor-preserving merging. |
| Retrieval metrics | `cem_wf_index/final_release/ranking_utils.py` | Computes per-query Recall@10/20, binary nDCG@10, and AP@10, then macro-averages over queries with nonempty relevance sets. |
| Physical consistency | `cem_wf_index/final_release/physical_errors.py` | Computes top-one track, pressure, wind, and approximate landfall errors after ranking; these CMA-derived values are evaluation-only. |
| Precomputed context archive | `cem_wf_index/context/fingerprints.py` | Validates and exposes separately supplied precomputed ERA5 background-context fingerprints and their timestamps. |
| Context retrieval | `cem_wf_index/context/index.py` | Builds the upstream context index and returns ranked context candidates. |
| Multi-channel candidate fusion | `cem_wf_index/context/candidates.py` | Unions event/context candidates, deduplicates identifiers, and emits source ranks, provenance, and the context reciprocal-rank prior. |
| Early diversity control | `cem_wf_index/context/temporal.py` | Applies early temporal NMS, event deduplication, and unique-event-ratio diagnostics before final ranking. |
| Candidate-stage audit | `cem_wf_index/context/audit.py` | Emits structured records for profile, sources/provenance, context stage, exclusions, diversity controls, frozen train-graph origin, and leakage checks. |
| Synthetic upstream composition | `cem_wf_index/context/pipeline.py`, `scripts/run_synthetic_pipeline.py` | Demonstrates the context-to-candidate contract using synthetic data only. |
| Paired uncertainty | `cem_wf_index/supplementary/statistics.py` | Computes paired query-bootstrap intervals from caller-supplied aligned values. |
| Focused development diagnostics | `cem_wf_index/supplementary/` | Validates and aggregates caller-supplied ClimateNet portability, matched-scoring, and physical-metric-ablation records. |
| Offline CLI entrypoint | `scripts/run_final_label_graph_lambdarank.py` | Thin wrapper; invoke it as a Python module from the repository root. |
| Online compatibility path | `online/final_online_inference_reference.py` | Re-exports the online reference implementation. |

## Exact Split and Leakage Protocol

The typhoon archive spans 1979--2020 at six-hour cadence and 0.25° × 0.25°
resolution. The chronological train/validation/test periods are 1979--2016,
2017--2018, and 2019--2020, with 1,035/63/55 query events over 1,153 mapped
archive events. Candidates from the same mapped CMA-STI cyclone are excluded.

Before any evaluated method is scored, each query's independent reference set
`G20` is fixed as the 20 eligible archive events nearest by Euclidean distance
in standardized event summaries of track geometry, pressure, and wind. The
summaries are latitude/longitude means and spans, track-point count,
minimum/mean pressure, and maximum/mean wind. `Recall@K` is
`|TopK ∩ G20| / 20`; landfall consistency remains a separately reported
physical-consistency metric.

| Stage | Allowed labels/data | Forbidden use |
| --- | --- | --- |
| Train graph and model | Train `is_gt_top20`; optional train `grade`; inference-available candidate features | Validation/test relevance in graph construction or fitting |
| Validation selection | Validation relevance for metrics, floors, and profile choice | Test relevance or test physical metrics for selection |
| Frozen test ranking | Candidate features, fitted model, frozen train graph, frozen blend/anchor profile | CMA numeric fields or test relevance as ranking features |
| Test evaluation | Test relevance and CMA-derived physical records after the ranking is fixed | Feeding evaluation values back into scoring or tuning |

The offline program materializes test feature matrices before selection for input preparation, but it does not fit on them, evaluate test labels, or predict the test ranking until after the validation profile is selected. The emitted policy flags are `uses_cma_gt_labels_for_training=True`, `uses_cma_numeric_at_inference=False`, and `uses_test_for_tuning=False`.

## Graph Definition

`_train_graph` retains train rows with `is_gt_top20=True`. A positive's edge weight is `1 + grade/max_train_grade`, using one when no grade is supplied. For each inference query, `_graph_scores`:

1. forms a seed score from 0.70 times the within-query standardized upstream final score plus 0.30 times the within-query standardized reciprocal source-minimum rank;
2. takes the first `seed_k` candidates;
3. activates train queries sharing those candidates with reciprocal seed-rank weighting;
4. keeps at most `activated_query_cap` train queries; and
5. propagates their train-positive candidates back into the current pool and normalizes by total activation weight.

Four graph views are generated with `(seed_k, cap)` equal to `(20,200)`, `(16,200)`, `(20,500)`, and `(40,500)`. Their raw and within-query standardized values, plus maximum/mean aggregates, become candidate features.

## Model and Search Configuration

The offline search fits three candidates:

| Candidate | Objective/labels | Trees | Learning rate | Leaves | Minimum child samples | Column sampling | Seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary LambdaRank | LambdaRank/binary | 420 | 0.04 | 31 | 45 | 0.85 | 20262201 |
| Graded LambdaRank | LambdaRank/graded | 520 | 0.03 | 63 | 35 | 0.90 | 20262202 |
| Balanced classifier | binary classifier | 520 | 0.035 | 63 | 40 | 0.90 | 20262203 |

The constructors retain their configured `subsample` values of 0.85 or 0.90;
this completion does not change the frozen training behavior.

Each model is evaluated with:

- model weight in `{0.35, 0.45, 0.55, 0.65}`;
- graph weight in `{0.10, 0.20, 0.30}`;
- calibrated-score weight in `{0.05, 0.15, 0.25}`;
- source-min-rank weight in `{0.00, 0.06}`;
- base-score weight set to the nonnegative residual; and
- retained anchor count in `{0, 1, 3, 5}`.

Validation selection first requires nDCG@10 ≥ 0.86, mAP@10 ≥ 0.80, Recall@10 ≥ 0.41, Recall@20 ≥ 0.735, and at least one retained anchor. If that set is empty, the fallback contains all profiles with at least one retained anchor. The winner is ordered by Recall@20, then the implemented composite, then mAP@10.

The frozen online profile is the graded LambdaRank model with weights `(base, model, graph, calibrated, source) = (0.10, 0.45, 0.30, 0.15, 0.00)` and `anchor_keep=5`. The letter `c` in the profile string denotes the calibrated-score term, not a context-channel weight.

## Feature Selection and Receipts

`_feature_columns` excludes identifiers, relevance labels, rank, split/method/profile fields, pool provenance fields, every `uses_*` field, and fields whose name contains `cma_numeric`. It then accepts remaining numeric columns. `_matrix` learns per-column median fill values from train rows and reuses them for validation and test.

Consequently:

- the generated feature-name CSV is the authoritative feature-order receipt for a concrete run;
- the fitted model, feature order, and train medians must travel together for online use; and
- callers must not add numeric label/evaluation fields outside the exclusion policy.

Upstream event/sequence/context information is represented through the candidate-row fields provided to this package. The final ranker does not construct or train the precomputed fingerprint archive; the upstream context stage carries candidate provenance and a context rank prior into the row contract.

## Upstream Context and Candidate Boundary

The public upstream reference has this one-way interface:

```text
precomputed background-context fingerprints
  -> ContextIndex ranked candidates
  -> event/context union + provenance + context reciprocal-rank prior
  -> early temporal NMS and event deduplication
  -> candidate rows consumed by the frozen ranking stage
```

`score_context_rank_prior` is candidate-rank evidence, not a probability or the final score. Source provenance records which candidate channels admitted an event. Early temporal NMS and event deduplication operate before the final ranker; the audit record identifies the applied exclusions and diversity steps. The synthetic pipeline exercises this contract without asserting measured effectiveness or main-table lineage.

`ContextIndex` uses 1,024-dimensional L2 vectors with HNSW `M=32`,
`efConstruction=200`, and `efSearch=200`; its exact fallback preserves the same
distance/output contract for small synthetic checks.

## Anchor and Top-20 Semantics

For the frozen profile, the first five candidates from the retained anchor ranking are placed first, without duplicates. Remaining positions are filled by descending proposal score until 20 candidates are present. The online function expects every retained top-five identifier to occur in the candidate pool because output rows are materialized from that pool.

## Metrics

- `Recall@k` is the number of distinct relevant candidates in the first `k` positions divided by the full relevant-set size.
- `nDCG@10` uses binary gains and logarithmic discounts.
- `mAP@10` is the mean of AP@10, with the AP denominator `min(10, relevant-set size)`.
- All reported retrieval metrics are macro-averaged over queries with a nonempty relevance set.
- Physical errors are computed only after top-one selection and never enter the model or blend.

## CLI and Output Contract

Use module invocation:

```bash
python -m scripts.run_final_label_graph_lambdarank --help
python -m scripts.run_synthetic_pipeline --help
python -m cem_wf_index.supplementary.run_evaluation --help
```

The offline CLI accepts `--root`, `--candidate-detail-dir`, `--retained-validation-ranking`, and `--retained-test-ranking`. It writes a search ledger, selected validation/test rankings, the actual feature-name receipt, a train-graph manifest, a selected summary, and progress records. Those are generated local artifacts and are not tracked in this source repository.

## Contribution Boundary

This release uses standard LightGBM, nearest-neighbor/upstream retrieval, ranking metrics, and statistical primitives as backends. The claimed contribution remains the system-level composition and its boundaries: event/context candidate organization, train-only supervision, validation-only selection, frozen online scoring, and auditable post-ranking evaluation.
