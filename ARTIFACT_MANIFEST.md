# Artifact Manifest

This manifest identifies what a reviewer can inspect directly and which
caller-supplied inputs activate each data-dependent stage.

## Included Source

| Path | Role | Runs without external experiment data? |
| --- | --- | --- |
| `cem_wf_index/final_release/offline_lambdarank.py` | Grouped training, validation-only profile selection, frozen test reporting | `--help` only |
| `cem_wf_index/final_release/online_inference.py` | Frozen online feature, graph, blend, and top-20 reference | Import only; ranking requires model and candidate inputs |
| `cem_wf_index/final_release/graph_features.py` | Train-label graph construction and propagation | Unit-level use with supplied frames |
| `cem_wf_index/final_release/candidate_rows.py` | Candidate/relevance normalization and anchor merge | Unit-level use with supplied frames |
| `cem_wf_index/final_release/ranking_utils.py` | Retrieval metrics | Yes, with in-memory ranked/relevance inputs |
| `cem_wf_index/final_release/physical_errors.py` | Evaluation-only top-one physical metrics | No; needs caller-prepared CMA/mapping/descriptor files |
| `cem_wf_index/context/fingerprints.py` | Precomputed ERA5 background-context archive contract | Yes, with synthetic or separately supplied arrays |
| `cem_wf_index/context/index.py` | Context candidate indexing/search | Yes, with synthetic arrays |
| `cem_wf_index/context/candidates.py` | Event/context union, source ranks, and provenance | Yes, with synthetic candidate lists |
| `cem_wf_index/context/temporal.py` | Early temporal NMS, event deduplication, and diversity diagnostics | Yes, with synthetic events |
| `cem_wf_index/context/audit.py` | Structured candidate-stage audit schema | Yes, with synthetic records |
| `cem_wf_index/context/pipeline.py` | Composition of the synthetic upstream reference | Yes |
| `cem_wf_index/supplementary/statistics.py` | Generic paired query-bootstrap interval | Yes, with paired arrays |
| `cem_wf_index/supplementary/metrics.py` | Supplementary continuous/binary ranking metrics | Yes, with in-memory arrays |
| `cem_wf_index/supplementary/climatenet_portability.py` | ClimateNet development-study contract and aggregation | No; needs query-level JSON |
| `cem_wf_index/supplementary/matched_scoring.py` | Matched downstream-scoring contract and aggregation | No; needs query-level JSON |
| `cem_wf_index/supplementary/physical_metric_ablation.py` | Task-conditioned physical-metric development ablation | No; needs query-level JSON |
| `cem_wf_index/supplementary/run_evaluation.py` | CLI dispatch for the three supplementary evaluators | `--help` only |
| `scripts/run_final_label_graph_lambdarank.py` | Thin offline CLI wrapper | `--help` only |
| `scripts/run_synthetic_pipeline.py` | Synthetic context/candidate demonstration | Yes |
| `online/` | Compatibility import for the frozen online reference | Import only |
| `tests/` | Source-policy and synthetic evaluator checks | Yes |

See `supplementary/README.md` for the external JSON contracts and the development-only scope of those studies.

## Fingerprint Archive Metadata

`release_assets/fingerprint/` contains the cleaned public metadata and verification list for precomputed ERA5 background-context fingerprints supplied separately through a GitHub Release. The arrays themselves are not tracked in Git.

| Metadata file | Purpose |
| --- | --- |
| `manifest.json` | Declares the expected 1,024-dimensional ERA5 context archive schema, variables, window/stride, normalization, and coverage |
| `SHA256SUMS` | Identifies the expected `all_fp1024.npy`, `start_times.npy`, and `end_times.npy` bytes |
| `ATTRIBUTION.txt` | Records Copernicus/ERA5 attribution and the licence link |

These files document and authenticate the released upstream context input. The
archive connects to the public context interface without altering the final
ranker's frozen scoring configuration.

## Generated Outputs Expected From a Full Local Run

With all required inputs, the offline runner produces:

- a candidate-profile search ledger and partial ledger;
- validation and test top-20 ranking details;
- the selected-summary CSV/JSON;
- the exact selected feature-name CSV;
- a train-label-graph manifest; and
- storage/progress/event records.

The synthetic upstream example emits caller-directed demonstration output and
structured audit data derived entirely from generated inputs.

Those outputs are ignored by Git and are not included here.

## Required External Inputs

| Input | Used by | Minimum role |
| --- | --- | --- |
| Three split-specific candidate CSVs | Offline runner | Query/candidate rows, relevance for the appropriate split, and inference-available numeric features |
| Retained validation and test ranking CSVs | Offline and online anchor merge | Supplies the retained ordering required by the selected anchor profile |
| Fitted model, feature order, train median fills | Online reference | Frozen inference state |
| Two train-label graph mappings | Online reference | Frozen graph state derived from train positives |
| Descriptor, evaluation mapping, CMA best-track table | Physical-error helper | Post-ranking evaluation only |
| Query-level development JSON | Supplementary evaluators | Paired development-study values and required leakage/seal declarations |
| Precomputed background-context fingerprint arrays | Context archive/index | Upstream context input; the GitHub Release metadata and checksums identify the expected files |

## Recorded Protocol Contract

For the typhoon evaluation, the source documentation records the 1979--2020
archive, 1979--2016/2017--2018/2019--2020 chronological split, 1,035/63/55
queries, and 1,153 mapped events. `G20` is fixed independently of evaluated
methods from standardized CMA-STI track/pressure/wind summaries after
same-cyclone exclusion; `Recall@K = |TopK ∩ G20| / 20`, while landfall is
evaluated separately. The declared context-index backend is 1,024-D L2 HNSW
with `M=32` and `efConstruction=efSearch=200`.

## External-asset boundary

Source datasets and large generated run assets remain external under their
applicable licences and the input contracts above. Paper tables are not
duplicated in this source release; focused supplementary aggregates are clearly
separated in `supplementary/README.md`.

## Integrity and Safe Interpretation

1. Run `python -m pytest -q` and the compile/import checks in `README.md`.
2. For a data-dependent rerun, retain the emitted feature-name receipt, selected profile, input hashes, and per-query outputs together.
3. Acquire the separately supplied archive from the GitHub Release, then verify the named files against `release_assets/fingerprint/SHA256SUMS` before use.
4. Never infer paper-table lineage merely because a compatible source module, evaluator, or fingerprint manifest is present.
