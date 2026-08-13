# Review Alignment

This document maps reviewer items to public source and documentation. It states the strongest claim supported by the repository and records remaining evidence boundaries.

## Reviewer 2

### R2-W1 — Manuscript/implementation ranking correspondence

- `SOURCE_GUIDE.md` maps full-candidate calibration, train-label graph propagation, frozen online scoring, anchor merging, metrics, and evaluation to their implementing modules.
- `offline_lambdarank.py` implements train-only fitting, validation-only profile selection, profile freezing, and report-only test scoring.
- `online_inference.py` exposes the frozen blend and anchor configuration.

The released ranking stages and configuration can therefore be traced directly to code.

### R2-W2 — Context-channel boundary

- The final ranker consumes candidate rows prepared by an upstream retrieval pipeline; it does not train a fingerprint archive.
- `cem_wf_index/context/fingerprints.py` and `index.py` define the `FingerprintArchive` to `ContextIndex` path; `candidates.py` emits context rank, source provenance, and `score_context_rank_prior` before frozen final scoring.
- `release_assets/fingerprint/manifest.json` and `SHA256SUMS` specify the separately supplied, precomputed ERA5 background-context fingerprints without storing the arrays in Git.
- The frozen blend's `c` term is the calibrated score, not a context weight; its explicit source-min-rank term has weight zero.

This makes context an explicit upstream candidate-evidence channel while preserving the frozen downstream scorer.

### R2-W3 — Experimental protocol and configuration

- `README.md` states the train/validation/frozen-test sequence, candidate-row contract, frozen model/blend/graph/anchor configuration, and commands.
- `SOURCE_GUIDE.md` records all three model constructors, the blend grid, validation floors/fallback, selection order, metric definitions, and leakage boundary.
- The public protocol states the 1979--2020 archive, chronological split and query counts, independent `G20` construction, same-cyclone exclusion, Recall definition, separate landfall evaluation, and HNSW parameters.
The public source now records the implemented protocol and configuration in one place.

### R2-W4 — Statistical uncertainty tooling

- `cem_wf_index/supplementary/statistics.py` provides deterministic paired query-bootstrap intervals.
- Supplementary evaluators apply it only to caller-supplied aligned query-level development records.

Paired-bootstrap tooling is publicly available for exact aligned per-query inputs, with deterministic resampling and explicit query counts.

### R2-W5 — Artifact scope

- `ARTIFACT_MANIFEST.md` separates included source/metadata, generated outputs, and required external inputs.
- Quick checks work without data; offline reruns, online ranking, physical evaluation, and development aggregations require their declared external inputs.

The release boundary and external-input contract are explicit and auditable.

### R2-D2 — Diversity and early T-NMS

The final-release ranker receives an already generated candidate pool and performs duplicate-free anchor/proposal merging; it is not the implementation point for upstream temporal non-maximum suppression. `cem_wf_index/context/temporal.py` provides `early_temporal_nms`, `deduplicate_events`, and `unique_event_ratio` at the candidate stage.

The synthetic workflow reports pre/post counts, suppression ratio, unique-event ratio, and an event-distinct top-K under the same public schema.

### R2-D3 — Audit outputs

- The offline runner emits the feature-name receipt, train-graph manifest, search ledger, selected summary, ranking details, progress state, event log, and policy flags.
- `cem_wf_index/context/audit.py` records the selected profile, source ranks/provenance, context stage, candidate exclusions, early T-NMS/deduplication, frozen train-graph provenance, and leakage checks.
The audit schema, writer, and a fully synthetic inspectable example are public.

## Reviewer 3

### System-level contribution

The contribution statement remains unchanged: CEM-WF-Index is presented as a system-level composition of event/context candidate organization, leakage-controlled supervision, frozen online ranking, and auditable evaluation. LightGBM, nearest-neighbor retrieval, ranking metrics, paired bootstrap, and other standard primitives are backends or evaluation tools, not individually claimed algorithmic novelties.

## Reviewer 4

### R4-W1 / R4-W3a — ClimateNet portability

- `cem_wf_index/supplementary/climatenet_portability.py` checks a development-only, three-fold chronological contract, unopened formal test, sealed scores, and disjoint input/evaluation annotations.
- `supplementary/README.md` reports the corresponding aggregate development study and makes its query count and metric explicit.

The shared event-retrieval interface now has a public evaluator for the fold-local ClimateNet atmospheric-river portability study; object and query counts are stated separately.

### R4-W3b — Component novelty and task metric

- `cem_wf_index/supplementary/physical_metric_ablation.py` evaluates caller-supplied chronological, paired development records for a task-conditioned physical metric.
- `physical_errors.py` separately defines post-ranking typhoon physical-consistency metrics.

Task-conditioned physical evidence can now be isolated with a public chronological development-ablation contract.

### R4-W4 — Matched supervision

- `cem_wf_index/supplementary/matched_scoring.py` requires a sealed scorer, identical shared-candidate scores, a backend shortlist contained in the full pool, and the same independent relevance set.
- It reports paired query-level differences and uncertainty from caller-supplied development records.

The evaluator holds downstream scoring fixed while comparing candidate support, directly isolating the multi-channel pool under the declared development protocol.

## Public evidence organization

Core paper code, input-asset metadata, synthetic checks, and focused
supplementary development studies are separated by directory so each item can
be inspected under its intended evaluation contract.
