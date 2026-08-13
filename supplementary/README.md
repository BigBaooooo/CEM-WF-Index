# Focused supplementary development results added in response to reviewer questions; separate from the manuscript's main test tables

These focused development controls address reviewer questions while remaining
separate from the manuscript's main held-out test tables.  They do not replace
or restate the paper's principal results.

| Study | Development queries | Metric | Control | CEM-WF-Index | Paired gain |
|---|---:|---|---:|---:|---:|
| ClimateNet atmospheric-river portability (492 objects) | 86 | continuous nDCG@10 | 0.575 | 0.663 | +0.088, 95% CI [0.068, 0.109] |
| Matched downstream scoring, strongest backend control (ACORN) | 62 | nDCG@10 | 0.397 | 0.498 | +0.101, 95% CI [0.054, 0.149] |

For ClimateNet, the gain is positive in all three temporal folds and the
reported aggregate averages five model seeds.  This development study uses
expert atmospheric-river objects to test portability through the shared
event-retrieval interface; the independent evaluation masks are not used as
model inputs, and the formal test remains unopened.

In the matched-scoring control, the same previously sealed scorer is applied
to the ACORN shortlist and the full CEM-WF-Index candidate pool.  The average
number of relevant events in the top 20 rises from 4.42 to 7.97, showing that
the broader relevant-event support from multi-channel candidate generation
remains beneficial when downstream supervision is held fixed.

A separate task-conditioned physical-state metric ablation yields a +0.148
typhoon-track nDCG@10 gain (paired 95% CI [0.115, 0.181]), is positive in all
three temporal folds, and reduces top-1 Track RMSE by 469 km.  This ablation
tests the contribution of task-conditioned physical evidence within the
shared retrieval workflow.

## Evaluation Code

The public evaluators aggregate externally supplied query-level development
records.  No raw annotations, event identifiers, candidate tables, model
files, or result bundles are stored in this repository.

Run any evaluator from the repository root:

```bash
python -m cem_wf_index.supplementary.run_evaluation climatenet-portability \
  --input INPUT.json --output SUMMARY.json
python -m cem_wf_index.supplementary.run_evaluation matched-scoring \
  --input INPUT.json --output SUMMARY.json --backend acorn --full-system cem
python -m cem_wf_index.supplementary.run_evaluation physical-metric-ablation \
  --input INPUT.json --output SUMMARY.json
```

All input and output paths are supplied by the caller.  The JSON contracts
contain no machine-specific path fields.

## Data Contracts and Leakage Guards

The ClimateNet input declares `study_scope="development"`,
`formal_test_opened=false`, `scores_sealed_before_relevance=true`, and
`input_and_evaluation_annotations_disjoint=true`.  It supplies three temporal
fold boundaries and, per query, its timestamp, fold, continuous relevance,
and five pairs of sealed candidate-score vectors.  It computes nDCG per seed,
then averages the five values at query level before paired query bootstrap.
The evaluator checks that every query is inside the evaluation portion of its
chronological fold.

The matched-scoring input declares a non-empty `scorer_seal` and
`scorer_sealed_before_labels=true`.  Each query supplies exactly 20 independent
ground-truth event identifiers, one shared sealed score map, and the candidate
identifiers included by the backend and full M=1,000 pools.  Both rankings are
constructed from that single score map.  The evaluator also requires the
backend shortlist to be a subset of the full pool, making the matched
downstream-scoring comparison explicit in the released logic.

The physical-metric ablation input supplies three chronological development
folds and query-level nDCG@10 and top-1 Track error values for the control and
task-conditioned metric.  It reports fold deltas, a paired query-bootstrap
interval, and the corresponding Track RMSE reduction.

`paired_bootstrap_interval` is also exposed as a generic query-level utility.
It resamples aligned query pairs and supports uncertainty analysis for other
comparisons when their exact per-query values are supplied.
