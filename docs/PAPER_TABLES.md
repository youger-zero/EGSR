# Paper Tables

## Table 1. Main Results on the PGDP5K Test Split

| Method | LS | AS | PR | TS |
| --- | --- | --- | --- | --- |
| InterGPS | 65.7 | 44.4 | 40.0 | 27.3 |
| PGDPNet w/o GNN | 98.4 | 93.1 | 79.7 | 78.2 |
| PGDPNet | 99.0 | 96.6 | 86.2 | 84.7 |
| EGSR (Ours) | 99.0 | 95.2 | 83.7 | 83.0 |

Note: EGSR denotes the current clean no-oracle replayable mainline under the locked public-safe protocol.

## Table 2. Locked EGSR Replay Chain and Artifact Trace

| Stage | Artifact / Source | Split | Output Type | Role |
| --- | --- | --- | --- | --- |
| Candidate source | detector + OCR/rules + light VLM | test | primitive/relation candidates | clean candidate proposal |
| Scene replay | locked replay candidates | test | replayed scene hypotheses | scene reconstruction |
| Constraint scoring | locked scoring module | test | residuals / scores | executable checking |
| Logic projection | canonical projection | test | PGDP logic forms | deterministic output |
| Evaluation | official evaluator | test | LS / AS / PR / TS | post hoc metric computation |
| Pipeline metadata | locked config and logs | test | configs / run records | reproducibility audit |

Note: The locked manuscript mainline uses K=50 replayed scene hypotheses. Ground-truth logic forms are used only by the final evaluator after prediction.

## Table 3. Top-K Candidate Quality and Oracle Upper-Bound Analysis

| K | Coverage@K | TS-Coverage@K | Oracle-LS@K | Oracle-AS@K | Oracle-PR@K | Oracle-TS@K |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 91.5 | 74.4 | 97.9 | 91.5 | 75.2 | 74.4 |
| 5 | 96.2 | 84.7 | 99.4 | 96.2 | 85.7 | 84.7 |
| 10 | 96.7 | 84.7 | 99.5 | 96.7 | 85.7 | 84.7 |
| 20 | 96.8 | 84.7 | 99.7 | 96.8 | 85.7 | 84.7 |

Note: Oracle@K is computed only for post hoc upper-bound diagnosis and is never used during inference.

## Table 4. Downstream Ablation on the Locked EGSR Mainline

| Variant | LS | AS | PR | TS |
| --- | --- | --- | --- | --- |
| Direct projection | 98.9 | 94.4 | 79.7 | 79.1 |
| Reranking only | 91.9 | 71.7 | 53.5 | 53.3 |
| Canonical projection only | 98.9 | 94.4 | 79.7 | 79.1 |
| Reranking + canonical projection | 91.8 | 71.6 | 53.3 | 53.1 |
| Locked EGSR mainline | 99.0 | 95.2 | 83.7 | 83.0 |

Note: The locked mainline includes the fixed stable selection, canonical projection, and generic deterministic postprocessing used in the final released inference path.

## Table 5. Internal Consistency Analysis of Locked-Mainline Variants

| Variant | Executability Rate | Mean Residual | Render Consistency |
| --- | --- | --- | --- |
| Direct projection | 0.991 | 0.296391 | 0.999569 |
| Reranking only | 0.992 | 0.155301 | 0.945965 |
| Canonical projection only | 0.991 | 0.296391 | 0.999569 |
| Reranking + canonical projection | 0.992 | 0.154416 | 0.945541 |
| Locked EGSR mainline | 0.992 | 0.294769 | 0.999522 |

## Table 6. Generic Postprocessing Ablation on the Locked EGSR Mainline

| Postprocess Mode | LS | AS | PR | TS |
| --- | --- | --- | --- | --- |
| None | 99.0 | 95.2 | 83.7 | 83.0 |
| Canonical normalization | 99.0 | 95.2 | 83.7 | 83.0 |
| Canonical + relation conflict handling | 99.0 | 95.2 | 83.7 | 83.0 |
| Canonical + scalar conflict handling | 98.0 | 90.5 | 78.3 | 77.7 |
| Canonical + all conflict handling | 98.0 | 90.5 | 78.3 | 77.7 |
| Canonical + parallel closure | 99.0 | 95.2 | 83.7 | 83.0 |
| Canonical + equal-length closure | 99.0 | 95.2 | 83.7 | 83.0 |
| Canonical + all generic closure | 99.0 | 95.2 | 83.7 | 83.0 |

Note: This ablation verifies that the locked mainline does not rely on aggressive or sample-specific rewriting. Most generic deterministic postprocessing modes preserve the result, while overly aggressive scalar-conflict removal degrades performance and is therefore excluded from the released path.

## Table 7. K-Value and Runtime Trade-off of the Locked EGSR Pipeline

| K | LS | AS | PR | TS | Assembly + Constraint Time | Rerender + Reranking Time | Total Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 99.0 | 94.6 | 81.1 | 80.7 | 0.04238 | 0.00616 | 0.04854 |
| 10 | 98.8 | 93.5 | 78.3 | 77.9 | 0.05358 | 0.00679 | 0.06037 |
| 20 | 98.8 | 94.8 | 82.6 | 81.9 | 0.0776 | 0.013 | 0.0906 |

Note: This K-sweep is a diagnostic trade-off study over retained timing traces for K=5, 10, and 20. The formal locked manuscript mainline uses K=50, as recorded in the bundled replay metadata, but no stage-separated K=50 timing trace was retained under the same reporting protocol.

Timing protocol:

- `time_unit`: seconds per sample
- `assembly_constraint`: assemble_scenes + execute_constraints
- `rerender_reranking`: score_visual + score_symbolic + rerank_scenes_public
- `excluded`: ['build_candidates_clean', 'project_logic_form', 'eval_pgdp_metrics']
