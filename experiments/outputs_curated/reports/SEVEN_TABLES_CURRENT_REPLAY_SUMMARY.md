# Seven Tables Current Replay Summary

Date: 2026-05-01

## Formal manuscript mainline

- manuscript label: `EGSR (Ours)`
- formal replay variant: `egsr_clean_mainline_replay`
- split: `PGDP5K test`
- official metrics: `LS 99.0 / AS 95.2 / PR 83.7 / TS 83.0`

## Why this is the formal row

- it is reproduced from current code plus current retained clean inputs
- it uses no oracle inference
- it uses no sample-specific rewrite
- it has a complete retained artifact chain

## Why the older row is not the default paper row

- historical retained variant: `postproc5_round1_canon`
- historical retained metrics: `LS 99.0 / AS 95.3 / PR 83.8 / TS 83.1`
- fresh replay comparison shows exactly one differing sample: `4945`
- therefore the historical row is kept only as archival audit evidence

## Seven table files

1. Table 1
   - `experiments/outputs_curated/reports/TABLE1_main_results_pgdp5k_test_current_replay.json`
2. Table 2
   - `experiments/outputs_curated/reports/TABLE2_locked_replay_chain_current_replay.json`
3. Table 3
   - `experiments/outputs_curated/reports/TABLE3_topk_candidate_quality_current_replay.json`
4. Table 4
   - `experiments/outputs_curated/reports/TABLE4_downstream_ablation_locked_mainline.json`
5. Table 5
   - `experiments/outputs_curated/reports/TABLE5_internal_consistency_locked_mainline_variants.json`
6. Table 6
   - `experiments/outputs_curated/reports/TABLE6_generic_postprocessing_ablation_locked_mainline.json`
7. Table 7
   - `experiments/outputs_curated/reports/TABLE7_k_value_runtime_tradeoff_locked_pipeline.json`

## Core numbers

### Table 3 Oracle@K

- `K=1: Coverage 91.5 / TS-Coverage 74.4 / Oracle-TS 74.4`
- `K=5: Coverage 96.2 / TS-Coverage 84.7 / Oracle-TS 84.7`
- `K=10: Coverage 96.7 / TS-Coverage 84.7 / Oracle-TS 84.7`
- `K=20: Coverage 96.8 / TS-Coverage 84.7 / Oracle-TS 84.7`

### Table 4 Downstream ablation

- `Direct projection: 98.9 / 94.4 / 79.7 / 79.1`
- `Reranking only: 91.9 / 71.7 / 53.5 / 53.3`
- `Canonical projection only: 98.9 / 94.4 / 79.7 / 79.1`
- `Reranking + canonical projection: 91.8 / 71.6 / 53.3 / 53.1`
- `Locked EGSR mainline (current replay row): 99.0 / 95.2 / 83.7 / 83.0`

### Table 6 Generic postprocessing

- neutral:
  - `None`
  - `Canonical normalization`
  - `Canonical + relation conflict handling`
  - `Canonical + parallel closure`
  - `Canonical + equal-length closure`
  - `Canonical + all generic closure`
- harmful:
  - `Canonical + scalar conflict handling`
  - `Canonical + all conflict handling`

### Table 7 K-value trade-off

- `K=5: 99.0 / 94.6 / 81.1 / 80.7`
- `K=10: 98.8 / 93.5 / 78.3 / 77.9`
- `K=20: 98.8 / 94.8 / 82.6 / 81.9`

## Paper-safe recommendation

Use the `egsr_clean_mainline_replay` row as the default manuscript result. The
older `postproc5_round1_canon` row is retained only as archival audit evidence,
not as a paper table default.
