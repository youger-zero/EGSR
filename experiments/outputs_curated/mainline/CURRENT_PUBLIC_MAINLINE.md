# Current Public Mainline

## Status

Current formal locked public-safe result:

- manuscript label: `EGSR (Ours)`
- variant: `egsr_clean_mainline_replay`
- status: `LOCKED`
- date locked: `2026-05-01`

## Official Metrics

### `egsr_clean_mainline_replay`

- LS: 99.0
- AS: 95.2
- PR: 83.7
- TS: 83.0

### Historical retained replay

- variant: `postproc5_round1_canon`
- metrics: `LS 99.0 / AS 95.3 / PR 83.8 / TS 83.1`
- status: archival retained evidence only
- note: this row is no longer the default paper result because a fresh replay
  from current code and current retained clean inputs yields the
  `egsr_clean_mainline_replay` row above

## Official Paths

- source candidates:
  - `experiments/outputs/candidates/detector_plus_ocr_rules_plus_light_vlm/test`
- replay candidates:
  - `experiments/outputs/candidates/egsr_clean_mainline_replay/test`
- scenes:
  - `experiments/outputs/scenes/egsr_clean_mainline_replay/test`
- scores:
  - `experiments/outputs/scores/egsr_clean_mainline_replay/test`
- logic forms:
  - `experiments/outputs/logic_forms/egsr_clean_mainline_replay/test`
- eval:
  - `experiments/outputs/eval/egsr_clean_mainline_replay/test`
- pipeline meta and audit:
  - `experiments/outputs/pipelines/egsr_clean_mainline_replay/test`

## Lock Rationale

- strongest currently reproducible clean replay from current code plus current
  retained clean `test` source artifacts
- no sample-specific rewrite path
- no approved non-oracle frozen `train/val` source artifacts are currently available
- therefore, further optimization is frozen to avoid drifting into test-driven development

## Replay Difference Audit

- a fresh replay differs from the historical retained row by exactly one test
  sample: `4945`
- the difference is caused by a geometry-to-logic projection normalization
  ambiguity in an angle-equality form
- therefore the current reproducible replay is the public-safe manuscript
  default, while the older retained row remains archival reference only

## Unlock Condition

The mainline may be reopened only if all of the following hold:

1. non-oracle frozen candidate source artifacts are restored or provided for `val`
2. ideally the same is done for `train`
3. all development and selection move to `val`
4. `test` is used only for final confirmation
5. the new candidate beats `egsr_clean_mainline_replay` under that protocol

## Notes

- Public-safe clean replays route through `experiments/scripts/rerank_scenes_public.py`.
- Legacy handcrafted reranking code is not part of the public-safe mainline.
- `egsr_clean_mainline_replay` uses `rewrite-mode none` and
  `public-postprocess-mode canonical_only`.
- historical retained `postproc5_round1_canon` remains stored for audit
  comparison only.
- Optimization against `test` is frozen by protocol until clean `val` source artifacts are restored.
