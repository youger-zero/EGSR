# Data Setup

This public release does not include the full PGDP5K data bundle or the
official PGDP evaluator repository.

## Expected local layout

Place the external resources in the following layout:

```text
experiments/
  data/
    PGDP5K/
      splits/
        train.txt
        val.txt
        test.txt
      ...
    PGDP/
      InterGPS/
      ...
```

## What this repository already includes

This release already includes the split files:

- `experiments/data/PGDP5K/splits/train.txt`
- `experiments/data/PGDP5K/splits/val.txt`
- `experiments/data/PGDP5K/splits/test.txt`

## What you still need to provide

- the official PGDP5K images and annotations
- the official PGDP repository clone used by `eval_pgdp_metrics.py`
- any additional upstream public resources required by your local PGDP setup

## What can already be verified without external data

Even without the external dataset bundle, this public repository already lets
you verify:

- the locked manuscript mainline identity
- the retained seven paper tables
- the no-oracle public projection path
- the public-release audit tests

Useful files:

- `experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md`
- `docs/PAPER_TABLES.md`
- `experiments/outputs/eval/egsr_clean_mainline_replay/test/pgdp_metrics.json`
- `experiments/outputs/pipelines/egsr_clean_mainline_replay/test/_run_meta.json`

## Notes

- the public replay path itself is no-oracle
- ground truth is used only by the final evaluator
- the manuscript tables in `experiments/outputs_curated/reports/` are already
  populated from the retained clean replay artifacts
- the formal public manuscript row is `egsr_clean_mainline_replay`
