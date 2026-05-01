# Public Release Audit

This repository was exported from a larger local working directory into a
public-safe release tree.

## Removed from the public release

- local machine paths
- personal environment references
- remote sync helpers
- local one-click wrappers
- historical notes not needed for the paper repository
- legacy tuning and exploratory scripts
- historical dirty or ambiguous result chains not needed for the clean replay

## Kept in the public release

- clean replay scripts
- no-oracle compliance test
- current manuscript mainline record
- current seven-table report artifacts
- current replay evaluation summaries
- split files needed for integrity checks

## Mainline policy

The formal public result in this release is:

- `egsr_clean_mainline_replay`
- `LS 99.0 / AS 95.2 / PR 83.7 / TS 83.0`

The older retained row:

- `postproc5_round1_canon`
- `LS 99.0 / AS 95.3 / PR 83.8 / TS 83.1`

is kept only as historical audit context in the manuscript-facing records and
is not the default paper result, because the current code and retained clean
inputs replay to the `egsr_clean_mainline_replay` row above.
