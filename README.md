# EGSR

EGSR is an executable geometric scene reconstruction pipeline for PGDP-style
geometry diagram understanding.

This repository is the clean public release used for manuscript reporting. It
contains the locked mainline record, the retained paper tables, replay
metadata, and the public verification tests needed to inspect and audit the
reported result.

For a quick audit, read in this order:

1. `experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md`
2. `docs/PAPER_TABLES.md`
3. `docs/PAPER_REPRODUCTION.md`

## Release scope

This release keeps only the files needed to:

- inspect the clean no-oracle mainline
- reproduce the reported pipeline with the official PGDP resources
- verify the public-safe protocol
- read the final seven manuscript tables

This release does not include:

- private local scripts
- remote sync wrappers
- historical dirty tuning artifacts
- sample-specific rewrite logic
- local machine paths or personal environment records

## Formal manuscript result

- method label: `EGSR (Ours)`
- formal replay variant: `egsr_clean_mainline_replay`
- split: `PGDP5K test`
- metrics: `LS 99.0 / AS 95.2 / PR 83.7 / TS 83.0`

In the official PGDP5K evaluation, this locked mainline is competitive with
PGDPNet and outperforms PGDPNet w/o GNN on all four reported metrics.

The authoritative public mainline tracker is:

- [experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md](experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md)

The seven-table summary used for the manuscript is:

- [experiments/outputs_curated/reports/SEVEN_TABLES_CURRENT_REPLAY_SUMMARY.md](experiments/outputs_curated/reports/SEVEN_TABLES_CURRENT_REPLAY_SUMMARY.md)

## Repository structure

- `experiments/scripts/`: clean replay, evaluation, and diagnostic scripts
- `tests/`: public compliance regression tests
- `experiments/outputs_curated/`: final report tables and mainline records
- `experiments/outputs/`: retained public result summaries for the formal replay
- `experiments/data/PGDP5K/splits/`: split files only
- `docs/`: data setup and public-release audit notes

## Data setup

This repository does not bundle the full PGDP5K dataset or the official PGDP
evaluation repository.

To run the pipeline, place the required external resources under:

- `experiments/data/PGDP5K/`
- `experiments/data/PGDP/`

See:

- [docs/DATA_SETUP.md](docs/DATA_SETUP.md)

## Public replay

The main replay entry point is:

```bash
python experiments/scripts/run_egsr_core_clean.py --help
```

The formal paper row is backed by these retained result files:

- `experiments/outputs/eval/egsr_clean_mainline_replay/test/pgdp_metrics.json`
- `experiments/outputs/eval/egsr_clean_mainline_replay/test/oracle_k_metrics.json`
- `experiments/outputs/pipelines/egsr_clean_mainline_replay/test/_run_meta.json`

To rebuild the manuscript tables from the bundled public artifacts:

```bash
python experiments/scripts/build_public_release_tables.py
```

The no-oracle compliance test is:

```bash
python -m unittest tests/test_no_oracle_compliance.py
```

For the full public release validation:

```bash
python -m unittest tests/test_no_oracle_compliance.py tests/test_public_release_artifacts.py
```

Recommended public verification flow:

1. run `python experiments/scripts/build_public_release_tables.py`
2. run `python -m unittest tests/test_no_oracle_compliance.py tests/test_public_release_artifacts.py`
3. inspect `docs/PAPER_TABLES.md`
4. inspect `experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md`

Paper reproduction details:

- [docs/PAPER_REPRODUCTION.md](docs/PAPER_REPRODUCTION.md)

## Code Availability

The public repository for the clean locked mainline is:

- [https://github.com/youger-zero/EGSR](https://github.com/youger-zero/EGSR)

The repository includes:

- the locked mainline record
- the seven manuscript tables
- replay metadata for the formal result
- public compliance and release-audit tests

For a paper-ready wording, see:

- [docs/CODE_AVAILABILITY.md](docs/CODE_AVAILABILITY.md)

## Release audit

The release-audit note is:

- [docs/PUBLIC_RELEASE_AUDIT.md](docs/PUBLIC_RELEASE_AUDIT.md)

It explains what was removed from the original working directory before this
public export was created.
