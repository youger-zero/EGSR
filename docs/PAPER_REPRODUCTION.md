# Paper Reproduction

This public repository supports two levels of reproduction.

## 1. Rebuild the manuscript tables from the bundled public artifacts

This path is fully self-contained inside the GitHub release tree.

Run:

```bash
python experiments/scripts/build_public_release_tables.py
```

This regenerates:

- `docs/PAPER_TABLES.md`

from the retained public report JSON files in:

- `experiments/outputs_curated/reports/`

This is the recommended path for checking that the paper tables bundled with
the public repository are internally consistent.

Expected output:

- `docs/PAPER_TABLES.md`

The generated markdown should match the retained JSON table artifacts under:

- `experiments/outputs_curated/reports/`

## 2. Replay the clean mainline pipeline

The repository also includes the clean replay code and the replay metadata for:

- `egsr_clean_mainline_replay`

To run the pipeline itself, you still need to provide the external PGDP5K data
bundle and the official PGDP evaluator repository under the expected local
layout described in:

- [DATA_SETUP.md](DATA_SETUP.md)

Main entry point:

```bash
python experiments/scripts/run_egsr_core_clean.py --help
```

Formal replay record:

- variant: `egsr_clean_mainline_replay`
- metrics file: `experiments/outputs/eval/egsr_clean_mainline_replay/test/pgdp_metrics.json`
- oracle-K file: `experiments/outputs/eval/egsr_clean_mainline_replay/test/oracle_k_metrics.json`
- run metadata: `experiments/outputs/pipelines/egsr_clean_mainline_replay/test/_run_meta.json`

Minimum paper-facing verification chain:

```bash
python experiments/scripts/build_public_release_tables.py
python -m unittest tests/test_no_oracle_compliance.py tests/test_public_release_artifacts.py
```

If external PGDP5K resources are available and a full replay is desired, run
the clean replay entry point with the same locked configuration recorded in the
bundled run metadata. The replay result must be compared against:

- `experiments/outputs_curated/mainline/CURRENT_PUBLIC_MAINLINE.md`

## Validation

Run the public release tests:

```bash
python -m unittest tests/test_no_oracle_compliance.py tests/test_public_release_artifacts.py
```

These tests verify:

- the public release keeps only the clean no-rewrite projection path
- the public release contains the expected seven table artifacts
- the public release main result matches the bundled manuscript result
- the release tree does not contain obvious local machine path residues
- the repository remains suitable for public audit after export cleanup
