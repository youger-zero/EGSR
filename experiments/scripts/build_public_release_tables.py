"""
Rebuild manuscript-facing paper tables from the retained public release JSON.

This script is designed for the GitHub release tree. It does not rerun the full
pipeline. Instead, it reconstructs the final manuscript tables from the locked
public report artifacts that are bundled in the release.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "experiments" / "outputs_curated" / "reports"
DOCS = ROOT / "docs"


TABLE_FILES = [
    "TABLE1_main_results_pgdp5k_test_current_replay.json",
    "TABLE2_locked_replay_chain_current_replay.json",
    "TABLE3_topk_candidate_quality_current_replay.json",
    "TABLE4_downstream_ablation_locked_mainline.json",
    "TABLE5_internal_consistency_locked_mainline_variants.json",
    "TABLE6_generic_postprocessing_ablation_locked_mainline.json",
    "TABLE7_k_value_runtime_tradeoff_locked_pipeline.json",
]


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "_No rows_"
    keys = [k for k in rows[0].keys() if k != "variant_slug" and k != "stage_timings_sec"]
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body = []
    for row in rows:
        vals = [str(row.get(k, "")) for k in keys]
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def main() -> None:
    sections: List[str] = ["# Paper Tables", ""]
    for file_name in TABLE_FILES:
        payload = read_json(REPORTS / file_name)
        title = payload.get("title", file_name)
        sections.append(f"## {title}")
        sections.append("")
        rows = payload.get("rows")
        if isinstance(rows, list):
            sections.append(markdown_table(rows))
        else:
            sections.append("```json")
            sections.append(json.dumps(payload, ensure_ascii=False, indent=2))
            sections.append("```")
        note = payload.get("note")
        if note:
            sections.append("")
            sections.append(f"Note: {note}")
        timing_protocol = payload.get("timing_protocol")
        if timing_protocol:
            sections.append("")
            sections.append("Timing protocol:")
            sections.append("")
            for k, v in timing_protocol.items():
                sections.append(f"- `{k}`: {v}")
        sections.append("")

    out = DOCS / "PAPER_TABLES.md"
    out.write_text("\n".join(sections), encoding="utf-8")
    print(f"[build_public_release_tables] wrote {out}")


if __name__ == "__main__":
    main()
