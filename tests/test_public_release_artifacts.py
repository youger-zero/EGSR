from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "experiments" / "outputs_curated" / "reports"
DOCS = ROOT / "docs"


class TestPublicReleaseArtifacts(unittest.TestCase):
    def test_required_release_files_exist(self) -> None:
        required = [
            REPORTS / "TABLE1_main_results_pgdp5k_test_current_replay.json",
            REPORTS / "TABLE2_locked_replay_chain_current_replay.json",
            REPORTS / "TABLE3_topk_candidate_quality_current_replay.json",
            REPORTS / "TABLE4_downstream_ablation_locked_mainline.json",
            REPORTS / "TABLE5_internal_consistency_locked_mainline_variants.json",
            REPORTS / "TABLE6_generic_postprocessing_ablation_locked_mainline.json",
            REPORTS / "TABLE7_k_value_runtime_tradeoff_locked_pipeline.json",
            ROOT / "experiments" / "outputs_curated" / "mainline" / "CURRENT_PUBLIC_MAINLINE.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"missing required release file: {path}")

    def test_main_result_values_match_release_claim(self) -> None:
        path = REPORTS / "TABLE1_main_results_pgdp5k_test_current_replay.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        egsr_rows = [row for row in payload["rows"] if row["Method"] == "EGSR (Ours)"]
        self.assertEqual(len(egsr_rows), 1)
        row = egsr_rows[0]
        self.assertEqual((row["LS"], row["AS"], row["PR"], row["TS"]), (99.0, 95.2, 83.7, 83.0))

    def test_release_has_no_obvious_local_paths(self) -> None:
        suspicious = ["C:" + "\\Users", "D:" + "\\夸克下载", "App" + "Data", "Z" + "XH"]
        ignored_parts = {".git", "__pycache__"}
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            if ignored_parts & set(path.parts) or path.suffix in {".pyc", ".pyo"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in suspicious:
                self.assertNotIn(token, text, f"{path} contains suspicious token {token}")

    def test_build_public_release_tables_script(self) -> None:
        cmd = [sys.executable, "experiments/scripts/build_public_release_tables.py"]
        completed = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        out = DOCS / "PAPER_TABLES.md"
        self.assertTrue(out.exists())
        text = out.read_text(encoding="utf-8")
        self.assertIn("Table 1. Main Results on the PGDP5K Test Split", text)
        self.assertIn("EGSR (Ours)", text)


if __name__ == "__main__":
    unittest.main()
