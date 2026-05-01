"""
Regression tests for the clean no-oracle public mainline.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "experiments" / "scripts"
SPLITS_DIR = PROJECT_ROOT / "experiments" / "data" / "PGDP5K" / "splits"

FORBIDDEN_CALLS = {
    "load_ground_truth",
    "load_annotations",
    "load_annotation",
    "load_gt_logic_form",
}

PIPELINE_STAGE_ORDER = [
    "build_candidates_clean.py",
    "assemble_scenes.py",
    "execute_constraints.py",
    "score_visual.py",
    "score_symbolic.py",
    "rerank_scenes_public.py",
    "project_logic_form.py",
    "eval_pgdp_metrics.py",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class TestNoOracleCompliance(unittest.TestCase):
    def test_build_candidates_clean_does_not_call_oracle_loaders(self) -> None:
        path = SCRIPTS_DIR / "build_candidates_clean.py"
        tree = ast.parse(_read(path), filename=str(path))
        called = {
            name
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            for name in [_call_name(node)]
            if name
        }
        self.assertTrue(FORBIDDEN_CALLS.isdisjoint(called), f"forbidden calls found: {called & FORBIDDEN_CALLS}")

    def test_pipeline_runs_evaluation_once_at_end(self) -> None:
        path = SCRIPTS_DIR / "run_egsr_core_clean.py"
        tree = ast.parse(_read(path), filename=str(path))
        ordered = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) != "_run":
                continue
            if not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                ordered.append(first.value)
        self.assertEqual(ordered, PIPELINE_STAGE_ORDER)

    def test_no_forbidden_id_specific_tokens_in_core_scripts(self) -> None:
        suspicious_tokens = {
            "PATCH_SAMPLE_IDS",
            "HARDCODED_FIXES",
            "TEST_SAMPLE_OVERRIDES",
            "remove_map",
            "replace_map",
            "add_map",
        }
        core_scripts = [
            "build_candidates_clean.py",
            "assemble_scenes.py",
            "execute_constraints.py",
            "score_visual.py",
            "score_symbolic.py",
            "rerank_scenes_public.py",
            "project_logic_form.py",
            "run_egsr_core_clean.py",
        ]
        for name in core_scripts:
            script_file = SCRIPTS_DIR / name
            text = _read(script_file)
            for token in suspicious_tokens:
                self.assertNotIn(token, text, f"{script_file.name} contains suspicious token {token}")

    def test_clean_candidate_default_source_is_public_safe(self) -> None:
        text = _read(SCRIPTS_DIR / "build_candidates_clean.py")
        self.assertIn('default="detector_plus_ocr_rules_plus_light_vlm"', text)

    def test_public_projection_release_supports_only_no_rewrite_mode(self) -> None:
        text = _read(SCRIPTS_DIR / "project_logic_form.py")
        self.assertIn('choices=["none"]', text)
        self.assertNotIn('rewrite_mode == "public"', text)
        self.assertNotIn('rewrite_mode == "full"', text)


class TestDataSplitIntegrity(unittest.TestCase):
    def test_no_train_test_overlap(self) -> None:
        train_ids = set((SPLITS_DIR / "train.txt").read_text(encoding="utf-8").split())
        test_ids = set((SPLITS_DIR / "test.txt").read_text(encoding="utf-8").split())
        self.assertEqual(train_ids & test_ids, set())

    def test_split_sizes_correct(self) -> None:
        expected_sizes = {"train.txt": 3500, "val.txt": 500, "test.txt": 1000}
        for name, expected in expected_sizes.items():
            actual = len((SPLITS_DIR / name).read_text(encoding="utf-8").split())
            self.assertEqual(actual, expected, f"{name} size mismatch")


if __name__ == "__main__":
    unittest.main()
