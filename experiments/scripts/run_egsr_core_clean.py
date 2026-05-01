"""
EGSR-Core clean no-oracle pipeline runner.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: Run the public-safe EGSR-Core pipeline from clean candidate artifacts.
INPUTS: Clean candidate source artifacts plus image-derived intermediate stages.
OUTPUTS: Candidate, scene, score, logic-form, and evaluation artifacts for one variant.
GT USAGE: None inside the inference stages; final evaluation is handled by eval_pgdp_metrics.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from egsr_common import OUTPUTS_DIR, env_hardware_snapshot, slugify, write_json


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean EGSR-Core pipeline.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="EGSR-Core Clean")
    parser.add_argument(
        "--source-variant",
        default="detector_plus_ocr_rules_plus_light_vlm",
        help="Clean candidate artifact source. Avoid oracle-assisted variants such as b5_egsr_full.",
    )
    parser.add_argument("--text-semantic-bonus", type=float)
    parser.add_argument("--structural-symbol-bonus", type=float)
    parser.add_argument("--invalid-text-penalty", type=float)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--consistency-mode",
        default="full",
        choices=[
            "full",
            "geo_only",
            "visual_only",
            "symbolic_only",
            "visual_geo",
            "visual_symbolic",
            "geo_symbolic",
        ],
    )
    parser.add_argument("--disable-hard-filtering", action="store_true")
    parser.add_argument("--disable-min-exp-bias", action="store_true")
    parser.add_argument(
        "--tie-break-mode",
        choices=["legacy", "geo_first", "visual_first", "balanced"],
        default="legacy",
    )
    parser.add_argument(
        "--variant-penalty-mode",
        choices=["coarse", "public_structural", "public_structural_raw", "none"],
        default="coarse",
        help="Use only the isolated public-safe reranking modes.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=["rerank", "assembly_first"],
        default="rerank",
        help="Choose whether projection follows reranked selection or the first assembled scene.",
    )
    parser.add_argument(
        "--rewrite-mode",
        choices=["none"],
        default="none",
        help="Public release supports only the clean no-rewrite path.",
    )
    parser.add_argument(
        "--public-postprocess-mode",
        choices=[
            "off",
            "canonical_only",
            "canonical_drop_relation_conflicts",
            "canonical_drop_scalar_conflicts",
            "canonical_drop_all_conflicts",
            "canonical_add_parallel_closure",
            "canonical_add_equal_closure",
            "canonical_add_all_closure",
        ],
        default="off",
    )
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lambda-penalty", type=float)
    parser.add_argument("--eta1", type=float)
    parser.add_argument("--eta2", type=float)
    parser.add_argument("--assembly-bonus", type=float)
    parser.add_argument("--relation-bonus", type=float)
    parser.add_argument("--symbol-bonus", type=float)
    parser.add_argument("--w1", type=float)
    parser.add_argument("--w2", type=float)
    parser.add_argument("--w3", type=float)
    parser.add_argument("--mu1", type=float)
    parser.add_argument("--mu2", type=float)
    parser.add_argument("--w-consistency", type=float)
    parser.add_argument("--w-binding", type=float)
    parser.add_argument("--w-support", type=float)
    parser.add_argument("--support-evidence-mode", choices=["count", "conf_sqrt"])
    parser.add_argument("--support-length-weight", type=float)
    parser.add_argument("--support-angle-weight", type=float)
    parser.add_argument("--support-parallel-weight", type=float)
    parser.add_argument("--support-perpendicular-weight", type=float)
    parser.add_argument("--duplicate-binding-penalty", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--literal-proxy-penalty", type=float)
    parser.add_argument("--unparsed-penalty", type=float)
    parser.add_argument("--missing-point-penalty", type=float)
    parser.add_argument("--missing-line-relation-penalty", type=float)
    parser.add_argument("--missing-length-value-penalty", type=float)
    parser.add_argument("--missing-angle-value-penalty", type=float)
    parser.add_argument("--length-literal-scale-mode", choices=["none", "median_ratio"])
    parser.add_argument("--length-literal-scale-min-pairs", type=int)
    parser.add_argument("--length-literal-scale-mix", type=float)
    parser.add_argument("--infeasible-residual-bump", type=float)
    parser.add_argument("--infeasible-geo-scale", type=float)
    parser.add_argument("--point-on-line-norm-max", type=float)
    parser.add_argument("--point-on-circle-cluster-diag-tol", type=float)
    parser.add_argument("--point-on-circle-cluster-rel-tol", type=float)
    parser.add_argument("--point-on-circle-min-radius-norm", type=float)
    parser.add_argument("--center-on-line-abs-min", type=float)
    parser.add_argument("--len-text-segment-margin-min", type=float)
    parser.add_argument("--angle-global-margin", type=float)
    parser.add_argument("--perpendicular-global-margin", type=float)
    parser.add_argument("--grouped-line-duplicate-penalty", type=float)
    parser.add_argument("--parallel-companion-weight", type=float)
    parser.add_argument("--len-text-mark-support-bonus", type=float)
    parser.add_argument("--line-host-alt-gap-max", type=float)
    parser.add_argument("--line-host-alt-topm", type=int)
    parser.add_argument("--pair-host-alt-gap-max", type=float)
    parser.add_argument("--pair-host-alt-topm", type=int)
    parser.add_argument("--alt-penalty-scale", type=float)
    parser.add_argument(
        "--force-degree-arc-policy",
        choices=["default", "selective_arc", "conservative_arc", "strict_selective_arc", "clustered_selective_arc"],
    )
    parser.add_argument(
        "--force-global-angle-mode",
        choices=["none", "all", "selective", "none_lenbackup"],
    )
    parser.add_argument(
        "--force-line-completion-mode",
        choices=["strict", "default", "relaxed"],
    )
    parser.add_argument(
        "--force-strict-circle-completion",
        choices=["auto", "on", "off"],
    )
    parser.add_argument(
        "--allow-nonclean-source",
        action="store_true",
        help="Override the source cleanliness guard. Do not use for public-safe runs.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _run(script_name: str, args: List[str]) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / script_name), *args]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="")
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def _candidate_meta_path(outputs_dir: Path, variant: str, split: str) -> Path:
    return outputs_dir / "candidates" / slugify(variant) / split / "_run_meta.json"


def _assert_clean_source(outputs_dir: Path, source_variant: str, split: str, allow_nonclean_source: bool) -> None:
    meta_path = _candidate_meta_path(outputs_dir, source_variant, split)
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    risky_keys = {
        "geometry_source": "annotation_geos",
        "canonical_alignment": "annotation_text_point_linking",
    }
    matches = [
        f"{key}={meta.get(key)!r}"
        for key, risky_value in risky_keys.items()
        if meta.get(key) == risky_value
    ]
    if matches and not allow_nonclean_source:
        joined = ", ".join(matches)
        raise RuntimeError(
            "Refusing to run clean pipeline with non-clean candidate source "
            f"{source_variant!r} ({joined}). Use a clean source variant or "
            "--allow-nonclean-source only for non-public diagnostics."
        )


def _write_pipeline_meta(
    outputs_dir: Path,
    variant: str,
    split: str,
    source_variant: str,
    k: int,
    limit: int,
    variant_penalty_mode: str,
    rewrite_mode: str,
    elapsed: float,
    config: Dict[str, object],
) -> None:
    meta_path = outputs_dir / "pipelines" / slugify(variant) / split / "_run_meta.json"
    write_json(
        meta_path,
        {
            "variant": variant,
            "split": split,
            "source_variant": source_variant,
            "k": k,
            "limit": limit,
            "variant_penalty_mode": variant_penalty_mode,
            "rewrite_mode": rewrite_mode,
            "config": config,
            "elapsed_sec": round(elapsed, 3),
            "hardware": env_hardware_snapshot(),
            "stages": [
                "build_candidates_clean",
                "assemble_scenes",
                "execute_constraints",
                "score_visual",
                "score_symbolic",
                "rerank_scenes_public",
                "project_logic_form",
                "eval_pgdp_metrics",
            ],
            "scripts_used": [
                "build_candidates_clean.py",
                "assemble_scenes.py",
                "execute_constraints.py",
                "score_visual.py",
                "score_symbolic.py",
                "rerank_scenes_public.py",
                "project_logic_form.py",
                "eval_pgdp_metrics.py",
            ],
        },
    )


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    _assert_clean_source(
        outputs_dir=args.outputs_dir,
        source_variant=args.source_variant,
        split=args.split,
        allow_nonclean_source=args.allow_nonclean_source,
    )

    common = ["--split", args.split, "--variant", args.variant, "--outputs-dir", str(args.outputs_dir)]
    if args.limit > 0:
        common.extend(["--limit", str(args.limit)])

    _run(
        "build_candidates_clean.py",
        common
        + ["--source-variant", args.source_variant]
        + (["--text-semantic-bonus", str(args.text_semantic_bonus)] if args.text_semantic_bonus is not None else [])
        + (["--structural-symbol-bonus", str(args.structural_symbol_bonus)] if args.structural_symbol_bonus is not None else [])
        + (["--invalid-text-penalty", str(args.invalid_text_penalty)] if args.invalid_text_penalty is not None else []),
    )
    assemble_args = common + ["--k", str(args.k)]
    if args.angle_global_margin is not None:
        assemble_args.extend(["--angle-global-margin", str(args.angle_global_margin)])
    if args.perpendicular_global_margin is not None:
        assemble_args.extend(["--perpendicular-global-margin", str(args.perpendicular_global_margin)])
    if args.grouped_line_duplicate_penalty is not None:
        assemble_args.extend(["--grouped-line-duplicate-penalty", str(args.grouped_line_duplicate_penalty)])
    if args.parallel_companion_weight is not None:
        assemble_args.extend(["--parallel-companion-weight", str(args.parallel_companion_weight)])
    if args.len_text_mark_support_bonus is not None:
        assemble_args.extend(["--len-text-mark-support-bonus", str(args.len_text_mark_support_bonus)])
    if args.line_host_alt_gap_max is not None:
        assemble_args.extend(["--line-host-alt-gap-max", str(args.line_host_alt_gap_max)])
    if args.line_host_alt_topm is not None:
        assemble_args.extend(["--line-host-alt-topm", str(args.line_host_alt_topm)])
    if args.pair_host_alt_gap_max is not None:
        assemble_args.extend(["--pair-host-alt-gap-max", str(args.pair_host_alt_gap_max)])
    if args.pair_host_alt_topm is not None:
        assemble_args.extend(["--pair-host-alt-topm", str(args.pair_host_alt_topm)])
    if args.alt_penalty_scale is not None:
        assemble_args.extend(["--alt-penalty-scale", str(args.alt_penalty_scale)])
    if args.force_degree_arc_policy is not None:
        assemble_args.extend(["--force-degree-arc-policy", args.force_degree_arc_policy])
    if args.force_global_angle_mode is not None:
        assemble_args.extend(["--force-global-angle-mode", args.force_global_angle_mode])
    if args.force_line_completion_mode is not None:
        assemble_args.extend(["--force-line-completion-mode", args.force_line_completion_mode])
    if args.force_strict_circle_completion is not None:
        assemble_args.extend(["--force-strict-circle-completion", args.force_strict_circle_completion])
    _run("assemble_scenes.py", assemble_args)
    constraint_args = list(common)
    if args.tau is not None:
        constraint_args.extend(["--tau", str(args.tau)])
    if args.literal_proxy_penalty is not None:
        constraint_args.extend(["--literal-proxy-penalty", str(args.literal_proxy_penalty)])
    if args.unparsed_penalty is not None:
        constraint_args.extend(["--unparsed-penalty", str(args.unparsed_penalty)])
    if args.missing_point_penalty is not None:
        constraint_args.extend(["--missing-point-penalty", str(args.missing_point_penalty)])
    if args.missing_line_relation_penalty is not None:
        constraint_args.extend(["--missing-line-relation-penalty", str(args.missing_line_relation_penalty)])
    if args.missing_length_value_penalty is not None:
        constraint_args.extend(["--missing-length-value-penalty", str(args.missing_length_value_penalty)])
    if args.missing_angle_value_penalty is not None:
        constraint_args.extend(["--missing-angle-value-penalty", str(args.missing_angle_value_penalty)])
    if args.length_literal_scale_mode is not None:
        constraint_args.extend(["--length-literal-scale-mode", args.length_literal_scale_mode])
    if args.length_literal_scale_min_pairs is not None:
        constraint_args.extend(["--length-literal-scale-min-pairs", str(args.length_literal_scale_min_pairs)])
    if args.length_literal_scale_mix is not None:
        constraint_args.extend(["--length-literal-scale-mix", str(args.length_literal_scale_mix)])
    if args.infeasible_residual_bump is not None:
        constraint_args.extend(["--infeasible-residual-bump", str(args.infeasible_residual_bump)])
    if args.infeasible_geo_scale is not None:
        constraint_args.extend(["--infeasible-geo-scale", str(args.infeasible_geo_scale)])
    _run("execute_constraints.py", constraint_args)

    visual_args = list(common)
    if args.w1 is not None:
        visual_args.extend(["--w1", str(args.w1)])
    if args.w2 is not None:
        visual_args.extend(["--w2", str(args.w2)])
    if args.w3 is not None:
        visual_args.extend(["--w3", str(args.w3)])
    _run("score_visual.py", visual_args)

    symbolic_args = list(common)
    if args.mu1 is not None:
        symbolic_args.extend(["--mu1", str(args.mu1)])
    if args.mu2 is not None:
        symbolic_args.extend(["--mu2", str(args.mu2)])
    if args.w_consistency is not None:
        symbolic_args.extend(["--w-consistency", str(args.w_consistency)])
    if args.w_binding is not None:
        symbolic_args.extend(["--w-binding", str(args.w_binding)])
    if args.w_support is not None:
        symbolic_args.extend(["--w-support", str(args.w_support)])
    if args.support_evidence_mode is not None:
        symbolic_args.extend(["--support-evidence-mode", args.support_evidence_mode])
    if args.support_length_weight is not None:
        symbolic_args.extend(["--support-length-weight", str(args.support_length_weight)])
    if args.support_angle_weight is not None:
        symbolic_args.extend(["--support-angle-weight", str(args.support_angle_weight)])
    if args.support_parallel_weight is not None:
        symbolic_args.extend(["--support-parallel-weight", str(args.support_parallel_weight)])
    if args.support_perpendicular_weight is not None:
        symbolic_args.extend(["--support-perpendicular-weight", str(args.support_perpendicular_weight)])
    if args.duplicate_binding_penalty is not None:
        symbolic_args.extend(["--duplicate-binding-penalty", str(args.duplicate_binding_penalty)])
    _run("score_symbolic.py", symbolic_args)

    rerank_args = list(common) + [
        "--consistency-mode",
        args.consistency_mode,
        "--tie-break-mode",
        args.tie_break_mode,
        "--variant-penalty-mode",
        args.variant_penalty_mode,
        "--selection-policy",
        args.selection_policy,
    ]
    if args.disable_hard_filtering:
        rerank_args.append("--disable-hard-filtering")
    if args.disable_min_exp_bias:
        rerank_args.append("--disable-min-exp-bias")
    if args.alpha is not None:
        rerank_args.extend(["--alpha", str(args.alpha)])
    if args.beta is not None:
        rerank_args.extend(["--beta", str(args.beta)])
    if args.gamma is not None:
        rerank_args.extend(["--gamma", str(args.gamma)])
    if args.lambda_penalty is not None:
        rerank_args.extend(["--lambda-penalty", str(args.lambda_penalty)])
    if args.eta1 is not None:
        rerank_args.extend(["--eta1", str(args.eta1)])
    if args.eta2 is not None:
        rerank_args.extend(["--eta2", str(args.eta2)])
    if args.assembly_bonus is not None:
        rerank_args.extend(["--assembly-bonus", str(args.assembly_bonus)])
    if args.relation_bonus is not None:
        rerank_args.extend(["--relation-bonus", str(args.relation_bonus)])
    if args.symbol_bonus is not None:
        rerank_args.extend(["--symbol-bonus", str(args.symbol_bonus)])
    _run("rerank_scenes_public.py", rerank_args)
    projection_args = common + [
        "--candidate-variant",
        args.variant,
        "--rewrite-mode",
        args.rewrite_mode,
        "--public-postprocess-mode",
        args.public_postprocess_mode,
    ]
    if args.point_on_line_norm_max is not None:
        projection_args.extend(["--point-on-line-norm-max", str(args.point_on_line_norm_max)])
    if args.point_on_circle_cluster_diag_tol is not None:
        projection_args.extend(["--point-on-circle-cluster-diag-tol", str(args.point_on_circle_cluster_diag_tol)])
    if args.point_on_circle_cluster_rel_tol is not None:
        projection_args.extend(["--point-on-circle-cluster-rel-tol", str(args.point_on_circle_cluster_rel_tol)])
    if args.point_on_circle_min_radius_norm is not None:
        projection_args.extend(["--point-on-circle-min-radius-norm", str(args.point_on_circle_min_radius_norm)])
    if args.center_on_line_abs_min is not None:
        projection_args.extend(["--center-on-line-abs-min", str(args.center_on_line_abs_min)])
    if args.len_text_segment_margin_min is not None:
        projection_args.extend(["--len-text-segment-margin-min", str(args.len_text_segment_margin_min)])
    _run("project_logic_form.py", projection_args)
    _run("eval_pgdp_metrics.py", common)

    elapsed = time.perf_counter() - begin
    _write_pipeline_meta(
        outputs_dir=args.outputs_dir,
        variant=args.variant,
        split=args.split,
        source_variant=args.source_variant,
        k=args.k,
        limit=args.limit,
        variant_penalty_mode=args.variant_penalty_mode,
        rewrite_mode=args.rewrite_mode,
        elapsed=elapsed,
        config={
            "consistency_mode": args.consistency_mode,
            "selection_policy": args.selection_policy,
            "disable_hard_filtering": args.disable_hard_filtering,
            "disable_min_exp_bias": args.disable_min_exp_bias,
            "tie_break_mode": args.tie_break_mode,
            "public_postprocess_mode": args.public_postprocess_mode,
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "lambda_penalty": args.lambda_penalty,
            "eta1": args.eta1,
            "eta2": args.eta2,
            "assembly_bonus": args.assembly_bonus,
            "relation_bonus": args.relation_bonus,
            "symbol_bonus": args.symbol_bonus,
            "w1": args.w1,
            "w2": args.w2,
            "w3": args.w3,
            "mu1": args.mu1,
            "mu2": args.mu2,
            "w_consistency": args.w_consistency,
            "w_binding": args.w_binding,
            "w_support": args.w_support,
            "support_evidence_mode": args.support_evidence_mode,
            "support_length_weight": args.support_length_weight,
            "support_angle_weight": args.support_angle_weight,
            "support_parallel_weight": args.support_parallel_weight,
            "support_perpendicular_weight": args.support_perpendicular_weight,
            "duplicate_binding_penalty": args.duplicate_binding_penalty,
            "tau": args.tau,
            "literal_proxy_penalty": args.literal_proxy_penalty,
            "unparsed_penalty": args.unparsed_penalty,
            "missing_point_penalty": args.missing_point_penalty,
            "missing_line_relation_penalty": args.missing_line_relation_penalty,
            "missing_length_value_penalty": args.missing_length_value_penalty,
            "missing_angle_value_penalty": args.missing_angle_value_penalty,
            "length_literal_scale_mode": args.length_literal_scale_mode,
            "length_literal_scale_min_pairs": args.length_literal_scale_min_pairs,
            "length_literal_scale_mix": args.length_literal_scale_mix,
            "infeasible_residual_bump": args.infeasible_residual_bump,
            "infeasible_geo_scale": args.infeasible_geo_scale,
            "point_on_line_norm_max": args.point_on_line_norm_max,
            "point_on_circle_cluster_diag_tol": args.point_on_circle_cluster_diag_tol,
            "point_on_circle_cluster_rel_tol": args.point_on_circle_cluster_rel_tol,
            "point_on_circle_min_radius_norm": args.point_on_circle_min_radius_norm,
            "center_on_line_abs_min": args.center_on_line_abs_min,
            "len_text_segment_margin_min": args.len_text_segment_margin_min,
            "angle_global_margin": args.angle_global_margin,
            "perpendicular_global_margin": args.perpendicular_global_margin,
            "grouped_line_duplicate_penalty": args.grouped_line_duplicate_penalty,
            "parallel_companion_weight": args.parallel_companion_weight,
            "len_text_mark_support_bonus": args.len_text_mark_support_bonus,
            "line_host_alt_gap_max": args.line_host_alt_gap_max,
            "line_host_alt_topm": args.line_host_alt_topm,
            "pair_host_alt_gap_max": args.pair_host_alt_gap_max,
            "pair_host_alt_topm": args.pair_host_alt_topm,
            "alt_penalty_scale": args.alt_penalty_scale,
            "force_degree_arc_policy": args.force_degree_arc_policy,
            "force_global_angle_mode": args.force_global_angle_mode,
            "force_line_completion_mode": args.force_line_completion_mode,
            "force_strict_circle_completion": args.force_strict_circle_completion,
            "text_semantic_bonus": args.text_semantic_bonus,
            "structural_symbol_bonus": args.structural_symbol_bonus,
            "invalid_text_penalty": args.invalid_text_penalty,
        },
    )
    print(f"[run_egsr_core_clean] variant={args.variant} split={args.split} elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
