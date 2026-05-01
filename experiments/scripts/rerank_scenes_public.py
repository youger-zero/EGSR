"""
Phase 5: public-safe joint reranking for EGSR candidate scenes.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: Public-safe scene reranking with only generic structural preferences.
INPUTS: Scene hypotheses plus visual / geometry / symbolic scores.
OUTPUTS: Ranked scene list written back into score artifacts.
GT USAGE: None.
PURPOSE: Public mainline reranking path isolated from legacy handcrafted logic.
LAST UPDATED: 2026-05-01
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    consistency_mode_weights,
    default_weights,
    read_json,
    read_split_ids,
    read_stage_sample,
    slugify,
    write_json,
    write_run_meta,
)


CONSISTENCY_MODES = (
    "full",
    "geo_only",
    "visual_only",
    "symbolic_only",
    "visual_geo",
    "visual_symbolic",
    "geo_symbolic",
)


def _coarse_variant_penalty(assembly: Dict[str, Any]) -> float:
    penalty = 0.0
    degree_arc_policy = str(assembly.get("degree_arc_policy", ""))
    global_angle_mode = str(assembly.get("global_angle_mode", "none"))

    if global_angle_mode == "all" and degree_arc_policy == "default":
        penalty += 0.002
    if not bool(assembly.get("refined_line_naming", True)):
        penalty += 0.001
    if global_angle_mode == "selective" and int(assembly.get("sample_circle_count", 0)) == 0:
        penalty += 0.001
    if float(assembly.get("assembly_score", 0.0)) <= 0.4:
        penalty += 0.05
    return penalty


def _public_structural_penalty(assembly: Dict[str, Any]) -> float:
    assembly_score = float(assembly.get("assembly_score", 0.0))
    relation_coverage = float(assembly.get("relation_coverage", 0.0))
    symbol_support = float(assembly.get("symbol_support", 0.0))
    level = int(assembly.get("level", 0))

    bonus = 0.06 * assembly_score + 0.015 * relation_coverage + 0.015 * symbol_support
    if level == 0:
        bonus += 0.01
    elif level >= 2:
        bonus -= min(0.012, 0.004 * (level - 1))
    return -bonus


def _public_exact_tie_preference(assembly: Dict[str, Any]) -> float:
    circle_count = int(assembly.get("sample_circle_count", 0))
    degree_text_count = int(assembly.get("sample_degree_text_count", 0))
    symbolic_degree_count = int(assembly.get("sample_symbolic_degree_count", 0))
    geometry_symbol_count = int(assembly.get("sample_geometry_symbol_count", 999))
    legacy_simple_regime = (
        circle_count == 0
        and degree_text_count == 0
        and symbolic_degree_count == 0
        and geometry_symbol_count <= 12
    )
    public_refined_regime = degree_text_count > 0 or geometry_symbol_count <= 8
    if not (legacy_simple_regime or public_refined_regime):
        return 0.0
    return 1.0 if bool(assembly.get("refined_line_naming", True)) else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerank EGSR scene hypotheses with the public-safe path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="EGSR Public Rerank")
    parser.add_argument(
        "--selection-policy",
        default="rerank",
        choices=("rerank", "assembly_first"),
        help="Choose whether projection should follow reranked selection or the first assembled scene.",
    )
    parser.add_argument(
        "--variant-penalty-mode",
        default="public_structural_raw",
        choices=("coarse", "public_structural", "public_structural_raw", "none"),
    )
    parser.add_argument("--consistency-mode", default="full", choices=CONSISTENCY_MODES)
    parser.add_argument("--disable-hard-filtering", action="store_true")
    parser.add_argument("--disable-min-exp-bias", action="store_true")
    parser.add_argument("--alpha", type=float, default=default_weights()["alpha"])
    parser.add_argument("--beta", type=float, default=default_weights()["beta"])
    parser.add_argument("--gamma", type=float, default=default_weights()["gamma"])
    parser.add_argument("--lambda-penalty", type=float, default=default_weights()["lambda"])
    parser.add_argument("--eta1", type=float, default=default_weights()["eta1"])
    parser.add_argument("--eta2", type=float, default=default_weights()["eta2"])
    parser.add_argument("--assembly-bonus", type=float, default=0.012)
    parser.add_argument("--relation-bonus", type=float, default=0.004)
    parser.add_argument("--symbol-bonus", type=float, default=0.004)
    parser.add_argument(
        "--tie-break-mode",
        default="legacy",
        choices=("legacy", "geo_first", "visual_first", "balanced"),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _scene_score_maps(score_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    maps: Dict[str, Dict[str, Any]] = {}
    for block_key, score_key in (
        ("constraint_execution", "s_geo"),
        ("visual_scoring", "s_visual"),
        ("symbolic_scoring", "s_sym"),
    ):
        block = score_payload.get(block_key, {})
        for row in block.get("scenes", []):
            scene_id = row.get("scene_id")
            if not scene_id:
                continue
            maps.setdefault(scene_id, {})
            maps[scene_id][score_key] = float(row.get(score_key, 0.0))
            if block_key == "constraint_execution":
                maps[scene_id]["feasible"] = bool(row.get("feasible", False))
                maps[scene_id]["residual"] = float(row.get("residual", 1.0))
    return maps


def _raw_min_exp(scene: Dict[str, Any], eta1: float, eta2: float) -> Tuple[float, Dict[str, float]]:
    min_exp = scene.get("min_explanation", {})
    extra = float(min_exp.get("extra_primitives", 0))
    redundant = float(min_exp.get("redundant_relations", 0))
    discrete = scene.get("discrete_structure", {})
    active_n = max(len(discrete.get("active_primitives", [])), 1)
    rel_n = max(len(discrete.get("relations", [])), 1)
    extra_ratio = extra / active_n
    redundant_ratio = redundant / rel_n
    c_raw = eta1 * extra_ratio + eta2 * redundant_ratio
    return c_raw, {
        "extra_primitives": extra,
        "redundant_relations": redundant,
        "active_count": float(active_n),
        "relation_count": float(rel_n),
        "extra_ratio": extra_ratio,
        "redundant_ratio": redundant_ratio,
    }


def _variant_penalty(args: argparse.Namespace, assembly: Dict[str, Any], c_raw: float) -> float:
    if args.variant_penalty_mode == "coarse":
        return _coarse_variant_penalty(assembly)
    if args.variant_penalty_mode == "public_structural":
        return _public_structural_penalty(assembly)
    if args.variant_penalty_mode == "none":
        return 0.0

    closure_penalty = float(assembly.get("closure_penalty", 0.0))
    invalid_closure = float(assembly.get("invalid_closure_count", 0.0))
    return (
        _public_structural_penalty(assembly)
        + 0.08 * c_raw
        + 0.04 * closure_penalty
        + 0.002 * min(invalid_closure, 10.0)
    )


def _final_score(
    args: argparse.Namespace,
    alpha: float,
    beta: float,
    gamma: float,
    s_visual: float,
    s_geo: float,
    s_sym: float,
    c_min: float,
    c_raw: float,
    feasible: bool,
    assembly: Dict[str, Any],
) -> Tuple[float, float]:
    variant_penalty = _variant_penalty(args, assembly, c_raw)
    total = alpha * s_visual + beta * s_geo + gamma * s_sym - args.lambda_penalty * c_min - variant_penalty
    if args.variant_penalty_mode == "public_structural_raw":
        total += (
            args.assembly_bonus * float(assembly.get("assembly_score", 0.0))
            + args.relation_bonus * float(assembly.get("relation_coverage", 0.0))
            + args.symbol_bonus * float(assembly.get("symbol_support", 0.0))
        )
    if (not args.disable_hard_filtering) and (not feasible):
        total -= 10.0
    return total, variant_penalty


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    base_weights = {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma}
    mode_weights = consistency_mode_weights(args.consistency_mode, base_weights)
    alpha = mode_weights["alpha"]
    beta = mode_weights["beta"]
    gamma = mode_weights["gamma"]

    written = 0
    for sample_id in split_ids:
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        score_file = args.outputs_dir / "scores" / slugify(args.variant) / args.split / f"{sample_id}.json"
        if not score_file.exists():
            continue
        score_payload = read_json(score_file)
        maps = _scene_score_maps(score_payload)

        scenes = scene_payload.get("scenes", [])
        raw_penalties: Dict[str, float] = {}
        penalty_meta: Dict[str, Dict[str, float]] = {}
        for scene in scenes:
            scene_id = scene.get("scene_id")
            c_raw, pmeta = _raw_min_exp(scene, args.eta1, args.eta2)
            if scene_id:
                raw_penalties[scene_id] = c_raw
                penalty_meta[scene_id] = pmeta
        min_c = min(raw_penalties.values()) if raw_penalties else 0.0
        max_c = max(raw_penalties.values()) if raw_penalties else 1.0
        scale = max(max_c - min_c, 0.05)

        ranked: List[Dict[str, Any]] = []
        scene_by_id = {scene.get("scene_id"): scene for scene in scenes if scene.get("scene_id")}
        for scene in scenes:
            scene_id = scene.get("scene_id")
            scene_scores = maps.get(scene_id, {})
            s_geo = float(scene_scores.get("s_geo", 0.0))
            s_visual = float(scene_scores.get("s_visual", 0.0))
            s_sym = float(scene_scores.get("s_sym", 0.0))
            feasible = bool(scene_scores.get("feasible", False))
            assembly = scene.get("assembly", {})
            c_raw = raw_penalties.get(scene_id, 0.0)
            c_min = (c_raw - min_c) / scale
            if args.disable_min_exp_bias:
                c_min = 0.0

            total, variant_penalty = _final_score(
                args, alpha, beta, gamma, s_visual, s_geo, s_sym, c_min, c_raw, feasible, assembly
            )
            ranked.append(
                {
                    "scene_id": scene_id,
                    "final_score": round(total, 6),
                    "score_breakdown": {
                        "s_visual": round(s_visual, 6),
                        "s_geo": round(s_geo, 6),
                        "s_sym": round(s_sym, 6),
                        "c_min_exp_raw": round(c_raw, 6),
                        "c_min_exp": round(c_min, 6),
                        "variant_penalty": round(variant_penalty, 6),
                        "feasible": feasible,
                        "penalty_meta": penalty_meta.get(scene_id, {}),
                    },
                }
            )

        def _rank_key(row: Dict[str, Any]):
            block = row["score_breakdown"]
            final_score = float(row["final_score"])
            feasible_key = 1.0 if block.get("feasible") else 0.0
            s_visual = float(block.get("s_visual", 0.0))
            s_geo = float(block.get("s_geo", 0.0))
            s_sym = float(block.get("s_sym", 0.0))
            c_min = float(block.get("c_min_exp", 0.0))
            c_raw = float(block.get("c_min_exp_raw", 0.0))
            residual = float(maps.get(row["scene_id"], {}).get("residual", 1.0))
            assembly = scene_by_id.get(row["scene_id"], {}).get("assembly", {})
            public_tie_pref = _public_exact_tie_preference(assembly)
            if args.tie_break_mode == "geo_first":
                return (final_score, feasible_key, s_geo, s_visual, -c_min, -c_raw, -residual, s_sym, row["scene_id"])
            if args.tie_break_mode == "visual_first":
                return (final_score, feasible_key, s_visual, s_geo, -c_min, -c_raw, -residual, s_sym, row["scene_id"])
            if args.tie_break_mode == "balanced":
                return (final_score, feasible_key, s_visual + s_geo, s_geo, s_visual, -c_min, -c_raw, -residual, s_sym, row["scene_id"])
            return (
                final_score,
                feasible_key,
                public_tie_pref,
                s_sym,
                s_geo,
                s_visual,
                -residual,
                -c_min,
                -c_raw,
                row["scene_id"],
            )

        ranked.sort(key=_rank_key, reverse=True)
        if args.selection_policy == "assembly_first":
            selected_scene_id = scenes[0].get("scene_id") if scenes else None
        else:
            selected_scene_id = ranked[0]["scene_id"] if ranked else None

        score_payload["reranking"] = {
            "module": "rerank_scenes_public",
            "selection_policy": args.selection_policy,
            "consistency_mode": args.consistency_mode,
            "tie_break_mode": args.tie_break_mode,
            "variant_penalty_mode": args.variant_penalty_mode,
            "weights": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "lambda_penalty": args.lambda_penalty,
                "eta1": args.eta1,
                "eta2": args.eta2,
                "assembly_bonus": args.assembly_bonus,
                "relation_bonus": args.relation_bonus,
                "symbol_bonus": args.symbol_bonus,
            },
            "hard_filtering_enabled": not args.disable_hard_filtering,
            "min_exp_enabled": not args.disable_min_exp_bias,
            "selected_scene_id": selected_scene_id,
            "ranked_scenes": ranked,
        }
        write_json(score_file, score_payload)
        written += 1

    elapsed = time.perf_counter() - begin
    write_run_meta(
        "scores",
        args.variant,
        args.split,
        {
            "module": "reranking_public",
            "selection_policy": args.selection_policy,
            "consistency_mode": args.consistency_mode,
            "variant_penalty_mode": args.variant_penalty_mode,
            "tie_break_mode": args.tie_break_mode,
            "hard_filtering_enabled": not args.disable_hard_filtering,
            "min_exp_enabled": not args.disable_min_exp_bias,
            "num_requested": len(split_ids),
            "num_written": written,
            "elapsed_sec": round(elapsed, 3),
        },
        args.outputs_dir,
    )
    print(
        f"[rerank_scenes_public] split={args.split} variant={args.variant} policy={args.selection_policy} mode={args.consistency_mode} "
        f"penalty={args.variant_penalty_mode} tie_break={args.tie_break_mode} "
        f"written={written} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
