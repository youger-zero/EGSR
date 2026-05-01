"""
Phase 4 (part A): compute visual consistency scores.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 计算场景与图像的视觉一致性分数
INPUTS: 场景假设、原始图像
OUTPUTS: 视觉一致性分数
GT USAGE: None - 仅比较场景渲染与原始图像
PURPOSE: 核心推理模块 - 三重一致性评分的视觉部分
LAST UPDATED: 2026-04-28

✅ 此脚本不访问测试集标注，可用于正式推理。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    default_weights,
    read_json,
    read_split_ids,
    read_stage_sample,
    slugify,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EGSR visual scores.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--w1", type=float, default=default_weights()["w1"])
    parser.add_argument("--w2", type=float, default=default_weights()["w2"])
    parser.add_argument("--w3", type=float, default=default_weights()["w3"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _candidate_maps(candidate_payload: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    for cand in candidate_payload.get("candidates", []):
        cid = str(cand.get("id", ""))
        if cid:
            by_id[cid] = cand
        name = cand.get("name")
        if name:
            by_name[str(name)] = cand
    return by_id, by_name


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2)


def _line_midpoint_from_location(location: Any) -> Optional[Tuple[float, float]]:
    if not isinstance(location, list) or len(location) < 4:
        return None
    x1, y1, x2, y2 = map(float, location[:4])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _line_score(scene: Dict[str, Any], candidate_by_id: Dict[str, Dict[str, Any]]) -> float:
    observed_lines = {
        str(c["id"]): c
        for c in candidate_by_id.values()
        if str(c.get("type")) == "line"
    }
    if not observed_lines:
        return 1.0

    active = set(scene.get("discrete_structure", {}).get("active_primitives", []))
    active_line_ids = [cid for cid in active if cid in observed_lines]
    if not active_line_ids:
        return 0.0

    score = 0.0
    for cid in active_line_ids:
        score += float(observed_lines[cid].get("confidence", 0.5))
    denom = sum(float(c.get("confidence", 0.5)) for c in observed_lines.values())
    return max(0.0, min(1.0, score / max(denom, 1e-6)))


def _point_score(scene: Dict[str, Any], candidate_by_name: Dict[str, Dict[str, Any]]) -> float:
    pred = scene.get("continuous_parameters", {}).get("point_coordinates", {})
    observed_points = {
        str(c.get("name")): c
        for c in candidate_by_name.values()
        if str(c.get("type")) == "point"
    }
    if not observed_points:
        return 1.0

    scores: List[float] = []
    for name, cand in observed_points.items():
        loc = cand.get("location", [])
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        if name not in pred:
            scores.append(0.0)
            continue
        observed_xy = [float(loc[0]), float(loc[1])]
        pred_xy = pred[name]
        dist = _distance(observed_xy, pred_xy)
        scores.append(math.exp(-dist / 24.0))
    return sum(scores) / max(len(scores), 1)


def _topo_score(scene: Dict[str, Any], candidate_payload: Dict[str, Any], candidate_by_id: Dict[str, Dict[str, Any]]) -> float:
    closure_penalty = float(scene.get("assembly", {}).get("closure_penalty", 1.0))
    relation_coverage = float(scene.get("assembly", {}).get("relation_coverage", 0.0))
    symbol_support = float(scene.get("assembly", {}).get("symbol_support", 0.0))

    observed_symbols = [
        c for c in candidate_payload.get("candidates", [])
        if str(c.get("type")) in {"text_label", "geometry_symbol"}
    ]
    active = set(scene.get("discrete_structure", {}).get("active_primitives", []))
    active_symbols = [c for c in observed_symbols if str(c.get("id")) in active]
    symbol_coverage = len(active_symbols) / max(len(observed_symbols), 1) if observed_symbols else 1.0

    raw = 0.45 * relation_coverage + 0.35 * symbol_support + 0.20 * symbol_coverage
    return max(0.0, min(1.0, raw * max(0.0, 1.0 - closure_penalty)))


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    weights_sum = args.w1 + args.w2 + args.w3
    if weights_sum <= 0:
        args.w1, args.w2, args.w3 = 1.0, 0.0, 0.0
    else:
        args.w1 /= weights_sum
        args.w2 /= weights_sum
        args.w3 /= weights_sum

    written = 0
    mean_visual = []

    for sample_id in split_ids:
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
            candidate_payload = read_stage_sample("candidates", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        candidate_by_id, candidate_by_name = _candidate_maps(candidate_payload)
        scene_scores = []
        for scene in scene_payload.get("scenes", []):
            s_line = _line_score(scene, candidate_by_id)
            s_point = _point_score(scene, candidate_by_name)
            s_topo = _topo_score(scene, candidate_payload, candidate_by_id)
            s_visual = args.w1 * s_line + args.w2 * s_point + args.w3 * s_topo
            scene_scores.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "s_line": round(s_line, 6),
                    "s_point": round(s_point, 6),
                    "s_topo": round(s_topo, 6),
                    "s_visual": round(s_visual, 6),
                }
            )

        avg_sv = sum(x["s_visual"] for x in scene_scores) / max(len(scene_scores), 1)
        mean_visual.append(avg_sv)

        score_file = args.outputs_dir / "scores" / slugify(args.variant) / args.split / f"{sample_id}.json"
        if score_file.exists():
            score_payload = read_json(score_file)
        else:
            score_payload = {"sample_id": sample_id, "variant": args.variant, "split": args.split}
        score_payload["visual_scoring"] = {
            "weights": {"w1": args.w1, "w2": args.w2, "w3": args.w3},
            "render_consistency": round(avg_sv, 6),
            "scenes": scene_scores,
            "visual_mode": "candidate_observation_consistency",
        }
        write_json(score_file, score_payload)
        written += 1

    elapsed = time.perf_counter() - begin
    print(
        f"[score_visual] split={args.split} variant={args.variant} "
        f"written={written} mean_render={sum(mean_visual)/max(len(mean_visual),1):.4f} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
