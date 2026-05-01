"""
Phase 7 (internal): evaluate Coverage@K and Oracle@K.

ORACLE STATUS: ORACLE-DIAG
DESCRIPTION: 诊断工具 - 分析候选池的覆盖率和Oracle上界
INPUTS: 候选池、测试集标注（ground truth）
OUTPUTS: Coverage@K 和 Oracle@K 指标（用于 Table 2）
GT USAGE: Diagnostic only - 用于分析候选池质量上界
PURPOSE: 回答"如果选择器是完美的，候选池能达到多少性能？"
LAST UPDATED: 2026-04-28

⚠️ 警告：此脚本使用测试集标注，仅用于诊断分析，不作为正式推理结果。
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    load_ground_truth,
    load_official_eval_module,
    metric_tag_from_scores,
    official_single_sample_scores,
    read_split_ids,
    read_stage_sample,
    slugify,
    write_json,
)

TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
KEYWORDS = {
    "PointLiesOnLine",
    "PointLiesOnCircle",
    "Perpendicular",
    "Parallel",
    "Equals",
    "LengthOf",
    "MeasureOf",
    "Line",
    "Circle",
    "Angle",
    "Arc",
    "Triangle",
    "Quadrilateral",
    "Polygon",
    "Add",
    "Mul",
    "SumOf",
    "Find",
    "RadiusOf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EGSR coverage@K and oracle@K metrics.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--k-values", default="1,5,10,20")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--pgdp-root", type=Path, default=Path("experiments/data/PGDP"))
    return parser.parse_args()


def _scene_to_graph(scene: Dict[str, Any]) -> Dict[str, Any]:
    discrete = scene.get("discrete_structure", {})
    active = set(discrete.get("active_primitives", []))
    continuous = scene.get("continuous_parameters", {})
    points_all = continuous.get("point_coordinates", {})
    lines_all = continuous.get("line_parameters", {})
    circles_all = continuous.get("circle_parameters", {})
    inferred_points = {name: [float(xy[0]), float(xy[1])] for name, xy in points_all.items()}
    for line in lines_all.values():
        endpoints = line.get("endpoints", [])
        location = line.get("location", [])
        if len(endpoints) == 2 and len(location) >= 4:
            inferred_points.setdefault(endpoints[0], [float(location[0]), float(location[1])])
            inferred_points.setdefault(endpoints[1], [float(location[2]), float(location[3])])
    for circle in circles_all.values():
        center = circle.get("center")
        location = circle.get("location", [])
        if isinstance(center, str) and len(location) >= 2:
            inferred_points.setdefault(center, [float(location[0]), float(location[1])])

    point_instances = sorted([x[3:] for x in active if isinstance(x, str) and x.startswith("pt_")])
    line_instances = sorted(lines_all.keys())
    circle_instances = sorted(circles_all.keys())
    if not point_instances:
        point_instances = sorted(points_all.keys())
    if not line_instances:
        line_instances = sorted(lines_all.keys())
    if not circle_instances:
        circle_instances = sorted(circles_all.keys())

    known_line_names = set(line_instances) | set(lines_all.keys())
    known_circle_names = set(circle_instances) | set(circles_all.keys())

    point_positions = {}
    for p in point_instances:
        if p in inferred_points:
            xy = inferred_points[p]
            point_positions[p] = [float(xy[0]), float(xy[1])]

    logic_forms = [str(x) for x in scene.get("projected_logic_forms", [])]
    referenced_points = set()
    for form in logic_forms:
        for tok in TOKEN_RE.findall(form):
            if tok in KEYWORDS or tok.islower():
                continue
            # Point labels must take precedence over line/circle instance names
            # because the scene representation can reuse a point name as the
            # identifier for a circle centered at that point.
            if tok in inferred_points:
                referenced_points.add(tok)
                continue
            if tok in known_line_names or tok in known_circle_names:
                continue
            if tok[0].isupper():
                referenced_points.add(tok)
    for p in sorted(referenced_points):
        if p not in point_positions:
            if p in inferred_points:
                xy = inferred_points[p]
                point_positions[p] = [float(xy[0]), float(xy[1])]
            else:
                point_positions[p] = [0.0, 0.0]
        if p not in point_instances:
            point_instances.append(p)
    point_instances = sorted(set(point_instances))

    return {
        "point_instances": point_instances,
        "line_instances": line_instances,
        "circle_instances": circle_instances,
        "diagram_logic_forms": logic_forms,
        "point_positions": point_positions,
    }


def _parse_k_values(text: str) -> List[int]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return sorted(set([k for k in out if k > 0]))


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()
    k_values = _parse_k_values(args.k_values)
    if not k_values:
        raise ValueError("k-values must include at least one positive integer")

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    gt_map = load_ground_truth(args.data_dir)
    eval_handle = load_official_eval_module(args.pgdp_root)

    # Aggregators
    coverage_as = {k: 0 for k in k_values}
    coverage_ts = {k: 0 for k in k_values}
    oracle_tags = {k: {"LS": 0, "AS": 0, "PR": 0, "TS": 0} for k in k_values}
    used = 0
    per_sample = {}

    max_k = max(k_values)
    for sample_id in split_ids:
        gt_graph = gt_map.get(sample_id)
        if gt_graph is None:
            continue
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        scenes = scene_payload.get("scenes", [])[:max_k]
        if not scenes:
            continue

        scene_metrics = []
        for scene in scenes:
            pred_graph = _scene_to_graph(scene)
            scores = official_single_sample_scores(eval_handle, gt_graph, pred_graph)
            tags = metric_tag_from_scores(scores)
            scene_metrics.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "accuracy": scores["accuracy"],
                    "recall": scores["recall"],
                    "f1": scores["f1"],
                    **tags,
                }
            )

        sample_report = {}
        for k in k_values:
            topk = scene_metrics[:k]
            as_hit = int(any(row["AS"] == 1 for row in topk))
            ts_hit = int(any(row["TS"] == 1 for row in topk))
            coverage_as[k] += as_hit
            coverage_ts[k] += ts_hit

            best = max(topk, key=lambda x: x["f1"])
            for key in ["LS", "AS", "PR", "TS"]:
                oracle_tags[k][key] += int(best[key])

            sample_report[k] = {
                "coverage_as": as_hit,
                "coverage_ts": ts_hit,
                "oracle_best_scene_id": best["scene_id"],
                "oracle_f1": round(float(best["f1"]), 6),
                "oracle_tags": {m: int(best[m]) for m in ["LS", "AS", "PR", "TS"]},
            }

        per_sample[sample_id] = sample_report
        used += 1

    n = max(used, 1)
    summary_rows = []
    for k in k_values:
        summary_rows.append(
            {
                "K": k,
                "Coverage@K": round(100.0 * coverage_as[k] / n, 4),
                "TS-Coverage@K": round(100.0 * coverage_ts[k] / n, 4),
                "Oracle-LS@K": round(100.0 * oracle_tags[k]["LS"] / n, 4),
                "Oracle-AS@K": round(100.0 * oracle_tags[k]["AS"] / n, 4),
                "Oracle-PR@K": round(100.0 * oracle_tags[k]["PR"] / n, 4),
                "Oracle-TS@K": round(100.0 * oracle_tags[k]["TS"] / n, 4),
            }
        )

    elapsed = time.perf_counter() - begin
    payload = {
        "split": args.split,
        "variant": args.variant,
        "k_values": k_values,
        "num_samples": used,
        "summary": summary_rows,
        "elapsed_sec": round(elapsed, 3),
        "per_sample": per_sample,
    }
    out_file = args.outputs_dir / "eval" / slugify(args.variant) / args.split / "oracle_k_metrics.json"
    write_json(out_file, payload)
    print(
        f"[eval_oracle_k] split={args.split} variant={args.variant} "
        f"samples={used} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
