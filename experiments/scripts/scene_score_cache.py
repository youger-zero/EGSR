"""
Build and read cached official scene scores for EGSR experiments.

ORACLE STATUS: ORACLE-DIAG
DESCRIPTION: 缓存官方评分以加速实验迭代
INPUTS: 场景假设、测试集标注（ground truth）
OUTPUTS: 缓存的场景评分
GT USAGE: Diagnostic only - 使用 load_ground_truth 计算场景质量
PURPOSE: 加速超参数调优和消融实验
LAST UPDATED: 2026-04-28

⚠️ 警告：此脚本使用测试集标注进行评分，仅用于加速实验，不影响推理逻辑。
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    load_ground_truth,
    load_official_eval_module,
    metric_tag_from_scores,
    official_single_sample_scores,
    read_json,
    read_split_ids,
    read_stage_sample,
    slugify,
    write_json,
)

TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
KEYWORDS = {
    "PointLiesOnLine",
    "PointLiesOnCircle",
    "PointLabel",
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


def cache_dir(outputs_dir: Path, variant: str, split: str) -> Path:
    return outputs_dir / "scene_score_cache" / slugify(variant) / split


def cache_path(outputs_dir: Path, variant: str, split: str, sample_id: str) -> Path:
    return cache_dir(outputs_dir, variant, split) / f"{sample_id}.json"


def scene_to_graph(scene: Dict[str, Any]) -> Dict[str, Any]:
    discrete = scene.get("discrete_structure", {})
    active = set(discrete.get("active_primitives", []))
    continuous = scene.get("continuous_parameters", {})

    point_coords = continuous.get("point_coordinates", {})
    line_params = continuous.get("line_parameters", {})
    circle_params = continuous.get("circle_parameters", {})
    inferred_points = {name: [float(xy[0]), float(xy[1])] for name, xy in point_coords.items()}
    for line in line_params.values():
        endpoints = line.get("endpoints", [])
        location = line.get("location", [])
        if len(endpoints) == 2 and len(location) >= 4:
            inferred_points.setdefault(endpoints[0], [float(location[0]), float(location[1])])
            inferred_points.setdefault(endpoints[1], [float(location[2]), float(location[3])])
    for circle in circle_params.values():
        center = circle.get("center")
        location = circle.get("location", [])
        if isinstance(center, str) and len(location) >= 2:
            inferred_points.setdefault(center, [float(location[0]), float(location[1])])

    point_instances = sorted([x[3:] for x in active if isinstance(x, str) and x.startswith("pt_")])
    line_instances = sorted([x[3:] for x in active if isinstance(x, str) and x.startswith("ln_")])
    circle_instances = sorted([x[3:] for x in active if isinstance(x, str) and x.startswith("cc_")])

    if not point_instances:
        point_instances = sorted(point_coords.keys())
    if not line_instances:
        line_instances = sorted(line_params.keys())
    if not circle_instances:
        circle_instances = sorted(circle_params.keys())

    point_positions = {}
    for name in point_instances:
        if name in inferred_points:
            xy = inferred_points[name]
            point_positions[name] = [float(xy[0]), float(xy[1])]

    logic_forms = [str(x) for x in scene.get("projected_logic_forms", [])]
    referenced_points = set()
    for form in logic_forms:
        for token in TOKEN_RE.findall(form):
            if token in KEYWORDS or token.islower():
                continue
            if token in inferred_points:
                referenced_points.add(token)
                continue
            if token[0].isupper():
                referenced_points.add(token)
    for name in sorted(referenced_points):
        if name not in point_positions:
            if name in inferred_points:
                xy = inferred_points[name]
                point_positions[name] = [float(xy[0]), float(xy[1])]
            else:
                point_positions[name] = [0.0, 0.0]
        if name not in point_instances:
            point_instances.append(name)
    point_instances = sorted(set(point_instances))

    return {
        "point_instances": point_instances,
        "line_instances": line_instances,
        "circle_instances": circle_instances,
        "diagram_logic_forms": logic_forms,
        "point_positions": point_positions,
    }


def read_cached_sample(outputs_dir: Path, variant: str, split: str, sample_id: str) -> Optional[Dict[str, Any]]:
    path = cache_path(outputs_dir, variant, split, sample_id)
    if not path.exists():
        return None
    return read_json(path)


def _score_one_sample(
    sample_id: str,
    split: str,
    variant: str,
    data_dir: str,
    outputs_dir: str,
    pgdp_root: str,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    try:
        gt_map = load_ground_truth(Path(data_dir))
        gt_graph = gt_map.get(sample_id)
        if gt_graph is None:
            return sample_id, None, "missing_ground_truth"
        scene_payload = read_stage_sample("scenes", variant, split, sample_id, Path(outputs_dir))
        scenes = scene_payload.get("scenes", [])
        if not scenes:
            return sample_id, None, "missing_scenes"
        eval_handle = load_official_eval_module(Path(pgdp_root))
        scene_rows = []
        for scene in scenes:
            scene_id = scene.get("scene_id")
            scores = official_single_sample_scores(eval_handle, gt_graph, scene_to_graph(scene))
            tags = metric_tag_from_scores(scores)
            scene_rows.append(
                {
                    "scene_id": scene_id,
                    "accuracy": round(scores["accuracy"], 6),
                    "recall": round(scores["recall"], 6),
                    "f1": round(scores["f1"], 6),
                    **tags,
                }
            )
        payload = {
            "sample_id": sample_id,
            "split": split,
            "variant": variant,
            "num_scenes": len(scene_rows),
            "scene_metrics": scene_rows,
        }
        return sample_id, payload, None
    except Exception as exc:  # pragma: no cover - operational path
        return sample_id, None, str(exc)


def build_cache(
    split: str,
    variant: str,
    data_dir: Path = DATA_DIR,
    outputs_dir: Path = OUTPUTS_DIR,
    pgdp_root: Path = Path("experiments/data/PGDP"),
    limit: int = 0,
    workers: int = 1,
    force: bool = False,
) -> Dict[str, Any]:
    begin = time.perf_counter()
    split_ids = read_split_ids(split, data_dir)
    if limit > 0:
        split_ids = split_ids[:limit]

    out_dir = cache_dir(outputs_dir, variant, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    reused = 0
    for sample_id in split_ids:
        target = cache_path(outputs_dir, variant, split, sample_id)
        if target.exists() and not force:
            reused += 1
            continue
        pending.append(sample_id)

    done = 0
    errors: List[Dict[str, str]] = []
    total = len(pending)
    started = time.perf_counter()

    if workers <= 1:
        for sample_id in pending:
            sid, payload, err = _score_one_sample(
                sample_id,
                split,
                variant,
                str(data_dir),
                str(outputs_dir),
                str(pgdp_root),
            )
            if payload is not None:
                write_json(cache_path(outputs_dir, variant, split, sid), payload)
                done += 1
            else:
                errors.append({"sample_id": sid, "error": err or "unknown_error"})
            if total:
                elapsed = time.perf_counter() - started
                avg = elapsed / max(done + len(errors), 1)
                eta = avg * max(total - done - len(errors), 0)
                print(
                    f"[scene_score_cache] {done + len(errors)}/{total} "
                    f"({100.0 * (done + len(errors)) / total:.1f}%) "
                    f"ETA={eta / 60.0:.1f}m"
                )
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _score_one_sample,
                    sample_id,
                    split,
                    variant,
                    str(data_dir),
                    str(outputs_dir),
                    str(pgdp_root),
                )
                for sample_id in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                sid, payload, err = future.result()
                if payload is not None:
                    write_json(cache_path(outputs_dir, variant, split, sid), payload)
                    done += 1
                else:
                    errors.append({"sample_id": sid, "error": err or "unknown_error"})
                processed = done + len(errors)
                if total:
                    elapsed = time.perf_counter() - started
                    avg = elapsed / max(processed, 1)
                    eta = avg * max(total - processed, 0)
                    print(
                        f"[scene_score_cache] {processed}/{total} "
                        f"({100.0 * processed / total:.1f}%) "
                        f"ETA={eta / 60.0:.1f}m"
                    )

    elapsed = time.perf_counter() - begin
    payload = {
        "split": split,
        "variant": variant,
        "requested": len(split_ids),
        "reused": reused,
        "built": done,
        "failed": len(errors),
        "workers": workers,
        "elapsed_sec": round(elapsed, 3),
        "errors": errors[:50],
    }
    write_json(out_dir / "_cache_meta.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached official scene scores.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--pgdp-root", type=Path, default=Path("experiments/data/PGDP"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_cache(
        split=args.split,
        variant=args.variant,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        pgdp_root=args.pgdp_root,
        limit=args.limit,
        workers=args.workers,
        force=args.force,
    )
    print(
        f"[scene_score_cache] split={args.split} variant={args.variant} "
        f"built={payload['built']} reused={payload['reused']} failed={payload['failed']} "
        f"elapsed={payload['elapsed_sec']:.2f}s"
    )


if __name__ == "__main__":
    main()
