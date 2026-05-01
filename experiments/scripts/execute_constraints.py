"""
Phase 3: execute lightweight constraints and compute geometry scores.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 执行几何约束并计算几何一致性分数
INPUTS: 场景假设
OUTPUTS: 可执行场景、几何分数、残差
GT USAGE: None - 纯几何约束求解
PURPOSE: 核心推理模块 - 三重一致性评分的几何部分
LAST UPDATED: 2026-04-28

✅ 此脚本不访问测试集标注，可用于正式推理。
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    default_weights,
    exp_score,
    read_json,
    read_split_ids,
    read_stage_sample,
    restore_stdout_if_needed,
    safe_float,
    stage_dir,
    write_json,
    write_run_meta,
)


POINT_ON_LINE_RE = re.compile(
    r"PointLiesOnLine\(\s*([A-Za-z0-9_]+)\s*,\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)"
)
POINT_ON_CIRCLE_RE = re.compile(
    r"PointLiesOnCircle\(\s*([A-Za-z0-9_]+)\s*,\s*Circle\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_.+\-*/\\]+)\s*\)\s*\)"
)
PAR_RE = re.compile(
    r"Parallel\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*,\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)"
)
PERP_RE = re.compile(
    r"Perpendicular\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*,\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)"
)
EQUAL_LINE_RE = re.compile(
    r"Equals\(\s*LengthOf\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*,\s*LengthOf\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*\)"
)
EQUAL_LEN_VALUE_RE = re.compile(
    r"Equals\(\s*LengthOf\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*,\s*([A-Za-z0-9_.+\-*/=\\{}()]+)\s*\)"
)
ANGLE_VALUE_RE = re.compile(
    r"Equals\(\s*MeasureOf\(\s*Angle\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*,\s*([A-Za-z0-9_.+\-*/=\\{}()]+)\s*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute EGSR constraints.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--tau", type=float, default=default_weights()["tau"])
    parser.add_argument("--circle-proxy-mode", default="none", choices=["none", "cluster"])
    parser.add_argument("--literal-proxy-penalty", type=float, default=0.2)
    parser.add_argument("--unparsed-penalty", type=float, default=0.15)
    parser.add_argument("--missing-point-penalty", type=float, default=1.0)
    parser.add_argument("--missing-line-relation-penalty", type=float, default=1.0)
    parser.add_argument("--missing-length-value-penalty", type=float, default=1.0)
    parser.add_argument("--missing-angle-value-penalty", type=float, default=1.0)
    parser.add_argument("--length-literal-scale-mode", default="none", choices=["none", "median_ratio"])
    parser.add_argument("--length-literal-scale-min-pairs", type=int, default=2)
    parser.add_argument("--length-literal-scale-mix", type=float, default=1.0)
    parser.add_argument("--infeasible-residual-bump", type=float, default=0.35)
    parser.add_argument("--infeasible-geo-scale", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _pair_key(a: str, b: str, c: str, d: str) -> Tuple[str, str]:
    left = "".join(sorted([a, b]))
    right = "".join(sorted([c, d]))
    if left <= right:
        return left, right
    return right, left


def _point_xy(scene: Dict[str, Any], point_name: str) -> Optional[Tuple[float, float]]:
    coords = scene.get("continuous_parameters", {}).get("point_coordinates", {})
    if point_name not in coords:
        return None
    xy = coords[point_name]
    if not isinstance(xy, list) or len(xy) < 2:
        return None
    return float(xy[0]), float(xy[1])


def _line_points(scene: Dict[str, Any], left: str, right: str) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    p1 = _point_xy(scene, left)
    p2 = _point_xy(scene, right)
    if p1 is None or p2 is None:
        return None
    return p1, p2


def _vector(p1: Sequence[float], p2: Sequence[float]) -> Tuple[float, float]:
    return float(p2[0]) - float(p1[0]), float(p2[1]) - float(p1[1])


def _norm(vec: Sequence[float]) -> float:
    return math.sqrt(float(vec[0]) ** 2 + float(vec[1]) ** 2)


def _distance_point_to_line(point: Sequence[float], line: Tuple[Sequence[float], Sequence[float]]) -> float:
    p1, p2 = line
    vx, vy = _vector(p1, p2)
    px, py = float(point[0]) - float(p1[0]), float(point[1]) - float(p1[1])
    denom = max(_norm((vx, vy)), 1e-6)
    cross = abs(vx * py - vy * px)
    return cross / denom


def _parallel_residual(line_a: Tuple[Sequence[float], Sequence[float]], line_b: Tuple[Sequence[float], Sequence[float]]) -> float:
    va = _vector(*line_a)
    vb = _vector(*line_b)
    denom = max(_norm(va) * _norm(vb), 1e-6)
    cross = abs(va[0] * vb[1] - va[1] * vb[0])
    return min(1.0, cross / denom)


def _perpendicular_residual(line_a: Tuple[Sequence[float], Sequence[float]], line_b: Tuple[Sequence[float], Sequence[float]]) -> float:
    va = _vector(*line_a)
    vb = _vector(*line_b)
    denom = max(_norm(va) * _norm(vb), 1e-6)
    dot = abs(va[0] * vb[0] + va[1] * vb[1])
    return min(1.0, dot / denom)


def _line_length(scene: Dict[str, Any], left: str, right: str) -> Optional[float]:
    line = _line_points(scene, left, right)
    if line is None:
        return None
    return _norm(_vector(*line))


def _angle_value(scene: Dict[str, Any], a: str, b: str, c: str) -> Optional[float]:
    pa = _point_xy(scene, a)
    pb = _point_xy(scene, b)
    pc = _point_xy(scene, c)
    if pa is None or pb is None or pc is None:
        return None
    v1 = (pa[0] - pb[0], pa[1] - pb[1])
    v2 = (pc[0] - pb[0], pc[1] - pb[1])
    denom = max(_norm(v1) * _norm(v2), 1e-6)
    cos_val = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / denom))
    return math.degrees(math.acos(cos_val))


def _parse_numeric(token: str) -> Optional[float]:
    token = str(token).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", token):
        return float(token)
    return None


def _circle_center_xy(scene: Dict[str, Any], circle_name: str) -> Optional[Tuple[float, float]]:
    circle_params = scene.get("continuous_parameters", {}).get("circle_parameters", {})
    payload = circle_params.get(circle_name)
    if not isinstance(payload, dict):
        return None
    loc = payload.get("location", [])
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    center_name = payload.get("center")
    if center_name:
        return _point_xy(scene, str(center_name))
    return None


def _circle_radius_stats(scene: Dict[str, Any]) -> Dict[Tuple[str, str], Tuple[float, List[float]]]:
    stats: Dict[Tuple[str, str], List[float]] = {}
    for form in [str(x) for x in scene.get("projected_logic_forms", [])]:
        no_space = re.sub(r"\s+", "", form)
        m = POINT_ON_CIRCLE_RE.fullmatch(no_space)
        if not m:
            continue
        point_name, circle_name, radius_token = m.groups()
        if not radius_token:
            continue
        point_xy = _point_xy(scene, point_name)
        center_xy = _circle_center_xy(scene, circle_name)
        if point_xy is None or center_xy is None:
            continue
        radius = _norm((point_xy[0] - center_xy[0], point_xy[1] - center_xy[1]))
        stats.setdefault((circle_name, radius_token), []).append(radius)
    out: Dict[Tuple[str, str], Tuple[float, List[float]]] = {}
    for key, radii in stats.items():
        if len(radii) >= 3:
            out[key] = (sum(radii) / len(radii), radii)
    return out


def _median(values: Sequence[float]) -> Optional[float]:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _length_literal_scale(
    scene: Dict[str, Any],
    min_pairs: int,
) -> Optional[float]:
    ratios: List[float] = []
    forms = [str(x) for x in scene.get("projected_logic_forms", [])]
    for form in forms:
        no_space = re.sub(r"\s+", "", form)
        m = EQUAL_LEN_VALUE_RE.fullmatch(no_space)
        if not m:
            continue
        a, b, token = m.groups()
        target = _parse_numeric(token)
        if target is None or abs(target) < 1e-6:
            continue
        observed = _line_length(scene, a, b)
        if observed is None:
            continue
        ratios.append(observed / abs(target))
    if len(ratios) < max(1, min_pairs):
        return None
    return _median(ratios)


def _hard_filter(scene: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    reasons: List[str] = []
    violated: List[str] = []
    discrete = scene.get("discrete_structure", {})
    active = set(discrete.get("active_primitives", []))
    relations = discrete.get("relations", [])
    continuous = scene.get("continuous_parameters", {})

    point_coords = continuous.get("point_coordinates", {})
    line_params = continuous.get("line_parameters", {})

    for cid in active:
        if isinstance(cid, str) and cid.startswith("pt_"):
            if cid[3:] not in point_coords:
                reasons.append("missing_point_coordinate")
                violated.append(cid)
        if isinstance(cid, str) and cid.startswith("ln_"):
            if cid[3:] not in line_params:
                reasons.append("missing_line_parameter")
                violated.append(cid)

    for rel in relations:
        if len(rel) < 2:
            reasons.append("malformed_relation")
            continue
        left = rel[0]
        right = rel[1]
        left_ok = isinstance(left, str) and left in active
        if isinstance(right, list):
            right_ok = all(isinstance(x, str) and x in active for x in right)
        else:
            right_ok = isinstance(right, str) and right in active
        if not left_ok or not right_ok:
            reasons.append("unclosed_reference")
            violated.append(str(rel))

    forms = [re.sub(r"\s+", "", x) for x in scene.get("projected_logic_forms", [])]
    parallel_pairs = set()
    perp_pairs = set()
    for form in forms:
        pm = PAR_RE.fullmatch(form)
        if pm:
            parallel_pairs.add(_pair_key(*pm.groups()))
            continue
        xm = PERP_RE.fullmatch(form)
        if xm:
            perp_pairs.add(_pair_key(*xm.groups()))
    conflicts = parallel_pairs & perp_pairs
    if conflicts:
        reasons.append("incompatible_parallel_perpendicular")
        violated.extend([f"conflict:{x[0]}|{x[1]}" for x in sorted(conflicts)])

    if not forms:
        reasons.append("empty_constraint_system")

    feasible = len(reasons) == 0
    return feasible, sorted(set(reasons)), violated


def _constraint_terms(
    scene: Dict[str, Any],
    circle_proxy_mode: str = "none",
    literal_proxy_penalty: float = 0.2,
    unparsed_penalty: float = 0.15,
    missing_point_penalty: float = 1.0,
    missing_line_relation_penalty: float = 1.0,
    missing_length_value_penalty: float = 1.0,
    missing_angle_value_penalty: float = 1.0,
    length_literal_scale_mode: str = "none",
    length_literal_scale_min_pairs: int = 2,
    length_literal_scale_mix: float = 1.0,
) -> List[Tuple[str, float]]:
    terms: List[Tuple[str, float]] = []
    forms = [str(x) for x in scene.get("projected_logic_forms", [])]
    circle_radius_stats = _circle_radius_stats(scene) if circle_proxy_mode == "cluster" else {}
    length_literal_scale = None
    if length_literal_scale_mode == "median_ratio":
        length_literal_scale = _length_literal_scale(scene, length_literal_scale_min_pairs)
    length_literal_scale_mix = max(0.0, min(1.0, length_literal_scale_mix))

    point_coords = scene.get("continuous_parameters", {}).get("point_coordinates", {})
    if point_coords:
        xs = [float(v[0]) for v in point_coords.values() if isinstance(v, list) and len(v) >= 2]
        ys = [float(v[1]) for v in point_coords.values() if isinstance(v, list) and len(v) >= 2]
        diag = math.sqrt((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) if xs and ys else 1.0
    else:
        diag = 1.0
    diag = max(diag, 1.0)

    for form in forms:
        no_space = re.sub(r"\s+", "", form)

        m = POINT_ON_LINE_RE.fullmatch(no_space)
        if m:
            p, a, b = m.groups()
            point = _point_xy(scene, p)
            line = _line_points(scene, a, b)
            if point is None or line is None:
                terms.append(("point_on_line_missing", missing_point_penalty))
            else:
                terms.append(("point_on_line", min(1.0, _distance_point_to_line(point, line) / diag)))
            continue

        m = POINT_ON_CIRCLE_RE.fullmatch(no_space)
        if m:
            p, circle_name, radius_token = m.groups()
            point = _point_xy(scene, p)
            center_xy = _circle_center_xy(scene, circle_name)
            radius = _parse_numeric(radius_token)
            if point is None or center_xy is None:
                terms.append(("point_on_circle_missing", missing_point_penalty))
            elif radius is not None:
                observed = _norm((point[0] - center_xy[0], point[1] - center_xy[1]))
                terms.append(("point_on_circle", abs(observed - radius) / max(radius, 1.0)))
            else:
                proxy = circle_radius_stats.get((circle_name, radius_token))
                if proxy is None:
                    terms.append(("point_on_circle_missing", missing_point_penalty))
                else:
                    mean_radius, _ = proxy
                    observed = _norm((point[0] - center_xy[0], point[1] - center_xy[1]))
                    terms.append(("point_on_circle_proxy", abs(observed - mean_radius) / max(mean_radius, 1.0)))
            continue

        m = PAR_RE.fullmatch(no_space)
        if m:
            a, b, c, d = m.groups()
            line_a = _line_points(scene, a, b)
            line_b = _line_points(scene, c, d)
            if line_a is None or line_b is None:
                terms.append(("parallel_missing", missing_line_relation_penalty))
            else:
                terms.append(("parallel", _parallel_residual(line_a, line_b)))
            continue

        m = PERP_RE.fullmatch(no_space)
        if m:
            a, b, c, d = m.groups()
            line_a = _line_points(scene, a, b)
            line_b = _line_points(scene, c, d)
            if line_a is None or line_b is None:
                terms.append(("perpendicular_missing", missing_line_relation_penalty))
            else:
                terms.append(("perpendicular", _perpendicular_residual(line_a, line_b)))
            continue

        m = EQUAL_LINE_RE.fullmatch(no_space)
        if m:
            a, b, c, d = m.groups()
            len1 = _line_length(scene, a, b)
            len2 = _line_length(scene, c, d)
            if len1 is None or len2 is None:
                terms.append(("equal_length_missing", missing_line_relation_penalty))
            else:
                terms.append(("equal_length", abs(len1 - len2) / max(len1, len2, 1.0)))
            continue

        m = EQUAL_LEN_VALUE_RE.fullmatch(no_space)
        if m:
            a, b, token = m.groups()
            target = _parse_numeric(token)
            if target is None:
                terms.append(("length_literal_proxy", literal_proxy_penalty))
            else:
                observed = _line_length(scene, a, b)
                if observed is None:
                    terms.append(("length_literal_missing", missing_length_value_penalty))
                else:
                    raw_residual = abs(observed - target) / max(abs(target), observed, 1.0)
                    scaled_residual = raw_residual
                    if length_literal_scale is not None:
                        scaled_target = abs(target) * length_literal_scale
                        scaled_residual = abs(observed - scaled_target) / max(scaled_target, observed, 1.0)
                    residual = raw_residual
                    if length_literal_scale is not None:
                        residual = (
                            (1.0 - length_literal_scale_mix) * raw_residual
                            + length_literal_scale_mix * scaled_residual
                        )
                    terms.append(("length_literal", min(1.0, residual)))
            continue

        m = ANGLE_VALUE_RE.fullmatch(no_space)
        if m:
            a, b, c, token = m.groups()
            target = _parse_numeric(token)
            if target is None:
                terms.append(("angle_literal_proxy", literal_proxy_penalty))
            else:
                observed = _angle_value(scene, a, b, c)
                if observed is None:
                    terms.append(("angle_literal_missing", missing_angle_value_penalty))
                else:
                    terms.append(("angle_literal", abs(observed - target) / 180.0))
            continue

        terms.append(("unparsed_constraint", unparsed_penalty))

    return terms


def _geometry_scores(
    scene: Dict[str, Any],
    tau: float,
    circle_proxy_mode: str = "none",
    literal_proxy_penalty: float = 0.2,
    unparsed_penalty: float = 0.15,
    missing_point_penalty: float = 1.0,
    missing_line_relation_penalty: float = 1.0,
    missing_length_value_penalty: float = 1.0,
    missing_angle_value_penalty: float = 1.0,
    length_literal_scale_mode: str = "none",
    length_literal_scale_min_pairs: int = 2,
    length_literal_scale_mix: float = 1.0,
) -> Dict[str, Any]:
    terms = _constraint_terms(
        scene,
        circle_proxy_mode=circle_proxy_mode,
        literal_proxy_penalty=literal_proxy_penalty,
        unparsed_penalty=unparsed_penalty,
        missing_point_penalty=missing_point_penalty,
        missing_line_relation_penalty=missing_line_relation_penalty,
        missing_length_value_penalty=missing_length_value_penalty,
        missing_angle_value_penalty=missing_angle_value_penalty,
        length_literal_scale_mode=length_literal_scale_mode,
        length_literal_scale_min_pairs=length_literal_scale_min_pairs,
        length_literal_scale_mix=length_literal_scale_mix,
    )
    if not terms:
        residual = 1.0
    else:
        residual = sum(score for _, score in terms) / len(terms)
    s_geo = exp_score(residual, tau)
    return {
        "residual": residual,
        "s_geo": s_geo,
        "constraint_terms": [
            {"name": name, "residual": round(score, 6)}
            for name, score in terms
        ],
    }


def _load_score_file(path: Path, sample_id: str, variant: str, split: str) -> Dict[str, Any]:
    if path.exists():
        return read_json(path)
    return {
        "sample_id": sample_id,
        "variant": variant,
        "split": split,
    }


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    scores_dir = stage_dir("scores", args.variant, args.split, args.outputs_dir)
    _ = scores_dir

    written = 0
    executability_acc = []
    residual_acc = []

    for sample_id in split_ids:
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        scene_scores: List[Dict[str, Any]] = []
        for scene in scene_payload.get("scenes", []):
            feasible, reasons, violated = _hard_filter(scene)
            geom = _geometry_scores(
                scene,
                args.tau,
                circle_proxy_mode=args.circle_proxy_mode,
                literal_proxy_penalty=args.literal_proxy_penalty,
                unparsed_penalty=args.unparsed_penalty,
                missing_point_penalty=args.missing_point_penalty,
                missing_line_relation_penalty=args.missing_line_relation_penalty,
                missing_length_value_penalty=args.missing_length_value_penalty,
                missing_angle_value_penalty=args.missing_angle_value_penalty,
                length_literal_scale_mode=args.length_literal_scale_mode,
                length_literal_scale_min_pairs=args.length_literal_scale_min_pairs,
                length_literal_scale_mix=args.length_literal_scale_mix,
            )
            residual = geom["residual"]
            s_geo = geom["s_geo"]
            if not feasible:
                s_geo *= max(0.0, min(1.0, args.infeasible_geo_scale))
                residual = min(1.0, residual + args.infeasible_residual_bump)

            scene_scores.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "feasible": feasible,
                    "hard_filter_reasons": reasons,
                    "violated_constraints": violated,
                    "residual": round(residual, 6),
                    "s_geo": round(s_geo, 6),
                    "constraint_terms": geom["constraint_terms"],
                }
            )

        feasible_scores = [x for x in scene_scores if x["feasible"]]
        executability = len(feasible_scores) / max(len(scene_scores), 1)
        mean_residual = (
            sum(safe_float(x["residual"]) for x in feasible_scores) / len(feasible_scores)
            if feasible_scores
            else 1.0
        )
        executability_acc.append(executability)
        residual_acc.append(mean_residual)

        from egsr_common import slugify

        out_file = args.outputs_dir / "scores" / slugify(args.variant) / args.split / f"{sample_id}.json"
        payload = _load_score_file(out_file, sample_id, args.variant, args.split)
        payload["constraint_execution"] = {
            "tau": args.tau,
            "executability_rate": round(executability, 6),
            "mean_residual": round(mean_residual, 6),
            "scenes": scene_scores,
        }
        write_json(out_file, payload)
        written += 1
        restore_stdout_if_needed()

    elapsed = time.perf_counter() - begin
    write_run_meta(
        "scores",
        args.variant,
        args.split,
        {
            "module": "constraint_execution",
            "num_requested": len(split_ids),
            "num_written": written,
            "avg_executability_rate": round(sum(executability_acc) / max(len(executability_acc), 1), 6),
            "avg_mean_residual": round(sum(residual_acc) / max(len(residual_acc), 1), 6),
            "elapsed_sec": round(elapsed, 3),
            "residual_mode": "rule_based_geometry_proxy",
            "circle_proxy_mode": args.circle_proxy_mode,
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
        },
        args.outputs_dir,
    )
    print(
        f"[execute_constraints] split={args.split} variant={args.variant} "
        f"written={written} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
