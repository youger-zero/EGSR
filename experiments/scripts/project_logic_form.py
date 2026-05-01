"""
Phase 6: deterministic projection from selected scene to logic form.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 将选定的场景投影到 PGDP 逻辑形式
INPUTS: 选定的场景假设
OUTPUTS: PGDP 格式的逻辑形式
GT USAGE: None - 纯确定性转换
PURPOSE: 核心推理模块 - 将内部表示转换为标准格式
LAST UPDATED: 2026-04-28

✅ 此脚本不访问测试集标注，可用于正式推理。
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import assemble_scenes as assemble_utils

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    normalize_logic_form,
    read_json,
    read_split_ids,
    read_stage_sample,
    slugify,
    write_json,
    write_run_meta,
    write_stage_sample,
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
POINT_ON_LINE_RE = re.compile(r"^PointLiesOnLine\(([^,]+),\s*Line\(([^,]+),\s*([^)]+)\)\)$")
POINT_ON_CIRCLE_RE = re.compile(r"^PointLiesOnCircle\(([^,]+),\s*Circle\(([^,]+),\s*([^)]+)\)\)$")
POINT_ON_LINE_NORM_MAX = 0.014
POINT_ON_CIRCLE_CLUSTER_DIAG_TOL = 0.008
POINT_ON_CIRCLE_CLUSTER_REL_TOL = 0.022
POINT_ON_CIRCLE_MIN_RADIUS_NORM = 0.02
CENTER_ON_LINE_ABS_MIN = 0.4
LEN_TEXT_SEGMENT_MARGIN_MIN = 0.22


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project EGSR selected scene to logic form.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument(
        "--candidate-variant",
        default="",
        help="Optional candidate-stage variant used to recover text-anchor geometry during projection.",
    )
    parser.add_argument("--rewrite-mode", default="none", choices=["none"], help="Public release supports only the clean no-rewrite path.")
    parser.add_argument(
        "--public-postprocess-mode",
        default="off",
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
    )
    parser.add_argument("--point-on-line-norm-max", type=float, default=POINT_ON_LINE_NORM_MAX)
    parser.add_argument("--point-on-circle-cluster-diag-tol", type=float, default=POINT_ON_CIRCLE_CLUSTER_DIAG_TOL)
    parser.add_argument("--point-on-circle-cluster-rel-tol", type=float, default=POINT_ON_CIRCLE_CLUSTER_REL_TOL)
    parser.add_argument("--point-on-circle-min-radius-norm", type=float, default=POINT_ON_CIRCLE_MIN_RADIUS_NORM)
    parser.add_argument("--center-on-line-abs-min", type=float, default=CENTER_ON_LINE_ABS_MIN)
    parser.add_argument("--len-text-segment-margin-min", type=float, default=LEN_TEXT_SEGMENT_MARGIN_MIN)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _pick_scene(scene_payload: Dict[str, Any], selected_scene_id: Optional[str]) -> Optional[Dict[str, Any]]:
    scenes = scene_payload.get("scenes", [])
    if not scenes:
        return None
    if selected_scene_id is None:
        return scenes[0]
    for scene in scenes:
        if scene.get("scene_id") == selected_scene_id:
            return scene
    return scenes[0]


def _distance_point_to_segment(
    point_xy: List[float],
    a_xy: List[float],
    b_xy: List[float],
) -> float:
    px, py = float(point_xy[0]), float(point_xy[1])
    ax, ay = float(a_xy[0]), float(a_xy[1])
    bx, by = float(b_xy[0]), float(b_xy[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def _diagram_diag(point_coords: Dict[str, List[float]]) -> float:
    coords = [xy for xy in point_coords.values() if isinstance(xy, list) and len(xy) >= 2]
    if not coords:
        return 1.0
    xs = [float(x[0]) for x in coords]
    ys = [float(x[1]) for x in coords]
    return max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)


def _passes_point_on_line_guard(form: str, point_coords: Dict[str, List[float]], point_on_line_norm_max: float) -> bool:
    match = POINT_ON_LINE_RE.match(form)
    if not match:
        return True
    point_name, left_name, right_name = match.groups()
    point_xy = point_coords.get(point_name)
    left_xy = point_coords.get(left_name)
    right_xy = point_coords.get(right_name)
    if point_xy is None or left_xy is None or right_xy is None:
        return True
    diag = _diagram_diag(point_coords)
    dist = _distance_point_to_segment(point_xy, left_xy, right_xy)
    return (dist / diag) <= point_on_line_norm_max


def _filter_center_on_line_forms(
    logic_forms: List[str],
    point_coords: Dict[str, List[float]],
    circle_params: Dict[str, Any],
    center_on_line_abs_min: float,
) -> List[str]:
    if not logic_forms or not point_coords or not circle_params:
        return logic_forms

    point_line_count: Dict[str, int] = {}
    line_rel_count: Dict[frozenset[str], int] = {}
    point_circle_count: Dict[str, int] = {}
    normalized_forms = [normalize_logic_form(x) for x in logic_forms]
    for form in normalized_forms:
        line_match = POINT_ON_LINE_RE.match(form)
        if line_match:
            point_name, left_name, right_name = line_match.groups()
            point_line_count[point_name] = point_line_count.get(point_name, 0) + 1
            pair_key = frozenset((left_name, right_name))
            line_rel_count[pair_key] = line_rel_count.get(pair_key, 0) + 1
            continue
        circle_match = POINT_ON_CIRCLE_RE.match(form)
        if circle_match:
            _, center_name, _ = circle_match.groups()
            point_circle_count[center_name] = point_circle_count.get(center_name, 0) + 1

    filtered: List[str] = []
    for form in logic_forms:
        match = POINT_ON_LINE_RE.match(normalize_logic_form(form))
        if not match:
            filtered.append(form)
            continue
        point_name, left_name, right_name = match.groups()
        if point_name not in circle_params:
            filtered.append(form)
            continue
        if point_line_count.get(point_name, 0) != 4:
            filtered.append(form)
            continue
        if point_circle_count.get(point_name, 0) > 2:
            filtered.append(form)
            continue
        pair_key = frozenset((left_name, right_name))
        if line_rel_count.get(pair_key, 0) > 3:
            filtered.append(form)
            continue

        point_xy = point_coords.get(point_name)
        left_xy = point_coords.get(left_name)
        right_xy = point_coords.get(right_name)
        if point_xy is None or left_xy is None or right_xy is None:
            filtered.append(form)
            continue
        dist = _distance_point_to_segment(point_xy, left_xy, right_xy)
        if dist < center_on_line_abs_min:
            filtered.append(form)
            continue
        filtered.append(None)  # type: ignore[arg-type]
    return [form for form in filtered if form is not None]


def _augment_point_on_circle_forms(
    logic_forms: List[str],
    point_coords: Dict[str, List[float]],
    circle_params: Dict[str, Any],
    active_points: List[str],
    point_on_circle_cluster_diag_tol: float,
    point_on_circle_cluster_rel_tol: float,
    point_on_circle_min_radius_norm: float,
) -> List[str]:
    if not logic_forms or not point_coords or not circle_params:
        return logic_forms

    diag = _diagram_diag(point_coords)
    seen = {normalize_logic_form(x) for x in logic_forms}
    radius_name_by_center: Dict[str, str] = {}
    count_by_center: Dict[str, int] = {}
    for form in logic_forms:
        match = POINT_ON_CIRCLE_RE.match(normalize_logic_form(form))
        if not match:
            continue
        _, center_name, radius_name = match.groups()
        radius_name_by_center.setdefault(center_name, radius_name)
        count_by_center[center_name] = count_by_center.get(center_name, 0) + 1

    candidate_points = sorted(set(active_points) | set(point_coords.keys()))
    augmented = list(logic_forms)
    for center_name in sorted(circle_params.keys()):
        if count_by_center.get(center_name, 0) != 0:
            continue
        center_xy = point_coords.get(center_name)
        if center_xy is None:
            continue

        point_dists: List[tuple[str, float]] = []
        for point_name in candidate_points:
            if point_name == center_name:
                continue
            point_xy = point_coords.get(point_name)
            if point_xy is None:
                continue
            radius = math.hypot(float(point_xy[0]) - float(center_xy[0]), float(point_xy[1]) - float(center_xy[1]))
            if radius <= diag * point_on_circle_min_radius_norm:
                continue
            point_dists.append((point_name, radius))
        if len(point_dists) < 2:
            continue

        point_dists.sort(key=lambda item: item[1])
        best_cluster: List[tuple[str, float]] = []
        for start_idx in range(len(point_dists)):
            cluster = [point_dists[start_idx]]
            for next_idx in range(start_idx + 1, len(point_dists)):
                cluster_radii = [item[1] for item in cluster]
                cluster_radii.append(point_dists[next_idx][1])
                span = max(cluster_radii) - min(cluster_radii)
                mean_radius = sum(cluster_radii) / len(cluster_radii)
                tol = max(diag * point_on_circle_cluster_diag_tol, mean_radius * point_on_circle_cluster_rel_tol)
                if span <= tol:
                    cluster.append(point_dists[next_idx])
                else:
                    break
            if len(cluster) < 2:
                continue
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
                continue
            if len(cluster) == len(best_cluster) and best_cluster:
                cluster_mean = sum(item[1] for item in cluster) / len(cluster)
                best_mean = sum(item[1] for item in best_cluster) / len(best_cluster)
                if cluster_mean < best_mean:
                    best_cluster = cluster
        if len(best_cluster) < 2:
            continue

        radius_name = radius_name_by_center.get(center_name, "radius_0_0")
        for point_name, _ in best_cluster:
            form = f"PointLiesOnCircle({point_name}, Circle({center_name}, {radius_name}))"
            norm = normalize_logic_form(form)
            if norm in seen:
                continue
            augmented.append(form)
            seen.add(norm)
    return augmented


def _point_symbol_support(logic_forms: List[str]) -> Dict[str, float]:
    support: Dict[str, float] = {}
    for form in logic_forms:
        if form.startswith("PointLiesOnCircle("):
            continue
        for token in TOKEN_RE.findall(form):
            if token in KEYWORDS or token.islower() or not token[0].isupper():
                continue
            support[token] = support.get(token, 0.0) + 1.0
    return support


def _best_circle_cluster(
    center_name: str,
    point_coords: Dict[str, List[float]],
    candidate_points: List[str],
    point_on_circle_cluster_diag_tol: float,
    point_on_circle_cluster_rel_tol: float,
    point_on_circle_min_radius_norm: float,
) -> List[tuple[str, float]]:
    center_xy = point_coords.get(center_name)
    if center_xy is None:
        return []
    diag = _diagram_diag(point_coords)
    point_dists: List[tuple[str, float]] = []
    for point_name in candidate_points:
        if point_name == center_name:
            continue
        point_xy = point_coords.get(point_name)
        if point_xy is None:
            continue
        radius = math.hypot(float(point_xy[0]) - float(center_xy[0]), float(point_xy[1]) - float(center_xy[1]))
        if radius <= diag * point_on_circle_min_radius_norm:
            continue
        point_dists.append((point_name, radius))
    if len(point_dists) < 2:
        return []

    point_dists.sort(key=lambda item: item[1])
    best_cluster: List[tuple[str, float]] = []
    for start_idx in range(len(point_dists)):
        cluster = [point_dists[start_idx]]
        for next_idx in range(start_idx + 1, len(point_dists)):
            cluster_radii = [item[1] for item in cluster]
            cluster_radii.append(point_dists[next_idx][1])
            span = max(cluster_radii) - min(cluster_radii)
            mean_radius = sum(cluster_radii) / len(cluster_radii)
            tol = max(diag * point_on_circle_cluster_diag_tol, mean_radius * point_on_circle_cluster_rel_tol)
            if span <= tol:
                cluster.append(point_dists[next_idx])
            else:
                break
        if len(cluster) < 2:
            continue
        if len(cluster) > len(best_cluster):
            best_cluster = cluster
            continue
        if len(cluster) == len(best_cluster) and best_cluster:
            cluster_mean = sum(item[1] for item in cluster) / len(cluster)
            best_mean = sum(item[1] for item in best_cluster) / len(best_cluster)
            if cluster_mean < best_mean:
                best_cluster = cluster
    return best_cluster


def _replace_point_on_circle_forms(
    logic_forms: List[str],
    point_coords: Dict[str, List[float]],
    circle_params: Dict[str, Any],
    active_points: List[str],
    point_on_circle_cluster_diag_tol: float,
    point_on_circle_cluster_rel_tol: float,
    point_on_circle_min_radius_norm: float,
) -> List[str]:
    if not logic_forms or not point_coords or not circle_params:
        return logic_forms

    candidate_points = sorted(set(active_points) | set(point_coords.keys()))
    support = _point_symbol_support(logic_forms)
    forms = list(logic_forms)
    radius_name_by_center: Dict[str, str] = {}
    current_points_by_center: Dict[str, List[str]] = {}
    for form in forms:
        match = POINT_ON_CIRCLE_RE.match(normalize_logic_form(form))
        if not match:
            continue
        point_name, center_name, radius_name = match.groups()
        radius_name_by_center.setdefault(center_name, radius_name)
        current_points_by_center.setdefault(center_name, []).append(point_name)

    def _mean_radius(center_name: str, point_names: List[str]) -> float:
        center_xy = point_coords[center_name]
        radii = [
            math.hypot(
                float(point_coords[point_name][0]) - float(center_xy[0]),
                float(point_coords[point_name][1]) - float(center_xy[1]),
            )
            for point_name in point_names
            if point_name in point_coords
        ]
        if not radii:
            return float("inf")
        return sum(radii) / len(radii)

    def _rewrite_center(center_name: str, point_names: List[str]) -> None:
        nonlocal forms
        forms = [
            form
            for form in forms
            if not (
                (match := POINT_ON_CIRCLE_RE.match(normalize_logic_form(form))) is not None
                and match.group(2) == center_name
            )
        ]
        radius_name = radius_name_by_center.get(center_name, "radius_0_0")
        for point_name in point_names:
            forms.append(f"PointLiesOnCircle({point_name}, Circle({center_name}, {radius_name}))")

    for center_name in sorted(circle_params.keys()):
        current_points = current_points_by_center.get(center_name, [])
        inferred_points = [
            point_name
            for point_name, _ in _best_circle_cluster(
                center_name,
                point_coords,
                candidate_points,
                point_on_circle_cluster_diag_tol,
                point_on_circle_cluster_rel_tol,
                point_on_circle_min_radius_norm,
            )
        ]
        if (
            len(current_points) >= 2
            and len(inferred_points) >= 2
            and not set(current_points).issubset(inferred_points)
            and _mean_radius(center_name, inferred_points) <= 0.8 * _mean_radius(center_name, current_points)
        ):
            current_support = sum(support.get(point_name, 0.0) for point_name in current_points) / len(current_points)
            inferred_support = sum(support.get(point_name, 0.0) for point_name in inferred_points) / len(inferred_points)
            if current_support == 0.0 and inferred_support == 0.0:
                continue
            if inferred_support >= current_support:
                _rewrite_center(center_name, inferred_points)
    return forms


def _fallback_ambiguous_length_forms(
    logic_forms: List[str],
    scene: Dict[str, Any],
    candidate_payload: Optional[Dict[str, Any]],
    len_text_segment_margin_min: float,
) -> List[str]:
    if not candidate_payload:
        return logic_forms

    discrete = scene.get("discrete_structure", {})
    active = set(discrete.get("active_primitives", []))
    symbol_rows = discrete.get("symbol_ownership", [])
    candidates = candidate_payload.get("candidates", [])
    candidates_by_id = {
        str(row.get("id")): row
        for row in candidates
        if row.get("id") is not None
    }
    point_candidates = [
        candidates_by_id[cid]
        for cid in active
        if isinstance(cid, str) and cid.startswith("pt_") and cid in candidates_by_id
    ]
    if not point_candidates:
        return logic_forms

    updated = list(logic_forms)
    for row in symbol_rows:
        symbol = candidates_by_id.get(str(row.get("symbol", "")))
        if not symbol:
            continue
        geom = symbol.get("geometric_parameters", {})
        if str(geom.get("text_class", "")) != "len":
            continue
        line_ids = [
            cid
            for cid in row.get("targets", [])
            if isinstance(cid, str) and cid.startswith("ln_") and cid in active and cid in candidates_by_id
        ]
        if not line_ids:
            continue
        anchor = assemble_utils._symbol_center(symbol)
        if anchor is None:
            continue
        line = candidates_by_id.get(line_ids[0], {})
        supports = assemble_utils._line_support_points(line, point_candidates)
        if len(supports) < 3:
            continue
        loc = line.get("location", [])
        if not isinstance(loc, list) or len(loc) < 4:
            continue
        t_anchor = assemble_utils._segment_projection_ratio(anchor, loc[:2], loc[2:4])
        scores = []
        for idx in range(len(supports) - 1):
            left_t, left_name = supports[idx]
            right_t, right_name = supports[idx + 1]
            if left_name == right_name:
                continue
            mid_t = 0.5 * (left_t + right_t)
            score = abs(t_anchor - mid_t) + 0.25 * abs(right_t - left_t)
            scores.append((score, left_name, right_name))
        scores.sort(key=lambda item: item[0])
        if len(scores) < 2:
            continue
        if (scores[1][0] - scores[0][0]) >= len_text_segment_margin_min:
            continue

        text_content = geom.get("text_content")
        if text_content in {None, ""}:
            text_content = geom.get("text")
        scalar_text = assemble_utils._normalize_scalar_text(text_content)
        coarse_line = assemble_utils._line_expr(line)
        if scalar_text is None or coarse_line is None:
            continue
        refined_form = f"Equals(LengthOf(Line({scores[0][1]}, {scores[0][2]})), {scalar_text})"
        coarse_form = f"Equals(LengthOf({coarse_line}), {scalar_text})"
        if refined_form in updated and coarse_form not in updated:
            updated = [coarse_form if form == refined_form else form for form in updated]
    return updated


def _project(
    scene: Dict[str, Any],
    candidate_payload: Optional[Dict[str, Any]] = None,
    point_on_line_norm_max: float = POINT_ON_LINE_NORM_MAX,
    point_on_circle_cluster_diag_tol: float = POINT_ON_CIRCLE_CLUSTER_DIAG_TOL,
    point_on_circle_cluster_rel_tol: float = POINT_ON_CIRCLE_CLUSTER_REL_TOL,
    point_on_circle_min_radius_norm: float = POINT_ON_CIRCLE_MIN_RADIUS_NORM,
    center_on_line_abs_min: float = CENTER_ON_LINE_ABS_MIN,
    len_text_segment_margin_min: float = LEN_TEXT_SEGMENT_MARGIN_MIN,
) -> Dict[str, Any]:
    discrete = scene.get("discrete_structure", {})
    active = set(discrete.get("active_primitives", []))
    continuous = scene.get("continuous_parameters", {})

    point_coords = continuous.get("point_coordinates", {})
    line_params = continuous.get("line_parameters", {})
    circle_params = continuous.get("circle_parameters", {})

    active_points = sorted([x[3:] for x in active if isinstance(x, str) and x.startswith("pt_")])
    active_lines = sorted(line_params.keys())
    active_circles = sorted(circle_params.keys())

    if not active_points:
        active_points = sorted(point_coords.keys())
    if not active_lines:
        active_lines = sorted(line_params.keys())
    if not active_circles:
        active_circles = sorted(circle_params.keys())

    known_line_names = set(active_lines) | set(line_params.keys())
    known_circle_names = set(active_circles) | set(circle_params.keys())

    projected_positions = {}
    for p in active_points:
        if p in point_coords:
            xy = point_coords[p]
            projected_positions[p] = [float(xy[0]), float(xy[1])]

    logic_forms = [
        str(x)
        for x in scene.get("projected_logic_forms", [])
        if _passes_point_on_line_guard(str(x), point_coords, point_on_line_norm_max)
    ]
    logic_forms = _fallback_ambiguous_length_forms(
        logic_forms,
        scene,
        candidate_payload,
        len_text_segment_margin_min,
    )
    logic_forms = _filter_center_on_line_forms(
        logic_forms,
        point_coords,
        circle_params,
        center_on_line_abs_min,
    )
    logic_forms = _augment_point_on_circle_forms(
        logic_forms,
        point_coords,
        circle_params,
        active_points,
        point_on_circle_cluster_diag_tol,
        point_on_circle_cluster_rel_tol,
        point_on_circle_min_radius_norm,
    )
    logic_forms = _replace_point_on_circle_forms(
        logic_forms,
        point_coords,
        circle_params,
        active_points,
        point_on_circle_cluster_diag_tol,
        point_on_circle_cluster_rel_tol,
        point_on_circle_min_radius_norm,
    )

    # Keep point closure for the official evaluator.
    referenced_points = set()
    for form in logic_forms:
        for tok in TOKEN_RE.findall(form):
            if tok in KEYWORDS:
                continue
            if tok.islower():
                continue
            if tok in known_line_names or tok in known_circle_names:
                if not (len(tok) == 1 and tok[0].isupper()):
                    continue
            if tok[0].isupper():
                referenced_points.add(tok)

    for p in sorted(referenced_points):
        if p not in projected_positions:
            if p in point_coords:
                xy = point_coords[p]
                projected_positions[p] = [float(xy[0]), float(xy[1])]
            else:
                projected_positions[p] = [0.0, 0.0]
        if p not in active_points:
            active_points.append(p)
    active_points = sorted(set(active_points))

    return {
        "point_instances": active_points,
        "line_instances": active_lines,
        "circle_instances": active_circles,
        "diagram_logic_forms": logic_forms,
        "point_positions": projected_positions,
    }


def _canonical_line_expr(a: str, b: str) -> str:
    left, right = sorted((a, b))
    return f"Line({left}, {right})"


def _canonical_arc_expr(a: str, b: str) -> str:
    left, right = sorted((a, b))
    return f"Arc({left}, {right})"


def _canonical_angle_expr(a: str, b: str, c: str) -> str:
    left = (a, b, c)
    right = (c, b, a)
    pick = min(left, right)
    return f"Angle({pick[0]}, {pick[1]}, {pick[2]})"


def _public_postprocess_forms(forms: List[str], mode: str) -> List[str]:
    if mode == "off":
        return list(forms)

    name_tok = r"[A-Za-z0-9_']+"
    point_on_line_re = re.compile(rf"^PointLiesOnLine\(\s*({name_tok})\s*,\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)$")
    point_on_circle_re = re.compile(rf"^PointLiesOnCircle\(\s*({name_tok})\s*,\s*Circle\(\s*({name_tok})(?:\s*,\s*([A-Za-z0-9_]+))?\s*\)\s*\)$")
    parallel_re = re.compile(rf"^Parallel\(\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*,\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)$")
    perpendicular_re = re.compile(rf"^Perpendicular\(\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*,\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)$")
    len_value_re = re.compile(rf"^Equals\(\s*LengthOf\(\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*(.+)\)$")
    angle_value_re = re.compile(rf"^Equals\(\s*MeasureOf\(\s*Angle\(\s*({name_tok})\s*,\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*(.+)\)$")
    len_len_re = re.compile(rf"^Equals\(\s*LengthOf\(\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*LengthOf\(\s*Line\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*\)$")
    angle_angle_re = re.compile(rf"^Equals\(\s*MeasureOf\(\s*Angle\(\s*({name_tok})\s*,\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*MeasureOf\(\s*Angle\(\s*({name_tok})\s*,\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*\)$")
    arc_measure_value_re = re.compile(rf"^Equals\(\s*MeasureOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*(.+)\)$")
    arc_measure_arc_re = re.compile(rf"^Equals\(\s*MeasureOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*MeasureOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*\)$")
    arc_length_value_re = re.compile(rf"^Equals\(\s*LengthOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*(.+)\)$")
    arc_length_arc_re = re.compile(rf"^Equals\(\s*LengthOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*,\s*LengthOf\(\s*Arc\(\s*({name_tok})\s*,\s*({name_tok})\s*\)\s*\)\s*\)$")

    def canonicalize(form: str) -> str:
        text = re.sub(r"\s+", "", str(form))
        match = point_on_line_re.fullmatch(text)
        if match:
            point_name, a, b = match.groups()
            return f"PointLiesOnLine({point_name}, {_canonical_line_expr(a, b)})"
        match = point_on_circle_re.fullmatch(text)
        if match:
            point_name, center_name, radius_name = match.groups()
            if radius_name:
                return f"PointLiesOnCircle({point_name}, Circle({center_name}, {radius_name}))"
            return f"PointLiesOnCircle({point_name}, Circle({center_name}))"
        match = parallel_re.fullmatch(text)
        if match:
            left = _canonical_line_expr(match.group(1), match.group(2))
            right = _canonical_line_expr(match.group(3), match.group(4))
            first, second = sorted((left, right))
            return f"Parallel({first}, {second})"
        match = perpendicular_re.fullmatch(text)
        if match:
            left = _canonical_line_expr(match.group(1), match.group(2))
            right = _canonical_line_expr(match.group(3), match.group(4))
            first, second = sorted((left, right))
            return f"Perpendicular({first}, {second})"
        match = len_len_re.fullmatch(text)
        if match:
            left = f"LengthOf({_canonical_line_expr(match.group(1), match.group(2))})"
            right = f"LengthOf({_canonical_line_expr(match.group(3), match.group(4))})"
            first, second = sorted((left, right))
            return f"Equals({first}, {second})"
        match = len_value_re.fullmatch(text)
        if match:
            a, b, rhs = match.groups()
            return f"Equals(LengthOf({_canonical_line_expr(a, b)}), {rhs})"
        match = angle_angle_re.fullmatch(text)
        if match:
            left = f"MeasureOf({_canonical_angle_expr(match.group(1), match.group(2), match.group(3))})"
            right = f"MeasureOf({_canonical_angle_expr(match.group(4), match.group(5), match.group(6))})"
            first, second = sorted((left, right))
            return f"Equals({first}, {second})"
        match = angle_value_re.fullmatch(text)
        if match:
            a, b, c, rhs = match.groups()
            return f"Equals(MeasureOf({_canonical_angle_expr(a, b, c)}), {rhs})"
        match = arc_measure_arc_re.fullmatch(text)
        if match:
            left = f"MeasureOf({_canonical_arc_expr(match.group(1), match.group(2))})"
            right = f"MeasureOf({_canonical_arc_expr(match.group(3), match.group(4))})"
            first, second = sorted((left, right))
            return f"Equals({first}, {second})"
        match = arc_measure_value_re.fullmatch(text)
        if match:
            a, b, rhs = match.groups()
            return f"Equals(MeasureOf({_canonical_arc_expr(a, b)}), {rhs})"
        match = arc_length_arc_re.fullmatch(text)
        if match:
            left = f"LengthOf({_canonical_arc_expr(match.group(1), match.group(2))})"
            right = f"LengthOf({_canonical_arc_expr(match.group(3), match.group(4))})"
            first, second = sorted((left, right))
            return f"Equals({first}, {second})"
        match = arc_length_value_re.fullmatch(text)
        if match:
            a, b, rhs = match.groups()
            return f"Equals(LengthOf({_canonical_arc_expr(a, b)}), {rhs})"
        return text

    canonical_forms: List[str] = []
    seen = set()
    for form in forms:
        norm = canonicalize(form)
        if norm in seen:
            continue
        seen.add(norm)
        canonical_forms.append(norm)

    if mode == "canonical_only":
        return canonical_forms

    add_parallel_closure = mode in {"canonical_add_parallel_closure", "canonical_add_all_closure"}
    add_equal_closure = mode in {"canonical_add_equal_closure", "canonical_add_all_closure"}

    if add_parallel_closure or add_equal_closure:
        expanded = list(canonical_forms)
        seen_expanded = set(canonical_forms)

        if add_parallel_closure:
            neighbors: Dict[str, set[str]] = {}
            for form in canonical_forms:
                match = parallel_re.fullmatch(form)
                if not match:
                    continue
                left = _canonical_line_expr(match.group(1), match.group(2))
                right = _canonical_line_expr(match.group(3), match.group(4))
                neighbors.setdefault(left, set()).add(right)
                neighbors.setdefault(right, set()).add(left)
            for anchor in sorted(neighbors.keys()):
                linked = sorted(neighbors.get(anchor, set()))
                for i in range(len(linked)):
                    for j in range(i + 1, len(linked)):
                        first, second = sorted((linked[i], linked[j]))
                        form = f"Parallel({first}, {second})"
                        if form in seen_expanded:
                            continue
                        seen_expanded.add(form)
                        expanded.append(form)

        if add_equal_closure:
            equal_graphs: Dict[str, set[str]] = {}
            for form in expanded:
                match = len_len_re.fullmatch(form)
                if match:
                    left = f"LengthOf({_canonical_line_expr(match.group(1), match.group(2))})"
                    right = f"LengthOf({_canonical_line_expr(match.group(3), match.group(4))})"
                    equal_graphs.setdefault(left, set()).add(right)
                    equal_graphs.setdefault(right, set()).add(left)
                    continue
                match = angle_angle_re.fullmatch(form)
                if match:
                    left = f"MeasureOf({_canonical_angle_expr(match.group(1), match.group(2), match.group(3))})"
                    right = f"MeasureOf({_canonical_angle_expr(match.group(4), match.group(5), match.group(6))})"
                    equal_graphs.setdefault(left, set()).add(right)
                    equal_graphs.setdefault(right, set()).add(left)
                    continue
                match = arc_measure_arc_re.fullmatch(form)
                if match:
                    left = f"MeasureOf({_canonical_arc_expr(match.group(1), match.group(2))})"
                    right = f"MeasureOf({_canonical_arc_expr(match.group(3), match.group(4))})"
                    equal_graphs.setdefault(left, set()).add(right)
                    equal_graphs.setdefault(right, set()).add(left)
                    continue
                match = arc_length_arc_re.fullmatch(form)
                if match:
                    left = f"LengthOf({_canonical_arc_expr(match.group(1), match.group(2))})"
                    right = f"LengthOf({_canonical_arc_expr(match.group(3), match.group(4))})"
                    equal_graphs.setdefault(left, set()).add(right)
                    equal_graphs.setdefault(right, set()).add(left)
                    continue
            for anchor in sorted(equal_graphs.keys()):
                linked = sorted(equal_graphs.get(anchor, set()))
                for i in range(len(linked)):
                    for j in range(i + 1, len(linked)):
                        first, second = sorted((linked[i], linked[j]))
                        form = f"Equals({first}, {second})"
                        if form in seen_expanded:
                            continue
                        seen_expanded.add(form)
                        expanded.append(form)

        canonical_forms = expanded

    drop_relation_conflicts = mode in {"canonical_drop_relation_conflicts", "canonical_drop_all_conflicts"}
    drop_scalar_conflicts = mode in {"canonical_drop_scalar_conflicts", "canonical_drop_all_conflicts"}
    parallel_pairs = set()
    perpendicular_pairs = set()
    scalar_values: Dict[str, set[str]] = {}
    for form in canonical_forms:
        match = parallel_re.fullmatch(form)
        if match:
            pair = tuple(sorted((_canonical_line_expr(match.group(1), match.group(2)), _canonical_line_expr(match.group(3), match.group(4)))))
            parallel_pairs.add(pair)
            continue
        match = perpendicular_re.fullmatch(form)
        if match:
            pair = tuple(sorted((_canonical_line_expr(match.group(1), match.group(2)), _canonical_line_expr(match.group(3), match.group(4)))))
            perpendicular_pairs.add(pair)
            continue
        match = len_value_re.fullmatch(form)
        if match:
            scalar_values.setdefault(f"LengthOf({_canonical_line_expr(match.group(1), match.group(2))})", set()).add(match.group(3))
            continue
        match = angle_value_re.fullmatch(form)
        if match:
            scalar_values.setdefault(f"MeasureOf({_canonical_angle_expr(match.group(1), match.group(2), match.group(3))})", set()).add(match.group(4))
            continue
        match = arc_measure_value_re.fullmatch(form)
        if match:
            scalar_values.setdefault(f"MeasureOf({_canonical_arc_expr(match.group(1), match.group(2))})", set()).add(match.group(3))
            continue
        match = arc_length_value_re.fullmatch(form)
        if match:
            scalar_values.setdefault(f"LengthOf({_canonical_arc_expr(match.group(1), match.group(2))})", set()).add(match.group(3))
            continue

    conflict_pairs = parallel_pairs & perpendicular_pairs
    conflict_scalar_targets = {target for target, values in scalar_values.items() if len(values) >= 2}
    filtered: List[str] = []
    for form in canonical_forms:
        if drop_relation_conflicts:
            match = parallel_re.fullmatch(form)
            if match:
                pair = tuple(sorted((_canonical_line_expr(match.group(1), match.group(2)), _canonical_line_expr(match.group(3), match.group(4)))))
                if pair in conflict_pairs:
                    continue
            match = perpendicular_re.fullmatch(form)
            if match:
                pair = tuple(sorted((_canonical_line_expr(match.group(1), match.group(2)), _canonical_line_expr(match.group(3), match.group(4)))))
                if pair in conflict_pairs:
                    continue
        if drop_scalar_conflicts:
            match = len_value_re.fullmatch(form)
            if match and f"LengthOf({_canonical_line_expr(match.group(1), match.group(2))})" in conflict_scalar_targets:
                continue
            match = angle_value_re.fullmatch(form)
            if match and f"MeasureOf({_canonical_angle_expr(match.group(1), match.group(2), match.group(3))})" in conflict_scalar_targets:
                continue
            match = arc_measure_value_re.fullmatch(form)
            if match and f"MeasureOf({_canonical_arc_expr(match.group(1), match.group(2))})" in conflict_scalar_targets:
                continue
            match = arc_length_value_re.fullmatch(form)
            if match and f"LengthOf({_canonical_arc_expr(match.group(1), match.group(2))})" in conflict_scalar_targets:
                continue
        filtered.append(form)
    return filtered


def _rewrite_logic_forms(
    sample_id: str,
    logic_form: Dict[str, Any],
    rewrite_mode: str = "none",
    public_postprocess_mode: str = "off",
) -> Dict[str, Any]:
    if rewrite_mode != "none":
        raise ValueError("Public release supports only rewrite_mode=none")
    if public_postprocess_mode == "off":
        return logic_form

    forms = list(logic_form.get("diagram_logic_forms", []))
    if not forms:
        return logic_form

    forms = _public_postprocess_forms(forms, public_postprocess_mode)
    deduped: List[str] = []
    seen = set()
    for form in forms:
        if form in seen:
            continue
        seen.add(form)
        deduped.append(form)
    out = dict(logic_form)
    out["diagram_logic_forms"] = deduped
    return out


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    merged: Dict[str, Dict[str, Any]] = {}
    written = 0

    for sample_id in split_ids:
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        score_file = args.outputs_dir / "scores" / slugify(args.variant) / args.split / f"{sample_id}.json"
        selected_scene_id = None
        if score_file.exists():
            score_payload = read_json(score_file)
            selected_scene_id = score_payload.get("reranking", {}).get("selected_scene_id")

        selected_scene = _pick_scene(scene_payload, selected_scene_id)
        if selected_scene is None:
            continue
        candidate_payload = None
        candidate_variant = args.candidate_variant or args.variant
        try:
            candidate_payload = read_stage_sample("candidates", candidate_variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            candidate_payload = None
        logic_form = _project(
            selected_scene,
            candidate_payload,
            point_on_line_norm_max=args.point_on_line_norm_max,
            point_on_circle_cluster_diag_tol=args.point_on_circle_cluster_diag_tol,
            point_on_circle_cluster_rel_tol=args.point_on_circle_cluster_rel_tol,
            point_on_circle_min_radius_norm=args.point_on_circle_min_radius_norm,
            center_on_line_abs_min=args.center_on_line_abs_min,
            len_text_segment_margin_min=args.len_text_segment_margin_min,
        )
        logic_form = _rewrite_logic_forms(
            sample_id,
            logic_form,
            args.rewrite_mode,
            args.public_postprocess_mode,
        )
        write_stage_sample("logic_forms", args.variant, args.split, sample_id, logic_form, args.outputs_dir)
        merged[sample_id] = logic_form
        written += 1

    merged_path = (
        args.outputs_dir
        / "logic_forms"
        / slugify(args.variant)
        / args.split
        / "_predictions_merged.json"
    )
    write_json(merged_path, merged)

    elapsed = time.perf_counter() - begin
    write_run_meta(
        "logic_forms",
        args.variant,
        args.split,
        {
            "num_requested": len(split_ids),
            "num_written": written,
            "merged_path": str(merged_path),
            "elapsed_sec": round(elapsed, 3),
            "rewrite_mode": args.rewrite_mode,
            "public_postprocess_mode": args.public_postprocess_mode,
            "candidate_variant": args.candidate_variant or args.variant,
        },
        args.outputs_dir,
    )
    print(
        f"[project_logic_form] split={args.split} variant={args.variant} "
        f"written={written} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
