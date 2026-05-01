"""
Phase 2: assemble top-K EGSR scene graph hypotheses.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 浠庡€欓€夋睜缁勮鍦烘櫙鍋囪
INPUTS: 鍊欓€夋睜 JSON
OUTPUTS: Top-K 鍦烘櫙鍋囪
GT USAGE: None - 浠呬娇鐢ㄥ€欓€夋睜涓殑淇℃伅
PURPOSE: 鏍稿績鎺ㄧ悊妯″潡 - 缁勫悎鍊欓€夊厓绱犲舰鎴愬畬鏁村満鏅?LAST UPDATED: 2026-04-28

鉁?姝よ剼鏈笉璁块棶娴嬭瘯闆嗘爣娉紝鍙敤浜庢寮忔帹鐞嗐€?娉ㄦ剰锛氬€欓€夋睜涓殑 annotation_id 瀛楁浠呯敤浜庤拷韪€欓€夋潵婧愶紝涓嶅奖鍝嶆帹鐞嗛€昏緫銆?"""

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
    deterministic_rng,
    read_split_ids,
    safe_float,
    stage_dir,
    write_run_meta,
    write_stage_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble top-K EGSR scene hypotheses.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--angle-global-margin",
        type=float,
        default=8.0,
        help="Global angle target must beat local incident binding by at least this margin.",
    )
    parser.add_argument(
        "--perpendicular-global-margin",
        type=float,
        default=1e9,
        help="Global perpendicular target must beat local incident binding by at least this margin.",
    )
    parser.add_argument(
        "--grouped-line-duplicate-penalty",
        type=float,
        default=0.0,
        help="Penalty for assigning repeated grouped line-mark symbols to the same host line.",
    )
    parser.add_argument(
        "--parallel-companion-weight",
        type=float,
        default=0.0,
        help="Weight for preferring parallel-like marks whose host line has a plausible parallel companion.",
    )
    parser.add_argument(
        "--len-text-mark-support-bonus",
        type=float,
        default=0.0,
        help="Bonus for attaching length text to lines already supported by nearby bar-like marks.",
    )
    parser.add_argument(
        "--line-host-alt-gap-max",
        type=float,
        default=0.0,
        help="If positive, expand alternate host-line scene variants for ambiguous line-attached symbols whose next-best gap is within this threshold.",
    )
    parser.add_argument(
        "--line-host-alt-topm",
        type=int,
        default=1,
        help="Maximum number of ambiguous line-attached symbols to branch per assembly preset.",
    )
    parser.add_argument(
        "--pair-host-alt-gap-max",
        type=float,
        default=0.0,
        help="If positive, expand alternate line-pair scene variants for ambiguous angle/perpendicular bindings within this score gap.",
    )
    parser.add_argument(
        "--pair-host-alt-topm",
        type=int,
        default=1,
        help="Maximum number of ambiguous pair-bound symbols to branch per assembly preset.",
    )
    parser.add_argument(
        "--alt-penalty-scale",
        type=float,
        default=0.01,
        help="Assembly score penalty multiplier applied to later alternative-branch variants.",
    )
    parser.add_argument(
        "--force-degree-arc-policy",
        choices=["default", "selective_arc", "conservative_arc", "strict_selective_arc", "clustered_selective_arc"],
        default=None,
        help="Override preset degree/arc relation generation policy with a fixed public-safe mode.",
    )
    parser.add_argument(
        "--force-global-angle-mode",
        choices=["none", "all", "selective", "none_lenbackup"],
        default=None,
        help="Override preset global angle search mode with a fixed public-safe mode.",
    )
    parser.add_argument(
        "--force-line-completion-mode",
        choices=["strict", "default", "relaxed"],
        default=None,
        help="Override preset non-endpoint point-on-line completion tolerance.",
    )
    parser.add_argument(
        "--force-strict-circle-completion",
        choices=["auto", "on", "off"],
        default="auto",
        help="Override preset point-on-circle completion strictness.",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _dedupe_keep_order(items: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for item in items:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _candidate_maps(candidates: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    ann_to_cid: Dict[str, str] = {}
    for cand in candidates:
        cid = str(cand.get("id", ""))
        if not cid:
            continue
        by_id[cid] = cand
        ann_id = cand.get("annotation_id")
        if ann_id is not None:
            ann_to_cid[str(ann_id)] = cid
    return by_id, ann_to_cid


def _geometry_candidate_ids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(c["id"])
        for c in candidates
        if str(c.get("type")) in {"point", "line", "circle"}
    ]


def _symbol_candidate_ids(candidates: Sequence[Dict[str, Any]]) -> List[str]:
    return [
        str(c["id"])
        for c in candidates
        if str(c.get("type")) in {"text_label", "geometry_symbol"}
    ]


def _line_expr(candidate: Dict[str, Any]) -> Optional[str]:
    params = candidate.get("geometric_parameters", {})
    endpoints = params.get("endpoints", [])
    if isinstance(endpoints, list) and len(endpoints) >= 2 and endpoints[0] and endpoints[1]:
        return f"Line({endpoints[0]}, {endpoints[1]})"
    name = str(candidate.get("name", ""))
    if len(name) >= 2 and name.isalpha():
        return f"Line({name[0]}, {name[1]})"
    return None


def _circle_expr(candidate: Dict[str, Any]) -> Optional[str]:
    geom = candidate.get("geometric_parameters", {})
    radius_token = geom.get("radius_token")
    name = str(candidate.get("name", ""))
    if name and radius_token:
        return f"Circle({name}, {radius_token})"
    center = geom.get("center")
    if center:
        return f"Circle({center})"
    name = str(candidate.get("name", ""))
    if name:
        return f"Circle({name})"
    return None


def _point_name(candidate: Dict[str, Any]) -> Optional[str]:
    name = candidate.get("name")
    if name:
        return str(name)
    cid = str(candidate.get("id", ""))
    if cid.startswith("pt_"):
        return cid[3:]
    return None


def _line_endpoints(candidate: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    params = candidate.get("geometric_parameters", {})
    endpoints = params.get("endpoints", [])
    if isinstance(endpoints, list) and len(endpoints) >= 2 and endpoints[0] and endpoints[1]:
        return str(endpoints[0]), str(endpoints[1])
    return None


def _angle_from_lines(line_a: Dict[str, Any], line_b: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    end_a = _line_endpoints(line_a)
    end_b = _line_endpoints(line_b)
    if not end_a or not end_b:
        return None
    shared = set(end_a) & set(end_b)
    if not shared:
        return None
    vertex = sorted(shared)[0]
    left = end_a[0] if end_a[1] == vertex else end_a[1]
    right = end_b[0] if end_b[1] == vertex else end_b[1]
    return left, vertex, right


def _best_ray_point_name(
    line: Dict[str, Any],
    vertex: str,
    point_candidates: Sequence[Dict[str, Any]],
) -> Optional[str]:
    endpoints = _line_endpoints(line)
    if not endpoints:
        return None
    other = endpoints[0] if endpoints[1] == vertex else endpoints[1]
    if other == vertex:
        return None

    point_by_name = {
        str(c.get("name")): c
        for c in point_candidates
        if c.get("name")
    }
    vertex_cand = point_by_name.get(vertex)
    other_cand = point_by_name.get(other)
    if vertex_cand is None or other_cand is None:
        return other
    vertex_xy = _point_xy_from_candidate(vertex_cand)
    other_xy = _point_xy_from_candidate(other_cand)
    if vertex_xy is None or other_xy is None:
        return other

    line_len = max(math.hypot(other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1]), 1.0)
    max_dist = max(4.0, line_len * 0.03)
    ray = (other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1])

    best_name = other
    best_proj = float("inf")
    for cand in point_candidates:
        point_name = _point_name(cand)
        if point_name in {None, vertex}:
            continue
        xy = _point_xy_from_candidate(cand)
        if xy is None:
            continue
        if _line_distance(line, xy) > max_dist:
            continue
        vec = (xy[0] - vertex_xy[0], xy[1] - vertex_xy[1])
        proj = vec[0] * ray[0] + vec[1] * ray[1]
        if proj <= 1e-6:
            continue
        if proj < best_proj:
            best_proj = proj
            best_name = point_name
    return best_name


def _named_angle_from_lines(
    line_a: Dict[str, Any],
    line_b: Dict[str, Any],
    point_candidates: Sequence[Dict[str, Any]],
) -> Optional[Tuple[str, str, str]]:
    end_a = _line_endpoints(line_a)
    end_b = _line_endpoints(line_b)
    if not end_a or not end_b:
        return None
    shared = set(end_a) & set(end_b)
    if not shared:
        return None
    vertex = sorted(shared)[0]
    left = _best_ray_point_name(line_a, vertex, point_candidates)
    right = _best_ray_point_name(line_b, vertex, point_candidates)
    if not left or not right:
        return None
    return left, vertex, right


def _line_support_points(
    line: Dict[str, Any],
    point_candidates: Sequence[Dict[str, Any]],
) -> List[Tuple[float, str]]:
    endpoints = _line_endpoints(line)
    loc = line.get("location", [])
    if not endpoints or not isinstance(loc, list) or len(loc) < 4:
        return []
    a = (float(loc[0]), float(loc[1]))
    b = (float(loc[2]), float(loc[3]))
    line_len = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1.0)
    max_dist = max(4.0, line_len * 0.03)
    rows: List[Tuple[float, str]] = []
    seen = set()
    for cand in point_candidates:
        point_name = _point_name(cand)
        xy = _point_xy_from_candidate(cand)
        if point_name is None or xy is None or point_name in seen:
            continue
        if _line_distance(line, xy) > max_dist:
            continue
        t = _segment_projection_ratio(xy, a, b)
        rows.append((t, point_name))
        seen.add(point_name)
    rows.sort(key=lambda x: x[0])
    return rows


def _segment_expr_near_anchor(
    line: Dict[str, Any],
    point_candidates: Sequence[Dict[str, Any]],
    anchor: Sequence[float],
) -> Optional[str]:
    supports = _line_support_points(line, point_candidates)
    if len(supports) < 2:
        return _line_expr(line)
    loc = line.get("location", [])
    if not isinstance(loc, list) or len(loc) < 4:
        return _line_expr(line)
    t_anchor = _segment_projection_ratio(anchor, loc[:2], loc[2:4])
    best_pair = None
    best_score = 1e18
    for i in range(len(supports) - 1):
        left_t, left_name = supports[i]
        right_t, right_name = supports[i + 1]
        if left_name == right_name:
            continue
        mid_t = 0.5 * (left_t + right_t)
        score = abs(t_anchor - mid_t) + 0.25 * abs(right_t - left_t)
        if score < best_score:
            best_score = score
            best_pair = (left_name, right_name)
    if best_pair is None:
        return _line_expr(line)
    return f"Line({best_pair[0]}, {best_pair[1]})"


def _ray_line_expr(
    line: Dict[str, Any],
    vertex: str,
    point_candidates: Sequence[Dict[str, Any]],
) -> Optional[str]:
    point_name = _best_ray_point_name(line, vertex, point_candidates)
    if point_name is None:
        return _line_expr(line)
    return f"Line({point_name}, {vertex})"


def _sanitize_value(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    text = re.sub(r"\s+", "", text)
    return text or "unknown"


def _canonical_symbol_class(value: Any) -> str:
    text = str(value or "").strip().lower()
    targeted = {
        "double bar": "double_bar",
        "triple bar": "triple_bar",
        "double parallel": "double_parallel",
        "double angle": "double_angle",
        "triple angle": "triple_angle",
        "quad angle": "quad_angle",
    }
    return targeted.get(text, text)


GROUPED_LINE_MARK_CLASSES = {"double_bar", "triple_bar", "double_parallel"}
GROUPED_ANGLE_MARK_CLASSES = {"double_angle", "triple_angle", "quad_angle"}
PARALLEL_LIKE_MARK_CLASSES = {"parallel", "double_parallel"}
LENGTH_LIKE_MARK_CLASSES = {"bar", "double_bar", "triple_bar"}


def _is_supported_scalar_text(value: Any) -> bool:
    return _normalize_scalar_text(value) is not None


def _normalize_scalar_text(value: Any) -> Optional[str]:
    text = _sanitize_value(value)
    if text == "unknown":
        return None
    # OCR can emit payloads like "FH=37"; keep only the scalar right-hand side.
    if "=" in text:
        parts = [p for p in text.split("=") if p]
        if parts:
            text = parts[-1]
    # Reject tuple-like / coordinate-like payloads such as "(h,k)".
    if "," in text:
        return None
    return text or None


def _normalize_group_text(value: Any) -> Optional[str]:
    text = _sanitize_value(value)
    if text == "unknown":
        return None
    return text or None


def _point_xy_from_candidate(candidate: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    loc = candidate.get("location", [])
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def _symbol_center(candidate: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    geom = candidate.get("geometric_parameters", {})
    center = geom.get("center")
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    loc = candidate.get("location", [])
    if isinstance(loc, list):
        if len(loc) >= 4:
            x, y, w, h = map(float, loc[:4])
            # Text/symbol detections are often stored as [x, y, w, h].
            if w >= 0.0 and h >= 0.0:
                return x + w / 2.0, y + h / 2.0
            return (float(loc[0]) + float(loc[2])) / 2.0, (float(loc[1]) + float(loc[3])) / 2.0
        if len(loc) >= 2:
            return float(loc[0]), float(loc[1])
    return None


def _distance_point_to_segment(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def _point_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _segment_projection_ratio(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    return max(0.0, min(1.0, t))


def _segment_projection_raw(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(point[0]), float(point[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-9:
        return 0.5
    return ((px - ax) * dx + (py - ay) * dy) / denom


def _line_distance(candidate: Dict[str, Any], point: Sequence[float]) -> float:
    loc = candidate.get("location", [])
    if not isinstance(loc, list) or len(loc) < 4:
        return 1e9
    return _distance_point_to_segment(point, loc[:2], loc[2:4])


def _line_incident_to_point(line: Dict[str, Any], point_name: str) -> bool:
    params = line.get("geometric_parameters", {})
    endpoints = params.get("endpoints", [])
    return isinstance(endpoints, list) and point_name in endpoints


def _nearest_lines(
    center: Sequence[float],
    line_candidates: Sequence[Dict[str, Any]],
    topn: int = 2,
) -> List[Dict[str, Any]]:
    ranked = sorted(line_candidates, key=lambda c: _line_distance(c, center))
    return ranked[:topn]


def _shared_vertex(line_a: Dict[str, Any], line_b: Dict[str, Any]) -> Optional[str]:
    end_a = _line_endpoints(line_a)
    end_b = _line_endpoints(line_b)
    if not end_a or not end_b:
        return None
    shared = sorted(set(end_a) & set(end_b))
    return shared[0] if shared else None


def _line_other_endpoint(line: Dict[str, Any], vertex: str) -> Optional[str]:
    endpoints = _line_endpoints(line)
    if not endpoints:
        return None
    if endpoints[0] == vertex:
        return endpoints[1]
    if endpoints[1] == vertex:
        return endpoints[0]
    return None


def _angle_between(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    ax, ay = float(vec_a[0]), float(vec_a[1])
    bx, by = float(vec_b[0]), float(vec_b[1])
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na <= 1e-9 or nb <= 1e-9:
        return math.pi
    cos_val = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.acos(cos_val)


def _line_direction(line: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    loc = line.get("location", [])
    if not isinstance(loc, list) or len(loc) < 4:
        return None
    return float(loc[2]) - float(loc[0]), float(loc[3]) - float(loc[1])


def _symbol_axis(symbol: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    loc = symbol.get("location", [])
    if not isinstance(loc, list) or len(loc) < 4:
        return None
    if len(loc) >= 4:
        w = float(loc[2])
        h = float(loc[3])
        if w <= 0.0 and h <= 0.0:
            return None
        return (1.0, 0.0) if w >= h else (0.0, 1.0)
    return None


def _symbol_box_aspect(symbol: Dict[str, Any]) -> float:
    loc = symbol.get("location", [])
    if not isinstance(loc, list) or len(loc) < 4:
        return 1.0
    w = abs(float(loc[2]))
    h = abs(float(loc[3]))
    if h <= 1e-6:
        return 1.0
    return max(w / h, h / w)


def _rank_host_lines_for_mark(
    symbol: Dict[str, Any],
    center: Sequence[float],
    lines: Sequence[Dict[str, Any]],
    prefer_parallel: bool,
    topn: int = 6,
    midpoint_weight: float = 16.0,
    overflow_weight: float = 30.0,
    parallel_companion_weight: float = 0.0,
) -> List[Tuple[Dict[str, Any], float]]:
    nearby = _nearest_lines(center, lines, topn=topn)
    if not nearby:
        return []
    axis = _symbol_axis(symbol)
    if axis is None:
        return [(line, _line_distance(line, center) + 1e-6 * idx) for idx, line in enumerate(nearby[:1])]
    ranked: List[Tuple[Dict[str, Any], float]] = []
    for idx, line in enumerate(nearby):
        direction = _line_direction(line)
        if direction is None:
            continue
        orient = min(_angle_between(axis, direction), abs(math.pi - _angle_between(axis, direction)))
        orient_pen = orient if prefer_parallel else abs((math.pi / 2.0) - orient)
        dist_pen = _line_distance(line, center)
        loc = line.get("location", [])
        t_pen = 0.0
        overflow_pen = 0.0
        if isinstance(loc, list) and len(loc) >= 4:
            t = _segment_projection_ratio(center, loc[:2], loc[2:4])
            raw_t = _segment_projection_raw(center, loc[:2], loc[2:4])
            t_pen = abs(t - 0.5)
            overflow_pen = max(0.0, 0.08 - raw_t, raw_t - 0.92)
        midpoint_w = midpoint_weight if not prefer_parallel else 6.0
        score = dist_pen + 10.0 * orient_pen + midpoint_w * t_pen + overflow_weight * overflow_pen + 1e-6 * idx
        if prefer_parallel and parallel_companion_weight > 0.0:
            companion_pen = 5.0
            for other in nearby:
                if other is line:
                    continue
                other_dir = _line_direction(other)
                if other_dir is None:
                    continue
                align = min(_angle_between(direction, other_dir), abs(math.pi - _angle_between(direction, other_dir)))
                shared_pen = 1.5 if _shared_vertex(line, other) is not None else 0.0
                other_dist = _line_distance(other, center)
                pair_pen = 40.0 * align + shared_pen + 0.25 * other_dist
                if pair_pen < companion_pen:
                    companion_pen = pair_pen
            score += parallel_companion_weight * companion_pen
        ranked.append((line, score))
    ranked.sort(key=lambda item: item[1])
    return ranked


def _best_host_line_for_mark(
    symbol: Dict[str, Any],
    center: Sequence[float],
    lines: Sequence[Dict[str, Any]],
    prefer_parallel: bool,
    topn: int = 6,
    midpoint_weight: float = 16.0,
    overflow_weight: float = 30.0,
    parallel_companion_weight: float = 0.0,
) -> List[Dict[str, Any]]:
    ranked = _rank_host_lines_for_mark(
        symbol,
        center,
        lines,
        prefer_parallel=prefer_parallel,
        topn=topn,
        midpoint_weight=midpoint_weight,
        overflow_weight=overflow_weight,
        parallel_companion_weight=parallel_companion_weight,
    )
    return [ranked[0][0]] if ranked else []


def _line_id(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("id", ""))


def _point_candidate_id_by_name(point_by_name: Dict[str, Dict[str, Any]], point_name: str) -> Optional[str]:
    point = point_by_name.get(point_name)
    if point is None:
        return None
    cid = str(point.get("id", ""))
    return cid or None


def _length_mark_support_bonus(
    center: Sequence[float],
    line_id: str,
    mark_rankings: Sequence[Dict[str, Any]],
) -> float:
    bonus = 0.0
    for row in mark_rankings:
        mark_center = row.get("center")
        if not isinstance(mark_center, tuple) or len(mark_center) < 2:
            continue
        dist = math.hypot(float(center[0]) - float(mark_center[0]), float(center[1]) - float(mark_center[1]))
        if dist > 140.0:
            continue
        proximity = max(0.0, 1.0 - dist / 140.0)
        for rank, candidate_line_id in enumerate(row.get("line_ids", [])[:3]):
            if candidate_line_id == line_id:
                bonus += proximity / (rank + 1.0)
                break
    return bonus


def _diversify_grouped_line_symbol_hosts(
    symbol_ownership: Sequence[Dict[str, Any]],
    duplicate_penalty: float,
) -> List[Dict[str, Any]]:
    if duplicate_penalty <= 0.0:
        return [dict(row) for row in symbol_ownership]

    updated = [dict(row) for row in symbol_ownership]
    groups: Dict[str, List[int]] = {}
    for idx, row in enumerate(updated):
        sym_class = str(row.get("sym_class", ""))
        if sym_class in GROUPED_LINE_MARK_CLASSES or sym_class in {"bar", "parallel"}:
            candidates = row.get("host_line_candidates", [])
            if isinstance(candidates, list) and candidates:
                groups.setdefault(sym_class, []).append(idx)

    for _, indices in groups.items():
        if len(indices) < 2:
            continue
        candidate_lists: List[List[Tuple[str, float]]] = []
        for idx in indices:
            pairs: List[Tuple[str, float]] = []
            for item in updated[idx].get("host_line_candidates", [])[:3]:
                if not isinstance(item, dict):
                    continue
                line_id = str(item.get("line_id", ""))
                score = safe_float(item.get("score"), 1e18)
                if line_id:
                    pairs.append((line_id, score))
            if not pairs:
                pairs = [(str(x), float(rank)) for rank, x in enumerate(updated[idx].get("targets", [])) if str(x).startswith("ln_")]
            candidate_lists.append(pairs[:3])

        best_assign: Optional[List[str]] = None
        best_cost = 1e18

        def _search(pos: int, chosen: List[str], score_sum: float, counts: Dict[str, int]) -> None:
            nonlocal best_assign, best_cost
            if score_sum >= best_cost:
                return
            if pos >= len(candidate_lists):
                dup_cost = sum(max(0, c - 1) for c in counts.values()) * duplicate_penalty
                total = score_sum + dup_cost
                if total < best_cost:
                    best_cost = total
                    best_assign = list(chosen)
                return
            for line_id, score in candidate_lists[pos]:
                chosen.append(line_id)
                counts[line_id] = counts.get(line_id, 0) + 1
                _search(pos + 1, chosen, score_sum + score, counts)
                counts[line_id] -= 1
                if counts[line_id] <= 0:
                    counts.pop(line_id, None)
                chosen.pop()

        _search(0, [], 0.0, {})
        if best_assign is None:
            continue
        for local_idx, assigned_line in enumerate(best_assign):
            row = updated[indices[local_idx]]
            row["targets"] = [assigned_line]
            row["host_line_assigned_by_group"] = True
            updated[indices[local_idx]] = row
    return updated


def _retarget_symbol_row_to_line(
    row: Dict[str, Any],
    new_line_id: str,
    candidates_by_id: Dict[str, Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    updated = dict(row)
    line = candidates_by_id.get(new_line_id, {})
    endpoints = _line_endpoints(line) or ()
    targets = [new_line_id]
    for point_name in endpoints:
        point_id = _point_candidate_id_by_name(point_by_name, point_name)
        if point_id:
            targets.append(point_id)
    updated["targets"] = _dedupe_keep_order(targets)
    updated["retargeted_host_line"] = new_line_id
    return updated


def _expand_symbol_ownership_variants(
    symbol_ownership: Sequence[Dict[str, Any]],
    candidates_by_id: Dict[str, Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
    line_host_alt_gap_max: float,
    line_host_alt_topm: int,
) -> List[List[Dict[str, Any]]]:
    base_variant = [dict(row) for row in symbol_ownership]
    if line_host_alt_gap_max <= 0.0 or line_host_alt_topm <= 0:
        return [base_variant]

    eligible: List[Tuple[float, int, List[Tuple[str, float]]]] = []
    for idx, row in enumerate(symbol_ownership):
        host_candidates = row.get("host_line_candidates", [])
        if not isinstance(host_candidates, list) or len(host_candidates) < 2:
            continue
        top_line_id = str(host_candidates[0].get("line_id", ""))
        top_score = safe_float(host_candidates[0].get("score"), 1e18)
        alts: List[Tuple[str, float]] = []
        for item in host_candidates[1:]:
            line_id = str(item.get("line_id", ""))
            score = safe_float(item.get("score"), 1e18)
            gap = score - top_score
            if line_id and gap <= line_host_alt_gap_max:
                alts.append((line_id, gap))
        if not top_line_id or not alts:
            continue
        best_gap = min(gap for _, gap in alts)
        eligible.append((best_gap, idx, alts))

    eligible.sort(key=lambda x: (x[0], x[1]))
    variants: List[List[Dict[str, Any]]] = [base_variant]
    for _, idx, alts in eligible[:line_host_alt_topm]:
        for line_id, _ in alts:
            updated_variant = [dict(row) for row in symbol_ownership]
            updated_variant[idx] = _retarget_symbol_row_to_line(updated_variant[idx], line_id, candidates_by_id, point_by_name)
            variants.append(updated_variant)
    return variants


def _retarget_symbol_row_to_pair(
    row: Dict[str, Any],
    vertex_id: str,
    line_ids: Sequence[str],
) -> Dict[str, Any]:
    updated = dict(row)
    updated["targets"] = _dedupe_keep_order([vertex_id, *[str(x) for x in line_ids if str(x)]])
    updated["retargeted_pair_binding"] = True
    return updated


def _expand_pair_symbol_ownership_variants(
    symbol_ownership: Sequence[Dict[str, Any]],
    pair_host_alt_gap_max: float,
    pair_host_alt_topm: int,
) -> List[List[Dict[str, Any]]]:
    base_variant = [dict(row) for row in symbol_ownership]
    if pair_host_alt_gap_max <= 0.0 or pair_host_alt_topm <= 0:
        return [base_variant]

    eligible: List[Tuple[float, int, List[Tuple[str, List[str], float]]]] = []
    for idx, row in enumerate(symbol_ownership):
        pair_candidates = row.get("pair_line_candidates", [])
        if not isinstance(pair_candidates, list) or len(pair_candidates) < 2:
            continue
        top_score = safe_float(pair_candidates[0].get("score"), 1e18)
        alts: List[Tuple[str, List[str], float]] = []
        for item in pair_candidates[1:]:
            vertex_id = str(item.get("vertex_id", ""))
            line_ids = [str(x) for x in item.get("line_ids", []) if str(x)]
            score = safe_float(item.get("score"), 1e18)
            gap = score - top_score
            if vertex_id and len(line_ids) >= 2 and gap <= pair_host_alt_gap_max:
                alts.append((vertex_id, line_ids[:2], gap))
        if not alts:
            continue
        best_gap = min(gap for _, _, gap in alts)
        eligible.append((best_gap, idx, alts))

    eligible.sort(key=lambda x: (x[0], x[1]))
    variants: List[List[Dict[str, Any]]] = [base_variant]
    for _, idx, alts in eligible[:pair_host_alt_topm]:
        for vertex_id, line_ids, _ in alts:
            updated_variant = [dict(row) for row in symbol_ownership]
            updated_variant[idx] = _retarget_symbol_row_to_pair(updated_variant[idx], vertex_id, line_ids)
            variants.append(updated_variant)
    return variants


def _best_incident_line_pair(
    center: Sequence[float],
    vertex_point: Dict[str, Any],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair, _ = _best_incident_line_pair_with_score(center, vertex_point, lines, point_by_name)
    return pair


def _best_incident_line_pair_with_score(
    center: Sequence[float],
    vertex_point: Dict[str, Any],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    vertex_name = str(vertex_point.get("name", ""))
    vertex_xy = _point_xy_from_candidate(vertex_point)
    if not vertex_name or vertex_xy is None:
        return [], 1e18
    ray = (float(center[0]) - vertex_xy[0], float(center[1]) - vertex_xy[1])
    candidates = []
    for line in lines:
        if not _line_incident_to_point(line, vertex_name):
            continue
        other = _line_other_endpoint(line, vertex_name)
        if not other or other not in point_by_name:
            continue
        other_xy = _point_xy_from_candidate(point_by_name[other])
        if other_xy is None:
            continue
        vec = (other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1])
        candidates.append((line, vec))
    if len(candidates) < 2:
        return [], 1e18

    best_pair: List[Dict[str, Any]] = []
    best_score = 1e18
    for i in range(len(candidates)):
        line_a, vec_a = candidates[i]
        for j in range(i + 1, len(candidates)):
            line_b, vec_b = candidates[j]
            pair_angle = _angle_between(vec_a, vec_b)
            if pair_angle >= math.pi - 1e-6:
                continue
            dev = min(
                abs((_angle_between(ray, vec_a) + _angle_between(ray, vec_b)) - pair_angle),
                abs((_angle_between(ray, (-vec_a[0], -vec_a[1])) + _angle_between(ray, (-vec_b[0], -vec_b[1]))) - pair_angle),
            )
            dist_pen = _line_distance(line_a, center) + _line_distance(line_b, center)
            # Prefer wedges that actually contain the text/marker ray, then closer lines, then tighter angle.
            score = dev * 1000.0 + dist_pen + 0.01 * pair_angle + 1e-6 * (i + j)
            if score < best_score:
                best_score = score
                best_pair = [line_a, line_b]
    if best_pair:
        return best_pair, best_score
    ranked = sorted(candidates, key=lambda x: _line_distance(x[0], center))
    fallback = [ranked[0][0], ranked[1][0]]
    fallback_score = _line_distance(fallback[0], center) + _line_distance(fallback[1], center) + 500.0
    return fallback, fallback_score


def _rank_incident_line_pairs_with_scores(
    center: Sequence[float],
    vertex_point: Dict[str, Any],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
    topn: int = 3,
) -> List[Tuple[List[Dict[str, Any]], float]]:
    vertex_name = str(vertex_point.get("name", ""))
    vertex_xy = _point_xy_from_candidate(vertex_point)
    if not vertex_name or vertex_xy is None:
        return []
    ray = (float(center[0]) - vertex_xy[0], float(center[1]) - vertex_xy[1])
    candidates = []
    for line in lines:
        if not _line_incident_to_point(line, vertex_name):
            continue
        other = _line_other_endpoint(line, vertex_name)
        if not other or other not in point_by_name:
            continue
        other_xy = _point_xy_from_candidate(point_by_name[other])
        if other_xy is None:
            continue
        vec = (other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1])
        candidates.append((line, vec))
    if len(candidates) < 2:
        return []
    ranked: List[Tuple[List[Dict[str, Any]], float]] = []
    seen = set()
    for i in range(len(candidates)):
        line_a, vec_a = candidates[i]
        for j in range(i + 1, len(candidates)):
            line_b, vec_b = candidates[j]
            pair_angle = _angle_between(vec_a, vec_b)
            if pair_angle >= math.pi - 1e-6:
                continue
            dev = min(
                abs((_angle_between(ray, vec_a) + _angle_between(ray, vec_b)) - pair_angle),
                abs((_angle_between(ray, (-vec_a[0], -vec_a[1])) + _angle_between(ray, (-vec_b[0], -vec_b[1]))) - pair_angle),
            )
            dist_pen = _line_distance(line_a, center) + _line_distance(line_b, center)
            score = dev * 1000.0 + dist_pen + 0.01 * pair_angle + 1e-6 * (i + j)
            pair = [line_a, line_b]
            key = tuple(sorted(str(x.get("id", "")) for x in pair))
            if key in seen:
                continue
            seen.add(key)
            ranked.append((pair, score))
    ranked.sort(key=lambda x: x[1])
    return ranked[:topn]


def _best_angle_targets(
    center: Sequence[float],
    point_candidates: Sequence[Dict[str, Any]],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], float]:
    best_vertex: Optional[Dict[str, Any]] = None
    best_lines: List[Dict[str, Any]] = []
    best_score = 1e18
    for point in point_candidates:
        pair, score = _best_incident_line_pair_with_score(center, point, lines, point_by_name)
        if len(pair) < 2:
            continue
        if score < best_score:
            best_score = score
            best_vertex = point
            best_lines = pair
    return best_vertex, best_lines, best_score


def _best_perpendicular_pair_with_score(
    center: Sequence[float],
    vertex_point: Dict[str, Any],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], float]:
    vertex_name = str(vertex_point.get("name", ""))
    vertex_xy = _point_xy_from_candidate(vertex_point)
    if not vertex_name or vertex_xy is None:
        return [], 1e18
    ray = (float(center[0]) - vertex_xy[0], float(center[1]) - vertex_xy[1])
    candidates = []
    for line in lines:
        if not _line_incident_to_point(line, vertex_name):
            continue
        other = _line_other_endpoint(line, vertex_name)
        if not other or other not in point_by_name:
            continue
        other_xy = _point_xy_from_candidate(point_by_name[other])
        if other_xy is None:
            continue
        vec = (other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1])
        candidates.append((line, vec))
    if len(candidates) < 2:
        return [], 1e18

    best_pair: List[Dict[str, Any]] = []
    best_score = 1e18
    for i in range(len(candidates)):
        line_a, vec_a = candidates[i]
        for j in range(i + 1, len(candidates)):
            line_b, vec_b = candidates[j]
            pair_angle = _angle_between(vec_a, vec_b)
            if pair_angle >= math.pi - 1e-6:
                continue
            dev = min(
                abs((_angle_between(ray, vec_a) + _angle_between(ray, vec_b)) - pair_angle),
                abs((_angle_between(ray, (-vec_a[0], -vec_a[1])) + _angle_between(ray, (-vec_b[0], -vec_b[1]))) - pair_angle),
            )
            right_dev = abs(pair_angle - (math.pi / 2.0))
            dist_pen = _line_distance(line_a, center) + _line_distance(line_b, center)
            score = right_dev * 600.0 + dev * 300.0 + dist_pen + 1e-6 * (i + j)
            if score < best_score:
                best_score = score
                best_pair = [line_a, line_b]
    if best_pair:
        return best_pair, best_score

    ranked = sorted(candidates, key=lambda x: _line_distance(x[0], center))
    fallback = [ranked[0][0], ranked[1][0]]
    fallback_score = _line_distance(fallback[0], center) + _line_distance(fallback[1], center) + 500.0
    return fallback, fallback_score


def _rank_perpendicular_pairs_with_scores(
    center: Sequence[float],
    vertex_point: Dict[str, Any],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
    topn: int = 3,
) -> List[Tuple[List[Dict[str, Any]], float]]:
    vertex_name = str(vertex_point.get("name", ""))
    vertex_xy = _point_xy_from_candidate(vertex_point)
    if not vertex_name or vertex_xy is None:
        return []
    ray = (float(center[0]) - vertex_xy[0], float(center[1]) - vertex_xy[1])
    candidates = []
    for line in lines:
        if not _line_incident_to_point(line, vertex_name):
            continue
        other = _line_other_endpoint(line, vertex_name)
        if not other or other not in point_by_name:
            continue
        other_xy = _point_xy_from_candidate(point_by_name[other])
        if other_xy is None:
            continue
        vec = (other_xy[0] - vertex_xy[0], other_xy[1] - vertex_xy[1])
        candidates.append((line, vec))
    if len(candidates) < 2:
        return []
    ranked: List[Tuple[List[Dict[str, Any]], float]] = []
    seen = set()
    for i in range(len(candidates)):
        line_a, vec_a = candidates[i]
        for j in range(i + 1, len(candidates)):
            line_b, vec_b = candidates[j]
            pair_angle = _angle_between(vec_a, vec_b)
            if pair_angle >= math.pi - 1e-6:
                continue
            dev = min(
                abs((_angle_between(ray, vec_a) + _angle_between(ray, vec_b)) - pair_angle),
                abs((_angle_between(ray, (-vec_a[0], -vec_a[1])) + _angle_between(ray, (-vec_b[0], -vec_b[1]))) - pair_angle),
            )
            right_dev = abs(pair_angle - (math.pi / 2.0))
            dist_pen = _line_distance(line_a, center) + _line_distance(line_b, center)
            score = right_dev * 600.0 + dev * 300.0 + dist_pen + 1e-6 * (i + j)
            pair = [line_a, line_b]
            key = tuple(sorted(str(x.get("id", "")) for x in pair))
            if key in seen:
                continue
            seen.add(key)
            ranked.append((pair, score))
    ranked.sort(key=lambda x: x[1])
    return ranked[:topn]


def _best_perpendicular_targets(
    center: Sequence[float],
    point_candidates: Sequence[Dict[str, Any]],
    lines: Sequence[Dict[str, Any]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], float]:
    best_vertex: Optional[Dict[str, Any]] = None
    best_lines: List[Dict[str, Any]] = []
    best_score = 1e18
    for point in point_candidates:
        pair, score = _best_perpendicular_pair_with_score(center, point, lines, point_by_name)
        if len(pair) < 2:
            continue
        if score < best_score:
            best_score = score
            best_vertex = point
            best_lines = pair
    return best_vertex, best_lines, best_score


def _line_pair_candidate_rows(
    vertex_point: Optional[Dict[str, Any]],
    ranked_pairs: Sequence[Tuple[List[Dict[str, Any]], float]],
    point_by_name: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if vertex_point is None:
        return []
    vertex_name = str(vertex_point.get("name", ""))
    vertex_id = _point_candidate_id_by_name(point_by_name, vertex_name)
    if not vertex_id:
        return []
    rows: List[Dict[str, Any]] = []
    for pair, score in ranked_pairs:
        line_ids = [str(x.get("id", "")) for x in pair if str(x.get("id", ""))]
        if len(line_ids) < 2:
            continue
        rows.append(
            {
                "vertex_id": vertex_id,
                "line_ids": line_ids[:2],
                "score": round(score, 6),
            }
        )
    return rows


def _best_parallel_line_pair(
    center: Sequence[float],
    lines: Sequence[Dict[str, Any]],
    topn: int = 6,
) -> List[Dict[str, Any]]:
    nearby = _nearest_lines(center, lines, topn=topn)
    if len(nearby) < 2:
        return nearby[:2]
    best_pair: List[Dict[str, Any]] = []
    best_score = 1e18
    for i in range(len(nearby)):
        line_a = nearby[i]
        dir_a = _line_direction(line_a)
        if dir_a is None:
            continue
        for j in range(i + 1, len(nearby)):
            line_b = nearby[j]
            dir_b = _line_direction(line_b)
            if dir_b is None:
                continue
            align = min(_angle_between(dir_a, dir_b), abs(math.pi - _angle_between(dir_a, dir_b)))
            dist_pen = _line_distance(line_a, center) + _line_distance(line_b, center)
            shared_pen = 0.0
            if _shared_vertex(line_a, line_b) is not None:
                shared_pen = 50.0
            score = align * 100.0 + dist_pen + shared_pen + 1e-6 * (i + j)
            if score < best_score:
                best_score = score
                best_pair = [line_a, line_b]
    if best_pair:
        return best_pair
    return nearby[:2]


def _nearest_point(center: Sequence[float], point_candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not point_candidates:
        return None
    ranked = []
    for cand in point_candidates:
        xy = _point_xy_from_candidate(cand)
        if xy is None:
            continue
        ranked.append((math.hypot(center[0] - xy[0], center[1] - xy[1]), cand))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[0][1]


def _nearest_symbol_distance(
    center: Sequence[float],
    symbols: Sequence[Dict[str, Any]],
    sym_classes: Sequence[str],
    exclude_id: str = "",
) -> float:
    best = 1e9
    allowed = set(sym_classes)
    for symbol in symbols:
        if str(symbol.get("id", "")) == exclude_id:
            continue
        sym_class = _canonical_symbol_class(symbol.get("geometric_parameters", {}).get("sym_class", ""))
        if sym_class not in allowed:
            continue
        sym_center = _symbol_center(symbol)
        if sym_center is None:
            continue
        dist = math.hypot(float(center[0]) - sym_center[0], float(center[1]) - sym_center[1])
        if dist < best:
            best = dist
    return best


def _best_arc_targets(
    center: Sequence[float],
    circles: Sequence[Dict[str, Any]],
    point_candidates: Sequence[Dict[str, Any]],
    radius_tol_scale: float = 0.045,
    radius_tol_floor: float = 3.5,
    min_cluster_size: int = 2,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]]:
    text_theta = math.atan2(float(center[1]), float(center[0]))
    best_choice: Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]] = None
    best_score = 1e18

    for circle in circles:
        loc = circle.get("location", [])
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        circle_center = (float(loc[0]), float(loc[1]))
        center_name = str(circle.get("geometric_parameters", {}).get("center_point") or circle.get("name") or "")
        point_rows = []
        for point in point_candidates:
            point_name = _point_name(point)
            xy = _point_xy_from_candidate(point)
            if point_name in {None, center_name} or xy is None:
                continue
            radius = math.hypot(xy[0] - circle_center[0], xy[1] - circle_center[1])
            point_rows.append((radius, point))
        if len(point_rows) < 2:
            continue
        point_rows.sort(key=lambda x: x[0])

        clusters: List[List[Tuple[float, Dict[str, Any]]]] = []
        for radius, point in point_rows:
            placed = False
            for cluster in clusters:
                ref = sum(x[0] for x in cluster) / len(cluster)
                if abs(radius - ref) <= max(radius_tol_floor, ref * radius_tol_scale):
                    cluster.append((radius, point))
                    placed = True
                    break
            if not placed:
                clusters.append([(radius, point)])

        text_radius = math.hypot(float(center[0]) - circle_center[0], float(center[1]) - circle_center[1])
        for cluster in clusters:
            if len(cluster) < min_cluster_size:
                continue
            cluster_radius = sum(x[0] for x in cluster) / len(cluster)
            boundary_pen = abs(text_radius - cluster_radius)
            members = [point for _, point in cluster]
            for i in range(len(members)):
                xy_a = _point_xy_from_candidate(members[i])
                if xy_a is None:
                    continue
                theta_a = math.atan2(xy_a[1] - circle_center[1], xy_a[0] - circle_center[0])
                for j in range(i + 1, len(members)):
                    xy_b = _point_xy_from_candidate(members[j])
                    if xy_b is None:
                        continue
                    theta_b = math.atan2(xy_b[1] - circle_center[1], xy_b[0] - circle_center[0])
                    diff = ((theta_b - theta_a + math.pi) % (2.0 * math.pi)) - math.pi
                    mid_theta = theta_a + diff / 2.0
                    theta_pen = abs(((math.atan2(float(center[1]) - circle_center[1], float(center[0]) - circle_center[0]) - mid_theta + math.pi) % (2.0 * math.pi)) - math.pi)
                    span_pen = abs(diff)
                    score = boundary_pen + 30.0 * theta_pen + 2.0 * span_pen
                    if score < best_score:
                        best_score = score
                        best_choice = (circle, members[i], members[j], boundary_pen)
    return best_choice


def _base_relations(
    candidate_payload: Dict[str, Any],
    candidate_ids: Sequence[str],
    degree_arc_policy: str = "default",
    global_angle_search: bool = False,
    selective_global_angle: bool = False,
    angle_global_margin: float = 8.0,
    perpendicular_global_margin: float = 1e9,
    grouped_line_duplicate_penalty: float = 0.0,
    parallel_companion_weight: float = 0.0,
    len_text_mark_support_bonus: float = 0.0,
) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    def _stable_cand_key(cand: Dict[str, Any]) -> Tuple[str, str, float, Tuple[float, ...]]:
        loc = cand.get("location", [])
        loc_key: Tuple[float, ...]
        if isinstance(loc, list):
            loc_key = tuple(float(x) for x in loc[:4] if isinstance(x, (int, float)) or str(x).replace(".", "", 1).replace("-", "", 1).isdigit())
        else:
            loc_key = ()
        return (
            str(cand.get("type", "")),
            str(cand.get("id", "")),
            safe_float(cand.get("confidence"), 0.0),
            loc_key,
        )

    candidates = candidate_payload.get("candidates", [])
    cand_set = set(candidate_ids)
    base_rel: List[List[Any]] = []
    symbol_ownership: List[Dict[str, Any]] = []

    points = sorted(
        [c for c in candidates if str(c.get("type")) == "point" and str(c.get("id")) in cand_set],
        key=_stable_cand_key,
    )
    lines = sorted(
        [c for c in candidates if str(c.get("type")) == "line" and str(c.get("id")) in cand_set],
        key=_stable_cand_key,
    )
    circles = sorted(
        [c for c in candidates if str(c.get("type")) == "circle" and str(c.get("id")) in cand_set],
        key=_stable_cand_key,
    )
    symbols = sorted(
        [
        c
        for c in candidates
        if str(c.get("type")) in {"text_label", "geometry_symbol"} and str(c.get("id")) in cand_set
        ],
        key=_stable_cand_key,
    )
    degree_symbol_count = sum(1 for c in symbols if str(c.get("geometric_parameters", {}).get("text_class", "")) == "degree")
    angle_symbol_count = sum(1 for c in symbols if str(c.get("geometric_parameters", {}).get("text_class", "")) == "angle")
    has_circle_candidates = bool(circles)
    length_mark_rankings: List[Dict[str, Any]] = []

    for symbol in symbols:
        if str(symbol.get("type")) != "geometry_symbol":
            continue
        mark_class = _canonical_symbol_class(symbol.get("geometric_parameters", {}).get("sym_class", ""))
        if mark_class not in LENGTH_LIKE_MARK_CLASSES:
            continue
        center = _symbol_center(symbol)
        if center is None:
            continue
        ranked = _rank_host_lines_for_mark(
            symbol,
            (float(center[0]), float(center[1])),
            lines,
            prefer_parallel=False,
            topn=6,
        )
        if not ranked:
            continue
        length_mark_rankings.append(
            {
                "symbol_id": str(symbol.get("id", "")),
                "center": (float(center[0]), float(center[1])),
                "line_ids": [_line_id(line) for line, _ in ranked if _line_id(line)],
            }
        )

    point_by_name = {str(c.get("name")): c for c in points if c.get("name")}
    for line in lines:
        cid = str(line.get("id"))
        endpoints = _line_endpoints(line)
        if not endpoints:
            continue
        for point_name in endpoints:
            point = point_by_name.get(point_name)
            if point is None:
                continue
            base_rel.append([str(point.get("id")), cid, "endpoint"])

    for symbol in symbols:
        center = _symbol_center(symbol)
        if center is None:
            continue
        center_xy = (float(center[0]), float(center[1]))
        symbol_id = str(symbol.get("id"))
        pair_candidate_rows: List[Dict[str, Any]] = []
        ranked_hosts: List[Tuple[Dict[str, Any], float]] = []
        geom = symbol.get("geometric_parameters", {})
        sym_class = _canonical_symbol_class(geom.get("sym_class", ""))
        text_class = str(geom.get("text_class", ""))
        targets: List[str] = []

        if str(symbol.get("type")) == "geometry_symbol" and sym_class in {
            "perpendicular",
            "parallel",
            "double_parallel",
            "bar",
            "double_bar",
            "triple_bar",
            "angle",
            "double_angle",
            "triple_angle",
            "quad_angle",
        }:
            nearest_point = _nearest_point(center_xy, points)
            nearest = []
            pair_candidate_rows: List[Dict[str, Any]] = []
            if nearest_point is not None and sym_class == "perpendicular":
                local_pair, local_score = _best_perpendicular_pair_with_score(center_xy, nearest_point, lines, point_by_name)
                best_vertex, best_lines, best_score = _best_perpendicular_targets(center_xy, points, lines, point_by_name)
                local_ranked_pairs = _rank_perpendicular_pairs_with_scores(center_xy, nearest_point, lines, point_by_name, topn=3)
                use_global_perpendicular = (
                    len(local_pair) < 2
                    or (
                        best_vertex is not None
                        and len(best_lines) >= 2
                        and best_score + perpendicular_global_margin < local_score
                    )
                )
                if use_global_perpendicular and best_vertex is not None and len(best_lines) >= 2:
                    nearest_point = best_vertex
                    nearest = best_lines
                    pair_candidate_rows = _line_pair_candidate_rows(
                        best_vertex,
                        _rank_perpendicular_pairs_with_scores(center_xy, best_vertex, lines, point_by_name, topn=3),
                        point_by_name,
                    )
                else:
                    nearest = local_pair
                    pair_candidate_rows = _line_pair_candidate_rows(nearest_point, local_ranked_pairs, point_by_name)
            elif sym_class in LENGTH_LIKE_MARK_CLASSES:
                ranked_hosts = _rank_host_lines_for_mark(symbol, center_xy, lines, prefer_parallel=False, topn=6)
                nearest = [ranked_hosts[0][0]] if ranked_hosts else []
            elif sym_class in PARALLEL_LIKE_MARK_CLASSES:
                ranked_hosts = _rank_host_lines_for_mark(
                    symbol,
                    center_xy,
                    lines,
                    prefer_parallel=True,
                    topn=6,
                    parallel_companion_weight=parallel_companion_weight,
                )
                nearest = [ranked_hosts[0][0]] if ranked_hosts else []
            elif nearest_point is not None and sym_class in {"angle"} | GROUPED_ANGLE_MARK_CLASSES:
                local_pair, local_score = _best_incident_line_pair_with_score(center_xy, nearest_point, lines, point_by_name)
                best_vertex, best_lines, best_score = _best_angle_targets(center_xy, points, lines, point_by_name)
                local_ranked_pairs = _rank_incident_line_pairs_with_scores(center_xy, nearest_point, lines, point_by_name, topn=3)
                use_global_angle = (
                    len(local_pair) < 2
                    or (
                        best_vertex is not None
                        and len(best_lines) >= 2
                        and best_score + angle_global_margin < local_score
                    )
                )
                if use_global_angle and best_vertex is not None and len(best_lines) >= 2:
                    nearest_point = best_vertex
                    nearest = best_lines
                    pair_candidate_rows = _line_pair_candidate_rows(
                        best_vertex,
                        _rank_incident_line_pairs_with_scores(center_xy, best_vertex, lines, point_by_name, topn=3),
                        point_by_name,
                    )
                else:
                    nearest = local_pair
                    pair_candidate_rows = _line_pair_candidate_rows(nearest_point, local_ranked_pairs, point_by_name)
            if not nearest:
                nearest = _nearest_lines(center_xy, lines, topn=2)
            targets.extend([str(x.get("id")) for x in nearest])
            vertex = None
            if len(nearest) >= 2:
                vertex = _shared_vertex(nearest[0], nearest[1])
            if vertex and vertex in point_by_name:
                targets.insert(0, str(point_by_name[vertex].get("id")))
        elif text_class == "len":
            ranked_hosts = _rank_host_lines_for_mark(
                symbol,
                center_xy,
                lines,
                prefer_parallel=False,
                topn=8,
                midpoint_weight=24.0,
                overflow_weight=48.0,
            )
            if len_text_mark_support_bonus > 0.0 and ranked_hosts:
                rescored: List[Tuple[Dict[str, Any], float]] = []
                for line, score in ranked_hosts:
                    support_bonus = _length_mark_support_bonus(center_xy, _line_id(line), length_mark_rankings)
                    rescored.append((line, score - len_text_mark_support_bonus * support_bonus))
                rescored.sort(key=lambda item: item[1])
                ranked_hosts = rescored
            nearest = [ranked_hosts[0][0]] if ranked_hosts else []
            if nearest:
                targets.append(str(nearest[0].get("id")))
                endpoints = _line_endpoints(nearest[0]) or ()
                for point_name in endpoints:
                    if point_name in point_by_name:
                        targets.append(str(point_by_name[point_name].get("id")))
        elif text_class in {"degree", "angle"}:
            vertex_cand = _nearest_point(center_xy, points)
            best_angle_vertex = None
            best_angle_lines: List[Dict[str, Any]] = []
            pair_candidate_rows: List[Dict[str, Any]] = []
            use_global_search_here = global_angle_search
            scalar_text = _normalize_scalar_text(geom.get("text_content") or geom.get("text"))
            is_numeric_degree = bool(scalar_text) and bool(re.fullmatch(r"[0-9.]+", scalar_text))
            prefer_local_symbolic_degree = (
                text_class == "degree"
                and has_circle_candidates
                and not is_numeric_degree
                and _symbol_box_aspect(symbol) >= 2.5
            )
            if global_angle_search and selective_global_angle:
                if text_class == "degree":
                    use_global_search_here = (not has_circle_candidates and degree_symbol_count == 1)
                else:
                    use_global_search_here = angle_symbol_count >= 3
            if prefer_local_symbolic_degree:
                use_global_search_here = False
            if use_global_search_here:
                best_angle_vertex, best_angle_lines, _ = _best_angle_targets(center_xy, points, lines, point_by_name)
                if best_angle_vertex is not None:
                    vertex_cand = best_angle_vertex
                    pair_candidate_rows = _line_pair_candidate_rows(
                        best_angle_vertex,
                        _rank_incident_line_pairs_with_scores(center_xy, best_angle_vertex, lines, point_by_name, topn=3),
                        point_by_name,
                    )
            if text_class == "degree" and degree_arc_policy in {"selective_arc", "conservative_arc", "strict_selective_arc", "clustered_selective_arc"}:
                vertex_dist = 1e9
                if vertex_cand is not None:
                    vertex_xy = _point_xy_from_candidate(vertex_cand)
                    if vertex_xy is not None:
                        vertex_dist = math.hypot(center_xy[0] - vertex_xy[0], center_xy[1] - vertex_xy[1])
                arc_choice = _best_arc_targets(
                    center_xy,
                    circles,
                    points,
                    radius_tol_scale=0.03 if degree_arc_policy in {"strict_selective_arc", "clustered_selective_arc"} else 0.045,
                    radius_tol_floor=2.5 if degree_arc_policy in {"strict_selective_arc", "clustered_selective_arc"} else 3.5,
                    min_cluster_size=3 if degree_arc_policy in {"strict_selective_arc", "clustered_selective_arc"} else 2,
                )
                arc_hint_dist = _nearest_symbol_distance(
                    center_xy,
                    symbols,
                    sym_classes=["arrow", "head"],
                    exclude_id=symbol_id,
                )
                if arc_choice is not None:
                    circle_cand, point_a, point_b, boundary_pen = arc_choice
                    loc = circle_cand.get("location", [])
                    cluster_radius = None
                    if isinstance(loc, list) and len(loc) >= 2:
                        circle_center = (float(loc[0]), float(loc[1]))
                        point_a_xy = _point_xy_from_candidate(point_a)
                        point_b_xy = _point_xy_from_candidate(point_b)
                        radii = []
                        if point_a_xy is not None:
                            radii.append(math.hypot(point_a_xy[0] - circle_center[0], point_a_xy[1] - circle_center[1]))
                        if point_b_xy is not None:
                            radii.append(math.hypot(point_b_xy[0] - circle_center[0], point_b_xy[1] - circle_center[1]))
                        if radii:
                            cluster_radius = sum(radii) / len(radii)
                    text_radius = None
                    if cluster_radius is not None and isinstance(loc, list) and len(loc) >= 2:
                        text_radius = math.hypot(center_xy[0] - float(loc[0]), center_xy[1] - float(loc[1]))
                    numeric_text = is_numeric_degree
                    use_arc = False
                    if prefer_local_symbolic_degree:
                        use_arc = False
                    elif degree_arc_policy in {"selective_arc", "clustered_selective_arc"}:
                        use_arc = (
                            arc_hint_dist <= 80.0
                            or (boundary_pen <= 45.0 and boundary_pen + 10.0 < vertex_dist)
                        )
                    else:
                        not_deep_inside = (
                            text_radius is None
                            or cluster_radius is None
                            or text_radius >= cluster_radius - (15.0 if numeric_text else 10.0)
                        )
                        if degree_arc_policy == "strict_selective_arc":
                            use_arc = (
                                not_deep_inside
                                and (
                                    arc_hint_dist <= 55.0
                                    or (boundary_pen <= 28.0 and boundary_pen + 5.0 < vertex_dist)
                                )
                            )
                        else:
                            use_arc = (
                                not_deep_inside
                                and (
                                    (arc_hint_dist <= 65.0 and boundary_pen <= max(40.0, vertex_dist + 5.0))
                                    or (boundary_pen <= 32.0 and boundary_pen + 8.0 < vertex_dist)
                                )
                            )
                    if use_arc:
                        targets.append(str(circle_cand.get("id")))
                        targets.append(str(point_a.get("id")))
                        targets.append(str(point_b.get("id")))
            if targets:
                pass
            elif vertex_cand is not None:
                vertex_name = str(vertex_cand.get("name"))
                targets.append(str(vertex_cand.get("id")))
                incident = list(best_angle_lines) if best_angle_lines and vertex_cand is best_angle_vertex else _best_incident_line_pair(center_xy, vertex_cand, lines, point_by_name)
                if not pair_candidate_rows:
                    pair_candidate_rows = _line_pair_candidate_rows(
                        vertex_cand,
                        _rank_incident_line_pairs_with_scores(center_xy, vertex_cand, lines, point_by_name, topn=3),
                        point_by_name,
                    )
                if len(incident) < 2:
                    incident = [line for line in lines if _line_incident_to_point(line, vertex_name)]
                    if len(incident) < 2:
                        incident = _nearest_lines(center_xy, lines, topn=2)
                    else:
                        incident = sorted(incident, key=lambda c: _line_distance(c, center_xy))[:2]
                targets.extend(str(x.get("id")) for x in incident[:2])
        elif text_class == "point":
            nearest_point = _nearest_point(center_xy, points)
            if nearest_point is not None:
                targets.append(str(nearest_point.get("id")))
        targets = [x for x in _dedupe_keep_order(targets) if x in cand_set]
        if not targets:
            continue
        row = {"symbol": symbol_id, "targets": targets, "inferred": True, "sym_class": sym_class}
        if sym_class in GROUPED_LINE_MARK_CLASSES | LENGTH_LIKE_MARK_CLASSES | PARALLEL_LIKE_MARK_CLASSES:
            ranked_hosts = locals().get("ranked_hosts", [])
            if ranked_hosts:
                row["host_line_candidates"] = [
                    {"line_id": str(line.get("id", "")), "score": round(score, 6)}
                    for line, score in ranked_hosts[:3]
                    if str(line.get("id", ""))
                ]
        pair_candidates = locals().get("pair_candidate_rows", [])
        if pair_candidates:
            row["pair_line_candidates"] = pair_candidates
        symbol_ownership.append(row)

    symbol_ownership = _diversify_grouped_line_symbol_hosts(symbol_ownership, grouped_line_duplicate_penalty)
    base_rel.extend(
        [[str(row.get("symbol")), list(row.get("targets", [])), "attached_to"] for row in symbol_ownership if row.get("targets")]
    )
    return _dedupe_keep_order(base_rel), symbol_ownership


def _relation_is_closed(relation: List[Any], active: set) -> bool:
    if len(relation) < 2:
        return False
    left = relation[0]
    right = relation[1]
    left_ok = isinstance(left, str) and left in active
    if isinstance(right, list):
        right_ok = all(isinstance(x, str) and x in active for x in right)
    else:
        right_ok = isinstance(right, str) and right in active
    return left_ok and right_ok


def _continuous_parameters(candidates_by_id: Dict[str, Dict[str, Any]], active_ids: Sequence[str]) -> Dict[str, Any]:
    points: Dict[str, List[float]] = {}
    lines: Dict[str, Dict[str, Any]] = {}
    circles: Dict[str, Dict[str, Any]] = {}
    symbols: Dict[str, Any] = {}

    for cid in active_ids:
        cand = candidates_by_id.get(cid)
        if not cand:
            continue
        ctype = str(cand.get("type"))
        name = str(cand.get("name", ""))
        if ctype == "point":
            loc = cand.get("location", [])
            if isinstance(loc, list) and len(loc) >= 2:
                points[name] = [float(loc[0]), float(loc[1])]
        elif ctype == "line":
            lines[name] = {
                "endpoints": list(cand.get("geometric_parameters", {}).get("endpoints", [None, None])),
                "location": list(cand.get("location", [])),
            }
        elif ctype == "circle":
            circles[name] = {
                "center": cand.get("geometric_parameters", {}).get("center"),
                "location": list(cand.get("location", [])),
            }
        elif ctype in {"text_label", "geometry_symbol"}:
            symbols[cid] = {
                "name": name,
                "location": list(cand.get("location", [])),
                "meta": cand.get("geometric_parameters", {}),
            }

    return {
        "point_coordinates": points,
        "line_parameters": lines,
        "circle_parameters": circles,
        "symbol_positions": symbols,
    }


def _logic_forms_from_symbol_targets(
    symbol_row: Dict[str, Any],
    candidates_by_id: Dict[str, Dict[str, Any]],
    active_ids: set,
    refined_line_naming: bool = True,
    coarse_length_text: bool = False,
) -> List[str]:
    ordered_active_ids = sorted(str(x) for x in active_ids)
    symbol_cid = symbol_row["symbol"]
    if symbol_cid not in active_ids:
        return []
    symbol = candidates_by_id.get(symbol_cid)
    if not symbol:
        return []
    targets = [x for x in symbol_row.get("targets", []) if x in active_ids]
    if not targets:
        return []

    geom = symbol.get("geometric_parameters", {})
    symbol_type = str(symbol.get("type", ""))
    sym_class = _canonical_symbol_class(geom.get("sym_class", ""))
    text_class = str(geom.get("text_class", ""))
    text_content = geom.get("text_content")
    if text_content in {None, ""}:
        text_content = geom.get("text")

    line_targets = [candidates_by_id[t] for t in targets if t.startswith("ln_") and t in candidates_by_id]
    point_targets = [candidates_by_id[t] for t in targets if t.startswith("pt_") and t in candidates_by_id]
    circle_targets = [candidates_by_id[t] for t in targets if t.startswith("cc_") and t in candidates_by_id]
    active_point_candidates = [
        candidates_by_id[t]
        for t in ordered_active_ids
        if isinstance(t, str) and t.startswith("pt_") and t in candidates_by_id
    ]
    active_circle_candidates = [
        candidates_by_id[t]
        for t in ordered_active_ids
        if isinstance(t, str) and t.startswith("cc_") and t in candidates_by_id
    ]

    forms: List[str] = []
    if sym_class == "perpendicular" and len(line_targets) >= 2:
        vertex = _shared_vertex(line_targets[0], line_targets[1])
        if vertex:
            l1 = _ray_line_expr(line_targets[0], vertex, active_point_candidates)
            l2 = _ray_line_expr(line_targets[1], vertex, active_point_candidates)
        else:
            l1 = _line_expr(line_targets[0])
            l2 = _line_expr(line_targets[1])
        if l1 and l2:
            forms.append(f"Perpendicular({l1}, {l2})")
    elif sym_class == "parallel" and len(line_targets) >= 2:
        anchor = _symbol_center(symbol)
        if refined_line_naming and anchor is not None:
            l1 = _segment_expr_near_anchor(line_targets[0], active_point_candidates, anchor)
            l2 = _segment_expr_near_anchor(line_targets[1], active_point_candidates, anchor)
        else:
            l1 = _line_expr(line_targets[0])
            l2 = _line_expr(line_targets[1])
        if l1 and l2:
            forms.append(f"Parallel({l1}, {l2})")
    elif sym_class == "bar" and len(line_targets) >= 2:
        anchor = _symbol_center(symbol)
        if refined_line_naming and anchor is not None:
            l1 = _segment_expr_near_anchor(line_targets[0], active_point_candidates, anchor)
            l2 = _segment_expr_near_anchor(line_targets[1], active_point_candidates, anchor)
        else:
            l1 = _line_expr(line_targets[0])
            l2 = _line_expr(line_targets[1])
        if l1 and l2:
            forms.append(f"Equals(LengthOf({l1}), LengthOf({l2}))")
    elif (sym_class == "text" or symbol_type == "text_label") and text_class == "len" and line_targets:
        anchor = _symbol_center(symbol)
        if coarse_length_text:
            line_expr = _line_expr(line_targets[0])
        elif refined_line_naming and anchor is not None:
            line_expr = _segment_expr_near_anchor(line_targets[0], active_point_candidates, anchor)
        else:
            line_expr = _line_expr(line_targets[0])
        scalar_text = _normalize_scalar_text(text_content)
        if line_expr and scalar_text is not None:
            forms.append(f"Equals(LengthOf({line_expr}), {scalar_text})")
    elif (sym_class == "text" or symbol_type == "text_label") and text_class == "degree" and circle_targets and len(point_targets) >= 2:
        point_names = sorted(
            [name for name in (_point_name(point_targets[0]), _point_name(point_targets[1])) if name]
        )
        scalar_text = _normalize_scalar_text(text_content)
        if len(point_names) >= 2 and scalar_text is not None:
            forms.append(f"Equals(MeasureOf(Arc({point_names[0]}, {point_names[1]})), {scalar_text})")
    elif (sym_class == "text" or symbol_type == "text_label") and text_class == "degree" and len(line_targets) >= 2:
        angle = _named_angle_from_lines(line_targets[0], line_targets[1], active_point_candidates)
        scalar_text = _normalize_scalar_text(text_content)
        if angle and scalar_text is not None:
            a, b, c = angle
            if len({a, b, c}) < 3:
                anchor = _symbol_center(symbol)
                if anchor is not None:
                    arc_choice = _best_arc_targets(anchor, active_circle_candidates, active_point_candidates)
                    if arc_choice is not None:
                        _, point_a, point_b, _ = arc_choice
                        point_names = sorted(
                            [name for name in (_point_name(point_a), _point_name(point_b)) if name]
                        )
                        if len(point_names) >= 2:
                            forms.append(f"Equals(MeasureOf(Arc({point_names[0]}, {point_names[1]})), {scalar_text})")
                            return forms
            forms.append(f"Equals(MeasureOf(Angle({a}, {b}, {c})), {scalar_text})")
    elif (sym_class == "text" or symbol_type == "text_label") and text_class == "angle" and len(line_targets) >= 2:
        angle = _named_angle_from_lines(line_targets[0], line_targets[1], active_point_candidates)
        label_text = _normalize_group_text(text_content)
        if angle and label_text is not None:
            a, b, c = angle
            forms.append(f"Equals(MeasureOf(Angle({a}, {b}, {c})), MeasureOf(angle {label_text}))")
    elif circle_targets and point_targets:
        point_name = _point_name(point_targets[0])
        circle_expr = _circle_expr(circle_targets[0])
        if point_name and circle_expr:
            forms.append(f"PointLiesOnCircle({point_name}, {circle_expr})")

    return forms


def _prune_conflicting_text_symbols(
    symbol_ownership: Sequence[Dict[str, Any]],
    candidates_by_id: Dict[str, Dict[str, Any]],
    active_ids: set,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    len_groups: Dict[str, List[Dict[str, Any]]] = {}
    deg_groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}

    for row in symbol_ownership:
        sid = str(row.get("symbol", ""))
        symbol = candidates_by_id.get(sid, {})
        geom = symbol.get("geometric_parameters", {})
        text_class = str(geom.get("text_class", ""))
        if text_class == "len":
            line_ids = [x for x in row.get("targets", []) if isinstance(x, str) and x.startswith("ln_") and x in active_ids]
            if line_ids:
                len_groups.setdefault(line_ids[0], []).append(row)
                continue
        if text_class == "degree":
            line_ids = [x for x in row.get("targets", []) if isinstance(x, str) and x.startswith("ln_") and x in active_ids]
            if len(line_ids) >= 2:
                angle = _angle_from_lines(candidates_by_id.get(line_ids[0], {}), candidates_by_id.get(line_ids[1], {}))
                if angle:
                    deg_groups.setdefault(angle, []).append(row)
                    continue
        kept.append(row)

    def _row_conf(row: Dict[str, Any]) -> float:
        sid = str(row.get("symbol", ""))
        return safe_float(candidates_by_id.get(sid, {}).get("confidence"), 0.0)

    def _row_text(row: Dict[str, Any]) -> Optional[str]:
        sid = str(row.get("symbol", ""))
        geom = candidates_by_id.get(sid, {}).get("geometric_parameters", {})
        return _normalize_scalar_text(geom.get("text_content") or geom.get("text"))

    for rows in len_groups.values():
        by_text: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            text = _row_text(row)
            if text is None:
                continue
            prev = by_text.get(text)
            if prev is None or _row_conf(row) > _row_conf(prev):
                by_text[text] = row
        if by_text:
            kept.append(max(by_text.values(), key=_row_conf))

    for rows in deg_groups.values():
        by_text: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            text = _row_text(row)
            if text is None:
                continue
            prev = by_text.get(text)
            if prev is None or _row_conf(row) > _row_conf(prev):
                by_text[text] = row
        if by_text:
            kept.append(max(by_text.values(), key=_row_conf))

    return kept


def _base_logic_forms(
    relations: Sequence[List[Any]],
    symbol_ownership: Sequence[Dict[str, Any]],
    candidates_by_id: Dict[str, Dict[str, Any]],
    active_ids: set,
    refined_line_naming: bool = True,
    strict_circle_completion: bool = False,
    line_completion_mode: str = "default",
    coarse_length_text: bool = False,
) -> List[str]:
    forms: List[str] = []
    ordered_active_ids = sorted(str(x) for x in active_ids)
    active_point_candidates = [
        candidates_by_id[t]
        for t in ordered_active_ids
        if isinstance(t, str) and t.startswith("pt_") and t in candidates_by_id
    ]
    class_counts: Dict[str, int] = {}
    for row in symbol_ownership:
        sym_class = str(row.get("sym_class", ""))
        class_counts[sym_class] = class_counts.get(sym_class, 0) + 1

    for row in symbol_ownership:
        sym_class = str(row.get("sym_class", ""))
        if sym_class == "bar":
            continue
        if sym_class == "parallel":
            continue
        if sym_class == "angle":
            continue
        forms.extend(
            _logic_forms_from_symbol_targets(
                row,
                candidates_by_id,
                active_ids,
                refined_line_naming=refined_line_naming,
                coarse_length_text=coarse_length_text,
            )
        )

    grouped_host_lines: Dict[str, List[str]] = {}
    grouped_host_angles: Dict[str, List[str]] = {}
    grouped_angle_texts: Dict[str, List[str]] = {}
    for row in symbol_ownership:
        sym_class = str(row.get("sym_class", ""))
        if sym_class not in GROUPED_LINE_MARK_CLASSES and not (
            sym_class in {"bar", "parallel"}
        ):
            if sym_class not in GROUPED_ANGLE_MARK_CLASSES and not (
                sym_class == "angle"
            ):
                continue
            line_ids = [x for x in row.get("targets", []) if isinstance(x, str) and x.startswith("ln_") and x in active_ids]
            if len(line_ids) < 2:
                continue
            active_point_candidates = [
                candidates_by_id[t]
                for t in ordered_active_ids
                if isinstance(t, str) and t.startswith("pt_") and t in candidates_by_id
            ]
            angle = _named_angle_from_lines(
                candidates_by_id.get(line_ids[0], {}),
                candidates_by_id.get(line_ids[1], {}),
                active_point_candidates,
            )
            if not angle:
                continue
            a, b, c = angle
            grouped_host_angles.setdefault(sym_class, []).append(f"MeasureOf(Angle({a}, {b}, {c}))")
            continue
        if sym_class == "" and candidates_by_id.get(row.get("symbol"), {}).get("geometric_parameters", {}).get("text_class") == "angle":
            line_ids = [x for x in row.get("targets", []) if isinstance(x, str) and x.startswith("ln_") and x in active_ids]
            if len(line_ids) < 2:
                continue
            active_point_candidates = [
                candidates_by_id[t]
                for t in ordered_active_ids
                if isinstance(t, str) and t.startswith("pt_") and t in candidates_by_id
            ]
            angle = _named_angle_from_lines(
                candidates_by_id.get(line_ids[0], {}),
                candidates_by_id.get(line_ids[1], {}),
                active_point_candidates,
            )
            if not angle:
                continue
            symbol = candidates_by_id.get(row.get("symbol"), {})
            geom = symbol.get("geometric_parameters", {})
            label = _normalize_group_text(geom.get("text_content") or geom.get("text"))
            if label is None:
                continue
            a, b, c = angle
            grouped_angle_texts.setdefault(label, []).append(f"MeasureOf(Angle({a}, {b}, {c}))")
            continue
        line_ids = [x for x in row.get("targets", []) if isinstance(x, str) and x.startswith("ln_") and x in active_ids]
        if not line_ids:
            continue
        symbol = candidates_by_id.get(row.get("symbol"), {})
        anchor = _symbol_center(symbol)
        line_expr = None
        if refined_line_naming and anchor is not None:
            line_expr = _segment_expr_near_anchor(
                candidates_by_id.get(line_ids[0], {}),
                active_point_candidates,
                anchor,
            )
        if line_expr is None:
            line_expr = _line_expr(candidates_by_id.get(line_ids[0], {}))
        if line_expr:
            grouped_host_lines.setdefault(sym_class, []).append(line_expr)

    for sym_class in sorted(grouped_host_lines):
        line_exprs = grouped_host_lines[sym_class]
        unique_line_exprs = _dedupe_keep_order(line_exprs)
        if len(unique_line_exprs) < 2:
            continue
        if len(unique_line_exprs) == 2:
            pairs = [(unique_line_exprs[0], unique_line_exprs[1])]
        else:
            pairs = []
            for i in range(len(unique_line_exprs)):
                for j in range(i + 1, len(unique_line_exprs)):
                    pairs.append((unique_line_exprs[i], unique_line_exprs[j]))
        for left, right in pairs:
            if sym_class in PARALLEL_LIKE_MARK_CLASSES:
                forms.append(f"Parallel({left}, {right})")
            else:
                forms.append(f"Equals(LengthOf({left}), LengthOf({right}))")

    for sym_class in sorted(grouped_host_angles):
        angle_exprs = grouped_host_angles[sym_class]
        unique_angle_exprs = _dedupe_keep_order(angle_exprs)
        if len(unique_angle_exprs) < 2:
            continue
        if len(unique_angle_exprs) == 2:
            pairs = [(unique_angle_exprs[0], unique_angle_exprs[1])]
        else:
            pairs = []
            for i in range(len(unique_angle_exprs)):
                for j in range(i + 1, len(unique_angle_exprs)):
                    pairs.append((unique_angle_exprs[i], unique_angle_exprs[j]))
        for left, right in pairs:
            forms.append(f"Equals({left}, {right})")

    for label in sorted(grouped_angle_texts):
        angle_exprs = grouped_angle_texts[label]
        unique_angle_exprs = _dedupe_keep_order(angle_exprs)
        if len(unique_angle_exprs) < 2:
            continue
        pairs = []
        for i in range(len(unique_angle_exprs)):
            for j in range(i + 1, len(unique_angle_exprs)):
                pairs.append((unique_angle_exprs[i], unique_angle_exprs[j]))
        for left, right in pairs:
            forms.append(f"Equals({left}, {right})")

    active_points = []
    active_lines = []
    active_circles = []
    coords = []
    for cid in active_ids:
        cand = candidates_by_id.get(cid)
        if not cand:
            continue
        ctype = str(cand.get("type"))
        if ctype == "point":
            active_points.append(cand)
            xy = _point_xy_from_candidate(cand)
            if xy is not None:
                coords.append(xy)
        elif ctype == "line":
            active_lines.append(cand)
        elif ctype == "circle":
            active_circles.append(cand)

    if coords:
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        diag = max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)
    else:
        diag = 1.0

    # Relation completion: add non-endpoint collinear points on active lines.
    line_completion_tol = {
        "strict": 0.012,
        "default": 0.018,
        "relaxed": 0.024,
    }.get(line_completion_mode, 0.018)
    for line in active_lines:
        loc = line.get("location", [])
        endpoints = set(_line_endpoints(line) or ())
        line_expr = _line_expr(line)
        if line_expr is None or not isinstance(loc, list) or len(loc) < 4:
            continue
        a = (float(loc[0]), float(loc[1]))
        b = (float(loc[2]), float(loc[3]))
        inferred_point_rows = []
        for point in active_points:
            point_name = _point_name(point)
            xy = _point_xy_from_candidate(point)
            if point_name is None or xy is None or point_name in endpoints:
                continue
            dist = _distance_point_to_segment(xy, a, b)
            if dist / diag <= line_completion_tol:
                inferred_point_rows.append((dist / diag, point_name))
        inferred_point_rows.sort(key=lambda x: x[0])
        for _, point_name in inferred_point_rows:
            forms.append(f"PointLiesOnLine({point_name}, {line_expr})")

    # Circle completion: infer co-circular points and separate radius groups for shared / nearly shared centers.
    circle_rows = []
    for circle in active_circles:
        center_point = str(circle.get("geometric_parameters", {}).get("center_point") or "")
        anchor_xy = None
        if center_point:
            for point in active_points:
                if _point_name(point) == center_point:
                    anchor_xy = _point_xy_from_candidate(point)
                    break
        if anchor_xy is None:
            loc = circle.get("location", [])
            if isinstance(loc, list) and len(loc) >= 2:
                anchor_xy = (float(loc[0]), float(loc[1]))
        if anchor_xy is None:
            continue
        circle_rows.append(
            {
                "circle": circle,
                "center_key": center_point or str(circle.get("name", "")),
                "anchor_xy": anchor_xy,
            }
        )

    merged_groups: List[List[Dict[str, Any]]] = []
    for row in circle_rows:
        placed = False
        for group in merged_groups:
            if _point_distance(row["anchor_xy"], group[0]["anchor_xy"]) <= 5.0:
                group.append(row)
                placed = True
                break
        if not placed:
            merged_groups.append([row])

    for group in merged_groups:
        anchor_xy = (
            sum(float(row["anchor_xy"][0]) for row in group) / len(group),
            sum(float(row["anchor_xy"][1]) for row in group) / len(group),
        )
        excluded_center_names = {str(row["center_key"]) for row in group if str(row["center_key"])}

        radii = []
        for point in active_points:
            point_name = _point_name(point)
            xy = _point_xy_from_candidate(point)
            if xy is None or point_name in excluded_center_names:
                continue
            radii.append((_point_distance(anchor_xy, xy), point))
        if len(radii) < 2:
            continue
        radii.sort(key=lambda x: x[0])
        clusters: List[List[Any]] = []
        for radius, point in radii:
            placed = False
            for cluster in clusters:
                ref = sum(x[0] for x in cluster) / len(cluster)
                tol = max(2.5, ref * 0.03) if strict_circle_completion else max(3.5, ref * 0.045)
                if abs(radius - ref) <= tol:
                    cluster.append((radius, point))
                    placed = True
                    break
            if not placed:
                clusters.append([(radius, point)])

        min_cluster_size = 3 if strict_circle_completion else 2
        clusters = [c for c in clusters if len(c) >= min_cluster_size]
        if not clusters:
            continue
        clusters.sort(key=lambda c: (sum(x[0] for x in c) / len(c), -len(c)))
        circles_sorted = sorted(
            [row["circle"] for row in group],
            key=lambda c: (
                float(c.get("geometric_parameters", {}).get("radius_hint", 0.0)),
                str(c.get("name", "")),
            ),
        )
        if len(circles_sorted) == 1:
            chosen = max(
                clusters,
                key=lambda c: (
                    len(c),
                    -abs((sum(x[0] for x in c) / len(c)) - (sum(x[0] for x in radii) / len(radii))),
                ),
            )
            circle_expr = _circle_expr(circles_sorted[0])
            if circle_expr:
                for _, point in chosen:
                    point_name = _point_name(point)
                    if point_name:
                        forms.append(f"PointLiesOnCircle({point_name}, {circle_expr})")
            continue

        # For multiple circles with the same or nearly the same center, match circles to distinct radius clusters.
        cluster_order = sorted(range(len(clusters)), key=lambda i: sum(x[0] for x in clusters[i]) / len(clusters[i]))
        for idx, circle in enumerate(circles_sorted):
            cluster_idx = cluster_order[min(idx, len(cluster_order) - 1)]
            circle_expr = _circle_expr(circle)
            if circle_expr is None:
                continue
            for _, point in clusters[cluster_idx]:
                point_name = _point_name(point)
                if point_name:
                    forms.append(f"PointLiesOnCircle({point_name}, {circle_expr})")

    return _dedupe_keep_order(forms)


def _mutate_logic_forms(
    base_logic_forms: Sequence[str],
    candidates_by_id: Dict[str, Dict[str, Any]],
    active_ids: Sequence[str],
    rng,
    level: int,
) -> List[str]:
    forms = [str(x) for x in base_logic_forms]
    if level <= 0:
        return forms

    drop_p = min(0.07 * level, 0.55)
    forms = [x for x in forms if rng.random() > drop_p]

    return _dedupe_keep_order(forms)


def _sample_active_ids(
    all_candidates: Sequence[Dict[str, Any]],
    candidate_source: str,
    rng,
    level: int,
) -> List[str]:
    source_penalty = {
        "detector_only": 0.18,
        "detector_ocr_rules": 0.09,
        "detector_ocr_rules_vlm": 0.04,
    }.get(candidate_source, 0.1)

    active: List[str] = []
    for cand in all_candidates:
        cid = str(cand.get("id"))
        ctype = str(cand.get("type"))
        conf = safe_float(cand.get("confidence"), 0.5)
        if level == 0:
            active.append(cid)
            continue
        if ctype in {"point", "line", "circle"}:
            drop_rate = min(0.02 * level + (1.0 - conf) * 0.08 + source_penalty * 0.2, 0.35)
        else:
            drop_rate = min(0.05 * level + (1.0 - conf) * 0.12 + source_penalty * 0.4, 0.6)
        if rng.random() > drop_rate:
            active.append(cid)

    geometry_active = [cid for cid in active if cid.startswith(("pt_", "ln_", "cc_"))]
    if len(geometry_active) < 3:
        conf_sorted = sorted(
            [c for c in all_candidates if str(c.get("type")) in {"point", "line", "circle"}],
            key=lambda x: safe_float(x.get("confidence"), 0.5),
            reverse=True,
        )
        for cand in conf_sorted:
            cid = str(cand.get("id"))
            if cid not in active:
                active.append(cid)
            geometry_active = [x for x in active if x.startswith(("pt_", "ln_", "cc_"))]
            if len(geometry_active) >= 3:
                break

    return _dedupe_keep_order(active)


def _deterministic_clean_active_ids(
    all_candidates: Sequence[Dict[str, Any]],
    profile: str,
) -> List[str]:
    active: List[str] = []
    for cand in all_candidates:
        cid = str(cand.get("id"))
        ctype = str(cand.get("type"))
        conf = safe_float(cand.get("confidence"), 0.5)
        if ctype in {"point", "line", "circle"}:
            active.append(cid)
            continue
        geom = cand.get("geometric_parameters", {})
        text_class = str(geom.get("text_class", ""))
        sym_class = _canonical_symbol_class(geom.get("sym_class", ""))
        keep = True
        if profile == "clean55":
            keep = conf >= 0.55
        elif profile == "text60_sym50":
            if ctype == "text_label" or text_class:
                keep = conf >= 0.60
            else:
                keep = conf >= 0.50
        elif profile == "strict70":
            if sym_class in GROUPED_LINE_MARK_CLASSES | GROUPED_ANGLE_MARK_CLASSES | {"parallel", "bar", "angle", "perpendicular"}:
                keep = conf >= 0.55
            elif ctype == "text_label" or text_class:
                keep = conf >= 0.70
            else:
                keep = conf >= 0.65
        elif profile == "precision80":
            if sym_class in GROUPED_LINE_MARK_CLASSES | GROUPED_ANGLE_MARK_CLASSES | {"parallel", "bar", "angle", "perpendicular"}:
                keep = conf >= 0.60
            elif ctype == "text_label" or text_class:
                keep = conf >= 0.80
            else:
                keep = conf >= 0.72
        elif profile == "precision85_marks65":
            if sym_class in GROUPED_LINE_MARK_CLASSES | GROUPED_ANGLE_MARK_CLASSES | {"parallel", "bar", "angle", "perpendicular"}:
                keep = conf >= 0.65
            elif ctype == "text_label" or text_class:
                keep = conf >= 0.85
            else:
                keep = conf >= 0.75
        if keep:
            active.append(cid)
    return _dedupe_keep_order(active)


def _build_scene_candidates(
    sample_id: str,
    candidate_payload: Dict[str, Any],
    k: int,
    seed: int,
    angle_global_margin: float = 8.0,
    perpendicular_global_margin: float = 1e9,
    grouped_line_duplicate_penalty: float = 0.0,
    parallel_companion_weight: float = 0.0,
    len_text_mark_support_bonus: float = 0.0,
    line_host_alt_gap_max: float = 0.0,
    line_host_alt_topm: int = 1,
    pair_host_alt_gap_max: float = 0.0,
    pair_host_alt_topm: int = 1,
    alt_penalty_scale: float = 0.01,
    force_degree_arc_policy: Optional[str] = None,
    force_global_angle_mode: Optional[str] = None,
    force_line_completion_mode: Optional[str] = None,
    force_strict_circle_completion: str = "auto",
) -> List[Dict[str, Any]]:
    rng = deterministic_rng(sample_id, seed=seed, namespace="assembly")
    all_candidates = candidate_payload.get("candidates", [])
    all_ids = [str(c["id"]) for c in all_candidates if c.get("id")]
    candidates_by_id, _ = _candidate_maps(all_candidates)
    point_by_name = {
        str(c.get("name")): c
        for c in all_candidates
        if str(c.get("type")) == "point" and c.get("name")
    }
    conf_map = {str(c["id"]): safe_float(c.get("confidence"), 0.5) for c in all_candidates if c.get("id")}
    relation_cache = {
        ("default", "none"): _base_relations(candidate_payload, all_ids, degree_arc_policy="default", global_angle_search=False, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("selective_arc", "none"): _base_relations(candidate_payload, all_ids, degree_arc_policy="selective_arc", global_angle_search=False, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("conservative_arc", "none"): _base_relations(candidate_payload, all_ids, degree_arc_policy="conservative_arc", global_angle_search=False, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("strict_selective_arc", "none"): _base_relations(candidate_payload, all_ids, degree_arc_policy="strict_selective_arc", global_angle_search=False, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("clustered_selective_arc", "none"): _base_relations(candidate_payload, all_ids, degree_arc_policy="clustered_selective_arc", global_angle_search=False, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("default", "all"): _base_relations(candidate_payload, all_ids, degree_arc_policy="default", global_angle_search=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("selective_arc", "all"): _base_relations(candidate_payload, all_ids, degree_arc_policy="selective_arc", global_angle_search=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("conservative_arc", "all"): _base_relations(candidate_payload, all_ids, degree_arc_policy="conservative_arc", global_angle_search=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("clustered_selective_arc", "all"): _base_relations(candidate_payload, all_ids, degree_arc_policy="clustered_selective_arc", global_angle_search=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("default", "selective"): _base_relations(candidate_payload, all_ids, degree_arc_policy="default", global_angle_search=True, selective_global_angle=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("selective_arc", "selective"): _base_relations(candidate_payload, all_ids, degree_arc_policy="selective_arc", global_angle_search=True, selective_global_angle=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("conservative_arc", "selective"): _base_relations(candidate_payload, all_ids, degree_arc_policy="conservative_arc", global_angle_search=True, selective_global_angle=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
        ("clustered_selective_arc", "selective"): _base_relations(candidate_payload, all_ids, degree_arc_policy="clustered_selective_arc", global_angle_search=True, selective_global_angle=True, angle_global_margin=angle_global_margin, perpendicular_global_margin=perpendicular_global_margin, grouped_line_duplicate_penalty=grouped_line_duplicate_penalty, parallel_companion_weight=parallel_companion_weight, len_text_mark_support_bonus=len_text_mark_support_bonus),
    }
    geometry_candidate_count = len(_geometry_candidate_ids(all_candidates))
    symbol_candidate_count = len(_symbol_candidate_ids(all_candidates))
    source = candidate_payload.get("candidate_source", "detector_ocr_rules_vlm")
    sample_circle_count = sum(1 for c in all_candidates if str(c.get("type")) == "circle")
    sample_geometry_symbol_count = sum(1 for c in all_candidates if str(c.get("type")) == "geometry_symbol")
    degree_text_values = [
        _normalize_scalar_text(c.get("geometric_parameters", {}).get("text_content") or c.get("geometric_parameters", {}).get("text"))
        for c in all_candidates
        if str(c.get("geometric_parameters", {}).get("text_class", "")) == "degree"
    ]
    sample_degree_text_count = sum(1 for x in degree_text_values if x)
    sample_symbolic_degree_count = sum(
        1 for x in degree_text_values if x and not re.fullmatch(r"[0-9.]+", x)
    )
    multi_len_circle_sample = (
        sum(1 for c in all_candidates if str(c.get("geometric_parameters", {}).get("text_class", "")) == "len") >= 3
        and sum(1 for c in all_candidates if str(c.get("type")) == "circle") >= 4
    )

    scene_pool: List[Dict[str, Any]] = []
    target_pool = max(k + 24, k * 5)
    preset_specs = [
        ("all", "default", True, False, "default", "none"),
        ("all", "default", True, False, "strict", "none"),
        ("all", "default", True, False, "relaxed", "none"),
        ("all", "default", True, False, "default", "none_lenbackup"),
        ("all", "selective_arc", True, False, "default", "none"),
        ("all", "conservative_arc", True, False, "default", "none"),
        ("all", "clustered_selective_arc", True, False, "strict", "none"),
        ("all", "default", True, False, "default", "selective"),
        ("all", "selective_arc", True, False, "default", "selective"),
        ("all", "conservative_arc", True, False, "strict", "selective"),
        ("all", "default", True, False, "default", "all"),
        ("all", "selective_arc", True, False, "strict", "all"),
        ("all", "default", False, False, "default", "none"),
        ("all", "selective_arc", False, False, "strict", "none"),
        ("all", "default", True, True, "strict", "none"),
        ("all", "selective_arc", True, True, "strict", "none"),
        ("all", "clustered_selective_arc", True, True, "strict", "none"),
        ("clean55", "default", True, False, "default", "none"),
        ("clean55", "selective_arc", True, False, "strict", "none"),
        ("clean55", "conservative_arc", True, False, "strict", "none"),
        ("text60_sym50", "default", True, False, "default", "none"),
        ("strict70", "default", True, False, "strict", "none"),
        ("strict70", "default", False, False, "strict", "none"),
        ("strict70", "default", True, True, "strict", "none"),
        ("strict70", "selective_arc", True, True, "strict", "none"),
        ("strict70", "clustered_selective_arc", True, True, "strict", "none"),
        ("precision80", "default", True, False, "strict", "none"),
        ("precision80", "default", True, False, "default", "none_lenbackup"),
        ("precision80", "default", False, False, "strict", "none"),
        ("precision80", "default", True, False, "default", "all"),
        ("precision80", "default", True, True, "strict", "none"),
        ("precision80", "selective_arc", True, True, "strict", "none"),
        ("precision80", "clustered_selective_arc", True, True, "strict", "none"),
        ("precision80", "conservative_arc", True, True, "strict", "none"),
        ("precision85_marks65", "default", True, False, "strict", "none"),
        ("precision85_marks65", "default", True, False, "default", "none_lenbackup"),
        ("precision85_marks65", "default", False, False, "strict", "none"),
        ("precision85_marks65", "default", True, False, "default", "all"),
        ("precision85_marks65", "default", True, True, "strict", "none"),
        ("precision85_marks65", "selective_arc", True, True, "strict", "none"),
        ("precision85_marks65", "clustered_selective_arc", True, True, "strict", "none"),
        ("precision85_marks65", "selective_arc", True, True, "strict", "selective"),
        ("precision85_marks65", "default", True, True, "strict", "selective"),
        ("precision85_marks65", "conservative_arc", True, True, "strict", "selective"),
        ("precision85_marks65", "clustered_selective_arc", True, True, "strict", "selective"),
        ("precision85_marks65", "default", True, True, "relaxed", "selective"),
        ("precision80", "selective_arc", True, True, "strict", "selective"),
        ("precision80", "default", True, True, "strict", "selective"),
        ("precision80", "conservative_arc", True, True, "strict", "selective"),
        ("precision80", "clustered_selective_arc", True, True, "strict", "selective"),
        ("precision80", "default", True, True, "relaxed", "selective"),
        ("all", "clustered_selective_arc", True, False, "default", "all"),
        ("all", "conservative_arc", True, False, "default", "all"),
    ]
    for idx in range(target_pool):
        if idx < len(preset_specs):
            profile, degree_arc_policy, refined_line_naming, strict_circle_completion, line_completion_mode, global_angle_mode = preset_specs[idx]
            if profile == "all":
                active = list(all_ids)
                level = 0
            else:
                active = _deterministic_clean_active_ids(all_candidates, profile)
                level = 0
        else:
            degree_arc_policy = "default"
            refined_line_naming = True
            strict_circle_completion = False
            line_completion_mode = "default"
            global_angle_mode = "none"
            level = idx - len(preset_specs) + 1
            active = _sample_active_ids(all_candidates, source, rng, level)
        if force_degree_arc_policy is not None:
            degree_arc_policy = force_degree_arc_policy
        if force_global_angle_mode is not None:
            global_angle_mode = force_global_angle_mode
        if force_line_completion_mode is not None:
            line_completion_mode = force_line_completion_mode
        if force_strict_circle_completion == "on":
            strict_circle_completion = True
        elif force_strict_circle_completion == "off":
            strict_circle_completion = False
        active_set = set(active)
        coarse_length_text = False
        if global_angle_mode.endswith("_lenbackup"):
            coarse_length_text = True
            global_angle_mode = global_angle_mode.replace("_lenbackup", "")

        base_rel, base_symbol_ownership = relation_cache[(degree_arc_policy, global_angle_mode)]
        relations = [rel for rel in base_rel if _relation_is_closed(rel, active_set)]
        relations.sort(key=lambda rel: repr(rel))
        invalid_closure = len(base_rel) - len(relations)
        symbol_ownership = [
            row for row in base_symbol_ownership if row["symbol"] in active_set and all(x in active_set for x in row["targets"])
        ]
        symbol_ownership = _prune_conflicting_text_symbols(symbol_ownership, candidates_by_id, active_set)
        symbol_ownership_variants = _expand_symbol_ownership_variants(
            symbol_ownership,
            candidates_by_id,
            point_by_name,
            line_host_alt_gap_max=line_host_alt_gap_max,
            line_host_alt_topm=line_host_alt_topm,
        )
        expanded_variants: List[List[Dict[str, Any]]] = []
        for symbol_variant in symbol_ownership_variants:
            expanded_variants.extend(
                _expand_pair_symbol_ownership_variants(
                    symbol_variant,
                    pair_host_alt_gap_max=pair_host_alt_gap_max,
                    pair_host_alt_topm=pair_host_alt_topm,
                )
            )
        for alt_idx, symbol_variant in enumerate(expanded_variants):
            symbol_variant.sort(
                key=lambda row: (
                    str(row.get("symbol", "")),
                    tuple(str(x) for x in row.get("targets", [])),
                    str(row.get("reason", "")),
                )
            )
            logic_forms = _base_logic_forms(
                relations,
                symbol_variant,
                candidates_by_id,
                active_set,
                refined_line_naming=refined_line_naming,
                strict_circle_completion=strict_circle_completion,
                line_completion_mode=line_completion_mode,
                coarse_length_text=coarse_length_text,
            )
            logic_forms = _mutate_logic_forms(logic_forms, candidates_by_id, active, rng, level)
            continuous = _continuous_parameters(candidates_by_id, active)

            avg_conf = sum(conf_map.get(cid, 0.0) for cid in active) / max(len(active), 1)
            relation_coverage = len(relations) / max(len(base_rel), 1) if base_rel else 1.0
            symbol_support = len(symbol_variant) / max(len(base_symbol_ownership), 1) if base_symbol_ownership else 1.0
            closure_penalty = invalid_closure / max(len(base_rel), 1) if base_rel else 0.0
            alt_penalty = max(0.0, alt_penalty_scale) * alt_idx
            assembly_score = max(
                0.0,
                0.5 * avg_conf + 0.3 * relation_coverage + 0.2 * symbol_support - 0.03 * level - alt_penalty,
            )

            active_geometry = len([cid for cid in active if cid.startswith(("pt_", "ln_", "cc_"))])
            active_symbols = len([cid for cid in active if cid.startswith(("tx_", "sm_"))])
            extra_prims = max(0, active_geometry - geometry_candidate_count) + max(0, active_symbols - symbol_candidate_count)
            redundant_relations = max(0, len(relations) - len(logic_forms))

            scene_pool.append(
                {
                    "scene_id": f"scene_{idx:03d}_alt{alt_idx:02d}",
                    "discrete_structure": {
                        "active_primitives": sorted(active),
                        "relations": relations,
                        "symbol_ownership": symbol_variant,
                    },
                    "continuous_parameters": continuous,
                    "projected_logic_forms": logic_forms,
                    "assembly": {
                        "avg_candidate_conf": round(avg_conf, 6),
                        "relation_coverage": round(relation_coverage, 6),
                        "symbol_support": round(symbol_support, 6),
                        "closure_penalty": round(closure_penalty, 6),
                        "invalid_closure_count": invalid_closure,
                        "assembly_score": round(assembly_score, 6),
                        "level": level,
                        "alt_variant_index": alt_idx,
                        "degree_arc_policy": degree_arc_policy,
                        "refined_line_naming": refined_line_naming,
                        "strict_circle_completion": strict_circle_completion,
                        "line_completion_mode": line_completion_mode,
                        "coarse_length_text": coarse_length_text,
                        "sample_circle_count": sample_circle_count,
                        "sample_geometry_symbol_count": sample_geometry_symbol_count,
                        "sample_degree_text_count": sample_degree_text_count,
                        "sample_symbolic_degree_count": sample_symbolic_degree_count,
                        "global_angle_search": global_angle_mode != "none",
                        "global_angle_mode": global_angle_mode,
                        "force_degree_arc_policy": force_degree_arc_policy,
                        "force_global_angle_mode": force_global_angle_mode,
                        "force_line_completion_mode": force_line_completion_mode,
                        "force_strict_circle_completion": force_strict_circle_completion,
                    },
                    "min_explanation": {
                        "extra_primitives": extra_prims,
                        "redundant_relations": redundant_relations,
                    },
                }
            )

    scene_pool.sort(
        key=lambda x: (
            -float(x["assembly"]["assembly_score"]),
            tuple(str(cid) for cid in x["discrete_structure"].get("active_primitives", [])),
            tuple(str(rel) for rel in x["discrete_structure"].get("relations", [])),
            tuple(str(form) for form in sorted(x.get("projected_logic_forms", []))),
            str(x.get("scene_id", "")),
        )
    )

    unique_pool: List[Dict[str, Any]] = []
    seen_logic = set()
    for scene in scene_pool:
        logic_key = tuple(sorted(str(x) for x in scene.get("projected_logic_forms", [])))
        if logic_key in seen_logic:
            continue
        seen_logic.add(logic_key)
        unique_pool.append(scene)
        if len(unique_pool) >= k:
            break
    return unique_pool


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    out_dir = stage_dir("scenes", args.variant, args.split, args.outputs_dir)
    _ = out_dir

    written = 0
    for sample_id in split_ids:
        try:
            from egsr_common import read_stage_sample

            candidate_payload = read_stage_sample("candidates", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        scenes = _build_scene_candidates(
            sample_id=sample_id,
            candidate_payload=candidate_payload,
            k=args.k,
            seed=args.seed,
            angle_global_margin=args.angle_global_margin,
            perpendicular_global_margin=args.perpendicular_global_margin,
            grouped_line_duplicate_penalty=args.grouped_line_duplicate_penalty,
            parallel_companion_weight=args.parallel_companion_weight,
            len_text_mark_support_bonus=args.len_text_mark_support_bonus,
            line_host_alt_gap_max=args.line_host_alt_gap_max,
            line_host_alt_topm=args.line_host_alt_topm,
            pair_host_alt_gap_max=args.pair_host_alt_gap_max,
            pair_host_alt_topm=args.pair_host_alt_topm,
            alt_penalty_scale=args.alt_penalty_scale,
            force_degree_arc_policy=args.force_degree_arc_policy,
            force_global_angle_mode=args.force_global_angle_mode,
            force_line_completion_mode=args.force_line_completion_mode,
            force_strict_circle_completion=args.force_strict_circle_completion,
        )

        payload = {
            "sample_id": sample_id,
            "variant": args.variant,
            "split": args.split,
            "k": args.k,
            "candidate_source": candidate_payload.get("candidate_source"),
            "scenes": scenes,
            "stats": {
                "num_scenes": len(scenes),
                "max_k": args.k,
            },
        }
        write_stage_sample("scenes", args.variant, args.split, sample_id, payload, args.outputs_dir)
        written += 1

    elapsed = time.perf_counter() - begin
    write_run_meta(
        "scenes",
        args.variant,
        args.split,
        {
            "k": args.k,
            "seed": args.seed,
            "num_requested": len(split_ids),
            "num_written": written,
            "elapsed_sec": round(elapsed, 3),
            "assembly_mode": "candidate_geometry_plus_attachment_heuristics",
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
        },
        args.outputs_dir,
    )
    print(
        f"[assemble_scenes] split={args.split} variant={args.variant} "
        f"k={args.k} written={written} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()

