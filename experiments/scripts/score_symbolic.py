"""
Phase 4 (part B): compute symbolic validity scores.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 计算场景的符号有效性分数
INPUTS: 场景假设
OUTPUTS: 符号有效性分数
GT USAGE: None - 仅检查符号引用的完整性和一致性
PURPOSE: 核心推理模块 - 三重一致性评分的符号部分
LAST UPDATED: 2026-04-28

✅ 此脚本不访问测试集标注，可用于正式推理。
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Set

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


PRED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\(.+\)$")
TOKEN_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*\b")
LEN_LITERAL_RE = re.compile(
    r"Equals\(\s*LengthOf\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*,\s*([A-Za-z0-9_.+\-*/=]+)\s*\)"
)
ANGLE_LITERAL_RE = re.compile(
    r"Equals\(\s*MeasureOf\(\s*Angle\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*,\s*([A-Za-z0-9_.+\-*/=]+)\s*\)"
)
PAR_RE = re.compile(
    r"Parallel\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*,\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)"
)
PERP_RE = re.compile(
    r"Perpendicular\(\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*,\s*Line\(\s*([A-Za-z0-9_]+)\s*,\s*([A-Za-z0-9_]+)\s*\)\s*\)"
)
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

LENGTH_MARK_CLASSES = {"bar", "double_bar", "triple_bar"}
ANGLE_MARK_CLASSES = {"angle", "double_angle", "triple_angle", "quad_angle"}
PARALLEL_MARK_CLASSES = {"parallel", "double_parallel"}
PERP_MARK_CLASSES = {"perpendicular", "right_angle"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute EGSR symbolic scores.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--mu1", type=float, default=default_weights()["mu1"])
    parser.add_argument("--mu2", type=float, default=default_weights()["mu2"])
    parser.add_argument("--w-consistency", type=float, default=0.35)
    parser.add_argument("--w-binding", type=float, default=0.35)
    parser.add_argument("--w-support", type=float, default=0.30)
    parser.add_argument(
        "--support-evidence-mode",
        choices=("count", "conf_sqrt"),
        default="count",
        help="How to convert active symbolic evidence into family support requirements.",
    )
    parser.add_argument("--support-length-weight", type=float, default=1.0)
    parser.add_argument("--support-angle-weight", type=float, default=1.0)
    parser.add_argument("--support-parallel-weight", type=float, default=1.0)
    parser.add_argument("--support-perpendicular-weight", type=float, default=1.0)
    parser.add_argument("--duplicate-binding-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _schema_score(forms: List[str]) -> float:
    if not forms:
        return 0.0
    valid = 0
    for form in forms:
        no_space = re.sub(r"\s+", "", str(form))
        if PRED_RE.match(no_space):
            valid += 1
    return valid / len(forms)


def _extract_symbolic_refs(forms: List[str]) -> Set[str]:
    refs: Set[str] = set()
    for form in forms:
        for token in TOKEN_RE.findall(str(form)):
            if token in KEYWORDS:
                continue
            refs.add(token)
    return refs


def _entity_registry(scene: Dict[str, Any]) -> Set[str]:
    registry: Set[str] = set()
    continuous = scene.get("continuous_parameters", {})
    registry.update(map(str, continuous.get("point_coordinates", {}).keys()))
    registry.update(map(str, continuous.get("line_parameters", {}).keys()))
    registry.update(map(str, continuous.get("circle_parameters", {}).keys()))

    for cid in scene.get("discrete_structure", {}).get("active_primitives", []):
        if not isinstance(cid, str):
            continue
        if cid.startswith(("pt_", "ln_", "cc_")):
            registry.add(cid[3:])
    return registry


def _candidate_by_id(candidate_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cand in candidate_payload.get("candidates", []):
        cid = str(cand.get("id", ""))
        if cid:
            out[cid] = cand
    return out


def _symbol_family(candidate: Dict[str, Any]) -> str | None:
    ctype = str(candidate.get("type", ""))
    geom = candidate.get("geometric_parameters", {}) or {}
    if ctype == "text_label":
        text_class = str(geom.get("text_class", "")).lower()
        if text_class == "point":
            return None
        if text_class in {"len", "length"}:
            return "length"
        if text_class in {"angle", "degree", "arc"}:
            return "angle"
        return "text"
    if ctype != "geometry_symbol":
        return None
    sym_class = str(geom.get("sym_class", "")).lower()
    text_class = str(geom.get("text_class", "")).lower()
    if sym_class == "text":
        if text_class in {"len", "length"}:
            return "length"
        if text_class in {"angle", "degree", "arc"}:
            return "angle"
        return "text"
    if sym_class in LENGTH_MARK_CLASSES:
        return "length"
    if sym_class in ANGLE_MARK_CLASSES:
        return "angle"
    if sym_class in PARALLEL_MARK_CLASSES:
        return "parallel"
    if sym_class in PERP_MARK_CLASSES:
        return "perpendicular"
    return None


def _is_literal_or_variable(token: str) -> bool:
    if token.islower():
        return True
    if re.fullmatch(r"-?\d+(\.\d+)?", token):
        return True
    if token.startswith("noise_"):
        return True
    return False


def _support_requirement(
    family: str,
    evidence_count: int,
    confidence_sum: float,
    mode: str,
) -> float:
    if evidence_count <= 0:
        return 0.0
    if mode == "conf_sqrt":
        return max(1.0, math.sqrt(max(confidence_sum, 0.0)))
    return max(1.0, math.sqrt(float(evidence_count)))


def _family_weight_map(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "length": max(args.support_length_weight, 0.0),
        "angle": max(args.support_angle_weight, 0.0),
        "parallel": max(args.support_parallel_weight, 0.0),
        "perpendicular": max(args.support_perpendicular_weight, 0.0),
    }


def _symbol_support_score(
    scene: Dict[str, Any],
    candidate_payload: Dict[str, Any],
    args: argparse.Namespace,
) -> float:
    forms = [re.sub(r"\s+", "", str(x)) for x in scene.get("projected_logic_forms", [])]
    if not forms:
        return 0.0

    family_forms = {
        "length": 0,
        "angle": 0,
        "parallel": 0,
        "perpendicular": 0,
    }
    for form in forms:
        if "LengthOf(Line(" in form:
            family_forms["length"] += 1
        if "MeasureOf(Angle(" in form or "MeasureOf(Arc(" in form:
            family_forms["angle"] += 1
        if form.startswith("Parallel("):
            family_forms["parallel"] += 1
        if form.startswith("Perpendicular("):
            family_forms["perpendicular"] += 1

    by_id = _candidate_by_id(candidate_payload)
    active = set(scene.get("discrete_structure", {}).get("active_primitives", []))
    family_evidence = {
        "length": 0,
        "angle": 0,
        "parallel": 0,
        "perpendicular": 0,
    }
    family_confidence = {
        "length": 0.0,
        "angle": 0.0,
        "parallel": 0.0,
        "perpendicular": 0.0,
    }
    for cid in active:
        cand = by_id.get(str(cid))
        if not cand:
            continue
        family = _symbol_family(cand)
        if family in family_evidence:
            family_evidence[family] += 1
            family_confidence[family] += float(cand.get("confidence", 0.5))

    family_weights = _family_weight_map(args)

    weighted_scores: List[float] = []
    total_weight = 0.0
    for family, evidence_count in family_evidence.items():
        if evidence_count <= 0:
            continue
        family_weight = family_weights.get(family, 1.0)
        if family_weight <= 0:
            continue
        required_count = _support_requirement(
            family=family,
            evidence_count=evidence_count,
            confidence_sum=family_confidence.get(family, 0.0),
            mode=args.support_evidence_mode,
        )
        total_weight += family_weight * required_count
        support = min(family_forms.get(family, 0), required_count) / required_count
        weighted_scores.append(family_weight * required_count * support)

    if total_weight <= 0:
        return 1.0
    return sum(weighted_scores) / total_weight


def _consistency_score(scene: Dict[str, Any]) -> float:
    forms = [str(x) for x in scene.get("projected_logic_forms", [])]
    refs = _extract_symbolic_refs(forms)
    registry = _entity_registry(scene)
    unknown = [x for x in refs if x not in registry and not _is_literal_or_variable(x)]
    if not refs:
        return 0.0
    return 1.0 - len(unknown) / len(refs)


def _binding_coherence_score(scene: Dict[str, Any], duplicate_binding_penalty: float = 0.0) -> float:
    forms = [re.sub(r"\s+", "", str(x)) for x in scene.get("projected_logic_forms", [])]
    if not forms:
        return 0.0

    line_literals: Dict[tuple, set] = {}
    angle_literals: Dict[tuple, set] = {}
    par_pairs = set()
    perp_pairs = set()
    line_literal_counts: Dict[tuple, int] = {}
    angle_literal_counts: Dict[tuple, int] = {}
    par_pair_counts: Dict[tuple, int] = {}
    perp_pair_counts: Dict[tuple, int] = {}
    endpoint_point_on_line = 0
    checks = 0

    for form in forms:
        m = LEN_LITERAL_RE.fullmatch(form)
        if m:
            a, b, token = m.groups()
            key = tuple(sorted((a, b)))
            line_literals.setdefault(key, set()).add(token)
            line_literal_counts[key] = line_literal_counts.get(key, 0) + 1
            checks += 1
            continue
        m = ANGLE_LITERAL_RE.fullmatch(form)
        if m:
            a, b, c, token = m.groups()
            key = (min(a, c), b, max(a, c))
            angle_literals.setdefault(key, set()).add(token)
            angle_literal_counts[key] = angle_literal_counts.get(key, 0) + 1
            checks += 1
            continue
        m = PAR_RE.fullmatch(form)
        if m:
            a, b, c, d = m.groups()
            left = "".join(sorted((a, b)))
            right = "".join(sorted((c, d)))
            key = tuple(sorted((left, right)))
            par_pairs.add(key)
            par_pair_counts[key] = par_pair_counts.get(key, 0) + 1
            checks += 1
            continue
        m = PERP_RE.fullmatch(form)
        if m:
            a, b, c, d = m.groups()
            left = "".join(sorted((a, b)))
            right = "".join(sorted((c, d)))
            key = tuple(sorted((left, right)))
            perp_pairs.add(key)
            perp_pair_counts[key] = perp_pair_counts.get(key, 0) + 1
            checks += 1
            continue
        if form.startswith("PointLiesOnLine("):
            inner = form[len("PointLiesOnLine("):-1]
            try:
                point_name, rest = inner.split(",Line(", 1)
                line_args = rest[:-1]
                l1, l2 = line_args.split(",", 1)
                if point_name in {l1, l2}:
                    endpoint_point_on_line += 1
                checks += 1
            except ValueError:
                pass

    contradictions = 0
    contradictions += sum(max(len(vals) - 1, 0) for vals in line_literals.values())
    contradictions += sum(max(len(vals) - 1, 0) for vals in angle_literals.values())
    contradictions += len(par_pairs & perp_pairs)
    contradictions += endpoint_point_on_line
    if duplicate_binding_penalty > 0.0:
        contradictions += duplicate_binding_penalty * sum(max(v - 1, 0) for v in line_literal_counts.values())
        contradictions += duplicate_binding_penalty * sum(max(v - 1, 0) for v in angle_literal_counts.values())
        contradictions += duplicate_binding_penalty * sum(max(v - 1, 0) for v in par_pair_counts.values())
        contradictions += duplicate_binding_penalty * sum(max(v - 1, 0) for v in perp_pair_counts.values())

    return max(0.0, 1.0 - contradictions / max(checks, 1))


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    norm = args.mu1 + args.mu2
    if norm <= 0:
        args.mu1, args.mu2 = 0.5, 0.5
    else:
        args.mu1 /= norm
        args.mu2 /= norm

    validity_norm = args.w_consistency + args.w_binding + args.w_support
    if validity_norm <= 0:
        args.w_consistency, args.w_binding, args.w_support = 0.35, 0.35, 0.30
    else:
        args.w_consistency /= validity_norm
        args.w_binding /= validity_norm
        args.w_support /= validity_norm

    written = 0
    avg_sym = []

    for sample_id in split_ids:
        try:
            scene_payload = read_stage_sample("scenes", args.variant, args.split, sample_id, args.outputs_dir)
            candidate_payload = read_stage_sample("candidates", args.variant, args.split, sample_id, args.outputs_dir)
        except FileNotFoundError:
            continue

        scene_scores = []
        for scene in scene_payload.get("scenes", []):
            forms = [str(x) for x in scene.get("projected_logic_forms", [])]
            s_schema = _schema_score(forms)
            s_consistency = _consistency_score(scene)
            s_binding = _binding_coherence_score(scene, duplicate_binding_penalty=args.duplicate_binding_penalty)
            s_support = _symbol_support_score(scene, candidate_payload, args)
            s_validity = (
                args.w_consistency * s_consistency
                + args.w_binding * s_binding
                + args.w_support * s_support
            )
            s_sym = args.mu1 * s_schema + args.mu2 * s_validity
            scene_scores.append(
                {
                    "scene_id": scene.get("scene_id"),
                    "s_schema": round(s_schema, 6),
                    "s_consistency": round(s_consistency, 6),
                    "s_binding": round(s_binding, 6),
                    "s_support": round(s_support, 6),
                    "s_sym": round(s_sym, 6),
                }
            )

        sample_ss = sum(x["s_sym"] for x in scene_scores) / max(len(scene_scores), 1)
        avg_sym.append(sample_ss)

        score_file = args.outputs_dir / "scores" / slugify(args.variant) / args.split / f"{sample_id}.json"
        if score_file.exists():
            score_payload = read_json(score_file)
        else:
            score_payload = {"sample_id": sample_id, "variant": args.variant, "split": args.split}
        score_payload["symbolic_scoring"] = {
                "weights": {
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
                },
            "symbolic_validity": round(sample_ss, 6),
            "scenes": scene_scores,
        }
        write_json(score_file, score_payload)
        written += 1

    elapsed = time.perf_counter() - begin
    print(
        f"[score_symbolic] split={args.split} variant={args.variant} "
        f"written={written} mean_sym={sum(avg_sym)/max(len(avg_sym),1):.4f} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
