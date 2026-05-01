"""
ORACLE STATUS: NO-ORACLE
DESCRIPTION: Build sanitized no-oracle candidate pools from frozen prediction-side artifacts.
INPUTS:
  - frozen candidate artifacts
OUTPUTS: candidates/{variant}/{split}/*.json
GT USAGE: None
"""

from __future__ import annotations

import argparse
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    read_json,
    read_split_ids,
    slugify,
    write_run_meta,
    write_stage_sample,
)


FORBIDDEN_KEYS = {
    "annotation_id",
    "canonical_name",
    "annotation_endpoints",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sanitized no-oracle EGSR candidates.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="EGSR-Core Clean")
    parser.add_argument(
        "--source-variant",
        default="detector_plus_ocr_rules_plus_light_vlm",
        help="Frozen clean candidate artifact variant to sanitize.",
    )
    parser.add_argument("--text-semantic-bonus", type=float, default=0.0)
    parser.add_argument("--structural-symbol-bonus", type=float, default=0.0)
    parser.add_argument("--invalid-text-penalty", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _clean_geom_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if key in FORBIDDEN_KEYS:
            continue
        out[key] = value
    return out


def _clean_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {
        "id": candidate.get("id"),
        "type": candidate.get("type"),
        "name": candidate.get("name"),
        "location": candidate.get("location", []),
        "geometric_parameters": _clean_geom_params(candidate.get("geometric_parameters", {})),
        "confidence": candidate.get("confidence", 0.5),
        "source": candidate.get("source", "frozen_artifact"),
    }
    return cleaned


STRUCTURAL_SYMBOL_CLASSES = {
    "perpendicular",
    "right_angle",
    "parallel",
    "double_parallel",
    "bar",
    "double_bar",
    "triple_bar",
    "angle",
    "double_angle",
    "triple_angle",
    "quad_angle",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "", text)
    return text


def _is_valid_point_text(candidate: Dict[str, Any]) -> bool:
    geom = candidate.get("geometric_parameters", {}) or {}
    text = _normalize_text(geom.get("text") or geom.get("text_content") or candidate.get("name"))
    if not text:
        return False
    return bool(re.fullmatch(r"[A-Z]", text))


def _is_valid_measure_text(candidate: Dict[str, Any]) -> bool:
    geom = candidate.get("geometric_parameters", {}) or {}
    text = _normalize_text(geom.get("text_content") or geom.get("text"))
    if not text:
        return False
    if "," in text:
        return False
    return True


def _recalibrate_confidence(
    candidate: Dict[str, Any],
    text_semantic_bonus: float,
    structural_symbol_bonus: float,
    invalid_text_penalty: float,
) -> None:
    conf = float(candidate.get("confidence", 0.5))
    geom = candidate.get("geometric_parameters", {}) or {}
    ctype = str(candidate.get("type", ""))
    text_class = str(geom.get("text_class", "")).lower()
    sym_class = str(geom.get("sym_class", "")).lower()

    delta = 0.0
    if ctype == "text_label" and text_class == "point":
        delta += text_semantic_bonus if _is_valid_point_text(candidate) else -invalid_text_penalty
    elif ctype == "geometry_symbol" and sym_class == "text" and text_class in {"len", "degree", "angle", "point"}:
        delta += text_semantic_bonus if _is_valid_measure_text(candidate) else -invalid_text_penalty

    if ctype == "geometry_symbol" and sym_class in STRUCTURAL_SYMBOL_CLASSES:
        delta += structural_symbol_bonus

    candidate["confidence"] = round(max(0.0, min(1.0, conf + delta)), 6)


def _point_xy(candidate: Dict[str, Any]) -> Any:
    loc = candidate.get("location", [])
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def _enrich_circle_candidates(candidates: List[Dict[str, Any]]) -> None:
    points = [c for c in candidates if c.get("type") == "point"]
    circles = [c for c in candidates if c.get("type") == "circle"]
    point_names = []
    for p in points:
        xy = _point_xy(p)
        if xy is not None:
            point_names.append((str(p.get("name")), xy))

    used: Dict[str, int] = {}
    for circle in circles:
        loc = circle.get("location", [])
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        cx, cy = float(loc[0]), float(loc[1])
        nearest_name = None
        nearest_dist = 1e9
        for pname, (px, py) in point_names:
            dist = math.hypot(cx - px, cy - py)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_name = pname
        if nearest_name is None:
            continue
        idx = used.get(nearest_name, 0)
        used[nearest_name] = idx + 1
        circle_name = nearest_name if idx == 0 else f"{nearest_name}{idx}"
        circle["name"] = circle_name
        geom = circle.setdefault("geometric_parameters", {})
        geom["center_point"] = nearest_name
        geom["radius_token"] = f"radius_{idx}_0"


def _source_payload(outputs_dir: Path, source_variant: str, split: str, sample_id: str) -> Dict[str, Any]:
    src = outputs_dir / "candidates" / slugify(source_variant) / split / f"{sample_id}.json"
    if not src.exists():
        raise FileNotFoundError(
            f"Missing clean candidate source artifact for split='{split}', variant='{source_variant}', sample_id='{sample_id}': {src}. "
            "The clean pipeline only replays from existing non-oracle candidate artifacts. "
            "Populate the source split first before using it as a clean development/evaluation split."
        )
    return read_json(src)


def build_candidates_no_oracle(
    split: str,
    variant: str,
    source_variant: str,
    data_dir: Path,
    outputs_dir: Path,
    limit: int = 0,
    text_semantic_bonus: float = 0.0,
    structural_symbol_bonus: float = 0.0,
    invalid_text_penalty: float = 0.0,
) -> int:
    split_ids = read_split_ids(split, data_dir)
    if limit > 0:
        split_ids = split_ids[:limit]

    written = 0
    for sample_id in split_ids:
        payload = _source_payload(outputs_dir, source_variant, split, sample_id)
        cleaned_candidates: List[Dict[str, Any]] = [_clean_candidate(c) for c in payload.get("candidates", [])]
        for cand in cleaned_candidates:
            _recalibrate_confidence(
                cand,
                text_semantic_bonus=text_semantic_bonus,
                structural_symbol_bonus=structural_symbol_bonus,
                invalid_text_penalty=invalid_text_penalty,
            )
        _enrich_circle_candidates(cleaned_candidates)
        clean_payload = {
            "sample_id": sample_id,
            "file_name": payload.get("file_name", f"{sample_id}.png"),
            "width": payload.get("width"),
            "height": payload.get("height"),
            "candidate_source": payload.get("candidate_source", "frozen_artifact"),
            "candidates": cleaned_candidates,
            "relations": {"geo2geo": [], "sym2sym": [], "sym2geo": []},
            "candidate_build_notes": {
                "candidate_origin": "frozen_prediction_artifact",
                "source_variant": source_variant,
                "relations_removed": True,
                "annotation_metadata_removed": True,
                "text_semantic_bonus": text_semantic_bonus,
                "structural_symbol_bonus": structural_symbol_bonus,
                "invalid_text_penalty": invalid_text_penalty,
            },
            "stats": {
                "num_candidates": len(cleaned_candidates),
                "num_points": sum(1 for c in cleaned_candidates if c.get("type") == "point"),
                "num_lines": sum(1 for c in cleaned_candidates if c.get("type") == "line"),
                "num_circles": sum(1 for c in cleaned_candidates if c.get("type") == "circle"),
                "num_symbols": sum(
                    1
                    for c in cleaned_candidates
                    if c.get("type") in {"text_label", "geometry_symbol"}
                ),
            },
        }
        write_stage_sample("candidates", variant, split, sample_id, clean_payload, outputs_dir)
        written += 1
    return written


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()
    written = build_candidates_no_oracle(
        split=args.split,
        variant=args.variant,
        source_variant=args.source_variant,
        data_dir=args.data_dir,
        outputs_dir=args.outputs_dir,
        limit=args.limit,
        text_semantic_bonus=args.text_semantic_bonus,
        structural_symbol_bonus=args.structural_symbol_bonus,
        invalid_text_penalty=args.invalid_text_penalty,
    )
    elapsed = time.perf_counter() - begin
    write_run_meta(
        "candidates",
        args.variant,
        args.split,
        {
            "source_variant": args.source_variant,
            "num_written": written,
            "num_requested": written,
            "elapsed_sec": round(elapsed, 3),
            "sanitized_relations_removed": True,
            "sanitized_annotation_metadata_removed": True,
            "text_semantic_bonus": args.text_semantic_bonus,
            "structural_symbol_bonus": args.structural_symbol_bonus,
            "invalid_text_penalty": args.invalid_text_penalty,
        },
        args.outputs_dir,
    )
    print(
        f"[build_candidates_clean] split={args.split} variant={args.variant} "
        f"source_variant={args.source_variant} written={written} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
