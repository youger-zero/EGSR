"""
Shared utilities for the EGSR experiment pipeline.

ORACLE STATUS: UTILITY
DESCRIPTION: 通用工具函数库
FUNCTIONS: 包含 load_ground_truth, load_annotations 等函数
GT USAGE: 提供 GT 访问接口，但不直接使用
PURPOSE: 工具库 - 被其他脚本调用
LAST UPDATED: 2026-04-28

⚠️ 注意：此文件包含 load_ground_truth 和 load_annotations 函数。
✅ 这些函数本身是中性的，关键在于调用它们的脚本如何使用。
✅ NO-ORACLE 脚本不应调用这些函数。
✅ ORACLE-DIAG 脚本可以调用这些函数进行诊断。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = SCRIPT_DIR.parent
PROJECT_DIR = EXPERIMENTS_DIR.parent
DATA_DIR = EXPERIMENTS_DIR / "data" / "PGDP5K"
OUTPUTS_DIR = EXPERIMENTS_DIR / "outputs"


def slugify(text: str) -> str:
    """Convert free-form experiment names into stable path-safe ids."""
    lowered = text.strip().lower()
    lowered = lowered.replace("+", "_plus_")
    lowered = lowered.replace("/", "_")
    lowered = lowered.replace("w/o", "wo")
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = re.sub(r"_+", "_", lowered).strip("_")
    return lowered or "default"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split_ids(split: str, data_dir: Path = DATA_DIR) -> List[str]:
    split_path = data_dir / "splits" / f"{split}.txt"
    ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()]
    return [sid for sid in ids if sid]


def load_annotations(split: str, data_dir: Path = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    return read_json(data_dir / "annotations" / f"{split}.json")


def load_ground_truth(data_dir: Path = DATA_DIR) -> Dict[str, Dict[str, Any]]:
    return read_json(data_dir / "our_diagram_logic_forms_annot.json")


def split_line_instance(line_name: str, point_names: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Split a line instance name (e.g. AB) into two known point names."""
    if not line_name:
        return None
    # Longest-match first in case point names have more than one char.
    candidates = sorted(point_names, key=lambda x: len(x), reverse=True)
    for left in candidates:
        if not line_name.startswith(left):
            continue
        right = line_name[len(left) :]
        if right and right in point_names:
            return left, right
    return None


def normalize_logic_form(expr: str) -> str:
    return re.sub(r"\s+", "", str(expr))


def logic_set(graph: Dict[str, Any]) -> set:
    forms = graph.get("diagram_logic_forms", [])
    return {normalize_logic_form(x) for x in forms if str(x).strip()}


def prf1_from_sets(gt_set: set, pred_set: set) -> Tuple[float, float, float]:
    if not pred_set and not gt_set:
        return 1.0, 1.0, 1.0
    overlap = len(gt_set & pred_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(gt_set) if gt_set else 1.0
    denom = precision + recall
    f1 = 0.0 if denom == 0 else (2.0 * precision * recall / denom)
    return precision, recall, f1


def deterministic_rng(sample_id: str, seed: int = 7, namespace: str = "egsr") -> random.Random:
    mix = f"{namespace}:{sample_id}:{seed}".encode("utf-8")
    digest = hashlib.sha256(mix).hexdigest()
    value = int(digest[:16], 16)
    return random.Random(value)


def stage_dir(stage: str, variant: str, split: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return ensure_dir(outputs_dir / stage / slugify(variant) / split)


def run_meta_path(stage: str, variant: str, split: str, outputs_dir: Path = OUTPUTS_DIR) -> Path:
    return outputs_dir / stage / slugify(variant) / split / "_run_meta.json"


def write_run_meta(stage: str, variant: str, split: str, meta: Dict[str, Any], outputs_dir: Path = OUTPUTS_DIR) -> None:
    payload = dict(meta)
    payload["stage"] = stage
    payload["variant"] = variant
    payload["variant_slug"] = slugify(variant)
    payload["split"] = split
    write_json(run_meta_path(stage, variant, split, outputs_dir), payload)


def read_stage_sample(stage: str, variant: str, split: str, sample_id: str, outputs_dir: Path = OUTPUTS_DIR) -> Dict[str, Any]:
    return read_json(outputs_dir / stage / slugify(variant) / split / f"{sample_id}.json")


def write_stage_sample(
    stage: str,
    variant: str,
    split: str,
    sample_id: str,
    payload: Dict[str, Any],
    outputs_dir: Path = OUTPUTS_DIR,
) -> None:
    write_json(outputs_dir / stage / slugify(variant) / split / f"{sample_id}.json", payload)


def list_stage_sample_ids(stage: str, variant: str, split: str, outputs_dir: Path = OUTPUTS_DIR) -> List[str]:
    base = outputs_dir / stage / slugify(variant) / split
    if not base.exists():
        return []
    out: List[str] = []
    for path in sorted(base.glob("*.json")):
        if path.name.startswith("_"):
            continue
        out.append(path.stem)
    return out


def exp_score(residual: float, tau: float) -> float:
    tau_safe = max(tau, 1e-6)
    return math.exp(-residual / tau_safe)


@dataclass
class OfficialEvalHandle:
    module: Any
    metric_key: str = "logic_forms_all"


def load_official_eval_module(pgdproot: Path) -> OfficialEvalHandle:
    """
    Load the official PGDP evaluation module.

    Notes:
    - The vendor evaluator can leave ``sys.stdout`` as ``None`` when latex parsing fails.
      We heal this after each call in wrapper functions.
    """

    eval_dir = pgdproot / "InterGPS" / "diagram_parser" / "evaluation_new"
    solver_dir = pgdproot / "InterGPS" / "symbolic_solver"
    sys.path.insert(0, str(eval_dir))
    sys.path.insert(0, str(solver_dir))
    import calc_diagram_accuracy as cda  # type: ignore

    return OfficialEvalHandle(module=cda)


def restore_stdout_if_needed() -> None:
    if sys.stdout is None and getattr(sys, "__stdout__", None) is not None:
        sys.stdout = sys.__stdout__


def official_single_sample_scores(
    handle: OfficialEvalHandle,
    gt_graph: Optional[Dict[str, Any]],
    pred_graph: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    cda = handle.module
    accuracy, recall, _ = cda.diagram_evaluaion(gt_graph, pred_graph)
    restore_stdout_if_needed()
    metric_key = handle.metric_key
    acc = float(accuracy[metric_key])
    rec = float(recall[metric_key])
    f1 = float(cda.calc_f1(acc, rec))
    return {"accuracy": acc, "recall": rec, "f1": f1}


def metric_tag_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    f1 = scores["f1"]
    rec = scores["recall"]
    return {
        "TS": int(abs(f1 - 1.0) < 1e-12),
        "PR": int(abs(rec - 1.0) < 1e-12),
        "AS": int(f1 >= 0.75),
        "LS": int(f1 >= 0.5),
    }


def summarize_metric_tags(tags: Iterable[Dict[str, int]]) -> Dict[str, float]:
    rows = list(tags)
    n = max(len(rows), 1)
    return {
        "LS": 100.0 * sum(x["LS"] for x in rows) / n,
        "AS": 100.0 * sum(x["AS"] for x in rows) / n,
        "PR": 100.0 * sum(x["PR"] for x in rows) / n,
        "TS": 100.0 * sum(x["TS"] for x in rows) / n,
        "count": len(rows),
    }


def default_weights() -> Dict[str, float]:
    return {
        "alpha": 0.35,
        "beta": 0.35,
        "gamma": 0.30,
        "lambda": 0.25,
        "tau": 0.35,
        "mu1": 0.25,
        "mu2": 0.75,
        "eta1": 0.6,
        "eta2": 0.4,
        "w1": 0.4,
        "w2": 0.3,
        "w3": 0.3,
    }


def consistency_mode_weights(mode: str, base: Dict[str, float]) -> Dict[str, float]:
    out = dict(base)
    mode = mode.lower()
    if mode == "full":
        return out
    keep = {
        "geo_only": {"beta"},
        "visual_only": {"alpha"},
        "symbolic_only": {"gamma"},
        "visual_geo": {"alpha", "beta"},
        "visual_symbolic": {"alpha", "gamma"},
        "geo_symbolic": {"beta", "gamma"},
    }.get(mode)
    if keep is None:
        return out
    if "alpha" not in keep:
        out["alpha"] = 0.0
    if "beta" not in keep:
        out["beta"] = 0.0
    if "gamma" not in keep:
        out["gamma"] = 0.0
    norm = out["alpha"] + out["beta"] + out["gamma"]
    if norm <= 0:
        out["alpha"], out["beta"], out["gamma"] = 1.0, 0.0, 0.0
    else:
        out["alpha"] /= norm
        out["beta"] /= norm
        out["gamma"] /= norm
    return out


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def env_hardware_snapshot() -> Dict[str, Any]:
    return {
        "platform": os.name,
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
