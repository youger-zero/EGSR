"""
Phase 7 (official): evaluate LS/AS/PR/TS on PGDP metrics.

ORACLE STATUS: ORACLE-DIAG
DESCRIPTION: 官方评估工具 - 计算 LS/AS/PR/TS 指标
INPUTS: 预测的逻辑形式、测试集标注（ground truth）
OUTPUTS: 评估指标（用于 Table 1）
GT USAGE: Diagnostic only - 使用 load_ground_truth 计算最终指标
PURPOSE: 最终评估 - 这是唯一允许访问测试集 GT 的阶段
LAST UPDATED: 2026-04-28

⚠️ 警告：此脚本使用测试集标注进行评估。
✅ 注意：评估是推理流程的最后一步，此时访问 GT 是标准实践。
✅ 关键：推理过程本身不访问 GT，只有评估时才访问。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

from egsr_common import (
    DATA_DIR,
    OUTPUTS_DIR,
    load_ground_truth,
    load_official_eval_module,
    metric_tag_from_scores,
    official_single_sample_scores,
    read_json,
    read_split_ids,
    slugify,
    summarize_metric_tags,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate EGSR logic forms with official PGDP metrics.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--variant", default="B5 EGSR Full")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument(
        "--pgdp-root",
        type=Path,
        default=Path("experiments/data/PGDP"),
        help="Root of the official PGDP repository clone.",
    )
    return parser.parse_args()


def _load_predictions(args: argparse.Namespace, split_ids) -> Dict[str, Dict[str, Any]]:
    merged_path = (
        args.outputs_dir
        / "logic_forms"
        / slugify(args.variant)
        / args.split
        / "_predictions_merged.json"
    )
    if merged_path.exists():
        return read_json(merged_path)

    pred: Dict[str, Dict[str, Any]] = {}
    base = args.outputs_dir / "logic_forms" / slugify(args.variant) / args.split
    for sid in split_ids:
        path = base / f"{sid}.json"
        if path.exists():
            pred[sid] = read_json(path)
    return pred


def main() -> None:
    args = parse_args()
    begin = time.perf_counter()

    split_ids = read_split_ids(args.split, args.data_dir)
    if args.limit > 0:
        split_ids = split_ids[: args.limit]

    gt = load_ground_truth(args.data_dir)
    pred = _load_predictions(args, split_ids)
    eval_handle = load_official_eval_module(args.pgdp_root)

    per_sample = {}
    tags = []
    for sid in split_ids:
        gt_graph = gt.get(sid)
        pred_graph = pred.get(sid)
        scores = official_single_sample_scores(eval_handle, gt_graph, pred_graph)
        tag = metric_tag_from_scores(scores)
        per_sample[sid] = {
            "accuracy": round(scores["accuracy"], 6),
            "recall": round(scores["recall"], 6),
            "f1": round(scores["f1"], 6),
            **tag,
        }
        tags.append(tag)

    summary = summarize_metric_tags(tags)
    elapsed = time.perf_counter() - begin
    payload = {
        "split": args.split,
        "variant": args.variant,
        "official_metrics": {
            "LS": round(summary["LS"], 4),
            "AS": round(summary["AS"], 4),
            "PR": round(summary["PR"], 4),
            "TS": round(summary["TS"], 4),
        },
        "num_samples": len(split_ids),
        "elapsed_sec": round(elapsed, 3),
        "per_sample": per_sample,
    }

    out_file = args.outputs_dir / "eval" / slugify(args.variant) / args.split / "pgdp_metrics.json"
    write_json(out_file, payload)
    print(
        f"[eval_pgdp_metrics] split={args.split} variant={args.variant} "
        f"LS={payload['official_metrics']['LS']:.2f} AS={payload['official_metrics']['AS']:.2f} "
        f"PR={payload['official_metrics']['PR']:.2f} TS={payload['official_metrics']['TS']:.2f} "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
