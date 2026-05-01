"""
Summarize experiment outputs into table-ready JSON artifacts.

ORACLE STATUS: NO-ORACLE
DESCRIPTION: 汇总实验输出生成表格数据
INPUTS: 实验输出目录（评估结果）
OUTPUTS: 表格 JSON 文件
GT USAGE: None - 仅读取已有的评估结果
PURPOSE: 后处理工具 - 生成论文表格
LAST UPDATED: 2026-04-28

✅ 此脚本不访问测试集标注，可用于正式推理。
注意：此脚本读取的评估结果可能来自 Oracle 评估，但脚本本身不访问 GT。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from egsr_common import OUTPUTS_DIR, read_json, slugify, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize EGSR outputs into fixed tables.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--main-variant", default="B5 EGSR Full")
    parser.add_argument(
        "--e3-variants",
        default="",
        help="Comma-separated variant names used in E3 triple-consistency ablation.",
    )
    parser.add_argument(
        "--e4-variants",
        default="B0 Direct Projection,B1 Top-1 Executable Scene,B5 w/o hard filtering,B5 w/o minimum explanation bias,B5 EGSR Full",
        help="Comma-separated variant names used in E4 module ablation.",
    )
    parser.add_argument(
        "--e5-variants",
        default="detector only,detector + OCR/rules,detector + OCR/rules + light VLM",
        help="Comma-separated variant names used in E5 candidate source ablation.",
    )
    parser.add_argument(
        "--e6-summary",
        type=Path,
        default=OUTPUTS_DIR / "eval" / "e6_k_efficiency_summary.json",
        help="Path to E6 timing+metric summary json.",
    )
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    return parser.parse_args()


def _load_pgdp(outputs_dir: Path, variant: str, split: str) -> Dict[str, Any]:
    path = outputs_dir / "eval" / slugify(variant) / split / "pgdp_metrics.json"
    return read_json(path)


def _load_oracle(outputs_dir: Path, variant: str, split: str) -> Dict[str, Any]:
    path = outputs_dir / "eval" / slugify(variant) / split / "oracle_k_metrics.json"
    return read_json(path)


def _load_score_meta(outputs_dir: Path, variant: str, split: str) -> Dict[str, Any]:
    path = outputs_dir / "scores" / slugify(variant) / split / "_run_meta.json"
    if path.exists():
        return read_json(path)
    return {}


def _compute_internal_metrics(outputs_dir: Path, variant: str, split: str) -> Dict[str, Any]:
    base = outputs_dir / "scores" / slugify(variant) / split
    if not base.exists():
        return {
            "Executability Rate": None,
            "Mean Residual": None,
            "Render Consistency": None,
        }
    exe = []
    res = []
    vis = []
    for path in base.glob("*.json"):
        if path.name.startswith("_"):
            continue
        data = read_json(path)
        ce = data.get("constraint_execution", {})
        vs = data.get("visual_scoring", {})
        if "executability_rate" in ce:
            exe.append(float(ce["executability_rate"]))
        if "mean_residual" in ce:
            res.append(float(ce["mean_residual"]))
        if "render_consistency" in vs:
            vis.append(float(vs["render_consistency"]))
    n = max(len(exe), 1)
    return {
        "Executability Rate": round(sum(exe) / n, 6) if exe else None,
        "Mean Residual": round(sum(res) / n, 6) if res else None,
        "Render Consistency": round(sum(vis) / n, 6) if vis else None,
    }


def _compute_selected_metrics(outputs_dir: Path, variant: str, split: str) -> Dict[str, Any]:
    base = outputs_dir / "scores" / slugify(variant) / split
    if not base.exists():
        return {
            "Executability Rate": None,
            "Mean Residual": None,
            "Render Consistency": None,
        }
    feasible_flags = []
    residuals = []
    render_vals = []
    for path in base.glob("*.json"):
        if path.name.startswith("_"):
            continue
        data = read_json(path)
        rer = data.get("reranking", {})
        sel = rer.get("selected_scene_id")
        if not sel:
            continue

        ce_rows = {x.get("scene_id"): x for x in data.get("constraint_execution", {}).get("scenes", [])}
        vs_rows = {x.get("scene_id"): x for x in data.get("visual_scoring", {}).get("scenes", [])}

        ce = ce_rows.get(sel, {})
        vs = vs_rows.get(sel, {})
        feasible = bool(ce.get("feasible", False))
        feasible_flags.append(1.0 if feasible else 0.0)
        if feasible and ("residual" in ce):
            residuals.append(float(ce["residual"]))
        if "s_visual" in vs:
            render_vals.append(float(vs["s_visual"]))

    n = max(len(feasible_flags), 1)
    return {
        "Executability Rate": round(sum(feasible_flags) / n, 6) if feasible_flags else None,
        "Mean Residual": round(sum(residuals) / max(len(residuals), 1), 6) if residuals else None,
        "Render Consistency": round(sum(render_vals) / max(len(render_vals), 1), 6) if render_vals else None,
    }


def _parse_csv_names(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    main_pgdp = _load_pgdp(args.outputs_dir, args.main_variant, args.split)
    main_oracle = _load_oracle(args.outputs_dir, args.main_variant, args.split)
    main_meta = _load_score_meta(args.outputs_dir, args.main_variant, args.split)
    main_internal = _compute_internal_metrics(args.outputs_dir, args.main_variant, args.split)

    tables: Dict[str, Any] = {}

    # Table 1: Main Results (external baselines as placeholders, EGSR filled).
    egsr = main_pgdp["official_metrics"]
    tables["Table1_MainResults"] = [
        {
            "Method": "InterGPS",
            "LS": 65.7,
            "AS": 44.4,
            "PR": 40.0,
            "TS": 27.3,
            "Source": "PGDP README (All row)",
        },
        {
            "Method": "PGDPNet w/o GNN",
            "LS": 98.4,
            "AS": 93.1,
            "PR": 79.7,
            "TS": 78.2,
            "Source": "PGDP README (All row)",
        },
        {
            "Method": "PGDPNet",
            "LS": 99.0,
            "AS": 96.6,
            "PR": 86.2,
            "TS": 84.7,
            "Source": "PGDP README (All row)",
        },
        {
            "Method": args.main_variant,
            "LS": egsr["LS"],
            "AS": egsr["AS"],
            "PR": egsr["PR"],
            "TS": egsr["TS"],
            "Source": "this run",
        },
    ]

    # Table 2: Top-K Candidate Quality
    tables["Table2_TopKCandidateQuality"] = main_oracle["summary"]

    # Table 3 + Table 4: E3 ablation if provided
    e3_variants = _parse_csv_names(args.e3_variants)
    table3 = []
    table4 = []
    for variant in e3_variants:
        pgdp = _load_pgdp(args.outputs_dir, variant, args.split)
        score_meta = _load_score_meta(args.outputs_dir, variant, args.split)
        m = pgdp["official_metrics"]
        table3.append(
            {
                "Variant": variant,
                "LS": m["LS"],
                "AS": m["AS"],
                "PR": m["PR"],
                "TS": m["TS"],
            }
        )
        table4.append({"Variant": variant, **_compute_selected_metrics(args.outputs_dir, variant, args.split)})
    if table3:
        tables["Table3_TripleConsistencyMain"] = table3
        tables["Table4_TripleConsistencyAnalysis"] = table4

    # Table 5: Module Ablation
    e4_variants = _parse_csv_names(args.e4_variants)
    table5 = []
    for variant in e4_variants:
        pgdp = _load_pgdp(args.outputs_dir, variant, args.split)
        internal = _compute_selected_metrics(args.outputs_dir, variant, args.split)
        m = pgdp["official_metrics"]
        table5.append(
            {
                "Variant": variant,
                "LS": m["LS"],
                "AS": m["AS"],
                "PR": m["PR"],
                "TS": m["TS"],
                "Executability Rate": internal["Executability Rate"],
                "Mean Residual": internal["Mean Residual"],
            }
        )
    if table5:
        tables["Table5_ModuleAblation"] = table5

    # Table 6: Candidate Source Ablation
    e5_variants = _parse_csv_names(args.e5_variants)
    table6 = []
    for variant in e5_variants:
        pgdp = _load_pgdp(args.outputs_dir, variant, args.split)
        oracle = _load_oracle(args.outputs_dir, variant, args.split)
        row10 = None
        for row in oracle.get("summary", []):
            if int(row.get("K", -1)) == 10:
                row10 = row
                break
        if row10 is None:
            continue
        table6.append(
            {
                "Candidate Source": variant,
                "Coverage@10": row10["Coverage@K"],
                "Oracle-AS@10": row10["Oracle-AS@K"],
                "Oracle-TS@10": row10["Oracle-TS@K"],
                "Final AS": pgdp["official_metrics"]["AS"],
                "Final TS": pgdp["official_metrics"]["TS"],
            }
        )
    if table6:
        tables["Table6_CandidateSourceAblation"] = table6

    # Table 7: K-Value and Efficiency
    table7 = []
    e6_protocol = {}
    if args.e6_summary.exists():
        e6 = read_json(args.e6_summary)
        e6_protocol = e6.get("hardware_protocol", {})
        for row in e6.get("rows", []):
            table7.append(
                {
                    "K": row["K"],
                    "LS": row["LS"],
                    "AS": row["AS"],
                    "PR": row["PR"],
                    "TS": row["TS"],
                    "Assembly + Constraint Time": row["assembly_constraint_time_sec_per_sample"],
                    "Rerender + Reranking Time": row["rerender_reranking_time_sec_per_sample"],
                    "Total Time": row["total_time_sec_per_sample"],
                }
            )
    if table7:
        tables["Table7_KValueAndEfficiency"] = table7

    payload = {
        "split": args.split,
        "main_variant": args.main_variant,
        "tables": tables,
        "notes": {
            "external_baseline_values": "Fill from quoted papers/repos as required by locked design.",
            "module_meta": main_meta,
            "main_internal_metrics": main_internal,
            "e6_timing_protocol": e6_protocol,
            "time_unit": "seconds per sample for Table 7 columns",
        },
    }

    out_file = args.outputs_dir / "eval" / slugify(args.main_variant) / args.split / "tables_summary.json"
    write_json(out_file, payload)
    print(f"[summarize_tables] wrote {out_file}")


if __name__ == "__main__":
    main()
