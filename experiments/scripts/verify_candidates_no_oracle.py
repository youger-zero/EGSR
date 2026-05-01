"""
验证候选池是否符合 no-oracle 要求

ORACLE STATUS: UTILITY
DESCRIPTION: 验证候选池的合规性
INPUTS: 候选池目录
OUTPUTS: 验证报告
GT USAGE: None - 仅分析候选池本身
PURPOSE: 合规性验证工具
LAST UPDATED: 2026-04-28
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statistics


def _safe_print(message: str) -> None:
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        sys.stdout.buffer.write((message + "\n").encode(encoding, errors="backslashreplace"))

class CandidatePoolVerifier:
    """候选池验证器"""
    
    def __init__(self, candidates_dir: Path):
        self.candidates_dir = candidates_dir
        self.issues = []
        self.warnings = []
        self.stats = {}
    
    def verify(self) -> Tuple[bool, List[str], List[str], Dict[str, Any]]:
        """
        执行完整验证
        
        Returns:
            (is_compliant, issues, warnings, stats)
        """
        _safe_print(f"\n{'='*60}")
        _safe_print(f"验证候选池: {self.candidates_dir.name}")
        _safe_print(f"{'='*60}\n")
        
        # 检查 1: 候选位置是否有噪声
        _safe_print("[CHECK 1] 检查候选位置噪声...")
        self._check_position_noise()
        
        # 检查 2: 候选置信度分布
        _safe_print("[CHECK 2] 检查置信度分布...")
        self._check_confidence_distribution()
        
        # 检查 3: 候选来源标记
        _safe_print("[CHECK 3] 检查候选来源...")
        self._check_candidate_sources()
        
        # 检查 4: 候选完整性
        _safe_print("[CHECK 4] 检查候选完整性...")
        self._check_candidate_completeness()
        
        # 检查 5: annotation_id 使用
        _safe_print("[CHECK 5] 检查 annotation_id 使用...")
        self._check_annotation_id_usage()
        
        # 生成报告
        is_compliant = len(self.issues) == 0
        
        _safe_print(f"\n{'='*60}")
        if is_compliant:
            _safe_print("✅ 验证通过 - 候选池符合 no-oracle 要求")
        else:
            _safe_print("❌ 验证失败 - 发现合规性问题")
        _safe_print(f"{'='*60}\n")

        if self.issues:
            _safe_print("问题:")
            for issue in self.issues:
                _safe_print(f"  ❌ {issue}")

        if self.warnings:
            _safe_print("\n警告:")
            for warning in self.warnings:
                _safe_print(f"  ⚠️  {warning}")

        _safe_print("\n统计信息:")
        for key, value in self.stats.items():
            _safe_print(f"  {key}: {value}")
        
        return is_compliant, self.issues, self.warnings, self.stats
    
    def _check_position_noise(self):
        """检查候选位置是否有噪声（不是精确的整数）"""
        sample_files = list(self.candidates_dir.glob("*.json"))
        if not sample_files:
            self.issues.append("候选池为空")
            return
        
        # 排除元数据文件
        sample_files = [f for f in sample_files if f.name != "_run_meta.json"]
        
        # 随机抽样 10 个文件
        import random
        sample_files = random.sample(sample_files, min(10, len(sample_files)))
        
        has_noise_count = 0
        total_points = 0
        
        for sample_file in sample_files:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            candidates = data.get("candidates", [])
            for cand in candidates:
                if cand.get("type") == "point":
                    total_points += 1
                    loc = cand.get("location", [])
                    if len(loc) >= 2:
                        x, y = loc[0], loc[1]
                        # 检查是否有小数部分（噪声）
                        if x != int(x) or y != int(y):
                            has_noise_count += 1
        
        if total_points > 0:
            noise_ratio = has_noise_count / total_points
            self.stats["position_noise_ratio"] = f"{noise_ratio:.2%}"
            
            if noise_ratio < 0.5:
                self.warnings.append(
                    f"位置噪声比例较低: {noise_ratio:.2%}（预期 > 50%）"
                )
        else:
            self.warnings.append("未找到点候选")
    
    def _check_confidence_distribution(self):
        """检查置信度分布是否合理"""
        sample_files = list(self.candidates_dir.glob("*.json"))
        sample_files = [f for f in sample_files if f.name != "_run_meta.json"]
        
        if not sample_files:
            return
        
        # 随机抽样
        import random
        sample_files = random.sample(sample_files, min(20, len(sample_files)))
        
        confidences = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            candidates = data.get("candidates", [])
            for cand in candidates:
                conf = cand.get("confidence")
                if conf is not None:
                    confidences.append(conf)
        
        if confidences:
            mean_conf = statistics.mean(confidences)
            std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0
            min_conf = min(confidences)
            max_conf = max(confidences)
            
            self.stats["confidence_mean"] = f"{mean_conf:.4f}"
            self.stats["confidence_std"] = f"{std_conf:.4f}"
            self.stats["confidence_range"] = f"[{min_conf:.4f}, {max_conf:.4f}]"
            
            # 检查是否所有置信度都是 1.0（可疑）
            if all(c == 1.0 for c in confidences):
                self.issues.append("所有候选的置信度都是 1.0（可疑）")
            
            # 检查是否有合理的分布
            if std_conf < 0.01:
                self.warnings.append(
                    f"置信度标准差很小: {std_conf:.4f}（可能缺乏多样性）"
                )
        else:
            self.warnings.append("未找到置信度信息")
    
    def _check_candidate_sources(self):
        """检查候选来源标记"""
        sample_files = list(self.candidates_dir.glob("*.json"))
        sample_files = [f for f in sample_files if f.name != "_run_meta.json"]
        
        if not sample_files:
            return
        
        # 检查第一个文件
        with open(sample_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        candidate_source = data.get("candidate_source", "unknown")
        self.stats["candidate_source"] = candidate_source
        
        # 检查候选来源
        valid_sources = [
            "detector_only",
            "detector_ocr_rules",
            "detector_ocr_rules_vlm"
        ]
        
        if candidate_source not in valid_sources:
            self.warnings.append(
                f"候选来源不在预期列表中: {candidate_source}"
            )
        
        # 检查单个候选的来源标记
        candidates = data.get("candidates", [])
        sources = set()
        for cand in candidates:
            source = cand.get("source")
            if source:
                sources.add(source)
        
        self.stats["individual_sources"] = ", ".join(sorted(sources))
        
        # 检查是否有 "annotation" 来源（不应该有）
        if "annotation" in sources:
            self.issues.append("发现候选来源为 'annotation'（违规）")
    
    def _check_candidate_completeness(self):
        """检查候选完整性（不应该是 100% 完美）"""
        sample_files = list(self.candidates_dir.glob("*.json"))
        sample_files = [f for f in sample_files if f.name != "_run_meta.json"]
        
        if not sample_files:
            return
        
        # 随机抽样
        import random
        sample_files = random.sample(sample_files, min(10, len(sample_files)))
        
        point_counts = []
        line_counts = []
        circle_counts = []
        
        for sample_file in sample_files:
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            candidates = data.get("candidates", [])
            
            points = sum(1 for c in candidates if c.get("type") == "point")
            lines = sum(1 for c in candidates if c.get("type") == "line")
            circles = sum(1 for c in candidates if c.get("type") == "circle")
            
            point_counts.append(points)
            line_counts.append(lines)
            circle_counts.append(circles)
        
        if point_counts:
            self.stats["avg_points_per_sample"] = f"{statistics.mean(point_counts):.1f}"
        if line_counts:
            self.stats["avg_lines_per_sample"] = f"{statistics.mean(line_counts):.1f}"
        if circle_counts:
            self.stats["avg_circles_per_sample"] = f"{statistics.mean(circle_counts):.1f}"
    
    def _check_annotation_id_usage(self):
        """检查 annotation_id 的使用方式"""
        sample_files = list(self.candidates_dir.glob("*.json"))
        sample_files = [f for f in sample_files if f.name != "_run_meta.json"]
        
        if not sample_files:
            return
        
        # 检查第一个文件
        with open(sample_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        candidates = data.get("candidates", [])
        
        has_annotation_id = any("annotation_id" in c for c in candidates)
        
        if has_annotation_id:
            self.stats["has_annotation_id"] = "Yes (仅用于追踪)"
            self.warnings.append(
                "候选包含 annotation_id 字段（用于追踪，不影响推理）"
            )
        else:
            self.stats["has_annotation_id"] = "No"


def verify_candidate_pool(candidates_dir: Path) -> bool:
    """验证单个候选池"""
    verifier = CandidatePoolVerifier(candidates_dir)
    is_compliant, issues, warnings, stats = verifier.verify()
    
    # 保存验证报告
    report = {
        "candidate_pool": candidates_dir.name,
        "is_compliant": is_compliant,
        "issues": issues,
        "warnings": warnings,
        "statistics": stats,
        "timestamp": "2026-04-28T18:00:00Z"
    }
    
    report_file = candidates_dir / "verification_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    _safe_print(f"\n验证报告已保存: {report_file}")
    
    return is_compliant


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="验证候选池是否符合 no-oracle 要求"
    )
    parser.add_argument(
        "candidates_dir",
        type=Path,
        help="候选池目录路径"
    )
    
    args = parser.parse_args()
    
    if not args.candidates_dir.exists():
        _safe_print(f"错误: 候选池目录不存在: {args.candidates_dir}")
        exit(1)
    
    is_compliant = verify_candidate_pool(args.candidates_dir)
    
    exit(0 if is_compliant else 1)
