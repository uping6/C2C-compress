#!/usr/bin/env python3
"""Compare LongBench-E per-dataset scores for Zero-sharer and original runs.

The script reads the newest ``*_summary.json`` from each supplied result
directory.  Scores are LongBench's task-specific metrics (F1/EM/ROUGE, etc.),
not a single classification accuracy metric.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ANALYSIS_ROWS = [
    ("单文档QA", "Qasper", "qasper_e", "证据定位、答案长度"),
    ("单文档QA", "MultiFieldQA-en", "multifieldqa_en_e", "通用文档理解"),
    ("多文档QA", "HotpotQA", "hotpotqa_e", "多跳推理、文档选择"),
    ("多文档QA", "2WikiMQA", "2wikimqa_e", "实体关系、多跳推理"),
    ("摘要", "GovReport", "gov_report_e", "输出长度、全局覆盖"),
    ("摘要", "MultiNews", "multi_news_e", "去重、跨文档整合"),
    ("Few-shot", "TREC", "trec_e", "标签格式、示例归纳"),
    ("Few-shot", "TriviaQA", "triviaqa_e", "短答案、知识问答"),
    ("Few-shot摘要", "SAMSum", "samsum_e", "风格模仿、对话摘要"),
    ("合成计数", "PassageCount", "passage_count_e", "全局计数"),
    ("合成检索", "PassageRetrieval-en", "passage_retrieval_en_e", "精确位置检索"),
    ("代码", "LCC", "lcc_e", "文件内长依赖"),
    ("代码", "RepoBench-P", "repobench-p_e", "跨文件依赖"),
]


def newest_summary(result_dir: Path) -> Path:
    summaries = sorted(result_dir.glob("*_summary.json"), key=lambda path: path.stat().st_mtime)
    if not summaries:
        raise FileNotFoundError(f"No *_summary.json found in {result_dir}")
    return summaries[-1]


def load_subjects(summary_path: Path) -> dict[str, dict[str, Any]]:
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    subjects = summary.get("subjects")
    if not isinstance(subjects, dict):
        raise ValueError(f"Missing subjects mapping in {summary_path}")
    return subjects


def score_and_count(subjects: dict[str, dict[str, Any]], key: str) -> tuple[float, int]:
    item = subjects.get(key)
    if not isinstance(item, dict):
        raise KeyError(f"Dataset {key!r} is absent from summary")
    return float(item["score"]), int(item.get("num_samples", 0))


def render_table(zero_subjects: dict[str, dict[str, Any]], original_subjects: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| 分析组 | 数据集 | Zero | Original | Original−Zero | 主要怀疑因素 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    grouped: dict[str, list[tuple[float, float, int]]] = defaultdict(list)

    for group, display_name, key, concern in ANALYSIS_ROWS:
        zero_score, zero_count = score_and_count(zero_subjects, key)
        original_score, original_count = score_and_count(original_subjects, key)
        if zero_count != original_count:
            concern = f"{concern}；样本数 Zero/Original={zero_count}/{original_count}"
        delta = original_score - zero_score
        lines.append(
            f"| {group} | {display_name} | {zero_score:.2%} | {original_score:.2%} | "
            f"{delta:+.2%} | {concern} |"
        )
        grouped[group].append((zero_score, original_score, min(zero_count, original_count)))

    lines.extend(["", "### 分析组加权均分", "", "| 分析组 | Zero | Original | Original−Zero | 样本数 |", "| --- | ---: | ---: | ---: | ---: |"])
    for group in dict.fromkeys(row[0] for row in ANALYSIS_ROWS):
        values = grouped[group]
        weight = sum(count for _, _, count in values)
        zero_avg = sum(zero * count for zero, _, count in values) / weight
        original_avg = sum(original * count for _, original, count in values) / weight
        lines.append(
            f"| {group} | {zero_avg:.2%} | {original_avg:.2%} | "
            f"{original_avg - zero_avg:+.2%} | {weight} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zero-dir",
        type=Path,
        default=Path("local/final_results/0.6+1.5B_instruct/zero_sharer_cache"),
    )
    parser.add_argument(
        "--original-dir",
        type=Path,
        default=Path("local/final_results/0.6+1.5B_instruct/jpegcache_rosetta"),
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    args = parser.parse_args()

    zero_summary = newest_summary(args.zero_dir)
    original_summary = newest_summary(args.original_dir)
    report = "\n".join(
        [
            "# Zero-sharer-cache vs JPEGCache-Rosetta",
            "",
            f"- Zero summary: `{zero_summary}`",
            f"- Original summary: `{original_summary}`",
            "",
            render_table(load_subjects(zero_summary), load_subjects(original_summary)),
            "",
        ]
    )
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
