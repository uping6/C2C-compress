#!/usr/bin/env python3
"""Analyse behavior differences between Zero-sharer and original predictions.

Outputs:
  - behavior_summary.json: aggregate behavior metrics for both methods.
  - per_sample_comparison.csv: matched samples with length and LongBench score deltas.
  - per_dataset_comparison.csv: matched per-dataset aggregate deltas.

The evaluator stores generated-token counts but not EOS/finish-reason metadata.
Consequently, ``empty_or_missing_rate`` covers detectable empty/missing outputs;
``max_length_rate`` is reported separately as a proxy for truncation risk.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


QA_F1 = {"qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "triviaqa"}
ROUGE_L = {"gov_report", "multi_news", "samsum"}
EXACT = {"trec", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"}
WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
EXPLANATION_RE = re.compile(
    r"\b(because|therefore|thus|since|explanation|reason|based on|the answer is|"
    r"答案是|因为|所以|因此|解释)\b",
    flags=re.IGNORECASE,
)


def normalized(text: Any) -> str:
    text = "" if text is None else str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^\w\s]", "", text)


def tokens(text: Any) -> list[str]:
    return WORD_RE.findall(normalized(text))


def f1(prediction: Any, gold: Any) -> float:
    pred, target = tokens(prediction), tokens(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    common = Counter(pred) & Counter(target)
    same = sum(common.values())
    if not same:
        return 0.0
    precision, recall = same / len(pred), same / len(target)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: Any, gold: Any) -> float:
    pred, target = tokens(prediction), tokens(gold)
    if not pred and not target:
        return 1.0
    if not pred or not target:
        return 0.0
    dp = [0] * (len(target) + 1)
    for left in pred:
        previous = 0
        for index, right in enumerate(target, start=1):
            saved = dp[index]
            dp[index] = previous + 1 if left == right else max(dp[index], dp[index - 1])
            previous = saved
    lcs = dp[-1]
    if not lcs:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(target)
    return 2 * precision * recall / (precision + recall)


def exact(prediction: Any, gold: Any) -> float:
    return float(normalized(prediction) == normalized(gold))


def metric_for(dataset: str):
    name = re.sub(r"_e$", "", dataset)
    if name in QA_F1:
        return f1
    if name in ROUGE_L:
        return rouge_l
    if name in EXACT:
        return exact
    raise KeyError(f"Unsupported LongBench-E dataset: {dataset}")


def latest_summary_dir(result_dir: Path) -> Path:
    if not list(result_dir.glob("*_summary.json")):
        raise FileNotFoundError(f"No summary JSON in {result_dir}")
    return result_dir


def load_rows(result_dir: Path) -> dict[str, dict[str, Any]]:
    prediction_root = result_dir / "pred_e" / "cachejpeg_rosetta"
    if not prediction_root.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_root}")
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(prediction_root.glob("*.jsonl")):
        dataset = f"{path.stem}_e"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["dataset"] = dataset
            row_id = str(row["_id"])
            if row_id in rows:
                raise ValueError(f"Duplicate _id {row_id} in {result_dir}")
            rows[row_id] = row
    return rows


def load_questions(data_dir: Path, datasets: set[str]) -> dict[str, str]:
    questions: dict[str, str] = {}
    for dataset in datasets:
        path = data_dir / f"{dataset}.jsonl"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    questions[str(row["_id"])] = str(row.get("input", ""))
    return questions


def best_score(row: dict[str, Any]) -> float:
    scorer = metric_for(row["dataset"])
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [answers]
    return max((scorer(row.get("pred", ""), answer) for answer in answers), default=0.0)


def stored_length(row: dict[str, Any]) -> int:
    value = row.get("gen_length")
    if isinstance(value, int) and value >= 0:
        return value
    return len(tokens(row.get("pred", "")))


def reference_wrapped_explanation(row: dict[str, Any]) -> bool:
    """Detect a gold answer embedded in extra generated text, plus discourse cues."""
    pred_raw = str(row.get("pred", "")).strip()
    pred_norm = normalized(pred_raw)
    if not pred_norm:
        return False
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [answers]
    wrapped_gold = any(
        (gold_norm := normalized(answer))
        and gold_norm in pred_norm
        and gold_norm != pred_norm
        for answer in answers
    )
    return wrapped_gold or bool(EXPLANATION_RE.search(pred_raw))


def copy_ratio(row: dict[str, Any], question: str) -> float:
    out = tokens(row.get("pred", ""))
    if not out:
        return 0.0
    question_terms = set(tokens(question))
    return sum(token in question_terms for token in out) / len(out)


def behavior(rows: Iterable[dict[str, Any]], questions: dict[str, str], max_generation_length: int) -> dict[str, Any]:
    rows = list(rows)
    lengths = [stored_length(row) for row in rows]
    repetition = []
    copied = []
    wrapped = 0
    empty_or_missing = 0
    capped = 0
    for row, length in zip(rows, lengths):
        out_tokens = tokens(row.get("pred", ""))
        repetition.append(1.0 - len(set(out_tokens)) / len(out_tokens) if out_tokens else 0.0)
        copied.append(copy_ratio(row, questions.get(str(row["_id"]), "")))
        wrapped += reference_wrapped_explanation(row)
        empty_or_missing += int(not out_tokens or row.get("gen_length") is None)
        capped += int(length >= max_generation_length)
    count = len(rows)
    return {
        "samples": count,
        "average_output_tokens": statistics.fmean(lengths) if lengths else 0.0,
        "median_output_tokens": statistics.median(lengths) if lengths else 0.0,
        "max_length_rate": capped / count if count else 0.0,
        "token_repetition_rate": statistics.fmean(repetition) if repetition else 0.0,
        "question_copy_ratio": statistics.fmean(copied) if copied else 0.0,
        "answer_wrapped_or_explained_rate": wrapped / count if count else 0.0,
        "empty_or_missing_rate": empty_or_missing / count if count else 0.0,
    }


def write_csv(path: Path, matched: list[tuple[str, dict[str, Any], dict[str, Any], str]]) -> None:
    columns = [
        "_id", "dataset", "length_bucket", "zero_gen_tokens", "original_gen_tokens",
        "original_minus_zero_tokens", "zero_score", "original_score", "original_minus_zero_score",
        "zero_copy_ratio", "original_copy_ratio", "zero_explanation", "original_explanation",
        "zero_empty_or_missing", "original_empty_or_missing",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row_id, zero, original, question in matched:
            zero_len, original_len = stored_length(zero), stored_length(original)
            zero_score, original_score = best_score(zero), best_score(original)
            writer.writerow({
                "_id": row_id,
                "dataset": zero["dataset"],
                "length_bucket": zero.get("length_bucket", "unknown"),
                "zero_gen_tokens": zero_len,
                "original_gen_tokens": original_len,
                "original_minus_zero_tokens": original_len - zero_len,
                "zero_score": f"{zero_score:.8f}",
                "original_score": f"{original_score:.8f}",
                "original_minus_zero_score": f"{original_score - zero_score:.8f}",
                "zero_copy_ratio": f"{copy_ratio(zero, question):.8f}",
                "original_copy_ratio": f"{copy_ratio(original, question):.8f}",
                "zero_explanation": int(reference_wrapped_explanation(zero)),
                "original_explanation": int(reference_wrapped_explanation(original)),
                "zero_empty_or_missing": int(not tokens(zero.get("pred", "")) or zero.get("gen_length") is None),
                "original_empty_or_missing": int(not tokens(original.get("pred", "")) or original.get("gen_length") is None),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-dir", type=Path, default=Path("local/final_results/0.6+1.5B_instruct/zero_sharer_cache"))
    parser.add_argument("--original-dir", type=Path, default=Path("local/final_results/0.6+1.5B_instruct/jpegcache_rosetta"))
    parser.add_argument("--longbench-data", type=Path, default=Path("/data/smy/KVCache-Factory/data/LongBench"))
    parser.add_argument("--max-generation-length", type=int, default=2048)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/zero_original_output_behavior"))
    args = parser.parse_args()

    zero_rows = load_rows(latest_summary_dir(args.zero_dir))
    original_rows = load_rows(latest_summary_dir(args.original_dir))
    questions = load_questions(
        args.longbench_data,
        {row["dataset"] for row in zero_rows.values()} | {row["dataset"] for row in original_rows.values()},
    )
    shared_ids = sorted(set(zero_rows) & set(original_rows))
    matched = [(row_id, zero_rows[row_id], original_rows[row_id], questions.get(row_id, "")) for row_id in shared_ids]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "matched_samples": len(matched),
        "zero_only_predictions": len(set(zero_rows) - set(original_rows)),
        "original_only_predictions": len(set(original_rows) - set(zero_rows)),
        "definitions": {
            "token_repetition_rate": "mean(1 - unique normalized output tokens / normalized output tokens)",
            "question_copy_ratio": "mean(output tokens also appearing in the raw LongBench question / output tokens)",
            "answer_wrapped_or_explained_rate": "output wraps a reference answer with extra text, or contains an explanation discourse cue",
            "empty_or_missing_rate": "empty normalized output or missing stored generation length; EOS finish reasons are not stored",
            "max_length_rate": "stored gen_length >= configured max generation length",
        },
        "zero": behavior((zero for _, zero, _, _ in matched), questions, args.max_generation_length),
        "original": behavior((original for _, _, original, _ in matched), questions, args.max_generation_length),
    }
    (args.output_dir / "behavior_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.output_dir / "per_sample_comparison.csv", matched)

    per_dataset: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for _, zero, original, _ in matched:
        per_dataset[zero["dataset"]].append((zero, original))
    with (args.output_dir / "per_dataset_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "samples", "zero_score", "original_score", "original_minus_zero_score", "zero_tokens", "original_tokens", "original_minus_zero_tokens"])
        for dataset, pairs in sorted(per_dataset.items()):
            zero_scores = [best_score(zero) for zero, _ in pairs]
            original_scores = [best_score(original) for _, original in pairs]
            zero_lengths = [stored_length(zero) for zero, _ in pairs]
            original_lengths = [stored_length(original) for _, original in pairs]
            writer.writerow([
                dataset, len(pairs), f"{statistics.fmean(zero_scores):.8f}",
                f"{statistics.fmean(original_scores):.8f}",
                f"{statistics.fmean(original_scores) - statistics.fmean(zero_scores):.8f}",
                f"{statistics.fmean(zero_lengths):.3f}", f"{statistics.fmean(original_lengths):.3f}",
                f"{statistics.fmean(original_lengths) - statistics.fmean(zero_lengths):.3f}",
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote analysis files to {args.output_dir}")


if __name__ == "__main__":
    main()
