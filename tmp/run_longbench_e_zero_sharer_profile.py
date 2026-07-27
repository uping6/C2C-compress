#!/usr/bin/env python3
"""Run CacheJPEG-Rosetta LongBench-E with receiver-side zeroed sharer KV and profile codec costs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from script.evaluation.unified_evaluator import UnifiedEvaluator


ROOT = Path("/data/smy_data")
BASE_CONFIG = ROOT / "recipe/eval_recipe/longbench_jpegcache.yaml"
DEFAULT_OUTPUT = ROOT / "tmp/longbench_e_cachejpeg_rosetta_zero_sharer_profile"


def sample_score(evaluator: UnifiedEvaluator, subject: str, row: dict) -> float:
    metric = evaluator._longbench_metric_type(subject)
    gold = row.get("answers", [])
    golds = gold if isinstance(gold, list) else [str(gold)]
    pred = row.get("pred", "")
    if metric == "rouge_l":
        return max((evaluator._rouge_l_score(pred, item) for item in golds), default=0.0)
    if metric == "f1":
        return max((evaluator._f1_score(pred, item) for item in golds), default=0.0)
    return max((evaluator._exact_match_score(pred, item) for item in golds), default=0.0)


def summarize(records: list[dict], evaluator: UnifiedEvaluator) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["subject"], "ALL")].append(record)
        groups[(record["subject"], record["length_bucket"])].append(record)
    rows = []
    for (subject, bucket), items in sorted(groups.items()):
        raw_bytes = sum(item["original_kv_bytes"] for item in items)
        payload_bytes = sum(item["payload_bytes"] for item in items)
        metric = evaluator._longbench_metric_type(subject)
        rows.append(
            {
                "subject": subject,
                "length_bucket": bucket,
                "metric": metric,
                "num_samples": len(items),
                "correctness_score": float(np.mean([item["score"] for item in items])),
                "aggregate_compression_factor": float(raw_bytes / payload_bytes) if payload_bytes else 0.0,
                "aggregate_space_saving_ratio": float(1.0 - payload_bytes / raw_bytes) if raw_bytes else 0.0,
                "mean_sample_compression_factor": float(np.mean([item["compression_factor"] for item in items])),
                "avg_encode_seconds": float(np.mean([item["encode_seconds"] for item in items])),
                "avg_decode_seconds": float(np.mean([item["decode_seconds"] for item in items])),
                "total_original_kv_bytes": raw_bytes,
                "total_payload_bytes": payload_bytes,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="Run only qasper_e rows [0, 4).")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["output"]["output_dir"] = str(args.output_dir)
    config["eval"]["longbench_e"] = True
    config["eval"]["skip_existing_longbench_subjects"] = False
    config["model"]["cachejpeg_rosetta_config"]["ablation"] = {
        "zero_sharer_cache_at_receiver": True
    }
    if args.smoke:
        config["eval"]["subjects"] = ["qasper"]
        config["eval"]["limit"] = [0, 4]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_config = args.output_dir / "experiment_config.yaml"
    generated_config.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    if not args.aggregate_only:
        subprocess.run(
            [sys.executable, "script/evaluation/unified_evaluator.py", "--config", str(generated_config)],
            cwd=ROOT,
            check=True,
        )

    evaluator = UnifiedEvaluator(config)
    prediction_dir = args.output_dir / "pred_e/cachejpeg_rosetta"
    records = []
    for prediction_file in sorted(prediction_dir.glob("*.jsonl")):
        subject = prediction_file.stem
        with prediction_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                stats = row.get("cachejpeg_stats")
                if not stats:
                    raise RuntimeError(f"Missing cachejpeg_stats in {prediction_file}: {row.get('_id')}")
                if not stats.get("zero_sharer_cache_at_receiver"):
                    raise RuntimeError(f"Zero-sharer ablation was not active for {row.get('_id')}")
                records.append(
                    {
                        "subject": subject,
                        "_id": row.get("_id"),
                        "input_length": row.get("input_length"),
                        "length_bucket": row.get("length_bucket") or evaluator._longbench_length_bucket(row.get("input_length")),
                        "score": sample_score(evaluator, subject, row),
                        **stats,
                    }
                )

    summary_rows = summarize(records, evaluator)
    raw_file = args.output_dir / "sample_codec_metrics.jsonl"
    raw_file.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
    )
    csv_file = args.output_dir / "dataset_length_codec_accuracy_summary.csv"
    if summary_rows:
        with csv_file.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
    report = {
        "experiment": "cachejpeg_rosetta_receiver_zero_sharer_cache",
        "compression_factor_definition": "sum(original teacher KV bytes) / sum(serialized EncodedPayload bytes)",
        "space_saving_definition": "1 - sum(payload bytes) / sum(original teacher KV bytes)",
        "timing_definition": {
            "encode_seconds": "DCT + quantization + adaptive bit packing + LZ4",
            "decode_seconds": "LZ4 unpack + dequantization + inverse DCT",
            "payload_size_serialization_excluded_from_encode_time": True,
        },
        "num_samples": len(records),
        "rows": summary_rows,
    }
    report_file = args.output_dir / "dataset_length_codec_accuracy_summary.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"num_samples": len(records), "csv": str(csv_file), "json": str(report_file)}, indent=2))


if __name__ == "__main__":
    main()
