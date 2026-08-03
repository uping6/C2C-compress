#!/usr/bin/env python3
"""HotpotQA-E ablation: zero decoded sharer KV at the receiver before fusing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

import script.evaluation.unified_evaluator as evaluator_module
from rosetta.cachejpeg_rosetta.wrapper import (
    CacheJPEGRosettaEvalWrapper,
    load_cachejpeg_rosetta_model,
)


PROJECT_ROOT = Path("/data/smy_data")
BASE_MODEL = "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
SHARER_MODEL = "/data/smy/lmy/C2C-compress-master/local/models/Qwen3-4B"
FUSER_ROOT = "/data/smy/lmy/C2C-compress-master/C2C_Fuser/qwen3_0.6b+qwen3_4b_Fuser"
DATA_DIR = "/data/smy/KVCache-Factory/data/LongBench"
ORIGINAL_DECODE_CACHE = CacheJPEGRosettaEvalWrapper.decode_cache


def decode_cache_then_zero_sharer_kv(self, payload):
    """Receiver-side hook: decode the transmitted cache, then erase sharer K/V."""
    decoded = ORIGINAL_DECODE_CACHE(self, payload)
    legacy = self._to_legacy_cache(decoded)
    zeroed = tuple((torch.zeros_like(key), torch.zeros_like(value)) for key, value in legacy)
    if not getattr(self, "_zero_sharer_kv_reported", False):
        nonzero = sum(
            int(torch.count_nonzero(key).item() + torch.count_nonzero(value).item())
            for key, value in zeroed
        )
        print(
            "[zero-sharer-kv] receiver hook active: "
            f"layers={len(zeroed)}, nonzero_after_zeroing={nonzero}; proceeding to fuser"
        )
        self._zero_sharer_kv_reported = True
    return zeroed


def make_config(output_dir: str, start: int, limit: int | None) -> dict:
    eval_config = {
        "dataset": "longbench",
        "subjects": ["hotpotqa"],
        "longbench_e": True,
        "longbench_local_data_dir": DATA_DIR,
        "longbench_prompt_format_path": "longbench/config/dataset2prompt.json",
        "longbench_maxlen_format_path": "longbench/config/dataset2maxlen.json",
        "gpu_ids": [1],
        "answer_method": "generate",
        "use_cot": False,
        "use_template": True,
        "sample_interval": 1,
        "skip_existing_longbench_subjects": False,
        "rosetta_checkpoint_subfolder": "final",
    }
    if limit is not None:
        eval_config["limit"] = [int(start), int(start) + int(limit)]
    return {
        "model": {
            # Route through evaluator's CacheJPEG branch; its loader is replaced below
            # by the CacheJPEG-Rosetta loader while keeping evaluator generation logic.
            "model_name": "cachejpeg",
            "rosetta_config": {
                "base_model": BASE_MODEL,
                "teacher_model": SHARER_MODEL,
                "checkpoints_dir": FUSER_ROOT,
            },
            "cachejpeg_rosetta_config": {
                "sharer_model_role": "teacher",
                "receiver_model_role": "base",
                "homo_c2c_kv_src": "/data/smy/HomoC2C-KV/src",
                "transport": {"mode": "none"},
                "codec": {
                    "method": "cachejpeg",
                    "compute": {"backend": "gpu", "transform_dtype": "float32"},
                    "anchors": {"sink_count": 1, "recent_count": 0, "preserve_options": False},
                    "block": {"mode": "global", "size": 64},
                    "quant": {
                        "q_global": 1.0, "low": 1.0, "high": 8.0, "curve": "quadratic",
                        "key_scale": 1.0, "value_scale": 1.0,
                        "layer_group_scales": {"early": 1.0, "middle": 1.0, "late": 1.0},
                        "clip_int16": True,
                    },
                    "zero_tail": {"mode": "none", "ratio": 0.0, "min_keep": 1},
                    "entropy": {"representation": "adaptive_int", "backend": "lz4"},
                },
            },
            "generation_config": {"do_sample": False, "max_new_tokens": 32},
        },
        "output": {"output_dir": output_dir},
        "eval": eval_config,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N rows.")
    parser.add_argument("--start", type=int, default=0, help="First dataset row for a worker shard.")
    parser.add_argument("--shard-size", type=int, default=None, help="Run all 300 rows in restartable shards.")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "tmp/hotpotqa_e_zero_sharer_kv"),
    )
    args = parser.parse_args()

    if args.shard_size:
        output_root = Path(args.output_dir)
        all_rows = {}
        for start in range(0, 300, args.shard_size):
            shard_size = min(args.shard_size, 300 - start)
            shard_dir = output_root / "shards" / f"{start:03d}_{start + shard_size:03d}"
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--start", str(start), "--limit", str(shard_size),
                "--output-dir", str(shard_dir),
            ]
            print(f"[shard] rows [{start}, {start + shard_size})")
            subprocess.run(command, check=True)
            shard_predictions = shard_dir / "pred_e/cachejpeg/hotpotqa.jsonl"
            with shard_predictions.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        all_rows[str(row.get("_id"))] = row

        merged_file = output_root / "pred_e/cachejpeg/hotpotqa.jsonl"
        merged_file.parent.mkdir(parents=True, exist_ok=True)
        with merged_file.open("w", encoding="utf-8") as stream:
            for row in all_rows.values():
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        scorer = evaluator_module.UnifiedEvaluator(make_config(str(output_root), 0, None))
        metrics = scorer._score_longbench_subject("hotpotqa_e", merged_file)
        metrics.update({
            "ablation": "receiver_zero_decoded_sharer_kv_before_fuser",
            "representation": "adaptive_int", "entropy_backend": "lz4", "transport": "none",
            "prediction_file": str(merged_file), "shard_size": args.shard_size,
        })
        metrics_file = output_root / "hotpotqa_e_zero_sharer_kv_metrics.json"
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return

    config = make_config(args.output_dir, args.start, args.limit)
    CacheJPEGRosettaEvalWrapper.decode_cache = decode_cache_then_zero_sharer_kv

    # UnifiedEvaluator currently dispatches model_name=cachejpeg via this symbol.
    # Adapt that signature to the CacheJPEG-Rosetta loader without changing source code.
    def load_ablation_model(model_config, device, generation_config=None):
        return load_cachejpeg_rosetta_model(
            model_config,
            config["eval"],
            device=device,
            generation_config=generation_config,
        )

    evaluator_module.load_cachejpeg_model = load_ablation_model
    evaluator = evaluator_module.UnifiedEvaluator(config)
    return_dict = {}
    evaluator.evaluate_on_gpu(0, 1, ["hotpotqa_e"], return_dict)

    prediction_file = Path(args.output_dir) / "pred_e/cachejpeg/hotpotqa.jsonl"
    metrics = evaluator._score_longbench_subject("hotpotqa_e", prediction_file)
    metrics.update(
        {
            "ablation": "receiver_zero_decoded_sharer_kv_before_fuser",
            "representation": "adaptive_int",
            "entropy_backend": "lz4",
            "transport": "none",
            "prediction_file": str(prediction_file),
        }
    )
    metrics_file = Path(args.output_dir) / "hotpotqa_e_zero_sharer_kv_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
