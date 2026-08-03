#!/usr/bin/env python3
"""Stream LongBench-E receiver/fused/residual KV distribution statistics on one GPU.

The script uses the evaluator's LongBench-E modulo-4 selection rule and does
not persist KV tensors.  Per-layer mean/std/RMS/min/max/zero fraction are exact
over all processed tensor elements; p01/p50/p99 are estimated from a bounded
reservoir sample, so the run remains memory bounded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rosetta.cachejpeg_rosetta.fuser_bridge import RosettaFuserBridge
from rosetta.cachejpeg_rosetta.wrapper import _load_rosetta_assets
from rosetta.model.aligner import AlignmentStrategy, TokenAligner


SUBJECTS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
    "passage_retrieval_en", "lcc", "repobench-p",
]


@dataclass
class Distribution:
    reservoir_size: int
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    total_abs: float = 0.0
    zero_count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    reservoir: list[float] = field(default_factory=list)

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1)
        n = int(flat.numel())
        if not n:
            return
        summary = torch.stack((
            flat.sum(dtype=torch.float64),
            flat.square().sum(dtype=torch.float64),
            flat.abs().sum(dtype=torch.float64),
            (flat == 0).sum(dtype=torch.float64),
            flat.min().to(torch.float64),
            flat.max().to(torch.float64),
        )).cpu().tolist()
        self.count += n
        self.total += summary[0]
        self.total_sq += summary[1]
        self.total_abs += summary[2]
        self.zero_count += int(summary[3])
        self.minimum = min(self.minimum, summary[4])
        self.maximum = max(self.maximum, summary[5])

        # Sample at most 128 positions per tensor.  Reservoir downsampling keeps
        # memory bounded while preserving a representative value distribution.
        sample_count = min(128, n)
        positions = torch.randint(n, (sample_count,), device=flat.device)
        self.reservoir.extend(flat[positions].cpu().tolist())
        if len(self.reservoir) > self.reservoir_size * 2:
            self.reservoir = random.sample(self.reservoir, self.reservoir_size)

    def as_dict(self) -> dict[str, float | int]:
        if not self.count:
            return {"numel": 0}
        if len(self.reservoir) > self.reservoir_size:
            self.reservoir = random.sample(self.reservoir, self.reservoir_size)
        samples = torch.tensor(self.reservoir, dtype=torch.float32)
        quantiles = torch.quantile(samples, torch.tensor([0.01, 0.50, 0.99])).tolist()
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return {
            "numel": self.count,
            "mean": mean,
            "std": variance ** 0.5,
            "min": self.minimum,
            "max": self.maximum,
            "abs_mean": self.total_abs / self.count,
            "rms": (self.total_sq / self.count) ** 0.5,
            "p01_estimate": quantiles[0],
            "p50_estimate": quantiles[1],
            "p99_estimate": quantiles[2],
            "zero_fraction": self.zero_count / self.count,
            "reservoir_values": len(self.reservoir),
        }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def truncate_prompt(prompt: str, tokenizer: Any, max_length: int) -> str:
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(ids) <= max_length:
        return prompt
    half = max_length // 2
    return tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(ids[-half:], skip_special_tokens=True)


def make_aligned_inputs(assets: Any, prompt: str, device: torch.device, strategy: str):
    aligner = TokenAligner(assets.base_tokenizer, assets.teacher_tokenizer, AlignmentStrategy(strategy))
    details = aligner.align_chat_messages(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_details=True,
        enable_thinking=False,
    )
    base_ids = torch.tensor(details["slm_ids_padded"], device=device).unsqueeze(0)
    teacher_ids = torch.tensor(details["llm_ids_padded"], device=device).unsqueeze(0)
    base_mask = (~torch.tensor(details["slm_padding_mask"])).to(device).unsqueeze(0)
    teacher_mask = (~torch.tensor(details["llm_padding_mask"])).to(device).unsqueeze(0)
    if base_ids.shape[1] != teacher_ids.shape[1]:
        raise ValueError("TokenAligner returned mismatched cache lengths")
    return base_ids, teacher_ids, base_mask, teacher_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receiver-model", default="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    parser.add_argument("--teacher-model", default="local/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--fuser-checkpoint", default="/data/smy_data/local/checkpoints/0.6+1.5B_instruct_longbench_e/final")
    parser.add_argument("--longbench-data", type=Path, default=Path("/data/smy/KVCache-Factory/data/LongBench"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-prompt-tokens", type=int, default=12000)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all selected LongBench-E samples.")
    parser.add_argument("--reservoir-size", type=int, default=10000)
    parser.add_argument("--alignment-strategy", choices=["longest", "prefix"], default="longest")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/longbench_receiver_fuser_kv_stats"))
    args = parser.parse_args()
    random.seed(42)
    device = torch.device(args.device)
    checkpoint = Path(args.fuser_checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    assets = _load_rosetta_assets(
        {"rosetta_config": {"base_model": args.receiver_model, "teacher_model": args.teacher_model, "checkpoints_dir": str(checkpoint)}},
        {}, device, {"do_sample": False},
    )
    bridge = RosettaFuserBridge(assets)
    prompt_templates = json.loads(
        # LongBench's distributed prompt file is UTF-8 with a BOM.
        Path("longbench/config/dataset2prompt.json").read_text(encoding="utf-8-sig")
    )
    stats: dict[tuple[int, str, str], Distribution] = {}
    processed_by_subject: dict[str, int] = {}
    failures: list[dict[str, str]] = []
    processed = 0
    start = time.monotonic()

    for subject in SUBJECTS:
        rows = load_jsonl(args.longbench_data / f"{subject}_e.jsonl")
        selected = [
            row for row in rows
            if int(hashlib.sha256(str(row["_id"]).encode("utf-8")).hexdigest(), 16) % 4 == 1
        ]
        for sample in selected:
            if args.max_samples and processed >= args.max_samples:
                break
            try:
                # Use LongBench's raw task prompt; TokenAligner applies one chat template.
                raw_prompt = prompt_templates[subject].format(**sample)
                raw_prompt = truncate_prompt(raw_prompt, assets.base_tokenizer, args.max_prompt_tokens)
                base_ids, teacher_ids, base_mask, teacher_mask = make_aligned_inputs(
                    assets, raw_prompt, device, args.alignment_strategy
                )
                with torch.no_grad():
                    receiver_out = assets.base_model(input_ids=base_ids, attention_mask=base_mask, use_cache=True)
                    teacher_out = assets.teacher_model(input_ids=teacher_ids, attention_mask=teacher_mask, use_cache=True)
                receiver_cache = bridge._to_dynamic_cache(receiver_out.past_key_values)
                fused_cache = bridge.fuse_teacher_cache_to_base(teacher_out.past_key_values, receiver_cache)
                for layer, tensors in enumerate(zip(receiver_cache.key_cache, receiver_cache.value_cache, fused_cache.key_cache, fused_cache.value_cache)):
                    receiver_k, receiver_v, fused_k, fused_v = tensors
                    for component, receiver_tensor, fused_tensor in (("K", receiver_k, fused_k), ("V", receiver_v, fused_v)):
                        for cache_name, tensor in (("receiver", receiver_tensor), ("fused", fused_tensor), ("residual", fused_tensor - receiver_tensor)):
                            key = (layer, component, cache_name)
                            stats.setdefault(key, Distribution(args.reservoir_size)).update(tensor)
                processed += 1
                processed_by_subject[subject] = processed_by_subject.get(subject, 0) + 1
                if processed % 10 == 0:
                    print(f"processed={processed} subject={subject} elapsed_s={time.monotonic() - start:.1f}", flush=True)
            except Exception as error:  # retain a complete audit without stopping the all-set run
                failures.append({"subject": subject, "_id": str(sample.get("_id")), "error": repr(error)})
                print(f"FAILED subject={subject} id={sample.get('_id')} error={error!r}", flush=True)
            finally:
                torch.cuda.empty_cache()
        if args.max_samples and processed >= args.max_samples:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for (layer, component, cache_name), distribution in sorted(stats.items()):
        rows.append({"layer": layer, "component": component, "cache": cache_name, **distribution.as_dict()})
    with (args.output_dir / "per_layer_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["layer"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "processed_samples": processed,
        "processed_by_subject": processed_by_subject,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - start,
        "receiver_model": args.receiver_model,
        "teacher_model": args.teacher_model,
        "fuser_checkpoint": str(checkpoint),
        "max_prompt_tokens": args.max_prompt_tokens,
        "reservoir_size": args.reservoir_size,
        "quantile_note": "p01/p50/p99 are reservoir estimates; other reported moments are exact over processed tensor values.",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE processed={processed} failures={len(failures)} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
