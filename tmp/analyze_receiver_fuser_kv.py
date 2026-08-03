#!/usr/bin/env python3
"""Measure receiver KV and fuser-induced KV residual distributions layer by layer.

Example (run from /data/smy_data):
  conda run -n rosetta bash -lc 'PYTHONPATH=/data/smy_data python \
    tmp/analyze_receiver_fuser_kv.py \
    --teacher-model local/models/Qwen2.5-1.5B-Instruct \
    --fuser-checkpoint /path/to/compatible/final'

The fuser checkpoint must have been trained for the selected receiver/sharer
pair.  The script can still execute with an incompatible checkpoint when tensor
shapes match, but those statistics must not be interpreted as model quality.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from rosetta.cachejpeg_rosetta.fuser_bridge import RosettaFuserBridge
from rosetta.cachejpeg_rosetta.wrapper import _load_rosetta_assets
from rosetta.model.aligner import AlignmentStrategy, TokenAligner


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | int | list[int]]:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return {"shape": list(tensor.shape), "numel": 0}
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.50, 0.99], device=values.device))
    return {
        "shape": list(tensor.shape),
        "numel": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "abs_mean": float(values.abs().mean().item()),
        "rms": float(values.square().mean().sqrt().item()),
        "p01": float(quantiles[0].item()),
        "p50": float(quantiles[1].item()),
        "p99": float(quantiles[2].item()),
        "zero_fraction": float((values == 0).float().mean().item()),
    }


def aligned_inputs(assets: Any, prompt: str, device: torch.device, strategy: str):
    aligner = TokenAligner(
        slm_tokenizer=assets.base_tokenizer,
        llm_tokenizer=assets.teacher_tokenizer,
        strategy=AlignmentStrategy(strategy),
    )
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
        raise ValueError("Token alignment did not produce equal receiver/sharer sequence lengths")
    return base_ids, teacher_ids, base_mask, teacher_mask, details


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer", "cache", "component", "shape", "numel", "mean", "std", "min", "max",
        "abs_mean", "rms", "p01", "p50", "p99", "zero_fraction", "residual_to_receiver_rms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receiver-model",
        default="/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
    )
    parser.add_argument("--teacher-model", default="local/models/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--fuser-checkpoint",
        required=True,
        help="Directory containing projector_*.pt/json and projector_config.json.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alignment-strategy", choices=["longest", "prefix"], default="longest")
    parser.add_argument(
        "--prompt",
        default="Read the passage carefully and answer briefly: What is the central claim of the passage?",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/receiver_fuser_kv_stats"))
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.fuser_checkpoint).resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fuser checkpoint not found: {checkpoint}")

    assets = _load_rosetta_assets(
        {"rosetta_config": {
            "base_model": args.receiver_model,
            "teacher_model": args.teacher_model,
            "checkpoints_dir": str(checkpoint),
        }},
        {},
        device=device,
        generation_config={"do_sample": False},
    )
    base_ids, teacher_ids, base_mask, teacher_mask, details = aligned_inputs(
        assets, args.prompt, device, args.alignment_strategy
    )
    with torch.no_grad():
        receiver = assets.base_model(input_ids=base_ids, attention_mask=base_mask, use_cache=True)
        sharer = assets.teacher_model(input_ids=teacher_ids, attention_mask=teacher_mask, use_cache=True)

    bridge = RosettaFuserBridge(assets)
    receiver_cache = bridge._to_dynamic_cache(receiver.past_key_values)
    fused_cache = bridge.fuse_teacher_cache_to_base(
        sharer.past_key_values, base_seed_cache=receiver_cache
    )

    rows: list[dict[str, Any]] = []
    for layer, (receiver_key, receiver_value, fused_key, fused_value) in enumerate(
        zip(receiver_cache.key_cache, receiver_cache.value_cache, fused_cache.key_cache, fused_cache.value_cache)
    ):
        for component, receiver_tensor, fused_tensor in (
            ("K", receiver_key, fused_key),
            ("V", receiver_value, fused_value),
        ):
            residual = fused_tensor - receiver_tensor
            receiver_stats = tensor_stats(receiver_tensor)
            fused_stats = tensor_stats(fused_tensor)
            residual_stats = tensor_stats(residual)
            receiver_rms = float(receiver_stats.get("rms", 0.0))
            ratio = float(residual_stats.get("rms", 0.0)) / receiver_rms if receiver_rms else None
            for cache_name, stats in (
                ("receiver", receiver_stats),
                ("fused", fused_stats),
                ("fused_minus_receiver", residual_stats),
            ):
                rows.append({
                    "layer": layer,
                    "cache": cache_name,
                    "component": component,
                    **stats,
                    "shape": "x".join(map(str, stats["shape"])),
                    "residual_to_receiver_rms": ratio if cache_name == "fused_minus_receiver" else "",
                })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_layer_kv_distribution.csv", rows)
    metadata = {
        "receiver_model": args.receiver_model,
        "teacher_model": args.teacher_model,
        "fuser_checkpoint": str(checkpoint),
        "device": str(device),
        "alignment_strategy": args.alignment_strategy,
        "prompt": args.prompt,
        "aligned_sequence_length": int(base_ids.shape[1]),
        "receiver_layers": len(receiver_cache.key_cache),
        "fused_layers": len(fused_cache.key_cache),
        "projector_mapping": assets.projector_dict,
        "outputs": {"per_layer_csv": "per_layer_kv_distribution.csv"},
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} K/V distribution rows to {args.output_dir}")


if __name__ == "__main__":
    main()
