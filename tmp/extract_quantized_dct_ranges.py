from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from homo_c2c_kv.codec.cachejpeg.config import resolve_cachejpeg_config
from homo_c2c_kv.codec.cachejpeg.quant_table import build_frequency_table, frequency_band_slice
from rosetta.cachejpeg.gpu_codec import (
    _anchor_indices,
    _body_indices,
    _forward_dct,
    _quantize,
)
from rosetta.cachejpeg.wrapper import load_cachejpeg_model
from script.evaluation.unified_evaluator import UnifiedEvaluator


PROJECT_ROOT = Path("/data/smy_data")
OUT_DIR = PROJECT_ROOT / "tmp"
DEVICE = torch.device("cuda:1")
BANDS = tuple(f"B{i}" for i in range(6))


def _dimension_stats(values: torch.Tensor) -> dict[str, np.ndarray]:
    # Reduce batch, heads and DCT frequency, retaining the head dimension.
    flattened = values.movedim(-1, 0).reshape(values.shape[-1], -1).float()
    absolute = flattened.abs()
    return {
        "min": flattened.amin(dim=1).cpu().numpy(),
        "max": flattened.amax(dim=1).cpu().numpy(),
        "abs_max": absolute.amax(dim=1).cpu().numpy(),
        "abs_p99": torch.quantile(absolute, 0.99, dim=1).cpu().numpy(),
    }


def main() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "recipe/eval_recipe/longbench_jpegcache.yaml").read_text()
    )
    codec_dict = dict(config["model"]["cachejpeg_config"])
    codec_cfg = resolve_cachejpeg_config(codec_dict)
    config["output"]["output_dir"] = "/tmp/cachejpeg_quantized_dct_analysis"
    evaluator = UnifiedEvaluator(config)
    model, tokenizer = load_cachejpeg_model(
        evaluator.model_config,
        device=DEVICE,
        generation_config={"max_new_tokens": 1, "do_sample": False},
    )
    subject = "qasper"
    data_path = Path(config["eval"]["longbench_local_data_dir"]) / f"{subject}.jsonl"
    rows = [json.loads(line) for line in data_path.read_text().splitlines() if line.strip()]
    sample_index = next(
        i for i, row in enumerate(rows)
        if int(hashlib.sha256(str(row["_id"]).encode()).hexdigest(), 16) % 4 == 1
    )
    evaluator.current_evaluating_subject = subject
    prompt = evaluator._format_longbench_example(rows[sample_index], tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    position_ids = model._build_position_ids(inputs.get("attention_mask"), inputs["input_ids"])
    with torch.no_grad():
        outputs = model.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            position_ids=position_ids,
            use_cache=True,
        )

    cache = model._to_legacy_cache(outputs.past_key_values)
    num_layers = len(cache)
    collected = {
        component: {stat: [] for stat in ("min", "max", "abs_max", "abs_p99")}
        for component in ("key", "value")
    }
    band_abs_max = {component: [] for component in ("key", "value")}

    with torch.no_grad():
        for layer_index, (key, value) in enumerate(cache):
            layer_bands = {"key": [], "value": []}
            for component, tensor in (("key", key), ("value", value)):
                tensor = tensor.detach().to(DEVICE)
                anchors = _anchor_indices(tensor.shape[2], codec_cfg, DEVICE)
                body = _body_indices(tensor.shape[2], anchors)
                coefficients, transform = _forward_dct(tensor.index_select(2, body), codec_cfg)
                table_np = build_frequency_table(
                    int(transform["freq_len"]), codec_cfg,
                    layer_idx=layer_index, num_layers=num_layers, kv_type=component,
                )
                table = torch.from_numpy(table_np).to(DEVICE)
                quantized, _ = _quantize(coefficients, table, codec_cfg, transform)
                for stat, values in _dimension_stats(quantized).items():
                    collected[component][stat].append(values)
                freq_axis = 3 if transform["mode"] == "fixed" else 2
                for band in BANDS:
                    slc = frequency_band_slice(int(transform["freq_len"]), band)
                    index = [slice(None)] * quantized.ndim
                    index[freq_axis] = slc
                    band_values = quantized[tuple(index)].abs()
                    reduce_dims = tuple(i for i in range(band_values.ndim) if i != band_values.ndim - 1)
                    layer_bands[component].append(
                        band_values.amax(dim=reduce_dims).cpu().numpy()
                        if band_values.numel() else np.zeros(quantized.shape[-1], dtype=np.float32)
                    )
            for component in ("key", "value"):
                band_abs_max[component].append(np.stack(layer_bands[component]))

    arrays = {}
    for component in ("key", "value"):
        for stat, rows_ in collected[component].items():
            arrays[f"{component}_{stat}"] = np.stack(rows_)
        arrays[f"{component}_band_abs_max"] = np.stack(band_abs_max[component])
    np.savez_compressed(OUT_DIR / "quantized_dct_ranges.npz", **arrays)

    metadata = {
        "dataset": subject,
        "sample_index": sample_index,
        "sample_id": rows[sample_index]["_id"],
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "device": str(DEVICE),
        "num_layers": num_layers,
        "num_heads": int(cache[0][0].shape[1]),
        "head_dimension": int(cache[0][0].shape[-1]),
        "quant_dtype": "int16",
        "bands": list(BANDS),
        "block_mode": codec_cfg.block.mode,
        "quantization": {
            "q_global": codec_cfg.quant.q_global,
            "low": codec_cfg.quant.low,
            "high": codec_cfg.quant.high,
            "curve": codec_cfg.quant.curve,
        },
    }
    (OUT_DIR / "quantized_dct_ranges.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
