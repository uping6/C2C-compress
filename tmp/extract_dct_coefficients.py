from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.fft import dct

from rosetta.cachejpeg.wrapper import load_cachejpeg_model
from script.evaluation.unified_evaluator import UnifiedEvaluator


PROJECT_ROOT = Path("/data/smy_data")
OUT_DIR = PROJECT_ROOT / "tmp"
DEVICE = torch.device("cuda:1")
NUM_FREQUENCY_BINS = 128


def bin_axis(values: np.ndarray, bins: int, axis: int) -> np.ndarray:
    chunks = np.array_split(values, bins, axis=axis)
    return np.stack([chunk.mean(axis=axis) for chunk in chunks], axis=axis)


def component_statistics(coefficients: np.ndarray) -> dict:
    flattened = coefficients.reshape(-1)
    abs_flattened = np.abs(flattened)
    return {
        "mean": float(flattened.mean()),
        "std": float(flattened.std()),
        "abs_mean": float(abs_flattened.mean()),
        "abs_max": float(abs_flattened.max()),
        "quantiles": {
            str(q): float(np.quantile(flattened, q))
            for q in (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)
        },
        "abs_quantiles": {
            str(q): float(np.quantile(abs_flattened, q))
            for q in (0.5, 0.9, 0.99, 0.999)
        },
    }


def main():
    config = yaml.safe_load((PROJECT_ROOT / "recipe/eval_recipe/longbench_jpegcache.yaml").read_text())
    config["output"]["output_dir"] = "/tmp/cachejpeg_dct_analysis"
    evaluator = UnifiedEvaluator(config)
    model, tokenizer = load_cachejpeg_model(
        evaluator.model_config,
        device=DEVICE,
        generation_config={"max_new_tokens": 1, "do_sample": False},
    )

    subject = "qasper"
    rows = [
        json.loads(line)
        for line in (Path(config["eval"]["longbench_local_data_dir"]) / f"{subject}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sample_index = next(
        index
        for index, row in enumerate(rows)
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

    legacy_cache = model._to_legacy_cache(outputs.past_key_values)
    num_layers = len(legacy_cache)
    representative_layers = sorted({0, num_layers // 2, num_layers - 1})
    layer_frequency = {"key": [], "value": []}
    representative = {"key": [], "value": []}
    layer_stats = []

    for layer_index, (key, value) in enumerate(legacy_cache):
        layer_record = {"layer": layer_index, "components": {}}
        for component_name, tensor in (("key", key), ("value", value)):
            original = tensor.detach().float().cpu().numpy()
            # Match CacheJPEG anchors.sink_count=1: the sink token is stored losslessly,
            # while DCT is applied to the remaining token axis.
            body = original[:, :, 1:, :]
            coefficients = dct(body, axis=2, norm="ortho").astype(np.float32, copy=False)
            frequency_profile = np.mean(np.abs(coefficients), axis=(0, 1, 3))
            layer_frequency[component_name].append(
                bin_axis(frequency_profile, NUM_FREQUENCY_BINS, axis=0)
            )
            if layer_index in representative_layers:
                frequency_dimension = np.mean(np.abs(coefficients), axis=(0, 1))
                representative[component_name].append(
                    bin_axis(frequency_dimension, NUM_FREQUENCY_BINS, axis=0)
                )
            layer_record["components"][component_name] = {
                "original_shape": list(original.shape),
                "body_shape": list(body.shape),
                "dct_shape": list(coefficients.shape),
                **component_statistics(coefficients),
            }
        layer_stats.append(layer_record)

    arrays = {
        "key_layer_frequency": np.stack(layer_frequency["key"]),
        "value_layer_frequency": np.stack(layer_frequency["value"]),
        "key_representative_frequency_dimension": np.stack(representative["key"]),
        "value_representative_frequency_dimension": np.stack(representative["value"]),
        "representative_layers": np.asarray(representative_layers, dtype=np.int64),
    }
    np.savez_compressed(OUT_DIR / "dct_binned_coefficients.npz", **arrays)
    stats = {
        "dataset": subject,
        "sample_index": sample_index,
        "sample_id": rows[sample_index]["_id"],
        "device": str(DEVICE),
        "input_token_count": int(inputs["input_ids"].shape[1]),
        "dct_axis": 2,
        "dct_norm": "ortho",
        "sink_anchor_count": 1,
        "num_layers": num_layers,
        "num_frequency_bins": NUM_FREQUENCY_BINS,
        "representative_layers": representative_layers,
        "layers": layer_stats,
    }
    (OUT_DIR / "dct_coefficient_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in stats.items() if key != "layers"}, indent=2))
    print("first_layer_shapes", stats["layers"][0]["components"])


if __name__ == "__main__":
    main()
