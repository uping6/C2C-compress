#!/usr/bin/env python3
"""Measure centered SVD cumulative energy of Sharer K/V channel spaces.

For every selected layer, cache tensors are rearranged as
``[B,H,S,D] -> [B,S,H*D]``.  Tokens collected from different prompts are
treated as observations of the same H*D-dimensional channel space.  The
script accumulates centered covariance matrices and obtains the squared
singular-value spectrum through ``eigvalsh`` without retaining activations.

K is captured before RoPE, matching the input used by the concat LCF path;
V comes from the model's ordinary KV cache.  If the total LCF transport
dimension is R, K and V are evaluated at R/2 independently.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-svd-energy")
import matplotlib.pyplot as plt  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = REPO_ROOT / "local/models/Qwen2.5-1.5B-Instruct"
DEFAULT_ARROW = (
    REPO_ROOT
    / "tmp/hf_datasets_mmlu/json/default-114c48bb05eeb75d/0.0.0"
    / "2752a09ea3d59feeb3ad5ea2af11086f41ecd725fd528c98a89d75f546aba397"
    / "json-test.arrow"
)
DEFAULT_OUTPUT = REPO_ROOT / "tmp/SVD_energy/Qwen2.5-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-arrow", type=Path, default=DEFAULT_ARROW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 14, 27])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--tokens-per-sample", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-r",
        type=int,
        default=128,
        help="Total K+V transport width; each of K and V is evaluated at R/2.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    args = parser.parse_args()
    if args.num_samples <= 0 or args.tokens_per_sample <= 0 or args.batch_size <= 0:
        parser.error("sample, token, and batch counts must be positive")
    if args.total_r <= 0 or args.total_r % 2:
        parser.error("--total-r must be a positive even integer")
    return args


@dataclass
class RunningSecondMoment:
    """Float64 sufficient statistics for a centered channel covariance."""

    channels: int

    def __post_init__(self) -> None:
        self.count = 0
        self.sum = torch.zeros(self.channels, dtype=torch.float64)
        self.sum_outer = torch.zeros(
            (self.channels, self.channels), dtype=torch.float64
        )

    def update(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2 or observations.shape[1] != self.channels:
            raise ValueError(
                f"Expected observations [N,{self.channels}], got {tuple(observations.shape)}"
            )
        values = observations.detach().to(device="cpu", dtype=torch.float64)
        self.count += int(values.shape[0])
        self.sum += values.sum(dim=0)
        self.sum_outer += values.T @ values

    def centered_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise ValueError("At least two observations are required for centered SVD")
        scatter = self.sum_outer - torch.outer(self.sum, self.sum) / self.count
        scatter = 0.5 * (scatter + scatter.T)
        eigenvalues = torch.linalg.eigvalsh(scatter).clamp_min(0).flip(0)
        total = eigenvalues.sum()
        if not bool(total > 0):
            raise ValueError("Centered channel energy is zero")
        cumulative = torch.cumsum(eigenvalues, dim=0) / total
        return eigenvalues.numpy(), cumulative.numpy()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[value]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 inference is not supported reliably on CPU")
    return dtype


def format_record(record: dict[str, Any]) -> str:
    if "ctx" in record and "endings" in record:
        labels = [chr(ord("A") + index) for index in range(len(record["endings"]))]
        choices = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, record["endings"])
        )
        return f"Question: {record['ctx']}\n\nChoices:\n{choices}\n\nSelect the correct answer."
    if "question" in record and "choices" in record:
        labels = [chr(ord("A") + index) for index in range(len(record["choices"]))]
        choices = "\n".join(
            f"{label}. {choice}" for label, choice in zip(labels, record["choices"])
        )
        return f"Question: {record['question']}\n\nChoices:\n{choices}\n\nSelect the correct answer."
    for field in ("prompt", "text", "input"):
        if field in record:
            return str(record[field])
    raise ValueError(f"Cannot construct a prompt from fields: {sorted(record)}")


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def cache_key_values(cache: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if hasattr(cache, "key_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    return [pair[0] for pair in legacy], [pair[1] for pair in legacy]


def uniform_valid_positions(attention_row: torch.Tensor, count: int) -> torch.Tensor:
    valid = attention_row.nonzero(as_tuple=False).flatten()
    if valid.numel() < count:
        raise ValueError(
            f"Sequence contains {valid.numel()} valid tokens, fewer than requested {count}"
        )
    offsets = torch.floor(
        (torch.arange(count, device=valid.device, dtype=torch.float64) + 0.5)
        * valid.numel()
        / count
    ).long()
    return valid[offsets]


def install_pre_rope_hooks(
    model: Any, layers: list[int], captured: dict[int, torch.Tensor]
) -> list[Any]:
    handles = []
    decoder_layers = model.model.layers
    for layer_index in layers:
        attention = decoder_layers[layer_index].self_attn
        head_dim = int(attention.head_dim)
        key_norm = getattr(attention, "k_norm", None)

        def hook(
            _module: Any,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
            *,
            index: int = layer_index,
            dim: int = head_dim,
            norm: Any = key_norm,
        ) -> None:
            batch, sequence, channels = output.shape
            if channels % dim:
                raise ValueError(
                    f"Layer {index} k_proj channels {channels} are not divisible by head_dim {dim}"
                )
            key = output.reshape(batch, sequence, channels // dim, dim).transpose(1, 2)
            if norm is not None:
                key = norm(key)
            captured[index] = key.detach()

        handles.append(attention.k_proj.register_forward_hook(hook))
    return handles


def threshold_rank(cumulative: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def summarize_spectrum(
    eigenvalues: np.ndarray,
    cumulative: np.ndarray,
    *,
    layer: int,
    component: str,
    observations: int,
    half_r: int,
) -> dict[str, Any]:
    probabilities = eigenvalues / eigenvalues.sum()
    nonzero = probabilities > 0
    effective_rank = math.exp(float(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum()))

    def energy_at(rank: int) -> float:
        return float(cumulative[min(rank, len(cumulative)) - 1])

    return {
        "layer": layer,
        "component": component,
        "observations": observations,
        "input_dimension_hd": int(len(cumulative)),
        "allocated_dimension_r_over_2": half_r,
        "energy_at_r_over_2": energy_at(half_r),
        "energy_at_16": energy_at(16),
        "energy_at_32": energy_at(32),
        "energy_at_64": energy_at(64),
        "energy_at_128": energy_at(128),
        "rank_90": threshold_rank(cumulative, 0.90),
        "rank_95": threshold_rank(cumulative, 0.95),
        "rank_99": threshold_rank(cumulative, 0.99),
        "effective_rank": effective_rank,
    }


def plot_component(
    spectra: dict[tuple[int, str], np.ndarray],
    layers: list[int],
    component: str,
    half_r: int,
    output_path: Path,
) -> None:
    max_rank = max(len(spectra[(layer, component)]) for layer in layers)
    break_rank = min(128, max_rank - 1)
    fig, (ax_low, ax_high) = plt.subplots(
        1,
        2,
        sharey=True,
        figsize=(8.4, 5.2),
        gridspec_kw={"width_ratios": (4, 1), "wspace": 0.05},
    )
    for layer in layers:
        cumulative = spectra[(layer, component)]
        ranks = np.arange(1, len(cumulative) + 1)
        ax_low.plot(ranks, cumulative, linewidth=2, label=f"Layer {layer}")
        ax_high.plot(ranks, cumulative, linewidth=2)
    ax_low.axvline(
        half_r,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"R/2 = {half_r}",
    )
    for threshold in (0.90, 0.95, 0.99):
        ax_low.axhline(threshold, color="gray", linestyle=":", linewidth=0.8)
        ax_high.axhline(threshold, color="gray", linestyle=":", linewidth=0.8)

    # Both panels remain linear. The 128..HD interval receives one quarter of
    # the horizontal space so the practically relevant low-rank region is
    # enlarged without distorting distances within either segment.
    ax_low.set_xlim(1, break_rank)
    ax_high.set_xlim(break_rank, max_rank)
    ax_low.set_xticks([1, 32, 64, 96, break_rank])
    ax_high.set_xticks([160, 208, max_rank])
    ax_low.set_ylim(0, 1.005)
    ax_low.set_ylabel("Centered cumulative energy")
    fig.supxlabel(
        f"Number of retained singular dimensions (piecewise linear; input HD = {max_rank})",
        y=0.01,
    )
    fig.suptitle(
        f"Qwen2.5-1.5B-Instruct — {component} channel-space SVD",
        y=0.98,
    )
    for ax in (ax_low, ax_high):
        ax.grid(alpha=0.2)

    ax_low.spines["right"].set_visible(False)
    ax_high.spines["left"].set_visible(False)
    ax_high.tick_params(axis="y", left=False, labelleft=False)
    break_mark = 0.012
    mark_kwargs = {"color": "black", "clip_on": False, "linewidth": 1.0}
    ax_low.plot(
        (1 - break_mark, 1 + break_mark),
        (-break_mark, +break_mark),
        transform=ax_low.transAxes,
        **mark_kwargs,
    )
    ax_low.plot(
        (1 - break_mark, 1 + break_mark),
        (1 - break_mark, 1 + break_mark),
        transform=ax_low.transAxes,
        **mark_kwargs,
    )
    ax_high.plot(
        (-break_mark, +break_mark),
        (-break_mark, +break_mark),
        transform=ax_high.transAxes,
        **mark_kwargs,
    )
    ax_high.plot(
        (-break_mark, +break_mark),
        (1 - break_mark, 1 + break_mark),
        transform=ax_high.transAxes,
        **mark_kwargs,
    )
    ax_low.legend(loc="lower right")
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.91)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    dataset_path = args.dataset_arrow.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset Arrow file not found: {dataset_path}")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(device)

    num_layers = int(model.config.num_hidden_layers)
    layers = sorted(set(int(layer) for layer in args.layers))
    invalid = [layer for layer in layers if not 0 <= layer < num_layers]
    if invalid:
        raise ValueError(f"Layers outside [0,{num_layers - 1}]: {invalid}")
    num_heads = int(model.config.num_key_value_heads)
    head_dim = int(model.model.layers[0].self_attn.head_dim)
    channels = num_heads * head_dim
    half_r = args.total_r // 2
    if half_r > channels:
        raise ValueError(f"R/2={half_r} exceeds H*D={channels}")

    dataset = Dataset.from_file(str(dataset_path))
    sample_count = min(args.num_samples, len(dataset))
    rng = random.Random(args.seed)
    selected_indices = sorted(rng.sample(range(len(dataset)), sample_count))
    prompts = [apply_chat_template(tokenizer, format_record(dataset[index])) for index in selected_indices]

    accumulators = {
        (layer, component): RunningSecondMoment(channels)
        for layer in layers
        for component in ("K", "V")
    }
    captured: dict[int, torch.Tensor] = {}
    handles = install_pre_rope_hooks(model, layers, captured)
    processed_indices: list[int] = []
    try:
        with torch.inference_mode():
            for start in range(0, sample_count, args.batch_size):
                batch_prompts = prompts[start : start + args.batch_size]
                batch_indices = selected_indices[start : start + args.batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    add_special_tokens=False,
                ).to(device)
                captured.clear()
                output = model(**encoded, use_cache=True, return_dict=True)
                _, values = cache_key_values(output.past_key_values)
                if set(captured) != set(layers):
                    raise RuntimeError(
                        f"Captured pre-RoPE K for layers {sorted(captured)}, expected {layers}"
                    )

                for batch_row, dataset_index in enumerate(batch_indices):
                    try:
                        positions = uniform_valid_positions(
                            encoded["attention_mask"][batch_row], args.tokens_per_sample
                        )
                    except ValueError as error:
                        raise ValueError(f"Dataset index {dataset_index}: {error}") from error
                    for layer in layers:
                        key = captured[layer][batch_row, :, positions, :]
                        value = values[layer][batch_row, :, positions, :]
                        # [H,T,D] -> [T,H*D], preserving head-major channel order.
                        key_rows = key.permute(1, 0, 2).reshape(len(positions), channels)
                        value_rows = value.permute(1, 0, 2).reshape(len(positions), channels)
                        accumulators[(layer, "K")].update(key_rows)
                        accumulators[(layer, "V")].update(value_rows)
                    processed_indices.append(dataset_index)
                print(
                    f"Processed {min(start + len(batch_prompts), sample_count)}/{sample_count} samples",
                    flush=True,
                )
                del output
    finally:
        for handle in handles:
            handle.remove()

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    cumulative_spectra: dict[tuple[int, str], np.ndarray] = {}
    npz_values: dict[str, np.ndarray] = {}
    for layer in layers:
        for component in ("K", "V"):
            accumulator = accumulators[(layer, component)]
            eigenvalues, cumulative = accumulator.centered_spectrum()
            cumulative_spectra[(layer, component)] = cumulative
            npz_values[f"layer_{layer}_{component.lower()}_eigenvalues"] = eigenvalues
            npz_values[f"layer_{layer}_{component.lower()}_cumulative_energy"] = cumulative
            summaries.append(
                summarize_spectrum(
                    eigenvalues,
                    cumulative,
                    layer=layer,
                    component=component,
                    observations=accumulator.count,
                    half_r=half_r,
                )
            )

    metadata = {
        "model": str(model_path),
        "dataset_arrow": str(dataset_path),
        "dataset_rows": len(dataset),
        "selected_dataset_indices": selected_indices,
        "processed_dataset_indices": processed_indices,
        "num_samples": sample_count,
        "tokens_per_sample": args.tokens_per_sample,
        "observations_per_layer": sample_count * args.tokens_per_sample,
        "max_length": args.max_length,
        "seed": args.seed,
        "layers": layers,
        "num_model_layers": num_layers,
        "num_kv_heads": num_heads,
        "head_dim": head_dim,
        "channel_dimension_hd": channels,
        "total_lcf_dimension_r": args.total_r,
        "per_component_dimension_r_over_2": half_r,
        "centering": "global per-layer per-component channel mean",
        "key_representation": "pre-RoPE K after optional k_norm",
        "value_representation": "ordinary cache V",
        "token_sampling": "uniform centers of equal-width bins over valid prompt tokens",
        "device": str(device),
        "model_dtype": str(dtype),
        "energy_definition": "sum of top-r centered squared singular values / total centered squared singular values",
    }
    result = {"metadata": metadata, "layers": summaries}
    (output_dir / "svd_energy_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "svd_energy_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    np.savez_compressed(output_dir / "svd_energy_spectra.npz", **npz_values)
    plot_component(cumulative_spectra, layers, "K", half_r, output_dir / "k_cumulative_energy.png")
    plot_component(cumulative_spectra, layers, "V", half_r, output_dir / "v_cumulative_energy.png")
    print(f"Saved centered SVD energy statistics to {output_dir}")


if __name__ == "__main__":
    main()
