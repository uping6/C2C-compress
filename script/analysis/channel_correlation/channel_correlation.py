#!/usr/bin/env python3
"""Plot channel- and head-level Pearson correlation for transformer KV caches.

The cache layout is kept as ``head 0: dim 0..D-1, head 1: ...``.  A run
produces separate K/V channel heatmaps, separate absolute head-level heatmaps,
the raw matrices, and machine-readable summary statistics.
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-channel-correlation")
import matplotlib.pyplot as plt  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


DEFAULT_PROMPTS = [
    "Explain why the sky appears blue during the day and red near sunset.",
    "Compare renewable energy with fossil fuels in terms of cost and reliability.",
    "A train travels 180 kilometers in 2.5 hours. Compute its average speed.",
    "Summarize the main causes and consequences of the Industrial Revolution.",
    "Write a short algorithm for finding duplicate values in a list.",
    "What evidence would distinguish correlation from causation in an experiment?",
    "Describe how attention mechanisms help a language model use context.",
    "Give three practical ways a city can reduce traffic congestion.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", type=Path, help="Local Hugging Face model directory.")
    source.add_argument(
        "--input-npz",
        type=Path,
        help="NPZ containing X_K and X_V [observations, heads*head_dim].",
    )
    parser.add_argument("--layer", type=int, default=None, help="Cache layer; default: middle layer.")
    parser.add_argument("--prompts-file", type=Path, help="One prompt per line, or a JSONL file.")
    parser.add_argument(
        "--dataset",
        choices=["prompts", "mmlu"],
        default="prompts",
        help="Input source for model collection.",
    )
    parser.add_argument(
        "--mmlu-arrow",
        type=Path,
        help="Cached cais/mmlu Arrow split. If omitted, search common Hugging Face caches.",
    )
    parser.add_argument("--mmlu-num-samples", type=int, default=100)
    parser.add_argument("--mmlu-tokens-per-sample", type=int, default=32)
    parser.add_argument("--mmlu-seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/channel_correlation"))
    parser.add_argument("--title-prefix", default="")
    args = parser.parse_args()
    if args.model is None and args.input_npz is None:
        parser.error("one of --model or --input-npz is required")
    if args.input_npz and args.dataset != "prompts":
        parser.error("--dataset only applies with --model")
    if args.mmlu_num_samples <= 0 or args.mmlu_tokens_per_sample <= 0:
        parser.error("MMLU sample and token counts must be positive")
    return args


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if path.suffix == ".jsonl":
            record = json.loads(line)
            value = record.get("prompt", record.get("text", record.get("input")))
            if value is None:
                raise ValueError("JSONL rows must contain prompt, text, or input")
            prompts.append(str(value))
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def cache_layers(cache: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    if hasattr(cache, "key_cache"):
        return list(cache.key_cache), list(cache.value_cache)
    legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
    return [item[0] for item in legacy], [item[1] for item in legacy]


def find_mmlu_arrow(explicit_path: Path | None, split: str = "test") -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MMLU Arrow file not found: {path}")
        return path
    patterns = [
        f"~/.cache/huggingface/datasets/cais___mmlu/all/*/*/mmlu-{split}.arrow",
        f"/home/*/.cache/huggingface/datasets/cais___mmlu/all/*/*/mmlu-{split}.arrow",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(item) for item in glob.glob(os.path.expanduser(pattern)))
    if not candidates:
        raise FileNotFoundError(
            "Could not find cached cais/mmlu/all Arrow data; pass --mmlu-arrow explicitly"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_mmlu_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, list[int]]:
    from datasets import Dataset

    arrow_path = find_mmlu_arrow(args.mmlu_arrow)
    dataset = Dataset.from_file(str(arrow_path))
    count = min(args.mmlu_num_samples, len(dataset))
    rng = np.random.default_rng(args.mmlu_seed)
    indices = np.sort(rng.choice(len(dataset), size=count, replace=False)).tolist()
    return [dataset[index] for index in indices], arrow_path, indices


def mmlu_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    labels = [chr(ord("A") + index) for index in range(len(sample["choices"]))]
    choices = "\n".join(
        f"{label}. {choice}" for label, choice in zip(labels, sample["choices"])
    )
    content = (
        f"Question: {sample['question']}\n\n"
        f"Choices:\n{choices}\n\n"
        "Select the correct answer."
    )
    return [{"role": "user", "content": content}]


def uniformly_spaced_token_indices(length: int, count: int) -> np.ndarray:
    """Select one token at the center of each of ``count`` equal-width bins."""
    if length < count:
        raise ValueError(f"sequence has {length} tokens, fewer than requested {count}")
    indices = np.floor((np.arange(count, dtype=np.float64) + 0.5) * length / count).astype(int)
    if len(np.unique(indices)) != count or indices[0] < 0 or indices[-1] >= length:
        raise AssertionError("uniform token sampling produced invalid indices")
    return indices


def collect_from_model(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to(args.device)
    num_layers = int(model.config.num_hidden_layers)
    layer = num_layers // 2 if args.layer is None else args.layer
    if not -num_layers <= layer < num_layers:
        raise ValueError(f"layer {layer} is outside [-{num_layers}, {num_layers - 1}]")
    layer %= num_layers

    k_rows: list[np.ndarray] = []
    v_rows: list[np.ndarray] = []
    lengths: list[int] = []
    original_lengths: list[int] = []
    if args.dataset == "mmlu":
        samples, mmlu_arrow, dataset_indices = load_mmlu_samples(args)
        records = [(mmlu_messages(sample), sample) for sample in samples]
    else:
        prompts = load_prompts(args.prompts_file)
        records = [([{"role": "user", "content": prompt}], None) for prompt in prompts]
        mmlu_arrow = None
        dataset_indices = []
    selected_token_indices: list[list[int]] = []
    subjects: list[str] = []
    with torch.inference_mode():
        for sample_number, (messages, sample) in enumerate(records):
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = messages[0]["content"]
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                add_special_tokens=True,
            ).to(args.device)
            output = model(**inputs, use_cache=True, return_dict=True)
            keys, values = cache_layers(output.past_key_values)
            key = keys[layer][0].detach().float().cpu()    # [H, T, D]
            value = values[layer][0].detach().float().cpu()
            if key.ndim != 3 or key.shape != value.shape:
                raise ValueError(f"Expected matching [H,T,D] K/V, got {key.shape}, {value.shape}")
            original_lengths.append(int(key.shape[1]))
            if args.dataset == "mmlu":
                try:
                    token_indices = uniformly_spaced_token_indices(
                        key.shape[1], args.mmlu_tokens_per_sample
                    )
                except ValueError as error:
                    raise ValueError(
                        f"MMLU dataset index {dataset_indices[sample_number]}: {error}. "
                        "Reduce --mmlu-tokens-per-sample."
                    ) from error
                key = key[:, token_indices, :]
                value = value[:, token_indices, :]
                selected_token_indices.append(token_indices.tolist())
                subjects.append(str(sample.get("subject", "unknown")))
                if (sample_number + 1) % 10 == 0 or sample_number + 1 == len(records):
                    print(f"Collected MMLU samples: {sample_number + 1}/{len(records)}", flush=True)
            # T,H,D -> T,HD keeps dimensions grouped within each head.
            k_rows.append(key.permute(1, 0, 2).reshape(key.shape[1], -1).numpy())
            v_rows.append(value.permute(1, 0, 2).reshape(value.shape[1], -1).numpy())
            lengths.append(int(key.shape[1]))

    head_dim = int(k_rows[0].shape[1] // key.shape[0])
    metadata = {
        "source": "model",
        "model": str(args.model.resolve()),
        "layer": layer,
        "num_layers": num_layers,
        "dataset": args.dataset,
        "num_samples": len(records),
        "sampled_tokens_per_sample": lengths,
        "input_tokens_per_sample": original_lengths,
        "max_length": args.max_length,
        "device": args.device,
        "dtype": str(next(model.parameters()).dtype),
    }
    if args.dataset == "mmlu":
        metadata.update({
            "dataset_name": "cais/mmlu",
            "dataset_config": "all",
            "dataset_split": "test",
            "dataset_arrow": str(mmlu_arrow),
            "dataset_sampling": "uniform random without replacement",
            "dataset_seed": args.mmlu_seed,
            "dataset_indices": dataset_indices,
            "subject_counts": dict(sorted(Counter(subjects).items())),
            "tokens_per_sample_requested": args.mmlu_tokens_per_sample,
            "token_sampling": "one token at the center of each equal-width sequence bin",
            "selected_token_indices": selected_token_indices,
            "total_sampled_tokens": len(records) * args.mmlu_tokens_per_sample,
        })
    return np.concatenate(k_rows), np.concatenate(v_rows), head_dim, metadata


def collect_from_npz(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    data = np.load(args.input_npz)
    if "X_K" not in data or "X_V" not in data:
        raise ValueError("Input NPZ must contain X_K and X_V")
    x_k, x_v = np.asarray(data["X_K"]), np.asarray(data["X_V"])
    if x_k.ndim != 2 or x_k.shape != x_v.shape:
        raise ValueError(f"X_K and X_V must have matching [M,HD] shapes, got {x_k.shape}, {x_v.shape}")
    if "head_dim" not in data:
        raise ValueError("Input NPZ must contain scalar head_dim")
    head_dim = int(np.asarray(data["head_dim"]).item())
    if head_dim <= 0 or x_k.shape[1] % head_dim:
        raise ValueError("head_dim must be positive and divide the channel count")
    return x_k, x_v, head_dim, {
        "source": "npz",
        "input_npz": str(args.input_npz.resolve()),
        "layer": args.layer,
    }


def pearson_and_stats(x: np.ndarray, head_dim: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        raise ValueError("At least two observations are required")
    centered = x - x.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0)
    stable = norms > np.finfo(np.float64).eps * max(1.0, float(norms.max(initial=0.0)))
    normalized = np.zeros_like(centered)
    normalized[:, stable] = centered[:, stable] / norms[stable]
    corr = np.clip(normalized.T @ normalized, -1.0, 1.0)
    np.fill_diagonal(corr, np.where(stable, 1.0, 0.0))

    channels = x.shape[1]
    heads = channels // head_dim
    valid_pairs = stable[:, None] & stable[None, :]
    off_diagonal = ~np.eye(channels, dtype=bool)
    pair_mask = valid_pairs & off_diagonal
    abs_pairs = np.abs(corr[pair_mask])
    if not abs_pairs.size:
        raise ValueError("No valid non-diagonal channel pairs")

    blocks = np.abs(corr).reshape(heads, head_dim, heads, head_dim)
    head_corr = blocks.mean(axis=(1, 3))
    within_mask = pair_mask & (
        np.arange(channels)[:, None] // head_dim == np.arange(channels)[None, :] // head_dim
    )
    cross_mask = pair_mask & ~(
        np.arange(channels)[:, None] // head_dim == np.arange(channels)[None, :] // head_dim
    )
    stats = {
        "observations": int(x.shape[0]),
        "channels": channels,
        "num_heads": heads,
        "head_dim": head_dim,
        "constant_channels": int((~stable).sum()),
        "ordered_off_diagonal_pairs": int(pair_mask.sum()),
        "mean_absolute_correlation": float(abs_pairs.mean()),
        "ratio_abs_correlation_gt_0_5": float((abs_pairs > 0.5).mean()),
        "ratio_abs_correlation_gt_0_8": float((abs_pairs > 0.8).mean()),
        "mean_absolute_within_head": float(np.abs(corr[within_mask]).mean()),
        "mean_absolute_cross_head": float(np.abs(corr[cross_mask]).mean()),
        "head_level_absolute_correlation": head_corr.tolist(),
    }
    return corr, head_corr, stats


def cross_head_dimension_matching(
    corr: np.ndarray, head_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Measure whether every dimension has a strongly correlated peer in another head.

    Returns directional best matches [H,H,D], a symmetric matched head matrix
    [H,H], and each source dimension's best match in any other head [H,D].
    """
    channels = corr.shape[0]
    heads = channels // head_dim
    if heads < 2:
        raise ValueError("Cross-head dimension matching requires at least two KV heads")
    blocks = np.abs(corr).reshape(heads, head_dim, heads, head_dim)
    # directional[h1,h2,d1] = max_d2 |rho((h1,d1),(h2,d2))|
    directional = blocks.max(axis=3).transpose(0, 2, 1)
    for head in range(heads):
        directional[head, head, :] = np.nan

    matched_head = np.full((heads, heads), np.nan, dtype=np.float64)
    for source in range(heads):
        for target in range(source + 1, heads):
            score = 0.5 * (
                np.nanmean(directional[source, target])
                + np.nanmean(directional[target, source])
            )
            matched_head[source, target] = score
            matched_head[target, source] = score

    best_any_head = np.nanmax(directional, axis=1)
    values = best_any_head.reshape(-1)
    stats = {
        "definition": "For each (head, dimension), max |rho| over every dimension in other heads.",
        "mean_best_cross_head_match": float(values.mean()),
        "median_best_cross_head_match": float(np.median(values)),
        "ratio_best_cross_head_match_gt_0_5": float((values > 0.5).mean()),
        "ratio_best_cross_head_match_gt_0_8": float((values > 0.8).mean()),
        "matched_head_level_correlation": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matched_head
        ],
    }
    return directional, matched_head, best_any_head, stats


def plot_channel(corr: np.ndarray, head_dim: int, component: str, path: Path, prefix: str) -> None:
    channels = corr.shape[0]
    heads = channels // head_dim
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
    for boundary in range(head_dim, channels, head_dim):
        ax.axhline(boundary - 0.5, color="black", lw=0.75, ls="--", alpha=0.75)
        ax.axvline(boundary - 0.5, color="black", lw=0.75, ls="--", alpha=0.75)
    ticks = np.arange(head_dim / 2 - 0.5, channels, head_dim)
    ax.set_xticks(ticks, [f"head {i}" for i in range(heads)], rotation=35, ha="right")
    ax.set_yticks(ticks, [f"head {i}" for i in range(heads)])
    ax.set_xlabel("Channel (head-major order)")
    ax.set_ylabel("Channel (head-major order)")
    title = f"{prefix + ' — ' if prefix else ''}{component} Channel Pearson Correlation"
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=r"Pearson $\rho$")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heads(head_corr: np.ndarray, component: str, path: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6), layout="constrained")
    image = ax.imshow(head_corr, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    heads = head_corr.shape[0]
    ax.set_xticks(range(heads), [str(i) for i in range(heads)])
    ax.set_yticks(range(heads), [str(i) for i in range(heads)])
    ax.set_xlabel("Head")
    ax.set_ylabel("Head")
    ax.set_title(
        f"{prefix + ' — ' if prefix else ''}{component} Mean |Channel Correlation|",
        fontsize=11,
        pad=10,
    )
    if heads <= 12:
        for row in range(heads):
            for col in range(heads):
                color = "white" if head_corr[row, col] < 0.5 else "black"
                ax.text(col, row, f"{head_corr[row, col]:.3f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label=r"mean $|\rho|$")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cross_head_dimension_block(
    corr: np.ndarray, head_dim: int, component: str, path: Path, prefix: str
) -> None:
    """Plot all D x D cross-head blocks; upper-triangle head pairs only."""
    heads = corr.shape[0] // head_dim
    pairs = [(left, right) for left in range(heads) for right in range(left + 1, heads)]
    columns = min(3, len(pairs))
    rows = (len(pairs) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.7 * rows),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for ax, (left, right) in zip(axes.flat, pairs):
        block = np.abs(
            corr[
                left * head_dim : (left + 1) * head_dim,
                right * head_dim : (right + 1) * head_dim,
            ]
        )
        image = ax.imshow(block, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"head {left} dimensions vs head {right} dimensions")
        ax.set_xlabel(f"head {right} dimension")
        ax.set_ylabel(f"head {left} dimension")
    for ax in axes.flat[len(pairs) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"{prefix + ' — ' if prefix else ''}{component} Cross-head Dimension Correlation",
        fontsize=12,
    )
    if image is not None:
        fig.colorbar(image, ax=list(axes.flat), label=r"$|\rho|$", shrink=0.85)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_dimension_matches(
    best: np.ndarray, component: str, path: Path, prefix: str
) -> None:
    heads, head_dim = best.shape
    fig, ax = plt.subplots(
        figsize=(max(8.0, head_dim / 14), max(2.8, heads * 0.65 + 1.8)),
        layout="constrained",
    )
    image = ax.imshow(best, cmap="magma", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(heads), [f"head {head}" for head in range(heads)])
    ax.set_xlabel("Source dimension")
    ax.set_ylabel("Source head")
    ax.set_title(
        f"{prefix + ' — ' if prefix else ''}{component}: Best Correlated Dimension in Another Head",
        fontsize=11,
    )
    fig.colorbar(image, ax=ax, label=r"$\max_{h'\ne h,d'} |\rho_{(h,d),(h',d')}|$")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_matched_heads(
    matched: np.ndarray, component: str, path: Path, prefix: str
) -> None:
    heads = matched.shape[0]
    masked = np.ma.masked_invalid(matched)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#e6e6e6")
    fig, ax = plt.subplots(figsize=(6.8, 5.6), layout="constrained")
    image = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(heads), [str(i) for i in range(heads)])
    ax.set_yticks(range(heads), [str(i) for i in range(heads)])
    ax.set_xlabel("Target head")
    ax.set_ylabel("Source head")
    ax.set_title(
        f"{prefix + ' — ' if prefix else ''}{component} Cross-head Dimension-matched Correlation",
        fontsize=11,
    )
    for row in range(heads):
        for col in range(heads):
            if np.isfinite(matched[row, col]):
                color = "white" if matched[row, col] < 0.5 else "black"
                ax.text(col, row, f"{matched[row, col]:.3f}", ha="center", va="center", color=color)
            elif row == col:
                ax.text(col, row, "same\nhead", ha="center", va="center", color="#666666")
    fig.colorbar(image, ax=ax, label="mean best dimension match")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.input_npz:
        x_k, x_v, head_dim, metadata = collect_from_npz(args)
    else:
        x_k, x_v, head_dim, metadata = collect_from_model(args)
    if x_k.shape[1] % head_dim:
        raise ValueError("Channel count is not divisible by head_dim")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrices: dict[str, np.ndarray] = {}
    results: dict[str, Any] = {"metadata": metadata}
    for component, x in (("K", x_k), ("V", x_v)):
        corr, head_corr, stats = pearson_and_stats(x, head_dim)
        directional, matched_head, best_dimensions, matching_stats = cross_head_dimension_matching(
            corr, head_dim
        )
        matrices[f"R_{component}"] = corr.astype(np.float32)
        matrices[f"R_{component}_head"] = head_corr.astype(np.float32)
        matrices[f"R_{component}_cross_head_best_directional"] = directional.astype(np.float32)
        matrices[f"R_{component}_head_dimension_matched"] = matched_head.astype(np.float32)
        matrices[f"R_{component}_best_cross_head_by_dimension"] = best_dimensions.astype(np.float32)
        stats["cross_head_dimension_matching"] = matching_stats
        results[component] = stats
        stem = component.lower()
        plot_channel(corr, head_dim, component, args.output_dir / f"{stem}_channel_correlation.png", args.title_prefix)
        plot_heads(head_corr, component, args.output_dir / f"{stem}_head_correlation.png", args.title_prefix)
        plot_cross_head_dimension_block(
            corr,
            head_dim,
            component,
            args.output_dir / f"{stem}_cross_head_dimension_correlation.png",
            args.title_prefix,
        )
        plot_best_dimension_matches(
            best_dimensions,
            component,
            args.output_dir / f"{stem}_best_cross_head_match_by_dimension.png",
            args.title_prefix,
        )
        plot_matched_heads(
            matched_head,
            component,
            args.output_dir / f"{stem}_head_dimension_matched_correlation.png",
            args.title_prefix,
        )

    np.savez_compressed(args.output_dir / "correlation_matrices.npz", **matrices, head_dim=head_dim)
    (args.output_dir / "correlation_stats.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for component in ("K", "V"):
        stats = results[component]
        print(
            f"{component}: mean|rho|={stats['mean_absolute_correlation']:.6f}, "
            f"P(|rho|>0.5)={stats['ratio_abs_correlation_gt_0_5']:.6f}, "
            f"P(|rho|>0.8)={stats['ratio_abs_correlation_gt_0_8']:.6f}, "
            f"within-head={stats['mean_absolute_within_head']:.6f}, "
            f"cross-head={stats['mean_absolute_cross_head']:.6f}"
        )
        matching = stats["cross_head_dimension_matching"]
        print(
            f"{component} cross-head dimension matching: "
            f"mean-best={matching['mean_best_cross_head_match']:.6f}, "
            f"median-best={matching['median_best_cross_head_match']:.6f}, "
            f"P(best>0.5)={matching['ratio_best_cross_head_match_gt_0_5']:.6f}, "
            f"P(best>0.8)={matching['ratio_best_cross_head_match_gt_0_8']:.6f}"
        )
    print(f"Outputs written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
