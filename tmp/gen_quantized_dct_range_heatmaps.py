from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper_plot_style import save_figure


ROOT = Path(__file__).resolve().parent
DATA = np.load(ROOT / "quantized_dct_ranges.npz")


def signed_range_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True, sharex=True, sharey=True)
    for row, component in enumerate(("key", "value")):
        limit = max(float(np.abs(DATA[f"{component}_{s}"]).max()) for s in ("min", "max"))
        for column, stat in enumerate(("min", "max")):
            ax = axes[row, column]
            sns.heatmap(
                DATA[f"{component}_{stat}"], ax=ax, cmap="coolwarm", center=0,
                vmin=-limit, vmax=limit, xticklabels=16, yticklabels=4,
                cbar_kws={"label": "Quantized coefficient"},
            )
            ax.set_xlabel("Head dimension index")
            ax.set_ylabel("Transformer layer" if column == 0 else "")
            ax.text(0.015, 0.96, f"{component.upper()} · {stat}", transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", color="black")
    save_figure(fig, "quantized_dct_layer_dimension_range_heatmap")


def magnitude_figure():
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.5), constrained_layout=True, sharex=True, sharey=True)
    all_log = [np.log10(1 + DATA[f"{c}_{s}"]) for c in ("key", "value") for s in ("abs_p99", "abs_max")]
    vmin, vmax = min(x.min() for x in all_log), max(x.max() for x in all_log)
    for row, component in enumerate(("key", "value")):
        for column, stat in enumerate(("abs_p99", "abs_max")):
            ax = axes[row, column]
            sns.heatmap(
                np.log10(1 + DATA[f"{component}_{stat}"]), ax=ax, cmap="mako",
                vmin=vmin, vmax=vmax, xticklabels=16, yticklabels=4,
                cbar_kws={"label": r"$\log_{10}(1 + |q|)$"},
            )
            ax.set_xlabel("Head dimension index")
            ax.set_ylabel("Transformer layer" if column == 0 else "")
            label = "P99 |q|" if stat == "abs_p99" else "max |q|"
            ax.text(0.015, 0.96, f"{component.upper()} · {label}", transform=ax.transAxes,
                    ha="left", va="top", fontweight="bold", color="white")
    save_figure(fig, "quantized_dct_layer_dimension_magnitude_heatmap")


def required_width_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True, sharey=True)
    cmap = colors.ListedColormap(["#2a9d8f", "#e9c46a", "#e76f51"])
    norm = colors.BoundaryNorm([0, 1, 2, 3], cmap.N)
    for ax, component in zip(axes, ("key", "value")):
        minimum, maximum = DATA[f"{component}_min"], DATA[f"{component}_max"]
        width_class = np.where((minimum >= -8) & (maximum <= 7), 0,
                              np.where((minimum >= -128) & (maximum <= 127), 1, 2))
        sns.heatmap(width_class, ax=ax, cmap=cmap, norm=norm, cbar=False,
                    xticklabels=16, yticklabels=4)
        ax.set_xlabel("Head dimension index")
        ax.set_ylabel("Transformer layer" if component == "key" else "")
        ax.text(0.015, 0.96, component.upper(), transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    fig.legend(handles, ["INT4", "INT8", "INT16"], loc="outside lower center", ncol=3, frameon=False)
    save_figure(fig, "quantized_dct_layer_dimension_required_bitwidth_heatmap")


signed_range_figure()
magnitude_figure()
required_width_figure()
