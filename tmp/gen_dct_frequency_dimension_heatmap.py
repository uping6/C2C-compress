from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper_plot_style import save_figure


DATA = np.load(Path(__file__).resolve().parent / "dct_binned_coefficients.npz")
LAYERS = DATA["representative_layers"].tolist()
EPSILON = 1e-8

fig, axes = plt.subplots(len(LAYERS), 2, figsize=(8.6, 8.0), constrained_layout=True)
for row, layer in enumerate(LAYERS):
    for column, (component, label) in enumerate((("key", "K"), ("value", "V"))):
        axis = axes[row, column]
        all_values = np.log10(DATA[f"{component}_representative_frequency_dimension"] + EPSILON)
        values = all_values[row]
        sns.heatmap(
            values,
            ax=axis,
            cmap="mako",
            cbar=row == 0,
            vmin=float(all_values.min()),
            vmax=float(all_values.max()),
            cbar_kws={"label": f"{label}: log10(mean |DCT coeff.|), shared across layers"},
            xticklabels=16,
            yticklabels=16,
        )
        axis.set_xlabel("Head dimension")
        axis.set_ylabel("DCT frequency group (low → high)" if column == 0 else "")
        axis.text(
            0.02,
            0.96,
            f"Layer {layer} · {label}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            color="white",
            fontweight="bold",
        )

fig.text(0.5, -0.015, "K and V use independent color scales; colors are not comparable across columns.", ha="center", fontsize=9)
save_figure(fig, "dct_frequency_dimension_heatmap")
