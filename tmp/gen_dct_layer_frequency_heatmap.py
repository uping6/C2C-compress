from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from paper_plot_style import save_figure


DATA = np.load(Path(__file__).resolve().parent / "dct_binned_coefficients.npz")
EPSILON = 1e-8

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True, sharey=True)
for axis, component, label in zip(axes, ("key", "value"), ("K", "V")):
    values = np.log10(DATA[f"{component}_layer_frequency"] + EPSILON)
    sns.heatmap(
        values,
        ax=axis,
        cmap="mako",
        cbar_kws={"label": f"{label}: log10(mean |DCT coeff.|)"},
        xticklabels=16,
        yticklabels=4,
    )
    axis.set_xlabel("DCT frequency group (low → high)")
    axis.set_ylabel("Transformer layer" if component == "key" else "")
    axis.text(0.02, 0.96, label, transform=axis.transAxes, ha="left", va="top", color="white", fontweight="bold")

fig.text(0.5, -0.035, "K and V use independent color scales; colors are not comparable across panels.", ha="center", fontsize=9)

save_figure(fig, "dct_layer_frequency_heatmap")
