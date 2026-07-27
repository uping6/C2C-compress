from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


OUT_DIR = Path(__file__).resolve().parent

matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
        "mathtext.fontset": "stix",
    }
)


def save_figure(fig, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}")
    plt.close(fig)
