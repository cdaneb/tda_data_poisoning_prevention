"""Create Figure V5: a conceptual view of how training-data poisoning affects a NIDS.

The figure intentionally uses schematic two-feature traffic space rather than
measured classifier performance.  It shows the mechanism: mislabeled attack
packets placed in the benign training set move the learned decision boundary,
so subsequently observed attacks can fall on the benign side.

Usage:
    python make_figure_v5_nids_poisoning.py --render
    python make_figure_v5_nids_poisoning.py --render --out figures/figure_v5_nids_poisoning.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from paths import FIGURES_DIR


INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PALE_BLUE = "#dceafb"
PALE_ORANGE = "#fbe4d9"


# Fixed coordinates make this a reproducible conceptual illustration, not a
# randomly sampled or empirically estimated decision boundary.
BENIGN = np.array([
    [0.13, 0.15], [0.19, 0.29], [0.28, 0.18], [0.31, 0.34],
    [0.37, 0.25], [0.22, 0.40], [0.42, 0.40], [0.10, 0.34],
])
ATTACK = np.array([
    [0.56, 0.61], [0.64, 0.73], [0.71, 0.63], [0.77, 0.79],
    [0.84, 0.70], [0.68, 0.87], [0.88, 0.86], [0.53, 0.75],
])
POISON = np.array([[0.48, 0.57], [0.57, 0.67], [0.62, 0.58]])


def setup_panel(ax, title):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)
        side.set_linewidth(0.9)
    ax.set_title(title, fontsize=13.0, fontweight="bold", color=INK, pad=8)


def scatter_training(ax, poisoned=False):
    ax.scatter(BENIGN[:, 0], BENIGN[:, 1], s=37, c=BLUE, edgecolors="white",
               linewidths=0.65, zorder=4)
    ax.scatter(ATTACK[:, 0], ATTACK[:, 1], s=43, marker="^", c=ORANGE,
               edgecolors="white", linewidths=0.65, zorder=4)
    if poisoned:
        # Blue fill says "benign label"; orange rim and triangle marker retain
        # the packet's true attack provenance for the explanatory diagram.
        ax.scatter(POISON[:, 0], POISON[:, 1], s=58, marker="^", c=BLUE,
                   edgecolors=ORANGE, linewidths=1.35, zorder=5)


def boundary(ax, intercept, color, label=None, style="-"):
    x = np.linspace(0.05, 0.98, 100)
    y = -0.95 * x + intercept
    keep = (y >= 0) & (y <= 1)
    ax.plot(x[keep], y[keep], color=color, lw=2.0, ls=style, zorder=3, label=label)


def render(out_path=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.72))
    fig.subplots_adjust(left=0.035, right=0.99, top=0.88, bottom=0.24, wspace=0.16)

    # Clean baseline: attacks lie above the learned boundary and are alerted.
    ax = axes[0]
    setup_panel(ax, "Clean training")
    scatter_training(ax)
    boundary(ax, 1.15, INK)

    # Poisoned training: attack-derived samples carry a benign label and the
    # fitted boundary moves toward the attack population.
    ax = axes[1]
    setup_panel(ax, "Poisoned training")
    scatter_training(ax, poisoned=True)
    boundary(ax, 1.15, INK_MUTED, style="--")
    boundary(ax, 1.47, INK)
    ax.annotate("shift", xy=(0.72, 0.78), xytext=(0.87, 0.38),
                fontsize=9.4, color=INK, fontweight="bold", ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 1.3})

    # Deployment: hold the traffic distribution fixed but apply the shifted
    # boundary.  The circled attacks below it now become false negatives.
    ax = axes[2]
    setup_panel(ax, "Deployment")
    scatter_training(ax)
    boundary(ax, 1.47, INK)
    missed = ATTACK[(ATTACK[:, 1] < -0.95 * ATTACK[:, 0] + 1.47)]
    ax.scatter(missed[:, 0], missed[:, 1], s=115, facecolors="none", edgecolors=ORANGE,
               linewidths=1.45, zorder=6)
    ax.annotate("missed attacks", xy=(0.68, 0.70), xytext=(0.25, 0.13),
                fontsize=9.2, color=ORANGE, fontweight="bold", ha="center", va="center",
                arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.3,
                            "connectionstyle": "arc3,rad=0.12"})

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=7.5, label="benign"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=ORANGE,
               markeredgecolor="white", markersize=8, label="attack"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=BLUE,
               markeredgecolor=ORANGE, markeredgewidth=1.2, markersize=7,
               label="attack labeled benign"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=9.2, handlelength=1.3, columnspacing=1.6, bbox_to_anchor=(0.5, 0.025))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v5_nids_poisoning.pdf"
    fig.savefig(out, dpi=250, transparent=(out.suffix.lower() == ".pdf"), bbox_inches="tight",
                pad_inches=0.03)
    plt.close(fig)
    return out, fig.get_size_inches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="write the figure")
    parser.add_argument("--out", default=None, help="override output path")
    args = parser.parse_args()
    if not args.render:
        parser.error("pass --render")
    out, size = render(args.out)
    print(f"Wrote {out}")
    print(f"Figure size: {size[0]:.2f} x {size[1]:.2f} in, vector PDF when output is .pdf")


if __name__ == "__main__":
    main()
