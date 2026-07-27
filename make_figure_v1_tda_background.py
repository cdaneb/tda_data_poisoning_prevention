"""Create Figure V1: a conceptual persistent-homology primer for the poster.

The poster's analysis uses cubical persistence on packet images. This figure is
deliberately labelled as a point-cloud illustration: it introduces the ideas of
components (H_0), loops (H_1), and persistence across a scale parameter without
claiming that this is the packet-data computation.

Usage:
    python make_figure_v1_tda_background.py --render
    python make_figure_v1_tda_background.py --render --out figures/figure_v1_tda_background.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np

from paths import FIGURES_DIR


INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#898781"
PALE_BLUE = "#dceafb"
PALE_ORANGE = "#fbe4d9"


def ring_points(center=(0.0, 0.0), radius=1.0, n=10):
    theta = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.column_stack((center[0] + radius * np.cos(theta),
                            center[1] + radius * np.sin(theta)))


RING = ring_points()
CLUSTER = np.array([[2.25, -0.30], [2.55, -0.12], [2.38, 0.20], [2.72, 0.24]])
POINTS = np.vstack((RING, CLUSTER))


def setup_panel(ax, title, subtitle):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 3.10)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title(title, fontsize=12.0, fontweight="bold", color=INK, pad=18)
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.4, color=INK_MUTED)


def draw_points(ax, radius, color=INK):
    for x, y in POINTS:
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=0.12))
        ax.plot(x, y, "o", ms=4.8, color=color, zorder=5)


def draw_edges(ax, pairs, color, lw=2.0):
    for i, j in pairs:
        ax.plot([POINTS[i, 0], POINTS[j, 0]], [POINTS[i, 1], POINTS[j, 1]],
                color=color, lw=lw, solid_capstyle="round", zorder=2)


def draw_bar(ax, y, start, end, color, label):
    ax.plot([start, end], [y, y], color=color, lw=4.0, solid_capstyle="butt")
    ax.text(-0.08, y, label, ha="right", va="center", fontsize=9.2,
            fontweight="bold", color=INK)


def render(out_path=None):
    fig = plt.figure(figsize=(7.5, 4.15), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3.15, 1.30], hspace=0.38,
                          left=0.065, right=0.985, top=0.80, bottom=0.10,
                          wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel 1: components have not yet connected.
    ax = axes[0]
    setup_panel(ax, r"Small radius $\varepsilon$", r"many separate components")
    draw_points(ax, radius=0.17, color=GREY)
    ax.text(0.5, -0.13, r"$H_0$: many components", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.3, color=INK_SECONDARY)

    # Panel 2: ring produces a one-dimensional hole while the cluster merges.
    ax = axes[1]
    setup_panel(ax, r"Intermediate $\varepsilon$", r"nearby points connect")
    draw_points(ax, radius=0.30, color=BLUE)
    draw_edges(ax, [(i, (i + 1) % 10) for i in range(10)], BLUE)
    draw_edges(ax, [(10, 11), (10, 12), (11, 13), (12, 13)], BLUE)
    ax.text(0.5, -0.13, r"$H_0$: 2 components  $\quad$ $H_1$: 1 loop", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.3, color=INK_SECONDARY)

    # Panel 3: a larger scale fills the loop and joins both regions.
    ax = axes[2]
    setup_panel(ax, r"Large radius $\varepsilon$", r"connections fill the loop")
    draw_points(ax, radius=0.47, color=ORANGE)
    draw_edges(ax, [(i, (i + 1) % 10) for i in range(10)], ORANGE, lw=1.8)
    draw_edges(ax, [(0, 2), (2, 4), (4, 6), (6, 8), (8, 0), (1, 5), (3, 7),
                    (8, 10), (9, 12), (10, 12), (11, 13)], ORANGE, lw=1.6)
    ax.add_patch(Polygon(RING, closed=True, facecolor=PALE_ORANGE,
                         edgecolor="none", alpha=0.70, zorder=1))
    ax.text(0.5, -0.13, r"$H_0$: 1 component  $\quad$ $H_1$: loop filled", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.3, color=INK_SECONDARY)

    # Persistence bars: the single long H0 bar and one finite H1 bar are the
    # visual payoff of scanning across all intermediate scales.
    axb = fig.add_subplot(gs[1, :])
    axb.set_xlim(-0.55, 10.25)
    axb.set_ylim(-0.25, 2.45)
    axb.set_yticks([])
    axb.set_xticks([0, 3.1, 6.2, 9.3])
    axb.set_xticklabels([r"small $\varepsilon$", r"", r"", r"large $\varepsilon$"],
                         fontsize=8.7, color=INK_MUTED)
    axb.grid(axis="x", color=GRID, lw=0.9)
    for side in ("left", "right", "top"):
        axb.spines[side].set_visible(False)
    axb.spines["bottom"].set_color(GRID)
    axb.tick_params(axis="x", length=0)
    draw_bar(axb, 1.68, 0.0, 9.3, BLUE, r"$H_0$")
    draw_bar(axb, 1.25, 0.0, 3.1, BLUE, r"$H_0$")
    draw_bar(axb, 0.82, 3.1, 6.2, ORANGE, r"$H_1$")
    axb.text(5.0, 2.15, "Persistence bars: long-lived features are the durable shape signal",
             ha="center", va="center", fontsize=10.0, fontweight="bold", color=INK)
    axb.text(5.0, 0.10,
             "Conceptual point-cloud illustration; this poster computes cubical persistence on packet images.",
             ha="center", va="center", fontsize=8.0, color=INK_MUTED)

    fig.text(0.035, 0.965, "Persistent homology records shape across scale",
             fontsize=15.0, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.035, 0.915,
             r"As $\varepsilon$ grows, components merge ($H_0$), loops appear ($H_1$), and short-lived features fade.",
             fontsize=9.2, color=INK_MUTED, ha="left", va="top")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v1_tda_background.pdf"
    fig.savefig(out, dpi=200, transparent=(out.suffix.lower() == ".pdf"))
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
