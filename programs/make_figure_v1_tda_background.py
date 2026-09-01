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

from programs.paths import FIGURES_DIR


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
# A square whose sides appear before its diagonals in a Rips filtration, giving
# the middle-scale panel a genuine second one-dimensional cycle.
CLUSTER = np.array([[2.15, -0.35], [2.65, -0.35], [2.65, 0.15], [2.15, 0.15]])
POINTS = np.vstack((RING, CLUSTER))


def setup_panel(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 3.10)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title(title, fontsize=13.0, fontweight="bold", color=INK, pad=8)


def draw_points(ax, radius, color=INK):
    for x, y in POINTS:
        ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor="none", alpha=0.12))
        ax.plot(x, y, "o", ms=5.8, color=color, zorder=5)


def draw_edges(ax, pairs, color, lw=2.0, alpha=1.0):
    for i, j in pairs:
        ax.plot([POINTS[i, 0], POINTS[j, 0]], [POINTS[i, 1], POINTS[j, 1]],
                color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=2)


def convex_hull(points):
    """Return the counterclockwise planar convex hull of a small point set."""
    ordered = sorted(map(tuple, points))

    def cross(origin, a, b):
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1])


def draw_bar(ax, y, start, end, color, label):
    ax.plot([start, end], [y, y], color=color, lw=5.2, solid_capstyle="butt")
    ax.text(-0.08, y, label, ha="right", va="center", fontsize=10.8,
            fontweight="bold", color=INK)


def render(out_path=None):
    fig = plt.figure(figsize=(7.5, 3.15), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[3.25, 1.05], hspace=0.30,
                          left=0.075, right=0.985, top=0.93, bottom=0.12,
                          wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Panel 1: components have not yet connected.
    ax = axes[0]
    setup_panel(ax, r"Small $\varepsilon$")
    draw_points(ax, radius=0.17, color=GREY)
    ax.text(0.5, -0.10, r"$\beta_0=14,\ \beta_1=0$", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.4, color=INK_SECONDARY)

    # Panel 2: the ring and square each produce a one-dimensional hole.
    ax = axes[1]
    setup_panel(ax, r"Middle $\varepsilon$")
    draw_points(ax, radius=0.30, color=BLUE)
    draw_edges(ax, [(i, (i + 1) % 10) for i in range(10)], BLUE)
    draw_edges(ax, [(10, 11), (11, 12), (12, 13), (13, 10)], BLUE)
    ax.text(0.5, -0.10, r"$\beta_0=2,\ \beta_1=2$", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.4, color=INK_SECONDARY)

    # Panel 3: at sufficiently large scale the Rips 1-skeleton is complete;
    # its filled higher-dimensional simplices kill every one-dimensional loop.
    ax = axes[2]
    setup_panel(ax, r"Large $\varepsilon$")
    hull = convex_hull(POINTS)
    ax.add_patch(Polygon(hull, closed=True, facecolor=PALE_ORANGE,
                         edgecolor="none", alpha=0.70, zorder=0))
    complete_graph = [(i, j) for i in range(len(POINTS))
                      for j in range(i + 1, len(POINTS))]
    draw_edges(ax, complete_graph, ORANGE, lw=0.70, alpha=0.42)
    draw_points(ax, radius=0.47, color=ORANGE)
    ax.text(0.5, -0.10, r"$\beta_0=1,\ \beta_1=0$", transform=ax.transAxes,
            ha="center", va="top", fontsize=10.4, color=INK_SECONDARY)

    # Representative persistence bars: both middle-scale H1 classes die before
    # the large-scale complex becomes fully connected and filled.
    axb = fig.add_subplot(gs[1, :])
    axb.set_xlim(-0.55, 10.25)
    axb.set_ylim(-0.15, 2.35)
    axb.set_yticks([])
    axb.set_xticks([0, 3.1, 6.2, 9.3])
    axb.set_xticklabels([r"small $\varepsilon$", r"", r"", r"large $\varepsilon$"],
                         fontsize=9.7, color=INK_MUTED)
    axb.grid(axis="x", color=GRID, lw=0.9)
    for side in ("left", "right", "top"):
        axb.spines[side].set_visible(False)
    axb.spines["bottom"].set_color(GRID)
    axb.tick_params(axis="x", length=0)
    draw_bar(axb, 1.88, 0.0, 9.3, BLUE, r"$H_0$")
    draw_bar(axb, 1.38, 0.0, 3.1, BLUE, r"$H_0$")
    draw_bar(axb, 0.78, 3.1, 6.2, ORANGE, r"$H_1$")
    draw_bar(axb, 0.28, 3.45, 5.45, ORANGE, r"$H_1$")

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
