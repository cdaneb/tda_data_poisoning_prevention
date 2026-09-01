"""Create Figure V9: the controlled single- vs multi-threshold TDA pipeline.

This is a methodological schematic, not an empirical result.  The same input
packet feeds both arms.  Phase Q changes one representation variable: one
Binarizer cutoff becomes a fixed bank of nine cutoffs.  The per-threshold TDA
map and OPTICS settings remain fixed.

Usage:
    python make_figure_v9_tda_pipeline.py --render
    python make_figure_v9_tda_pipeline.py --render --out figures/figure_v9_tda_pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from programs.paths import FIGURES_DIR


INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#deddd7"
NAVY = "#39424f"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
PALE_BLUE = "#eef4fb"
PALE_ORANGE = "#fff1ea"
WHITE = "#ffffff"


def rounded_box(ax, x, y, w, h, facecolor=WHITE, edgecolor=GRID,
                linewidth=1.1, radius=0.09, zorder=1):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color=INK_MUTED, linewidth=1.6,
          mutation_scale=12, zorder=5):
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=mutation_scale,
        linewidth=linewidth, color=color, shrinkA=0, shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def illustrative_raster():
    """Return one fixed, data-like raster used only to explain thresholding."""
    yy, xx = np.mgrid[0:30, 0:50]
    raster = (
        0.47
        + 0.24 * np.sin(0.29 * xx + 0.11 * yy)
        + 0.18 * np.cos(0.18 * xx - 0.31 * yy)
        + 0.10 * np.sin(0.08 * xx * yy)
    )
    raster[:, 7:10] += 0.18
    raster[19:23, :] -= 0.15
    return (raster - raster.min()) / (raster.max() - raster.min())


def draw_image(ax, image, x, y, w, h, cmap="gray", edgecolor=GRID,
               linewidth=0.9, zorder=3):
    ax.imshow(
        image, extent=(x, x + w, y, y + h), origin="lower",
        cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
        aspect="auto", zorder=zorder,
    )
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor="none", edgecolor=edgecolor,
        linewidth=linewidth, zorder=zorder + 1,
    ))


def draw_input(ax, raster):
    rounded_box(ax, 0.18, 0.40, 2.45, 3.80, facecolor=WHITE,
                edgecolor=GRID, linewidth=1.25, radius=0.11)
    ax.text(1.405, 3.91, "INPUT PACKET", ha="center", va="center",
            fontsize=14.2, fontweight="bold", color=INK)
    ax.text(1.405, 3.60, "1,500 payload bytes", ha="center", va="center",
            fontsize=11.0, color=INK_SECONDARY)

    byte_values = ("7F", "1A", "C4", "58", "09", "E2")
    bx = 0.40
    for i, value in enumerate(byte_values):
        face = PALE_BLUE if i % 2 == 0 else "#f4f3ef"
        rounded_box(ax, bx + 0.29 * i, 3.23, 0.25, 0.29,
                    facecolor=face, edgecolor=GRID, linewidth=0.7,
                    radius=0.025, zorder=2)
        ax.text(bx + 0.29 * i + 0.125, 3.375, value,
                ha="center", va="center", fontsize=8.8,
                fontweight="bold", color=INK_SECONDARY, zorder=4)
    ax.text(2.22, 3.375, "...", ha="center", va="center",
            fontsize=12.0, color=INK_MUTED)

    draw_image(ax, raster, 0.40, 1.18, 2.00, 1.56,
               cmap="gray", edgecolor=GRID, linewidth=0.9)


def draw_control(ax, raster):
    rounded_box(ax, 3.13, 2.51, 2.42, 1.69, facecolor=PALE_BLUE,
                edgecolor=NAVY, linewidth=1.5, radius=0.10)
    ax.text(4.34, 3.98, "CONTROL", ha="center", va="center",
            fontsize=11.0, fontweight="bold", color=NAVY)
    ax.text(4.34, 3.73, r"one cutoff: $t=0.4$",
            ha="center", va="center", fontsize=10.4,
            fontweight="bold", color=INK_SECONDARY)
    draw_image(ax, raster > 0.4, 3.47, 2.72, 1.74, 0.84,
               cmap="gray_r", edgecolor=GRID, linewidth=0.8)


def draw_multithreshold(ax, raster):
    rounded_box(ax, 3.13, 0.40, 2.42, 1.82, facecolor=PALE_ORANGE,
                edgecolor=ORANGE, linewidth=1.55, radius=0.10)
    ax.text(4.34, 1.99, "MULTI-THRESHOLD", ha="center", va="center",
            fontsize=10.6, fontweight="bold", color=ORANGE)
    ax.text(4.34, 1.75, r"nine cutoffs: $t=0.1,\ldots,0.9$",
            ha="center", va="center", fontsize=9.8,
            fontweight="bold", color=INK_SECONDARY)

    # Three visible faces communicate the larger nine-member nested stack.
    for zorder, (threshold, x, y) in enumerate(
        ((0.1, 3.41, 0.66), (0.4, 3.60, 0.74), (0.9, 3.79, 0.82)),
        start=4,
    ):
        draw_image(ax, raster > threshold, x, y, 1.42, 0.72,
                   cmap="gray_r", edgecolor=GRID, linewidth=0.75,
                   zorder=zorder)


def draw_filtration_icon(ax, cx, cy, color):
    starts = ((-0.26, 0.18), (-0.26, 0.00), (-0.26, -0.18),
              (-0.06, 0.27), (0.08, -0.27))
    ends = ((0.22, 0.18), (0.22, 0.00), (0.22, -0.18),
            (0.20, -0.05), (-0.18, 0.04))
    for (sx, sy), (ex, ey) in zip(starts, ends):
        arrow(ax, (cx + sx, cy + sy), (cx + ex, cy + ey),
              color=color, linewidth=1.0, mutation_scale=7, zorder=5)


def draw_barcode_icon(ax, cx, cy, color):
    bars = ((-0.28, 0.24, 0.20), (-0.28, 0.08, 0.42),
            (-0.10, -0.08, 0.35), (0.02, -0.24, 0.26))
    for x, y, width in bars:
        ax.plot([cx + x, cx + x + width], [cy + y, cy + y],
                color=color, linewidth=2.1, solid_capstyle="butt", zorder=5)


def draw_vector_icon(ax, cx, cy, color, rows=1):
    shades = (color, BLUE if color == NAVY else "#f28b62")
    for row in range(rows):
        for col in range(8):
            ax.add_patch(Rectangle(
                (cx - 0.32 + col * 0.09, cy + 0.10 - row * 0.18),
                0.065, 0.12, facecolor=shades[(row + col) % 2],
                edgecolor="none", zorder=5,
            ))


def draw_cluster_icon(ax, cx, cy, color):
    points = ((-0.25, 0.11), (-0.08, 0.27), (0.06, 0.02),
              (0.25, 0.21), (-0.18, -0.20), (0.20, -0.24))
    for i, (dx, dy) in enumerate(points):
        fill = color if i not in (3, 4) else INK_MUTED
        ax.add_patch(Circle((cx + dx, cy + dy), 0.065,
                            facecolor=fill, edgecolor=WHITE,
                            linewidth=0.5, zorder=5))


def draw_downstream_row(ax, cy, color, multi=False):
    xs = (6.98, 8.28, 9.58, 10.93)
    draw_filtration_icon(ax, xs[0], cy, color)
    draw_barcode_icon(ax, xs[1], cy, color)
    draw_vector_icon(ax, xs[2], cy, color, rows=3 if multi else 1)
    draw_cluster_icon(ax, xs[3], cy, color)

    for left, right in zip(xs[:-1], xs[1:]):
        arrow(ax, (left + 0.42, cy), (right - 0.42, cy),
              color=color, linewidth=1.35, mutation_scale=10)

    if multi:
        labels = (r"same map $\times9$", r"concat. / $\sqrt{9}$",
                  "540 features", "OPTICS")
    else:
        labels = ("5 filtrations", r"$H_0$ and $H_1$",
                  "60 features", "OPTICS")
    for x, label in zip(xs, labels):
        ax.text(x, cy - 0.49, label, ha="center", va="center",
                fontsize=9.1, fontweight="bold",
                color=color if multi else INK_SECONDARY)


def draw_combined_pipeline(ax):
    rounded_box(ax, 6.22, 0.40, 5.58, 3.80, facecolor=WHITE,
                edgecolor=GRID, linewidth=1.3, radius=0.11)
    ax.text(9.01, 3.93, "SAME TDA MAP + OPTICS",
            ha="center", va="center", fontsize=13.4,
            fontweight="bold", color=INK)
    ax.plot([6.52, 11.50], [2.30, 2.30], color=GRID, linewidth=1.0)
    draw_downstream_row(ax, 3.18, NAVY, multi=False)
    draw_downstream_row(ax, 1.46, ORANGE, multi=True)


def render(out_path=None):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(11.7, 4.45), facecolor=WHITE)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.025)
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0.20, 4.45)
    ax.axis("off")

    raster = illustrative_raster()
    draw_input(ax, raster)
    draw_control(ax, raster)
    draw_multithreshold(ax, raster)
    draw_combined_pipeline(ax)

    # One shared packet forks into two separately evaluated representations.
    ax.plot([2.63, 2.86], [2.30, 2.30], color=INK_MUTED, linewidth=1.5, zorder=4)
    ax.plot([2.86, 2.86], [1.31, 3.35], color=INK_MUTED, linewidth=1.5, zorder=4)
    arrow(ax, (2.86, 3.35), (3.08, 3.35), color=NAVY, linewidth=1.6)
    arrow(ax, (2.86, 1.31), (3.08, 1.31), color=ORANGE, linewidth=1.6)

    # Separate labeled arrows keep the two arms distinct while reusing the
    # same downstream method and settings.
    arrow(ax, (5.58, 3.18), (6.17, 3.18), color=NAVY, linewidth=1.7)
    ax.text(5.875, 3.37, "1 mask", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=NAVY)
    arrow(ax, (5.58, 1.46), (6.17, 1.46), color=ORANGE, linewidth=1.7)
    ax.text(5.875, 1.66, "9 masks", ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=ORANGE)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v9_tda_pipeline.pdf"
    fig.savefig(
        out, dpi=250, facecolor=WHITE, transparent=False,
        bbox_inches="tight", pad_inches=0.025,
    )
    size = tuple(fig.get_size_inches())
    plt.close(fig)
    return out, size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="write the figure")
    parser.add_argument("--out", default=None, help="override output path")
    args = parser.parse_args()
    if not args.render:
        parser.error("pass --render")
    out, size = render(args.out)
    print(f"Wrote {out}")
    print(
        f"Figure size: {size[0]:.2f} x {size[1]:.2f} in, "
        "vector PDF when output is .pdf"
    )


if __name__ == "__main__":
    main()
