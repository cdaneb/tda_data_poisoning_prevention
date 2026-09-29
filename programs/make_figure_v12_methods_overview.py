"""Create Figure V12: overview of the paper's methods and evidence flow.

The upper lane separates reconstruction of the supplied research artifacts
from the controlled experiments.  The lower ladder shows why evidence at one
level (for example, fewer representation collisions) does not establish the
next level (for example, downstream classifier utility).

Usage:
    python -m programs.make_figure_v12_methods_overview --render
    python -m programs.make_figure_v12_methods_overview --render --out PATH.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from programs.paths import FIGURES_DIR


INK = "#111315"
INK_SECONDARY = "#4d535a"
INK_MUTED = "#7c838b"
GRID = "#d9dde2"
NAVY = "#34495e"
BLUE = "#2a78d6"
ORANGE = "#e56a3a"
PALE_BLUE = "#eef5fc"
PALE_ORANGE = "#fff1ea"
PALE_GRAY = "#f5f6f7"
WHITE = "#ffffff"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    facecolor=WHITE,
    edgecolor=GRID,
    linewidth=1.15,
    radius=0.08,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=INK_MUTED,
    linewidth=1.45,
    mutation_scale=11,
    zorder=5,
    connectionstyle="arc3",
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def box_text(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    *,
    facecolor=WHITE,
    edgecolor=GRID,
    title_color=INK,
    title_size=10.0,
    body_size=8.8,
    linewidth=1.15,
):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.text(
        x + w / 2,
        y + h - 0.23,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=6,
    )
    ax.text(
        x + w / 2,
        y + h / 2 - 0.11,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=INK_SECONDARY,
        linespacing=1.25,
        zorder=6,
    )


def lane_label(ax, y, number, title, color):
    rounded_box(
        ax,
        0.10,
        y,
        0.55,
        0.55,
        facecolor=color,
        edgecolor=color,
        linewidth=0.8,
        radius=0.12,
        zorder=4,
    )
    ax.text(
        0.375,
        y + 0.275,
        str(number),
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=WHITE,
        zorder=6,
    )
    ax.text(
        0.78,
        y + 0.275,
        title,
        ha="left",
        va="center",
        fontsize=10.2,
        fontweight="bold",
        color=color,
        zorder=6,
    )


def draw_forensic_lane(ax):
    lane_label(ax, 7.18, 1, "SUPPLIED-ARTIFACT RECONSTRUCTION", NAVY)

    box_text(
        ax,
        0.55,
        5.82,
        3.15,
        1.08,
        "Delivered materials",
        "6 Jupyter notebooks  |  2 Excel tables\n18 poison payloads\nraw 1,500-byte packet payloads",
        facecolor=PALE_BLUE,
        edgecolor=NAVY,
        title_color=NAVY,
    )
    box_text(
        ax,
        4.16,
        5.82,
        4.02,
        1.08,
        "Forensic checks",
        "execution order  |  sample construction\nfit boundaries  |  feature order\nconstant coordinates  |  raw-row alignment",
        facecolor=PALE_BLUE,
        edgecolor=NAVY,
        title_color=NAVY,
    )
    box_text(
        ax,
        8.64,
        5.82,
        4.80,
        1.08,
        "What can be reconstructed",
        "Reconstruct the 126- and 280-feature configurations\nCompare separate-batch fitting with one shared fit\nPrespecified random seed: 60",
        facecolor=PALE_BLUE,
        edgecolor=NAVY,
        title_color=NAVY,
    )
    arrow(ax, (3.74, 6.36), (4.08, 6.36), color=NAVY)
    arrow(ax, (8.22, 6.36), (8.56, 6.36), color=NAVY)


def draw_controlled_lane(ax):
    lane_label(ax, 5.02, 2, "CONTROLLED EXPERIMENTS", ORANGE)

    box_text(
        ax,
        0.55,
        3.17,
        2.25,
        1.48,
        "Payload populations",
        "UNSW and CICIDS\n5 prespecified random seeds\nunmodified rows\n+ poisoned copies",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
    )
    box_text(
        ax,
        3.17,
        3.17,
        2.30,
        1.48,
        "Four attack families",
        "transpositions  |  reversal\nblock swap  |  cyclic shift\nbyte multiset preserved",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
    )
    arrow(ax, (2.84, 3.91), (3.09, 3.91), color=ORANGE)

    box_text(
        ax,
        6.08,
        4.05,
        2.18,
        0.92,
        "Frame A: first match",
        "permissive transpositions;\nsome payloads unchanged",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
        body_size=8.4,
    )
    box_text(
        ax,
        8.73,
        4.05,
        4.71,
        0.92,
        "First stage creating an exact match",
        "raw payload -> binary mask -> persistence -> 60-vector\nassign each repeated final vector\nto the first stage where it matches another",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
        body_size=8.4,
    )

    box_text(
        ax,
        6.08,
        2.84,
        2.18,
        0.92,
        "Frame B: raw changes",
        "redraw attacks until every\npoisoned payload changes",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
        body_size=8.4,
    )
    box_text(
        ax,
        8.73,
        2.84,
        4.71,
        0.92,
        "One controlled representation change",
        "control: one cutoff -> 60 features\nrepair: nine fixed cutoffs ->\n540 concatenated features",
        facecolor=PALE_ORANGE,
        edgecolor=ORANGE,
        title_color=ORANGE,
        body_size=8.4,
    )

    ax.plot([5.53, 5.77], [3.91, 3.91], color=ORANGE, linewidth=1.35, zorder=4)
    ax.plot([5.77, 5.77], [3.30, 4.51], color=ORANGE, linewidth=1.35, zorder=4)
    arrow(ax, (5.77, 4.51), (6.00, 4.51), color=ORANGE)
    arrow(ax, (5.77, 3.30), (6.00, 3.30), color=ORANGE)
    arrow(ax, (8.31, 4.51), (8.65, 4.51), color=ORANGE)
    arrow(ax, (8.31, 3.30), (8.65, 3.30), color=ORANGE)


def draw_evidence_ladder(ax):
    lane_label(ax, 2.18, 3, "SEPARATE EVIDENCE CLAIMS", NAVY)

    boxes = (
        (
            0.55,
            2.22,
            "1  Reproduction",
            "Can the supplied\nprocedure be reconstructed?",
            PALE_BLUE,
            NAVY,
        ),
        (
            3.17,
            2.22,
            "2  Representation",
            "Where do exact feature\nmatches arise, and are\nthey reduced?",
            PALE_BLUE,
            NAVY,
        ),
        (
            5.79,
            2.22,
            "3  Cluster separation",
            "Does fixed OPTICS form\npoison-dominant clusters?",
            PALE_BLUE,
            NAVY,
        ),
        (
            8.41,
            2.22,
            "4  Label-free removal",
            "Isolation Forest and kNN\ncalibrated on trusted data;\nreport poison capture and\nunmodified-row removal",
            PALE_BLUE,
            NAVY,
        ),
        (
            11.03,
            2.41,
            "5  Classifier utility",
            "Six training conditions;\ntest payload identities absent\nfrom training; report malicious\nrecall and benign FPR",
            PALE_BLUE,
            NAVY,
        ),
    )
    for x, w, title, body, face, edge in boxes:
        box_text(
            ax,
            x,
            0.42,
            w,
            1.18,
            title,
            body,
            facecolor=face,
            edgecolor=edge,
            title_color=edge,
            title_size=8.8,
            body_size=8.4,
            linewidth=1.25,
        )

    for x in (2.85, 5.47, 8.09, 10.71):
        arrow(ax, (x, 1.01), (x + 0.24, 1.01), color=NAVY, linewidth=1.25)

def render(out_path: str | Path | None = None):
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(11.7, 6.4), facecolor=WHITE)
    fig.subplots_adjust(left=0.012, right=0.988, top=0.985, bottom=0.02)
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0.0, 7.85)
    ax.axis("off")

    draw_forensic_lane(ax)
    draw_controlled_lane(ax)
    draw_evidence_ladder(ax)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = (
        Path(out_path)
        if out_path
        else FIGURES_DIR / "figure_v12_methods_overview.pdf"
    )
    fig.savefig(
        out,
        dpi=300,
        facecolor=WHITE,
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.025,
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
