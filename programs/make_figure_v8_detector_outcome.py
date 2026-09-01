"""Plot the downstream Phase Q detector outcome from the frozen R1 artifact.

The figure reports the preregistered matched-clean-cost removal delta.

Usage:
    python make_figure_v8_detector_outcome.py --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from programs.paths import FIGURES_DIR, RESULTS_DIR


INK = "#0b0b0b"
INK_MUTED = "#67655f"
GRID = "#deddd7"
NAVY = "#39424f"
ORANGE = "#eb6834"

FAMILIES = (
    ("transpositions", "Transpositions"),
    ("block_reversal", "Block reversal"),
    ("block_swap", "Block swap"),
    ("cyclic_shift", "Cyclic shift"),
)
SEEDS = ("42", "123", "456", "789", "1024")


def load_summary(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for key, label in FAMILIES:
        family = doc["runs"][key]
        if set(family) != set(SEEDS):
            raise ValueError(f"{key}: unexpected seed set")
        cells = [family[seed] for seed in SEEDS]
        deltas = [c["matched_clean_cost"][0]["poison_removal_rate_delta"] for c in cells]
        if any(value is None for value in deltas):
            raise ValueError(f"{key}: missing exact-purity matched-cost delta")
        delta_pp = 100 * np.asarray(deltas, dtype=float)
        rows.append({
            "label": label,
            "delta_mean": delta_pp.mean(),
            "delta_sd": delta_pp.std(ddof=0),
        })
    return rows


def render(out_path=None, artifact=None):
    artifact_path = Path(artifact) if artifact else RESULTS_DIR / "phase_q_r1_multithreshold_capture.json"
    rows = load_summary(artifact_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    fig, ax_delta = plt.subplots(figsize=(6.6, 3.65), facecolor="white")
    fig.subplots_adjust(left=0.31, right=0.97, top=0.94, bottom=0.23)

    y = np.arange(len(rows))[::-1]
    labels = [r["label"] for r in rows]

    # Matched-clean-cost detector effect.
    delta_mean = np.asarray([r["delta_mean"] for r in rows])
    delta_sd = np.asarray([r["delta_sd"] for r in rows])
    ax_delta.axvline(0, color=NAVY, linewidth=1.5, zorder=1)
    ax_delta.errorbar(delta_mean, y, xerr=delta_sd, fmt="o", ms=8.5,
                      color=ORANGE, ecolor=ORANGE, elinewidth=1.8,
                      capsize=4, capthick=1.5, zorder=3)
    for xpos, ypos in zip(delta_mean, y):
        offset = 0.10 if xpos >= 0 else -0.10
        ax_delta.text(xpos + offset, ypos + 0.20, f"{xpos:+.2f}",
                      ha="left" if xpos >= 0 else "right", va="bottom",
                      fontsize=11.0, fontweight="bold", color=ORANGE)
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels(labels, fontsize=11.6, fontweight="bold")
    ax_delta.set_xlim(-1.0, 1.45)
    ax_delta.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax_delta.set_xlabel("Matched-cost removal change\n(percentage points)",
                        fontsize=11.5, fontweight="bold", labelpad=8)

    ax_delta.tick_params(axis="x", labelsize=10.2, colors=INK_MUTED, length=0)
    ax_delta.tick_params(axis="y", length=0, pad=7)
    ax_delta.grid(axis="x", color=GRID, linewidth=0.9)
    ax_delta.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax_delta.spines[side].set_visible(False)
    ax_delta.spines["bottom"].set_color(GRID)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v8_detector_outcome.pdf"
    fig.savefig(out, dpi=250, transparent=(out.suffix.lower() == ".pdf"),
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="write the figure")
    parser.add_argument("--out", default=None, help="override output path")
    parser.add_argument("--artifact", default=None, help="override the R1 JSON artifact")
    args = parser.parse_args()
    if not args.render:
        parser.error("pass --render")
    out, rows = render(args.out, args.artifact)
    print(f"Wrote {out}")
    for row in rows:
        print(
            f"{row['label']}: delta {row['delta_mean']:+.2f} "
            f"+/- {row['delta_sd']:.2f} pp"
        )


if __name__ == "__main__":
    main()
