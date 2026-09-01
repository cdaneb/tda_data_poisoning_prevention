"""Plot the Phase Q representation-level repair from the frozen R1 artifact.

The chart is intentionally sparse: explanatory prose belongs in the poster's
LaTeX caption, while this file owns the exact artifact-to-mark mapping.

Usage:
    python make_figure_v6_representation_repair.py --render
    python make_figure_v6_representation_repair.py --render --out figures/figure_v6_representation_repair.png
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
            raise ValueError(f"{key}: expected seeds {SEEDS}, found {tuple(family)}")
        cells = [family[seed] for seed in SEEDS]
        for cell in cells:
            if cell["n_clean"] != 5000 or cell["n_poison"] != 500:
                raise ValueError(f"{key}: unexpected sample counts")
            if cell["raw_noop_count"] != 0:
                raise ValueError(f"{key}: raw no-op entered the paired experiment")
            if cell["control"]["n_features"] != 60 or cell["repair"]["n_features"] != 540:
                raise ValueError(f"{key}: unexpected feature dimensions")

        control = 100 * np.asarray(
            [c["control"]["duplicate_with_any_clean_fraction"] for c in cells],
            dtype=float,
        )
        repair = 100 * np.asarray(
            [c["repair"]["duplicate_with_any_clean_fraction"] for c in cells],
            dtype=float,
        )
        rows.append({
            "label": label,
            "control_mean": control.mean(),
            "control_sd": control.std(ddof=0),
            "repair_mean": repair.mean(),
            "repair_sd": repair.std(ddof=0),
            "reduction": 100 * (1 - repair.mean() / control.mean()),
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

    fig, ax = plt.subplots(figsize=(9.2, 3.65), facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.985, top=0.83, bottom=0.22)

    y = np.arange(len(rows))[::-1]
    height = 0.32
    control_mean = np.array([r["control_mean"] for r in rows])
    control_sd = np.array([r["control_sd"] for r in rows])
    repair_mean = np.array([r["repair_mean"] for r in rows])
    repair_sd = np.array([r["repair_sd"] for r in rows])

    ax.barh(y + height / 2, control_mean, height=height, color=NAVY,
            xerr=control_sd, capsize=3.5, error_kw={"lw": 1.2, "capthick": 1.2},
            label=r"one threshold ($t=0.4$)")
    ax.barh(y - height / 2, repair_mean, height=height, color=ORANGE,
            xerr=repair_sd, capsize=3.5, error_kw={"lw": 1.2, "capthick": 1.2},
            label="nine thresholds (0.1-0.9)")

    label_gap = 0.12
    for ypos, cval, csd, rval, rsd in zip(
            y, control_mean, control_sd, repair_mean, repair_sd):
        ax.text(cval + csd + label_gap, ypos + height / 2, f"{cval:.2f}",
                ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=NAVY)
        ax.text(rval + rsd + label_gap, ypos - height / 2, f"{rval:.2f}",
                ha="left", va="center",
                fontsize=11.5, fontweight="bold", color=ORANGE)

    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=12.5, fontweight="bold")
    ax.set_xlim(0, 5.35)
    ax.set_xticks(np.arange(0, 6, 1))
    ax.set_xlabel("Poison vectors exactly matching a clean vector (%)",
                  fontsize=12.5, fontweight="bold", labelpad=8)
    ax.tick_params(axis="x", labelsize=10.5, colors=INK_MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.015), ncol=2,
              frameon=False, fontsize=11.0, handlelength=1.5, columnspacing=1.4)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v6_representation_repair.pdf"
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
            f"{row['label']}: {row['control_mean']:.2f} +/- {row['control_sd']:.2f} -> "
            f"{row['repair_mean']:.2f} +/- {row['repair_sd']:.2f}; "
            f"reduction {row['reduction']:.1f}%"
        )


if __name__ == "__main__":
    main()
