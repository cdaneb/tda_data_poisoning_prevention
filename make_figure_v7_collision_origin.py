"""Plot the Q3 earliest-merger attribution from its frozen audit artifact.

This contextual chart uses the legacy single-threshold R60 frame.  It shows
where final repeated-member collision mass first arose; it is not a Phase Q
repair-arm result.

Usage:
    python make_figure_v7_collision_origin.py --render
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paths import FIGURES_DIR, RESULTS_DIR


INK = "#0b0b0b"
WHITE = "#ffffff"
GREY = "#898781"
NAVY = "#39424f"
ORANGE = "#eb6834"
SEEDS = ("42", "123", "456", "789", "1024")
DISPLAY_STAGES = (
    ("raw_payload", "Raw payload", GREY),
    ("binary_mask", "Binarization", NAVY),
    ("unscaled_diagrams", "Diagrams", ORANGE),
)


def load_summary(path: Path):
    doc = json.loads(path.read_text(encoding="utf-8"))
    if set(doc["seeds"]) != set(SEEDS):
        raise ValueError(f"unexpected seed set: {tuple(doc['seeds'])}")

    per_stage = {stage: [] for stage, _, _ in DISPLAY_STAGES}
    later_shares = []
    displayed = set(per_stage)
    for seed in SEEDS:
        attribution = doc["seeds"][seed]["Q3_C_population_attribution"][
            "earliest_merger_attribution"
        ]
        if not attribution["attribution_sums_to_final_repeated_mass"]:
            raise ValueError(f"seed {seed}: attribution does not sum to final repeated mass")
        rows = attribution["rows"]
        if abs(sum(r["share_of_final_repeated_member_mass"] for r in rows) - 1.0) > 1e-9:
            raise ValueError(f"seed {seed}: attribution shares do not sum to one")
        lookup = {r["stage"]: r["share_of_final_repeated_member_mass"] for r in rows}
        for stage in per_stage:
            per_stage[stage].append(100 * float(lookup[stage]))
        later_shares.extend(100 * float(value) for stage, value in lookup.items() if stage not in displayed)

    if not np.allclose(later_shares, 0.0, atol=1e-12):
        raise ValueError("a stage after the displayed three has nonzero attribution")

    summary = []
    for stage, label, color in DISPLAY_STAGES:
        values = np.asarray(per_stage[stage], dtype=float)
        summary.append({
            "stage": stage,
            "label": label,
            "color": color,
            "mean": values.mean(),
            "sd": values.std(ddof=0),
        })
    if not np.isclose(sum(item["mean"] for item in summary), 100.0):
        raise ValueError("displayed stage means do not sum to 100%")
    return summary


def render(out_path=None, artifact=None):
    artifact_path = Path(artifact) if artifact else RESULTS_DIR / "phase_q3_collision_audit.json"
    summary = load_summary(artifact_path)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(8.6, 1.48), facecolor="white")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.06)

    left = 0.0
    centers = {}
    for item in summary:
        ax.barh(0, item["mean"], left=left, height=0.52, color=item["color"],
                edgecolor="white", linewidth=1.6)
        centers[item["stage"]] = left + item["mean"] / 2
        left += item["mean"]

    for item in summary[:2]:
        ax.text(centers[item["stage"]], 0, f"{item['label']}\n{item['mean']:.1f}%",
                ha="center", va="center", fontsize=13.2, fontweight="bold", color=WHITE)

    diagram = summary[2]
    ax.annotate(f"{diagram['label']}  {diagram['mean']:.1f}%",
                xy=(centers[diagram["stage"]], 0.25), xytext=(93.5, 0.56),
                ha="center", va="bottom", fontsize=11.8, fontweight="bold", color=ORANGE,
                arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 1.4})

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.42, 0.78)
    ax.axis("off")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(out_path) if out_path else FIGURES_DIR / "figure_v7_collision_origin.pdf"
    fig.savefig(out, dpi=250, transparent=(out.suffix.lower() == ".pdf"),
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="write the figure")
    parser.add_argument("--out", default=None, help="override output path")
    parser.add_argument("--artifact", default=None, help="override the Q3 JSON artifact")
    args = parser.parse_args()
    if not args.render:
        parser.error("pass --render")
    out, summary = render(args.out, args.artifact)
    print(f"Wrote {out}")
    for item in summary:
        print(f"{item['label']}: {item['mean']:.2f} +/- {item['sd']:.2f}%")


if __name__ == "__main__":
    main()
