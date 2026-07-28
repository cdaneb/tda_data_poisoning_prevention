"""Figure V4: random swaps, surrogate guidance, and a trivial cyclic shift.

This companion to V3 uses the same source artifacts and visual palette. It
separates two claims that the four-family plot alone can blur: surrogate guidance
raises strict-purity capture above random swaps, but an unguided cyclic shift
reaches essentially the same capture level.

Usage:
    python make_figure_v4_guidance_shift.py --check
    python make_figure_v4_guidance_shift.py --render
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from paths import FIGURES_DIR, RESULTS_DIR

SEEDS = ["42", "123", "456", "789", "1024"]
ALGO = "OPTICS"

REFERENCE = {
    "random_swaps": (1.80, 0.51),
    "guided_swaps": (6.48, 1.24),
    "cyclic_shift": (6.28, 1.31),
}

C_SERIES = "#2a78d6"
C_REFERENCE = "#eb6834"
INK = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def capture_summary(per_seed):
    values = np.asarray(per_seed, dtype=float)
    return float(values.mean()), float(values.std(ddof=0)), values.tolist()


def parse():
    """Read every plotted number from the committed multiseed artifacts."""
    m7 = json.load(open(RESULTS_DIR / "phase_m_m7_capture.json", encoding="utf-8"))
    lens = json.load(open(RESULTS_DIR / "lens4_baseline_multiseed.json", encoding="utf-8"))

    random = [m7["transpositions"][seed]["per_algo"][ALGO]["red_poison_capture_pct"]
              for seed in SEEDS]
    guided = [lens["G60-MLP"][seed][ALGO]["red_poison_capture_pct"] for seed in SEEDS]
    cyclic = [m7["cyclic_shift"][seed]["per_algo"][ALGO]["red_poison_capture_pct"]
              for seed in SEEDS]

    return [
        {"key": "random_swaps", "label": "Random\nswaps", "values": random, "color": C_SERIES},
        {"key": "guided_swaps", "label": "Surrogate-guided\nswaps", "values": guided,
         "color": C_REFERENCE},
        {"key": "cyclic_shift", "label": "Unguided\ncyclic shift", "values": cyclic,
         "color": "#f0a35e"},
    ]


def check(rows, tol=5e-3):
    failures = []
    print(f"{'Condition':<22}{'mean':>10}{'pop. SD':>11}{'per seed'}")
    print("-" * 76)
    for row in rows:
        mean, sd, values = capture_summary(row["values"])
        row["mean"] = mean
        row["sd"] = sd
        ref_mean, ref_sd = REFERENCE[row["key"]]
        print(f"{row['key']:<22}{mean:>10.4f}{sd:>11.4f}  {values}")
        if abs(mean - ref_mean) > tol or abs(sd - ref_sd) > tol:
            failures.append(
                f"{row['key']}: parsed {mean:.4f} +/- {sd:.4f}; "
                f"expected {ref_mean:.2f} +/- {ref_sd:.2f}"
            )
    return failures


def render(rows, out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(7.5, 3.25))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.23)
    x = np.arange(len(rows))
    means = [row["mean"] for row in rows]
    sds = [row["sd"] for row in rows]

    ax.bar(x, means, width=.62, color=[row["color"] for row in rows], zorder=3)
    ax.errorbar(x, means, yerr=sds, fmt="none", ecolor=INK, elinewidth=1.5,
                capsize=6, capthick=1.5, zorder=4)
    for xi, mean, sd in zip(x, means, sds):
        ax.annotate(f"{mean:.2f}", (xi, mean + sd), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=12.5, fontweight="bold", color=INK)

    ax.annotate("guidance: +4.68 points", xy=(1, means[1] + sds[1]),
                xytext=(.23, 8.35), textcoords="data", ha="center", va="bottom",
                fontsize=9.2, color=C_REFERENCE, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": C_REFERENCE, "lw": 1.4,
                            "connectionstyle": "angle3,angleA=0,angleB=90"})
    ax.set_ylim(0, 10.0)
    ax.set_xticks(x, [row["label"] for row in rows], fontsize=10.2, fontweight="bold")
    ax.set_ylabel("100%-pure capture (%)", fontsize=10.5, color=INK)
    ax.yaxis.grid(True, color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9.5, length=0, colors=INK_MUTED)
    ax.tick_params(axis="x", length=0, pad=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)

    fig.text(.03, .95, "Cyclic shift matches surrogate-guided capture",
             fontsize=13.2, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(.03, .895, "UNSW-NB15  .  OPTICS  .  5 seeds  .  same threshold",
             fontsize=9.0, color=INK_MUTED, ha="left", va="top")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = out_path or (FIGURES_DIR / "figure_v4_guidance_shift.pdf")
    fig.savefig(out, dpi=200, transparent=(out.suffix == ".pdf"))
    plt.close(fig)
    return out, fig.get_size_inches()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not (args.check or args.render):
        parser.error("pass --check or --render")

    rows = parse()
    failures = check(rows)
    if failures:
        print("FAIL: source artifacts do not match the recorded values:")
        for failure in failures:
            print(f"  {failure}")
        sys.exit(1)
    print("PASS: all V4 values match the five-seed record.")

    if args.render:
        out, size = render(rows, Path(args.out) if args.out else None)
        print(f"Wrote {out} ({size[0]:.2f} x {size[1]:.2f} in, vector PDF)")


if __name__ == "__main__":
    main()
