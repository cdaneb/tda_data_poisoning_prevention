"""Plot the Phase M8 read-only purity sweep; this is a diagnostic, not poster art."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "results" / "phase_m_m8_purity_sweep.json"
OUT = ROOT / "figures" / "figure_m8_purity_sweep.pdf"
ORDER = ["block_reversal", "block_swap", "transpositions", "cyclic_shift", "noise"]
LABELS = {
    "block_reversal": "Block reversal",
    "block_swap": "Block swap",
    "transpositions": "Random swaps",
    "cyclic_shift": "Cyclic shift",
    "noise": "Noise (orphaned N)",
}
COLORS = ["#C14E3D", "#D2863A", "#537A95", "#5A9A72", "#8465A8"]


def main():
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    levels = [item["label"] for item in payload["_meta"]["purity_levels"]]
    x = range(len(levels))
    fig, ax = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)

    for family, color in zip(ORDER, COLORS):
        record = payload["families"][family]
        curve = record["capture_curve"]
        values = [curve[level]["mean"] for level in levels]
        sd = [curve[level]["population_sd"] for level in levels]
        ax.errorbar(x, values, yerr=sd, marker="o", lw=2.4, capsize=3.5,
                    color=color, label=LABELS[family])
        ceiling = record["unclustered_ceiling"]["mean"]
        ax.axhline(ceiling, color=color, lw=1.05, ls="--", alpha=.55)
        union = record["union_ceiling_at_100pct_only"]["mean"]
        ax.scatter([0], [union], marker="x", s=52, linewidths=1.8, color=color, zorder=4)

    ax.axhline(40, color="#333333", lw=1, ls=":", label="40% reference")
    ax.set_xticks(list(x), levels)
    ax.set_xlim(-.2, len(levels) - .8)
    ax.set_ylim(-2, 70)
    ax.set_ylabel("Poison capture (%)")
    ax.set_xlabel("Minimum poison purity in a cluster")
    ax.set_title("M8: relaxed-purity capture remains far below each family’s unclustered ceiling")
    ax.grid(axis="y", alpha=.2)
    ax.legend(ncol=2, loc="upper right", frameon=False)
    ax.text(.015, .03,
            "Solid curves: capture. Dashed lines: unclustered ceilings (valid at every threshold).\n"
            "x markers: union ceilings at exact 100% only; they are not relaxed-purity bounds.",
            transform=ax.transAxes, fontsize=8.4, va="bottom",
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": .93, "pad": 4})
    fig.savefig(OUT, dpi=300)
    print(OUT)


if __name__ == "__main__":
    main()
