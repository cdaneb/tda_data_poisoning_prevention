"""Publication figures generated exclusively from the merged confirmation artifact."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "results/clean_novelty_confirmation.json"
OUT_CAPTURE = ROOT / "figures/clean_novelty_confirmation_capture.png"
OUT_EFFECT = ROOT / "figures/clean_novelty_confirmation_paired_effects.png"


def main():
    result = json.load(open(DATA))
    if not result["complete"] or len(result["records"]) != 90:
        raise RuntimeError("refusing to plot a partial confirmation")
    records = result["records"]
    populations = ["unsw_matched", "cicids_matched", "cicids_scale"]
    titles = ["UNSW unseen seeds", "CICIDS matched size", "CICIDS scale"]
    colors = {"control": "#0072B2", "stack": "#009E73"}
    labels = {"control": "60 features", "stack": "540 features"}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, pop, title in zip(axes, populations, titles):
        for rep in ("control", "stack"):
            rr = [r for r in records if r["population"] == pop and r["representation"] == rep]
            x = np.array([r["detectors"]["isolation_forest"]["budgets"]["0.05"]["clean_removal_rate"] for r in rr]) * 100
            y = np.array([r["detectors"]["isolation_forest"]["budgets"]["0.05"]["poison_capture"] for r in rr]) * 100
            ax.scatter(x, y, s=22, alpha=.65, color=colors[rep], label=labels[rep])
            ax.scatter(x.mean(), y.mean(), s=75, marker="D", edgecolor="black", linewidth=.7, color=colors[rep])
        ax.axvspan(4, 6, color="0.85", zorder=-2); ax.set_title(title); ax.set_xlabel("Held-out clean removal (%)"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Poison capture (%)"); axes[-1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT_CAPTURE, dpi=300); plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex="col")
    for row, detector in enumerate(("isolation_forest", "knn_distance")):
        for ax, pop, title in zip(axes[row], populations, titles):
            fams = sorted({r["family"] for r in records if r["population"] == pop})
            vals = []
            for family in fams:
                diffs = []
                for seed in sorted({r["seed"] for r in records if r["population"] == pop}):
                    pair = {r["representation"]: r for r in records if r["population"] == pop and r["family"] == family and r["seed"] == seed}
                    diffs.append(100 * (pair["stack"]["detectors"][detector]["budgets"]["0.05"]["poison_capture"] - pair["control"]["detectors"][detector]["budgets"]["0.05"]["poison_capture"]))
                vals.append(diffs)
            ax.boxplot(vals, tick_labels=[x.replace("_", "\n") for x in fams], showmeans=True)
            ax.axhline(0, color="black", lw=.8); ax.grid(axis="y", alpha=.2)
            if row == 0: ax.set_title(title)
            if row == 1: ax.set_xlabel("Attack family")
        axes[row, 0].set_ylabel(("IF" if row == 0 else "kNN") + " 540−60 capture (pp)")
    fig.tight_layout(); fig.savefig(OUT_EFFECT, dpi=300); plt.close(fig)
    print(f"wrote {OUT_CAPTURE.relative_to(ROOT)} and {OUT_EFFECT.relative_to(ROOT)}")


if __name__ == "__main__": main()
