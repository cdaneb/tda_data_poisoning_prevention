"""
Comparative visualization: Baseline (single-pass) vs Iterative filtration.

Produces the key figures for the conference presentation:
  1. Side-by-side bar chart: baseline vs iterative poison capture (both datasets)
  2. Combined convergence plot across datasets
  3. Summary statistics table as a figure (for poster/slides)
  4. Per-seed spread plots showing statistical robustness
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path(r"C:\TDA\results")
FIGURES_DIR = Path(r"C:\TDA\figures")
FIGURES_DIR.mkdir(exist_ok=True)

ALGORITHMS = ["DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]
ALGO_COLORS = {
    "DBSCAN": "#1f77b4",
    "HDBSCAN": "#ff7f0e",
    "OPTICS": "#2ca02c",
    "MeanShift": "#d62728",
}


def load_aggregated(dataset_name):
    """Load aggregated multi-seed results."""
    safe_name = dataset_name.replace("-", "_").replace(" ", "_").lower()
    path = RESULTS_DIR / f"multi_seed_aggregated_{safe_name}.json"
    with open(path) as f:
        return json.load(f)


def load_per_seed(dataset_name):
    """Load per-seed multi-seed results."""
    safe_name = dataset_name.replace("-", "_").replace(" ", "_").lower()
    path = RESULTS_DIR / f"multi_seed_per_seed_{safe_name}.json"
    with open(path) as f:
        return json.load(f)


def plot_baseline_vs_iterative(unsw_agg, cicids_agg):
    """
    Figure 1: Side-by-side grouped bar chart comparing poison capture rates.
    Baseline (single-pass) vs Iterative, for each algorithm, across both datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Poison Capture Rate: Single-Pass Baseline vs Iterative Filtration",
                 fontsize=14, fontweight="bold")

    for ax, agg, title in zip(axes, [unsw_agg, cicids_agg],
                               ["UNSW-NB15", "CICIDS2017"]):
        x = np.arange(len(ALGORITHMS))
        width = 0.35

        baseline_means = []
        baseline_stds = []
        iterative_means = []
        iterative_stds = []

        for algo in ALGORITHMS:
            b = agg["baseline"].get(algo, {}).get("red_capture_pct", {"mean": 0, "std": 0})
            i = agg["iterative"].get(algo, {}).get("poison_capture_pct", {"mean": 0, "std": 0})
            baseline_means.append(b["mean"])
            baseline_stds.append(b["std"])
            iterative_means.append(i["mean"])
            iterative_stds.append(i["std"])

        bars1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                       label="Baseline (Single-Pass)", color="#7faadb",
                       edgecolor="black", linewidth=0.5, capsize=4)
        bars2 = ax.bar(x + width/2, iterative_means, width, yerr=iterative_stds,
                       label="Iterative Filtration", color="#2ca02c",
                       edgecolor="black", linewidth=0.5, capsize=4)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Poison Capture (%)" if ax == axes[0] else "")
        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHMS, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0.5:
                    ax.annotate(f"{height:.1f}%",
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "baseline_vs_iterative_capture.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sanitized_purity(unsw_agg, cicids_agg):
    """
    Figure 2: Sanitized pool purity and poisoned pool precision across algorithms.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Iterative Filtration — Classification Quality",
                 fontsize=14, fontweight="bold")

    for ax, agg, title in zip(axes, [unsw_agg, cicids_agg],
                               ["UNSW-NB15", "CICIDS2017"]):
        x = np.arange(len(ALGORITHMS))
        width = 0.35

        purity_means = []
        purity_stds = []
        precision_means = []
        precision_stds = []

        for algo in ALGORITHMS:
            it = agg["iterative"].get(algo, {})
            sp = it.get("sanitized_purity", {"mean": 100, "std": 0})
            pp = it.get("poisoned_pool_precision", {"mean": 100, "std": 0})
            purity_means.append(sp["mean"])
            purity_stds.append(sp["std"])
            precision_means.append(pp["mean"])
            precision_stds.append(pp["std"])

        ax.bar(x - width/2, purity_means, width, yerr=purity_stds,
               label="Sanitized Pool Purity", color="#2ca02c",
               edgecolor="black", linewidth=0.5, capsize=4)
        ax.bar(x + width/2, precision_means, width, yerr=precision_stds,
               label="Poisoned Pool Precision", color="#d62728",
               edgecolor="black", linewidth=0.5, capsize=4)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(ALGORITHMS, fontsize=10)
        ax.set_ylim([0, 110])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / "purity_and_precision.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_seed_spread(unsw_seeds, cicids_seeds):
    """
    Figure 3: Strip/swarm plot showing per-seed poison capture rates.
    Demonstrates statistical robustness — results are consistent across seeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    fig.suptitle("Per-Seed Poison Capture Rate — Statistical Robustness",
                 fontsize=14, fontweight="bold")

    for ax, seeds, title in zip(axes, [unsw_seeds, cicids_seeds],
                                 ["UNSW-NB15", "CICIDS2017"]):
        for j, algo in enumerate(ALGORITHMS):
            # Baseline values
            baseline_vals = [s["baseline"][algo]["red_capture_pct"]
                           for s in seeds if algo in s["baseline"]]
            # Iterative values
            iterative_vals = [s["iterative"][algo]["poison_capture_pct"]
                            for s in seeds if algo in s["iterative"]]

            # Plot individual seed points
            x_base = np.full(len(baseline_vals), j - 0.15)
            x_iter = np.full(len(iterative_vals), j + 0.15)

            # Add small jitter for visibility
            jitter = np.random.RandomState(42).uniform(-0.05, 0.05, len(baseline_vals))
            jitter2 = np.random.RandomState(43).uniform(-0.05, 0.05, len(iterative_vals))

            ax.scatter(x_base + jitter, baseline_vals, color="#7faadb",
                      s=40, zorder=3, edgecolors="black", linewidth=0.5,
                      label="Baseline" if j == 0 else "")
            ax.scatter(x_iter + jitter2, iterative_vals, color="#2ca02c",
                      s=40, zorder=3, edgecolors="black", linewidth=0.5,
                      label="Iterative" if j == 0 else "")

            # Add mean lines
            if baseline_vals:
                ax.hlines(np.mean(baseline_vals), j - 0.25, j - 0.05,
                         colors="#7faadb", linewidth=2)
            if iterative_vals:
                ax.hlines(np.mean(iterative_vals), j + 0.05, j + 0.25,
                         colors="#2ca02c", linewidth=2)

        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Algorithm")
        ax.set_ylabel("Poison Capture (%)" if ax == axes[0] else "")
        ax.set_xticks(range(len(ALGORITHMS)))
        ax.set_xticklabels(ALGORITHMS, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / "per_seed_spread.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_table_figure(unsw_agg, cicids_agg):
    """
    Figure 4: Render the summary statistics as a formatted table figure.
    This can be placed directly on a poster or slide.
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis('off')

    # Build table data
    col_labels = [
        "Algorithm",
        "UNSW Baseline\nCapture%",
        "UNSW Iterative\nCapture%",
        "UNSW\nImprovement",
        "CICIDS Baseline\nCapture%",
        "CICIDS Iterative\nCapture%",
        "CICIDS\nImprovement",
    ]

    table_data = []
    for algo in ALGORITHMS:
        ub = unsw_agg["baseline"].get(algo, {}).get("red_capture_pct", {"mean": 0, "std": 0})
        ui = unsw_agg["iterative"].get(algo, {}).get("poison_capture_pct", {"mean": 0, "std": 0})
        cb = cicids_agg["baseline"].get(algo, {}).get("red_capture_pct", {"mean": 0, "std": 0})
        ci = cicids_agg["iterative"].get(algo, {}).get("poison_capture_pct", {"mean": 0, "std": 0})

        u_imp = ui["mean"] - ub["mean"]
        c_imp = ci["mean"] - cb["mean"]

        table_data.append([
            algo,
            f"{ub['mean']:.1f} ± {ub['std']:.1f}",
            f"{ui['mean']:.1f} ± {ui['std']:.1f}",
            f"{u_imp:+.1f}%",
            f"{cb['mean']:.1f} ± {cb['std']:.1f}",
            f"{ci['mean']:.1f} ± {ci['std']:.1f}",
            f"{c_imp:+.1f}%",
        ])

    table = ax.table(cellText=table_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

    # Alternate row colors
    for i in range(len(table_data)):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    # Highlight improvement columns
    for i in range(len(table_data)):
        for j in [3, 6]:  # improvement columns
            val = table_data[i][j]
            if val.startswith("+") and float(val.replace("%", "").replace("+", "")) > 0:
                table[i + 1, j].set_text_props(color="#2ca02c", fontweight="bold")

    ax.set_title("Iterative Topological Filtration — Cross-Dataset Results (mean ± std, n=5 seeds)",
                 fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = FIGURES_DIR / "summary_table.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all_comparison_figures():
    """Generate all comparative figures from multi-seed results."""
    print("\nGenerating comparative figures...")

    # Load aggregated results
    try:
        unsw_agg = load_aggregated("UNSW-NB15")
        print("  Loaded UNSW-NB15 aggregated results")
    except FileNotFoundError:
        print("  ERROR: UNSW-NB15 aggregated results not found. Run run_multi_seed.py first.")
        return

    try:
        cicids_agg = load_aggregated("CICIDS2017")
        print("  Loaded CICIDS2017 aggregated results")
    except FileNotFoundError:
        print("  ERROR: CICIDS2017 aggregated results not found. Run run_multi_seed.py first.")
        return

    # Load per-seed results
    unsw_seeds = load_per_seed("UNSW-NB15")
    cicids_seeds = load_per_seed("CICIDS2017")

    # Generate figures
    plot_baseline_vs_iterative(unsw_agg, cicids_agg)
    plot_sanitized_purity(unsw_agg, cicids_agg)
    plot_per_seed_spread(unsw_seeds, cicids_seeds)
    plot_summary_table_figure(unsw_agg, cicids_agg)

    print(f"\nAll comparative figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_comparison_figures()
