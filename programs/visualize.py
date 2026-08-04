"""
Visualization module for iterative topological filtration results.

Produces publication-quality figures for conference presentation:
  1. Baseline comparison table (RAW vs TDA, reproducing paper's format)
  2. Convergence curves across iterations
  3. Sanitized pool composition analysis
  4. Algorithm comparison bar charts
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


RESULTS_DIR = Path(r"C:\TDA\results")
FIGURES_DIR = Path(r"C:\TDA\figures")


def load_results(dataset_name="unsw_nb15"):
    """Load saved JSON results."""
    path = RESULTS_DIR / f"iterative_results_{dataset_name}.json"
    with open(path) as f:
        return json.load(f)


def plot_convergence_curves(results, dataset_name="UNSW-NB15"):
    """
    Plot per-iteration convergence metrics for all algorithms.

    Creates a 2x2 subplot figure:
      - Top left: Residual size over iterations
      - Top right: Poisoned samples remaining in residual
      - Bottom left: Green cluster % per iteration
      - Bottom right: Red cluster % per iteration
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Iterative Topological Filtration — Convergence\n{dataset_name}",
                 fontsize=14, fontweight="bold")

    colors = {"DBSCAN": "#1f77b4", "HDBSCAN": "#ff7f0e",
              "OPTICS": "#2ca02c", "MeanShift": "#d62728"}

    for algo_name, algo_data in results.items():
        log = algo_data["iteration_log"]
        if not log:
            continue

        iters = [e["iteration"] for e in log]
        color = colors.get(algo_name, "gray")

        # Top left: Residual size
        axes[0, 0].plot(iters, [e["n_residual"] for e in log],
                        marker="o", label=algo_name, color=color, linewidth=2)
        axes[0, 0].set_ylabel("Residual Size")
        axes[0, 0].set_title("Residual Data Remaining")

        # Top right: Poisoned remaining
        axes[0, 1].plot(iters, [e["n_poisoned_in_residual"] for e in log],
                        marker="s", label=algo_name, color=color, linewidth=2)
        axes[0, 1].set_ylabel("Poisoned Samples")
        axes[0, 1].set_title("Poisoned Data in Residual")

        # Bottom left: Green %
        axes[1, 0].plot(iters, [e["green_pct"] for e in log],
                        marker="^", label=algo_name, color=color, linewidth=2)
        axes[1, 0].set_ylabel("Green Cluster %")
        axes[1, 0].set_title("Clean Data Identified per Iteration")

        # Bottom right: Red %
        axes[1, 1].plot(iters, [e["red_pct"] for e in log],
                        marker="D", label=algo_name, color=color, linewidth=2)
        axes[1, 1].set_ylabel("Red Cluster %")
        axes[1, 1].set_title("Poisoned Data Identified per Iteration")

    for ax in axes.flat:
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = FIGURES_DIR / "convergence_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_algorithm_comparison(results, dataset_name="UNSW-NB15"):
    """
    Bar chart comparing final outcomes across algorithms.

    Shows: sanitized pool size, poisoned pool size, residual size,
    sanitized purity, and poison capture rate.
    """
    # We need is_poisoned to compute metrics — load from iteration log
    algorithms = list(results.keys())
    n_algos = len(algorithms)

    # Extract metrics from iteration logs
    san_sizes = []
    pp_sizes = []
    res_sizes = []
    n_iters_list = []

    for algo in algorithms:
        r = results[algo]
        san_sizes.append(len(r["sanitized_indices"]))
        pp_sizes.append(len(r["poisoned_pool_indices"]))
        res_sizes.append(len(r["residual_indices"]))
        n_iters_list.append(len(r["iteration_log"]))

    x = np.arange(n_algos)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, san_sizes, width, label="Sanitized (Clean Pool)",
                   color="#2ca02c", alpha=0.85)
    bars2 = ax.bar(x, pp_sizes, width, label="Poisoned Pool (Removed)",
                   color="#d62728", alpha=0.85)
    bars3 = ax.bar(x + width, res_sizes, width, label="Residual (Unseparated)",
                   color="#ffcc00", alpha=0.85)

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Iterative Topological Filtration — Final Outcome\n{dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n({n}iter)" for a, n in zip(algorithms, n_iters_list)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = FIGURES_DIR / "algorithm_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_wasserstein_convergence(results, dataset_name="UNSW-NB15"):
    """
    Plot Wasserstein distance between successive iterations.
    This measures how much the topological structure changes as we
    iteratively remove clusters — the core mathematical contribution.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"DBSCAN": "#1f77b4", "HDBSCAN": "#ff7f0e",
              "OPTICS": "#2ca02c", "MeanShift": "#d62728"}

    has_data = False
    for algo_name, algo_data in results.items():
        log = algo_data["iteration_log"]
        if not log:
            continue

        iters = [e["iteration"] for e in log]
        w_dists = [e["wasserstein_distance"] for e in log]

        # Filter out NaN values (first iteration has no distance)
        valid = [(i, w) for i, w in zip(iters, w_dists) if not np.isnan(w)]
        if valid:
            vi, vw = zip(*valid)
            ax.plot(vi, vw, marker="o", label=algo_name,
                    color=colors.get(algo_name, "gray"), linewidth=2)
            has_data = True

    if has_data:
        ax.set_xlabel("Iteration", fontsize=12)
        ax.set_ylabel("Wasserstein Distance (from previous iteration)", fontsize=12)
        ax.set_title(f"Topological Convergence — Wasserstein Distance Between Iterations\n{dataset_name}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        path = FIGURES_DIR / "wasserstein_convergence.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
    else:
        plt.close()
        print("  No valid Wasserstein distance data to plot.")


def plot_iteration_snapshots(results, algo="OPTICS", dataset_name="UNSW-NB15"):
    """
    Stacked bar chart showing cluster composition at each iteration
    for a single algorithm. Uses the paper's color scheme.
    """
    if algo not in results or not results[algo]["iteration_log"]:
        print(f"  No data for {algo}")
        return

    log = results[algo]["iteration_log"]
    iters = [e["iteration"] for e in log]
    greens = [e["n_green_clusters"] for e in log]
    reds = [e["n_red_clusters"] for e in log]
    yellows = [e["n_yellow_clusters"] for e in log]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(iters))
    width = 0.6

    ax.bar(x, greens, width, label="Green (Clean Only)", color="#2ca02c")
    ax.bar(x, reds, width, bottom=greens, label="Red (Poisoned Only)", color="#d62728")
    ax.bar(x, yellows, width,
           bottom=[g + r for g, r in zip(greens, reds)],
           label="Yellow (Mixed)", color="#ffcc00")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Number of Clusters", fontsize=12)
    ax.set_title(f"Cluster Composition per Iteration — {algo}\n{dataset_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = FIGURES_DIR / f"iteration_snapshots_{algo.lower()}.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_all_figures(dataset_name="unsw_nb15", display_name="UNSW-NB15"):
    """Generate all figures from saved results."""
    FIGURES_DIR.mkdir(exist_ok=True)

    print(f"\nGenerating figures for {display_name}...")
    results = load_results(dataset_name)

    plot_convergence_curves(results, display_name)
    plot_algorithm_comparison(results, display_name)
    plot_wasserstein_convergence(results, display_name)
    plot_iteration_snapshots(results, "OPTICS", display_name)
    plot_iteration_snapshots(results, "HDBSCAN", display_name)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_figures()
