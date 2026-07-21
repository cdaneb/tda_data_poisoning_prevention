"""
Full iterative experiment runner.

Runs the iterative topological filtration algorithm on UNSW-NB15 dataset
with all four clustering algorithms, tracks convergence, and produces
comprehensive results for analysis and visualization.
"""
import numpy as np
import json
import time
from pathlib import Path
from data_loader import load_unsw
from poison import poison_dataset
from iterative_filter import iterative_filter
from results_io import convert_for_json


def run_iterative_experiment(dataset_name, X, y, max_samples=5000,
                             poison_rate=0.10, max_iterations=10):
    """
    Run the iterative filtration experiment with all four algorithms.
    """
    print(f"\n{'='*70}")
    print(f"ITERATIVE TOPOLOGICAL FILTRATION EXPERIMENT")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # Subsample if needed
    if len(X) > max_samples:
        print(f"\nSubsampling from {len(X)} to {max_samples}...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X[idx]
        y = y[idx]

    # Poison the data
    print(f"\nPoisoning data (rate={poison_rate})...")
    X_poisoned, y_poisoned, is_poisoned = poison_dataset(X, y, poison_rate=poison_rate)
    print(f"  Total: {len(X_poisoned)} samples ({is_poisoned.sum()} poisoned)")

    # Run iterative filter with each algorithm
    algorithms = ["DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]
    all_results = {}

    for algo in algorithms:
        print(f"\n{'~'*70}")
        print(f"Running iterative filter with {algo}")
        print(f"{'~'*70}")

        t0 = time.time()
        result = iterative_filter(
            X_poisoned, is_poisoned,
            algorithm=algo,
            max_iterations=max_iterations,
            verbose=True
        )
        total_time = time.time() - t0

        result["total_time_s"] = total_time
        result["algorithm"] = algo
        all_results[algo] = result

    # Print comparison summary
    print(f"\n\n{'='*70}")
    print(f"COMPARISON SUMMARY — {dataset_name}")
    print(f"{'='*70}")

    total_poisoned = int(is_poisoned.sum())
    total_clean = int((~is_poisoned).sum())

    print(f"\nDataset: {len(X_poisoned)} samples ({total_poisoned} poisoned, {total_clean} clean)")
    print(f"\n{'Algorithm':<12} {'Iters':>5} {'Sanitized':>10} {'San.Purity':>11} "
          f"{'PoisonPool':>10} {'PP.Prec':>8} {'Residual':>9} {'PoisonCapt':>11} {'Time':>7}")
    print("-" * 95)

    for algo in algorithms:
        r = all_results[algo]
        n_san = len(r["sanitized_indices"])
        n_pp = len(r["poisoned_pool_indices"])
        n_res = len(r["residual_indices"])
        n_iters = len(r["iteration_log"])

        # Sanitized pool purity
        if n_san > 0:
            san_poison = is_poisoned[r["sanitized_indices"]].sum()
            san_purity = (1 - san_poison / n_san) * 100
        else:
            san_purity = 0

        # Poisoned pool precision
        if n_pp > 0:
            pp_true = is_poisoned[r["poisoned_pool_indices"]].sum()
            pp_prec = pp_true / n_pp * 100
        else:
            pp_true = 0
            pp_prec = 0

        # Poison capture rate
        poison_capt = pp_true / total_poisoned * 100 if total_poisoned > 0 else 0

        print(f"{algo:<12} {n_iters:>5} {n_san:>10} {san_purity:>10.1f}% "
              f"{n_pp:>10} {pp_prec:>7.1f}% {n_res:>9} {poison_capt:>10.1f}% "
              f"{r['total_time_s']:>6.1f}s")

    # Print per-iteration convergence for best algorithm
    print(f"\n\n{'='*70}")
    print(f"PER-ITERATION CONVERGENCE LOG (OPTICS)")
    print(f"{'='*70}")

    if "OPTICS" in all_results and all_results["OPTICS"]["iteration_log"]:
        log = all_results["OPTICS"]["iteration_log"]
        print(f"\n{'Iter':>4} {'Residual':>9} {'Poisoned':>9} {'Green#':>7} {'Red#':>6} "
              f"{'Yellow#':>8} {'GreenPct':>9} {'RedPct':>8} {'W-Dist':>10}")
        print("-" * 80)
        for entry in log:
            w_str = f"{entry['wasserstein_distance']:.6f}" if not np.isnan(entry['wasserstein_distance']) else "N/A"
            print(f"{entry['iteration']:>4} {entry['n_residual']:>9} "
                  f"{entry['n_poisoned_in_residual']:>9} "
                  f"{entry['n_green_clusters']:>7} {entry['n_red_clusters']:>6} "
                  f"{entry['n_yellow_clusters']:>8} "
                  f"{entry['green_pct']:>8.2f}% {entry['red_pct']:>7.2f}% "
                  f"{w_str:>10}")

    # Save results to JSON for later visualization
    output_dir = Path(r"C:\TDA\results")
    output_dir.mkdir(exist_ok=True)

    json_results = {}
    for algo, r in all_results.items():
        json_results[algo] = {
            "sanitized_indices": convert_for_json(r["sanitized_indices"]),
            "poisoned_pool_indices": convert_for_json(r["poisoned_pool_indices"]),
            "residual_indices": convert_for_json(r["residual_indices"]),
            "iteration_log": r["iteration_log"],
            "total_time_s": r["total_time_s"],
        }

    results_path = output_dir / f"iterative_results_{dataset_name.replace('-','_').lower()}.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2, default=convert_for_json)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("ITERATIVE TOPOLOGICAL FILTRATION EXPERIMENT")
    print("=" * 70)

    X_unsw, y_unsw = load_unsw()

    # Start with 5000 samples for feasibility.
    # Increase max_samples once verified.
    all_results = run_iterative_experiment(
        "UNSW-NB15", X_unsw, y_unsw,
        max_samples=5000,
        poison_rate=0.10,
        max_iterations=10
    )

    print("\n" + "=" * 70)
    print("ITERATIVE EXPERIMENT COMPLETE")
    print("=" * 70)
