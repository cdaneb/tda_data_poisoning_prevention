"""
Multi-seed experiment runner.

Runs the iterative topological filtration experiment multiple times
with different random seeds to establish statistical robustness.
Reports means and standard deviations for all key metrics.

Runs on BOTH datasets: UNSW-NB15 and CICIDS2017.
"""
import numpy as np
import json
import time
from pathlib import Path
from data_loader import load_unsw, load_cicids
from poison import poison_dataset
from iterative_filter import iterative_filter
from clustering import run_all_clustering, classify_clusters
from results_io import convert_for_json


# ============================================================
# CONFIGURATION
# ============================================================
SEEDS = [42, 123, 456, 789, 1024]  # 5 random seeds
MAX_SAMPLES = 5000                  # samples per run
POISON_RATE = 0.10
MAX_ITERATIONS = 10
ALGORITHMS = ["DBSCAN", "HDBSCAN", "OPTICS", "MeanShift"]
RESULTS_DIR = Path(r"C:\TDA\results")
# ============================================================


def run_single_seed_experiment(X_full, y_full, seed, dataset_name, algorithms):
    """
    Run the full experiment (baseline + iterative) for a single random seed.

    Returns a dict with all metrics for this seed.
    """
    rng = np.random.RandomState(seed)

    # Subsample
    if len(X_full) > MAX_SAMPLES:
        idx = rng.choice(len(X_full), size=MAX_SAMPLES, replace=False)
        X = X_full[idx]
        y = y_full[idx]
    else:
        X = X_full.copy()
        y = y_full.copy()

    # Poison (use the seed for reproducibility)
    X_poisoned, y_poisoned, is_poisoned = poison_dataset(
        X, y, poison_rate=POISON_RATE, random_state=seed
    )

    total_poisoned = int(is_poisoned.sum())
    total_clean = int((~is_poisoned).sum())

    seed_results = {
        "seed": seed,
        "n_samples": len(X),
        "n_poisoned": total_poisoned,
        "n_clean": total_clean,
        "baseline": {},
        "iterative": {},
    }

    # ---- BASELINE (single-pass TDA + clustering) ----
    from tda_pipeline import extract_tda_features

    print(f"    Extracting TDA features for baseline...")
    X_tda, pipeline = extract_tda_features(X_poisoned)

    for algo in algorithms:
        print(f"    Baseline clustering with {algo}...")
        all_cluster_results = run_all_clustering(X_tda)
        labels = all_cluster_results[algo]
        cluster_info, summary = classify_clusters(labels, is_poisoned)

        # Compute baseline metrics
        red_capture = 0
        if total_poisoned > 0:
            red_samples = []
            for ci in cluster_info:
                if ci["color"] == "Red":
                    red_samples.append(ci["n_poisoned"])
            red_capture = sum(red_samples) / total_poisoned * 100

        seed_results["baseline"][algo] = {
            "green_pct": summary["green_pct"],
            "red_pct": summary["red_pct"],
            "red_capture_pct": red_capture,
            "n_green_clusters": summary["colors"].get("Green", 0),
            "n_red_clusters": summary["colors"].get("Red", 0),
            "n_yellow_clusters": summary["colors"].get("Yellow", 0),
        }

    # ---- ITERATIVE ----
    for algo in algorithms:
        print(f"    Iterative filter with {algo}...")
        result = iterative_filter(
            X_poisoned, is_poisoned,
            algorithm=algo,
            max_iterations=MAX_ITERATIONS,
            verbose=False
        )

        n_san = len(result["sanitized_indices"])
        n_pp = len(result["poisoned_pool_indices"])
        n_res = len(result["residual_indices"])
        n_iters = len(result["iteration_log"])

        # Sanitized pool purity
        if n_san > 0:
            san_poison = int(is_poisoned[result["sanitized_indices"]].sum())
            san_purity = (1 - san_poison / n_san) * 100
        else:
            san_poison = 0
            san_purity = 100.0

        # Poisoned pool precision
        if n_pp > 0:
            pp_true = int(is_poisoned[result["poisoned_pool_indices"]].sum())
            pp_prec = pp_true / n_pp * 100
        else:
            pp_true = 0
            pp_prec = 100.0

        # Poison capture rate
        poison_capt = pp_true / total_poisoned * 100 if total_poisoned > 0 else 0

        # Wasserstein distances
        w_dists = [e["wasserstein_distance"] for e in result["iteration_log"]
                   if not np.isnan(e["wasserstein_distance"])]

        seed_results["iterative"][algo] = {
            "n_iterations": n_iters,
            "n_sanitized": n_san,
            "n_poisoned_pool": n_pp,
            "n_residual": n_res,
            "sanitized_purity": san_purity,
            "poisoned_pool_precision": pp_prec,
            "poison_capture_pct": poison_capt,
            "false_positives": san_poison,
            "false_negatives": n_pp - pp_true if n_pp > 0 else 0,
            "wasserstein_distances": w_dists,
            "iteration_log": result["iteration_log"],
        }

    return seed_results


def aggregate_results(all_seed_results, algorithms):
    """
    Compute means and standard deviations across seeds for each algorithm.
    """
    agg = {"baseline": {}, "iterative": {}}

    for algo in algorithms:
        # Baseline aggregation
        baseline_metrics = {
            "green_pct": [],
            "red_pct": [],
            "red_capture_pct": [],
            "n_green_clusters": [],
            "n_red_clusters": [],
        }
        for sr in all_seed_results:
            if algo in sr["baseline"]:
                for k in baseline_metrics:
                    baseline_metrics[k].append(sr["baseline"][algo][k])

        agg["baseline"][algo] = {}
        for k, vals in baseline_metrics.items():
            if vals:
                agg["baseline"][algo][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

        # Iterative aggregation
        iter_metrics = {
            "n_iterations": [],
            "n_sanitized": [],
            "n_poisoned_pool": [],
            "n_residual": [],
            "sanitized_purity": [],
            "poisoned_pool_precision": [],
            "poison_capture_pct": [],
            "false_positives": [],
            "false_negatives": [],
        }
        for sr in all_seed_results:
            if algo in sr["iterative"]:
                for k in iter_metrics:
                    iter_metrics[k].append(sr["iterative"][algo][k])

        agg["iterative"][algo] = {}
        for k, vals in iter_metrics.items():
            if vals:
                agg["iterative"][algo][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "values": vals,
                }

    return agg


def print_aggregated_report(dataset_name, agg, algorithms):
    """Print a formatted report of aggregated results."""
    print(f"\n{'='*90}")
    print(f"AGGREGATED RESULTS — {dataset_name} ({len(SEEDS)} seeds)")
    print(f"{'='*90}")

    # Baseline summary
    print(f"\n--- BASELINE (Single-Pass TDA + Clustering) ---")
    print(f"{'Algorithm':<12} {'Green% (mean±std)':>22} {'Red% (mean±std)':>22} "
          f"{'RedCapture% (mean±std)':>25}")
    print("-" * 85)
    for algo in algorithms:
        b = agg["baseline"].get(algo, {})
        g = b.get("green_pct", {"mean": 0, "std": 0})
        r = b.get("red_pct", {"mean": 0, "std": 0})
        rc = b.get("red_capture_pct", {"mean": 0, "std": 0})
        print(f"{algo:<12} {g['mean']:>10.2f} ± {g['std']:<8.2f} "
              f"{r['mean']:>10.2f} ± {r['std']:<8.2f} "
              f"{rc['mean']:>10.2f} ± {rc['std']:<8.2f}")

    # Iterative summary
    print(f"\n--- ITERATIVE TOPOLOGICAL FILTRATION ---")
    print(f"{'Algorithm':<12} {'Iters':>12} {'Sanitized':>12} {'Purity%':>16} "
          f"{'PoisonCapt%':>16} {'Precision%':>16}")
    print("-" * 90)
    for algo in algorithms:
        it = agg["iterative"].get(algo, {})
        ni = it.get("n_iterations", {"mean": 0, "std": 0})
        ns = it.get("n_sanitized", {"mean": 0, "std": 0})
        sp = it.get("sanitized_purity", {"mean": 0, "std": 0})
        pc = it.get("poison_capture_pct", {"mean": 0, "std": 0})
        pp = it.get("poisoned_pool_precision", {"mean": 0, "std": 0})
        print(f"{algo:<12} {ni['mean']:>5.1f}±{ni['std']:<4.1f} "
              f"{ns['mean']:>6.0f}±{ns['std']:<4.0f} "
              f"{sp['mean']:>8.1f}±{sp['std']:<5.1f} "
              f"{pc['mean']:>8.1f}±{pc['std']:<5.1f} "
              f"{pp['mean']:>8.1f}±{pp['std']:<5.1f}")

    # Improvement comparison
    print(f"\n--- IMPROVEMENT: Iterative vs Baseline (Poison Capture %) ---")
    print(f"{'Algorithm':<12} {'Baseline':>12} {'Iterative':>12} {'Improvement':>14}")
    print("-" * 55)
    for algo in algorithms:
        b_rc = agg["baseline"].get(algo, {}).get("red_capture_pct", {"mean": 0})["mean"]
        i_pc = agg["iterative"].get(algo, {}).get("poison_capture_pct", {"mean": 0})["mean"]
        diff = i_pc - b_rc
        print(f"{algo:<12} {b_rc:>11.2f}% {i_pc:>11.2f}% {diff:>+12.2f}%")


def run_dataset_experiment(dataset_name, X_full, y_full):
    """Run the full multi-seed experiment on one dataset."""
    print(f"\n{'#'*90}")
    print(f"# DATASET: {dataset_name}")
    print(f"# Seeds: {SEEDS}")
    print(f"# Max samples per seed: {MAX_SAMPLES}")
    print(f"# Poison rate: {POISON_RATE}")
    print(f"# Max iterations: {MAX_ITERATIONS}")
    print(f"{'#'*90}")

    all_seed_results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*70}")
        print(f"  Seed {seed} ({i+1}/{len(SEEDS)}) — {dataset_name}")
        print(f"{'='*70}")

        t0 = time.time()
        seed_result = run_single_seed_experiment(
            X_full, y_full, seed, dataset_name, ALGORITHMS
        )
        elapsed = time.time() - t0

        print(f"  Seed {seed} completed in {elapsed:.1f}s")
        all_seed_results.append(seed_result)

    # Aggregate across seeds
    agg = aggregate_results(all_seed_results, ALGORITHMS)

    # Print report
    print_aggregated_report(dataset_name, agg, ALGORITHMS)

    # Save all results
    RESULTS_DIR.mkdir(exist_ok=True)
    safe_name = dataset_name.replace("-", "_").replace(" ", "_").lower()

    # Save per-seed results
    per_seed_path = RESULTS_DIR / f"multi_seed_per_seed_{safe_name}.json"
    with open(per_seed_path, "w") as f:
        json.dump(all_seed_results, f, indent=2, default=convert_for_json)
    print(f"\n  Per-seed results saved to: {per_seed_path}")

    # Save aggregated results
    agg_path = RESULTS_DIR / f"multi_seed_aggregated_{safe_name}.json"
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2, default=convert_for_json)
    print(f"  Aggregated results saved to: {agg_path}")

    return all_seed_results, agg


if __name__ == "__main__":
    print("=" * 90)
    print("MULTI-SEED EXPERIMENT: Iterative Topological Filtration")
    print(f"Seeds: {SEEDS}")
    print("=" * 90)

    # ---- UNSW-NB15 ----
    print("\nLoading UNSW-NB15...")
    X_unsw, y_unsw = load_unsw()
    unsw_seeds, unsw_agg = run_dataset_experiment("UNSW-NB15", X_unsw, y_unsw)

    # ---- CICIDS2017 ----
    # This is a large dataset (~4.9 GB CSV). We load with a max_samples cap.
    # load_cicids() may take a few minutes to read the CSV.
    print("\nLoading CICIDS2017 (this may take several minutes for the large CSV)...")
    X_cicids, y_cicids = load_cicids(max_samples=50000)
    cicids_seeds, cicids_agg = run_dataset_experiment("CICIDS2017", X_cicids, y_cicids)

    print("\n" + "=" * 90)
    print("ALL MULTI-SEED EXPERIMENTS COMPLETE")
    print("=" * 90)
