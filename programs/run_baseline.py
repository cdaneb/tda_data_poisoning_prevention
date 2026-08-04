"""
Baseline experiment runner.

Reproduces the core experiment from Monkam et al. (2024):
  1. Load dataset
  2. Create poisoned version
  3. Run clustering on raw poisoned data (expect: only Yellow clusters)
  4. Extract TDA features from poisoned data
  5. Run clustering on TDA-transformed poisoned data (expect: Green + Red clusters emerge)

This validates the paper's central claim: TDA preprocessing enables
unsupervised clustering to separate poisoned from clean data.

NOTE: Due to memory/compute constraints, we run on UNSW-NB15 first (smaller dataset).
For CICIDS2017, use max_samples to subsample.
"""
import numpy as np
import time
from data_loader import load_unsw, load_cicids
from poison import poison_dataset
from tda_pipeline import extract_tda_features
from clustering import run_all_clustering, classify_clusters, print_cluster_report


def run_experiment(dataset_name, X, y, max_samples_for_tda=5000):
    """
    Run the full baseline experiment on one dataset.

    Args:
        dataset_name: str — name for display
        X: np.ndarray (N, 1500) — payload bytes
        y: np.ndarray (N,) — labels
        max_samples_for_tda: int — subsample size for TDA (TDA is compute-heavy)
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {dataset_name}")
    print(f"{'='*70}")

    # Subsample if needed for computational feasibility
    if len(X) > max_samples_for_tda:
        print(f"\nSubsampling from {len(X)} to {max_samples_for_tda} for TDA feasibility...")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), size=max_samples_for_tda, replace=False)
        X = X[idx]
        y = y[idx]
        print(f"  Subsampled: {X.shape}")

    # Step 1: Poison the data
    print(f"\n--- Step 1: Poisoning data ---")
    X_poisoned, y_poisoned, is_poisoned = poison_dataset(X, y, poison_rate=0.10)

    # Step 2: Cluster RAW poisoned data (no TDA)
    print(f"\n--- Step 2: Clustering RAW poisoned data (no TDA) ---")
    print(f"  This should produce ONLY Yellow clusters (no separation).")
    t0 = time.time()
    raw_results = run_all_clustering(X_poisoned.astype(np.float64))
    print(f"  Clustering took {time.time() - t0:.1f}s")

    for algo_name, labels in raw_results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned)
        print_cluster_report(f"{algo_name} (RAW)", cluster_info, summary)

    # Step 3: Extract TDA features from poisoned data
    print(f"\n--- Step 3: Extracting TDA features ---")
    t0 = time.time()
    X_poisoned_tda, pipeline = extract_tda_features(X_poisoned)
    print(f"  TDA extraction took {time.time() - t0:.1f}s")
    print(f"  TDA feature shape: {X_poisoned_tda.shape}")

    # Step 4: Cluster TDA-transformed poisoned data
    print(f"\n--- Step 4: Clustering TDA-transformed poisoned data ---")
    print(f"  This should produce Green AND Red clusters (separation achieved).")
    t0 = time.time()
    tda_results = run_all_clustering(X_poisoned_tda)
    print(f"  Clustering took {time.time() - t0:.1f}s")

    for algo_name, labels in tda_results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned)
        print_cluster_report(f"{algo_name} (TDA)", cluster_info, summary)

    # Step 5: Summary comparison
    print(f"\n--- SUMMARY: {dataset_name} ---")
    print(f"{'Algorithm':<15} {'RAW Green%':>12} {'RAW Red%':>10} {'TDA Green%':>12} {'TDA Red%':>10}")
    print("-" * 60)
    for algo_name in raw_results:
        _, raw_sum = classify_clusters(raw_results[algo_name], is_poisoned)
        _, tda_sum = classify_clusters(tda_results[algo_name], is_poisoned)
        print(f"{algo_name:<15} {raw_sum['green_pct']:>11.2f}% {raw_sum['red_pct']:>9.2f}% "
              f"{tda_sum['green_pct']:>11.2f}% {tda_sum['red_pct']:>9.2f}%")


if __name__ == "__main__":
    print("=" * 70)
    print("BASELINE REPRODUCTION: Monkam et al. (2024)")
    print("TDA + Unsupervised Learning for Data Poisoning Detection")
    print("=" * 70)

    # Run on UNSW-NB15 first (smaller dataset)
    # Use a subsample of 5000 for initial testing.
    # Increase max_samples_for_tda once verified working.
    X_unsw, y_unsw = load_unsw()
    run_experiment("UNSW-NB15", X_unsw, y_unsw, max_samples_for_tda=5000)

    # Optionally run on CICIDS2017 (much larger — subsample heavily for initial test)
    # Uncomment the lines below once UNSW experiment works:
    #
    # X_cicids, y_cicids = load_cicids()
    # run_experiment("CICIDS2017", X_cicids, y_cicids, max_samples_for_tda=5000)

    print("\n" + "=" * 70)
    print("BASELINE EXPERIMENT COMPLETE")
    print("=" * 70)
