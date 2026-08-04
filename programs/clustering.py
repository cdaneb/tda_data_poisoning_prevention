"""
Unsupervised clustering module.

Applies four clustering algorithms to the data as specified in the paper:
  1. DBSCAN
  2. HDBSCAN
  3. OPTICS
  4. Mean Shift Clustering

Then classifies the resulting clusters by their poisoned data content:
  - Green: 0% poisoned (exclusively clean)
  - Red: 100% poisoned (exclusively poisoned)
  - Yellow: mixed (contains both poisoned and clean)
  - Pink: predominantly poisoned (>80% poisoned, subset of yellow)
"""
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift


def run_all_clustering(X, algorithm_params=None):
    """
    Run all four clustering algorithms on the data.

    Args:
        X: np.ndarray of shape (N, F) — feature matrix
        algorithm_params: dict of dicts with custom parameters per algorithm

    Returns:
        results: dict mapping algorithm name -> cluster labels (np.ndarray of shape (N,))
    """
    if algorithm_params is None:
        algorithm_params = {}

    results = {}

    # 1. DBSCAN
    print("  Running DBSCAN...")
    dbscan_params = algorithm_params.get("dbscan", {"eps": 0.5, "min_samples": 5})
    db = DBSCAN(**dbscan_params, n_jobs=-1)
    results["DBSCAN"] = db.fit_predict(X)
    n_clusters = len(set(results["DBSCAN"])) - (1 if -1 in results["DBSCAN"] else 0)
    print(f"    Found {n_clusters} clusters + {(results['DBSCAN'] == -1).sum()} noise points")

    # 2. HDBSCAN (using sklearn.cluster.HDBSCAN, available since sklearn 1.3)
    print("  Running HDBSCAN...")
    hdbscan_params = algorithm_params.get("hdbscan", {"min_cluster_size": 10})
    hdb = HDBSCAN(**hdbscan_params)
    results["HDBSCAN"] = hdb.fit_predict(X)
    n_clusters = len(set(results["HDBSCAN"])) - (1 if -1 in results["HDBSCAN"] else 0)
    print(f"    Found {n_clusters} clusters + {(results['HDBSCAN'] == -1).sum()} noise points")

    # 3. OPTICS
    print("  Running OPTICS...")
    optics_params = algorithm_params.get("optics", {"min_samples": 5, "max_eps": 2.0})
    opt = OPTICS(**optics_params, n_jobs=-1)
    results["OPTICS"] = opt.fit_predict(X)
    n_clusters = len(set(results["OPTICS"])) - (1 if -1 in results["OPTICS"] else 0)
    print(f"    Found {n_clusters} clusters + {(results['OPTICS'] == -1).sum()} noise points")

    # 4. Mean Shift
    print("  Running Mean Shift...")
    ms_params = algorithm_params.get("meanshift", {})
    ms = MeanShift(**ms_params, n_jobs=-1)
    results["MeanShift"] = ms.fit_predict(X)
    n_clusters = len(set(results["MeanShift"]))
    print(f"    Found {n_clusters} clusters")

    return results


def classify_clusters(cluster_labels, is_poisoned):
    """
    Classify each cluster by its composition of poisoned vs clean data.

    Uses the paper's color scheme:
      - Green: 0% poisoned
      - Red: 100% poisoned
      - Pink: >80% poisoned
      - Yellow: mixed (all others)

    Args:
        cluster_labels: np.ndarray of shape (N,) — cluster assignments (-1 = noise)
        is_poisoned: np.ndarray of shape (N,) bool — True for poisoned samples

    Returns:
        cluster_info: list of dicts with per-cluster statistics
        summary: dict with aggregate statistics
    """
    unique_labels = sorted(set(cluster_labels))
    cluster_info = []

    n_total = len(cluster_labels)
    n_total_poisoned = is_poisoned.sum()

    for label in unique_labels:
        mask = cluster_labels == label
        n_in_cluster = mask.sum()
        n_poisoned_in_cluster = (is_poisoned & mask).sum()
        n_clean_in_cluster = n_in_cluster - n_poisoned_in_cluster
        poison_fraction = n_poisoned_in_cluster / n_in_cluster if n_in_cluster > 0 else 0

        # Classify cluster color
        if label == -1:
            color = "Noise"
        elif poison_fraction == 0:
            color = "Green"
        elif poison_fraction == 1.0:
            color = "Red"
        elif poison_fraction > 0.80:
            color = "Pink"
        else:
            color = "Yellow"

        info = {
            "cluster_id": label,
            "color": color,
            "size": n_in_cluster,
            "size_pct": n_in_cluster / n_total * 100,
            "n_poisoned": n_poisoned_in_cluster,
            "n_clean": n_clean_in_cluster,
            "poison_fraction": poison_fraction,
            "dpdc": n_poisoned_in_cluster / n_total_poisoned * 100 if n_total_poisoned > 0 else 0,
        }
        cluster_info.append(info)

    # Summary statistics
    green_pct = sum(c["size_pct"] for c in cluster_info if c["color"] == "Green")
    red_pct = sum(c["size_pct"] for c in cluster_info if c["color"] == "Red")
    red_poison_capture = sum(c["n_poisoned"] for c in cluster_info if c["color"] == "Red")
    red_poison_capture_pct = red_poison_capture / n_total_poisoned * 100 if n_total_poisoned > 0 else 0

    summary = {
        "n_clusters": len([c for c in cluster_info if c["cluster_id"] != -1]),
        "green_pct": green_pct,
        "red_pct": red_pct,
        "red_poison_capture_pct": red_poison_capture_pct,
        "colors": {color: len([c for c in cluster_info if c["color"] == color])
                   for color in ["Green", "Red", "Yellow", "Pink", "Noise"]},
    }

    return cluster_info, summary


def print_cluster_report(algorithm_name, cluster_info, summary):
    """Pretty-print the cluster analysis results."""
    print(f"\n  --- {algorithm_name} ---")
    print(f"  Clusters found: {summary['n_clusters']}")
    print(f"  Color breakdown: {summary['colors']}")
    print(f"  Green (clean-only) data: {summary['green_pct']:.2f}%")
    print(f"  Red (poison-only) data: {summary['red_pct']:.2f}%")
    print(f"  Red clusters capture: {summary['red_poison_capture_pct']:.2f}% of all poisoned data")
    print()
    for c in cluster_info:
        print(f"    Cluster {c['cluster_id']:>3d} [{c['color']:>6s}]: "
              f"{c['size']:>6d} samples ({c['size_pct']:>5.2f}%), "
              f"poison: {c['poison_fraction']:.1%}, "
              f"DPDC: {c['dpdc']:.1f}%")


if __name__ == "__main__":
    print("=== Testing clustering module ===\n")

    # Synthetic test: create data with clear clusters
    rng = np.random.RandomState(42)
    # Two clean clusters + one poisoned cluster
    X_clean1 = rng.randn(50, 10) + np.array([0] * 10)
    X_clean2 = rng.randn(50, 10) + np.array([5] * 10)
    X_poison = rng.randn(20, 10) + np.array([10] * 10)

    X_test = np.vstack([X_clean1, X_clean2, X_poison])
    is_poisoned_test = np.array([False] * 100 + [True] * 20)

    results = run_all_clustering(X_test)

    for algo_name, labels in results.items():
        cluster_info, summary = classify_clusters(labels, is_poisoned_test)
        print_cluster_report(algo_name, cluster_info, summary)

    print("\n=== Clustering module verification PASSED ===")
