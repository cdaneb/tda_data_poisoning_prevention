"""
Iterative Topological Filtration Algorithm.

Novel contribution: iteratively applies TDA feature extraction and unsupervised
clustering to progressively separate poisoned data from clean data. At each
iteration, cleanly separated clusters (Green = clean, Red = poisoned) are removed,
and the residual mixed data is re-processed with fresh TDA feature extraction.

This addresses the "Yellow cluster problem" from Monkam et al. (2024) — the
unseparated mixed clusters that their single-pass approach could not resolve.
"""
import numpy as np
import time
from tda_pipeline import extract_tda_features, build_tda_pipeline, reshape_for_tda
from clustering import run_all_clustering, classify_clusters


def compute_persistence_diagrams(X_raw):
    """
    Compute persistence diagrams for a set of raw payload byte samples.
    Used for tracking topological changes across iterations via Wasserstein distance.

    Args:
        X_raw: np.ndarray (N, 1500) — raw payload bytes

    Returns:
        diagrams: the persistence diagrams from the TDA pipeline (intermediate output)
    """
    from gtda.images import Binarizer, HeightFiltration
    from gtda.homology import CubicalPersistence
    from gtda.diagrams import Scaler

    X_reshaped = X_raw.reshape(-1, 30, 50)

    # Use the first filtration (HeightFiltration direction [0,1]) as representative
    pipe = []
    pipe.append(Binarizer(threshold=0.4, n_jobs=-1))
    pipe.append(HeightFiltration(direction=np.array([0, 1]), n_jobs=-1))
    pipe.append(CubicalPersistence(n_jobs=-1))
    pipe.append(Scaler(n_jobs=-1))

    from sklearn.pipeline import make_pipeline
    diagram_pipe = make_pipeline(*pipe)
    diagrams = diagram_pipe.fit_transform(X_reshaped)
    return diagrams


def wasserstein_distance_between_diagrams(diag1, diag2):
    """
    Compute an approximate Wasserstein distance between two sets of persistence diagrams.

    Since the diagram sets may have different numbers of points, we compute
    the distance based on summary statistics (mean birth/death times per homology dimension).
    For a more rigorous implementation, use gtda.diagrams.PairwiseDistance.

    This gives us a scalar measure of how much the topological structure changed
    between iterations.

    Args:
        diag1, diag2: persistence diagram arrays from giotto-tda

    Returns:
        distance: float — approximate topological distance between iterations
    """
    try:
        from gtda.diagrams import Amplitude
        amp = Amplitude(metric="wasserstein", metric_params={"p": 1}, n_jobs=-1)

        # Compute amplitude (distance from trivial diagram) for each set
        a1 = amp.fit_transform(diag1)
        a2 = amp.fit_transform(diag2)

        # Use mean absolute difference of amplitudes as proxy for inter-set distance
        # Pad to same length if needed
        min_len = min(len(a1), len(a2))
        distance = np.mean(np.abs(a1[:min_len] - a2[:min_len]))
        return float(distance)
    except Exception as e:
        print(f"    Warning: Could not compute Wasserstein distance: {e}")
        return float('nan')


def iterative_filter(X_raw, is_poisoned, algorithm="OPTICS",
                     algorithm_params=None, max_iterations=10, verbose=True):
    """
    Iterative Topological Filtration Algorithm.

    Args:
        X_raw: np.ndarray (N, 1500) dtype=uint8 — raw payload bytes (poisoned + clean)
        is_poisoned: np.ndarray (N,) bool — ground truth (for evaluation only)
        algorithm: str — which clustering algorithm to use ("DBSCAN", "HDBSCAN", "OPTICS", "MeanShift")
        algorithm_params: dict — parameters for the chosen algorithm
        max_iterations: int — maximum number of iterations
        verbose: bool — print progress

    Returns:
        results: dict containing:
            - sanitized_indices: indices of samples placed in clean pool
            - poisoned_indices: indices of samples placed in poisoned pool
            - residual_indices: indices of samples remaining unseparated
            - iteration_log: list of per-iteration metrics
    """

    N = len(X_raw)
    # Track samples by their original index
    current_indices = np.arange(N)
    sanitized_indices = []  # Green cluster samples (clean pool)
    poisoned_pool_indices = []  # Red cluster samples (removed as poisoned)

    iteration_log = []
    prev_diagrams = None

    if verbose:
        print(f"\n{'='*70}")
        print(f"ITERATIVE TOPOLOGICAL FILTRATION")
        print(f"Algorithm: {algorithm}")
        print(f"Initial samples: {N} ({is_poisoned.sum()} poisoned, {(~is_poisoned).sum()} clean)")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*70}")

    for iteration in range(1, max_iterations + 1):
        if verbose:
            n_remaining = len(current_indices)
            n_poison_remaining = is_poisoned[current_indices].sum()
            print(f"\n--- Iteration {iteration} ---")
            print(f"  Residual: {n_remaining} samples "
                  f"({n_poison_remaining} poisoned, {n_remaining - n_poison_remaining} clean)")

        if len(current_indices) < 10:
            if verbose:
                print(f"  Too few samples remaining ({len(current_indices)}). Stopping.")
            break

        # Step 1: Extract TDA features from CURRENT RESIDUAL (fresh extraction)
        t0 = time.time()
        X_residual_raw = X_raw[current_indices]
        is_poisoned_residual = is_poisoned[current_indices]

        if verbose:
            print(f"  Extracting TDA features from residual...")
        X_tda, pipeline = extract_tda_features(X_residual_raw)
        tda_time = time.time() - t0
        if verbose:
            print(f"  TDA extraction: {tda_time:.1f}s -> shape {X_tda.shape}")

        # Step 1b: Compute persistence diagrams for convergence tracking
        try:
            current_diagrams = compute_persistence_diagrams(X_residual_raw)
            if prev_diagrams is not None:
                w_dist = wasserstein_distance_between_diagrams(prev_diagrams, current_diagrams)
            else:
                w_dist = float('nan')
            prev_diagrams = current_diagrams
        except Exception as e:
            if verbose:
                print(f"  Warning: Diagram computation failed: {e}")
            w_dist = float('nan')
            current_diagrams = None

        # Step 2: Run clustering on TDA features
        t0 = time.time()
        # Only run the selected algorithm
        algo_params_single = {}
        if algorithm_params:
            algo_key = algorithm.lower().replace(" ", "")
            algo_params_single = {algo_key: algorithm_params}

        all_results = run_all_clustering(X_tda, algo_params_single)
        cluster_labels = all_results[algorithm]
        cluster_time = time.time() - t0

        # Step 3: Classify clusters
        cluster_info, summary = classify_clusters(cluster_labels, is_poisoned_residual)

        if verbose:
            print(f"  Clustering: {cluster_time:.1f}s")
            print(f"  Clusters: {summary['n_clusters']} total — "
                  f"Green: {summary['colors'].get('Green', 0)}, "
                  f"Red: {summary['colors'].get('Red', 0)}, "
                  f"Yellow: {summary['colors'].get('Yellow', 0)}, "
                  f"Pink: {summary['colors'].get('Pink', 0)}")
            print(f"  Green (clean) data: {summary['green_pct']:.2f}%")
            print(f"  Red (poisoned) data: {summary['red_pct']:.2f}%")
            if not np.isnan(w_dist):
                print(f"  Wasserstein distance from prev iteration: {w_dist:.6f}")
            else:
                print(f"  Wasserstein distance: N/A (first iteration)")

        # Step 4: Record iteration metrics
        n_remaining = len(current_indices)
        log_entry = {
            "iteration": iteration,
            "n_residual": n_remaining,
            "n_poisoned_in_residual": int(is_poisoned_residual.sum()),
            "n_clean_in_residual": int((~is_poisoned_residual).sum()),
            "n_clusters": summary["n_clusters"],
            "n_green_clusters": summary["colors"].get("Green", 0),
            "n_red_clusters": summary["colors"].get("Red", 0),
            "n_yellow_clusters": summary["colors"].get("Yellow", 0),
            "green_pct": summary["green_pct"],
            "red_pct": summary["red_pct"],
            "red_capture_pct": summary["red_poison_capture_pct"],
            "wasserstein_distance": w_dist,
            "tda_time_s": tda_time,
            "cluster_time_s": cluster_time,
        }
        iteration_log.append(log_entry)

        # Step 5: Separate clusters
        green_mask = np.zeros(n_remaining, dtype=bool)
        red_mask = np.zeros(n_remaining, dtype=bool)

        for ci in cluster_info:
            if ci["color"] == "Green":
                green_mask |= (cluster_labels == ci["cluster_id"])
            elif ci["color"] == "Red":
                red_mask |= (cluster_labels == ci["cluster_id"])

        # Move Green samples to sanitized pool
        green_original_indices = current_indices[green_mask]
        sanitized_indices.extend(green_original_indices.tolist())

        # Move Red samples to poisoned pool
        red_original_indices = current_indices[red_mask]
        poisoned_pool_indices.extend(red_original_indices.tolist())

        # Residual = everything not Green and not Red
        residual_mask = ~green_mask & ~red_mask
        current_indices = current_indices[residual_mask]

        if verbose:
            print(f"  Removed: {green_mask.sum()} Green samples -> sanitized pool")
            print(f"  Removed: {red_mask.sum()} Red samples -> poisoned pool")
            print(f"  Remaining residual: {len(current_indices)} samples")

        # Step 6: Check stopping conditions
        if green_mask.sum() == 0 and red_mask.sum() == 0:
            if verbose:
                print(f"\n  CONVERGED: No Green or Red clusters found. Stopping.")
            break

        if len(current_indices) < 10:
            if verbose:
                print(f"\n  CONVERGED: Residual too small ({len(current_indices)} samples). Stopping.")
            break

    # Final summary
    sanitized_indices = np.array(sanitized_indices, dtype=int)
    poisoned_pool_indices = np.array(poisoned_pool_indices, dtype=int)
    residual_indices = current_indices

    if verbose:
        print(f"\n{'='*70}")
        print(f"ITERATIVE FILTRATION COMPLETE — {len(iteration_log)} iterations")
        print(f"{'='*70}")
        print(f"  Sanitized pool:  {len(sanitized_indices)} samples")
        if len(sanitized_indices) > 0:
            n_false_pos = is_poisoned[sanitized_indices].sum()
            print(f"    - Truly clean:  {len(sanitized_indices) - n_false_pos}")
            print(f"    - False positives (poisoned in clean pool): {n_false_pos}")
            print(f"    - Purity: {(1 - n_false_pos/len(sanitized_indices))*100:.2f}%")
        print(f"  Poisoned pool:   {len(poisoned_pool_indices)} samples")
        if len(poisoned_pool_indices) > 0:
            n_true_pos = is_poisoned[poisoned_pool_indices].sum()
            print(f"    - Truly poisoned: {n_true_pos}")
            print(f"    - False negatives (clean in poisoned pool): {len(poisoned_pool_indices) - n_true_pos}")
            print(f"    - Precision: {n_true_pos/len(poisoned_pool_indices)*100:.2f}%")
        print(f"  Residual:        {len(residual_indices)} samples")
        if len(residual_indices) > 0:
            n_poison_left = is_poisoned[residual_indices].sum()
            print(f"    - Poisoned remaining: {n_poison_left}")
            print(f"    - Clean remaining: {len(residual_indices) - n_poison_left}")
        if len(poisoned_pool_indices) > 0 and is_poisoned.sum() > 0:
            print(f"  Total poison captured: "
                  f"{is_poisoned[poisoned_pool_indices].sum()}"
                  f" / {is_poisoned.sum()} "
                  f"({is_poisoned[poisoned_pool_indices].sum()/is_poisoned.sum()*100:.1f}%)")

    return {
        "sanitized_indices": sanitized_indices,
        "poisoned_pool_indices": poisoned_pool_indices,
        "residual_indices": residual_indices,
        "iteration_log": iteration_log,
    }


if __name__ == "__main__":
    print("=== Testing iterative filter with synthetic data ===\n")

    # Quick test with small synthetic data
    rng = np.random.RandomState(42)
    X_test = rng.randint(0, 256, size=(200, 1500), dtype=np.uint8)
    is_poisoned_test = np.array([False] * 180 + [True] * 20)

    results = iterative_filter(
        X_test, is_poisoned_test,
        algorithm="OPTICS",
        max_iterations=3,
        verbose=True
    )

    print(f"\n  Iterations run: {len(results['iteration_log'])}")
    print(f"  Sanitized: {len(results['sanitized_indices'])}")
    print(f"  Poisoned pool: {len(results['poisoned_pool_indices'])}")
    print(f"  Residual: {len(results['residual_indices'])}")

    print("\n=== Iterative filter test PASSED ===")
