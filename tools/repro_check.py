"""
Reproduction regression check — cell S / R60 (malicious_random_attack,
n_swaps=60, no guidance), seed 42, UNSW-NB15, OPTICS, threshold 0.4.

This is the exact job Phase R (R3) and Phase W (W2) both ran as the
blocking reproduction gate. It calls the project's real pipeline functions
end to end (data loading -> attack generation -> TDA feature extraction ->
clustering -> classification) with no surrogate models needed and nothing
retrained. Kept here, tracked, as the project's standing regression test
for "does this environment/these path changes still reproduce the recorded
number" rather than re-deriving it ad hoc each time.

Usage:
    python tools/repro_check.py               # just report the number
    python tools/repro_check.py --expect 2.2  # exit nonzero on mismatch
"""
import argparse
import sys
import time
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from data_loader import load_unsw
from adversarial_attack import malicious_random_attack
from tda_pipeline import extract_tda_features
from clustering import run_all_clustering, classify_clusters

MAX_SAMPLES = 5000
POISON_RATE = 0.10
SEED = 42


def run():
    t_start = time.time()

    print("Loading full UNSW-NB15 dataset...")
    X_all, y_all = load_unsw(max_samples=None)

    # Matches run_lens4_baseline.py::subsample_for_seed exactly
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X_all), size=MAX_SAMPLES, replace=False)
    X_full, y_full = X_all[idx], y_all[idx]

    print(f"\nGenerating R60/'S' attack (malicious_random_attack, n_swaps=60, seed={SEED})...")
    t0 = time.time()
    Xc, yc, ip, log = malicious_random_attack(X_full, y_full, poison_rate=POISON_RATE,
                                               random_state=SEED, n_swaps=60)
    gen_time = time.time() - t0
    validity_pct = 100.0 * sum(l["valid"] for l in log) / len(log)
    print(f"  gen_time={gen_time:.1f}s validity={validity_pct:.1f}% n_poison={ip.sum()}")

    print("\nExtracting TDA features (60-dim, threshold=0.4 binarizer)...")
    t0 = time.time()
    X_tda, _ = extract_tda_features(Xc)
    tda_time = time.time() - t0
    print(f"  tda_time={tda_time:.1f}s shape={X_tda.shape}")

    print("\nRunning all 4 clustering algorithms...")
    t0 = time.time()
    results = run_all_clustering(X_tda)
    cluster_time = time.time() - t0
    print(f"  cluster_time={cluster_time:.1f}s")

    captures = {}
    print("\n=== CAPTURE RESULTS (cell S / R60, seed 42) ===")
    for algo_name, labels in results.items():
        cluster_info, summary = classify_clusters(labels, ip)
        captures[algo_name] = summary["red_poison_capture_pct"]
        print(f"  {algo_name:<10} capture={summary['red_poison_capture_pct']:.4f}%  "
              f"colors={summary['colors']}")

    total_time = time.time() - t_start
    print(f"\nTotal wall-clock: {total_time:.1f}s")
    print(f"X_tda.shape: {X_tda.shape}")

    return captures, X_tda.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", type=float, default=None,
                         help="Expected OPTICS capture %% (e.g. 2.2). Exit nonzero on mismatch.")
    parser.add_argument("--tol", type=float, default=1e-6,
                         help="Absolute tolerance for the --expect comparison.")
    args = parser.parse_args()

    captures, shape = run()

    if shape != (5500, 60):
        print(f"\nFAIL: expected X_tda.shape == (5500, 60), got {shape}")
        sys.exit(1)

    if args.expect is not None:
        observed = captures["OPTICS"]
        if abs(observed - args.expect) > args.tol:
            print(f"\nFAIL: OPTICS capture {observed:.4f}% != expected {args.expect:.4f}% "
                  f"(tol={args.tol})")
            sys.exit(1)
        print(f"\nPASS: OPTICS capture {observed:.4f}% matches expected {args.expect:.4f}%")


if __name__ == "__main__":
    main()
